/* ------------------------------------------------------------------ */
/* Static Chatbot — multi-session frontend logic                      */
/* ------------------------------------------------------------------ */

(function () {
  "use strict";

  /* ---- DOM references ---------------------------------------------- */

  const sidebarEl = document.getElementById("sidebar");
  const sidebarToggleEl = document.getElementById("sidebarToggle");
  const sessionListEl = document.getElementById("sessionList");
  const sessionListStatusEl = document.getElementById("sessionListStatus");
  const newChatBtn = document.getElementById("newChatButton");
  const profileSelectEl = document.getElementById("profileSelect");
  const profileSelectorStatusEl =
    document.getElementById("profileSelectorStatus");
  const sessionHeaderEl = document.getElementById("sessionHeader");
  const sessionTitleEl = document.getElementById("sessionTitle");
  const sessionModelBadgeEl = document.getElementById("sessionModelBadge");
  const sessionCompatibilityNoticeEl =
    document.getElementById("sessionCompatibilityNotice");
  const messagesEl = document.getElementById("messages");
  const statusEl = document.getElementById("status");
  const inputEl = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendButton");

  /* ---- State ------------------------------------------------------- */

  let sessions = [];            // [{id, title, ..., llm_profile_*}, ...]
  let currentSessionId = null;  // int | null
  let isSending = false;        // prevent double-submit
  let isCreatingSession = false; // prevent double-create of New Chat
  let isRenaming = false;       // editing mode active
  let renamingSessionId = null; // which session is being edited
  let isRenameSaving = false;   // saving in progress (prevent double-submit)
  let isInitializing = true;    // block user actions during page init
  let sessionLoadRequestId = 0;  // race-condition guard for message loads
  let sessionLastMessageId = {}; // sessionId -> int (last known message id)

  // Model selector state
  let profiles = [];               // raw profile list from the server
  let selectedProfileId = null;    // string | null
  let profilesLoadState = "loading"; // "loading" | "ready" | "error"
  let profilesLoadError = null;    // persistent — #profileSelectorStatus

  // Session list state
  let sessionsLoadState = "loading"; // "loading" | "ready" | "error"
  let sessionsEverLoaded = false;    // ever loaded the full list
  let sessionsLoadError = null;      // persistent — #sessionListStatus

  // Temporary per-session send blocks set after 409/503 responses.
  // sessionId -> "conflict" | "profile_unavailable"
  let sessionSendBlocks = {};

  /* ---- Derived helpers --------------------------------------------- */

  /** The registry analysis of the current profile list. */
  function registry() {
    return analyzeProfileRegistry(profiles);
  }

  /** The SessionResponse for currentSessionId, or null. */
  function currentSession() {
    if (currentSessionId === null) return null;
    for (let i = 0; i < sessions.length; i++) {
      if (sessions[i].id === currentSessionId) return sessions[i];
    }
    return null;
  }

  /** Whether creating new sessions is currently possible. */
  function registryUsable() {
    return profilesLoadState === "ready" && registry().status === "valid";
  }

  /** Whether the current session (or none) can accept a message. */
  function currentSessionWritable() {
    return isSessionWritable(
      currentSession(),
      sessionSendBlocks[currentSessionId] || null,
    );
  }

  /** kind of a profile — only from a structurally valid registry. */
  function lookupProfileKind(profileId) {
    return profileKindFromRegistry(registry(), profileId);
  }

  /* ---- Custom error ------------------------------------------------ */

  class ApiError extends Error {
    constructor(message, status = 0) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  /* ---- Unified API request ----------------------------------------- */

  /**
   * Thin wrapper around fetch() that normalises errors.
   *
   * Returns the Response on success.  Throws ApiError with a
   * human-readable English message on any failure.
   */
  async function apiRequest(url, options = {}) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (_) {
      throw new ApiError(
        "Network error. Please check your connection.",
        0,
      );
    }

    if (!response.ok) {
      let detail = "Something went wrong. Please try again.";
      try {
        const body = await response.json();
        if (typeof body.detail === "string") {
          detail = body.detail;
        } else if (
          Array.isArray(body.detail) &&
          body.detail.length > 0 &&
          body.detail[0].msg
        ) {
          detail = body.detail[0].msg;
        }
      } catch (_) {
        /* non-JSON body — use default detail */
      }
      throw new ApiError(detail, response.status);
    }

    return response;
  }

  /* ---- API calls --------------------------------------------------- */

  async function fetchSessions() {
    const resp = await apiRequest("/api/sessions");
    return resp.json();
  }

  async function fetchSession(sessionId) {
    const resp = await apiRequest("/api/sessions/" + sessionId);
    return resp.json();
  }

  async function fetchProfiles() {
    const resp = await apiRequest("/api/llm/profiles");
    return resp.json();
  }

  async function createSessionRequest(profileId) {
    const resp = await apiRequest("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildCreateSessionPayload(profileId)),
    });
    return resp.json();
  }

  async function fetchMessages(sessionId) {
    const resp = await apiRequest(
      "/api/sessions/" + sessionId + "/messages",
    );
    return resp.json();
  }

  async function sendSessionMessageRequest(sessionId, text) {
    const resp = await apiRequest(
      "/api/sessions/" + sessionId + "/messages",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      },
    );
    return resp.json();
  }

  async function deleteSessionRequest(sessionId) {
    await apiRequest("/api/sessions/" + sessionId, { method: "DELETE" });
  }

  async function renameSessionRequest(sessionId, title) {
    const resp = await apiRequest(
      "/api/sessions/" + sessionId,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title }),
      },
    );
    return resp.json();
  }

  /* ---- Rendering: profile selector ---------------------------------- */

  /**
   * Rebuild the selector options and its persistent status text.
   *
   * Called ONLY when profile content or the resolved selection changes
   * (initial load, 422 reload).  Never called from routine control
   * synchronisation — rebuilding options would churn the DOM and drop
   * the select's focus.
   */
  function renderProfileSelectorContent() {
    const hadFocus = document.activeElement === profileSelectEl;
    const reg = registry();

    profileSelectEl.replaceChildren();

    if (reg.status === "valid") {
      for (const p of reg.profiles) {
        const option = document.createElement("option");
        option.value = p.id;
        option.textContent = p.label;
        profileSelectEl.appendChild(option);
      }
      if (selectedProfileId !== null) {
        profileSelectEl.value = selectedProfileId;
      }
    } else {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No models available";
      profileSelectEl.appendChild(option);
      profileSelectEl.value = "";
    }

    profileSelectorStatusEl.textContent = profileRegistryStatusText(
      reg.status,
      profilesLoadError,
    );

    // Minimal, explainable focus restoration: only when the select
    // held focus before the rebuild and is still usable.
    if (hadFocus && !profileSelectEl.disabled) {
      profileSelectEl.focus();
    }
  }

  /* ---- Rendering: session list -------------------------------------- */

  /**
   * Render the session list and its persistent status area.
   *
   * Handles list content, the empty-list message, initial load errors
   * and stale-list warnings in one place so the list is never hidden
   * or faked when an error exists.
   */
  function renderSessionListOrError() {
    if (
      sessionsLoadState === "error" &&
      sessions.length === 0 &&
      !sessionsEverLoaded
    ) {
      // Initial load failed — show an error, never "No conversations
      // yet.".
      sessionListEl.replaceChildren();
      const errDiv = document.createElement("div");
      errDiv.className = "sidebar-empty";
      errDiv.textContent = "Couldn't load conversations.";
      sessionListEl.appendChild(errDiv);

      sessionListStatusEl.textContent =
        sessionsLoadError || "Couldn't load conversations.";
      sessionListStatusEl.classList.remove("warning");
      sessionListStatusEl.hidden = false;
      return;
    }

    renderSessionList();

    if (sessionsLoadState === "error") {
      // Keep the cached/known list visible with a warning.
      if (sessionsEverLoaded) {
        sessionListStatusEl.textContent =
          "Conversation list may be out of date.";
      } else {
        sessionListStatusEl.textContent =
          "Some conversations may not have loaded.";
      }
      sessionListStatusEl.classList.add("warning");
      sessionListStatusEl.hidden = false;
    } else {
      sessionListStatusEl.textContent = "";
      sessionListStatusEl.classList.remove("warning");
      sessionListStatusEl.hidden = true;
    }
  }

  /* ---- Rendering: current session meta ------------------------------ */

  /**
   * Title, model badge, and compatibility notice for the current
   * session.  Everything comes from the SessionResponse itself; the
   * kind chip only when the current registry is valid.
   */
  function renderCurrentSessionMeta() {
    const session = currentSession();

    if (session === null) {
      sessionTitleEl.textContent = "";
      sessionModelBadgeEl.replaceChildren();
      sessionModelBadgeEl.hidden = true;
      sessionCompatibilityNoticeEl.textContent = "";
      sessionCompatibilityNoticeEl.hidden = true;
      return;
    }

    // -- title ----------------------------------------------------------
    sessionTitleEl.textContent = session.title;

    // -- badge: kind chip (valid registry only) + session's own label ---
    sessionModelBadgeEl.replaceChildren();
    const kindText = profileKindBadgeText(
      lookupProfileKind(session.llm_profile_id),
    );
    if (kindText !== null) {
      const kindSpan = document.createElement("span");
      kindSpan.className = "badge-kind";
      kindSpan.textContent = kindText;
      sessionModelBadgeEl.appendChild(kindSpan);
    }
    const labelSpan = document.createElement("span");
    labelSpan.className = "badge-label";
    labelSpan.textContent = session.llm_profile_label || "";
    sessionModelBadgeEl.appendChild(labelSpan);
    sessionModelBadgeEl.hidden = false;

    // -- compatibility notice: temporary block first, then server status
    const block = sessionSendBlocks[session.id] || null;
    const blockText = temporaryBlockExplanation(block);
    const noticeText =
      blockText !== null
        ? blockText
        : readOnlyExplanation(session.llm_profile_status);

    if (noticeText) {
      sessionCompatibilityNoticeEl.textContent = noticeText;
      sessionCompatibilityNoticeEl.hidden = false;
    } else {
      sessionCompatibilityNoticeEl.textContent = "";
      sessionCompatibilityNoticeEl.hidden = true;
    }
  }

  /* ---- Control state helpers --------------------------------------- */

  /**
   * Sync disabled states and placeholders only.  Never touches focus
   * and never rebuilds selector options.
   *
   * Tiered blocks:
   * - blockSessionSelection — session switching disabled
   * - blockSessionActions  — rename/delete disabled
   * - blockCreate          — New Chat disabled
   * - blockProfileSelect   — selector disabled
   * - blockSend            — input/Send disabled
   *
   * Session switching stays available while a send is in progress
   * (sendingSessionId guards stale renders); everything else is
   * locked during send/create.
   */
  function updateControlStates() {
    const blockBase = isInitializing || isRenaming || isRenameSaving;
    const blockSessionSelection = blockBase || isCreatingSession;
    const blockSessionActions = blockBase || isCreatingSession || isSending;
    const usable = registryUsable();
    const writable = currentSessionWritable();

    const blockCreate = blockSessionActions || !usable;
    const blockProfileSelect =
      blockSessionActions || !usable || profiles.length <= 1;
    const blockSend =
      blockBase || isCreatingSession || isSending || !writable ||
      (currentSessionId === null && !usable);

    sendBtn.disabled = blockSend;
    inputEl.disabled = blockSend;
    newChatBtn.disabled = blockCreate;
    profileSelectEl.disabled = blockProfileSelect;

    if (blockSend && currentSessionId !== null && !writable) {
      inputEl.placeholder =
        "This conversation is read-only. Start a new chat to continue.";
    } else {
      inputEl.placeholder = "Type a message...";
    }

    const deleteBtns = document.querySelectorAll(".delete-session-btn");
    deleteBtns.forEach(function (btn) {
      btn.disabled = blockSessionActions;
    });

    const renameBtns = document.querySelectorAll(".rename-session-btn");
    renameBtns.forEach(function (btn) {
      btn.disabled = blockSessionActions;
    });

    const selectBtns = document.querySelectorAll(".session-select-btn");
    selectBtns.forEach(function (btn) {
      btn.disabled = blockSessionSelection;
    });

    // Rename input / Save / Cancel are only controlled by isRenameSaving:
    // - Editing (isRenaming=true, isRenameSaving=false): enabled
    // - Saving  (isRenaming=true, isRenameSaving=true):  disabled
    if (isRenaming) {
      const input = document.querySelector(".session-rename-input");
      const saveBtn = document.querySelector(".session-rename-save-btn");
      const cancelBtn = document.querySelector(".session-rename-cancel-btn");
      if (input) input.disabled = isRenameSaving;
      if (saveBtn) saveBtn.disabled = isRenameSaving;
      if (cancelBtn) cancelBtn.disabled = isRenameSaving;
    }
  }

  /**
   * Synchronise the current-session UI.  Only renders meta and
   * controls — never rebuilds selector options.
   */
  function syncCurrentSessionUI() {
    renderCurrentSessionMeta();
    updateControlStates();
  }

  function setSendingState(on) {
    isSending = on;
    updateControlStates();
  }

  /* ---- Status helpers ---------------------------------------------- */

  function showStatus(text, isError) {
    statusEl.textContent = text;
    statusEl.className = isError ? "chat-status error" : "chat-status";
  }

  function clearStatus() {
    statusEl.textContent = "";
    statusEl.className = "chat-status";
  }

  /* ---- Messages ---------------------------------------------------- */

  function scrollMessagesToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function clearMessages() {
    messagesEl.replaceChildren();
  }

  /** Remove any .welcome-state node from the messages container. */
  function removeEmptyState() {
    const el = messagesEl.querySelector(".welcome-state");
    if (el) {
      el.remove();
    }
  }

  /**
   * Append a single message bubble using textContent (XSS-safe).
   *
   * Removes the welcome / empty-state placeholder on the first real
   * message so it never coexists with message bubbles.
   */
  function appendMessage(role, content) {
    removeEmptyState();

    var wrapper = document.createElement("div");
    wrapper.className = "message " + role;

    var contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = content;

    wrapper.appendChild(contentDiv);

    // Assistant messages get a copy button
    if (role === "assistant") {
      var actionsWrapper = document.createElement("div");
      actionsWrapper.className = "message-actions";

      var copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "copy-btn";
      copyBtn.setAttribute("aria-label", "Copy response");
      copyBtn.textContent = "Copy";
      copyBtn.dataset.state = "idle";

      var liveRegion = document.createElement("span");
      liveRegion.className = "sr-only";
      liveRegion.setAttribute("aria-live", "polite");
      liveRegion.setAttribute("aria-atomic", "true");

      var ctrl = createCopyController(
        function (t) {
          return copyToClipboard(t, navigator.clipboard, document);
        },
        function (fn, ms) { return setTimeout(fn, ms); },
        function (id) { clearTimeout(id); },
        function (newState) {
          if (!copyBtn.isConnected) return;

          copyBtn.dataset.state = newState;

          switch (newState) {
            case "copying":
              copyBtn.textContent = "Copying…";
              copyBtn.disabled = true;
              liveRegion.textContent = "Copying response to clipboard.";
              break;
            case "copied":
              copyBtn.textContent = "Copied";
              copyBtn.disabled = false;
              liveRegion.textContent = "Response copied to clipboard.";
              break;
            case "failed":
              copyBtn.textContent = "Failed";
              copyBtn.disabled = false;
              liveRegion.textContent = "Failed to copy response. Press to retry.";
              break;
            default:
              copyBtn.textContent = "Copy";
              copyBtn.disabled = false;
              liveRegion.textContent = "";
              break;
          }
        },
        function () { return copyBtn.isConnected; }
      );

      copyBtn.addEventListener("click", function () {
        ctrl.handleClick(content);
      });

      actionsWrapper.appendChild(copyBtn);
      wrapper.appendChild(actionsWrapper);
      wrapper.appendChild(liveRegion);
    }

    messagesEl.appendChild(wrapper);
    scrollMessagesToBottom();
  }

  /**
   * Replace the messages area with a full history render.
   *
   * Skips unknown roles silently (XSS-safe by design — only
   * "user" and "assistant" are rendered).
   */
  function renderMessages(messages) {
    clearMessages();
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role === "user" || msg.role === "assistant") {
        appendMessage(msg.role, msg.content);
      }
    }
    // Track the last known message id for this session so network-error
    // recovery can use an ID boundary instead of text matching.
    if (messages.length > 0) {
      sessionLastMessageId[currentSessionId] = messages[messages.length - 1].id;
    } else {
      sessionLastMessageId[currentSessionId] = 0;
    }
  }

  /* ---- Welcome / empty states -------------------------------------- */

  function renderWelcome() {
    clearMessages();
    const div = document.createElement("div");
    div.className = "welcome-state";

    const h2 = document.createElement("h2");
    h2.textContent = "Welcome to Static Chatbot";

    const p = document.createElement("p");
    p.textContent =
      "Click + New Chat to start a conversation, " +
      "or type a message below to begin.";

    div.appendChild(h2);
    div.appendChild(p);
    messagesEl.appendChild(div);
  }

  function renderEmptyChat() {
    clearMessages();
    const div = document.createElement("div");
    div.className = "welcome-state";

    const p = document.createElement("p");
    p.textContent = "Start a conversation...";

    div.appendChild(p);
    messagesEl.appendChild(div);
  }

  /* ---- Time formatting --------------------------------------------- */

  /**
   * Parse an ISO-8601 string from the API.
   *
   * Backend timestamps are UTC but may lack a timezone suffix
   * (e.g. "2026-08-06T14:00:00").  Browsers parse such strings as
   * local time, causing an offset equal to the local timezone.
   *
   * This helper detects the missing suffix and appends "Z" so the
   * string is always interpreted as UTC.  Strings that already carry
   * a timezone ("Z", "+HH:MM", "-HH:MM") are left unchanged.
   *
   * Returns a Date, or null when the input is not a valid string.
   */
  function parseApiDate(isoString) {
    if (typeof isoString !== "string" || !isoString) {
      return null;
    }

    const hasTimezone =
      /Z$/i.test(isoString) ||
      /[+-]\d{2}:\d{2}$/.test(isoString);

    const date = new Date(hasTimezone ? isoString : isoString + "Z");

    return Number.isNaN(date.getTime()) ? null : date;
  }

  function formatSessionTime(isoString) {
    const now = new Date();
    const date = parseApiDate(isoString);

    // Fallback for unparseable timestamps
    if (date === null) {
      return "";
    }

    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);

    if (diffMin < 1) return "Just now";
    if (diffMin < 60) return diffMin + "m ago";
    if (diffHour < 24) return diffHour + "h ago";
    if (diffDay < 7) return diffDay + "d ago";

    const months = [
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    return months[date.getMonth()] + " " + date.getDate();
  }

  /* ---- Session list rendering -------------------------------------- */

  function renderSessionList() {
    sessionListEl.replaceChildren();

    if (sessions.length === 0) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "sidebar-empty";
      emptyDiv.textContent = "No conversations yet.";
      sessionListEl.appendChild(emptyDiv);
      return;
    }

    // Use for…of with const — each iteration gets a fresh binding
    // so event listeners capture the correct session id.
    for (const session of sessions) {
      const sid = session.id;

      // Layout container — no interactive role
      const item = document.createElement("div");
      item.className = "session-item";
      if (sid === currentSessionId) {
        item.classList.add("active");
      }

      // -- Inline-edit mode for this session -----------------------------
      if (isRenaming && renamingSessionId === sid) {
        item.classList.add("editing");

        const input = document.createElement("input");
        input.type = "text";
        input.className = "session-rename-input";
        input.value = session.title;
        input.setAttribute("aria-label", "Rename session");

        const saveBtn = document.createElement("button");
        saveBtn.type = "button";
        saveBtn.className = "session-rename-save-btn";
        saveBtn.textContent = "Save";

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "session-rename-cancel-btn";
        cancelBtn.textContent = "Cancel";

        input.addEventListener("keydown", function (event) {
          if (event.key === "Enter" && !event.isComposing) {
            event.preventDefault();
            saveRename(sid, input);
          } else if (event.key === "Escape") {
            event.preventDefault();
            cancelRename();
          }
        });

        saveBtn.addEventListener("click", function () {
          saveRename(sid, input);
        });
        cancelBtn.addEventListener("click", function () {
          cancelRename();
        });

        item.appendChild(input);
        item.appendChild(saveBtn);
        item.appendChild(cancelBtn);

        // Auto-focus the input on next render frame
        setTimeout(function () { input.focus(); input.select(); }, 0);
      } else {
        // -- Select button (native <button> — keyboard/ARIA for free) -----
        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "session-select-btn";
        selectBtn.dataset.sessionId = String(sid);
        if (sid === currentSessionId) {
          selectBtn.classList.add("active");
        }

        const label = document.createElement("span");
        label.className = "session-item-label";
        label.textContent = session.title;

        const time = document.createElement("span");
        time.className = "session-time";
        time.textContent = formatSessionTime(session.updated_at);

        selectBtn.appendChild(label);
        selectBtn.appendChild(time);
        selectBtn.addEventListener("click", function () {
          selectSession(sid);
        });

        // -- Rename button -----------------------------------------------
        const renameBtn = document.createElement("button");
        renameBtn.type = "button";
        renameBtn.className = "rename-session-btn";
        renameBtn.setAttribute("aria-label", "Rename session " + sid);
        renameBtn.title = "Rename";
        renameBtn.textContent = "✎";  // U+270E
        renameBtn.addEventListener("click", function (event) {
          event.stopPropagation();
          startRename(sid);
        });

        // -- Delete button -----------------------------------------------
        const delBtn = document.createElement("button");
        delBtn.className = "delete-session-btn";
        delBtn.type = "button";
        delBtn.setAttribute("aria-label", "Delete session " + sid);
        delBtn.title = "Delete";
        delBtn.textContent = "×";
        delBtn.addEventListener("click", function (event) {
          event.stopPropagation();
          handleDeleteSession(sid, event);
        });

        item.appendChild(selectBtn);
        item.appendChild(renameBtn);
        item.appendChild(delBtn);
      }

      sessionListEl.appendChild(item);
    }
  }

  /* ---- Session operations ------------------------------------------ */

  /**
   * Refresh the full session list.
   *
   * Returns true on success, false on failure.  On failure the
   * existing sessions array, current selection and rendered messages
   * are kept; a persistent stale warning is shown.  On success the
   * server list is authoritative: temporary send blocks for sessions
   * it contains are cleared (their real llm_profile_status now
   * governs writability) and blocks for sessions no longer present
   * are dropped.
   */
  async function refreshSessions(preserveSelection) {
    if (preserveSelection === undefined) preserveSelection = true;

    let data;
    try {
      data = await fetchSessions();
    } catch (err) {
      sessionsLoadState = "error";
      sessionsLoadError = err.message;
      renderSessionListOrError();
      showStatus(err.message, true);
      return false;
    }

    // Validate BEFORE touching any local state: a malformed response
    // must never replace the cached list, the current selection, or
    // any temporary send block.
    if (!isValidSessionList(data)) {
      sessionsLoadState = "error";
      sessionsLoadError = "The server returned an invalid response.";
      renderSessionListOrError();
      showStatus(sessionsLoadError, true);
      return false;
    }

    sessions = data;
    sessionsLoadState = "ready";
    sessionsEverLoaded = true;
    sessionsLoadError = null;

    // Server list is authoritative for temporary blocks.
    const presentIds = {};
    for (const s of sessions) {
      presentIds[s.id] = true;
      delete sessionSendBlocks[s.id];
    }
    for (const sid of Object.keys(sessionSendBlocks)) {
      if (!presentIds[sid]) delete sessionSendBlocks[sid];
    }

    // Decide the final selection BEFORE rendering the list so the
    // highlight, title, badge and message area all point at the same
    // session.
    const next = resolveNextSelectionId(
      sessions,
      currentSessionId,
      preserveSelection,
    );
    const selectionChanged = next.changed;
    currentSessionId = next.selectionId;

    // Only the null-selection case bumps the request guard manually:
    // it must invalidate an in-flight load without starting a new one.
    // A real switch delegates the bump to loadMessages().
    if (next.selectionId === null) {
      sessionLoadRequestId++;
    }

    renderSessionListOrError();
    syncCurrentSessionUI();

    if (next.selectionId === null) {
      renderWelcome();
      return true;
    }

    if (selectionChanged) {
      loadMessages(next.selectionId);
    }

    return true;
  }

  /**
   * Precisely refresh one session after a 409/503 response.
   *
   * Clears the temporary block only when the GET succeeded, the
   * response is an object with the expected id, and it was written
   * back into the local sessions array.  On 404 the session is
   * removed locally.  Any other failure keeps the block, the draft
   * and the read-only notice.  Never touches sessionsLoadState.
   */
  async function refreshOneSessionCompatibility(sessionId) {
    let fresh;
    try {
      fresh = await fetchSession(sessionId);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        removeSessionLocally(sessionId);
        return false;
      }
      return false;
    }

    if (!isValidSessionResponse(fresh) || fresh.id !== sessionId) {
      // Invalid or mismatched response — keep the block and the
      // cached session, never overwrite anything.
      return false;
    }

    const idx = sessions.findIndex(function (s) {
      return s.id === sessionId;
    });
    if (idx >= 0) {
      sessions[idx] = fresh;
    } else {
      sessions.unshift(fresh);
    }
    delete sessionSendBlocks[sessionId];
    renderSessionListOrError();
    syncCurrentSessionUI();
    return true;
  }

  /**
   * Remove a session from the local list and handle UI transitions
   * if it was the currently-selected session.
   */
  function removeSessionLocally(sessionId) {
    sessions = sessions.filter(function (s) { return s.id !== sessionId; });
    delete sessionLastMessageId[sessionId];
    delete sessionSendBlocks[sessionId];

    if (sessionId === currentSessionId) {
      sessionLoadRequestId++;
      clearMessages();
      clearStatus();

      if (sessions.length > 0) {
        currentSessionId = sessions[0].id;
        syncCurrentSessionUI();
        loadMessages(currentSessionId);
      } else {
        currentSessionId = null;
        syncCurrentSessionUI();
        renderWelcome();
      }
    }

    renderSessionListOrError();
  }

  async function loadMessages(sessionId) {
    sessionLoadRequestId++;
    const requestId = sessionLoadRequestId;

    clearMessages();
    showStatus("Loading...", false);

    let data;
    try {
      data = await fetchMessages(sessionId);
    } catch (err) {
      // On 404 the session no longer exists — remove it locally
      if (err instanceof ApiError && err.status === 404) {
        if (requestId === sessionLoadRequestId) {
          removeSessionLocally(sessionId);
        }
        return;
      }
      // Other errors — show if still current
      if (requestId === sessionLoadRequestId &&
          currentSessionId === sessionId) {
        renderEmptyChat();
        showStatus(err.message, true);
      }
      return;
    }

    // Discard stale responses
    if (requestId !== sessionLoadRequestId) return;
    if (currentSessionId !== sessionId) return;

    // A malformed list must never reach renderMessages or touch
    // sessionLastMessageId.
    if (!isValidMessageList(data, sessionId)) {
      renderEmptyChat();
      showStatus("The server returned an invalid response.", true);
      return;
    }

    if (data.length === 0) {
      renderEmptyChat();
    } else {
      renderMessages(data);
    }
    clearStatus();
  }

  /**
   * Switch to a different session.  Allowed during send — the
   * sendingSessionId guard in sendMessage prevents the old response
   * from rendering into the new session.
   *
   * Focus rule: on desktop the focus stays on the clicked session
   * button.  On mobile (sidebar closes) focus moves to the input when
   * the session is writable, or to the compatibility notice when it
   * is read-only.
   */
  function selectSession(sessionId) {
    if (sessionId === currentSessionId) return;

    currentSessionId = sessionId;
    renderSessionListOrError();
    syncCurrentSessionUI();
    loadMessages(sessionId);

    if (isMobile()) {
      closeSidebarOnMobile();
      const session = currentSession();
      const writable = isSessionWritable(
        session,
        sessionSendBlocks[sessionId] || null,
      );
      if (session !== null && writable) {
        inputEl.focus();
      } else if (session !== null) {
        sessionCompatibilityNoticeEl.focus();
      }
    } else {
      // The list was rebuilt above — restore focus to the freshly
      // created button for this session (user-initiated switches
      // only; programmatic flows never call selectSession).
      const btn = findSessionButton(
        sessionListEl.querySelectorAll(".session-select-btn"),
        sessionId,
      );
      if (btn !== null && !btn.disabled) {
        btn.focus();
      }
    }
  }

  /** Handle a failed create-session request. */
  async function handleCreateSessionFailure(err) {
    if (err instanceof ApiError && err.status === 422) {
      // The profile no longer exists — no fallback, no local session.
      showStatus(err.message, true);
      await reloadProfiles();
      return;
    }
    showStatus(err.message, true);
  }

  /**
   * Reload the profile list (after a 422, or retry).  Blocks creation
   * while loading; on failure keeps the persistent selector error and
   * disables creation.
   */
  async function reloadProfiles() {
    profilesLoadState = "loading";
    profilesLoadError = null;
    updateControlStates();

    try {
      const data = await fetchProfiles();
      profiles = data;
      profilesLoadState = "ready";
      profilesLoadError = null;
    } catch (err) {
      profiles = [];
      profilesLoadState = "error";
      profilesLoadError = err.message;
    }

    selectedProfileId = resolveSelectedProfileId(
      registry(),
      selectedProfileId,
    );
    renderProfileSelectorContent();
    syncCurrentSessionUI();
  }

  /** Create a new session (called by "New Chat" button). */
  async function newChat() {
    if (isCreatingSession || isSending || isInitializing) return;

    // Capture the profile id at request start so async changes cannot
    // make the UI and the request disagree.
    const profileId = selectedProfileId;
    if (profileId === null) {
      showStatus("No model available. New chats are disabled.", true);
      return;
    }

    let created = false;
    let createdReadOnly = false;

    isCreatingSession = true;
    updateControlStates();

    try {
      showStatus("Creating session...", false);
      const session = await createSessionRequest(profileId);

      // Validate BEFORE modifying local state — an invalid response
      // must never create a local ghost session.
      if (!isValidSessionResponse(session)) {
        showStatus("The server returned an invalid response.", true);
        return;
      }
      for (let i = 0; i < sessions.length; i++) {
        if (sessions[i].id === session.id) {
          showStatus("The server returned an invalid response.", true);
          return;
        }
      }

      clearStatus();

      // The server SessionResponse is the source of truth for model
      // fields — never the selector's current guess.
      sessions.unshift(session);
      currentSessionId = session.id;
      sessionLastMessageId[session.id] = 0;
      sessionLoadRequestId++;
      renderSessionListOrError();
      syncCurrentSessionUI();
      renderEmptyChat();
      clearStatus();
      inputEl.value = "";
      inputEl.style.height = "";
      created = true;
      createdReadOnly = session.llm_profile_status !== "ready";
      closeSidebarOnMobile();
    } catch (err) {
      await handleCreateSessionFailure(err);
    } finally {
      isCreatingSession = false;
      updateControlStates();
      // Focus only after the controls are re-enabled, and only on
      // focusable, visible elements.
      if (created) {
        if (!inputEl.disabled) {
          inputEl.focus();
        } else if (createdReadOnly) {
          if (isMobile() && !sessionCompatibilityNoticeEl.hidden) {
            // Mobile: the sidebar is closed, so focus moves to the
            // visible compatibility notice.
            sessionCompatibilityNoticeEl.focus();
          } else if (!newChatBtn.disabled) {
            // Desktop: explicitly return focus to the re-enabled
            // New Chat button (never rely on natural focus, since it
            // was disabled during the async request).
            newChatBtn.focus();
          }
        }
      }
    }
  }

  /** Ensure a session exists, creating one if necessary. */
  async function ensureSession() {
    if (currentSessionId !== null) return true;

    if (isCreatingSession || isSending || isInitializing) return false;

    const profileId = selectedProfileId;
    if (profileId === null) {
      showStatus("No model available. New chats are disabled.", true);
      return false;
    }

    isCreatingSession = true;
    updateControlStates();

    try {
      showStatus("Creating session...", false);
      const session = await createSessionRequest(profileId);

      // Validate BEFORE modifying local state.
      if (!isValidSessionResponse(session)) {
        showStatus("The server returned an invalid response.", true);
        return false;
      }
      for (let i = 0; i < sessions.length; i++) {
        if (sessions[i].id === session.id) {
          showStatus("The server returned an invalid response.", true);
          return false;
        }
      }

      sessions.unshift(session);
      currentSessionId = session.id;
      sessionLastMessageId[session.id] = 0;
      sessionLoadRequestId++;
      renderSessionListOrError();
      syncCurrentSessionUI();
      clearStatus();
      return true;
    } catch (err) {
      await handleCreateSessionFailure(err);
      return false;
    } finally {
      isCreatingSession = false;
      updateControlStates();
    }
  }

  async function handleDeleteSession(sessionId, event) {
    event.stopPropagation();
    if (isSending) return;

    if (!confirm("Delete this conversation?")) return;

    try {
      await deleteSessionRequest(sessionId);
    } catch (err) {
      // 404 → session already gone on server, remove locally
      if (err instanceof ApiError && err.status === 404) {
        removeSessionLocally(sessionId);
        return;
      }
      // Other errors — keep list, show error
      showStatus(err.message, true);
      return;
    }

    removeSessionLocally(sessionId);
  }

  /* ---- Rename session ---------------------------------------------- */

  function startRename(sessionId) {
    if (isSending || isRenaming || isInitializing) return;

    isRenaming = true;
    renamingSessionId = sessionId;
    updateControlStates();
    renderSessionListOrError();
  }

  async function saveRename(sessionId, renameInputEl) {
    if (isRenameSaving) return;  // prevent double-submit

    var rawTitle = renameInputEl.value;
    if (!rawTitle.trim()) {
      showStatus("Title must not be blank.", true);
      renameInputEl.focus();
      return;
    }

    isRenameSaving = true;
    updateControlStates();  // disable input, Save, Cancel

    try {
      var updated = await renameSessionRequest(sessionId, rawTitle);

      // Validate BEFORE overwriting anything: a malformed response or
      // an id mismatch must not replace a local session, exit editing
      // or drop the draft.
      if (!isValidSessionResponse(updated) || updated.id !== sessionId) {
        isRenameSaving = false;
        updateControlStates();  // re-enable editing controls
        showStatus("The server returned an invalid response.", true);
        // Stay in Editing state — rename input is still connected.
        renameInputEl.focus();
        return;
      }

      // Use server-normalised title from the full SessionResponse
      for (var i = 0; i < sessions.length; i++) {
        if (sessions[i].id === sessionId) {
          sessions[i] = updated;
          break;
        }
      }
      // Move renamed session to top (updated_at has changed)
      sessions = [updated].concat(
        sessions.filter(function (s) { return s.id !== sessionId; })
      );
      clearStatus();
      isRenaming = false;
      renamingSessionId = null;
      isRenameSaving = false;
      updateControlStates();
      renderSessionListOrError();
      syncCurrentSessionUI();
      inputEl.focus();  // explicitly focus chat message input
    } catch (err) {
      // sessions was never modified — no rollback needed
      isRenameSaving = false;
      updateControlStates();  // re-enable input, Save, Cancel
      showStatus(err.message, true);
      // Stay in Editing state — do NOT call renderSessionListOrError()
      renameInputEl.focus();
    }
  }

  function cancelRename() {
    // No optimistic update was made, no restore needed
    isRenaming = false;
    renamingSessionId = null;
    isRenameSaving = false;
    updateControlStates();
    renderSessionListOrError();
    inputEl.focus();  // explicitly focus chat message input
  }

  /* ---- Send message ------------------------------------------------ */

  async function sendMessage(text) {
    if (isSending || isInitializing || isCreatingSession) return;

    // Auto-create session on first send
    if (currentSessionId === null) {
      if (!registryUsable()) return;  // defensive — UI already blocked
      const ok = await ensureSession();
      if (!ok) return;  // create failed or already in progress
    } else {
      // Defensive writability check — the UI is already disabled for
      // read-only sessions.
      if (!currentSessionWritable()) return;
    }

    const sendingSessionId = currentSessionId;
    const lastMessageIdBeforeSend = sessionLastMessageId[sendingSessionId] || 0;

    setSendingState(true);
    showStatus("Generating...", false);

    try {
      const data = await sendSessionMessageRequest(sendingSessionId, text);

      // Validate BEFORE touching any UI or state.  A malformed 2xx
      // body means the message may already be saved on the server —
      // route it through the same uncertain-send inspection as a
      // network error, never through the plain success path.
      if (!isValidSendMessageResponse(data, sendingSessionId)) {
        await handleUncertainSendOutcome(
          sendingSessionId, text, lastMessageIdBeforeSend,
        );
        return;
      }

      // --- Success --------------------------------------------------
      clearStatus();
      inputEl.value = "";
      inputEl.style.height = "";

      if (currentSessionId === sendingSessionId) {
        appendMessage(data.user_message.role, data.user_message.content);
        appendMessage(data.assistant_message.role, data.assistant_message.content);
        scrollMessagesToBottom();
      }

      sessionLastMessageId[sendingSessionId] = data.assistant_message.id;

      // Refresh session list (sending session moves to top).  On
      // failure the rendered messages stay and a stale warning shows.
      await refreshSessions(true);
    } catch (err) {
      await handleSendFailure(err, sendingSessionId, text, lastMessageIdBeforeSend);
      return;
    } finally {
      setSendingState(false);
    }
  }

  /**
   * Re-fetch the messages for a session and analyse what actually
   * happened to an uncertain send.
   *
   * Pure analysis: no DOM, no status text, no input clearing, no
   * sessionLastMessageId writes, no refreshSessions call.
   *
   * @returns {{status: string}} One of:
   *   {status: "succeeded", messages, lastMessageId}
   *   {status: "user_saved", messages, userMessageId}
   *   {status: "unknown"}
   */
  async function inspectUncertainSend(
    sendingSessionId,
    originalText,
    lastMessageIdBeforeSend,
  ) {
    let msgs;
    try {
      msgs = await fetchMessages(sendingSessionId);
    } catch (_) {
      return { status: "unknown" };
    }

    // A malformed list must never reach findSentMessages — fail closed.
    if (!isValidMessageList(msgs, sendingSessionId)) {
      return { status: "unknown" };
    }

    const result = findSentMessages(
      msgs, lastMessageIdBeforeSend, originalText,
    );
    if (result.userIdx < 0) {
      return { status: "unknown" };
    }

    if (result.hasAssistant) {
      // The list is validated and strictly increasing, so the last
      // element carries the highest known message id.
      return {
        status: "succeeded",
        messages: msgs,
        lastMessageId: msgs[msgs.length - 1].id,
      };
    }

    return {
      status: "user_saved",
      messages: msgs,
      userMessageId: result.userMessageId,
    };
  }

  /**
   * Apply the outcome of an uncertain send to the UI.  Shared by
   * network errors (status 0) and malformed 2xx send responses — one
   * recovery implementation, no duplicated logic.
   *
   * - succeeded  → render history on the sending session, clear the
   *                (now saved) draft, update the last id, refresh.
   * - user_saved → clear the saved draft, update the user id, show
   *                the saved-but-unanswered state.
   * - unknown    → keep the draft and show an uncertain warning;
   *                never assume success or failure.
   *
   * Rendering still guards on ``currentSessionId === sendingSessionId``
   * because switching sessions is allowed during a send.
   */
  async function handleUncertainSendOutcome(
    sendingSessionId,
    originalText,
    lastMessageIdBeforeSend,
  ) {
    const outcome = await inspectUncertainSend(
      sendingSessionId, originalText, lastMessageIdBeforeSend,
    );

    if (outcome.status === "succeeded") {
      if (currentSessionId === sendingSessionId) {
        renderMessages(outcome.messages);
        scrollMessagesToBottom();
        clearStatus();
      }
      inputEl.value = "";
      inputEl.style.height = "";
      if (outcome.lastMessageId > 0) {
        sessionLastMessageId[sendingSessionId] = outcome.lastMessageId;
      }
      await refreshSessions(true);
      return;
    }

    if (outcome.status === "user_saved") {
      sessionLastMessageId[sendingSessionId] = outcome.userMessageId;
      inputEl.value = "";
      inputEl.style.height = "";
      if (currentSessionId === sendingSessionId) {
        showStatus(
          "Your message was saved, but the assistant could not respond.",
          true,
        );
        // outcome.messages was already validated by
        // inspectUncertainSend — no second network request.
        if (outcome.messages.length === 0) {
          renderEmptyChat();
        } else {
          renderMessages(outcome.messages);
        }
      } else {
        showStatus(
          "Message saved in session #" + sendingSessionId +
          ", but the assistant could not respond.",
          true,
        );
      }
      return;
    }

    // unknown — keep the draft; never append, never update the last
    // id, never claim the request failed.
    showStatus(
      "Request status is uncertain. Review the conversation before resending.",
      true,
    );
  }

  /**
   * Handle a failed send — strategy depends on the error type.
   *
   * 409 / 503 → model compatibility conflict.  Handled as an early
   *              branch: a temporary block is set immediately, the
   *              session becomes read-only, the draft is kept, and
   *              only that session is refreshed precisely.  Never
   *              routed through network-recovery logic.
   * 502 / 504  → user message was saved (Phase 1 committed).
   *               Clear input, re-sync messages from DB.
   * 404 / 422  → user message was NOT saved.
   *               Keep input, show error.
   * Network (0) → uncertain.  Inspect the message log and decide by
   *               the message-ID boundary.
   *
   * *lastMessageIdBeforeSend* is the highest known message id in the
   * session right before the API call.  After a network error we only
   * look at messages with id > that boundary — this handles the case
   * where the user sends the same text twice consecutively.
   */
  async function handleSendFailure(
    err,
    sendingSessionId,
    originalText,
    lastMessageIdBeforeSend,
  ) {
    const errStatus = (err instanceof ApiError) ? err.status : 0;

    // --- 409/503: model compatibility — early branch --------------------
    if (errStatus === 409 || errStatus === 503) {
      // Never guess the exact 409 status from the English detail; use
      // a generic conflict block.  503 maps to profile_unavailable.
      sessionSendBlocks[sendingSessionId] =
        errStatus === 503 ? "profile_unavailable" : "conflict";

      // Immediately make the session read-only — do not wait for the
      // refresh to succeed.
      syncCurrentSessionUI();
      showStatus(err.message, true);
      // Draft stays in the textarea.

      // Precisely refresh only the affected session.  If the user has
      // switched to another session, the block is still recorded for
      // sendingSessionId and the notice is not rendered elsewhere.
      await refreshOneSessionCompatibility(sendingSessionId);
      return;
    }

    let isSaved = false;

    // --- Determine save status ----------------------------------------
    if (errStatus === 502 || errStatus === 504) {
      isSaved = true;
    } else if (errStatus === 0) {
      // Network error — inspect the log by ID boundary.  The shared
      // recovery handles succeeded / user-saved / unknown.
      await handleUncertainSendOutcome(
        sendingSessionId, originalText, lastMessageIdBeforeSend,
      );
      return;
    }
    // 404, 422, other HTTP → isSaved stays false

    // --- Act based on save status -------------------------------------
    if (currentSessionId === sendingSessionId) {
      if (isSaved) {
        showStatus(
          "Your message was saved, but the assistant could not respond.",
          true,
        );
        inputEl.value = "";
        inputEl.style.height = "";

        // Reload history to show the saved user message.  Only a
        // validated list may reach renderMessages; an invalid body
        // must not replace the current message area and must not be
        // mistaken for a plain send failure.
        try {
          const history = await fetchMessages(sendingSessionId);
          if (currentSessionId === sendingSessionId) {
            if (isValidMessageList(history, sendingSessionId)) {
              if (history.length === 0) {
                renderEmptyChat();
              } else {
                renderMessages(history);
              }
            } else {
              showStatus(
                "Your message was saved, but the conversation history " +
                "could not be refreshed.",
                true,
              );
            }
          }
        } catch (_) {
          /* best-effort sync */
        }
      } else {
        // 404, 422, or other non-saved errors
        if (errStatus === 404) {
          await refreshSessions(true);
        }
        showStatus(err.message || "Something went wrong.", true);
        // Keep input for retry
      }
    } else {
      // User switched away — just show a brief warning
      showStatus(
        "Failed to send message in session #" + sendingSessionId + ".",
        true,
      );
    }
  }

  /* ---- Mobile sidebar ---------------------------------------------- */

  function isMobile() {
    return window.innerWidth <= 767;
  }

  function toggleSidebar() {
    sidebarEl.classList.toggle("collapsed");
    const expanded = !sidebarEl.classList.contains("collapsed");
    sidebarToggleEl.setAttribute("aria-expanded", String(expanded));
  }

  function closeSidebarOnMobile() {
    if (isMobile()) {
      sidebarEl.classList.add("collapsed");
      sidebarToggleEl.setAttribute("aria-expanded", "false");
    }
  }

  /* ---- Event bindings ---------------------------------------------- */

  // Send button
  sendBtn.addEventListener("click", function () {
    if (isSending || isInitializing || isCreatingSession) return;
    const text = inputEl.value.trim();
    if (!text) return;
    sendMessage(text);
  });

  // Textarea: Enter to send, Shift+Enter for newline
  inputEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendBtn.click();
    }
  });

  // Auto-resize textarea (clamped at CSS max-height)
  inputEl.addEventListener("input", function () {
    inputEl.style.height = "";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + "px";
  });

  // New Chat button
  newChatBtn.addEventListener("click", function () {
    newChat();
  });

  // Model selector: only records the choice for the NEXT new chat.
  // No option rebuild, no focus move, no current-session change.
  profileSelectEl.addEventListener("change", function () {
    selectedProfileId = profileSelectEl.value;
  });

  // Mobile sidebar toggle
  sidebarToggleEl.addEventListener("click", function () {
    toggleSidebar();
  });

  // Close sidebar when clicking chat area on mobile
  document.getElementById("chatArea").addEventListener("click", function (e) {
    if (isMobile() && !sidebarEl.classList.contains("collapsed")) {
      if (e.target !== sidebarToggleEl &&
          !sidebarToggleEl.contains(e.target)) {
        closeSidebarOnMobile();
      }
    }
  });

  // Keep aria-expanded in sync when the window is resized
  window.addEventListener("resize", function () {
    if (isMobile()) {
      var expanded = !sidebarEl.classList.contains("collapsed");
      sidebarToggleEl.setAttribute("aria-expanded", String(expanded));
    } else {
      sidebarToggleEl.setAttribute("aria-expanded", "true");
    }
  });

  /* ---- Init -------------------------------------------------------- */

  async function init() {
    isInitializing = true;
    updateControlStates();

    // Mobile: start with sidebar collapsed
    if (isMobile()) {
      sidebarEl.classList.add("collapsed");
      sidebarToggleEl.setAttribute("aria-expanded", "false");
    } else {
      sidebarToggleEl.setAttribute("aria-expanded", "true");
    }

    renderSessionListOrError();
    renderWelcome();
    showStatus("Loading...", false);

    // Load profiles and sessions in parallel — results are handled
    // independently so one failure never blocks the other.
    const [profilesResult, sessionsResult] = await Promise.allSettled([
      fetchProfiles(),
      fetchSessions(),
    ]);

    if (profilesResult.status === "fulfilled") {
      profiles = profilesResult.value;
      profilesLoadState = "ready";
      profilesLoadError = null;
    } else {
      profiles = [];
      profilesLoadState = "error";
      profilesLoadError =
        profilesResult.reason && profilesResult.reason.message
          ? profilesResult.reason.message
          : "Failed to load models.";
    }
    selectedProfileId = resolveSelectedProfileId(
      registry(),
      selectedProfileId,
    );
    renderProfileSelectorContent();

    if (sessionsResult.status === "fulfilled") {
      if (isValidSessionList(sessionsResult.value)) {
        sessions = sessionsResult.value;
        sessionsLoadState = "ready";
        sessionsEverLoaded = true;
        sessionsLoadError = null;
      } else {
        sessions = [];
        sessionsLoadState = "error";
        sessionsLoadError = "The server returned an invalid response.";
      }
    } else {
      sessions = [];
      sessionsLoadState = "error";
      sessionsLoadError =
        sessionsResult.reason && sessionsResult.reason.message
          ? sessionsResult.reason.message
          : "Failed to load conversations.";
    }

    // Decide the initial selection BEFORE rendering the list so the
    // first highlight, title, badge and message area are consistent.
    if (sessionsLoadState === "ready" && sessions.length > 0) {
      currentSessionId = sessions[0].id;
    }

    renderSessionListOrError();

    if (sessionsLoadState === "ready") {
      if (sessions.length > 0) {
        syncCurrentSessionUI();
        await loadMessages(currentSessionId);
      } else {
        renderWelcome();
        clearStatus();
      }
    } else {
      renderWelcome();
      if (sessionsLoadError) showStatus(sessionsLoadError, true);
    }

    isInitializing = false;
    updateControlStates();

    // Focus rule: only when the page has no explicit user focus and
    // typing is actually possible.
    const canType = currentSessionId === null
      ? registryUsable()
      : currentSessionWritable();
    if (document.activeElement === document.body && canType) {
      inputEl.focus();
    }
  }

  init();
})();
