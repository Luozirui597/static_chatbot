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
  const currentProfileBarEl = document.getElementById("currentProfileBar");
  const currentProfileSelectEl =
    document.getElementById("currentProfileSelect");
  const applyProfileBtn = document.getElementById("applyProfileBtn");
  const currentProfileStatusEl =
    document.getElementById("currentProfileStatus");
  const profileSwitchDialogEl =
    document.getElementById("profileSwitchDialog");
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
  let selectedProfileId = null;    // string | null (new chats only)
  let profilesLoadState = "loading"; // "loading" | "ready" | "error"
  let profilesLoadError = null;    // persistent — #profileSelectorStatus

  // Session list state
  let sessionsLoadState = "loading"; // "loading" | "ready" | "error"
  let sessionsEverLoaded = false;    // ever loaded the full list
  let sessionsLoadError = null;      // persistent — #sessionListStatus

  // Temporary per-session send blocks set after 409/503 responses.
  // sessionId -> "conflict" | "profile_unavailable"
  let sessionSendBlocks = {};

  // Current-session model switcher state (fully independent from the
  // new-chat selector above).
  let currentProfileDraftId = null;   // string | null (current chat only)
  let isProfileSwitching = false;     // busy — covers outcome application
  let isDeletingSession = false;      // delete busy (programmatic guard)
  let deletingSessionId = null;
  let profileSwitchGeneration = 0;    // monotonic token / generation

  // Per-session records.  Keys are positive safe integer ids.
  const sessionHasMessages = Object.create(null);      // true | false | undefined
  const sessionSwitchUncertain = Object.create(null);  // uncertain records
  const profileSwitchStatusBySession = Object.create(null); // {text, isError}

  // Controller + confirmer, initialised once in init().
  let switchController = null;
  let profileSwitchInitializationError = null;   // string | null

  const UNCERTAIN_REAPPLY_TEXT =
    "The previous model switch could not be confirmed. Apply again to " +
    "check the current binding before retrying.";

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

  /** Whether the current session (or none) can accept a message.
   *  An uncertain-switch record for the current session always blocks
   *  sending — even when the cached llm_profile_status is "ready". */
  function currentSessionWritable() {
    return isSessionWritable(
      currentSession(),
      sessionSendBlocks[currentSessionId] || null,
      sessionSwitchUncertain[currentSessionId],
    );
  }

  /** kind of a profile — only from a structurally valid registry. */
  function lookupProfileKind(profileId) {
    return profileKindFromRegistry(registry(), profileId);
  }

  /** Find a session in the local list by id (no insertion). */
  function findSessionInList(sessionId) {
    if (!Number.isSafeInteger(sessionId)) return null;
    for (let i = 0; i < sessions.length; i++) {
      if (sessions[i].id === sessionId) return sessions[i];
    }
    return null;
  }

  /** Per-session record keys must be positive safe integers. */
  function isValidSessionIdKey(id) {
    return Number.isSafeInteger(id) && id >= 1;
  }

  /** Set the ordinary switch status for ONE session (target-scoped). */
  function setProfileSwitchStatus(sessionId, text, isError) {
    if (!isValidSessionIdKey(sessionId)) return;
    profileSwitchStatusBySession[sessionId] = {
      text: typeof text === "string" ? text : "",
      isError: isError === true,
    };
    renderCurrentProfileStatus();
  }

  /** Clear the ordinary switch status for ONE session. */
  function clearProfileSwitchStatus(sessionId) {
    if (!isValidSessionIdKey(sessionId)) return;
    delete profileSwitchStatusBySession[sessionId];
    renderCurrentProfileStatus();
  }

  /** Render the status area for the CURRENT session only.  The
   * uncertain record has the highest display priority; ordinary
   * per-session status comes second; otherwise the DOM is cleared. */
  function renderCurrentProfileStatus() {
    let text = "";
    let isError = false;

    if (profileSwitchInitializationError !== null &&
        currentSessionId !== null) {
      text = profileSwitchInitializationError;
      isError = true;
    } else if (currentSessionId !== null &&
               sessionSwitchUncertain[currentSessionId] !== undefined) {
      text = UNCERTAIN_REAPPLY_TEXT;
      isError = true;
    } else if (currentSessionId !== null &&
               profileSwitchStatusBySession[currentSessionId] !== undefined) {
      text = profileSwitchStatusBySession[currentSessionId].text;
      isError = profileSwitchStatusBySession[currentSessionId].isError;
    }

    currentProfileStatusEl.textContent = text;
    currentProfileStatusEl.className =
      "current-profile-status" + (isError ? " error" : "");
  }

  /** Apply the outcome's ordinary-status effect to the TARGET session
   * only — independent of which session is currently visible. */
  function applyTargetProfileSwitchStatus(targetSessionId, outcome, plan) {
    if (!isValidSessionIdKey(targetSessionId)) return;
    switch (outcome.status) {
      case "switched":
      case "cancelled":
        clearProfileSwitchStatus(targetSessionId);
        break;
      case "not_changed":
      case "validation_error":
      case "failed":
        if (typeof plan.showStatus === "string" &&
            plan.showStatus.trim() !== "") {
          setProfileSwitchStatus(targetSessionId, plan.showStatus, true);
        }
        break;
      case "uncertain":
        // The uncertain record lives in sessionSwitchUncertain; the
        // persistent hint is rendered from it (single source).
        break;
      case "busy":
      default:
        break;
    }
  }

  /** Raw history state → enum for needsRemoteHistoryConfirmation. */
  function historyStateFor(sessionId) {
    return historyStateFromValue(sessionHasMessages[sessionId]);
  }

  /** Write the has-messages state for a session (undefined = unknown). */
  function setHistoryState(sessionId, state) {
    if (!isValidSessionIdKey(sessionId)) return;
    if (state === undefined) {
      delete sessionHasMessages[sessionId];
    } else {
      sessionHasMessages[sessionId] = state === true;
    }
  }

  /* ---- Custom error ------------------------------------------------ */

  class ApiError extends Error {
    constructor(message, status = 0, code = null) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
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
      let code = null;
      try {
        const body = await response.json();
        if (typeof body.detail === "string") {
          detail = body.detail;
        } else if (
          body !== null && typeof body === "object" &&
          !Array.isArray(body) &&
          typeof body.detail === "object" && body.detail !== null &&
          !Array.isArray(body.detail) &&
          typeof body.detail.code === "string" &&
          typeof body.detail.message === "string"
        ) {
          // Structured error body (e.g. the model-switch 409) — the
          // stable code is preserved; callers never match on text.
          code = body.detail.code;
          detail = body.detail.message;
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
      throw new ApiError(detail, response.status, code);
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

  /* ---- Model switch API wrappers ----------------------------------- */

  /** PATCH the switch endpoint.  Success returns parsed JSON (validated
   * by the controller); failures throw normalised error objects with
   * failureKind "http" | "network" | "response_parse". */
  async function switchProfileRequest(sessionId, profileId, ack) {
    let response;
    try {
      response = await fetch("/api/sessions/" + sessionId + "/llm-profile", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSwitchSessionProfilePayload(profileId, ack)),
      });
    } catch (_) {
      throw {
        failureKind: "network", status: 0, code: null,
        message: "Network error. Please check your connection.",
      };
    }

    if (!response.ok) {
      let code = null;
      let message = "Something went wrong. Please try again.";
      let body = null;
      try { body = await response.json(); } catch (_) { body = null; }
      const structured = parseRemoteHistoryAckRequired(body);
      if (structured !== null) {
        code = structured.code;
        message = structured.message;
      } else if (body !== null && typeof body === "object" &&
                 !Array.isArray(body) && typeof body.detail === "string") {
        message = body.detail;
      } else if (body !== null && typeof body === "object" &&
                 !Array.isArray(body) && Array.isArray(body.detail) &&
                 body.detail.length > 0 && body.detail[0] &&
                 typeof body.detail[0].msg === "string") {
        message = body.detail[0].msg;
      }
      throw {
        failureKind: "http", status: response.status, code: code,
        message: message,
      };
    }

    try {
      return await response.json();
    } catch (_) {
      throw {
        failureKind: "response_parse", status: 0, code: null,
        message: "The server returned an unreadable response.",
      };
    }
  }

  /** GET a single session as parsed JSON; errors follow the same
   * normalised contract as switchProfileRequest. */
  async function fetchOneSessionRaw(sessionId) {
    let response;
    try {
      response = await fetch("/api/sessions/" + sessionId);
    } catch (_) {
      throw {
        failureKind: "network", status: 0, code: null,
        message: "Network error. Please check your connection.",
      };
    }
    if (!response.ok) {
      let body = null;
      try { body = await response.json(); } catch (_) { body = null; }
      let message = "Something went wrong. Please try again.";
      if (body !== null && typeof body === "object" && !Array.isArray(body) &&
          typeof body.detail === "string") {
        message = body.detail;
      }
      throw {
        failureKind: "http", status: response.status, code: null,
        message: message,
      };
    }
    try {
      return await response.json();
    } catch (_) {
      throw {
        failureKind: "response_parse", status: 0, code: null,
        message: "The server returned an unreadable response.",
      };
    }
  }

  /** Structured single-session fetch with NO cache side effects. */
  async function fetchAndValidateOneSession(sessionId) {
    let raw;
    try {
      raw = await fetchOneSessionRaw(sessionId);
    } catch (err) {
      if (err && typeof err === "object" && err.failureKind === "http" &&
          err.status === 404) {
        return { status: "not_found" };
      }
      return { status: "failed", message: err && err.message ? err.message : "" };
    }
    if (!isValidSessionResponse(raw) || raw.id !== sessionId) {
      return { status: "invalid_response", message: "Invalid session response." };
    }
    return { status: "ok", session: raw };
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
  function appendPlaceholderOption(selectEl) {
    const option = document.createElement("option");
    option.value = "";
    option.disabled = true;
    option.selected = true;
    option.textContent = "Select a model";
    selectEl.appendChild(option);
  }

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
      } else {
        // Explicit placeholder — never silently fall back to the
        // first profile.
        appendPlaceholderOption(profileSelectEl);
        profileSelectEl.value = "";
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
    const blockSessionActions =
      blockBase || isCreatingSession || isSending ||
      isProfileSwitching || isDeletingSession;
    const usable = registryUsable();
    const writable = currentSessionWritable();

    const blockCreate =
      blockSessionActions || !usable || selectedProfileId === null;
    const blockProfileSelect =
      blockSessionActions || !usable || profiles.length <= 1;
    const blockSend =
      blockBase || isCreatingSession || isSending ||
      isProfileSwitching || isDeletingSession || !writable ||
      (currentSessionId === null && !usable);

    const switchInitialized =
      switchController !== null && profileSwitchInitializationError === null;
    const blockCurrentProfile = blockSessionActions || !usable ||
      !switchInitialized || currentSessionId === null;

    sendBtn.disabled = blockSend;
    inputEl.disabled = blockSend;
    newChatBtn.disabled = blockCreate;
    profileSelectEl.disabled = blockProfileSelect;
    currentProfileSelectEl.disabled = blockCurrentProfile;
    applyProfileBtn.disabled = blockCurrentProfile || !applyEnabled();

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

  /**
   * Apply a profile-draft plan to the current-session model controls.
   *
   * ``plan`` comes from ``planProfileDraftForSelection``.  An
   * unchanged selection keeps the draft untouched and never rebuilds
   * the dropdown (no option churn, no focus loss); a changed
   * selection re-renders the bar — the ONLY place that shows or hides
   * it — which also re-renders the status area.  With no current
   * session the bar hides and the status text clears, while the
   * per-session records of other sessions stay untouched.
   */
  function applySelectionDraftPlan(plan) {
    currentProfileDraftId = plan.draftId;
    if (plan.selectionChanged) {
      renderCurrentProfileBar();
    } else {
      renderCurrentProfileStatus();
    }
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

    // Server list is authoritative for temporary blocks and other
    // per-session records.
    const presentIds = {};
    for (const s of sessions) {
      presentIds[s.id] = true;
      delete sessionSendBlocks[s.id];
    }
    for (const sid of Object.keys(sessionSendBlocks)) {
      if (!presentIds[sid]) delete sessionSendBlocks[sid];
    }
    for (const sid of Object.keys(sessionSwitchUncertain)) {
      if (!presentIds[sid]) delete sessionSwitchUncertain[sid];
    }
    for (const sid of Object.keys(sessionHasMessages)) {
      if (!presentIds[sid]) delete sessionHasMessages[sid];
    }
    for (const sid of Object.keys(profileSwitchStatusBySession)) {
      if (!presentIds[sid]) delete profileSwitchStatusBySession[sid];
    }

    // Decide the final selection BEFORE rendering the list so the
    // highlight, title, badge and message area all point at the same
    // session.
    const previousSelectionId = currentSessionId;
    const next = resolveNextSelectionId(
      sessions,
      currentSessionId,
      preserveSelection,
    );
    const selectionChanged = next.changed;
    currentSessionId = next.selectionId;

    // The model controls must follow the selection: a changed
    // selection re-derives the draft from the NEW session's server
    // binding (an old session's un-applied draft is never inherited),
    // while an unchanged selection keeps any un-applied draft — an
    // ordinary refresh never overwrites a user choice and never
    // rebuilds the dropdown.
    applySelectionDraftPlan(planProfileDraftForSelection({
      sessions: sessions,
      previousSessionId: previousSelectionId,
      nextSessionId: currentSessionId,
      previousDraftId: currentProfileDraftId,
      registry: registry(),
    }));

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
    // Side-effect-free structured fetch first; the cache is only ever
    // replaced when the session still exists locally — a deleted
    // session is never re-inserted (no ghost sessions).
    const result = await fetchAndValidateOneSession(sessionId);
    if (result.status === "not_found") {
      removeSessionLocally(sessionId);
      return false;
    }
    if (result.status !== "ok") {
      // failed / invalid_response — keep the block and the cached
      // session, never overwrite anything.
      return false;
    }

    const cachePlan = planSessionCacheUpdate({
      sessions: sessions,
      requestedSessionId: sessionId,
      fresh: result.session,
    });
    if (cachePlan.kind !== "replace") {
      return false;
    }
    sessions = cachePlan.sessions;

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
    delete sessionSwitchUncertain[sessionId];
    delete sessionHasMessages[sessionId];
    delete profileSwitchStatusBySession[sessionId];

    if (sessionId === currentSessionId) {
      sessionLoadRequestId++;
      clearMessages();
      clearStatus();

      // The model controls follow the new selection — the deleted
      // session's draft is never inherited; with no session left the
      // bar hides and the status text clears.  Deleting a NON-current
      // session never reaches this branch, so its draft and focus
      // stay untouched.
      const previousSelectionId = currentSessionId;
      currentSessionId = sessions.length > 0 ? sessions[0].id : null;
      applySelectionDraftPlan(planProfileDraftForSelection({
        sessions: sessions,
        previousSessionId: previousSelectionId,
        nextSessionId: currentSessionId,
        previousDraftId: currentProfileDraftId,
        registry: registry(),
      }));
      syncCurrentSessionUI();

      if (currentSessionId !== null) {
        loadMessages(currentSessionId);
      } else {
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

    setHistoryState(sessionId, data.length > 0);
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
    // Recompute the draft from the server binding of the newly visible
    // session; never carry over another session's un-applied draft.
    const session = findSessionInList(sessionId);
    currentProfileDraftId = session !== null
      ? resolveSessionProfileDraft(session, registry())
      : null;
    renderSessionListOrError();
    syncCurrentSessionUI();
    renderCurrentProfileBar();
    renderCurrentProfileStatus();
    loadMessages(sessionId);

    if (isMobile()) {
      closeSidebarOnMobile();
      const session = currentSession();
      const writable = isSessionWritable(
        session,
        sessionSendBlocks[sessionId] || null,
        sessionSwitchUncertain[sessionId],
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
  async function reloadProfiles({ allowDefaultFallback = true } = {}) {
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
      profilesLoadError = err && err.message ? err.message : "Failed to load models.";
      // On failure keep BOTH selectors' current values; the controls
      // are disabled via profilesLoadState === "error".
      renderProfileSelectorContent();
      renderCurrentProfileBar();
      renderCurrentProfileStatus();
      syncCurrentSessionUI();
      return;
    }

    // Read the truly visible state AFTER the await — never capture
    // drafts before the request.
    const visibleSessionIdAtApply = currentSessionId;
    const visibleDraftIdAtApply = currentProfileDraftId;
    const newRegistry = registry();

    if (allowDefaultFallback) {
      selectedProfileId = resolveSelectedProfileId(
        newRegistry, selectedProfileId,
      );
    } else {
      selectedProfileId = preserveSelectedProfileWithoutFallback(
        newRegistry, selectedProfileId,
      );
    }

    const draftStillValid =
      typeof visibleDraftIdAtApply === "string" &&
      captureRequestedProfile(newRegistry, visibleDraftIdAtApply) !== null;
    if (!draftStillValid) {
      const visibleAtApply = findSessionInList(visibleSessionIdAtApply);
      currentProfileDraftId = visibleAtApply !== null
        ? resolveSessionProfileDraft(visibleAtApply, newRegistry)
        : null;
    }
    // draftStillValid → keep the visible draft untouched (e.g. an
    // un-applied choice made on another session while reloading).

    renderProfileSelectorContent();
    renderCurrentProfileBar();
    renderCurrentProfileStatus();
    syncCurrentSessionUI();
  }

  /** Create a new session (called by "New Chat" button). */
  async function newChat() {
    if (isCreatingSession || isSending || isInitializing ||
        isProfileSwitching || isDeletingSession) return;

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
      setHistoryState(session.id, false);        // brand-new session
      currentProfileDraftId = session.llm_profile_id;
      sessionLoadRequestId++;
      renderSessionListOrError();
      syncCurrentSessionUI();
      renderCurrentProfileBar();
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

    if (isCreatingSession || isSending || isInitializing ||
        isProfileSwitching || isDeletingSession) return false;

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
      setHistoryState(session.id, false);
      currentProfileDraftId = session.llm_profile_id;
      sessionLoadRequestId++;
      renderSessionListOrError();
      syncCurrentSessionUI();
      renderCurrentProfileBar();
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
    if (isSending || isProfileSwitching || isDeletingSession ||
        isCreatingSession) return;

    if (!confirm("Delete this conversation?")) return;

    // Enter the delete busy state after confirmation and BEFORE the
    // first await — double clicks can only produce one DELETE.
    isDeletingSession = true;
    deletingSessionId = sessionId;
    updateControlStates();

    try {
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
    } finally {
      isDeletingSession = false;
      deletingSessionId = null;
      updateControlStates();
    }
  }

  /* ---- Rename session ---------------------------------------------- */

  function startRename(sessionId) {
    if (isSending || isRenaming || isInitializing ||
        isProfileSwitching || isDeletingSession) return;

    isRenaming = true;
    renamingSessionId = sessionId;
    updateControlStates();
    renderSessionListOrError();
  }

  async function saveRename(sessionId, renameInputEl) {
    if (isRenameSaving) return;  // prevent double-submit
    if (isProfileSwitching || isDeletingSession) return;

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
    if (isSending || isInitializing || isCreatingSession ||
        isProfileSwitching || isDeletingSession) return;

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
      setHistoryState(sendingSessionId, true);   // a user message is saved

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
      setHistoryState(sendingSessionId, true);
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
      setHistoryState(sendingSessionId, true);
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
        setHistoryState(sendingSessionId, true);
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

  /* ---- Current-session model switcher ------------------------------ */

  /** Render the options and draft of the current-session control bar. */
  function renderCurrentProfileBar() {
    const reg = registry();

    if (currentSessionId === null) {
      currentProfileBarEl.hidden = true;
      // Clear the status area too — the null-selection render must
      // leave no stale visible text or error class behind.  The
      // shared renderer only reads the per-session maps, so other
      // sessions' records stay untouched and the next selection
      // restores the right hint from them.
      renderCurrentProfileStatus();
      return;
    }
    currentProfileBarEl.hidden = false;

    currentProfileSelectEl.replaceChildren();
    if (reg.status === "valid") {
      for (const p of reg.profiles) {
        const option = document.createElement("option");
        option.value = p.id;
        option.textContent = p.label;
        currentProfileSelectEl.appendChild(option);
      }
      if (currentProfileDraftId !== null) {
        currentProfileSelectEl.value = currentProfileDraftId;
      } else {
        // Explicit placeholder — no silent fallback to the first
        // profile for a missing/unavailable binding.
        appendPlaceholderOption(currentProfileSelectEl);
        currentProfileSelectEl.value = "";
      }
    } else {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No models available";
      currentProfileSelectEl.appendChild(option);
      currentProfileSelectEl.value = "";
    }

    renderCurrentProfileStatus();
  }

  /** Whether Apply is available for the current session/draft.  An
   *  uncertain-switch record keeps Apply available for convergence
   *  even when the draft equals the cached binding. */
  function applyEnabled() {
    if (switchController === null ||
        profileSwitchInitializationError !== null) return false;
    if (isProfileSwitching || isDeletingSession || isSending ||
        isCreatingSession || isRenaming || isRenameSaving ||
        isInitializing) return false;
    if (registry().status !== "valid") return false;
    const session = currentSession();
    if (session === null) return false;
    return canApplySessionProfileWithUncertain({
      session: session,
      registry: registry(),
      draftProfileId: currentProfileDraftId,
      uncertain: sessionSwitchUncertain[session.id] !== undefined,
    });
  }

  /** Real <dialog> adapter matching createRemoteHistoryConfirmer. */
  function createProfileSwitchDialogAdapter() {
    const dlg = profileSwitchDialogEl;
    const cancelBtn = document.getElementById("psd-cancel");
    const continueBtn = document.getElementById("psd-continue");
    const bodyEl = document.getElementById("psd-body");

    function on(target, event, cb) {
      function handler(e) {
        if (event === "cancel") e.preventDefault();
        cb();
      }
      target.addEventListener(event, handler);
      return function () { target.removeEventListener(event, handler); };
    }

    return {
      showModal: function () { dlg.showModal(); },
      close: function () { if (dlg.open) dlg.close(); },
      setMessage: function (text) { bodyEl.textContent = text; },
      focusInitial: function () { cancelBtn.focus(); },
      isConnected: function () { return dlg.isConnected; },
      onDialogCancel: function (cb) { return on(dlg, "cancel", cb); },
      onCancelClick: function (cb) { return on(cancelBtn, "click", cb); },
      onContinueClick: function (cb) { return on(continueBtn, "click", cb); },
    };
  }

  /** One-time, fail-closed controller/confirmer initialisation. */
  function initializeProfileSwitching() {
    try {
      const dialogAdapter = createProfileSwitchDialogAdapter();
      const historyConfirmer = createRemoteHistoryConfirmer(dialogAdapter);
      switchController = createSessionProfileSwitchController({
        patchSwitch: switchProfileRequest,
        fetchOneSession: fetchOneSessionRaw,
        confirmRemoteHistory: historyConfirmer.confirm,
        validateSessionResponse: isValidSessionResponse,
      });
      profileSwitchInitializationError = null;
      return true;
    } catch (_) {
      switchController = null;
      profileSwitchInitializationError = "模型切换功能暂不可用。";
      return false;
    }
  }

  /** Apply the current draft to the current session. */
  async function applyProfileSwitch() {
    if (switchController === null) {
      if (currentSessionId !== null) {
        setProfileSwitchStatus(
          currentSessionId,
          profileSwitchInitializationError || "模型切换功能暂不可用。",
          true,
        );
      }
      return;
    }
    if (isInitializing || isSending || isCreatingSession || isRenaming ||
        isRenameSaving || isDeletingSession || isProfileSwitching) return;
    if (currentSessionId === null) return;
    if (!applyEnabled()) return;

    const targetSessionId = currentSessionId;
    const requestedDraftId = currentProfileDraftId;
    const generation = ++profileSwitchGeneration;
    isProfileSwitching = true;
    updateControlStates();

    let focusIntent = null;
    try {
      let effectiveSession = findSessionInList(targetSessionId);
      if (effectiveSession === null) {
        setProfileSwitchStatus(targetSessionId,
          "The conversation no longer exists.", true);
        return;
      }

      // -- uncertain convergence: refresh the authoritative session
      //    BEFORE building any operation.
      if (sessionSwitchUncertain[targetSessionId] !== undefined) {
        const result = await fetchAndValidateOneSession(targetSessionId);
        if (result.status === "ok") {
          const cachePlan = planSessionCacheUpdate({
            sessions: sessions,
            requestedSessionId: targetSessionId,
            fresh: result.session,
          });
          if (cachePlan.kind !== "replace") {
            setProfileSwitchStatus(targetSessionId,
              "The conversation could not be refreshed.", true);
            return;
          }
          sessions = cachePlan.sessions;
          renderSessionListOrError();
          effectiveSession = findSessionInList(targetSessionId);
          if (effectiveSession === null) {
            setProfileSwitchStatus(targetSessionId,
              "The conversation no longer exists.", true);
            return;
          }
          const classified = classifyUncertainRefresh({
            targetSessionId: targetSessionId,
            uncertainRecord: sessionSwitchUncertain[targetSessionId],
            fresh: result.session,
          });
          if (classified.status === "invalid") {
            setProfileSwitchStatus(targetSessionId, UNCERTAIN_REAPPLY_TEXT, true);
            return;
          }
          if (classified.status === "confirmed_target") {
            delete sessionSendBlocks[targetSessionId];
            delete sessionSwitchUncertain[targetSessionId];
            clearProfileSwitchStatus(targetSessionId);
            if (effectiveSession.llm_profile_id === requestedDraftId) {
              if (currentSessionId === targetSessionId) {
                currentProfileDraftId = requestedDraftId;
                renderCurrentProfileBar();
                syncCurrentSessionUI();
              }
              return;
            }
            // Draft differs — continue with a fresh switch below.
          }
          // different_binding → keep uncertain, continue with a fresh
          // PATCH for the current draft (same lock on the server
          // serialises the new request after the old one).
        } else if (result.status === "not_found") {
          removeSessionLocally(targetSessionId);
          return;
        } else {
          setProfileSwitchStatus(targetSessionId, UNCERTAIN_REAPPLY_TEXT, true);
          return;
        }
      }

      const requestedProfile = captureRequestedProfile(
        registry(), requestedDraftId,
      );
      if (requestedProfile === null) {
        setProfileSwitchStatus(targetSessionId,
          "Choose an available model first.", true);
        focusIntent = "apply";
        return;
      }

      const operation = {
        generation: generation,
        targetSessionId: targetSessionId,
        originalProfileId: effectiveSession.llm_profile_id,
        originalModelSnapshot: effectiveSession.llm_model_snapshot,
        requestedProfile: requestedProfile,
        needsConfirmHint: needsRemoteHistoryConfirmation({
          session: effectiveSession,
          registry: registry(),
          targetProfileId: requestedDraftId,
          historyState: historyStateFor(targetSessionId),
        }),
      };

      let outcome;
      try {
        outcome = await switchController.apply(operation);
      } catch (_) {
        outcome = { status: "failed", message: "Unexpected error." };
      }

      const plan = planSwitchOutcomeEffects({
        outcome: outcome,
        operation: operation,
        targetSessionId: targetSessionId,
        currentSessionId: currentSessionId,
        hasTarget: findSessionInList(targetSessionId) !== null,
      });

      focusIntent = await executeSwitchEffects(plan, outcome, operation);
    } finally {
      isProfileSwitching = false;
      updateControlStates();
      applyFocusIntent(focusIntent, targetSessionId, generation);
    }
  }

  /** Execute the effect plan.  Target-scoped state is always applied;
   * visible effects require the target to still be the current
   * session after every await. */
  async function executeSwitchEffects(plan, outcome, operation) {
    const targetSessionId = operation.targetSessionId;

    if (plan.kind === "ignore") return null;
    if (plan.removeSession) { removeSessionLocally(targetSessionId); return null; }

    // 1) cache: only a successful replace may unlock protective state
    if (plan.updateCache) {
      const cachePlan = planSessionCacheUpdate({
        sessions: sessions,
        requestedSessionId: targetSessionId,
        fresh: plan.session,
      });
      if (cachePlan.kind !== "replace") {
        setProfileSwitchStatus(targetSessionId,
          "The conversation could not be updated safely.", true);
        return null;
      }
      sessions = cachePlan.sessions;
    }

    // 2) protective state (target-scoped)
    if (plan.clearBlock) delete sessionSendBlocks[targetSessionId];
    if (plan.clearUncertain) delete sessionSwitchUncertain[targetSessionId];
    if (plan.uncertainRecord !== null) {
      sessionSwitchUncertain[targetSessionId] = plan.uncertainRecord;
    }

    // 3) ordinary target status BEFORE any await / visibility check
    applyTargetProfileSwitchStatus(targetSessionId, outcome, plan);

    // 4) profiles reload
    if (plan.reloadProfiles) {
      await reloadProfiles({ allowDefaultFallback: false });
    }

    // 5) re-read visibility after awaits
    renderSessionListOrError();
    if (currentSessionId !== targetSessionId) {
      renderCurrentProfileStatus();
      return null;
    }

    // 6) visible effects — target is still current
    switch (outcome.status) {
      case "switched":
        currentProfileDraftId = plan.session.llm_profile_id;
        break;
      case "not_changed":
        currentProfileDraftId = plan.session.llm_profile_id;
        break;
      case "cancelled":
      case "failed":
      case "validation_error": {
        const current = findSessionInList(targetSessionId);
        currentProfileDraftId = current !== null
          ? resolveSessionProfileDraft(current, registry())
          : null;
        break;
      }
      case "uncertain":
        // keep the user's draft; the record holds the persistent hint
        break;
      default:
        break;
    }

    renderCurrentProfileBar();
    renderCurrentProfileStatus();
    if (plan.syncVisibleUI) syncCurrentSessionUI();
    return plan.focus;
  }

  /** Focus application — triple-checked, never steals focus. */
  function applyFocusIntent(intent, targetSessionId, generation) {
    if (intent === null || intent === undefined) return;
    if (generation !== profileSwitchGeneration) return;
    if (currentSessionId !== targetSessionId) return;
    const el = intent === "input" ? inputEl : applyProfileBtn;
    if (el && el.isConnected && !el.disabled && !el.hidden) el.focus();
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

  // Current-session selector: records the draft only — Apply is the
  // only action that sends a request.
  currentProfileSelectEl.addEventListener("change", function () {
    currentProfileDraftId = currentProfileSelectEl.value;
    updateControlStates();
  });

  applyProfileBtn.addEventListener("click", function () {
    applyProfileSwitch();
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
      // Keep the latent mobile state collapsed while the desktop CSS
      // displays the sidebar normally.  If the viewport later crosses
      // back into mobile, the drawer is already off-screen instead of
      // jumping over the chat area at the breakpoint.
      sidebarEl.classList.add("collapsed");
      sidebarToggleEl.setAttribute("aria-expanded", "true");
    }
  });

  /* ---- Init -------------------------------------------------------- */

  async function init() {
    isInitializing = true;
    updateControlStates();

    // Keep the latent drawer state collapsed on every viewport.  The
    // class only has a visual effect inside the mobile media query, so
    // the desktop sidebar remains visible while future desktop-to-mobile
    // transitions start in the correct closed state.
    sidebarEl.classList.add("collapsed");
    if (isMobile()) {
      sidebarToggleEl.setAttribute("aria-expanded", "false");
    } else {
      sidebarToggleEl.setAttribute("aria-expanded", "true");
    }

    renderSessionListOrError();
    renderWelcome();
    showStatus("Loading...", false);

    // One-time, fail-closed switch controller/confirmer setup — runs
    // synchronously before the first await.  On failure the rest of
    // the app keeps working; only the model switch bar is disabled.
    initializeProfileSwitching();

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
        const first = findSessionInList(currentSessionId);
        currentProfileDraftId = first !== null
          ? resolveSessionProfileDraft(first, registry())
          : null;
        syncCurrentSessionUI();
        renderCurrentProfileBar();
        renderCurrentProfileStatus();
        await loadMessages(currentSessionId);
      } else {
        renderWelcome();
        clearStatus();
      }
    } else {
      renderWelcome();
      if (sessionsLoadError) showStatus(sessionsLoadError, true);
    }

    renderCurrentProfileBar();
    renderCurrentProfileStatus();

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
