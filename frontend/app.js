/* ------------------------------------------------------------------ */
/* Static Chatbot — multi-session frontend logic                      */
/* ------------------------------------------------------------------ */

(function () {
  "use strict";

  /* ---- DOM references ---------------------------------------------- */

  const sidebarEl = document.getElementById("sidebar");
  const sidebarToggleEl = document.getElementById("sidebarToggle");
  const sessionListEl = document.getElementById("sessionList");
  const newChatBtn = document.getElementById("newChatButton");
  const sessionHeaderEl = document.getElementById("sessionHeader");
  const messagesEl = document.getElementById("messages");
  const statusEl = document.getElementById("status");
  const inputEl = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendButton");

  /* ---- State ------------------------------------------------------- */

  let sessions = [];            // [{id, title, created_at, updated_at}, ...]
  let currentSessionId = null;  // int | null
  let isSending = false;        // prevent double-submit
  let isCreatingSession = false; // prevent double-create of New Chat
  let isRenaming = false;       // editing mode active
  let renamingSessionId = null; // which session is being edited
  let isRenameSaving = false;   // saving in progress (prevent double-submit)
  let isInitializing = true;    // block user actions during page init
  let sessionLoadRequestId = 0;  // race-condition guard for message loads
  let sessionLastMessageId = {}; // sessionId -> int (last known message id)

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

  async function createSessionRequest() {
    const resp = await apiRequest("/api/sessions", { method: "POST" });
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

  /* ---- Control state helpers --------------------------------------- */

  /** Sync button disabled states from isSending + isCreatingSession +
   *  isRenaming + isRenameSaving + isInitializing. */
  function updateControlStates() {
    const blockAll = isInitializing || isRenaming || isRenameSaving;
    const blockSend = blockAll || isSending;
    const blockCreate = blockAll || isSending || isCreatingSession;

    sendBtn.disabled = blockSend;
    inputEl.disabled = blockSend;
    newChatBtn.disabled = blockCreate;

    const deleteBtns = document.querySelectorAll(".delete-session-btn");
    deleteBtns.forEach(function (btn) { btn.disabled = blockAll; });

    const renameBtns = document.querySelectorAll(".rename-session-btn");
    renameBtns.forEach(function (btn) { btn.disabled = blockAll; });

    const selectBtns = document.querySelectorAll(".session-select-btn");
    selectBtns.forEach(function (btn) { btn.disabled = blockAll; });

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

    if (!blockSend) {
      inputEl.focus();
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

  /* ---- Session header ---------------------------------------------- */

  function updateSessionHeader() {
    if (currentSessionId === null) {
      sessionHeaderEl.textContent = "";
      return;
    }
    for (let i = 0; i < sessions.length; i++) {
      if (sessions[i].id === currentSessionId) {
        sessionHeaderEl.textContent = sessions[i].title;
        return;
      }
    }
    sessionHeaderEl.textContent = "New Chat";
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
        if (isSending || isRenaming || isInitializing) {
          renameBtn.disabled = true;
        }
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
        if (isSending || isRenaming || isInitializing) {
          delBtn.disabled = true;
        }
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

  async function refreshSessions(preserveSelection) {
    if (preserveSelection === undefined) preserveSelection = true;

    try {
      const data = await fetchSessions();
      sessions = data;
    } catch (err) {
      showStatus(err.message, true);
      return;
    }

    renderSessionList();
    updateSessionHeader();

    if (sessions.length === 0) {
      currentSessionId = null;
      sessionLoadRequestId++;
      renderWelcome();
      updateSessionHeader();
      return;
    }

    // If preserving selection and the current session still exists
    if (preserveSelection && currentSessionId !== null) {
      let found = false;
      for (let i = 0; i < sessions.length; i++) {
        if (sessions[i].id === currentSessionId) {
          found = true;
          break;
        }
      }
      if (!found) {
        // Current session was deleted by someone else
        currentSessionId = sessions[0].id;
        sessionLoadRequestId++;
        updateSessionHeader();
        loadMessages(currentSessionId);
      }
    }

    // If no selection, auto-select first
    if (currentSessionId === null && sessions.length > 0) {
      currentSessionId = sessions[0].id;
      updateSessionHeader();
      loadMessages(currentSessionId);
    }
  }

  /**
   * Remove a session from the local list and handle UI transitions
   * if it was the currently-selected session.
   */
  function removeSessionLocally(sessionId) {
    sessions = sessions.filter(function (s) { return s.id !== sessionId; });
    delete sessionLastMessageId[sessionId];

    if (sessionId === currentSessionId) {
      sessionLoadRequestId++;
      clearMessages();
      clearStatus();

      if (sessions.length > 0) {
        currentSessionId = sessions[0].id;
        updateSessionHeader();
        loadMessages(currentSessionId);
      } else {
        currentSessionId = null;
        updateSessionHeader();
        renderWelcome();
      }
    }

    renderSessionList();
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
   */
  function selectSession(sessionId) {
    if (sessionId === currentSessionId) return;

    currentSessionId = sessionId;
    renderSessionList();
    updateSessionHeader();
    loadMessages(sessionId);
    closeSidebarOnMobile();
  }

  /** Create a new session (called by "New Chat" button). */
  async function newChat() {
    if (isCreatingSession || isSending || isInitializing) return;

    isCreatingSession = true;
    updateControlStates();

    try {
      showStatus("Creating session...", false);
      const session = await createSessionRequest();
      clearStatus();

      sessions.unshift(session);
      currentSessionId = session.id;
      sessionLastMessageId[session.id] = 0;
      sessionLoadRequestId++;
      renderSessionList();
      updateSessionHeader();
      renderEmptyChat();
      clearStatus();
      inputEl.value = "";
      inputEl.style.height = "";
      inputEl.focus();
      closeSidebarOnMobile();
    } catch (err) {
      showStatus(err.message, true);
    } finally {
      isCreatingSession = false;
      updateControlStates();
    }
  }

  /** Ensure a session exists, creating one if necessary. */
  async function ensureSession() {
    if (currentSessionId !== null) return true;

    if (isCreatingSession || isSending || isInitializing) return false;

    isCreatingSession = true;
    updateControlStates();

    try {
      showStatus("Creating session...", false);
      const session = await createSessionRequest();
      sessions.unshift(session);
      currentSessionId = session.id;
      renderSessionList();
      updateSessionHeader();
      clearStatus();
      return true;
    } catch (err) {
      showStatus(err.message, true);
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
    renderSessionList();
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
      renderSessionList();
      updateSessionHeader();
      inputEl.focus();  // explicitly focus chat message input
    } catch (err) {
      // sessions was never modified — no rollback needed
      isRenameSaving = false;
      updateControlStates();  // re-enable input, Save, Cancel
      showStatus(err.message, true);
      // Stay in Editing state — do NOT call renderSessionList()
      renameInputEl.focus();
    }
  }

  function cancelRename() {
    // No optimistic update was made, no restore needed
    isRenaming = false;
    renamingSessionId = null;
    isRenameSaving = false;
    updateControlStates();
    renderSessionList();
    inputEl.focus();  // explicitly focus chat message input
  }

  /* ---- Send message ------------------------------------------------ */

  async function sendMessage(text) {
    if (isSending || isInitializing) return;

    // Auto-create session on first send
    if (currentSessionId === null) {
      const ok = await ensureSession();
      if (!ok) return;  // create failed or already in progress
    }

    const sendingSessionId = currentSessionId;
    const lastMessageIdBeforeSend = sessionLastMessageId[sendingSessionId] || 0;

    setSendingState(true);
    showStatus("Generating...", false);

    try {
      const data = await sendSessionMessageRequest(sendingSessionId, text);
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

      // Refresh session list (sending session moves to top)
      await refreshSessions(true);
    } catch (err) {
      await handleSendFailure(err, sendingSessionId, text, lastMessageIdBeforeSend);
      return;
    } finally {
      setSendingState(false);
    }
  }

  /**
   * Handle a failed send — strategy depends on the error type.
   *
   * 502 / 504  → user message was saved (Phase 1 committed).
   *               Clear input, re-sync messages from DB.
   * 404 / 422  → user message was NOT saved.
   *               Keep input, show error.
   * Network (0) → uncertain.  Re-fetch and use message-ID boundary
   *               to determine whether the request reached the server.
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
    let isSaved = false;

    // --- Determine save status ----------------------------------------
    if (errStatus === 502 || errStatus === 504) {
      isSaved = true;
    } else if (errStatus === 0) {
      // Network error — re-fetch and check by ID boundary
      try {
        const msgs = await fetchMessages(sendingSessionId);

        var result = findSentMessages(msgs, lastMessageIdBeforeSend, originalText);

        if (result.userIdx >= 0) {
          if (result.hasAssistant) {
            // Request actually succeeded — full response was saved.
            // Render silently on the sending session.
            if (currentSessionId === sendingSessionId) {
              renderMessages(msgs);
              scrollMessagesToBottom();
              clearStatus();
            }
            inputEl.value = "";
            inputEl.style.height = "";
            sessionLastMessageId[sendingSessionId] = msgs[msgs.length - 1].id;
            await refreshSessions(true);
            return;
          }
          // User message saved, no assistant
          isSaved = true;
          sessionLastMessageId[sendingSessionId] = result.userMessageId;
        }
        // userIdx < 0 → isSaved stays false (uncertain)
      } catch (_) {
        /* stay with isSaved = false */
      }
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

        // Reload history to show the saved user message
        try {
          const history = await fetchMessages(sendingSessionId);
          if (currentSessionId === sendingSessionId) {
            if (history.length === 0) {
              renderEmptyChat();
            } else {
              renderMessages(history);
            }
          }
        } catch (_) {
          /* best-effort sync */
        }
      } else if (errStatus === 0) {
        // Network error, uncertain
        showStatus(
          "Request status is uncertain. Review the conversation before resending.",
          true,
        );
        // Keep input — user can decide
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
    if (isSending || isInitializing) return;
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

    try {
      // Mobile: start with sidebar collapsed
      if (isMobile()) {
        sidebarEl.classList.add("collapsed");
        sidebarToggleEl.setAttribute("aria-expanded", "false");
      } else {
        sidebarToggleEl.setAttribute("aria-expanded", "true");
      }

      renderSessionList();
      renderWelcome();
      showStatus("Loading...", false);

      try {
        sessions = await fetchSessions();
      } catch (err) {
        showStatus(err.message, true);
        renderSessionList();
        return;
      }

      renderSessionList();

      if (sessions.length > 0) {
        currentSessionId = sessions[0].id;
        updateSessionHeader();
        await loadMessages(currentSessionId);
      } else {
        renderWelcome();
        clearStatus();
      }

      inputEl.focus();
    } finally {
      isInitializing = false;
      updateControlStates();
    }
  }

  init();
})();
