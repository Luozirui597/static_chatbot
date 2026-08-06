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
  let sessionLoadRequestId = 0;  // race-condition guard for message loads

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

  /* ---- Control state helpers --------------------------------------- */

  /** Sync button disabled states from isSending + isCreatingSession. */
  function updateControlStates() {
    const blockSend = isSending;
    const blockCreate = isSending || isCreatingSession;

    sendBtn.disabled = blockSend;
    inputEl.disabled = blockSend;
    newChatBtn.disabled = blockCreate;

    const deleteBtns = document.querySelectorAll(".delete-session-btn");
    deleteBtns.forEach(function (btn) { btn.disabled = blockSend; });

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

    const wrapper = document.createElement("div");
    wrapper.className = "message " + role;

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = content;

    wrapper.appendChild(contentDiv);
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
        sessionHeaderEl.textContent = "New Chat · #" + sessions[i].id;
        return;
      }
    }
    sessionHeaderEl.textContent = "New Chat · #" + currentSessionId;
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

      const item = document.createElement("div");
      item.className = "session-item";
      if (sid === currentSessionId) {
        item.classList.add("active");
      }

      // Label
      const textDiv = document.createElement("div");
      textDiv.className = "session-item-text";

      const label = document.createElement("div");
      label.className = "session-item-label";
      label.textContent = "New Chat · #" + sid;

      const time = document.createElement("div");
      time.className = "session-time";
      time.textContent = formatSessionTime(session.updated_at);

      textDiv.appendChild(label);
      textDiv.appendChild(time);

      // Delete button
      const delBtn = document.createElement("button");
      delBtn.className = "delete-session-btn";
      delBtn.type = "button";
      delBtn.setAttribute("aria-label", "Delete session " + sid);
      delBtn.textContent = "×";
      if (isSending) {
        delBtn.disabled = true;
      }
      delBtn.addEventListener("click", function (event) {
        handleDeleteSession(sid, event);
      });

      // Click on item → select
      item.addEventListener("click", function () {
        selectSession(sid);
      });

      item.appendChild(textDiv);
      item.appendChild(delBtn);
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
    if (isCreatingSession || isSending) return;

    isCreatingSession = true;
    updateControlStates();

    try {
      showStatus("Creating session...", false);
      const session = await createSessionRequest();
      clearStatus();

      sessions.unshift(session);
      currentSessionId = session.id;
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

    if (isCreatingSession || isSending) return false;

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

  /* ---- Send message ------------------------------------------------ */

  async function sendMessage(text) {
    if (isSending) return;

    // Auto-create session on first send
    if (currentSessionId === null) {
      const ok = await ensureSession();
      if (!ok) return;  // create failed or already in progress
    }

    const sendingSessionId = currentSessionId;

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

      // Refresh session list (sending session moves to top)
      await refreshSessions({ preserveSelection: true });
    } catch (err) {
      handleSendFailure(err, sendingSessionId, text);
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
   * Network (0) → uncertain.  Re-fetch and compare.
   */
  async function handleSendFailure(err, sendingSessionId, originalText) {
    const errStatus = (err instanceof ApiError) ? err.status : 0;
    let isSaved = false;

    // --- Determine save status ----------------------------------------
    if (errStatus === 502 || errStatus === 504) {
      isSaved = true;
    } else if (errStatus === 0) {
      // Network error — re-fetch and check
      try {
        const msgs = await fetchMessages(sendingSessionId);
        // Find last user message matching what we sent
        let userIdx = -1;
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].role === "user" && msgs[i].content === originalText) {
            userIdx = i;
            break;
          }
        }
        if (userIdx >= 0) {
          // Check if an assistant reply exists after this user message
          const hasAssistant = msgs.some(function (m, idx) {
            return idx > userIdx && m.role === "assistant";
          });
          if (hasAssistant) {
            // Request actually succeeded — full response was saved.
            // Render silently on the sending session.
            if (currentSessionId === sendingSessionId) {
              renderMessages(msgs);
              scrollMessagesToBottom();
              clearStatus();
            }
            inputEl.value = "";
            inputEl.style.height = "";
            await refreshSessions({ preserveSelection: true });
            return;
          }
          // User message saved, no assistant
          isSaved = true;
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
          await refreshSessions({ preserveSelection: true });
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
    if (isSending) return;
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

  // Auto-resize textarea
  inputEl.addEventListener("input", function () {
    inputEl.style.height = "";
    inputEl.style.height = inputEl.scrollHeight + "px";
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

  /* ---- Init -------------------------------------------------------- */

  async function init() {
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
  }

  init();
})();
