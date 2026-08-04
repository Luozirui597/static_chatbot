/* ------------------------------------------------------------------ */
/* Static Chatbot — frontend logic                                    */
/* ------------------------------------------------------------------ */

(function () {
  "use strict";

  /* ---- DOM references -------------------------------------------- */
  const messagesEl = document.getElementById("messages");
  const statusEl = document.getElementById("status");
  const inputEl = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendButton");

  /* ---- State ----------------------------------------------------- */
  let loading = false;

  /* ---- Status helpers -------------------------------------------- */

  /** Disable or enable the send button and input. */
  function setLoading(on) {
    loading = on;
    sendBtn.disabled = on;
    inputEl.disabled = on;
    if (!on) {
      inputEl.focus();
    }
  }

  /** Show the "generating…" indicator. */
  function showLoading() {
    statusEl.textContent = "Generating...";
    statusEl.className = "chat-status";
  }

  /** Show an error message in the status bar (red). */
  function showError(text) {
    statusEl.textContent = text;
    statusEl.className = "chat-status error";
  }

  /** Clear the status bar. */
  function clearStatus() {
    statusEl.textContent = "";
    statusEl.className = "chat-status";
  }

  /* ---- Messages -------------------------------------------------- */

  /** Safely add a message bubble.  Uses textContent — never innerHTML. */
  function addMessage(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = "message " + role;

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;

    wrapper.appendChild(content);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  /* ---- API call -------------------------------------------------- */

  async function sendMessage(text) {
    setLoading(true);
    clearStatus();
    showLoading();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        let detail = "Something went wrong. Please try again.";
        try {
          const body = await response.json();
          if (body.detail && Array.isArray(body.detail) && body.detail.length > 0) {
            detail = body.detail[0].msg || detail;
          } else if (typeof body.detail === "string") {
            detail = body.detail;
          }
        } catch (_) {
          // ignore parse errors, keep default message
        }
        showError(detail);
        return;
      }

      const data = await response.json();
      addMessage("bot", data.reply);
      clearStatus();
    } catch (_err) {
      // Network error (e.g. server unreachable)
      showError("Network error. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  /* ---- Event handlers -------------------------------------------- */

  sendBtn.addEventListener("click", function () {
    if (loading) return;
    const text = inputEl.value.trim();
    if (!text) return;

    addMessage("user", text);
    inputEl.value = "";
    inputEl.style.height = "";
    sendMessage(text);
  });

  inputEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendBtn.click();
    }
    // Shift+Enter inserts a newline (default browser behaviour).
  });

  /* Auto-resize the textarea as the user types. */
  inputEl.addEventListener("input", function () {
    inputEl.style.height = "";
    inputEl.style.height = inputEl.scrollHeight + "px";
  });
})();
