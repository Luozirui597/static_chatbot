"use strict";

(function () {
  var HISTORY_USER =
    "history **user** plain\nsecond line";
  var HISTORY_ASSISTANT =
    "# History heading\n\n**History** and *emphasis*\n\n- one\n- two\n\n" +
    "```js\n" +
    "function historyCode() {\n" +
    "  var token = \"" + "x".repeat(160) + "\";\n" +
    "  return token;\n" +
    "}\n" +
    "```";
  var NEW_USER = "new **user** plain";
  var NEW_ASSISTANT = "**New reply**\n\n*new emphasis*";
  var FALLBACK_USER = "fallback user";
  var FALLBACK_ASSISTANT = "**fallback assistant**";

  var report = {
    kind: "app",
    done: false,
    total: 0,
    executed: 0,
    passed: 0,
    failed: 0,
    failures: [],
    appRequests: []
  };
  window.__markdownTestReport = report;

  var nativeFetch = typeof window.fetch === "function"
    ? window.fetch.bind(window)
    : null;
  var clipboardCalls = [];
  var postIndex = 0;
  var nextMessageId = 3;
  var postSpecs = [
    { user: NEW_USER, assistant: NEW_ASSISTANT },
    { user: FALLBACK_USER, assistant: FALLBACK_ASSISTANT }
  ];

  function fail(name, error) {
    report.failures.push({
      name: name,
      message: String(error && error.stack ? error.stack : error)
    });
  }

  async function runTest(name, body) {
    report.total += 1;
    report.executed += 1;
    try {
      await body();
      report.passed += 1;
    } catch (error) {
      report.failed += 1;
      fail(name, error);
    }
  }

  function assert(condition, message) {
    if (condition) return;
    throw new Error(message || "assertion failed");
  }

  function sleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  async function waitFor(predicate, timeoutMs, label) {
    var deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        if (predicate() === true) return;
      } catch (_) {}
      await sleep(50);
    }
    throw new Error("timed out waiting for " + label);
  }

  function message(id, role, content) {
    return {
      id: id,
      session_id: 23,
      role: role,
      content: content,
      created_at: "2026-10-01T12:00:00Z",
      llm_profile_id_snapshot: null,
      llm_profile_kind_snapshot: null,
      llm_model_snapshot: null
    };
  }

  function sessionResponse() {
    return {
      id: 23,
      title: "Synthetic Markdown session",
      created_at: "2026-10-01T12:00:00Z",
      updated_at: "2026-10-01T12:00:00Z",
      llm_profile_id: "fake",
      llm_profile_label: "Fake",
      llm_profile_status: "ready",
      llm_model_snapshot: "fake-model",
      interaction_mode: "receive_teaching"
    };
  }

  function jsonResponse(value, status) {
    var code = typeof status === "number" ? status : 200;
    return Promise.resolve({
      ok: code >= 200 && code < 300,
      status: code,
      json: function () { return Promise.resolve(value); }
    });
  }

  function installFetchMock() {
    var profile = {
      id: "fake",
      label: "Fake",
      kind: "fake",
      model: "fake-model",
      is_default: true
    };

    window.fetch = function (url, options) {
      var method = (options && options.method) || "GET";
      report.appRequests.push({ url: String(url), method: method });

      if (String(url) === "/api/llm/profiles" && method === "GET") {
        return jsonResponse([profile]);
      }
      if (String(url) === "/api/sessions" && method === "GET") {
        return jsonResponse([sessionResponse()]);
      }
      if (String(url) === "/api/sessions/23/messages" && method === "GET") {
        return jsonResponse([
          message(1, "user", HISTORY_USER),
          message(2, "assistant", HISTORY_ASSISTANT)
        ]);
      }
      if (String(url) === "/api/sessions/23/messages" && method === "POST") {
        var spec = postSpecs[postIndex];
        postIndex += 1;
        assert(spec !== undefined, "unexpected extra POST /messages");
        var userId = nextMessageId;
        nextMessageId += 1;
        var assistantId = nextMessageId;
        nextMessageId += 1;
        return jsonResponse({
          user_message: message(userId, "user", spec.user),
          assistant_message: message(assistantId, "assistant", spec.assistant)
        });
      }
      if (String(url) === "/api/sessions/23/mode-switch-events" &&
          method === "GET") {
        return jsonResponse([]);
      }
      if (String(url) === "/api/sessions/23/history-reviews" &&
          method === "GET") {
        return jsonResponse([]);
      }
      return jsonResponse({ detail: "not mocked: " + method + " " + url }, 404);
    };
  }

  function installClipboardMock() {
    var fakeClipboard = {
      writeText: function (value) {
        clipboardCalls.push(value);
        return Promise.resolve();
      }
    };

    try {
      Object.defineProperty(window.navigator, "clipboard", {
        configurable: true,
        get: function () { return fakeClipboard; }
      });
    } catch (_) {
      try {
        Object.defineProperty(Navigator.prototype, "clipboard", {
          configurable: true,
          get: function () { return fakeClipboard; }
        });
      } catch (_) {
        /* The copy test will fail loudly if neither override works. */
      }
    }
  }

  async function loadIndexMarkup() {
    var response = await nativeFetch("/frontend/index.html");
    var html = await response.text();
    var parsed = new DOMParser().parseFromString(html, "text/html");
    var scripts = parsed.querySelectorAll("script");
    for (var i = 0; i < scripts.length; i += 1) {
      scripts[i].remove();
    }
    document.body.innerHTML = parsed.body.innerHTML;
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = src;
      script.async = false;
      script.onload = function () { resolve(); };
      script.onerror = function () {
        reject(new Error("failed to load " + src));
      };
      document.head.appendChild(script);
    });
  }

  async function loadAppScripts() {
    var scripts = [
      "/frontend/network-recovery.js",
      "/frontend/clipboard.js",
      "/frontend/copy-controller.js",
      "/frontend/interaction-mode.js",
      "/frontend/model-selection.js",
      "/frontend/session-profile-switch.js",
      "/frontend/history-review.js",
      "/frontend/vendor/marked.umd.js",
      "/frontend/vendor/purify.min.js",
      "/frontend/markdown-renderer.js",
      "/frontend/app.js"
    ];
    for (var i = 0; i < scripts.length; i += 1) {
      await loadScript(scripts[i]);
    }
  }

  function assistantContents() {
    return Array.prototype.slice.call(
      document.querySelectorAll("#messages .message.assistant .message-content")
    );
  }

  function userContents() {
    return Array.prototype.slice.call(
      document.querySelectorAll("#messages .message.user .message-content")
    );
  }

  function setInputAndSend(value) {
    var input = document.getElementById("messageInput");
    var button = document.getElementById("sendButton");
    input.value = value;
    button.click();
  }

  async function main() {
    assert(nativeFetch !== null, "native fetch is unavailable");

    installClipboardMock();
    await loadIndexMarkup();
    installFetchMock();
    await loadAppScripts();

    await waitFor(
      function () {
        return document.querySelectorAll("#messages .message").length >= 2;
      },
      8000,
      "synthetic history rendering"
    );

    await runTest("history user message remains exact plain text", function () {
      var contents = userContents();
      assert(contents.length >= 1, "history user message missing");
      assert(contents[0].textContent === HISTORY_USER,
        "user message text changed");
      assert(contents[0].querySelector("strong,em,ul,ol") === null,
        "user message received Markdown elements");
      assert(
        contents[0].classList.contains("message-content--markdown") === false,
        "user message received the Markdown class"
      );
    });

    await runTest("history assistant message uses Markdown rendering", function () {
      var contents = assistantContents();
      assert(contents.length >= 1, "history assistant message missing");
      assert(
        contents[0].classList.contains("message-content--markdown") === true,
        "history assistant did not use the Markdown renderer"
      );
      assert(contents[0].querySelector("strong") !== null,
        "history assistant bold missing");
      assert(contents[0].querySelector("em") !== null,
        "history assistant emphasis missing");
      assert(contents[0].querySelector("li") !== null,
        "history assistant list missing");
      assert(
        contents[0].closest(".message").querySelector(".copy-btn") !== null,
        "history assistant copy button missing"
      );
    });

    await runTest("actual copy button copies the original Markdown", async function () {
      var button = assistantContents()[0]
        .closest(".message")
        .querySelector(".copy-btn");
      button.click();
      await waitFor(
        function () { return clipboardCalls.length >= 1; },
        3000,
        "clipboard write"
      );
      assert(clipboardCalls[0] === HISTORY_ASSISTANT,
        "copy button did not copy the raw Markdown");
    });

    await runTest("new assistant reply goes through appendMessage", async function () {
      var before = assistantContents().length;
      setInputAndSend(NEW_USER);
      await waitFor(
        function () { return assistantContents().length > before; },
        8000,
        "new assistant reply"
      );
      await waitFor(
        function () {
          return document.getElementById("sendButton").disabled === false;
        },
        5000,
        "send button re-enabled"
      );

      var contents = assistantContents();
      var last = contents[contents.length - 1];
      assert(
        last.classList.contains("message-content--markdown") === true,
        "new assistant reply did not use Markdown rendering"
      );
      assert(last.querySelector("strong") !== null,
        "new assistant reply bold missing");
      assert(last.querySelector("em") !== null,
        "new assistant reply emphasis missing");
      assert(last.textContent.indexOf("New reply") >= 0,
        "new assistant reply text missing");
    });

    await runTest("new user message remains exact plain text", async function () {
      var contents = userContents();
      var last = contents[contents.length - 1];
      assert(last.textContent === NEW_USER,
        "new user message text changed");
      assert(last.querySelector("strong,em") === null,
        "new user message received Markdown elements");
      assert(
        last.classList.contains("message-content--markdown") === false,
        "new user message received the Markdown class"
      );
    });

    await runTest("history and new assistant replies share the same entry", function () {
      var contents = assistantContents();
      assert(contents.length >= 2, "expected history and new replies");
      for (var i = 0; i < contents.length; i += 1) {
        assert(
          contents[i].classList.contains("message-content--markdown") === true,
          "assistant entry " + i + " did not use Markdown"
        );
        assert(
          contents[i].closest(".message").querySelector(".copy-btn") !== null,
          "assistant entry " + i + " has no copy button"
        );
      }
    });

    await runTest("apply failure falls back to raw text in the real page", async function () {
      window.TeachableMarkdownRenderer = {
        apply: function () {
          throw new Error("synthetic apply failure");
        }
      };

      var before = assistantContents().length;
      setInputAndSend(FALLBACK_USER);

      await waitFor(
        function () { return assistantContents().length > before; },
        8000,
        "fallback assistant reply"
      );
      await waitFor(
        function () {
          return document.getElementById("sendButton").disabled === false;
        },
        5000,
        "send button re-enabled after fallback"
      );

      var assistants = assistantContents();
      var lastAssistant = assistants[assistants.length - 1];
      assert(lastAssistant.textContent === FALLBACK_ASSISTANT,
        "fallback assistant text changed");
      assert(
        lastAssistant.classList.contains("message-content--markdown") === false,
        "fallback assistant kept the Markdown class"
      );

      var users = userContents();
      var lastUser = users[users.length - 1];
      assert(lastUser.textContent === FALLBACK_USER,
        "fallback user text changed");
      assert(
        lastUser.classList.contains("message-content--markdown") === false,
        "fallback user received the Markdown class"
      );
    });

    await runTest("page controls remain usable after a rendering failure", function () {
      var input = document.getElementById("messageInput");
      var button = document.getElementById("sendButton");
      assert(input !== null, "message input disappeared");
      assert(button !== null, "send button disappeared");
      assert(input.disabled === false, "message input is disabled");
      assert(button.disabled === false, "send button is disabled");
    });

    await runTest("document stays in standards mode with the real stylesheet", async function () {
      assert(document.compatMode === "CSS1Compat",
        "document is not in standards mode: " + document.compatMode);
      assert(document.doctype, "document.doctype is missing");
      assert(document.doctype.name === "html",
        "doctype name is not html: " + document.doctype.name);
      assert(document.body.textContent.indexOf("DOCTYPE") === -1,
        "doctype source text leaked into the page");

      var styleLinks = Array.prototype.slice.call(
        document.querySelectorAll('link[rel="stylesheet"]')
      ).filter(function (link) {
        return (link.getAttribute("href") || "").indexOf("style.css") >= 0;
      });
      assert(styleLinks.length >= 1,
        "real style.css link was lost after body replacement");

      await waitFor(function () {
        var sheets = Array.prototype.slice.call(document.styleSheets);
        return sheets.some(function (sheet) {
          return typeof sheet.href === "string" &&
            sheet.href.indexOf("style.css") >= 0 &&
            sheet.cssRules &&
            sheet.cssRules.length > 0;
        });
      }, 5000, "real style.css cssRules");

      var userStyle = getComputedStyle(userContents()[0]);
      assert(userStyle.whiteSpace === "pre-wrap",
        "user plain-text white-space is " + userStyle.whiteSpace);
      assert(userStyle.backgroundColor === "rgb(0, 122, 255)",
        "user bubble CSS was not applied: " + userStyle.backgroundColor);
    });

    await runTest("plain-text and Markdown white-space contract holds in the real app", function () {
      var assistant = assistantContents()[0];
      var user = userContents()[0];

      assert(user.textContent === HISTORY_USER,
        "history user text changed in the real page");
      assert(getComputedStyle(user).whiteSpace === "pre-wrap",
        "user messages must preserve newlines with pre-wrap");
      assert(
        assistant.classList.contains("message-content--markdown") === true,
        "assistant Markdown class missing"
      );
      assert(getComputedStyle(assistant).whiteSpace === "normal",
        "Markdown body must use normal white-space");

      var pre = assistant.querySelector("pre");
      var code = pre === null ? null : pre.querySelector("code");
      assert(pre, "history fenced code block is missing");
      assert(code, "history fenced code element is missing");
      assert(code.textContent.indexOf("  var token") >= 0,
        "code indentation was lost");
      assert(code.textContent.indexOf("\n") >= 0,
        "code newline was lost");
      assert(getComputedStyle(pre).whiteSpace === "pre",
        "pre white-space is " + getComputedStyle(pre).whiteSpace);
      assert(getComputedStyle(code).whiteSpace === "pre",
        "code white-space is " + getComputedStyle(code).whiteSpace);
      assert(getComputedStyle(pre).overflowX === "auto",
        "pre does not own horizontal scrolling: " +
        getComputedStyle(pre).overflowX);
      assert(pre.scrollWidth > pre.clientWidth,
        "synthetic long code did not actually overflow the pre");
    });

    await runTest("real app messages stay inside the chat viewport", function () {
      var messages = document.getElementById("messages");
      assert(messages, "messages container is missing");

      var chatRect = messages.getBoundingClientRect();
      var page = document.documentElement;
      assert(page.scrollWidth <= page.clientWidth + 1,
        "page has unexpected horizontal overflow");
      assert(messages.scrollWidth <= messages.clientWidth + 1,
        "chat messages area has unexpected horizontal overflow");

      var contents = Array.prototype.slice.call(
        messages.querySelectorAll(".message-content")
      );
      assert(contents.length >= 2, "not enough synthetic messages");
      for (var i = 0; i < contents.length; i += 1) {
        var contentRect = contents[i].getBoundingClientRect();
        assert(contentRect.left >= chatRect.left - 1 &&
          contentRect.right <= chatRect.right + 1,
          "message content " + i + " escaped the chat viewport");
      }

      var copy = assistantContents()[0]
        .closest(".message")
        .querySelector(".copy-btn");
      assert(copy, "assistant copy button is missing");
      copy.scrollIntoView({ block: "nearest", inline: "nearest" });
      var copyRect = copy.getBoundingClientRect();
      assert(copyRect.width > 0 && copyRect.height > 0,
        "copy button has no hit target");
      assert(copyRect.top >= chatRect.top - 1 &&
        copyRect.bottom <= chatRect.bottom + 1,
        "copy button is outside the visible chat area");
      var hit = document.elementFromPoint(
        copyRect.left + copyRect.width / 2,
        copyRect.top + copyRect.height / 2
      );
      assert(hit === copy || copy.contains(hit),
        "copy button is covered or not actionable");
    });

    report.done = true;
  }

  main().catch(function (error) {
    fail("app integration suite", error);
    report.done = true;
  });
})();
