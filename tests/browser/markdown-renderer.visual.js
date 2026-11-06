"use strict";

(function () {
  var messages = document.getElementById("visual-messages");
  var report = {
    kind: "visual",
    done: false,
    total: 0,
    executed: 0,
    passed: 0,
    failed: 0,
    failures: [],
    executedNames: [],
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight
    }
  };
  window.__markdownVisualReport = report;

  function fail(name, error) {
    report.failures.push({
      name: name,
      message: String(error && error.stack ? error.stack : error)
    });
  }

  function assert(condition, message) {
    if (condition) return;
    throw new Error(message || "assertion failed");
  }

  function find(selector) {
    var el = document.querySelector(selector);
    assert(el, "missing element " + selector);
    return el;
  }

  function test(name, body) {
    report.total += 1;
    report.executed += 1;
    report.executedNames.push(name);
    try {
      body();
      report.passed += 1;
    } catch (error) {
      report.failed += 1;
      fail(name, error);
    }
  }

  function createMessage(role, id) {
    var wrapper = document.createElement("div");
    wrapper.className = "message " + role;
    if (id) wrapper.id = id;

    var content = document.createElement("div");
    content.className = "message-content";
    wrapper.appendChild(content);

    var result = {
      wrapper: wrapper,
      content: content,
      copyButton: null,
      liveRegion: null
    };

    if (role === "assistant") {
      var actions = document.createElement("div");
      actions.className = "message-actions";

      var copyButton = document.createElement("button");
      copyButton.type = "button";
      copyButton.className = "copy-btn";
      copyButton.setAttribute("aria-label", "Copy response");
      copyButton.textContent = "Copy";
      copyButton.dataset.state = "idle";
      actions.appendChild(copyButton);

      var liveRegion = document.createElement("span");
      liveRegion.className = "sr-only";
      liveRegion.setAttribute("aria-live", "polite");
      liveRegion.textContent = "";

      wrapper.appendChild(actions);
      wrapper.appendChild(liveRegion);

      result.copyButton = copyButton;
      result.liveRegion = liveRegion;
    }

    messages.appendChild(wrapper);
    return result;
  }

  function addUserMessage(id, text) {
    var message = createMessage("user", id);
    message.content.textContent = text;
    return message;
  }

  function addAssistantMessage(id, markdown) {
    var message = createMessage("assistant", id);
    var ok = window.TeachableMarkdownRenderer.apply(
      message.content, markdown
    );
    if (ok === false) {
      message.content.classList.remove("message-content--markdown");
      message.content.textContent = markdown;
    }
    return message;
  }

  var userLongToken = "U".repeat(90);
  var userSource =
    "纯文本用户消息第一行\n" +
    "第二行包含超长无空格片段：" + userLongToken + "\n" +
    "第三行保留换行。";

  var codeLine =
    "  var veryLongToken = \"" + "A".repeat(180) + "\";";
  var codeSource =
    "function visualCode() {\n" +
    codeLine + "\n" +
    "  return veryLongToken;\n" +
    "}";
  var codeMarkdown = "## 长代码块\n\n```js\n" + codeSource + "\n```";

  var tableCells = [];
  for (var tableIndex = 0; tableIndex < 8; tableIndex += 1) {
    tableCells.push("列" + (tableIndex + 1) + " " + "T".repeat(42));
  }
  var tableMarkdown =
    "## 宽表格\n\n" +
    "| " + tableCells.join(" | ") + " |\n" +
    "| " + tableCells.map(function () { return "---"; }).join(" | ") + " |\n" +
    "| " + tableCells.join(" | ") + " |\n";

  var linkToken = "L".repeat(150);
  var linkMarkdown =
    "## 长链接\n\n" +
    "[" + linkToken + "](https://example.com/" +
    "long/path/segment?token=" + "q".repeat(120) + ")";

  // .message is fit-content up to 80% of the chat width. Keep this display
  // row well beyond that cap so 1280px and 375px viewports both exercise
  // the math container's own horizontal scroll instead of widening the
  // message to fit the whole formula.
  var longMathTerms = [
    "\\frac{a_i+b_i+c_i+d_i+e_i}{c_i+d_i+e_i+f_i}",
    "\\frac{a_i+b_i+c_i+d_i+e_i+f_i}{c_i+d_i+e_i}",
    "\\frac{a_i+b_i+c_i}{c_i+d_i+e_i+f_i+g_i}",
    "\\frac{a_i+b_i+c_i+d_i}{c_i+d_i+e_i+f_i+g_i}"
  ];
  var mathTail = [];
  for (var mathTermIndex = 0; mathTermIndex < 20; mathTermIndex += 1) {
    mathTail.push(longMathTerms[mathTermIndex % longMathTerms.length]);
  }
  var mathMarkdown =
    "## 数学公式\n\n" +
    "行内公式 \\(y_i\\)、\\(\\frac{a}{b}\\) 与 \\(\\sqrt{x}\\)。\n\n" +
    "\\[\\sum_{i=1}^{n} " +
    "\\frac{a_i+b_i+c_i+d_i+e_i}{\\sqrt{c_i+d_i+e_i}} + " +
    mathTail.join(" + ") + "\\]";

  window.__visualFixtures = {
    userSource: userSource,
    userLongToken: userLongToken,
    codeSource: codeSource,
    codeLine: codeLine,
    codeLineLength: codeLine.length,
    tableCellCount: tableCells.length,
    tableCellTokenLength: 42,
    linkToken: linkToken,
    mathMarkdown: mathMarkdown,
    minimumScrollOverflow: 100
  };

  addUserMessage("visual-user-plain", userSource);

  var copyMessage = addAssistantMessage(
    "visual-copy-message",
    "## 可操作复制按钮\n\n短消息，用于验证按钮可见性和点击操作。"
  );
  copyMessage.copyButton.addEventListener("click", function () {
    copyMessage.copyButton.dataset.state = "copied";
    copyMessage.copyButton.textContent = "Copied";
    copyMessage.liveRegion.textContent = "Visual copy check completed.";
  });

  addAssistantMessage("visual-code-message", codeMarkdown);

  var tableMessage = addAssistantMessage(
    "visual-table-message", tableMarkdown
  );
  var tableElement = tableMessage.content.querySelector("table");
  if (tableElement) {
    var tableCellsAll = tableElement.querySelectorAll("th, td");
    for (var cellIndex = 0; cellIndex < tableCellsAll.length; cellIndex += 1) {
      tableCellsAll[cellIndex].style.whiteSpace = "nowrap";
    }
  }

  addAssistantMessage("visual-link-message", linkMarkdown);
  addAssistantMessage("visual-math-message", mathMarkdown);

  for (var fillerIndex = 1; fillerIndex <= 12; fillerIndex += 1) {
    addAssistantMessage(
      "visual-filler-" + fillerIndex,
      "### 滚动填充 " + fillerIndex + "\n\n" +
      "填充段落用于触发聊天区域纵向滚动。".repeat(24)
    );
  }
  addAssistantMessage("visual-last", "最后一条消息。");

  function resetReport() {
    report.done = false;
    report.total = 0;
    report.executed = 0;
    report.passed = 0;
    report.failed = 0;
    report.failures = [];
    report.executedNames = [];
    report.viewport = {
      width: window.innerWidth,
      height: window.innerHeight
    };
  }

  window.__runMarkdownVisualChecks = function () {
    resetReport();

    test("standards mode and real style.css are active", function () {
      assert(document.compatMode === "CSS1Compat",
        "document is not in standards mode: " + document.compatMode);
      assert(document.doctype, "document.doctype is missing");
      assert(document.doctype.name === "html",
        "doctype name is not html: " + document.doctype.name);
      assert(document.body.textContent.indexOf("DOCTYPE") === -1,
        "doctype source text leaked into the page");

      var sheet = Array.prototype.slice.call(document.styleSheets)
        .filter(function (candidate) {
          return typeof candidate.href === "string" &&
            candidate.href.indexOf("/frontend/style.css") >= 0;
        })[0];
      assert(sheet, "real style.css stylesheet is not loaded");
      assert(sheet.cssRules && sheet.cssRules.length > 0,
        "style.css has no readable rules");

      var userContent = find("#visual-user-plain .message-content");
      assert(getComputedStyle(userContent).whiteSpace === "pre-wrap",
        "key style.css rule not applied: " +
        getComputedStyle(userContent).whiteSpace);
    });

    test("plain user fallback keeps newlines and pre-wrap", function () {
      var fixture = window.__visualFixtures;
      var userContent = find("#visual-user-plain .message-content");
      assert(userContent.textContent === fixture.userSource,
        "plain text content changed");
      assert(userContent.textContent.indexOf("\n") >= 0,
        "plain text newlines were lost");
      assert(getComputedStyle(userContent).whiteSpace === "pre-wrap",
        "plain user white-space is " +
        getComputedStyle(userContent).whiteSpace);
      assert(userContent.scrollWidth <= userContent.clientWidth + 1,
        "plain user fallback overflows its bubble");
    });

    test("Markdown body uses normal white-space and block elements", function () {
      var copyContent = find("#visual-copy-message .message-content");
      assert(
        copyContent.classList.contains("message-content--markdown") === true,
        "assistant markdown class missing"
      );
      assert(getComputedStyle(copyContent).whiteSpace === "normal",
        "markdown body white-space is " +
        getComputedStyle(copyContent).whiteSpace);
      assert(copyContent.querySelector("h2"), "markdown heading missing");
    });

    test("long code keeps white-space and scrolls inside pre", function () {
      var fixture = window.__visualFixtures;
      var pre = find("#visual-code-message pre");
      var code = find("#visual-code-message pre code");
      assert(getComputedStyle(pre).whiteSpace === "pre",
        "pre white-space is " + getComputedStyle(pre).whiteSpace);
      assert(getComputedStyle(code).whiteSpace === "pre",
        "code white-space is " + getComputedStyle(code).whiteSpace);
      assert(getComputedStyle(pre).overflowX === "auto",
        "pre does not own horizontal scrolling: " +
        getComputedStyle(pre).overflowX);
      assert(code.textContent.indexOf(fixture.codeLine) >= 0,
        "code indentation or content was not preserved");
      assert(pre.scrollWidth > pre.clientWidth,
        "stress code did not overflow the pre container");
      pre.scrollLeft = 24;
      assert(pre.scrollLeft > 0, "pre did not scroll horizontally");
      pre.scrollLeft = 0;
    });

    test("wide table scrolls inside its own container", function () {
      var table = find("#visual-table-message table");
      assert(getComputedStyle(table).overflowX === "auto",
        "table does not own horizontal scrolling: " +
        getComputedStyle(table).overflowX);
      assert(table.scrollWidth > table.clientWidth,
        "stress table did not overflow its container");
      table.scrollLeft = 24;
      assert(table.scrollLeft > 0, "table did not scroll horizontally");
      table.scrollLeft = 0;
    });

    test("long link wraps without escaping the chat viewport", function () {
      var link = find("#visual-link-message a");
      var chat = find("#visual-messages");
      var chatRect = chat.getBoundingClientRect();
      var linkRect = link.getBoundingClientRect();
      assert(getComputedStyle(link).overflowWrap === "anywhere",
        "long link overflow-wrap is " +
        getComputedStyle(link).overflowWrap);
      assert(linkRect.left >= chatRect.left - 1 &&
        linkRect.right <= chatRect.right + 1,
        "long link escaped the chat viewport");
    });

    test("copy button is visible and actionable", function () {
      var chat = find("#visual-messages");
      var button = find("#visual-copy-message .copy-btn");
      button.scrollIntoView({ block: "nearest", inline: "nearest" });
      var chatRect = chat.getBoundingClientRect();
      var buttonRect = button.getBoundingClientRect();
      assert(buttonRect.width > 0 && buttonRect.height > 0,
        "copy button has no hit target");
      assert(buttonRect.left >= chatRect.left - 1 &&
        buttonRect.right <= chatRect.right + 1 &&
        buttonRect.top >= chatRect.top - 1 &&
        buttonRect.bottom <= chatRect.bottom + 1,
        "copy button is outside the visible chat area");

      var hit = document.elementFromPoint(
        buttonRect.left + buttonRect.width / 2,
        buttonRect.top + buttonRect.height / 2
      );
      assert(hit === button || button.contains(hit),
        "copy button is covered or not actionable");

      button.click();
      assert(button.dataset.state === "copied",
        "copy button click was not handled");
      button.dataset.state = "idle";
      button.textContent = "Copy";
    });

    test("messages stay within the chat viewport without page overflow", function () {
      var chat = find("#visual-messages");
      var chatRect = chat.getBoundingClientRect();
      var page = document.documentElement;
      assert(page.scrollWidth <= page.clientWidth + 1,
        "page has unexpected horizontal overflow");
      assert(chat.scrollWidth <= chat.clientWidth + 1,
        "chat area has unexpected horizontal overflow");

      var contents = document.querySelectorAll(
        "#visual-messages .message-content"
      );
      assert(contents.length >= 10, "not enough synthetic messages");
      for (var i = 0; i < contents.length; i += 1) {
        var rect = contents[i].getBoundingClientRect();
        assert(rect.left >= chatRect.left - 1 &&
          rect.right <= chatRect.right + 1,
          "message content " + i + " escaped the chat viewport");
      }
    });

    test("chat content taller than viewport can scroll to the last message", function () {
      var chat = find("#visual-messages");
      assert(getComputedStyle(chat).overflowY === "auto",
        "chat area does not own vertical scrolling: " +
        getComputedStyle(chat).overflowY);
      assert(chat.scrollHeight > chat.clientHeight,
        "synthetic transcript did not overflow the chat height");

      chat.scrollTop = chat.scrollHeight;
      assert(chat.scrollTop > 0, "chat area did not scroll vertically");

      var chatRect = chat.getBoundingClientRect();
      var lastRect = find("#visual-last").getBoundingClientRect();
      assert(lastRect.bottom <= chatRect.bottom + 1,
        "last message cannot be reached by scrolling");
      chat.scrollTop = 0;
    });

    test("long math display scrolls inside its own container", function () {
      var display = find("#visual-math-message .math-display");
      assert(display, "math display container missing");
      assert(getComputedStyle(display).overflowX === "auto",
        "math display does not own horizontal scrolling");
      assert(display.scrollWidth > display.clientWidth,
        "long math did not overflow its container");
      display.scrollLeft = 24;
      assert(display.scrollLeft > 0, "math display did not scroll");
      display.scrollLeft = 0;

      var math = display.querySelector("math");
      assert(math, "math element missing inside display");
      assert(math.getBoundingClientRect().width > 0,
        "math element has no visible width");
      var page = document.documentElement;
      var chat = find("#visual-messages");
      assert(page.scrollWidth <= page.clientWidth + 1,
        "math caused page horizontal overflow");
      assert(chat.scrollWidth <= chat.clientWidth + 1,
        "math caused chat horizontal overflow");
    });

    test("synthetic stress data is long enough to exercise layout", function () {
      var fixture = window.__visualFixtures;
      assert(fixture.userLongToken.length >= 60,
        "user token is too short to trigger wrapping");
      assert(fixture.codeLineLength >= 150,
        "code line is too short to trigger scrolling");
      assert(fixture.tableCellCount >= 8,
        "table is too narrow to trigger scrolling");
      assert(fixture.tableCellTokenLength >= 40,
        "table cells are too short to trigger scrolling");
      assert(fixture.linkToken.length >= 120,
        "link text is too short to trigger wrapping");
      assert(messages.scrollHeight >
        messages.clientHeight + fixture.minimumScrollOverflow,
        "synthetic transcript is not tall enough");
    });

    report.done = true;
    return report;
  };

  window.__visualReady = true;
})();
