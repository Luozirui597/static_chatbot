"use strict";

(function () {
  var testRoot = document.getElementById("markdown-test-root");
  var report = {
    kind: "integration",
    done: false,
    total: 0,
    executed: 0,
    passed: 0,
    failed: 0,
    failures: [],
    dependencyVersions: {}
  };
  window.__markdownTestReport = report;

  function fail(name, error) {
    report.failures.push({
      name: name,
      message: String(error && error.stack ? error.stack : error)
    });
  }

  function test(name, body) {
    report.total += 1;
    report.executed += 1;
    try {
      body();
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

  function render(markdown) {
    var el = document.createElement("div");
    el.className = "message-content";
    testRoot.appendChild(el);
    var ok = window.TeachableMarkdownRenderer.apply(el, markdown);
    assert(ok === true, "apply returned false for: " + markdown);
    return el;
  }

  function element(root, selector) {
    return root.querySelector(selector);
  }

  function text(el) {
    return el.textContent || "";
  }

  try {
    report.dependencyVersions.marked = window.marked ? "18.0.14" : null;
    report.dependencyVersions.dompurify = window.DOMPurify
      ? window.DOMPurify.version
      : null;

    test("real dependencies and renderer API are loaded", function () {
      assert(window.marked && typeof window.marked.Marked === "function",
        "marked.Marked is missing");
      assert(window.DOMPurify && window.DOMPurify.isSupported === true,
        "DOMPurify is not supported in this environment");
      assert(window.DOMPurify.version === "3.4.16",
        "unexpected DOMPurify version " + window.DOMPurify.version);
      assert(typeof window.TeachableMarkdownRenderer.apply === "function",
        "TeachableMarkdownRenderer.apply is missing");
      assert(typeof window.TeachableMarkdownRenderer.render === "function",
        "TeachableMarkdownRenderer.render is missing");
    });

    test("headings, paragraphs, bold and italic render as real elements", function () {
      var el = render("# 标题\n\n**粗体** 和 *斜体*");
      assert(element(el, "h1") !== null, "h1 missing");
      assert(element(el, "p") !== null, "p missing");
      assert(element(el, "strong") !== null, "strong missing");
      assert(element(el, "em") !== null, "em missing");
    });

    test("ordered, unordered and nested lists render as real elements", function () {
      var el = render("- 一\n- 二\n  - 嵌套\n\n1. 甲\n2. 乙");
      assert(element(el, "ul") !== null, "ul missing");
      assert(element(el, "ul ul") !== null, "nested ul missing");
      assert(element(el, "ol") !== null, "ol missing");
      assert(element(el, "li") !== null, "li missing");
    });

    test("blockquotes render as real elements", function () {
      var el = render("> 引用");
      var quote = element(el, "blockquote");
      assert(quote !== null, "blockquote missing");
      assert(text(quote).indexOf("引用") >= 0, "quote text missing");
    });

    test("inline code renders as code text", function () {
      var el = render("行内 `code`");
      var code = element(el, "code");
      assert(code !== null, "inline code missing");
      assert(element(el, "pre") === null, "inline code became a block");
      assert(text(code) === "code", "inline code text mismatch");
    });

    test("fenced code preserves HTML and Markdown literally", function () {
      var marker = "window.__markdown_code_executed = true;";
      var fence = "```html\n<script>" + marker + "</script>\n**not strong**\n```";
      var el = render(fence);
      var pre = element(el, "pre");
      assert(pre !== null, "pre missing");
      assert(element(pre, "strong") === null,
        "Markdown inside a fence was parsed as emphasis");
      assert(element(pre, "script") === null,
        "HTML inside a fence became a script element");
      assert(text(pre).indexOf("<script>" + marker + "</script>") >= 0,
        "fenced HTML text was not preserved");
      assert(text(pre).indexOf("**not strong**") >= 0,
        "fenced Markdown text was not preserved");
      assert(window.__markdown_code_executed === undefined,
        "fenced script executed");
    });

    test("horizontal rules render as hr", function () {
      var el = render("---");
      assert(element(el, "hr") !== null, "hr missing");
    });

    test("GFM tables render as real table elements", function () {
      var el = render("| A | B |\n|---|---|\n| 1 | 2 |");
      assert(element(el, "table") !== null, "table missing");
      assert(element(el, "th") !== null, "th missing");
      assert(element(el, "td") !== null, "td missing");
    });

    test("strikethrough renders as del", function () {
      var el = render("~~删除~~");
      assert(element(el, "del") !== null, "del missing");
    });

    test("escaped Markdown markers remain literal text", function () {
      var el = render("\\*not italic\\*");
      assert(element(el, "em") === null, "escaped marker became em");
      assert(text(el).indexOf("*not italic*") >= 0,
        "literal escaped marker missing");
    });

    test("raw HTML is shown as text and does not execute", function () {
      var el = render(
        '<img src="https://example.invalid/md-raw.png" ' +
        'onerror="window.__markdown_raw_executed = true">'
      );
      assert(element(el, "img") === null, "raw img element survived");
      assert(text(el).indexOf("<img") >= 0, "raw img text missing");
      assert(window.__markdown_raw_executed === undefined,
        "raw HTML event handler executed");
    });

    test("raw script tags are escaped and do not execute", function () {
      var el = render(
        "<script>window.__markdown_script_executed = true</script>"
      );
      assert(element(el, "script") === null, "raw script element survived");
      assert(text(el).indexOf("<script>") >= 0,
        "raw script source was not shown as text");
      assert(window.__markdown_script_executed === undefined,
        "raw script executed");
    });

    test("iframe, form, SVG, MathML and style tags are shown as text", function () {
      var el = render(
        '<iframe src="https://example.invalid"></iframe>' +
        '<form><input></form>' +
        '<svg><circle></circle></svg>' +
        '<math><mi>x</mi></math>' +
        '<style>body { color: red; }</style>'
      );
      assert(element(el, "iframe") === null, "iframe survived");
      assert(element(el, "form") === null, "form survived");
      assert(element(el, "input") === null, "input survived");
      assert(element(el, "svg") === null, "svg survived");
      assert(element(el, "math") === null, "math survived");
      assert(element(el, "style") === null, "style survived");
      assert(text(el).indexOf("<iframe") >= 0, "iframe source missing");
    });

    test("event attributes are rejected with the tag as text", function () {
      var el = render(
        '<a href="https://example.com" ' +
        'onclick="window.__markdown_click_executed = true">x</a>'
      );
      assert(element(el, "a") === null, "raw anchor with handler survived");
      assert(text(el).indexOf("<a") >= 0, "raw anchor source missing");
      assert(window.__markdown_click_executed === undefined,
        "event handler attribute executed");
    });

    test("javascript: links degrade to readable text", function () {
      var el = render("[bad](javascript:alert(1))");
      assert(element(el, "a") === null, "javascript link became an anchor");
      assert(text(el).indexOf("bad") >= 0, "link label missing");
    });

    test("data: links degrade to readable text", function () {
      var el = render("[data](data:text/html,hi)");
      assert(element(el, "a") === null, "data link became an anchor");
      assert(text(el).indexOf("data") >= 0, "link label missing");
    });

    test("fragment and relative links degrade to readable text", function () {
      var el = render("[frag](#here) [rel](docs/page)");
      assert(element(el, "a") === null,
        "fragment or relative link became an anchor");
      assert(text(el).indexOf("frag") >= 0, "fragment label missing");
      assert(text(el).indexOf("rel") >= 0, "relative label missing");
    });

    test("safe https links keep text and fixed target/rel", function () {
      var el = render("[safe](https://example.com/path?a=1&b=2)");
      var link = element(el, "a");
      assert(link !== null, "https link missing");
      assert(text(link).indexOf("safe") >= 0, "https link text missing");
      assert((link.getAttribute("href") || "").indexOf(
        "https://example.com/path?a=1&b=2"
      ) === 0, "https href missing");
      assert(link.getAttribute("target") === "_blank",
        "target is not fixed to _blank");
      var rel = link.getAttribute("rel") || "";
      assert(rel.indexOf("noopener") >= 0, "noopener missing");
      assert(rel.indexOf("noreferrer") >= 0, "noreferrer missing");
    });

    test("safe mailto links keep text and fixed target/rel", function () {
      var el = render("[mail](mailto:a@example.com)");
      var link = element(el, "a");
      assert(link !== null, "mailto link missing");
      assert((link.getAttribute("href") || "").indexOf("mailto:") === 0,
        "mailto href missing");
      assert(link.getAttribute("target") === "_blank",
        "mailto target is not fixed to _blank");
      var rel = link.getAttribute("rel") || "";
      assert(rel.indexOf("noopener") >= 0, "mailto noopener missing");
      assert(rel.indexOf("noreferrer") >= 0, "mailto noreferrer missing");
    });

    test("images degrade to escaped alt text without an img", function () {
      var el = render(
        "![图片 **alt**](https://example.invalid/md-image-alt.png)"
      );
      assert(element(el, "img") === null, "image element was generated");
      assert(text(el).indexOf("图片") >= 0, "alt text missing");
      assert(text(el).indexOf("alt") >= 0, "alt text missing");
    });

    test("images do not trigger automatic resource requests", function () {
      var token = "md-no-request-" + String(Date.now());
      var url = "https://example.invalid/" + token + ".png";
      var el = render("![" + token + "](" + url + ")");
      assert(element(el, "img") === null, "image element was generated");
      var entries = performance.getEntriesByType("resource");
      var requested = entries.some(function (entry) {
        return typeof entry.name === "string" &&
          entry.name.indexOf(token) >= 0;
      });
      assert(requested === false, "an image resource request was observed");
    });

    test("Chinese text and emoji survive rendering", function () {
      var el = render("中文 **粗体** 🎉");
      assert(text(el).indexOf("中文") >= 0, "Chinese text missing");
      assert(text(el).indexOf("🎉") >= 0, "emoji missing");
      assert(element(el, "strong") !== null, "Chinese bold missing");
    });

    test("repeated renders are stable", function () {
      var source = "**重复**\n\n- 一\n- 二";
      var first = render(source).innerHTML;
      for (var i = 0; i < 10; i += 1) {
        var current = render(source).innerHTML;
        assert(current === first, "render changed on repeat " + i);
      }
    });

    test("wrapper falls back to exact text when dependencies are missing", function () {
      var el = document.createElement("div");
      var local = window.TeachableMarkdownRenderer.createRenderer(
        undefined, undefined
      );
      var source = "**fallback**\n\nsecond line";
      var ok = local.apply(el, source);
      assert(ok === false, "missing dependencies did not fail closed");
      assert(el.textContent === source, "fallback text changed");
      assert(el.classList.contains("message-content--markdown") === false,
        "fallback kept the Markdown class");
    });

    test("wrapper falls back when the sanitizer returns a non-string", function () {
      function FakeMarked() {}
      FakeMarked.prototype.use = function () { return this; };
      FakeMarked.prototype.parse = function () { return "<p>x</p>"; };
      var fakePurify = {
        version: "3.4.16",
        isSupported: true,
        sanitize: function () { return { html: "<p>x</p>" }; }
      };
      var el = document.createElement("div");
      var local = window.TeachableMarkdownRenderer.createRenderer(
        { Marked: FakeMarked }, fakePurify
      );
      var source = "**fallback**";
      var ok = local.apply(el, source);
      assert(ok === false, "non-string sanitizer result was accepted");
      assert(el.textContent === source, "fallback text changed");
      assert(el.classList.contains("message-content--markdown") === false,
        "fallback kept the Markdown class");
    });

    test("document uses standards mode and the real style.css", function () {
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
            candidate.href.indexOf("style.css") >= 0;
        })[0];
      assert(sheet, "real style.css stylesheet is not loaded");
      assert(sheet.cssRules && sheet.cssRules.length > 0,
        "style.css has no readable rules");

      var sample = document.createElement("div");
      sample.className = "message-content";
      var source = "line one\nline two";
      sample.textContent = source;
      testRoot.appendChild(sample);
      try {
        assert(sample.textContent === source, "plain text fallback changed");
        assert(getComputedStyle(sample).whiteSpace === "pre-wrap",
          "plain text white-space is " + getComputedStyle(sample).whiteSpace);
      } finally {
        testRoot.removeChild(sample);
      }
    });

    test("Markdown body and code keep their white-space contract", function () {
      var el = render("正文\n\n```js\n  indented line\n```");
      var pre = element(el, "pre");
      var code = element(el, "pre code");

      assert(pre, "pre element missing");
      assert(code, "code element missing");
      assert(getComputedStyle(el).whiteSpace === "normal",
        "Markdown body white-space is " + getComputedStyle(el).whiteSpace);
      assert(getComputedStyle(pre).whiteSpace === "pre",
        "pre white-space is " + getComputedStyle(pre).whiteSpace);
      assert(getComputedStyle(code).whiteSpace === "pre",
        "code white-space is " + getComputedStyle(code).whiteSpace);
      assert(code.textContent.indexOf("  indented line") >= 0,
        "code indentation was not preserved");
    });

    report.done = true;
  } catch (error) {
    fail("suite setup", error);
    report.done = true;
  }
})();
