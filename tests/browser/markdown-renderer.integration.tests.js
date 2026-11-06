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
    executedNames: [],
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
    report.executedNames.push(name);
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

    test("inline and display math render MathML", function () {
      var el = render("Inline \\(y_i\\) and block:\n\n\\[\\sum_{i=1}^{n} i\\]");
      var maths = el.querySelectorAll("math");
      assert(maths.length === 2, "expected two math nodes, got " + maths.length);
      for (var i = 0; i < maths.length; i += 1) {
        assert(maths[i].namespaceURI === "http://www.w3.org/1998/Math/MathML",
          "wrong MathML namespace");
        assert(maths[i].querySelector("annotation[encoding='application/x-tex']"),
          "annotation missing");
      }
      assert(el.textContent.indexOf("\\(") === -1, "literal inline delimiter remains");
      assert(el.textContent.indexOf("\\[") === -1, "literal block delimiter remains");
    });

    test("dollar delimiters render inline and display MathML", function () {
      var el = render("Inline $x$ and display:\n\n$$y$$");
      var maths = el.querySelectorAll("math");
      assert(maths.length === 2, "expected two dollar math nodes, got " + maths.length);
      assert(maths[0].querySelector("annotation[encoding='application/x-tex']").textContent === "x",
        "inline dollar annotation mismatch");
      assert(maths[1].querySelector("annotation[encoding='application/x-tex']").textContent === "y",
        "display dollar annotation mismatch");
      assert(maths[1].getAttribute("display") === "block",
        "display dollar math is not block MathML");
      var visible = el.cloneNode(true);
      visible.querySelectorAll("annotation").forEach(function (a) { a.remove(); });
      assert(visible.textContent.indexOf("$x$") === -1, "inline dollar delimiter remains");
      assert(visible.textContent.indexOf("$$y$$") === -1, "display dollar delimiter remains");
    });

    test("dollar delimiters do not consume currency, escapes or code", function () {
      var currency = render("**Price** is $5 and $10");
      assert(currency.querySelectorAll("math").length === 0,
        "currency became math");
      assert(currency.querySelector("strong"), "currency bold was lost");
      assert(currency.textContent.indexOf("$5 and $10") >= 0,
        "currency text changed");

      var escaped = render("\\$x\\$ and `$y$`\n\n```\n$z$\n```\n\nvalid $w$");
      assert(escaped.querySelectorAll("math").length === 1,
        "code or escaped dollar became math");
      assert(escaped.textContent.indexOf("$x$") >= 0, "escaped dollar lost");
      assert(escaped.textContent.indexOf("$y$") >= 0, "inline code lost");
      assert(escaped.textContent.indexOf("$z$") >= 0, "fenced code lost");
    });

    test("empty, unclosed and over-budget dollar formulas fail locally", function () {
      var empty = render("empty $ $ and valid $x$");
      assert(empty.querySelectorAll("math").length === 1,
        "empty dollar removed adjacent valid math");
      assert(empty.textContent.indexOf("$ $") >= 0, "empty dollar source lost");

      var unclosed = render("unclosed $x");
      assert(unclosed.querySelectorAll("math").length === 0,
        "unclosed dollar became math");
      assert(unclosed.textContent.indexOf("$x") >= 0, "unclosed dollar source lost");

      var unclosedCurrency = render("Price is $5");
      assert(unclosedCurrency.querySelectorAll("math").length === 0,
        "unclosed currency became math");
      assert(unclosedCurrency.textContent.indexOf("$5") >= 0,
        "unclosed currency source lost");

      var over = document.createElement("div");
      over.className = "message-content";
      testRoot.appendChild(over);
      var ok = window.TeachableMarkdownRenderer.apply(over, "$" + "x".repeat(9000) + "$");
      assert(ok === false, "scan-limit dollar did not fail closed");
      assert(over.textContent.indexOf("$") >= 0, "scan-limit dollar source lost");
    });

    test("math structures: fraction, scripts, hat, root and sums", function () {
      var src = "\\(\\frac{a}{b}\\) \\(x_i^2\\) \\(\\hat{y}_i\\) " +
        "\\(\\sqrt{x}\\) \\(\\sum_{i=1}^{n} i\\)";
      var el = render(src);
      assert(el.querySelector("mfrac"), "fraction missing");
      assert(el.querySelector("msubsup, msup, msub"), "scripts missing");
      assert(el.querySelector("mover"), "hat/accent missing");
      assert(el.querySelector("msqrt"), "root missing");
      assert(el.querySelector("munderover, munder, mover"), "sum limits missing");
      assert(el.querySelectorAll("math").length === 5, "math count mismatch");
    });

    test("317 and 323 math fixtures", function () {
      var src317 = "Understood: \\(y_i\\) is the target value; " +
        "\\(\\hat{y}_i\\) is the predicted value, with " +
        "\\(\\hat{y}_i = w x_i + b\\).";
      var el317 = render(src317);
      assert(el317.querySelectorAll("math").length === 3,
        "317 should have three math nodes");
      var visible = el317.cloneNode(true);
      visible.querySelectorAll("annotation").forEach(function (a) { a.remove(); });
      assert(visible.textContent.indexOf("\\hat") === -1,
        "317 still shows a TeX command visibly");
      var src323 = "好，预测值是 ŷ_i = w * x_i + b。";
      var el323 = render(src323);
      assert(el323.querySelectorAll("math").length === 0,
        "323 must not become MathML");
      assert(el323.textContent === src323 + "\n", "323 text changed");
    });

    test("math in lists and tables", function () {
      var src = "- \\(x_i\\)\n- \\(\\frac{a}{b}\\)\n\n" +
        "| header | value |\n| --- | --- |\n| \\(\\sqrt{x}\\) | \\(\\sum_{i=1}^{n} i\\) |";
      var el = render(src);
      assert(el.querySelectorAll("math").length === 4,
        "list/table math count mismatch");
      assert(el.querySelector("li math"), "list math missing");
      assert(el.querySelector("td math, th math"), "table math missing");
    });

    test("math excluded in code, link targets and raw HTML", function () {
      var src = "`\\(x_i\\)`\n\n```\n\\(y_i\\)\n```\n\n" +
        "[label](https://example.com/\\(z_i\\))\n\n" +
        "[label \\(x_i\\)](https://example.com)\n\n" +
        "![alt \\(y_i\\)](https://example.com/a.png)\n\n" +
        "<code>\\(bad\\)</code>\n\n<math><mi>raw</mi></math>";
      var el = render(src);
      assert(el.querySelectorAll("math").length === 0,
        "math appeared in an excluded region");
      assert(el.querySelector("pre"), "code fence missing");
      assert(el.textContent.indexOf("\\(x_i\\)") >= 0,
        "inline code source lost");
      assert(el.textContent.indexOf("<math>") >= 0,
        "raw MathML source lost");
    });

    test("unclosed and scan-limit formulas fall back whole message", function () {
      var unclosed = "**keep** \\(x_i\\) \\(\\frac{x}{y} * _z";
      var el = document.createElement("div");
      var ok = window.TeachableMarkdownRenderer.apply(el, unclosed);
      assert(ok === false, "unclosed formula did not fail whole message");
      assert(el.textContent === unclosed, "unclosed source changed");
      assert(el.classList.contains("message-content--markdown") === false,
        "unclosed fallback kept Markdown class");

      var distant = "\\(" + "x".repeat(9000) + "\\)";
      var el2 = document.createElement("div");
      var ok2 = window.TeachableMarkdownRenderer.apply(el2, distant);
      assert(ok2 === false, "scan-limit formula did not fail whole message");
      assert(el2.textContent === distant, "scan-limit source changed");
      assert(el2.classList.contains("message-content--markdown") === false,
        "scan-limit fallback kept Markdown class");
    });

    test("excluded unclosed delimiters do not fail whole message", function () {
      var sources = [
        "```\nA \\(\\frac{x}{y}\n```",
        "`A \\(\\frac{x}{y}`",
        "[A \\(\\frac{x}{y}](https://example.com)",
        "<code>A \\(\\frac{x}{y}</code>"
      ];
      sources.forEach(function (src) {
        var el = document.createElement("div");
        var ok = window.TeachableMarkdownRenderer.apply(el, src);
        assert(ok === true, "excluded unclosed delimiter failed: " + src);
        assert(el.querySelectorAll("math").length === 0,
          "excluded region generated math");
      });
    });

    test("trust commands fall back locally without injected attributes/resources", function () {
      var commands = [
        "\\href{https://example.com}{x}",
        "\\url{https://example.com}",
        "\\includegraphics{example.png}",
        "\\htmlClass{evil-class}{x}",
        "\\htmlStyle{color:red}{x}"
      ];
      commands.forEach(function (tex) {
        var src = "**keep** \\(x_i\\) \\(" + tex + "\\)";
        var el = render(src);
        assert(el.querySelectorAll("math").length === 1,
          "adjacent valid formula missing for " + tex);
        assert(el.textContent.indexOf(tex) >= 0,
          "trusted command source missing for " + tex);
        assert(el.querySelector("a") === null, "unexpected anchor for " + tex);
        assert(el.querySelector("img") === null, "unexpected image for " + tex);
        assert(el.querySelector(".evil-class") === null,
          "injected class survived for " + tex);
        assert(el.querySelector("[style]") === null,
          "injected style survived for " + tex);
      });
    });

    test("math output has no placeholders and reparses as MathML", function () {
      var src = "Inline \\(y_i\\) and \\(\\frac{a}{b}\\).";
      var el = render(src);
      assert(el.innerHTML.indexOf("MATHPH") === -1, "placeholder leaked");
      var holder = document.createElement("div");
      holder.innerHTML = el.innerHTML;
      assert(holder.querySelectorAll("math").length === 2,
        "serialized/reparsed math count mismatch");
      for (var i = 0; i < holder.querySelectorAll("math").length; i += 1) {
        assert(holder.querySelectorAll("math")[i].namespaceURI ===
          "http://www.w3.org/1998/Math/MathML", "namespace lost after reparse");
      }
    });

    test("math limits and malformed formulas fall back locally", function () {
      var empty = render("empty \\(\\) and valid \\(x_i\\)");
      assert(empty.querySelectorAll("math").length === 1,
        "empty formula removed the valid adjacent formula");
      assert(empty.textContent.indexOf("\\(\\)") >= 0,
        "empty formula source lost");

      var tooLong = "\\(" + "x".repeat(4001) + "\\)";
      var elLong = render("before \\(x_i\\) " + tooLong + " after");
      assert(elLong.querySelectorAll("math").length === 1,
        "expr-limit formula removed the valid adjacent formula");
      assert(elLong.textContent.indexOf(tooLong) >= 0,
        "expr-limit source lost");

      var many = [];
      for (var i = 0; i < 21; i += 1) many.push("\\(x_" + i + "\\)");
      var elMany = render(many.join(" "));
      assert(elMany.querySelectorAll("math").length === 20,
        "count-limit should retain exactly 20 formulas");
      assert(elMany.textContent.indexOf("\\(") >= 0,
        "count-limit formula was not kept as source");

      var matrix = render("\\(\\begin{matrix}a&b\\end{matrix}\\)");
      assert(matrix.querySelectorAll("math").length === 0,
        "unsupported matrix structure became MathML");
      assert(matrix.textContent.indexOf("\\begin{matrix}") >= 0,
        "unsupported matrix source lost");

      var dollars = render("$x$ and $$y$$");
      assert(dollars.querySelectorAll("math").length === 2,
        "$ delimiters did not become math");
      var visibleDollars = dollars.cloneNode(true);
      visibleDollars.querySelectorAll("annotation").forEach(function (a) { a.remove(); });
      assert(visibleDollars.textContent.indexOf("$x$") === -1,
        "$ source remains");
    });

    test("placeholder limit and whole fallback do not leak state", function () {
      var tooMany = [];
      for (var i = 0; i < 129; i += 1) tooMany.push("\\(\\)");
      var source = tooMany.join(" ");
      var el = document.createElement("div");
      var ok = window.TeachableMarkdownRenderer.apply(el, source);
      assert(ok === false, "placeholder limit did not fail whole message");
      assert(el.textContent === source, "placeholder-limit source changed");
      assert(el.classList.contains("message-content--markdown") === false,
        "placeholder-limit fallback kept Markdown class");

      var next = document.createElement("div");
      var nextOk = window.TeachableMarkdownRenderer.apply(next, "Next \\(x_i\\)");
      assert(nextOk === true, "next message failed after placeholder limit");
      assert(next.querySelectorAll("math").length === 1,
        "next message did not recover");
    });

    test("missing KaTeX falls back per formula without disabling Markdown", function () {
      var saved = window.katex;
      try {
        window.katex = undefined;
        var renderer = window.TeachableMarkdownRenderer.createRenderer(
          window.marked, window.DOMPurify, undefined
        );
        var el = document.createElement("div");
        var ok = renderer.apply(el, "**keep** \\(x_i\\)");
        assert(ok === true, "missing KaTeX disabled Markdown");
        assert(el.querySelectorAll("math").length === 0,
          "missing KaTeX produced MathML");
        assert(el.querySelector("strong") !== null,
          "missing KaTeX removed Markdown");
        assert(el.textContent.indexOf("\\(x_i\\)") >= 0,
          "missing KaTeX lost the source formula");
      } finally {
        window.katex = saved;
      }
    });

    test("final math count corruption falls back whole message", function () {
      var original = window.DOMPurify.sanitize;
      var source = "**keep** \\(x_i\\)";
      try {
        window.DOMPurify.sanitize = function (html, config) {
          var clean = original.call(this, html, config);
          if (config && Array.isArray(config.ALLOWED_TAGS) &&
              config.ALLOWED_TAGS.indexOf("math") < 0) {
            return clean + "<math><mi>injected</mi></math>";
          }
          return clean;
        };
        var el = document.createElement("div");
        var ok = window.TeachableMarkdownRenderer.apply(el, source);
        assert(ok === false, "final count corruption did not fail whole message");
        assert(el.textContent === source, "final corruption source changed");
        assert(el.classList.contains("message-content--markdown") === false,
          "final corruption kept Markdown class");
      } finally {
        window.DOMPurify.sanitize = original;
      }
    });

    test("excluded math does not consume valid budgets", function () {
      var links20 = [];
      for (var i = 0; i < 20; i += 1) {
        links20.push("[\\(x\\)](https://example.com)");
      }
      var el20 = render(links20.join(" ") + " outside \\(y\\)");
      assert(el20.querySelectorAll("math").length === 1,
        "20 excluded formulas starved the valid formula");
      assert(el20.querySelectorAll("a").length === 20,
        "excluded link formulas were lost");

      var links129 = [];
      for (var j = 0; j < 129; j += 1) {
        links129.push("[\\(x\\)](https://example.com)");
      }
      var el129 = render(links129.join(" ") + " outside \\(y\\)");
      assert(el129.querySelectorAll("math").length === 1,
        "129 excluded formulas blocked the valid formula");
      assert(el129.querySelectorAll("a").length === 129,
        "129 excluded link formulas were not preserved");
    });

    test("raw HTML boundaries do not swallow outside math", function () {
      var el = render("<span>\\(unclosed</span> outside \\(y\\)");
      assert(el.querySelectorAll("math").length === 1,
        "outside formula was swallowed by the HTML candidate");
      assert(el.textContent.indexOf("<span>\\(unclosed</span>") >= 0,
        "raw HTML candidate text changed");
      assert(el.textContent.indexOf("outside yy") >= 0,
        "outside text/formula was lost");

      var complete = render("<span>\\(x_i\\)</span> outside \\(y\\)");
      assert(complete.querySelectorAll("math").length === 1,
        "complete raw HTML candidate swallowed the outside formula");
      assert(complete.textContent.indexOf("\\(x_i\\)") >= 0,
        "complete raw HTML candidate source was lost");

      var nested = render(
        "**keep** <span><b>\\(unclosed</b></span> outside \\(y\\)"
      );
      assert(nested.querySelectorAll("math").length === 1,
        "nested raw HTML candidate swallowed the outside formula");
      assert(nested.querySelector("strong") !== null,
        "adjacent Markdown was lost around nested HTML");
      assert(nested.textContent.indexOf("</b></span>") >= 0,
        "nested closing tags were lost");

      var adjacent = render("<span>plain</span> **keep** \\(y\\)");
      assert(adjacent.querySelectorAll("math").length === 1,
        "valid formula after raw HTML did not render");
      assert(adjacent.querySelector("strong") !== null,
        "Markdown after raw HTML did not render");
    });

    test("escaped delimiters remain ordinary Markdown", function () {
      var cases = [
        {
          source: "**keep** before \\\\[x\\\\] after",
          html: "<p><strong>keep</strong> before \\[x\\] after</p>\n",
          text: "keep before \\[x\\] after\n"
        },
        {
          source: "**keep** before \\\\(x\\\\) after",
          html: "<p><strong>keep</strong> before \\(x\\) after</p>\n",
          text: "keep before \\(x\\) after\n"
        },
        {
          source: "**keep** before \\\\[x\\] after",
          html: "<p><strong>keep</strong> before \\[x] after</p>\n",
          text: "keep before \\[x] after\n"
        },
        {
          source: "**keep** before \\) after",
          html: "<p><strong>keep</strong> before ) after</p>\n",
          text: "keep before ) after\n"
        }
      ];
      cases.forEach(function (entry) {
        var el = render(entry.source);
        assert(el.querySelectorAll("math").length === 0,
          "escaped delimiter generated MathML: " + entry.source);
        assert(el.innerHTML === entry.html,
          "escaped delimiter HTML differs from baseline: " + entry.source);
        assert(el.textContent === entry.text,
          "escaped delimiter text differs from baseline: " + entry.source);
        assert(el.querySelector("strong") !== null,
          "escaped delimiter destroyed adjacent Markdown: " + entry.source);
      });
    });

    test("leading escaped delimiters and block contexts", function () {
      var leadingBoth = render("\\\\[x\\\\]");
      assert(leadingBoth.querySelectorAll("math").length === 0,
        "leading escaped opener/closer generated MathML");
      assert(leadingBoth.innerHTML === "<p>\\[x\\]</p>\n",
        "leading escaped opener/closer differs from baseline");
      assert(leadingBoth.textContent === "\\[x\\]\n",
        "leading escaped opener/closer text changed");

      var leadingOpen = render("\\\\[x\\]");
      assert(leadingOpen.querySelectorAll("math").length === 0,
        "leading escaped opener generated MathML");
      assert(leadingOpen.innerHTML === "<p>\\[x]</p>\n",
        "leading escaped opener differs from baseline");

      var leadingInline = render("\\\\(x\\\\)");
      assert(leadingInline.querySelectorAll("math").length === 0,
        "leading escaped inline delimiter generated MathML");
      assert(leadingInline.innerHTML === "<p>\\(x\\)</p>\n",
        "leading escaped inline delimiter differs from baseline");

      var paragraph = render("before\n\n\\\\[x\\\\]");
      assert(paragraph.querySelectorAll("math").length === 0,
        "escaped block opener at paragraph start generated MathML");
      assert(paragraph.textContent.indexOf("\\[x\\]") >= 0,
        "paragraph-start escaped source changed");

      var line = render("before\n\\\\[x\\\\]");
      assert(line.querySelectorAll("math").length === 0,
        "escaped block opener after newline generated MathML");

      var list = render("- \\\\[x\\\\]");
      assert(list.querySelectorAll("math").length === 0,
        "escaped block opener in a list generated MathML");
      assert(list.querySelector("li") !== null,
        "list structure was lost");

      var normalInline = render("before \\(x\\)");
      assert(normalInline.querySelectorAll("math").length === 1,
        "normal inline formula stopped working");
      var normalBlock = render("before\n\n\\[x\\]");
      assert(normalBlock.querySelectorAll("math").length === 1,
        "normal block formula stopped working");
    });

    test("formula text containing HTML tags remains MathML", function () {
      var inline = render(
        "**keep** \\(\\text{</span>}\\) and \\(y\\)"
      );
      assert(inline.querySelectorAll("math").length === 2,
        "literal tag text inside formula blocked adjacent math");
      assert(inline.querySelector("strong") !== null,
        "adjacent Markdown was lost");
      assert(inline.querySelector("mtext") !== null,
        "literal tag text did not remain in MathML mtext");
      assert(inline.textContent.indexOf("</span>") >= 0,
        "literal tag text was removed from the formula");

      var display = render(
        "**keep**\n\n\\[\\text{</span>}\\]\n\nafter"
      );
      assert(display.querySelectorAll("math").length === 1,
        "display formula with literal tag text did not render");
      assert(display.querySelector("mtext") !== null,
        "display literal tag text did not remain in MathML");
      assert(display.textContent.indexOf("</span>") >= 0,
        "display literal tag text was removed");
      assert(display.textContent.indexOf("after") >= 0,
        "display adjacent content was lost");
    });

    test("nested Markdown inherits raw HTML context", function () {
      var emphasis = render(
        "<span>*\\(unclosed</span> outside \\(y\\)*"
      );
      assert(emphasis.querySelectorAll("math").length === 1,
        "emphasis swallowed the outside formula");
      assert(emphasis.querySelector("em") !== null,
        "emphasis structure was lost");
      assert(emphasis.textContent.indexOf("\\(unclosed") >= 0,
        "emphasis raw HTML candidate text changed");

      var strong = render(
        "<span>**\\(unclosed</span> outside \\(y\\)**"
      );
      assert(strong.querySelectorAll("math").length === 1,
        "strong swallowed the outside formula");
      assert(strong.querySelector("strong") !== null,
        "strong structure was lost");

      var normal = render(
        "<span>**bold**</span> outside \\(y\\)"
      );
      assert(normal.querySelectorAll("math").length === 1,
        "normal nested HTML lost the outside formula");
      assert(normal.querySelector("strong") !== null,
        "normal nested Markdown structure was lost");

      var link = render(
        "<span>[\\(unclosed</span> outside \\(y\\)](" +
        "https://example.com)"
      );
      assert(link.querySelectorAll("math").length === 0,
        "link-label formula escaped exclusion");
      assert(link.querySelector("a") !== null,
        "link structure was lost");

      var image = render(
        "<span>![\\(unclosed</span> outside \\(y\\)](" +
        "https://example.com/a.png)"
      );
      assert(image.querySelectorAll("math").length === 0,
        "image-label formula escaped exclusion");
    });

    test("raw HTML context is isolated between blocks and cells", function () {
      var paragraph = render(
        "<span>\\(unclosed\n\noutside \\(y\\)"
      );
      assert(paragraph.querySelectorAll("math").length === 1,
        "raw HTML context leaked across paragraphs");

      var list = render(
        "- <span>\\(unclosed\n- outside \\(y\\)"
      );
      assert(list.querySelectorAll("math").length === 1,
        "raw HTML context leaked across list items");
      assert(list.querySelectorAll("li").length === 2,
        "list item structure was lost");

      var table = render(
        "| a | b |\n|---|---|\n| <span>\\(unclosed | \\(y\\) |"
      );
      assert(table.querySelectorAll("math").length === 1,
        "raw HTML context leaked across table cells");
      assert(table.querySelectorAll("td").length === 2,
        "table cell structure was lost");
    });

    test("link and image HTML state does not leak outward", function () {
      var link = render(
        "[<span>](https://example.com) outside \\(y\\)"
      );
      assert(link.querySelectorAll("math").length === 1,
        "link-label HTML leaked into the outside scope");
      assert(link.querySelector("a") !== null,
        "link structure was lost");

      var image = render(
        "![<span>](https://example.com/a.png) outside \\(y\\)"
      );
      assert(image.querySelectorAll("math").length === 1,
        "image-alt HTML leaked into the outside scope");

      var linkLiteral = render(
        "[<span>](https://example.com) outside \\(\\text{</span>}\\)"
      );
      assert(linkLiteral.querySelectorAll("math").length === 1,
        "link label HTML blocked an outside literal-tag formula");
      assert(linkLiteral.querySelector("mtext") !== null,
        "outside literal-tag formula did not render as mtext");

      var imageLiteral = render(
        "![<span>](https://example.com/a.png) outside " +
        "\\(\\text{</span>}\\)"
      );
      assert(imageLiteral.querySelectorAll("math").length === 1,
        "image alt HTML blocked an outside literal-tag formula");
      assert(imageLiteral.querySelector("mtext") !== null,
        "image-side literal-tag formula did not render as mtext");
    });

    test("link and image HTML state is isolated across scopes", function () {
      var linkClose = render(
        "<span>[</span>](https://example.com) \\(x\\)</span> " +
        "outside \\(y\\)"
      );
      assert(linkClose.querySelectorAll("math").length === 1,
        "link-label close tag closed the genuine outer scope");
      assert(linkClose.textContent.indexOf("\\(x\\)") >= 0,
        "formula inside the genuine outer scope was not excluded");

      var imageClose = render(
        "<span>![</span>](https://example.com/a.png) \\(x\\)</span> " +
        "outside \\(y\\)"
      );
      assert(imageClose.querySelectorAll("math").length === 1,
        "image-alt close tag closed the genuine outer scope");

      var strong = render(
        "<span>**[</span>](https://example.com)** \\(x\\)</span> " +
        "outside \\(y\\)"
      );
      assert(strong.querySelectorAll("math").length === 1,
        "strong/link scope leaked HTML state");
      assert(strong.querySelector("strong") !== null,
        "strong structure was lost");

      var emphasis = render(
        "<span>*[</span>](https://example.com)* \\(x\\)</span> " +
        "outside \\(y\\)"
      );
      assert(emphasis.querySelectorAll("math").length === 1,
        "emphasis/link scope leaked HTML state");
      assert(emphasis.querySelector("em") !== null,
        "emphasis structure was lost");

      var paragraph = render(
        "[<span>](https://example.com)\n\noutside \\(y\\)"
      );
      assert(paragraph.querySelectorAll("math").length === 1,
        "link HTML leaked across paragraphs");

      var list = render(
        "- [<span>](https://example.com)\n- outside \\(y\\)"
      );
      assert(list.querySelectorAll("math").length === 1,
        "link HTML leaked across list items");
      assert(list.querySelectorAll("li").length === 2,
        "list structure was lost");

      var table = render(
        "| a | b |\n|---|---|\n" +
        "| [<span>](https://example.com) | outside \\(y\\) |"
      );
      assert(table.querySelectorAll("math").length === 1,
        "link HTML leaked across table cells");

      var firstMessage = render("[<span>](https://example.com)");
      assert(firstMessage.querySelectorAll("math").length === 0,
        "link-only message unexpectedly rendered math");
      var secondMessage = render("outside \\(y\\)");
      assert(secondMessage.querySelectorAll("math").length === 1,
        "link HTML leaked into the next message");
    });

    test("review: standalone currency remains literal", function () {
      var el = render("$5 and $10");
      assert(el.querySelectorAll("math").length === 0, "standalone currency became math");
      assert(el.textContent.indexOf("$5 and $10") >= 0, "currency source changed");
    });

    test("review: prefixed currency control", function () {
      var el = render("Price $5 and $10");
      assert(el.querySelectorAll("math").length === 0, "prefixed currency became math");
    });

    test("review: leading-space dollar boundary", function () {
      var el = render("$ x$");
      assert(el.querySelectorAll("math").length === 0, "leading-space dollar became math");
    });

    test("review: trailing-space dollar boundary", function () {
      var el = render("$x $");
      assert(el.querySelectorAll("math").length === 0, "trailing-space dollar became math");
    });

    report.done = true;
  } catch (error) {
    fail("suite setup", error);

    report.done = true;
  }
})();
