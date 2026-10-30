/**
 * Wrapper-level tests for frontend/markdown-renderer.js.
 *
 * Real marked and DOMPurify behavior is covered by the browser
 * integration test.  These tests only exercise dependency checks,
 * return-type checks, fallback behavior and configuration wiring with
 * injected fakes.
 *
 * Run: node --test tests/test_markdown_renderer.test.js
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const markdown = require(path.resolve(
  __dirname, "..", "frontend", "markdown-renderer.js"
));
const indexHtml = fs.readFileSync(
  path.resolve(__dirname, "..", "frontend", "index.html"), "utf8"
);

const MARKDOWN_CLASS = "message-content--markdown";

function makeClassList() {
  var classes = [];
  return {
    add: function (name) {
      if (classes.indexOf(name) === -1) classes.push(name);
    },
    remove: function (name) {
      var at = classes.indexOf(name);
      if (at >= 0) classes.splice(at, 1);
    },
    contains: function (name) {
      return classes.indexOf(name) >= 0;
    },
    values: function () {
      return classes.slice();
    }
  };
}

function makeElement(options) {
  options = options || {};
  var el = {
    textContent: options.textContent === undefined ? "" : options.textContent,
    innerHTML: "",
    classList: makeClassList()
  };
  if (options.throwOnInnerHTML === true) {
    Object.defineProperty(el, "innerHTML", {
      configurable: true,
      get: function () { return ""; },
      set: function () { throw new Error("innerHTML write failed"); }
    });
  }
  return el;
}

function makeFakePurify(options) {
  options = options || {};
  var captured = { config: null, calls: 0 };
  var fake = {
    version: "3.4.16",
    isSupported: options.isSupported === undefined ? true : options.isSupported,
    sanitize: function (html, config) {
      captured.calls += 1;
      captured.config = config;
      if (options.throwOnSanitize === true) throw new Error("sanitize failed");
      if (options.sanitizeResult !== undefined) return options.sanitizeResult;
      return options.sanitizeOutput === undefined
        ? "<p>sanitized:" + html + "</p>"
        : options.sanitizeOutput;
    },
    getCaptured: function () { return captured; },
    addHook: function () { fake.hookCalls = (fake.hookCalls || 0) + 1; },
    removeAllHooks: function () { fake.removeCalls = (fake.removeCalls || 0) + 1; },
    setConfig: function () { fake.setConfigCalls = (fake.setConfigCalls || 0) + 1; }
  };
  return fake;
}

function makeFakeMarked(options) {
  options = options || {};
  function FakeMarked() {
    this.constructed = true;
    this.useCalls = [];
    this.parseCalls = 0;
  }
  FakeMarked.prototype.use = function (extension) {
    this.useCalls.push(extension);
    return this;
  };
  FakeMarked.prototype.parse = function (source) {
    this.parseCalls += 1;
    if (options.throwOnParse === true) throw new Error("parse failed");
    if (options.parseResult !== undefined) return options.parseResult;
    return options.parseOutput === undefined
      ? "<p>parsed:" + source + "</p>"
      : options.parseOutput;
  };
  var namespace = {
    Marked: FakeMarked,
    globalUseCalls: 0,
    use: function () {
      namespace.globalUseCalls += 1;
    }
  };
  return namespace;
}

function makeConfigured(renderer) {
  return renderer.configured;
}

describe("markdown-renderer wrapper: dependency checks", function () {
  it("exposes its public API and pinned version constants", function () {
    assert.equal(markdown.MARKDOWN_CLASS, MARKDOWN_CLASS);
    assert.equal(markdown.PURIFY_VERSION, "3.4.16");
    assert.equal(typeof markdown.createRenderer, "function");
    assert.equal(typeof markdown.render, "function");
    assert.equal(typeof markdown.apply, "function");
  });

  it("loads the vendored parser, sanitizer and renderer before app.js", function () {
    var markedAt = indexHtml.indexOf("/static/vendor/marked.umd.js");
    var purifyAt = indexHtml.indexOf("/static/vendor/purify.min.js");
    var rendererAt = indexHtml.indexOf("/static/markdown-renderer.js");
    var appAt = indexHtml.indexOf("/static/app.js");

    assert.equal(markedAt > -1, true);
    assert.equal(purifyAt > -1, true);
    assert.equal(rendererAt > -1, true);
    assert.equal(appAt > -1, true);
    assert.equal(markedAt < rendererAt, true);
    assert.equal(purifyAt < rendererAt, true);
    assert.equal(rendererAt < appAt, true);
  });

  it("falls back when both real dependencies are missing", function () {
    var renderer = markdown.createRenderer(undefined, undefined);
    var el = makeElement();

    assert.equal(makeConfigured(renderer), false);
    assert.equal(renderer.render("**x**").ok, false);
    assert.equal(renderer.apply(el, "**x**"), false);
    assert.equal(el.textContent, "**x**");
    assert.equal(el.classList.contains(MARKDOWN_CLASS), false);
  });

  it("falls back before constructing marked when DOMPurify is unsupported", function () {
    var marked = makeFakeMarked();
    var purify = makeFakePurify({ isSupported: false });
    var renderer = markdown.createRenderer(marked, purify);

    assert.equal(renderer.configured, false);
    assert.equal(marked.Marked.prototype.constructed, undefined);
    assert.equal(renderer.render("x").ok, false);
  });

  it("falls back when marked is unavailable", function () {
    var purify = makeFakePurify();
    var renderer = markdown.createRenderer({ Marked: null }, purify);
    var el = makeElement();

    assert.equal(renderer.apply(el, "**x**"), false);
    assert.equal(el.textContent, "**x**");
  });
});

describe("markdown-renderer wrapper: return-type checks", function () {
  it("falls back when parse throws", function () {
    var renderer = markdown.createRenderer(
      makeFakeMarked({ throwOnParse: true }), makeFakePurify()
    );
    var el = makeElement();

    assert.equal(renderer.apply(el, "**x**"), false);
    assert.equal(el.textContent, "**x**");
  });

  it("falls back when parse returns a Promise", function () {
    var renderer = markdown.createRenderer(
      makeFakeMarked({ parseResult: Promise.resolve("<p>x</p>") }),
      makeFakePurify()
    );

    assert.equal(renderer.render("**x**").ok, false);
  });

  it("falls back when parse returns an object", function () {
    var renderer = markdown.createRenderer(
      makeFakeMarked({ parseResult: { html: "<p>x</p>" } }),
      makeFakePurify()
    );

    assert.equal(renderer.render("**x**").ok, false);
  });

  it("falls back when sanitize throws", function () {
    var renderer = markdown.createRenderer(
      makeFakeMarked(), makeFakePurify({ throwOnSanitize: true })
    );
    var el = makeElement();

    assert.equal(renderer.apply(el, "**x**"), false);
    assert.equal(el.textContent, "**x**");
  });

  it("falls back when sanitize returns an object", function () {
    var renderer = markdown.createRenderer(
      makeFakeMarked(), makeFakePurify({ sanitizeResult: { html: "x" } })
    );

    assert.equal(renderer.render("**x**").ok, false);
  });

  it("falls back when sanitize returns a Promise", function () {
    var renderer = markdown.createRenderer(
      makeFakeMarked(),
      makeFakePurify({ sanitizeResult: Promise.resolve("<p>x</p>") })
    );

    assert.equal(renderer.render("**x**").ok, false);
  });
});

describe("markdown-renderer wrapper: successful apply and config", function () {
  it("writes only the sanitized string and adds the Markdown class", function () {
    var purify = makeFakePurify({ sanitizeOutput: "<p><strong>x</strong></p>" });
    var renderer = markdown.createRenderer(makeFakeMarked(), purify);
    var el = makeElement();

    var result = renderer.apply(el, "**x**");

    assert.equal(makeConfigured(renderer), true);
    assert.equal(result, true);
    assert.equal(el.innerHTML, "<p><strong>x</strong></p>");
    assert.equal(el.classList.contains(MARKDOWN_CLASS), true);
  });

  it("passes the strict whitelist config to sanitize", function () {
    var purify = makeFakePurify();
    var renderer = markdown.createRenderer(makeFakeMarked(), purify);

    renderer.render("**x**");
    var config = purify.getCaptured().config;

    assert.deepEqual(config.ALLOWED_TAGS, [
      "a", "blockquote", "br", "code", "del", "em",
      "h1", "h2", "h3", "h4", "h5", "h6", "hr",
      "li", "ol", "p", "pre", "strong",
      "table", "tbody", "td", "th", "thead", "tr", "ul"
    ]);
    assert.deepEqual(config.ALLOWED_ATTR, [
      "href", "title", "target", "rel", "align", "start"
    ]);
    assert.deepEqual(config.ADD_URI_SAFE_ATTR, [
      "target", "rel", "align", "start"
    ]);
    assert.equal(config.ALLOW_DATA_ATTR, false);
    assert.equal(config.ALLOW_ARIA_ATTR, false);
    assert.equal(config.ALLOW_UNKNOWN_PROTOCOLS, false);
    assert.equal(config.RETURN_TRUSTED_TYPE, false);
    assert.equal(config.KEEP_CONTENT, true);
    assert.equal(config.ALLOWED_URI_REGEXP.test("https://example.com"), true);
    assert.equal(config.ALLOWED_URI_REGEXP.test("mailto:a@example.com"), true);
    assert.equal(config.ALLOWED_URI_REGEXP.test("javascript:alert(1)"), false);
    assert.equal(config.ALLOWED_URI_REGEXP.test("#fragment"), false);
    assert.equal(config.ALLOWED_URI_REGEXP.test("relative/path"), false);
  });

  it("does not mutate the global marked namespace or DOMPurify hooks", function () {
    var marked = makeFakeMarked();
    var purify = makeFakePurify();
    var renderer = markdown.createRenderer(marked, purify);

    renderer.render("**x**");

    assert.equal(marked.globalUseCalls, 0);
    assert.equal(purify.hookCalls, undefined);
    assert.equal(purify.removeCalls, undefined);
    assert.equal(purify.setConfigCalls, undefined);
  });

  it("creates isolated marked instances per renderer", function () {
    var firstMarked = makeFakeMarked();
    var secondMarked = makeFakeMarked();
    var first = markdown.createRenderer(firstMarked, makeFakePurify());
    var second = markdown.createRenderer(secondMarked, makeFakePurify());

    assert.equal(firstMarked.globalUseCalls, 0);
    assert.equal(secondMarked.globalUseCalls, 0);
    assert.equal(first.configured, true);
    assert.equal(second.configured, true);
  });

  it("does not write unsafe HTML when the DOM write throws", function () {
    var purify = makeFakePurify({ sanitizeOutput: "<p>safe</p>" });
    var renderer = markdown.createRenderer(makeFakeMarked(), purify);
    var el = makeElement({ throwOnInnerHTML: true });

    var result = renderer.apply(el, "**x**");

    assert.equal(result, false);
    assert.equal(el.textContent, "**x**");
    assert.equal(el.classList.contains(MARKDOWN_CLASS), false);
  });

  it("preserves exact text, including newlines, on fallback", function () {
    var renderer = markdown.createRenderer(undefined, undefined);
    var el = makeElement();
    var source = "line one\n**line two**\nline three";

    renderer.apply(el, source);

    assert.equal(el.textContent, source);
  });
});
