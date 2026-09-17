const { describe, it, beforeEach } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const indexHtml = fs.readFileSync(
  path.join(ROOT, "frontend", "index.html"), "utf8");
const appSource = fs.readFileSync(
  path.join(ROOT, "frontend", "app.js"), "utf8");
const reviewSource = fs.readFileSync(
  path.join(ROOT, "frontend", "history-review.js"), "utf8");
const styleCss = fs.readFileSync(
  path.join(ROOT, "frontend", "style.css"), "utf8");

/**
 * Tiny CSS reader for the stylesheet contract tests.  It walks the source
 * character by character, so nested at-rules keep their condition and a
 * rule inside @media can still be asserted.  No browser engine needed.
 */
function parseCss(source) {
  const rules = [];
  let buffer = "";
  let i = 0;

  function readBlock(start) {
    let depth = 1;
    let j = start;
    while (j < source.length && depth > 0) {
      if (source[j] === "{") depth += 1;
      else if (source[j] === "}") depth -= 1;
      j += 1;
    }
    return { text: source.slice(start, j - 1), next: j };
  }

  while (i < source.length) {
    const ch = source[i];
    if (ch === "/" && source[i + 1] === "*") {
      // Comments carry prose that must never look like a selector.
      i += 2;
      while (i < source.length &&
             !(source[i] === "*" && source[i + 1] === "/")) i += 1;
      i += 2;
      continue;
    }
    if (ch === "{") {
      const selector = buffer.trim();
      buffer = "";
      const block = readBlock(i + 1);
      i = block.next;
      if (selector.startsWith("@media")) {
        for (const inner of parseCss(block.text)) {
          inner.media = selector.slice(6).trim() +
            (inner.media ? " and " + inner.media : "");
          rules.push(inner);
        }
      } else if (selector.startsWith("@") === false && selector !== "") {
        rules.push({ selector: selector, body: block.text, media: null });
      }
      continue;
    }
    if (ch === ";") { buffer = ""; i += 1; continue; }
    buffer += ch;
    i += 1;
  }
  return rules;
}

/** Splits a selector list and normalises its whitespace. */
function selectorParts(rule) {
  return rule.selector.split(",").map(function (part) {
    return part.replace(/\s+/g, " ").trim();
  });
}

/** Normalises a selector-list string the same way. */
function normaliseSelector(selector) {
  return selector.split(",").map(function (part) {
    return part.replace(/\s+/g, " ").trim();
  }).join(", ");
}

/** The first rule whose selector list contains *selector* verbatim. */
function cssRule(source, selector) {
  const wanted = normaliseSelector(selector);
  return parseCss(source).find(function (rule) {
    return normalizeRuleSelector(rule) === wanted;
  }) || null;
}

function normalizeRuleSelector(rule) {
  return rule.selector.split(",").map(function (part) {
    return part.replace(/\s+/g, " ").trim();
  }).join(", ");
}

/** Declarations for *selector*; the last matching rule wins. */
function cssDeclarations(source, selector) {
  const wanted = normaliseSelector(selector);
  const declarations = Object.create(null);
  for (const rule of parseCss(source)) {
    if (normalizeRuleSelector(rule) !== wanted) continue;
    const body = rule.body.replace(/\/\*[\s\S]*?\*\//g, "");
    const declRe = /([-a-zA-Z]+)\s*:\s*([^;]+)/g;
    let decl;
    while ((decl = declRe.exec(body)) !== null) {
      declarations[decl[1].trim()] = decl[2].trim();
    }
  }
  return declarations;
}

/** Every rule that matches *selector*, in source order. */
function cssRulesFor(source, selector) {
  const wanted = normaliseSelector(selector);
  return parseCss(source).filter(function (rule) {
    return normalizeRuleSelector(rule) === wanted;
  });
}

const historyReview = require(path.join(ROOT, "frontend", "history-review.js"));
const {
  CORRECTIVE_MODE,
  isValidHistoryReviewSummaryList,
  isValidHistoryReviewDetail,
  buildHistoryReviewPanelModel,
  reconcileHistoryReviewCaches,
  upsertHistoryReviewSummary,
  historyReviewSummaryFromDetail,
  isHistoryReviewDetailCurrent,
  isHistoryReviewTargetBusy,
} = historyReview;
const {
  RECEIVE_TEACHING_MODE,
  isValidModeSwitchEvent,
} = require(path.join(ROOT, "frontend", "interaction-mode.js"));
const { isValidApiTimestamp } = require(
  path.join(ROOT, "frontend", "model-selection.js"));
const { createRemoteHistoryConfirmer } = require(
  path.join(ROOT, "frontend", "session-profile-switch.js"));

const PANEL_IDS = [
  "historyReviewPanel", "historyReviewTitle", "historyReviewStatusBadge",
  "historyReviewToggleBtn", "historyReviewBody", "historyReviewLive",
  "historyReviewProposal", "historyReviewProposalText",
  "historyReviewStartBtn", "historyReviewDismissBtn",
  "historyReviewDetail", "historyReviewSelector", "historyReviewSelect",
  "historyReviewSummary",
  "historyReviewCoverage", "historyReviewFindings", "historyReviewError",
  "historyReviewRecheckBtn", "historyReviewContinueBtn",
];

describe("history review DOM contract", function () {
  it("defines every required panel and dialog id", function () {
    for (const id of PANEL_IDS) {
      assert.equal(indexHtml.includes('id="' + id + '"'), true, id);
    }
    assert.equal(indexHtml.includes('id="historyReviewStartDialog"'), true);
    assert.equal(indexHtml.includes('id="historyReviewConsentDialog"'), true);
  });

  it("places the panel after the model bar and before messages", function () {
    const model = indexHtml.indexOf('id="currentProfileBar"');
    const panel = indexHtml.indexOf('id="historyReviewPanel"');
    const messages = indexHtml.indexOf('id="messages"');
    assert.equal(model > -1, true);
    assert.equal(panel > model, true);
    assert.equal(messages > panel, true);
  });

  it("loads history-review.js before app.js", function () {
    const review = indexHtml.indexOf('src="/static/history-review.js');
    const app = indexHtml.indexOf('src="/static/app.js');
    assert.equal(review > -1, true);
    assert.equal(app > -1, true);
    assert.equal(review < app, true);
  });

  it("exposes the review error region as a live status", function () {
    const marker = indexHtml.indexOf('id="historyReviewError"');
    const snippet = indexHtml.slice(marker, marker + 180);
    assert.equal(snippet.includes('role="status"'), true);
    assert.equal(snippet.includes('hidden'), true);
  });

  it("gives both dialogs accessible labels and descriptions", function () {
    assert.equal(indexHtml.includes('aria-labelledby="hrs-title"'), true);
    assert.equal(indexHtml.includes('aria-describedby="hrs-body"'), true);
    assert.equal(indexHtml.includes('aria-labelledby="hrc-title"'), true);
    assert.equal(indexHtml.includes('aria-describedby="hrc-body"'), true);
  });

  it("bumps the style and app cache-busting query", function () {
    assert.equal(
      indexHtml.includes('style.css?v=20260914-history-review3'), true);
    assert.equal(
      indexHtml.includes('app.js?v=20260914-history-review3'), true);
    // app.js depends on the confirmer's setTitle support, so the module
    // that carries it must be cache-busted too.
    assert.equal(
      indexHtml.includes(
        'session-profile-switch.js?v=20260914-history-review3'), true);
  });

  it("groups the review selector with its label", function () {
    const wrapper = indexHtml.indexOf('id="historyReviewSelector"');
    assert.equal(wrapper > -1, true);
    const label = indexHtml.indexOf('for="historyReviewSelect"', wrapper);
    const select = indexHtml.indexOf('id="historyReviewSelect"', wrapper);
    assert.equal(label > wrapper, true);
    assert.equal(select > label, true);
    // The wrapper starts hidden: a single review must not show an empty
    // dropdown before the first render.
    const snippet = indexHtml.slice(wrapper, wrapper + 120);
    assert.equal(snippet.includes('hidden'), true);
  });
});


/* ------------------------------------------------------------------ */
/* Modal, layout and visibility CSS contract                          */
/* ------------------------------------------------------------------ */

describe("history review modal and layout CSS contract", function () {
  it("centres every modal without relying on UA defaults", function () {
    const dialog = cssDeclarations(styleCss, "dialog");
    assert.equal(dialog.position, "fixed");
    assert.equal(dialog.inset, "0");
    assert.equal(dialog.margin, "auto");
    // Explicit auto margins keep centring even where flex layout for a
    // top-layer dialog is not applied.
    assert.equal(dialog.display === "flex" || dialog.margin === "auto", true);
    // All three dialogs share the single rule, so they cannot drift.
    for (const id of ["profileSwitchDialog", "historyReviewStartDialog",
                      "historyReviewConsentDialog"]) {
      const own = cssDeclarations(styleCss, "#" + id);
      assert.equal(own.position === undefined || own.position === "fixed",
        true, id);
      assert.equal(own.margin === undefined || own.margin !== "0px",
        true, id);
    }
  });

  it("keeps modals inside narrow viewports and scrollable when tall", function () {
    const dialog = cssDeclarations(styleCss, "dialog");
    assert.equal(dialog["max-width"], "calc(100vw - 32px)");
    assert.equal(dialog["max-height"], "calc(100vh - 32px)");
    assert.equal(dialog.overflow, "auto");
    assert.notEqual(dialog.width, undefined);
  });

  it("styles dialog actions with one primary and one secondary action", function () {
    const secondary = cssDeclarations(
      styleCss, '.dialog-actions button[data-variant="secondary"]');
    const primary = cssDeclarations(
      styleCss, '.dialog-actions button[data-variant="primary"]');
    assert.equal(secondary.background, "#f0f0f2");
    assert.equal(primary.background, "#007aff");
    assert.equal(primary.color, "#ffffff");

    // Each dialog marks exactly one primary and one secondary button.
    const dialogs = ["profileSwitchDialog", "historyReviewStartDialog",
                     "historyReviewConsentDialog"];
    for (const id of dialogs) {
      const start = indexHtml.indexOf('id="' + id + '"');
      const end = indexHtml.indexOf("</dialog>", start);
      const markup = indexHtml.slice(start, end);
      const primaryCount = (markup.match(/data-variant="primary"/g) || []).length;
      const secondaryCount =
        (markup.match(/data-variant="secondary"/g) || []).length;
      assert.equal(primaryCount, 1, id + " primary buttons");
      assert.equal(secondaryCount, 1, id + " secondary buttons");
      const cancelAt = markup.indexOf("Cancel");
      const primaryAt = markup.indexOf('data-variant="primary"');
      assert.equal(cancelAt > -1 && primaryAt > cancelAt, true,
        id + " primary action follows Cancel");
    }
  });

  it("keeps the review select hidden when the model hides it", function () {
    const hiddenRule = cssDeclarations(
      styleCss, "#historyReviewSelector[hidden], #historyReviewSelect[hidden]");
    assert.equal(hiddenRule.display, "none");
    // The base rule still sets display, which is exactly why the [hidden]
    // override must exist and stay more specific.
    assert.equal(cssDeclarations(styleCss, "#historyReviewSelect").display,
      "block");
    assert.equal(cssDeclarations(styleCss, "#historyReviewSelector[hidden], #historyReviewSelect[hidden]").display,
      "none");
    const label = cssDeclarations(styleCss, "#historyReviewDetail > label");
    assert.equal(label.display, "block");
    assert.equal(
      cssDeclarations(styleCss, "#historyReviewDetail > label[hidden]").display,
      "none");
  });

  it("bounds the review panel and keeps the message list scrollable", function () {
    const panel = cssDeclarations(styleCss, ".history-review-panel");
    assert.notEqual(panel["max-height"], undefined);
    assert.equal(panel["min-height"], "0");
    // The panel yields space before the messages/input do, so the
    // composer stays on screen on a short viewport.
    assert.equal(panel["flex"], "0 1 auto");
    assert.equal(panel.overflow, "hidden");
    const body = cssDeclarations(styleCss, ".history-review-body");
    assert.equal(body["overflow-y"], "auto");
    const header = cssDeclarations(styleCss, ".history-review-header");
    assert.equal(header["flex-shrink"], "0");
    const messages = cssDeclarations(styleCss, ".chat-messages");
    assert.notEqual(messages["min-height"], undefined);
    assert.notEqual(messages["min-height"], "0");
    assert.equal(messages["overflow-y"], "auto");
  });

  it("stops the root document from scrolling on mobile", function () {
    const root = cssDeclarations(styleCss, "html, body");
    assert.equal(root.overflow, "hidden");
    // The chat column itself must be allowed to shrink below its content,
    // which is what keeps the input area on screen.
    const chatAreaRules = cssRulesFor(styleCss, ".chat-area");
    assert.equal(chatAreaRules.length > 0, true);
    assert.equal(chatAreaRules.some(function (rule) {
      return /min-height\s*:\s*0/.test(rule.body);
    }), true);
    assert.equal(chatAreaRules.some(function (rule) {
      return /overflow\s*:\s*hidden/.test(rule.body);
    }), true);
    const input = cssDeclarations(styleCss, ".chat-input-area");
    assert.equal(input["flex-shrink"], "0");
  });

  it("contains the copy-status live region inside the message box", function () {
    // Regression: the absolutely positioned .sr-only copy-status region
    // used to resolve against the initial containing block because
    // .message was not positioned.  It then escaped .chat-messages'
    // scrollport and grew the document, so focusing the composer
    // scrolled the whole page (320x568: scrollY 0 -> 51.5).
    const message = cssDeclarations(styleCss, ".message");
    assert.equal(message.position, "relative");
    // The region itself stays absolutely positioned and visually
    // hidden, so the fix cannot regress screen-reader announcements.
    const srOnly = cssDeclarations(styleCss, ".sr-only");
    assert.equal(srOnly.position, "absolute");
    assert.equal(srOnly.width, "1px");
    assert.equal(srOnly.height, "1px");
    assert.notEqual(srOnly.overflow, undefined);
    // No ancestor of the message may reintroduce the escape path by
    // clipping away the positioned box.
    const scroller = cssDeclarations(styleCss, ".chat-messages");
    assert.equal(scroller["overflow-y"], "auto");
    assert.equal(scroller.position === undefined ||
      scroller.position === "static", true);
  });

  it("appends the live region inside the assistant message wrapper", function () {
    const start = appSource.indexOf('var liveRegion = document.createElement("span");');
    assert.equal(start > -1, true);
    const scope = appSource.slice(start, start + 2600);
    assert.equal(scope.includes('liveRegion.className = "sr-only";'), true);
    assert.equal(scope.includes('liveRegion.setAttribute("aria-live", "polite")'), true);
    assert.equal(scope.includes('liveRegion.setAttribute("aria-atomic", "true")'), true);
    // It is mounted on the message wrapper (the positioned element),
    // never on the scrolling container or the document body.
    assert.equal(scope.includes("wrapper.appendChild(liveRegion);"), true);
    assert.equal(scope.includes("messagesEl.appendChild(liveRegion)"), false);
    assert.equal(scope.includes("document.body.appendChild(liveRegion)"), false);
    // The announcement text is still driven by the copy state machine,
    // including the success announcement the fix must not break.
    assert.equal(
      scope.includes('liveRegion.textContent = "Response copied to clipboard.";'),
      true);
    assert.equal(
      scope.includes('liveRegion.textContent = "Failed to copy response. Press to retry.";'),
      true);
  });

  it("keeps the review panel from inflating the message list item", function () {
    // A long transcript must not grow its flex item's hypothetical size,
    // or the review panel would be shrunk to nothing on short viewports.
    const messages = cssDeclarations(styleCss, ".chat-messages");
    assert.equal(messages.flex, "1 1 0");
    const body = cssDeclarations(styleCss, ".history-review-body");
    assert.notEqual(body["max-height"], undefined);
    // The panel asks for a bounded share of the column and may shrink.
    const panel = cssDeclarations(styleCss, ".history-review-panel");
    assert.equal(panel["flex-grow"] === undefined || panel["flex-grow"] === "0",
      true);
    assert.equal(/0 1 auto|1 1/.test(panel.flex || ""), true);
  });

  it("constrains the mobile layout for 320x568 and 390x844", function () {
    const appLayout = cssRulesFor(styleCss, ".app-layout").filter(
      function (rule) { return rule.media !== null; });
    assert.equal(appLayout.length > 0, true);
    const body = appLayout.map(function (rule) { return rule.body; }).join(";");
    assert.equal(body.includes("100vh"), true);
    assert.equal(body.includes("100dvh"), true);
    assert.equal(body.includes("flex-direction: column"), true);

    // The chat column must be able to shrink below its content (base
    // rule) and the review panel's share must stay capped on mobile.
    const chatRules = cssRulesFor(styleCss, ".chat-area");
    assert.equal(chatRules.some(function (rule) {
      return /min-height\s*:\s*0/.test(rule.body);
    }), true);
    const mobilePanel = cssRulesFor(styleCss, ".history-review-panel").filter(
      function (rule) { return rule.media !== null; });
    assert.equal(mobilePanel.length > 0, true);
    assert.equal(mobilePanel.some(function (rule) {
      return /max-height/.test(rule.body);
    }), true);
  });
});


/**
 * Extracts a top-level function body by brace matching that skips string
 * literals, template literals and comments, so source containing '"{"'
 * cannot truncate the result.
 */
function extractFunctionSource(source, signature) {
  const start = source.indexOf(signature);
  if (start === -1) return null;
  const braceStart = source.indexOf("{", start);
  if (braceStart === -1) return null;
  let depth = 0;
  let i = braceStart;
  while (i < source.length) {
    const ch = source[i];
    if (ch === "/" && source[i + 1] === "/") {
      while (i < source.length && source[i] !== "\n") i += 1;
      continue;
    }
    if (ch === "/" && source[i + 1] === "*") {
      i += 2;
      while (i < source.length &&
             !(source[i] === "*" && source[i + 1] === "/")) i += 1;
      i += 2;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      const quote = ch;
      i += 1;
      while (i < source.length) {
        if (source[i] === "\\") { i += 2; continue; }
        if (source[i] === quote) { i += 1; break; }
        i += 1;
      }
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(start, i + 1);
    }
    i += 1;
  }
  return null;
}

function stripCommonJsExport(source) {
  const marker = "if (typeof module !== \"undefined\" && module.exports) {";
  const index = source.indexOf(marker);
  const body = index === -1 ? source : source.slice(0, index);
  return body.replace('"use strict";', "");
}

describe("history review initialization behavior", function () {
  it("executes the real initializeHistoryReview body with strict scope", function () {
    const fnSource = extractFunctionSource(
      appSource, "function initializeHistoryReview() {");
    assert.notEqual(fnSource, null);

    const declaresConsent =
      appSource.includes("let historyReviewConsentDialogAdapter = null;");

    const wrapper = [
      '"use strict";',
      "let historyReviewStartDialogAdapter = null;",
      declaresConsent
        ? "let historyReviewConsentDialogAdapter = null;"
        : "",
      "let historyReviewController = null;",
      "let historyReviewInitializationError = null;",
      "let historyReviewStartConfirmer = null;",
      "let historyReviewConsentConfirmer = null;",
      "const reviewDialogContextBySession = Object.create(null);",
      "let currentSessionId = null;",
      "function findSessionInList() { return null; }",
      "function getReviewStateIfExists() { return null; }",
      "function reviewEpoch() { return 0; }",
      "function confirmWithReviewFocus() { return Promise.resolve(true); }",
      "function createCancelableDialogAdapter() {",
      "  return { cancelPending: function () {} };",
      "}",
      "function createRemoteHistoryConfirmer() {",
      "  return { confirm: function () { return Promise.resolve(true); } };",
      "}",
      "function createHistoryReviewController() {",
      "  return { marker: 'controller' };",
      "}",
      "function createHistoryReviewRequest() {}",
      "function fetchReviewSummariesRequest() {}",
      "function fetchReviewDetailRequest() {}",
      "const isValidApiTimestamp = function () { return true; };",
      "const historyReviewStartDialogEl = {};",
      "const historyReviewStartTitleEl = {};",
      "const historyReviewStartBodyEl = {};",
      "const historyReviewStartCancelBtn = {};",
      "const historyReviewStartConfirmBtn = {};",
      "const historyReviewConsentDialogEl = {};",
      "const historyReviewConsentBodyEl = {};",
      "const historyReviewConsentCancelBtn = {};",
      "const historyReviewConsentConfirmBtn = {};",
      fnSource,
      "initializeHistoryReview();",
      "return { controller: historyReviewController,",
      "         error: historyReviewInitializationError };",
    ].filter(Boolean).join("\n");

    const run = new Function(wrapper);
    const result = run();
    assert.equal(result.error, null);
    assert.deepEqual(result.controller, { marker: "controller" });
  });
});


/* ------------------------------------------------------------------ */
/* app.js runtime wiring                                              */
/* ------------------------------------------------------------------ */

const APP_FUNCTIONS = [
  "function ensureReviewState(sessionId) {",
  "function reviewTargetBusy(sessionId) {",
  "async function loadHistoryReviewDetail(sessionId, reviewId) {",
  "async function loadHistoryReviewSession(sessionId, options) {",
  "function renderHistoryReviewPanel() {",
  "async function confirmWithReviewFocus(confirmFn, targetSessionId, triggerEl) {",
  "function applyAuthoritativeReviewDetail(state, sessionId, detail) {",
  "async function startHistoryReview(kind, eventId, triggerEl) {",
];

function makeFakeElement() {
  const classes = new Set();
  const listeners = Object.create(null);
  const element = {
    hidden: false,
    textContent: "",
    value: "",
    disabled: false,
    open: false,
    isConnected: true,
    focusCount: 0,
    children: [],
    classList: {
      toggle: function (name, on) {
        if (on === true) classes.add(name);
        else classes.delete(name);
      },
    },
    replaceChildren: function () { this.children = []; },
    appendChild: function (child) { this.children.push(child); },
    setAttribute: function () {},
    focus: function () { element.focusCount += 1; },
    showModal: function () { element.open = true; },
    close: function () { element.open = false; },
    addEventListener: function (name, handler) {
      if (Object.hasOwn(listeners, name) === false) {
        listeners[name] = [];
      }
      listeners[name].push(handler);
    },
    removeEventListener: function (name, handler) {
      if (Object.hasOwn(listeners, name) === false) return;
      listeners[name] = listeners[name].filter(function (item) {
        return item === handler ? false : true;
      });
    },
    dispatchEvent: function (event) {
      const name = typeof event === "string"
        ? event
        : (event && typeof event === "object" ? event.type : "");
      const handlers = Object.hasOwn(listeners, name)
        ? listeners[name].slice()
        : [];
      for (const handler of handlers) {
        handler(event && typeof event === "object"
          ? event
          : { type: name });
      }
      return true;
    },
  };
  return element;
}

/**
 * Builds a scope holding the real app.js review functions plus the real
 * history-review.js pure helpers.  Only genuinely outside-world effects
 * (fetch, DOM elements, session list, render sinks) are stubbed, so the
 * reconciliation decisions under test are the production ones.
 */
function createAppHarness() {
  const fixtures = {
    sessions: [{ id: 1 }, { id: 2 }],
    currentSessionId: 1,
    request: null,
  };

  const elements = Object.create(null);
  for (const id of PANEL_IDS) elements[id] = makeFakeElement();
  // The "Review" label lives inside the selector wrapper.
  elements.historyReviewSelectorLabel = makeFakeElement();
  for (const id of [
    "historyReviewStartDialog", "hrsTitle", "hrsBody", "hrsCancel", "hrsStart",
  ]) {
    elements[id] = makeFakeElement();
  }

  const inner = [
    "let currentSessionId = __fixtures.currentSessionId;",
    "let currentInteractionMode = __receiveMode;",
    "const reviewStateBySession = Object.create(null);",
    "const reviewLifecycleEpochBySession = Object.create(null);",
    "const reviewDialogContextBySession = Object.create(null);",
    "let historyReviewController = null;",
    "let historyReviewInitializationError = null;",
    "let historyReviewStartConfirmer = null;",
    "let historyReviewStartDialogAdapter = null;",
    "const historyReviewPanelEl = __elements.historyReviewPanel;",
    "const historyReviewTitleEl = __elements.historyReviewTitle;",
    "const historyReviewStatusBadgeEl = __elements.historyReviewStatusBadge;",
    "const historyReviewToggleBtn = __elements.historyReviewToggleBtn;",
    "const historyReviewBodyEl = __elements.historyReviewBody;",
    "const historyReviewLiveEl = __elements.historyReviewLive;",
    "const historyReviewProposalEl = __elements.historyReviewProposal;",
    "const historyReviewProposalTextEl = __elements.historyReviewProposalText;",
    "const historyReviewDetailEl = __elements.historyReviewDetail;",
    "const historyReviewSelectorEl = __elements.historyReviewSelector;",
    "const historyReviewSelectEl = __elements.historyReviewSelect;",
    "const historyReviewSummaryEl = __elements.historyReviewSummary;",
    "const historyReviewCoverageEl = __elements.historyReviewCoverage;",
    "const historyReviewFindingsEl = __elements.historyReviewFindings;",
    "const historyReviewErrorEl = __elements.historyReviewError;",
    "const historyReviewRecheckBtn = __elements.historyReviewRecheckBtn;",
    "const historyReviewContinueBtn = __elements.historyReviewContinueBtn;",
    "const historyReviewStartBtn = __elements.historyReviewStartBtn;",
    "const historyReviewDismissBtn = __elements.historyReviewDismissBtn;",
    "const historyReviewStartDialogEl = __elements.historyReviewStartDialog;",
    "const historyReviewStartTitleEl = __elements.hrsTitle;",
    "const historyReviewStartBodyEl = __elements.hrsBody;",
    "const historyReviewStartCancelBtn = __elements.hrsCancel;",
    "const historyReviewStartConfirmBtn = __elements.hrsStart;",
    "function showStatus(text, isError) {",
    "  __fixtures.lastStatus = { text: text, isError: isError === true };",
    "}",
    "const document = {",
    "  createElement: function () { return __makeElement(); },",
    "  querySelector: function (selector) {",
    "    if (selector === 'label[for=\"historyReviewSelect\"]') {",
    "      return __elements.historyReviewSelectorLabel;",
    "    }",
    "    return null;",
    "  },",
    "};",
    "let lastRenderedModel = null;",
    "function isValidSessionIdKey(value) {",
    "  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;",
    "}",
    "function findSessionInList(sessionId) {",
    "  for (const s of __fixtures.sessions) { if (s.id === sessionId) return s; }",
    "  return null;",
    "}",
    "function currentSessionInteractionMode() { return currentInteractionMode; }",
    "function removeSessionLocally(sessionId) {",
    "  __fixtures.sessions = __fixtures.sessions.filter(function (s) {",
    "    return s.id === sessionId ? false : true;",
    "  });",
    "}",
    "function getReviewStateIfExists(sessionId) {",
    "  if (isValidSessionIdKey(sessionId) === false) return null;",
    "  const key = String(sessionId);",
    "  if (Object.hasOwn(reviewStateBySession, key) === false) return null;",
    "  return reviewStateBySession[key];",
    "}",
    "function reviewEpoch(sessionId) {",
    "  return reviewLifecycleEpochBySession[String(sessionId)] || 0;",
    "}",
    "function bumpReviewEpoch(sessionId) {",
    "  const key = String(sessionId);",
    "  reviewLifecycleEpochBySession[key] = reviewEpoch(sessionId) + 1;",
    "}",
    "function canApplyCaptured(captured, kind) {",
    "  if (findSessionInList(captured.targetSessionId) === null) return false;",
    "  const current = getReviewStateIfExists(captured.targetSessionId);",
    "  if (current === null || current !== captured.stateIdentity) return false;",
    "  if (reviewEpoch(captured.targetSessionId) !== captured.epoch) return false;",
    "  if (kind === 'batch') {",
    "    return current.sessionLoadGeneration === captured.generation;",
    "  }",
    "  if (kind === 'detail') {",
    "    return current.detailGenerationByReviewId[captured.reviewId] ===",
    "      captured.generation;",
    "  }",
    "  return false;",
    "}",
    "function fetchModeSwitchEventsRequest(sessionId) {",
    "  return __fixtures.request('events:' + sessionId);",
    "}",
    "function fetchReviewSummariesRequest(sessionId) {",
    "  return __fixtures.request('summaries:' + sessionId);",
    "}",
    "function fetchReviewDetailRequest(sessionId, reviewId) {",
    "  return __fixtures.request('detail:' + sessionId + ':' + reviewId);",
    "}",
    "function updateControlStates() { return false; }",
    "function safeReviewError(err) {",
    "  if (err && typeof err.message === 'string' && err.message) return err.message;",
    "  return 'History review request failed.';",
    "}",
    "function isExactSessionNotFound(err) {",
    "  return err !== null && typeof err === 'object' &&",
    "    err.failureKind === 'http' && err.status === 404 &&",
    "    err.code === 'history_review_session_not_found';",
    "}",
    "function clearReviewSessionState(sessionId) {",
    "  if (isValidSessionIdKey(sessionId) === false) return;",
    "  const key = String(sessionId);",
    "  bumpReviewEpoch(sessionId);",
    "  delete reviewStateBySession[key];",
    "}",
    "function renderHistoryReviewFindings(findings) {",
    "  lastRenderedFindings = findings;",
    "}",
    "let lastRenderedFindings = null;",
    stripCommonJsExport(reviewSource),
    "const historyReviewDialogCopy = HISTORY_REVIEW_DIALOG_COPY;",
    "function __buildReviewModel(state) {",
    "  return buildHistoryReviewPanelModel({",
    "    sessionId: currentSessionId, state: state,",
    "    currentMode: currentSessionInteractionMode(),",
    "    validateTimestamp: isValidApiTimestamp,",
    "    validateModeSwitchEvent: isValidModeSwitchEvent,",
    "  });",
    "}",
    "function __renderHistoryReviewPanel() {",
    "  if (currentSessionId === null) return null;",
    "  const state = getReviewStateIfExists(currentSessionId);",
    "  if (state === null) return null;",
    "  const model = __buildReviewModel(state);",
    "  lastRenderedModel = model;",
    "  if (model === null || model.visible === false) return null;",
    "  __renderReviewPanelDom(model, state);",
    "  return model;",
    "}",
    ...APP_FUNCTIONS.map(function (signature) {
      let body = extractFunctionSource(appSource, signature);
      if (body === null) throw new Error("missing app.js function " + signature);
      // The real DOM renderer is exposed under a distinct name; the shim
      // above calls it after building the model.
      body = body.split("function renderHistoryReviewPanel() {")
        .join("function __renderReviewPanelDom(model, state) {");
      body = body.split("renderHistoryReviewPanelIfCurrent(")
        .join("__renderHistoryReviewPanelIfCurrent(");
      // The shim already resolved the state and built the model, so the
      // renamed body starts after those lookups.
      body = body.replace(
        "if (currentSessionId === null) { historyReviewPanelEl.hidden = true; return; }",
        "");
      body = body.replace(
        "const state = getReviewStateIfExists(currentSessionId);", "");
      body = body.replace(
        "if (state === null) { historyReviewPanelEl.hidden = true; return; }",
        "");
      body = body.replace(
        "const model = buildHistoryReviewPanelModel({\n" +
        "      sessionId: currentSessionId,\n" +
        "      state: state,\n" +
        "      currentMode: currentSessionInteractionMode(),\n" +
        "      validateTimestamp: isValidApiTimestamp,\n" +
        "      validateModeSwitchEvent: isValidModeSwitchEvent,\n" +
        "    });", "");
      body = body.replace(
        "function __renderReviewPanelDom(model, state) {",
        "function __renderReviewPanelDom(model, state) {\n" +
        "    if (model === null || model.visible === false) return;");
      return body;
    }),
    "function __renderHistoryReviewPanelIfCurrent(sessionId) {",
    "  if (currentSessionId !== sessionId) return;",
    "  __renderHistoryReviewPanel();",
    "}",
    "return {",
    "  reviewStateBySession: reviewStateBySession,",
    "  reviewLifecycleEpochBySession: reviewLifecycleEpochBySession,",
    "  ensureReviewState: ensureReviewState,",
    "  reviewTargetBusy: reviewTargetBusy,",
    "  loadHistoryReviewSession: loadHistoryReviewSession,",
    "  loadHistoryReviewDetail: loadHistoryReviewDetail,",
    "  renderHistoryReviewPanel: __renderHistoryReviewPanel,",
    "  setRequest: function (fn) { __fixtures.request = fn; },",
    "  elements: __elements,",
    "  sessions: __fixtures.sessions,",
    "  currentSession: function () { return currentSessionId; },",,
    "  setCurrentSession: function (id) { currentSessionId = id; },",
    "  setCurrentMode: function (mode) { currentInteractionMode = mode; },",
    "  setController: function (controller) { historyReviewController = controller; },",
    "  setupStartFlow: function () {",
    "    historyReviewStartDialogAdapter = createCancelableReviewDialogAdapter({",
    "      dialog: historyReviewStartDialogEl,",
    "      bodyEl: historyReviewStartBodyEl,",
    "      cancelBtn: historyReviewStartCancelBtn,",
    "      continueBtn: historyReviewStartConfirmBtn,",
    "      titleEl: historyReviewStartTitleEl,",
    "    });",
    "    historyReviewStartConfirmer = __createRemoteHistoryConfirmer(",
    "      historyReviewStartDialogAdapter);",
    "  },",
    "  startHistoryReview: startHistoryReview,",
    "  reviewDialogContextBySession: reviewDialogContextBySession,",
    "  lastRenderedModel: function () { return lastRenderedModel; },",
    "};",
  ].join("\n");

  const run = new Function(
    "__fixtures", "__elements", "__request", "__makeElement",
    "__receiveMode", "__reviewSource", "isValidApiTimestamp",
    "isValidModeSwitchEvent", "__createRemoteHistoryConfirmer", inner,
  );

  return run(
    fixtures, elements, fixtures.request, makeFakeElement,
    RECEIVE_TEACHING_MODE, reviewSource, isValidApiTimestamp,
    isValidModeSwitchEvent, createRemoteHistoryConfirmer,
  );
}


/* ------------------------------------------------------------------ */
/* fixtures                                                           */
/* ------------------------------------------------------------------ */

function makeSummary(overrides) {
  const value = {
    id: 10,
    session_id: 1,
    mode_switch_event_id: 5,
    status: "completed",
    eligible_message_count: 2,
    source_message_count: 2,
    source_from_message_id: 1,
    source_through_message_id: 2,
    truncated: false,
    summary: "summary text",
    coverage_note: "coverage note",
    error_code: null,
    error_message: null,
    findings_count: 1,
    created_at: "2026-08-06T12:00:00",
    started_at: "2026-08-06T12:00:01",
    completed_at: "2026-08-06T12:00:02",
    updated_at: "2026-08-06T12:00:02",
  };
  return Object.assign(value, overrides || {});
}

function makeFinding(overrides) {
  const value = {
    seq: 1,
    source_message_id: 1,
    verdict: "correct",
    claim_text: "claim",
    correction_text: null,
    explanation_text: null,
  };
  return Object.assign(value, overrides || {});
}

function makeDetail(overrides) {
  const value = Object.assign(makeSummary(), {
    findings_count: 1,
    findings: [makeFinding()],
  });
  return Object.assign(value, overrides || {});
}

function makeEvent(overrides) {
  const value = {
    id: 5,
    session_id: 1,
    from_mode: RECEIVE_TEACHING_MODE,
    to_mode: CORRECTIVE_MODE,
    created_at: "2026-08-06T12:00:00",
    history_through_message_id: 2,
    history_boundary_version: "history-boundary-v1",
    reviewable_user_message_count: 2,
    review_supported: true,
  };
  return Object.assign(value, overrides || {});
}

/**
 * Request router: entries are keyed by "kind:sessionId[:reviewId]" and map
 * to a payload or a function returning one.  Anything not configured
 * rejects, so a test can never silently depend on an unplanned request.
 * The router replaces the raw request layer (one level below
 * loadHistoryReviewSession), so error shapes stay faithful to what the
 * production request helper throws.
 */
function makeRouter() {
  const table = Object.create(null);
  const log = [];
  function route(key, value) {
    table[key] = value;
  }
  function request(key) {
    log.push(key);
    if (Object.hasOwn(table, key) === false) {
      return Promise.reject(new Error("unexpected request " + key));
    }
    const entry = table[key];
    return Promise.resolve(typeof entry === "function" ? entry() : entry);
  }
  return { route: route, request: request, log: log };
}


/* ------------------------------------------------------------------ */
/* execution-path assertions                                          */
/* ------------------------------------------------------------------ */

describe("history review app.js execution path", function () {
  let harness;
  let router;

  beforeEach(function () {
    harness = createAppHarness();
    router = makeRouter();
    harness.setRequest(router.request);
    assert.equal(harness.currentSession(), 1);
    assert.equal(harness.sessions.length, 2);
  });

  /**
   * Returns the last model the real renderHistoryReviewPanel produced.
   * The harness shim mirrors production by returning null for a hidden
   * panel, so the recorded model is the observable assertion target.
   */
  function renderedModel() {
    return harness.lastRenderedModel();
  }

  it("loadHistoryReviewSession reconciles caches through the real helper", function () {
    const source = extractFunctionSource(
      appSource, "async function loadHistoryReviewSession(sessionId, options) {");
    assert.notEqual(source, null);
    assert.equal(source.includes("reconcileHistoryReviewCaches("), true);
    assert.equal(source.includes("isValidHistoryReviewSummaryList("), true);
    assert.equal(source.includes("isHistoryReviewDetailCurrent("), true);
  });

  it("loadHistoryReviewDetail re-reads the summary after the await", function () {
    const source = extractFunctionSource(
      appSource, "async function loadHistoryReviewDetail(sessionId, reviewId) {");
    assert.notEqual(source, null);
    assert.equal(source.includes("historyReviewDetailMayUpdateSummary("), true);
    assert.equal(source.includes("upsertHistoryReviewSummary("), true);
    assert.equal(source.includes("historyReviewSummaryFromDetail("), true);

    const awaitIndex = source.indexOf("await fetchReviewDetailRequest(");
    assert.equal(awaitIndex > -1, true);
    const freshRead = source.indexOf(
      "const currentSummary = Array.isArray(state.summaries)", awaitIndex);
    assert.equal(freshRead > awaitIndex, true,
      "the summary must be re-read from state after the await");
    // The request-time summary must never be reused once the response lands.
    assert.equal(source.indexOf("summary.mode_switch_event_id", awaitIndex), -1);
    assert.equal(source.indexOf("summary.updated_at", awaitIndex), -1);
  });

  it("renderHistoryReviewPanel builds its model from history-review.js", function () {
    const source = extractFunctionSource(
      appSource, "function renderHistoryReviewPanel() {");
    assert.notEqual(source, null);
    assert.equal(source.includes("buildHistoryReviewPanelModel("), true);
    assert.equal(source.includes("renderHistoryReviewFindings("), true);
  });

  it("hides the review selector and its label for a single review", async function () {
    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", [makeSummary({
      id: 10, mode_switch_event_id: 5, status: "completed", findings_count: 1,
    })]);
    router.route("detail:1:10", makeDetail({
      id: 10, mode_switch_event_id: 5, status: "completed",
    }));

    await harness.loadHistoryReviewSession(1);

    const state = harness.ensureReviewState(1);
    const select = harness.elements.historyReviewSelect;
    const wrapper = harness.elements.historyReviewSelector;
    const label = harness.elements.historyReviewSelectorLabel;
    assert.equal(select.hidden, true);
    assert.equal(wrapper.hidden, true);
    assert.equal(label.hidden, true);
    // No fake placeholder option may be used to hide the control.
    assert.equal(select.children.length, 0);
    const model = harness.lastRenderedModel();
    assert.equal(model.selectorVisible, false);
    assert.equal(model.summaries.length, 1);
  });

  it("shows a populated selector with the selection for several reviews", async function () {
    const summaries = [
      makeSummary({ id: 13, mode_switch_event_id: 8, status: "completed",
        findings_count: 1, updated_at: "2026-08-06T12:10:00" }),
      makeSummary({ id: 12, mode_switch_event_id: 7, status: "failed",
        findings_count: 0, summary: null, coverage_note: null,
        error_code: "boom", error_message: "review failed",
        completed_at: null, updated_at: "2026-08-06T12:05:00" }),
      makeSummary({ id: 11, mode_switch_event_id: 6, status: "running",
        findings_count: 0, summary: null, coverage_note: null,
        completed_at: null, updated_at: "2026-08-06T12:03:00" }),
      makeSummary({ id: 10, mode_switch_event_id: 5, status: "pending",
        findings_count: 0, summary: null, coverage_note: null,
        completed_at: null, updated_at: "2026-08-06T12:01:00" }),
    ];
    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", summaries);
    router.route("detail:1:13", makeDetail({
      id: 13, mode_switch_event_id: 8, status: "completed",
      updated_at: "2026-08-06T12:10:00",
    }));

    await harness.loadHistoryReviewSession(1);

    const state = harness.ensureReviewState(1);
    // Select the failed review so the control must reflect a non-default
    // selection rather than falling back to the first summary.
    state.selectedReviewId = 12;
    harness.renderHistoryReviewPanel();

    const select = harness.elements.historyReviewSelect;
    const wrapper = harness.elements.historyReviewSelector;
    const label = harness.elements.historyReviewSelectorLabel;
    assert.equal(wrapper.hidden, false);
    assert.equal(select.hidden, false);
    assert.equal(label.hidden, false);
    assert.equal(select.children.length, 4);
    assert.deepEqual(select.children.map(function (option) {
      return option.value;
    }), ["13", "12", "11", "10"]);
    assert.deepEqual(select.children.map(function (option) {
      return option.textContent;
    }), [
      "Review #13 · completed",
      "Review #12 · failed",
      "Review #11 · running",
      "Review #10 · pending",
    ]);
    // The current selection is reflected in the control.
    assert.equal(select.value, "12");
    const model = harness.lastRenderedModel();
    assert.equal(model.selectorVisible, true);
    assert.equal(model.selectedReviewId, 12);
    assert.equal(model.badgeText, "Failed");
  });

  it("updateControlStates consumes the shared busy predicate", function () {
    const source = extractFunctionSource(
      appSource, "function updateControlStates() {");
    assert.notEqual(source, null);
    assert.equal(source.includes("reviewTargetBusy(currentSessionId)"), true);
    assert.equal(source.includes("reviewBusy"), true);
    assert.equal(source.includes("sendBtn.disabled = blockSend;"), true);
    assert.equal(source.includes("inputEl.disabled = blockSend;"), true);
  });

  it("reviewTargetBusy delegates to the shared busy predicate", function () {
    const source = extractFunctionSource(
      appSource, "function reviewTargetBusy(sessionId) {");
    assert.notEqual(source, null);
    assert.equal(source.includes("isHistoryReviewTargetBusy("), true);
  });

  it("A. summaries with completed status clear busy despite an unselected running cache", async function () {
    const runningSummary = makeSummary({
      id: 12, mode_switch_event_id: 7, status: "running",
      findings_count: 0, updated_at: "2026-08-06T12:00:00",
    });
    const state = harness.ensureReviewState(1);
    state.detailsById[12] = makeDetail({
      id: 12, mode_switch_event_id: 7, status: "running",
      findings_count: 0, findings: [], updated_at: "2026-08-06T12:00:00",
    });
    assert.equal(harness.reviewTargetBusy(1), true);

    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", [
      makeSummary({ id: 10, mode_switch_event_id: 5, status: "completed" }),
      makeSummary({ id: 11, mode_switch_event_id: 6, status: "completed" }),
    ]);
    router.route("detail:1:10", makeDetail({
      id: 10, mode_switch_event_id: 5, status: "completed",
    }));

    await harness.loadHistoryReviewSession(1);

    assert.equal(state.summaries.length, 2);
    assert.equal(Object.hasOwn(state.detailsById, "12"), false);
    assert.equal(harness.reviewTargetBusy(1), false);
    const model = renderedModel();
    assert.equal(model.badgeText, "Completed");
    assert.equal(model.busy, false);
    assert.equal(model.selectedReviewId, 10);
  });

  it("B. the app drops a cached running detail missing from summaries", async function () {
    const state = harness.ensureReviewState(1);
    state.detailsById[11] = makeDetail({
      id: 11, mode_switch_event_id: 6, status: "running",
      findings_count: 0, findings: [],
    });
    assert.equal(harness.reviewTargetBusy(1), true);

    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", [
      makeSummary({ id: 10, mode_switch_event_id: 5, status: "completed" }),
    ]);
    router.route("detail:1:10", makeDetail({
      id: 10, mode_switch_event_id: 5, status: "completed",
    }));

    await harness.loadHistoryReviewSession(1);

    assert.equal(Object.hasOwn(state.detailsById, "11"), false);
    assert.equal(harness.reviewTargetBusy(1), false);
    assert.equal(renderedModel().busy, false);
  });

  it("C. summaries error keeps running caches fail-closed and offers Reload", async function () {
    const state = harness.ensureReviewState(1);
    state.detailsById[12] = makeDetail({
      id: 12, mode_switch_event_id: 7, status: "running",
      findings_count: 0, findings: [], updated_at: "2026-08-06T12:00:00",
    });

    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", function () {
      return Promise.reject({
        failureKind: "network", status: 0, code: null,
        message: "Network error. Please check your connection.", body: null,
      });
    });

    await harness.loadHistoryReviewSession(1);

    assert.equal(state.summariesStatus, "error");
    assert.equal(harness.reviewTargetBusy(1), true);
    const model = renderedModel();
    assert.equal(model.busy, true);
    assert.equal(model.actionKind, "reload");
    assert.equal(model.actionLabel, "Reload");
  });

  it("D. a newer completed detail promotes the running summary and clears busy", async function () {
    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", [makeSummary({
      id: 10, mode_switch_event_id: 5, status: "running",
      findings_count: 0, summary: null, coverage_note: null,
      completed_at: null, updated_at: "2026-08-06T12:00:00",
    })]);
    router.route("detail:1:10", makeDetail({
      id: 10, mode_switch_event_id: 5, status: "completed",
      summary: "final summary text", coverage_note: "full coverage",
      updated_at: "2026-08-06T12:05:00",
      completed_at: "2026-08-06T12:05:00",
    }));

    await harness.loadHistoryReviewSession(1);

    const state = harness.ensureReviewState(1);
    assert.equal(state.summaries.length, 1);
    assert.equal(state.summaries[0].status, "completed");
    assert.equal(harness.reviewTargetBusy(1), false);

    const model = renderedModel();
    assert.equal(model.busy, false);
    assert.equal(model.badgeText, "Completed");
    assert.equal(model.summaryText, "final summary text");
    assert.equal(model.coverageText, "full coverage");
    assert.equal(model.findings.length, 1);
    assert.equal(model.actionKind, "none");
  });

  it("E. an older runtime detail never blocks a newer completed summary", async function () {
    const completed = makeSummary({
      id: 10, mode_switch_event_id: 5, status: "completed",
      findings_count: 1, summary: "authoritative summary",
      updated_at: "2026-08-06T12:10:00",
    });
    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", [completed]);
    // The server still serves an older running detail for the same review.
    router.route("detail:1:10", makeDetail({
      id: 10, mode_switch_event_id: 5, status: "running",
      findings_count: 0, findings: [], summary: null, coverage_note: null,
      updated_at: "2026-08-06T12:00:00",
    }));

    await harness.loadHistoryReviewSession(1);

    const state = harness.ensureReviewState(1);
    assert.equal(state.summaries.length, 1);
    assert.equal(state.summaries[0].status, "completed");
    assert.equal(state.detailsById[10], undefined);
    assert.equal(harness.reviewTargetBusy(1), false);

    const model = renderedModel();
    assert.equal(model.badgeText, "Completed");
    assert.equal(model.busy, false);
    assert.equal(model.detail, null);
    assert.equal(model.summaryText, null);
    assert.equal(model.findings.length, 0);
    assert.equal(model.actionKind, "reload");
    assert.equal(model.actionReviewId, 10);
  });

  it("F. one session's running cache never blocks another session", async function () {
    const first = harness.ensureReviewState(1);
    first.summaries = [makeSummary({
      id: 10, mode_switch_event_id: 5, status: "running",
      findings_count: 0, updated_at: "2026-08-06T12:00:00",
    })];
    first.summariesStatus = "ready";
    first.selectedReviewId = 10;
    first.detailsById[10] = makeDetail({
      id: 10, mode_switch_event_id: 5, status: "running",
      findings_count: 0, findings: [], updated_at: "2026-08-06T12:00:00",
    });
    assert.equal(harness.reviewTargetBusy(1), true);

    const second = harness.ensureReviewState(2);
    second.detailsById[11] = makeDetail({
      id: 11, session_id: 2, mode_switch_event_id: 6, status: "running",
      findings_count: 0, findings: [],
    });

    router.route("events:2", [makeEvent({ id: 6, session_id: 2 })]);
    router.route("summaries:2", [makeSummary({
      id: 20, session_id: 2, mode_switch_event_id: 6, status: "completed",
      findings_count: 1, updated_at: "2026-08-06T12:10:00",
    })]);
    router.route("detail:2:20", makeDetail({
      id: 20, session_id: 2, mode_switch_event_id: 6, status: "completed",
      updated_at: "2026-08-06T12:10:00",
    }));

    harness.setCurrentSession(2);
    await harness.loadHistoryReviewSession(2);

    assert.equal(harness.reviewTargetBusy(2), false);
    assert.equal(harness.reviewTargetBusy(1), true);
    assert.equal(Object.hasOwn(second.detailsById, "11"), false);
    const model = renderedModel();
    assert.equal(model.badgeText, "Completed");
    assert.equal(model.busy, false);
  });

  it("ignores a late detail response whose generation was superseded", async function () {
    const staleDetail = makeDetail({
      id: 10, mode_switch_event_id: 5, status: "running",
      findings_count: 0, findings: [], updated_at: "2026-08-06T12:00:00",
    });

    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", [makeSummary({
      id: 10, mode_switch_event_id: 5, status: "running",
      findings_count: 0, summary: null, coverage_note: null,
      completed_at: null, updated_at: "2026-08-06T12:00:00",
    })]);
    const pending = harness.loadHistoryReviewSession(1);

    // A newer load supersedes the in-flight detail request before it lands.
    const state = harness.ensureReviewState(1);
    state.detailGenerationByReviewId[10] =
      (state.detailGenerationByReviewId[10] || 0) + 1;

    await pending;

    assert.equal(Object.hasOwn(state.detailsById, "10"), false);
    assert.equal(state.summaries[0].status, "running");
    assert.equal(isValidHistoryReviewSummaryList(
      state.summaries, 1, isValidApiTimestamp), true);
  });

  it("keeps a trusted cached detail when the summaries reload fails", async function () {
    const trusted = makeDetail({
      id: 10, mode_switch_event_id: 5, status: "completed",
      updated_at: "2026-08-06T12:10:00",
    });
    const state = harness.ensureReviewState(1);
    state.summaries = [makeSummary({
      id: 10, mode_switch_event_id: 5, status: "completed",
      findings_count: 1, updated_at: "2026-08-06T12:10:00",
    })];
    state.summariesStatus = "ready";
    state.selectedReviewId = 10;
    state.detailsById[10] = trusted;

    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", function () {
      return Promise.reject({
        failureKind: "http", status: 503, code: null,
        message: "Service unavailable.", body: null,
      });
    });

    await harness.loadHistoryReviewSession(1);

    assert.equal(state.summariesStatus, "error");
    assert.equal(state.detailsById[10], trusted);
    assert.equal(harness.reviewTargetBusy(1), false);
    const model = renderedModel();
    assert.equal(model.badgeText, "Completed");
    assert.equal(model.summaryText, "summary text");
    assert.equal(model.actionKind, "reload");
  });

  it("reconcile helper agrees with the app-level busy predicate", function () {
    const completedSummary = makeSummary({
      id: 10, mode_switch_event_id: 5, status: "completed",
      findings_count: 1, updated_at: "2026-08-06T12:10:00",
    });
    const staleRunning = makeDetail({
      id: 10, mode_switch_event_id: 5, status: "running",
      findings_count: 0, findings: [], updated_at: "2026-08-06T12:00:00",
    });
    const reconciled = reconcileHistoryReviewCaches(
      { "10": staleRunning }, [completedSummary], 1, isValidApiTimestamp);
    assert.notEqual(reconciled, null);
    assert.deepEqual(reconciled.staleReviewIds, [10]);
    assert.equal(
      isHistoryReviewTargetBusy(1, {
        operation: null, uncertainEventId: null,
        summaries: reconciled.summaries,
        summariesStatus: "ready",
        detailsById: reconciled.detailsById,
      }, isValidApiTimestamp),
      false,
    );
    assert.equal(
      isHistoryReviewDetailCurrent(staleRunning, completedSummary, 1, isValidApiTimestamp),
      false,
    );
    assert.equal(
      isValidHistoryReviewDetail(staleRunning, 1, 10, 5, isValidApiTimestamp),
      true,
    );
    assert.equal(
      buildHistoryReviewPanelModel({
        sessionId: 1,
        state: {
          events: [], eventsStatus: "ready", summaries: [completedSummary],
          summariesStatus: "ready", detailsById: reconciled.detailsById,
          detailErrorByReviewId: Object.create(null),
          selectedReviewId: 10, uncertainEventId: null, operation: null,
        },
        currentMode: CORRECTIVE_MODE,
        validateTimestamp: isValidApiTimestamp,
        validateModeSwitchEvent: isValidModeSwitchEvent,
      }).badgeText,
      "Completed",
    );
    // The same reconciled state, rendered through the real app.js panel
    // function, must produce a compatible model.
    const state = harness.ensureReviewState(1);
    state.events = [];
    state.eventsStatus = "ready";
    state.summaries = reconciled.summaries;
    state.summariesStatus = "ready";
    state.detailsById = reconciled.detailsById;
    state.selectedReviewId = 10;
    const rendered = harness.renderHistoryReviewPanel();
    assert.notEqual(rendered, null);
    assert.equal(rendered.badgeText, "Completed");
    assert.equal(rendered.detail, null);
    assert.equal(rendered.summaryText, null);
    assert.equal(rendered.actionKind, "reload");
    assert.equal(rendered.actionReviewId, 10);
    assert.equal(harness.reviewTargetBusy(1), false);
    assert.equal(
      upsertHistoryReviewSummary(
        [completedSummary],
        historyReviewSummaryFromDetail(
          makeDetail({ id: 10, mode_switch_event_id: 5 }), 1, isValidApiTimestamp),
        1, isValidApiTimestamp).length,
      1,
    );
  });

  function flushAsync() {
    return new Promise(function (resolve) { setImmediate(resolve); });
  }

  function seedStartProposal() {
    harness.setCurrentMode(CORRECTIVE_MODE);
    const state = harness.ensureReviewState(1);
    state.events = [makeEvent()];
    state.eventsStatus = "ready";
    state.eventsError = null;
    state.summaries = [];
    state.summariesStatus = "ready";
    state.summariesError = null;
    state.detailsById = Object.create(null);
    state.selectedReviewId = null;
    state.operation = null;
    state.panelError = null;
    harness.renderHistoryReviewPanel();
    return state;
  }

  function makeStartController(postReview) {
    return historyReview.createHistoryReviewController({
      postReview: postReview || async function () {
        throw new Error("unexpected review POST");
      },
      fetchReviewSummaries: function (sessionId) {
        return router.request("summaries:" + sessionId);
      },
      fetchReviewDetail: function (sessionId, reviewId) {
        return router.request("detail:" + sessionId + ":" + reviewId);
      },
      confirmRemoteHistory: async function () { return false; },
      validateTimestamp: isValidApiTimestamp,
    });
  }

  function completedPayload() {
    return makeDetail({
      id: 10, session_id: 1, mode_switch_event_id: 5,
      status: "completed", summary: "final summary text",
      coverage_note: "full coverage",
      updated_at: "2026-08-06T12:05:00",
      completed_at: "2026-08-06T12:05:00",
    });
  }

  function routeCompletedPayload(payload) {
    router.route("events:1", [makeEvent()]);
    router.route("summaries:1", [makeSummary({
      id: payload.id, session_id: payload.session_id,
      mode_switch_event_id: payload.mode_switch_event_id,
      status: payload.status, findings_count: payload.findings_count,
      summary: payload.summary, coverage_note: payload.coverage_note,
      updated_at: payload.updated_at, completed_at: payload.completed_at,
    })]);
    router.route("detail:1:" + payload.id, payload);
  }

  it("keeps a busy reservation without Working or POST while the start dialog is open", async function () {
    const state = seedStartProposal();
    harness.setupStartFlow();
    let posts = 0;
    harness.setController(makeStartController(async function () {
      posts += 1;
      throw new Error("POST must not run before confirmation");
    }));

    const pending = harness.startHistoryReview(
      "start", 5, harness.elements.historyReviewStartBtn);

    assert.equal(state.operation.phase, "confirming_start");
    assert.equal(harness.reviewTargetBusy(1), true);
    const model = renderedModel();
    assert.equal(model.busy, true);
    assert.equal(model.workingVisible, false);
    assert.equal(model.workingLiveText, "");
    assert.equal(model.badgeText, "Review available");
    assert.notEqual(model.proposal, null);
    assert.equal(model.startVisible, false);
    assert.equal(harness.elements.historyReviewStatusBadge.textContent,
      "Review available");
    assert.equal(
      harness.elements.historyReviewLive.textContent.includes(
        "Working on history review..."),
      false,
    );
    assert.equal(harness.elements.historyReviewLive.textContent, "");
    assert.equal(harness.elements.historyReviewProposal.hidden, false);
    assert.equal(posts, 0);

    harness.elements.hrsCancel.dispatchEvent({ type: "click" });
    await flushAsync();
    await pending;
    assert.equal(state.operation, null);
    assert.equal(posts, 0);
  });

  it("shows Working and posts exactly once after Start review is clicked", async function () {
    const state = seedStartProposal();
    harness.setupStartFlow();
    const payload = completedPayload();
    let posts = 0;
    let resolvePost = null;

    harness.setController(makeStartController(function (sessionId, body) {
      posts += 1;
      assert.equal(sessionId, 1);
      assert.deepEqual(body, {
        mode_switch_event_id: 5,
        acknowledge_remote_history: false,
      });
      return new Promise(function (resolve) { resolvePost = resolve; });
    }));

    const pending = harness.startHistoryReview(
      "start", 5, harness.elements.historyReviewStartBtn);
    assert.equal(state.operation.phase, "confirming_start");
    assert.equal(posts, 0);

    harness.elements.hrsStart.dispatchEvent({ type: "click" });
    await flushAsync();

    assert.equal(state.operation.phase, "posting");
    assert.equal(renderedModel().badgeText, "Working");
    assert.equal(renderedModel().workingVisible, true);
    assert.equal(harness.elements.historyReviewStatusBadge.textContent,
      "Working");
    assert.equal(harness.elements.historyReviewLive.textContent,
      "Working on history review...");
    assert.equal(posts, 1);

    routeCompletedPayload(payload);
    resolvePost(payload);
    await flushAsync();
    await pending;

    assert.equal(posts, 1);
    assert.equal(state.operation, null);
    assert.equal(harness.reviewTargetBusy(1), false);
  });

  it("cancels without POST, restores focus, keeps the proposal, and allows a restart", async function () {
    const state = seedStartProposal();
    harness.setupStartFlow();
    let posts = 0;
    harness.setController(makeStartController(async function () {
      posts += 1;
      throw new Error("POST must not run on cancel");
    }));
    const startBtn = harness.elements.historyReviewStartBtn;

    const first = harness.startHistoryReview("start", 5, startBtn);
    assert.equal(state.operation.phase, "confirming_start");
    assert.equal(startBtn.focusCount, 0);

    harness.elements.hrsCancel.dispatchEvent({ type: "click" });
    await flushAsync();
    await first;

    assert.equal(posts, 0);
    assert.equal(state.operation, null);
    assert.equal(startBtn.focusCount, 1);
    assert.equal(harness.elements.historyReviewStatusBadge.textContent,
      "Review available");
    const model = renderedModel();
    assert.equal(model.badgeText, "Review available");
    assert.equal(model.workingVisible, false);
    assert.notEqual(model.proposal, null);
    assert.equal(model.startVisible, true);

    const second = harness.startHistoryReview("start", 5, startBtn);
    assert.equal(state.operation.phase, "confirming_start");
    harness.elements.hrsCancel.dispatchEvent({ type: "click" });
    await flushAsync();
    await second;
    assert.equal(state.operation, null);
    assert.equal(posts, 0);
    assert.equal(startBtn.focusCount, 2);
  });

  it("does not double-fire the review POST on repeated start actions or Start clicks", async function () {
    const state = seedStartProposal();
    harness.setupStartFlow();
    const payload = completedPayload();
    let posts = 0;
    let resolvePost = null;
    harness.setController(makeStartController(function () {
      posts += 1;
      return new Promise(function (resolve) { resolvePost = resolve; });
    }));

    const first = harness.startHistoryReview(
      "start", 5, harness.elements.historyReviewStartBtn);
    assert.equal(state.operation.phase, "confirming_start");

    await harness.startHistoryReview(
      "start", 5, harness.elements.historyReviewStartBtn);
    assert.equal(posts, 0);
    assert.equal(state.operation.phase, "confirming_start");

    harness.elements.hrsStart.dispatchEvent({ type: "click" });
    harness.elements.hrsStart.dispatchEvent({ type: "click" });
    await flushAsync();
    assert.equal(state.operation.phase, "posting");
    assert.equal(posts, 1);

    await harness.startHistoryReview(
      "start", 5, harness.elements.historyReviewStartBtn);
    assert.equal(posts, 1);

    routeCompletedPayload(payload);
    resolvePost(payload);
    await flushAsync();
    await first;
    assert.equal(posts, 1);
    assert.equal(state.operation, null);
  });

  it("keeps session A's confirmation reservation out of session B", async function () {
    const stateA = seedStartProposal();
    harness.setupStartFlow();
    let posts = 0;
    harness.setController(makeStartController(async function () {
      posts += 1;
      throw new Error("POST must not run for a cancelled confirmation");
    }));

    const pendingA = harness.startHistoryReview(
      "start", 5, harness.elements.historyReviewStartBtn);
    assert.equal(stateA.operation.phase, "confirming_start");
    assert.equal(Object.hasOwn(harness.reviewDialogContextBySession, "1"), true);
    assert.equal(Object.hasOwn(harness.reviewDialogContextBySession, "2"), false);

    harness.setCurrentSession(2);
    const summaryB = makeSummary({
      id: 20, session_id: 2, mode_switch_event_id: 6,
      status: "completed", findings_count: 1,
      updated_at: "2026-08-06T12:06:00",
    });
    const detailB = makeDetail({
      id: 20, session_id: 2, mode_switch_event_id: 6,
      status: "completed", updated_at: "2026-08-06T12:06:00",
    });
    const stateB = harness.ensureReviewState(2);
    stateB.events = [];
    stateB.eventsStatus = "ready";
    stateB.summaries = [summaryB];
    stateB.summariesStatus = "ready";
    stateB.detailsById["20"] = detailB;
    stateB.selectedReviewId = 20;
    stateB.operation = null;
    harness.setCurrentMode(CORRECTIVE_MODE);

    const modelB = harness.renderHistoryReviewPanel();
    assert.equal(modelB.badgeText, "Completed");
    assert.equal(harness.elements.historyReviewStatusBadge.textContent,
      "Completed");
    assert.equal(modelB.workingVisible, false);
    assert.equal(modelB.busy, false);
    assert.equal(harness.reviewTargetBusy(2), false);
    assert.equal(harness.reviewTargetBusy(1), true);
    assert.equal(stateA.operation.phase, "confirming_start");
    assert.equal(Object.hasOwn(harness.reviewDialogContextBySession, "2"), false);

    harness.elements.hrsCancel.dispatchEvent({ type: "click" });
    await flushAsync();
    await pendingA;

    assert.equal(posts, 0);
    assert.equal(stateA.operation, null);
    assert.equal(stateB.operation, null);
    assert.equal(harness.reviewTargetBusy(2), false);
    assert.equal(harness.reviewDialogContextBySession["1"], undefined);
  });
});
