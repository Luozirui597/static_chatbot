/**
 * Execute the real app.js send-failure handlers in a small VM harness.
 * These tests intentionally do not inspect source strings: the functions
 * under test are read from app.js and executed, with only browser I/O stubbed.
 */

const { describe, it, beforeEach } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.resolve(__dirname, "..");
const appSource = fs.readFileSync(path.join(ROOT, "frontend", "app.js"), "utf8");
const { findSentMessages } = require(
  path.join(ROOT, "frontend", "network-recovery.js"));

function extractFunction(name) {
  const marker = "async function " + name + "(";
  const start = appSource.indexOf(marker);
  if (start < 0) throw new Error("missing function " + name);
  let i = appSource.indexOf("{", start);
  let depth = 0;
  let quote = null;
  let lineComment = false;
  let blockComment = false;
  let escaped = false;
  for (; i < appSource.length; i += 1) {
    const ch = appSource[i];
    const next = appSource[i + 1];
    if (lineComment) {
      if (ch === "\n") lineComment = false;
      continue;
    }
    if (blockComment) {
      if (ch === "*" && next === "/") { blockComment = false; i += 1; }
      continue;
    }
    if (quote) {
      if (escaped) { escaped = false; continue; }
      if (ch === "\\") { escaped = true; continue; }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === "/" && next === "/") { lineComment = true; i += 1; continue; }
    if (ch === "/" && next === "*") { blockComment = true; i += 1; continue; }
    if (ch === "'" || ch === '"' || ch === "`") { quote = ch; continue; }
    if (ch === "{") depth += 1;
    if (ch === "}") {
      depth -= 1;
      if (depth === 0) return appSource.slice(start, i + 1);
    }
  }
  throw new Error("unterminated function " + name);
}

function buildHarness() {
  const context = {
    ApiError: class ApiError extends Error {
      constructor(status, message, code) {
        super(message);
        this.status = status;
        this.code = code || null;
      }
    },
    findSentMessages,
    currentSessionId: 7,
    sessionLastMessageId: {},
    sessionLastMessageBoundaryKnown: {},
    sessionSendBlocks: {},
    inputEl: { value: "draft", style: { height: "20px" } },
    lastStatus: null,
    rendered: null,
    historyStates: {},
    refreshCount: 0,
    fetchCount: 0,
    compatibilityRefreshes: [],
    fetchMode: "list",
    messages: [],
    deferredResolve: null,
    async fetchMessages() {
      context.fetchCount += 1;
      if (context.fetchMode === "throw") throw new Error("network down");
      if (context.fetchMode === "defer") {
        return new Promise(function (resolve) { context.deferredResolve = resolve; });
      }
      return context.messages;
    },
    isValidMessageList(list) { return Array.isArray(list); },
    setHistoryState(id, value) { context.historyStates[id] = value; },
    showStatus(text, isError) {
      context.lastStatus = { text: text, isError: isError === true };
    },
    clearStatus() { context.lastStatus = null; },
    renderMessages(messages) { context.rendered = messages; },
    renderEmptyChat() { context.rendered = []; },
    scrollMessagesToBottom() {},
    syncCurrentSessionUI() {},
    async refreshSessions() { context.refreshCount += 1; },
    async refreshOneSessionCompatibility(id) {
      context.compatibilityRefreshes.push(id);
    },
  };
  vm.createContext(context);
  const names = ["handleSendFailure", "handleUncertainSendOutcome", "inspectUncertainSend"];
  vm.runInContext(names.map(extractFunction).join("\n\n"), context);
  return context;
}

function reset(harness, options) {
  const opts = options || {};
  harness.currentSessionId = opts.currentSessionId || 7;
  harness.sessionLastMessageId = {};
  harness.sessionLastMessageBoundaryKnown = {};
  harness.sessionLastMessageId[7] = opts.boundary === undefined ? 5 : opts.boundary;
  harness.sessionLastMessageBoundaryKnown[7] = opts.known !== false;
  harness.inputEl.value = "draft";
  harness.inputEl.style.height = "20px";
  harness.lastStatus = null;
  harness.rendered = null;
  harness.historyStates = {};
  harness.refreshCount = 0;
  harness.fetchCount = 0;
  harness.fetchMode = opts.fetchMode || "list";
  harness.messages = opts.messages || [];
  harness.compatibilityRefreshes = [];
}

describe("real send-failure recovery handlers", function () {
  let h;
  beforeEach(function () { h = buildHarness(); });

  it("500 with only the new user message preserves the saved draft state", async function () {
    reset(h, { messages: [
      { id: 6, role: "user", content: "hello" },
      { id: 7, role: "user", content: "next" },
    ] });
    await h.handleSendFailure(new h.ApiError(500, "boom"), 7, "hello", 5, true);
    assert.equal(h.fetchCount, 1);
    assert.equal(h.historyStates[7], true);
    assert.equal(h.inputEl.value, "");
    assert.match(h.lastStatus.text, /saved/i);
  });

  it("500 with user+assistant renders success and clears only the sending draft", async function () {
    reset(h, { messages: [
      { id: 6, role: "user", content: "hello" },
      { id: 7, role: "assistant", content: "answer" },
    ] });
    await h.handleSendFailure(new h.ApiError(500, "boom"), 7, "hello", 5, true);
    assert.equal(h.fetchCount, 1);
    assert.equal(h.inputEl.value, "");
    assert.equal(h.rendered.length, 2);
  });

  it("500 query failure keeps the input and reports uncertainty", async function () {
    reset(h, { fetchMode: "throw", messages: [] });
    await h.handleSendFailure(new h.ApiError(500, "boom"), 7, "hello", 5, true);
    assert.equal(h.inputEl.value, "draft");
    assert.match(h.lastStatus.text, /uncertain/i);
  });

  it("500 with an unknown boundary refuses to fetch or clear", async function () {
    reset(h, { known: false, messages: [
      { id: 6, role: "user", content: "hello" },
      { id: 7, role: "assistant", content: "answer" },
    ] });
    await h.handleSendFailure(new h.ApiError(500, "boom"), 7, "hello", 5, false);
    assert.equal(h.fetchCount, 0);
    assert.equal(h.inputEl.value, "draft");
    assert.match(h.lastStatus.text, /uncertain/i);
  });

  it("500 with multiple identical candidates keeps the input", async function () {
    reset(h, { messages: [
      { id: 6, role: "user", content: "hello" },
      { id: 8, role: "user", content: "hello" },
    ] });
    await h.handleSendFailure(new h.ApiError(500, "boom"), 7, "hello", 5, true);
    assert.equal(h.inputEl.value, "draft");
    assert.match(h.lastStatus.text, /uncertain/i);
  });

  it("switching sessions during the recovery query protects the new draft", async function () {
    reset(h, { fetchMode: "defer", messages: [
      { id: 6, role: "user", content: "hello" },
      { id: 7, role: "assistant", content: "answer" },
    ] });
    const pending = h.handleSendFailure(
      new h.ApiError(500, "boom"), 7, "hello", 5, true);
    h.currentSessionId = 8;
    h.inputEl.value = "new session draft";
    h.deferredResolve(h.messages);
    await pending;
    assert.equal(h.inputEl.value, "new session draft");
    assert.equal(h.rendered, null);
    assert.equal(h.sessionLastMessageBoundaryKnown[7], true);
  });

  it("next user's assistant is not treated as this send's reply", async function () {
    reset(h, { messages: [
      { id: 6, role: "user", content: "hello" },
      { id: 7, role: "user", content: "later" },
      { id: 8, role: "assistant", content: "later answer" },
    ] });
    await h.handleSendFailure(new h.ApiError(500, "boom"), 7, "hello", 5, true);
    assert.equal(h.inputEl.value, "");
    assert.match(h.lastStatus.text, /saved/i);
    assert.equal(h.historyStates[7], true);
  });

  it("preserves 404/422 as confirmed-not-saved", async function () {
    for (const status of [404, 422]) {
      reset(h, { messages: [] });
      await h.handleSendFailure(new h.ApiError(status, "no"), 7, "hello", 5, true);
      assert.equal(h.fetchCount, 0);
      assert.equal(h.inputEl.value, "draft");
    }
  });

  it("preserves 409/503 compatibility branches", async function () {
    for (const status of [409, 503]) {
      reset(h, { messages: [] });
      await h.handleSendFailure(new h.ApiError(status, "conflict"), 7, "hello", 5, true);
      assert.equal(h.fetchCount, 0);
      assert.deepEqual(h.compatibilityRefreshes, [7]);
      assert.equal(h.inputEl.value, "draft");
    }
  });

  it("preserves 502/504 as saved and clears only when still current", async function () {
    reset(h, { messages: [] });
    await h.handleSendFailure(new h.ApiError(502, "bad gateway"), 7, "hello", 5, true);
    assert.equal(h.inputEl.value, "");
    assert.equal(h.historyStates[7], true);

    reset(h, { currentSessionId: 8, messages: [] });
    h.inputEl.value = "other draft";
    await h.handleSendFailure(new h.ApiError(504, "timeout"), 7, "hello", 5, true);
    assert.equal(h.inputEl.value, "other draft");
  });
  it("user_saved on a different session does not clear the current draft", async function () {
    reset(h, { currentSessionId: 8, messages: [
      { id: 6, role: "user", content: "hello" },
      { id: 7, role: "user", content: "later" },
    ] });
    h.inputEl.value = "other draft";
    await h.handleSendFailure(new h.ApiError(500, "boom"), 7, "hello", 5, true);
    assert.equal(h.inputEl.value, "other draft");
    assert.equal(h.historyStates[7], true);
    assert.equal(h.lastStatus, null);
  });

  it("network status 0 uses the same conservative inspection", async function () {
    reset(h, { messages: [
      { id: 6, role: "user", content: "hello" },
      { id: 7, role: "assistant", content: "answer" },
    ] });
    await h.handleSendFailure(new Error("network"), 7, "hello", 5, true);
    assert.equal(h.inputEl.value, "");
    assert.equal(h.rendered.length, 2);
  });

});

function buildSendHarness() {
  const h = buildHarness();
  h.isSending = false;
  h.isInitializing = false;
  h.isCreatingSession = false;
  h.isProfileSwitching = false;
  h.isDeletingSession = false;
  h.isInteractionModeSwitching = false;
  h.isTeachingStrategySwitching = false;
  h.reviewTargetBusy = function () { return false; };
  h.ensureSession = async function () { return true; };
  h.currentSessionWritable = function () { return true; };
  h.setSendingState = function (value) { h.isSending = value; };
  h.isValidSendMessageResponse = function () { return true; };
  h.sendSessionMessageRequest = function () {
    return new Promise(function (resolve) { h.deferredResolve = resolve; });
  };
  h.appendMessage = function (role, content) {
    h.appended.push({ role: role, content: content });
  };
  h.appended = [];
  h.scrollMessagesToBottom = function () {};
  vm.runInContext(extractFunction("sendMessage"), h);
  return h;
}

describe("real sendMessage success/session guard", function () {
  it("does not clear a draft typed after switching sessions", async function () {
    const h = buildSendHarness();
    h.currentSessionId = 7;
    h.sessionLastMessageId[7] = 5;
    h.sessionLastMessageBoundaryKnown[7] = true;
    h.inputEl.value = "old draft";
    h.inputEl.style.height = "20px";

    const pending = h.sendMessage("old draft");
    h.currentSessionId = 8;
    h.inputEl.value = "new session draft";
    h.inputEl.style.height = "34px";
    h.deferredResolve({
      user_message: { id: 10, role: "user", content: "old draft" },
      assistant_message: { id: 11, role: "assistant", content: "answer" },
    });
    await pending;

    assert.equal(h.inputEl.value, "new session draft");
    assert.equal(h.inputEl.style.height, "34px");
    assert.equal(h.appended.length, 0);
    assert.equal(h.sessionLastMessageId[7], 11);
    assert.equal(h.sessionLastMessageBoundaryKnown[7], true);
    assert.equal(h.refreshCount, 1);
  });

  it("still clears and renders when the sending session remains selected", async function () {
    const h = buildSendHarness();
    h.currentSessionId = 7;
    h.sessionLastMessageId[7] = 5;
    h.sessionLastMessageBoundaryKnown[7] = true;
    h.inputEl.value = "draft";
    h.inputEl.style.height = "20px";

    const pending = h.sendMessage("draft");
    h.deferredResolve({
      user_message: { id: 10, role: "user", content: "draft" },
      assistant_message: { id: 11, role: "assistant", content: "answer" },
    });
    await pending;

    assert.equal(h.inputEl.value, "");
    assert.equal(h.inputEl.style.height, "");
    assert.equal(h.appended.length, 2);
    assert.equal(h.lastStatus, null);
  });
});


describe("Reply-language saved-error message", function () {
  it("shows the structured 502 language message and clears the saved draft", async function () {
    const h = buildHarness();
    await h.handleSendFailure(
      new h.ApiError(
        502,
        "你的消息已保存。助手回复的语言不符合要求，该回复未保存。",
        "receive_reply_language_mismatch",
      ),
      7,
      "hello",
      5,
      true,
    );
    assert.equal(
      h.lastStatus.text,
      "你的消息已保存。助手回复的语言不符合要求，该回复未保存。",
    );
    assert.equal(h.inputEl.value, "");
    assert.equal(h.historyStates[7], true);
  });

  it("keeps a 422 language-conflict draft", async function () {
    const h = buildHarness();
    await h.handleSendFailure(
      new h.ApiError(
        422,
        "The request contains conflicting reply-language instructions.",
        "receive_reply_language_conflict",
      ),
      7,
      "hello",
      5,
      true,
    );
    assert.equal(h.inputEl.value, "draft");
  });
});
