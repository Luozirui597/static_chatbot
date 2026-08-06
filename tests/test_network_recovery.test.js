/**
 * Tests for network-recovery.js — the ID-boundary send-failure logic.
 *
 * Run:  node --test tests/test_network_recovery.test.js
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const { findSentMessages } = require(
  path.resolve(__dirname, "..", "frontend", "network-recovery.js"),
);

describe("findSentMessages(allMessages, lastKnownId, originalText)", function () {
  // Boundary = id 5.  Messages with id <= 5 are before the send attempt.

  it("1. no new messages after the boundary", function () {
    // Messages before boundary, none after
    const all = [
      { id: 4, role: "user", content: "hello" },
      { id: 5, role: "assistant", content: "hi" },
    ];
    const result = findSentMessages(all, 5, "hello");
    assert.equal(result.userIdx, -1);
    assert.equal(result.hasAssistant, false);
    assert.equal(result.userMessageId, 0);
  });

  it("2. same text before boundary, none after", function () {
    // "hello" exists before the boundary (id 3 <= 5), but nothing after
    const all = [
      { id: 3, role: "user", content: "hello" },
      { id: 4, role: "assistant", content: "hi" },
      { id: 5, role: "user", content: "other" },
    ];
    const result = findSentMessages(all, 5, "hello");
    assert.equal(result.userIdx, -1);
    assert.equal(result.hasAssistant, false);
    assert.equal(result.userMessageId, 0);
  });

  it("3. only user message after boundary, no assistant", function () {
    const all = [
      { id: 5, role: "user", content: "old" },
      { id: 6, role: "user", content: "hello" },
    ];
    const result = findSentMessages(all, 5, "hello");
    assert.equal(result.userIdx, 0);
    assert.equal(result.hasAssistant, false);
    assert.equal(result.userMessageId, 6);
  });

  it("4. user and assistant both saved after boundary", function () {
    const all = [
      { id: 5, role: "assistant", content: "bye" },
      { id: 6, role: "user", content: "hello" },
      { id: 7, role: "assistant", content: "Hi there!" },
    ];
    const result = findSentMessages(all, 5, "hello");
    assert.equal(result.userIdx, 0);
    assert.equal(result.hasAssistant, true);
    assert.equal(result.userMessageId, 6);
  });

  it("5. same text appears both before and after boundary", function () {
    // "hello" at id=3 (before boundary) and id=7 (after boundary).
    // Only id=7 should match.
    const all = [
      { id: 3, role: "user", content: "hello" },
      { id: 4, role: "assistant", content: "first reply" },
      { id: 5, role: "user", content: "second msg" },
      { id: 6, role: "assistant", content: "second reply" },
      { id: 7, role: "user", content: "hello" },
      { id: 8, role: "assistant", content: "third reply" },
    ];
    const result = findSentMessages(all, 5, "hello");
    assert.equal(result.userIdx, 1);
    assert.equal(result.hasAssistant, true);
    assert.equal(result.userMessageId, 7);
  });
});
