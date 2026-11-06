"use strict";

/**
 * Pure helper: examine *allMessages* to determine whether a send
 * request reached the server.
 *
 * Only messages with ``id > lastKnownId`` are considered — earlier
 * messages (including identical text from a prior turn) are ignored.
 * The caller must also supply whether that boundary is *known*.  When it
 * is not known, this function deliberately refuses to associate any
 * message with the send.
 *
 * A send is associated only when exactly one new user message has the
 * original text.  Its assistant reply, if any, must occur before the next
 * user message; an assistant belonging to a later user request can never
 * be treated as this send's reply.
 *
 * Returns:
 *   {userIdx, hasAssistant, userMessageId, association, observedLastMessageId, candidateCount}
 * where *association* is "matched", "none", or "unknown".
 *
 * This function is pure — it does not touch the DOM or network, so
 * ``node:test`` can verify it directly.  It is defined as a top-level
 * function in a classic script; ``app.js`` calls
 * ``findSentMessages(…)`` directly after ``network-recovery.js``
 * loads.
 *
 * @param {Array<{id: number, role: string, content: string}>} allMessages
 * @param {number} lastKnownId
 * @param {string} originalText
 * @param {boolean} [boundaryKnown=true]
 * @returns {{userIdx: number, hasAssistant: boolean, userMessageId: number,
 *   association: string, observedLastMessageId: number, candidateCount: number}}
 */
function findSentMessages(
  allMessages,
  lastKnownId,
  originalText,
  boundaryKnown,
) {
  var unknown = {
    userIdx: -1,
    hasAssistant: false,
    userMessageId: 0,
    association: "unknown",
    observedLastMessageId: 0,
    candidateCount: 0,
  };

  if (boundaryKnown === false) return unknown;
  if (!Array.isArray(allMessages) ||
      typeof lastKnownId !== "number" || !isFinite(lastKnownId) ||
      typeof originalText !== "string") {
    return unknown;
  }

  var newMessages = [];
  for (var i = 0; i < allMessages.length; i++) {
    if (allMessages[i].id > lastKnownId) newMessages.push(allMessages[i]);
  }

  var candidates = [];
  for (var j = 0; j < newMessages.length; j++) {
    if (newMessages[j].role === "user" &&
        newMessages[j].content === originalText) {
      candidates.push(j);
    }
  }

  var observedLastMessageId = newMessages.length > 0
    ? newMessages[newMessages.length - 1].id
    : 0;

  if (candidates.length === 0) {
    return {
      userIdx: -1,
      hasAssistant: false,
      userMessageId: 0,
      association: "none",
      observedLastMessageId: observedLastMessageId,
      candidateCount: 0,
    };
  }

  // Multiple identical user messages after the frozen boundary cannot be
  // associated with this request without a request id.  Never guess.
  if (candidates.length !== 1) {
    return {
      userIdx: -1,
      hasAssistant: false,
      userMessageId: 0,
      association: "unknown",
      observedLastMessageId: observedLastMessageId,
      candidateCount: candidates.length,
    };
  }

  var userIdx = candidates[0];
  var hasAssistant = false;
  for (var k = userIdx + 1; k < newMessages.length; k++) {
    if (newMessages[k].role === "user") break;
    if (newMessages[k].role === "assistant") {
      hasAssistant = true;
      break;
    }
  }

  return {
    userIdx: userIdx,
    hasAssistant: hasAssistant,
    userMessageId: newMessages[userIdx].id,
    association: "matched",
    observedLastMessageId: observedLastMessageId,
    candidateCount: 1,
  };
}

// Dual export: browser global for app.js, module.exports for node:test
if (typeof module !== "undefined" && module.exports) {
  module.exports = { findSentMessages: findSentMessages };
}
