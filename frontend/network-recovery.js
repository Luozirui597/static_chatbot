"use strict";

/**
 * Pure helper: examine *allMessages* to determine whether a send
 * request reached the server.
 *
 * Only messages with ``id > lastKnownId`` are considered — earlier
 * messages (including identical text from a prior turn) are ignored.
 *
 * Returns {userIdx, hasAssistant, userMessageId} where:
 *
 * - *userIdx* — index within the filtered subset of the first user
 *   message matching *originalText* (or -1);
 * - *hasAssistant* — true if an assistant message follows that user
 *   message;
 * - *userMessageId* — the ``id`` field of the matched user message
 *   (0 when *userIdx* is -1).
 *
 * This function is pure — it does not touch the DOM or network, so
 * ``node:test`` can verify it directly.  It is defined as a top-level
 * function in a classic script; ``app.js`` calls
 * ``findSentMessages(…)`` directly after ``network-recovery.js``
 * loads.
 *
 * @param {Array<{id: number, role: string, content: string}>} allMessages
 *   The full message list for the session (fetched from the API).
 * @param {number} lastKnownId
 *   The highest message id known before the send attempt.  Messages
 *   with id <= this value are ignored.
 * @param {string} originalText
 *   The exact text the user tried to send.
 * @returns {{userIdx: number, hasAssistant: boolean, userMessageId: number}}
 */
function findSentMessages(allMessages, lastKnownId, originalText) {
  // Filter to messages that appeared after the send attempt
  var newMessages = [];
  for (var i = 0; i < allMessages.length; i++) {
    if (allMessages[i].id > lastKnownId) {
      newMessages.push(allMessages[i]);
    }
  }

  var userIdx = -1;
  var userMessageId = 0;
  for (var j = 0; j < newMessages.length; j++) {
    if (newMessages[j].role === "user" && newMessages[j].content === originalText) {
      userIdx = j;
      userMessageId = newMessages[j].id;
      break;
    }
  }

  var hasAssistant = false;
  if (userIdx >= 0) {
    for (var k = userIdx + 1; k < newMessages.length; k++) {
      if (newMessages[k].role === "assistant") {
        hasAssistant = true;
        break;
      }
    }
  }

  return { userIdx: userIdx, hasAssistant: hasAssistant, userMessageId: userMessageId };
}

// Dual export: browser global for app.js, module.exports for node:test
if (typeof module !== "undefined" && module.exports) {
  module.exports = { findSentMessages: findSentMessages };
}
