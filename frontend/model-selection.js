"use strict";

/**
 * Pure helpers for the LLM model selector.
 *
 * Every function here is pure — no DOM, no fetch, no localStorage —
 * and never throws on unexpected input.  ``node:test`` verifies them
 * directly; ``app.js`` calls them after ``model-selection.js`` loads.
 *
 * The registry is fail-closed: any malformed response makes the whole
 * registry invalid, and nothing silently falls back to the first
 * profile.
 */

/**
 * Analyse the profile list returned by ``GET /api/llm/profiles``.
 *
 * Handles any JavaScript value without throwing.  A valid element is a
 * non-null object with:
 *
 * - ``id``          — non-empty string after trim
 * - ``label``       — non-empty string after trim
 * - ``kind``        — ``"api"`` | ``"local"`` | ``"fake"``
 * - ``model``       — non-empty string after trim
 * - ``is_default``  — boolean
 *
 * Extra fields are ignored.  One malformed element invalidates the
 * whole registry (nothing is silently dropped).  The input array and
 * its elements are never modified.
 *
 * @param {*} value
 *   The parsed response body (or any other value).
 * @returns {{
 *   status: "valid" | "empty" | "invalid_response"
 *         | "duplicate_ids" | "missing_default" | "multiple_defaults",
 *   profiles: Array,
 *   defaultProfileId: (string|null)
 * }}
 *   ``status`` is ``"valid"`` only when every element is well-formed,
 *   ids are unique, and exactly one profile has ``is_default === true``.
 *   ``defaultProfileId`` is non-null only for ``"valid"``.
 */
function analyzeProfileRegistry(value) {
  var emptyResult = { status: "empty", profiles: [], defaultProfileId: null };

  if (!Array.isArray(value)) {
    return {
      status: "invalid_response",
      profiles: [],
      defaultProfileId: null,
    };
  }
  if (value.length === 0) {
    return emptyResult;
  }

  var i;
  for (i = 0; i < value.length; i++) {
    var p = value[i];
    if (p === null || typeof p !== "object" || Array.isArray(p)) {
      return {
        status: "invalid_response",
        profiles: [],
        defaultProfileId: null,
      };
    }
    if (typeof p.id !== "string" || p.id.trim() === "") {
      return {
        status: "invalid_response",
        profiles: [],
        defaultProfileId: null,
      };
    }
    if (typeof p.label !== "string" || p.label.trim() === "") {
      return {
        status: "invalid_response",
        profiles: [],
        defaultProfileId: null,
      };
    }
    if (typeof p.kind !== "string" ||
        (p.kind !== "api" && p.kind !== "local" && p.kind !== "fake")) {
      return {
        status: "invalid_response",
        profiles: [],
        defaultProfileId: null,
      };
    }
    if (typeof p.model !== "string" || p.model.trim() === "") {
      return {
        status: "invalid_response",
        profiles: [],
        defaultProfileId: null,
      };
    }
    if (typeof p.is_default !== "boolean") {
      return {
        status: "invalid_response",
        profiles: [],
        defaultProfileId: null,
      };
    }
  }

  // -- duplicate ids -----------------------------------------------------
  // A Set is immune to prototype-chain keys like "__proto__".
  var seen = new Set();
  for (i = 0; i < value.length; i++) {
    if (seen.has(value[i].id)) {
      return {
        status: "duplicate_ids",
        profiles: [],
        defaultProfileId: null,
      };
    }
    seen.add(value[i].id);
  }

  // -- exactly one default ------------------------------------------------
  var defaults = [];
  for (i = 0; i < value.length; i++) {
    if (value[i].is_default === true) {
      defaults.push(value[i]);
    }
  }
  if (defaults.length === 0) {
    return {
      status: "missing_default",
      profiles: [],
      defaultProfileId: null,
    };
  }
  if (defaults.length > 1) {
    return {
      status: "multiple_defaults",
      profiles: [],
      defaultProfileId: null,
    };
  }

  return {
    status: "valid",
    profiles: value,
    defaultProfileId: defaults[0].id,
  };
}

/**
 * Resolve the selected profile id against a registry result.
 *
 * Fail-closed: a registry that is not ``"valid"`` yields ``null`` —
 * the first profile is never used as a silent fallback.
 *
 * @param {object} registry
 *   The result of ``analyzeProfileRegistry`` (any value tolerated).
 * @param {string|null} current
 *   The previously selected profile id.
 * @returns {string|null}
 */
function resolveSelectedProfileId(registry, current) {
  if (!registry || typeof registry !== "object" ||
      registry.status !== "valid" || !Array.isArray(registry.profiles)) {
    return null;
  }

  if (typeof current === "string" && current !== "") {
    for (var i = 0; i < registry.profiles.length; i++) {
      if (registry.profiles[i].id === current) {
        return current;
      }
    }
  }

  return typeof registry.defaultProfileId === "string"
    ? registry.defaultProfileId
    : null;
}

/**
 * Whether the given session can accept new messages.
 *
 * A temporary send block (set immediately after a 409/503 response)
 * always wins over a stale ``"ready"`` status.
 *
 * @param {object|null|undefined} session
 *   The current session (a ``SessionResponse``) or null when none.
 * @param {string|null|undefined} block
 *   The temporary block value for this session
 *   (``"conflict"`` / ``"profile_unavailable"``) or null.
 * @returns {boolean}
 *   ``true`` when there is no session (typing triggers auto-create) or
 *   when the session is ``ready`` and not blocked.  Unknown statuses
 *   fail closed (``false``).
 */
function isSessionWritable(session, block) {
  if (session === null || session === undefined) {
    return true;
  }
  if (block) {
    return false;
  }
  return session.llm_profile_status === "ready";
}

/**
 * Long-lived read-only explanation for a session status.
 *
 * @param {string} status
 *   ``llm_profile_status`` from a SessionResponse (any value tolerated).
 * @returns {string}
 *   ``""`` for ``"ready"``; a specific message for the three known
 *   non-ready statuses; a generic read-only message for anything else.
 */
function readOnlyExplanation(status) {
  switch (status) {
    case "ready":
      return "";
    case "profile_unavailable":
      return "This conversation's model is no longer available on the " +
        "server. You can read the history but cannot send new messages. " +
        "Choose a model and start a new chat to continue.";
    case "model_changed":
      return "This conversation's model configuration has changed since " +
        "it was created. You can read the history but cannot send new " +
        "messages. Choose a model and start a new chat to continue.";
    case "legacy_unknown":
      return "This conversation was created before model tracking was " +
        "added. You can read the history but cannot send new messages. " +
        "Choose a model and start a new chat to continue.";
    default:
      return "This conversation cannot accept new messages. You can read " +
        "the history. Choose a model and start a new chat to continue.";
  }
}

/**
 * Explanation for a temporary send block (set after 409/503).
 *
 * @param {string|null|undefined} block
 *   ``"conflict"`` or ``"profile_unavailable"``.
 * @returns {string|null}
 *   The explanation, or ``null`` when the block value is unknown —
 *   callers then fall back to the session's server status.
 */
function temporaryBlockExplanation(block) {
  if (block === "profile_unavailable") {
    return readOnlyExplanation("profile_unavailable");
  }
  if (block === "conflict") {
    return "This conversation cannot accept new messages due to a model " +
      "compatibility issue. You can read the history. Choose a model and " +
      "start a new chat to continue.";
  }
  return null;
}

/**
 * Persistent text for the selector status area.
 *
 * @param {string} status
 *   Registry status from ``analyzeProfileRegistry``.
 * @param {string|null|undefined} loadError
 *   The last profile load error message, if any.
 * @returns {string}
 *   ``""`` when the registry is valid and there is no load error.
 */
function profileRegistryStatusText(status, loadError) {
  if (typeof loadError === "string" && loadError !== "") {
    return "Models unavailable. " + loadError;
  }
  switch (status) {
    case "valid":
      return "";
    case "empty":
      return "No models available on the server.";
    case "invalid_response":
      return "The server returned an invalid model list.";
    case "duplicate_ids":
      return "The model list is invalid (duplicate model IDs).";
    case "missing_default":
      return "The model list is invalid (no default model).";
    case "multiple_defaults":
      return "The model list is invalid (multiple default models).";
    default:
      return "The model list is invalid.";
  }
}

/**
 * Build the JSON body for ``POST /api/sessions``.
 *
 * The caller must have already confirmed *profileId* is valid — this
 * function performs no fallback.
 *
 * @param {string} profileId
 * @returns {{llm_profile_id: string}}
 */
function buildCreateSessionPayload(profileId) {
  return { llm_profile_id: profileId };
}

/**
 * Short badge text for a profile kind.
 *
 * @param {string} kind
 * @returns {string|null}
 *   ``"API"``, ``"Local"``, ``"Fake"``, or ``null`` for unknown kinds
 *   (callers hide the badge instead of guessing).
 */
function profileKindBadgeText(kind) {
  switch (kind) {
    case "api":
      return "API";
    case "local":
      return "Local";
    case "fake":
      return "Fake";
    default:
      return null;
  }
}

/**
 * The four authoritative ``llm_profile_status`` values, matching
 * ``SessionProfileStatus`` on the backend.
 */
var VALID_SESSION_STATUSES = [
  "ready",
  "profile_unavailable",
  "model_changed",
  "legacy_unknown",
];

/**
 * Whether *value* is a parseable API timestamp string.
 *
 * Backend timestamps are UTC but may lack a timezone suffix; this
 * applies the same normalisation rule as ``parseApiDate`` in app.js.
 *
 * @param {*} value
 * @returns {boolean}
 */
function isValidApiTimestamp(value) {
  if (typeof value !== "string" || value === "") {
    return false;
  }
  var hasTimezone =
    /Z$/i.test(value) ||
    /[+-]\d{2}:\d{2}$/.test(value);
  var date = new Date(hasTimezone ? value : value + "Z");
  return !Number.isNaN(date.getTime());
}

/**
 * Whether *value* is a complete, structurally valid SessionResponse.
 *
 * Validates every field of the backend ``SessionResponse`` schema:
 * id (positive safe integer), title, created_at / updated_at
 * (parseable timestamps), llm_profile_id, llm_profile_label,
 * llm_profile_status (one of the four authoritative values) and
 * llm_model_snapshot (string or null).  Extra fields are ignored.
 * Never throws.
 *
 * @param {*} value
 * @returns {boolean}
 */
function isValidSessionResponse(value) {
  if (value === null || typeof value !== "object" ||
      Array.isArray(value)) {
    return false;
  }
  if (!Number.isSafeInteger(value.id) || value.id < 1) {
    return false;
  }
  if (typeof value.title !== "string") {
    return false;
  }
  if (!isValidApiTimestamp(value.created_at)) {
    return false;
  }
  if (!isValidApiTimestamp(value.updated_at)) {
    return false;
  }
  if (typeof value.llm_profile_id !== "string" ||
      value.llm_profile_id === "") {
    return false;
  }
  if (typeof value.llm_profile_label !== "string") {
    return false;
  }
  if (VALID_SESSION_STATUSES.indexOf(value.llm_profile_status) === -1) {
    return false;
  }
  if (value.llm_model_snapshot !== null &&
      typeof value.llm_model_snapshot !== "string") {
    return false;
  }
  return true;
}

/**
 * Whether *value* is a complete session list: an array whose every
 * element passes ``isValidSessionResponse`` and whose session ids are
 * unique.  Any malformed element or duplicate id fails the whole
 * list.  Never throws; never mutates the input.
 *
 * @param {*} value
 * @returns {boolean}
 */
function isValidSessionList(value) {
  if (!Array.isArray(value)) {
    return false;
  }
  var seenIds = new Set();
  for (var i = 0; i < value.length; i++) {
    if (!isValidSessionResponse(value[i])) {
      return false;
    }
    if (seenIds.has(value[i].id)) {
      return false;
    }
    seenIds.add(value[i].id);
  }
  return true;
}

/**
 * Whether *value* is a complete, structurally valid MessageResponse.
 *
 * Validates id and session_id (positive safe integers), role
 * ("user" | "assistant"), content (string) and created_at (parseable
 * timestamp).  Never throws.
 *
 * @param {*} value
 * @returns {boolean}
 */
function isValidMessageResponse(value) {
  if (value === null || typeof value !== "object" ||
      Array.isArray(value)) {
    return false;
  }
  if (!Number.isSafeInteger(value.id) || value.id < 1) {
    return false;
  }
  if (!Number.isSafeInteger(value.session_id) || value.session_id < 1) {
    return false;
  }
  if (value.role !== "user" && value.role !== "assistant") {
    return false;
  }
  if (typeof value.content !== "string") {
    return false;
  }
  if (!isValidApiTimestamp(value.created_at)) {
    return false;
  }
  return true;
}

/**
 * Whether *value* is a complete, ordered message list for
 * *expectedSessionId*.
 *
 * - *expectedSessionId* must be a positive safe integer.
 * - *value* must be an array; the empty array is valid.
 * - Every element must pass ``isValidMessageResponse``.
 * - Every message's ``session_id`` must equal *expectedSessionId*.
 * - Message ids must be unique and strictly increasing (the backend
 *   returns messages ordered by id ascending, and the network-recovery
 *   logic relies on that order).
 *
 * Never throws; never mutates the input.
 *
 * @param {*} value
 * @param {number} expectedSessionId
 * @returns {boolean}
 */
function isValidMessageList(value, expectedSessionId) {
  if (!Number.isSafeInteger(expectedSessionId) || expectedSessionId < 1) {
    return false;
  }

  if (!Array.isArray(value)) {
    return false;
  }

  var seenIds = new Set();
  var previousId = 0;

  for (var i = 0; i < value.length; i++) {
    var message = value[i];

    if (!isValidMessageResponse(message)) {
      return false;
    }
    if (message.session_id !== expectedSessionId) {
      return false;
    }
    if (seenIds.has(message.id)) {
      return false;
    }
    if (message.id <= previousId) {
      return false;
    }

    seenIds.add(message.id);
    previousId = message.id;
  }

  return true;
}

/**
 * Whether *value* is a valid SendMessageResponse for *expectedSessionId*.
 *
 * The expected id must itself be a positive safe integer, otherwise
 * this returns false.  Both messages must be complete MessageResponses,
 * the roles must be exactly user/assistant, and both session_ids must
 * equal *expectedSessionId*.  Never throws.
 *
 * @param {*} value
 * @param {number} expectedSessionId
 * @returns {boolean}
 */
function isValidSendMessageResponse(value, expectedSessionId) {
  if (!Number.isSafeInteger(expectedSessionId) || expectedSessionId < 1) {
    return false;
  }
  if (value === null || typeof value !== "object" ||
      Array.isArray(value)) {
    return false;
  }
  if (!isValidMessageResponse(value.user_message)) {
    return false;
  }
  if (!isValidMessageResponse(value.assistant_message)) {
    return false;
  }
  if (value.user_message.role !== "user") {
    return false;
  }
  if (value.assistant_message.role !== "assistant") {
    return false;
  }
  if (value.user_message.session_id !== expectedSessionId) {
    return false;
  }
  if (value.assistant_message.session_id !== expectedSessionId) {
    return false;
  }
  return true;
}

/**
 * The kind of *profileId*, but only from a structurally trustworthy
 * registry.
 *
 * Re-analyses ``registry.profiles`` with ``analyzeProfileRegistry`` —
 * a forged "valid" registry containing malformed elements fails that
 * re-analysis and yields ``null`` (elements are never skipped to find
 * another match).  Never throws; never guesses from the id.
 *
 * @param {object|null|undefined} registry
 *   A registry result (as produced by ``analyzeProfileRegistry``).
 * @param {string} profileId
 * @returns {string|null}
 *   ``"api"`` / ``"local"`` / ``"fake"``, or ``null``.
 */
function profileKindFromRegistry(registry, profileId) {
  if (registry === null || typeof registry !== "object" ||
      Array.isArray(registry)) {
    return null;
  }
  if (registry.status !== "valid") {
    return null;
  }
  if (!Array.isArray(registry.profiles)) {
    return null;
  }
  var rechecked = analyzeProfileRegistry(registry.profiles);
  if (rechecked.status !== "valid") {
    return null;
  }
  for (var i = 0; i < rechecked.profiles.length; i++) {
    if (rechecked.profiles[i].id === profileId) {
      return rechecked.profiles[i].kind;
    }
  }
  return null;
}

/**
 * Decide the next selected session id for a freshly fetched list.
 *
 * Never throws: a non-array input yields the defensive
 * ``{selectionId: null, changed: false}`` (app.js only calls this
 * after ``isValidSessionList``, so malformed elements are merely
 * skipped here as a backstop).
 *
 * @param {*} freshSessions
 *   The validated session list.
 * @param {number|null} currentId
 * @param {boolean} preserveSelection
 * @returns {{selectionId: (number|null), changed: boolean}}
 *   ``changed`` is true when the selection must be reloaded
 *   (including switching to a new session after a previous one
 *   disappeared, and the null-selection case when a selection
 *   existed before).
 */
function resolveNextSelectionId(freshSessions, currentId, preserveSelection) {
  if (!Array.isArray(freshSessions)) {
    return { selectionId: null, changed: false };
  }
  if (freshSessions.length === 0) {
    return { selectionId: null, changed: currentId !== null };
  }

  if (preserveSelection === true && currentId !== null) {
    for (var i = 0; i < freshSessions.length; i++) {
      var s = freshSessions[i];
      if (s !== null && typeof s === "object" && !Array.isArray(s) &&
          s.id === currentId) {
        return { selectionId: currentId, changed: false };
      }
    }
  }

  var first = freshSessions[0];
  if (first === null || typeof first !== "object" || Array.isArray(first) ||
      !Number.isSafeInteger(first.id)) {
    return { selectionId: null, changed: currentId !== null };
  }
  return { selectionId: first.id, changed: first.id !== currentId };
}

/**
 * Find the session-select button for *sessionId* among *buttons*.
 *
 * Compares ``String(btn.dataset.sessionId)`` against
 * ``String(sessionId)``.  Safe for null input, non-array-like input
 * and elements without a dataset.
 *
 * @param {*} buttons
 *   An array-like collection of button elements.
 * @param {number} sessionId
 * @returns {object|null}
 */
function findSessionButton(buttons, sessionId) {
  if (sessionId === null || sessionId === undefined) {
    return null;
  }
  if (buttons === null || typeof buttons !== "object") {
    return null;
  }
  var length = buttons.length;
  if (typeof length !== "number" || !Number.isFinite(length)) {
    return null;
  }

  var target = String(sessionId);
  for (var i = 0; i < length; i++) {
    var btn = buttons[i];
    if (btn === null || typeof btn !== "object") {
      continue;
    }
    var ds = btn.dataset;
    if (ds !== null && typeof ds === "object" &&
        String(ds.sessionId) === target) {
      return btn;
    }
  }
  return null;
}

// Dual export: browser global for app.js, module.exports for node:test
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    analyzeProfileRegistry: analyzeProfileRegistry,
    resolveSelectedProfileId: resolveSelectedProfileId,
    isSessionWritable: isSessionWritable,
    readOnlyExplanation: readOnlyExplanation,
    temporaryBlockExplanation: temporaryBlockExplanation,
    profileRegistryStatusText: profileRegistryStatusText,
    buildCreateSessionPayload: buildCreateSessionPayload,
    profileKindBadgeText: profileKindBadgeText,
    isValidApiTimestamp: isValidApiTimestamp,
    isValidSessionResponse: isValidSessionResponse,
    isValidSessionList: isValidSessionList,
    isValidMessageResponse: isValidMessageResponse,
    isValidMessageList: isValidMessageList,
    isValidSendMessageResponse: isValidSendMessageResponse,
    profileKindFromRegistry: profileKindFromRegistry,
    resolveNextSelectionId: resolveNextSelectionId,
    findSessionButton: findSessionButton,
  };
}
