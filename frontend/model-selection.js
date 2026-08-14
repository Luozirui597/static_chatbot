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
 * The three message provenance snapshot fields, in canonical order.
 */
var MESSAGE_SNAPSHOT_FIELDS = [
  "llm_profile_id_snapshot",
  "llm_profile_kind_snapshot",
  "llm_model_snapshot",
];

/**
 * Analyse the provenance snapshot triple of a message.
 *
 * INTERNAL helper — not exported; ``isValidMessageResponse`` and
 * ``messageProvenanceLabel`` are the only consumers, so the snapshot
 * rules have exactly one implementation.
 *
 * Legal states are exactly two:
 *
 * - ``"legacy"`` — all three fields are explicitly ``null``
 *   (messages created before snapshot tracking);
 * - ``"tracked"`` — all three fields are valid non-blank strings
 *   (id 1-50 chars, kind one of api/local/fake, model 1-255 chars).
 *
 * Everything else — missing own property, partial nulls, mixed types,
 * whitespace-only strings, over-long values, invalid kinds, inherited
 * properties — is ``"invalid"``.  Own-property checks use
 * ``Object.prototype.hasOwnProperty`` (never ``in``, which would
 * accept prototype-chain forgeries).  Never throws; never mutates.
 *
 * @param {*} value
 * @returns {{
 *   status: "legacy"|"tracked"|"invalid",
 *   profileId: (string|null),
 *   kind: ("api"|"local"|"fake"|null),
 *   model: (string|null)
 * }}
 */
function analyzeMessageSnapshot(value) {
  var empty = {
    status: "invalid", profileId: null, kind: null, model: null,
  };
  if (value === null || typeof value !== "object" ||
      Array.isArray(value)) {
    return empty;
  }
  for (var i = 0; i < MESSAGE_SNAPSHOT_FIELDS.length; i++) {
    if (!Object.prototype.hasOwnProperty.call(value, MESSAGE_SNAPSHOT_FIELDS[i])) {
      return empty;
    }
  }

  var profileId = value.llm_profile_id_snapshot;
  var kind = value.llm_profile_kind_snapshot;
  var model = value.llm_model_snapshot;

  if (profileId === null && kind === null && model === null) {
    return { status: "legacy", profileId: null, kind: null, model: null };
  }

  if (typeof profileId !== "string" || profileId.trim() === "" ||
      profileId.length > 50) {
    return empty;
  }
  if (kind !== "api" && kind !== "local" && kind !== "fake") {
    return empty;
  }
  if (typeof model !== "string" || model.trim() === "" ||
      model.length > 255) {
    return empty;
  }
  return { status: "tracked", profileId: profileId, kind: kind, model: model };
}

/**
 * Whether *value* is a complete, structurally valid MessageResponse.
 *
 * Validates id and session_id (positive safe integers), role
 * ("user" | "assistant"), content (string), created_at (parseable
 * timestamp) and the provenance snapshot triple (via
 * ``analyzeMessageSnapshot`` — missing or partial snapshots are
 * malformed responses, not legacy messages).  Never throws.
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
  return analyzeMessageSnapshot(value).status !== "invalid";
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
 * Look up *profileId* in a structurally trustworthy registry and
 * return the profile object itself.
 *
 * INTERNAL helper — not exported.  Re-analyses ``registry.profiles``
 * with ``analyzeProfileRegistry``: a forged "valid" registry
 * containing malformed elements fails that re-analysis and yields
 * ``null`` (elements are never skipped to find another match).  Never
 * throws; never guesses from the id.
 *
 * @param {object|null|undefined} registry
 *   A registry result (as produced by ``analyzeProfileRegistry``).
 * @param {string} profileId
 * @returns {object|null}
 *   The matching profile object, or ``null``.
 */
function profileFromRegistry(registry, profileId) {
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
      return rechecked.profiles[i];
    }
  }
  return null;
}

/**
 * The kind of *profileId*, but only from a structurally trustworthy
 * registry.  Delegates to ``profileFromRegistry`` — one
 * implementation of the trust gate, no duplicated checks.
 *
 * @param {object|null|undefined} registry
 *   A registry result (as produced by ``analyzeProfileRegistry``).
 * @param {string} profileId
 * @returns {string|null}
 *   ``"api"`` / ``"local"`` / ``"fake"``, or ``null``.
 */
function profileKindFromRegistry(registry, profileId) {
  var profile = profileFromRegistry(registry, profileId);
  return profile === null ? null : profile.kind;
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

/**
 * Resolve the initial draft profile id for the current-session model
 * switcher.
 *
 * The draft and the server binding (``session.llm_profile_id``) are
 * two independent values — this function only computes the initial
 * draft.  For every legal session status, when the session's stored
 * profile id still exists in a trusted registry the draft is that id
 * (including ``profile_unavailable`` sessions whose profile
 * reappeared after a registry refresh — applying the same id then
 * repairs the snapshot).  When the id is absent the draft is ``null``
 * and the UI must show a placeholder; the default or first profile is
 * NEVER silently chosen.
 *
 * Fail-closed: invalid session or invalid registry → ``null``.
 * Never throws; never mutates the inputs.
 *
 * @param {*} session
 * @param {*} registry
 * @returns {string|null}
 */
function resolveSessionProfileDraft(session, registry) {
  if (!isValidSessionResponse(session)) {
    return null;
  }
  var currentId = session.llm_profile_id;
  if (typeof currentId !== "string" || currentId === "") {
    return null;
  }
  if (profileFromRegistry(registry, currentId) === null) {
    return null;
  }
  return currentId;
}

/**
 * Whether the Apply action of the current-session model switcher is
 * available.
 *
 * ``options`` must be a plain object (null / arrays / primitives fail
 * closed with ``false``) carrying ``session``, ``registry``,
 * ``draftProfileId`` and ``isSwitching``.  Apply is available only
 * when:
 *
 * - the session and registry are valid and the draft id belongs to
 *   the trusted registry;
 * - ``isSwitching === false`` exactly (undefined / null / truthy
 *   values are NOT treated as "not switching");
 * - the binding actually needs a change: a ready session with the
 *   same draft id is strictly idempotent (false), while
 *   model_changed / legacy_unknown / profile_unavailable sessions may
 *   apply even the same id to repair the snapshot.
 *
 * Never throws; never mutates the inputs.
 *
 * @param {*} options
 * @returns {boolean}
 */
function canApplySessionProfile(options) {
  if (options === null || typeof options !== "object" ||
      Array.isArray(options)) {
    return false;
  }
  var session = options.session;
  var registry = options.registry;
  var draftProfileId = options.draftProfileId;
  var isSwitching = options.isSwitching;

  if (!isValidSessionResponse(session)) {
    return false;
  }
  if (profileFromRegistry(registry, draftProfileId) === null) {
    return false;
  }
  if (isSwitching !== false) {
    return false;
  }

  var status = session.llm_profile_status;
  if (status === "ready") {
    return draftProfileId !== session.llm_profile_id;
  }
  if (status === "model_changed" || status === "legacy_unknown" ||
      status === "profile_unavailable") {
    return true;
  }
  return false;
}

/**
 * Build the exact JSON body for
 * ``PATCH /api/sessions/{id}/llm-profile``.
 *
 * Construction only — the caller must have already ensured
 * *profileId* belongs to a trusted registry and that *acknowledge…*
 * is boolean (mirroring ``buildCreateSessionPayload``'s contract, to
 * avoid duplicating the backend's validation rules).  The
 * acknowledge flag is normalised to a strict boolean.
 *
 * @param {string} profileId
 * @param {*} acknowledgeRemoteHistory
 * @returns {{llm_profile_id: string, acknowledge_remote_history: boolean}}
 */
function buildSwitchSessionProfilePayload(profileId, acknowledgeRemoteHistory) {
  return {
    llm_profile_id: profileId,
    acknowledge_remote_history: acknowledgeRemoteHistory === true,
  };
}

/**
 * Whether switching the current session to *targetProfileId* needs
 * the remote-history privacy confirmation BEFORE sending the PATCH.
 *
 * ``options`` is a plain object carrying ``session``, ``registry``,
 * ``targetProfileId`` and ``historyState`` (``"empty"`` |
 * ``"present"`` | ``"unknown"``).
 *
 * - Local / fake targets never need the confirmation.
 * - The SAME reliable API binding (ready session, same profile id AND
 *   same model snapshot) never needs it — Apply is disabled there
 *   anyway.
 * - A real switch (or a non-ready repair) to an API target needs it
 *   when history is present, and also when the history state is
 *   UNKNOWN: "no history" cannot be proven, so privacy fails closed.
 * - An invalid historyState for an API target also returns ``true``.
 *
 * This is a frontend HINT only — the backend's structured 409 is the
 * final authority.  The 4B caller must run ``canApplySessionProfile``
 * first, so invalid inputs block Apply instead of popping a dialog.
 *
 * @param {*} options
 * @returns {boolean}
 */
function needsRemoteHistoryConfirmation(options) {
  if (options === null || typeof options !== "object" ||
      Array.isArray(options)) {
    return true;
  }
  var session = options.session;
  var targetProfileId = options.targetProfileId;
  var registry = options.registry;
  var historyState = options.historyState;

  if (!isValidSessionResponse(session)) {
    return true;
  }
  var target = profileFromRegistry(registry, targetProfileId);
  if (target === null) {
    return true;
  }
  if (target.kind !== "api") {
    return false;
  }
  if (session.llm_profile_status === "ready" &&
      session.llm_profile_id === target.id &&
      session.llm_model_snapshot === target.model) {
    return false;
  }
  if (historyState === "empty") {
    return false;
  }
  // "present" → true, "unknown" → true, anything else → true.
  return true;
}

/**
 * Parse a structured ``remote_history_ack_required`` 409 body.
 *
 * Succeeds only when *value* is a non-array object whose ``detail``
 * is a non-array object with ``code === "remote_history_ack_required"``
 * and a non-blank string ``message``.  Extra fields on either level
 * are ignored (future metadata must not break clients).  The message
 * is returned verbatim — callers never match on English text.
 *
 * @param {*} value
 * @returns {{code: "remote_history_ack_required", message: string}|null}
 */
function parseRemoteHistoryAckRequired(value) {
  if (value === null || typeof value !== "object" ||
      Array.isArray(value)) {
    return null;
  }
  var detail = value.detail;
  if (detail === null || typeof detail !== "object" ||
      Array.isArray(detail)) {
    return null;
  }
  if (detail.code !== "remote_history_ack_required") {
    return null;
  }
  if (typeof detail.message !== "string" || detail.message.trim() === "") {
    return null;
  }
  return {
    code: "remote_history_ack_required",
    message: detail.message,
  };
}

/**
 * Plain-text provenance label for a message, based ONLY on the
 * message's own snapshot triple.
 *
 * - legacy (all three null) → ``null``;
 * - tracked → ``"API · <model>"`` / ``"Local · <model>"`` /
 *   ``"Fake · <model>"``;
 * - invalid → ``null``.
 *
 * Never depends on the current registry or session binding — a
 * deleted profile or a later switch does not change old messages'
 * labels.  The function does not know about roles or the DOM; the 4C
 * rendering layer calls it only for assistant messages.
 *
 * @param {*} message
 * @returns {string|null}
 */
function messageProvenanceLabel(message) {
  var snapshot = analyzeMessageSnapshot(message);
  if (snapshot.status !== "tracked") {
    return null;
  }
  var prefix = profileKindBadgeText(snapshot.kind);
  if (prefix === null) {
    return null;
  }
  return prefix + " · " + snapshot.model;
}

/**
 * Decide whether a PATCH switch response can be applied.
 *
 * ``options`` is a plain object carrying ``requestedSessionId``,
 * ``currentSessionId`` and ``response``.  Returns two independent
 * booleans:
 *
 * - ``validForCache`` — the response is a valid SessionResponse whose
 *   id matches the requested session, so the caller may replace that
 *   session in its cache;
 * - ``stillCurrent`` — the visible session is still the requested
 *   one, so the caller may also update the visible title/controls/
 *   focus.
 *
 * A trustworthy response for a session the user has since left yields
 * ``{true, false}`` — cache update allowed, visible UI untouched.
 * Malformed responses, id mismatches or an invalid requested id
 * yield ``{false, false}``.
 *
 * The 4B caller must additionally confirm the target still exists in
 * its sessions array before writing — this function never re-inserts
 * a deleted session.
 *
 * @param {*} options
 * @returns {{validForCache: boolean, stillCurrent: boolean}}
 */
function canApplySwitchResponse(options) {
  var fail = { validForCache: false, stillCurrent: false };
  if (options === null || typeof options !== "object" ||
      Array.isArray(options)) {
    return fail;
  }
  var requestedSessionId = options.requestedSessionId;
  var currentSessionId = options.currentSessionId;
  var response = options.response;

  if (!Number.isSafeInteger(requestedSessionId) || requestedSessionId < 1) {
    return fail;
  }
  if (!isValidSessionResponse(response) ||
      response.id !== requestedSessionId) {
    return fail;
  }

  var stillCurrent =
    Number.isSafeInteger(currentSessionId) &&
    currentSessionId >= 1 &&
    currentSessionId === requestedSessionId;

  return { validForCache: true, stillCurrent: stillCurrent };
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
    resolveSessionProfileDraft: resolveSessionProfileDraft,
    canApplySessionProfile: canApplySessionProfile,
    buildSwitchSessionProfilePayload: buildSwitchSessionProfilePayload,
    needsRemoteHistoryConfirmation: needsRemoteHistoryConfirmation,
    parseRemoteHistoryAckRequired: parseRemoteHistoryAckRequired,
    messageProvenanceLabel: messageProvenanceLabel,
    canApplySwitchResponse: canApplySwitchResponse,
  };
}
