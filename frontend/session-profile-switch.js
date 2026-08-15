"use strict";

/**
 * Session model-switching controller and pure planning helpers.
 *
 * Everything here is side-effect free logic with injected dependencies
 * — no DOM, no fetch, no access to app.js state, no third-party
 * dependencies.  ``node:test`` drives every path with fakes.
 *
 * Trusted-registry and SessionResponse validation are reused from
 * ``model-selection.js``: required directly in Node, used as globals
 * in the browser (the script loads after ``model-selection.js``).
 */

var _analyzeProfileRegistry;
var _isValidSessionResponse;

if (typeof module !== "undefined" && module.exports) {
  var _modelSelection = require("./model-selection.js");
  _analyzeProfileRegistry = _modelSelection.analyzeProfileRegistry;
  _isValidSessionResponse = _modelSelection.isValidSessionResponse;
} else {
  _analyzeProfileRegistry = analyzeProfileRegistry;
  _isValidSessionResponse = isValidSessionResponse;
}

var REMOTE_HISTORY_CONFIRM_TEXT =
  "切换到 API 后，最近的聊天历史将发送给远程 API 服务。";

var UNCERTAIN_REAPPLY_TEXT =
  "The previous model switch could not be confirmed. Apply again to " +
  "check the current binding before retrying.";

/* ------------------------------------------------------------------ */
/* Shared validation helpers                                          */
/* ------------------------------------------------------------------ */

var VALID_PROFILE_KINDS = ["api", "local", "fake"];

/** Profile id rule, identical to the backend schema. */
var PROFILE_ID_PATTERN = /^[a-z][a-z0-9-]*$/;

function _isValidProfileId(value) {
  if (typeof value !== "string") return false;
  if (value.length < 1 || value.length > 50) return false;
  return PROFILE_ID_PATTERN.test(value);
}

/** An already-normalised, non-empty model name (1-255 chars). */
function _isValidModel(value) {
  if (typeof value !== "string") return false;
  if (value !== value.trim()) return false;      // must be normalised
  if (value === "" || value.length > 255) return false;
  return true;
}

/** Validate a requested-profile triple (pure values only). */
function _isValidRequestedProfile(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  if (!_isValidProfileId(value.id)) return false;
  if (VALID_PROFILE_KINDS.indexOf(value.kind) === -1) return false;
  if (!_isValidModel(value.model)) return false;
  if (value.kind === "fake" && value.model !== "fake") return false;
  return true;
}

/** Original binding profile id: null (legacy) or a valid profile id. */
function _isValidOriginalProfileId(value) {
  return value === null || _isValidProfileId(value);
}

/** Original binding model snapshot: null (legacy) or a normalised
 * non-empty model string. */
function _isValidOriginalModelSnapshot(value) {
  return value === null || _isValidModel(value);
}

/** Strict validation of a switch operation. */
function _isValidOperation(operation) {
  if (operation === null || typeof operation !== "object" ||
      Array.isArray(operation)) {
    return false;
  }
  if (!Number.isSafeInteger(operation.generation) ||
      operation.generation < 1) {
    return false;
  }
  if (!Number.isSafeInteger(operation.targetSessionId) ||
      operation.targetSessionId < 1) {
    return false;
  }
  if (!_isValidRequestedProfile(operation.requestedProfile)) {
    return false;
  }
  if (typeof operation.needsConfirmHint !== "boolean") {
    return false;
  }
  if (!_isValidOriginalProfileId(operation.originalProfileId)) {
    return false;
  }
  if (!_isValidOriginalModelSnapshot(operation.originalModelSnapshot)) {
    return false;
  }
  return true;
}

/** Validation that can never throw even for getter/Proxy objects. */
function _safeIsValidOperation(operation) {
  try {
    return _isValidOperation(operation);
  } catch (_) {
    return false;
  }
}

/** Copy a requested-profile triple into fresh pure values. */
function _copyRequestedProfile(value) {
  return { id: value.id, kind: value.kind, model: value.model };
}

/* ------------------------------------------------------------------ */
/* Pure helpers                                                       */
/* ------------------------------------------------------------------ */

/**
 * Map a raw history-state value (true / false / null / undefined) to
 * the ``historyState`` enum consumed by
 * ``needsRemoteHistoryConfirmation``.
 *
 * Anything else fails closed to ``"unknown"``.  Never throws.
 *
 * @param {*} value
 * @returns {"present"|"empty"|"unknown"}
 */
function historyStateFromValue(value) {
  if (value === true) return "present";
  if (value === false) return "empty";
  return "unknown";
}

/**
 * Capture a pure-value snapshot of the requested profile from a
 * trusted registry.
 *
 * Re-validates the registry via ``analyzeProfileRegistry`` and
 * requires a strict id match.  The returned snapshot is detached from
 * the inputs — later registry mutations cannot affect it.  Never
 * throws; never mutates.
 *
 * @param {*} registry
 * @param {string} profileId
 * @returns {{id: string, kind: string, model: string}|null}
 */
function captureRequestedProfile(registry, profileId) {
  if (registry === null || typeof registry !== "object" ||
      Array.isArray(registry)) {
    return null;
  }
  if (registry.status !== "valid" || !Array.isArray(registry.profiles)) {
    return null;
  }
  var rechecked = _analyzeProfileRegistry(registry.profiles);
  if (rechecked.status !== "valid") {
    return null;
  }
  for (var i = 0; i < rechecked.profiles.length; i++) {
    if (rechecked.profiles[i].id === profileId) {
      return {
        id: rechecked.profiles[i].id,
        kind: rechecked.profiles[i].kind,
        model: rechecked.profiles[i].model,
      };
    }
  }
  return null;
}

/**
 * Keep the new-chat selection without any default fallback.
 *
 * Returns *currentId* when it still exists in a trusted registry,
 * otherwise ``null`` — the default or first profile is NEVER chosen.
 * Never throws.
 *
 * @param {*} registry
 * @param {string|null} currentId
 * @returns {string|null}
 */
function preserveSelectedProfileWithoutFallback(registry, currentId) {
  if (typeof currentId !== "string" || currentId === "") {
    return null;
  }
  if (captureRequestedProfile(registry, currentId) === null) {
    return null;
  }
  return currentId;
}

/** Parse an API timestamp to epoch millis, or null when unparseable. */
function _parseTimestamp(value) {
  if (typeof value !== "string" || value === "") return null;
  var hasTimezone =
    /Z$/i.test(value) || /[+-]\d{2}:\d{2}$/.test(value);
  var date = new Date(hasTimezone ? value : value + "Z");
  return Number.isNaN(date.getTime()) ? null : date.getTime();
}

/**
 * Plan a cache update for a fresh authoritative SessionResponse.
 *
 * ``options`` = { sessions, requestedSessionId, fresh }.
 *
 * Every cached element must be a complete valid SessionResponse and
 * session ids must be unique — malformed or duplicate lists fail
 * closed with ``"invalid_list"``.  Only an EXISTING entry with the
 * same id is replaced; a deleted session is never re-inserted (no
 * unshift, ever).  The returned array is a NEW array sorted by
 * updated_at DESC, then id DESC; ``newIndex`` is the target's index
 * in that sorted array.  (Timestamps are already validated by
 * ``isValidSessionResponse``.)  Never mutates the inputs.
 *
 * @param {*} options
 * @returns {{
 *   kind: "replace", sessions: Array, newIndex: number
 * } | {kind: "noop", reason: "id_mismatch"|"missing_target"|"invalid_list"}}
 */
function planSessionCacheUpdate(options) {
  if (options === null || typeof options !== "object" ||
      Array.isArray(options)) {
    return { kind: "noop", reason: "invalid_list" };
  }
  var sessions = options.sessions;
  var requestedSessionId = options.requestedSessionId;
  var fresh = options.fresh;

  if (!Array.isArray(sessions)) {
    return { kind: "noop", reason: "invalid_list" };
  }
  if (!Number.isSafeInteger(requestedSessionId) || requestedSessionId < 1) {
    return { kind: "noop", reason: "id_mismatch" };
  }
  if (!_isValidSessionResponse(fresh)) {
    return { kind: "noop", reason: "id_mismatch" };
  }
  if (fresh.id !== requestedSessionId) {
    return { kind: "noop", reason: "id_mismatch" };
  }

  // -- validate every cached element; ids must be unique --------------
  var seen = new Set();
  var found = false;
  for (var i = 0; i < sessions.length; i++) {
    var s = sessions[i];
    if (!_isValidSessionResponse(s)) {
      return { kind: "noop", reason: "invalid_list" };
    }
    if (seen.has(s.id)) {
      return { kind: "noop", reason: "invalid_list" };
    }
    seen.add(s.id);
    if (s.id === requestedSessionId) found = true;
  }
  if (!found) {
    return { kind: "noop", reason: "missing_target" };
  }

  // -- build the new array: replace same-id entry, keep the rest ------
  var next = [];
  for (var j = 0; j < sessions.length; j++) {
    next.push(sessions[j].id === requestedSessionId ? fresh : sessions[j]);
  }

  next.sort(function (a, b) {
    var ta = _parseTimestamp(a.updated_at);
    var tb = _parseTimestamp(b.updated_at);
    if (tb !== ta) return tb - ta;          // updated_at DESC
    return b.id - a.id;                     // id DESC on ties
  });

  var newIndex = -1;
  for (var k = 0; k < next.length; k++) {
    if (next[k].id === requestedSessionId) {
      newIndex = k;
      break;
    }
  }

  return { kind: "replace", sessions: next, newIndex: newIndex };
}

/**
 * Classify the result of an uncertain-switch refresh.
 *
 * ``options`` = { targetSessionId, uncertainRecord, fresh }.
 * ``uncertainRecord`` = { generation, requestedProfile:
 * {id,kind,model}, originalProfileId, originalModelSnapshot }.
 *
 * Every field is strictly validated — including that ``fresh`` is a
 * valid SessionResponse whose id equals *targetSessionId*.
 *
 * - ``confirmed_target`` — the TARGET session is bound to the
 *   requested profile id AND model and is ready.
 * - ``different_binding`` — the target session is valid but bound to
 *   the original or a third profile.
 * - ``invalid`` — any input is malformed or the id does not match.
 *
 * Never throws; never mutates.
 *
 * @param {*} options
 * @returns {{status: "confirmed_target"|"different_binding"|"invalid"}}
 */
function classifyUncertainRefresh(options) {
  if (options === null || typeof options !== "object" ||
      Array.isArray(options)) {
    return { status: "invalid" };
  }
  var targetSessionId = options.targetSessionId;
  var record = options.uncertainRecord;
  var fresh = options.fresh;

  if (!Number.isSafeInteger(targetSessionId) || targetSessionId < 1) {
    return { status: "invalid" };
  }

  if (record === null || typeof record !== "object" ||
      Array.isArray(record)) {
    return { status: "invalid" };
  }
  if (!Number.isSafeInteger(record.generation) || record.generation < 1) {
    return { status: "invalid" };
  }
  if (!_isValidRequestedProfile(record.requestedProfile)) {
    return { status: "invalid" };
  }
  if (!_isValidOriginalProfileId(record.originalProfileId)) {
    return { status: "invalid" };
  }
  if (!_isValidOriginalModelSnapshot(record.originalModelSnapshot)) {
    return { status: "invalid" };
  }

  if (!_isValidSessionResponse(fresh)) {
    return { status: "invalid" };
  }
  if (fresh.id !== targetSessionId) {
    return { status: "invalid" };
  }

  if (fresh.llm_profile_id === record.requestedProfile.id &&
      fresh.llm_model_snapshot === record.requestedProfile.model &&
      fresh.llm_profile_status === "ready") {
    return { status: "confirmed_target" };
  }
  return { status: "different_binding" };
}

/**
 * Plan the side effects for a switch outcome — pure, no execution.
 *
 * ``options`` = { outcome, operation, targetSessionId,
 * currentSessionId, hasTarget }.  Exhaustively handles the eight
 * outcomes; unknown statuses fail closed with ``kind: "ignore"``.
 *
 * ``switched`` / ``not_changed`` only clear protective state when the
 * outcome carries a complete valid SessionResponse whose id matches
 * the target — malformed or mismatched sessions yield ``ignore`` and
 * never lift a temporary block.  ``hasTarget === false`` forbids
 * cache writes.  ``uncertain`` returns a complete, independent record
 * (``uncertainRecord``) built from the operation so the caller never
 * has to reassemble domain state.
 *
 * ``showStatus`` is TARGET-SCOPED: it describes the status the target
 * session should store, so it does NOT depend on ``stillCurrent``.
 * Only ``focus`` and ``syncVisibleUI`` are visible effects gated by
 * ``stillCurrent``.  ``switched`` and ``cancelled`` carry no
 * ``showStatus`` — the executor clears the target's ordinary status
 * itself.  ``uncertain`` carries ``showStatus: null``: the persistent
 * re-apply hint is rendered exclusively from the uncertain record.
 *
 * Never throws; never mutates.
 *
 * @param {*} options
 * @returns {{
 *   kind: "apply"|"ignore",
 *   updateCache: boolean, session: (object|null),
 *   clearBlock: boolean, clearUncertain: boolean,
 *   uncertainRecord: (object|null),
 *   reloadProfiles: boolean, removeSession: boolean,
 *   showStatus: (string|null), focus: ("input"|"apply"|null),
 *   syncVisibleUI: boolean
 * }}
 */
function planSwitchOutcomeEffects(options) {
  function ignore() {
    return {
      kind: "ignore", updateCache: false, session: null,
      clearBlock: false, clearUncertain: false, uncertainRecord: null,
      reloadProfiles: false, removeSession: false,
      showStatus: null, focus: null, syncVisibleUI: false,
    };
  }
  if (options === null || typeof options !== "object" ||
      Array.isArray(options)) {
    return ignore();
  }
  var outcome = options.outcome;
  var operation = options.operation;
  var targetSessionId = options.targetSessionId;
  var currentSessionId = options.currentSessionId;
  var hasTarget = options.hasTarget === true;

  if (outcome === null || typeof outcome !== "object" ||
      Array.isArray(outcome) || typeof outcome.status !== "string") {
    return ignore();
  }
  if (!Number.isSafeInteger(targetSessionId) || targetSessionId < 1) {
    return ignore();
  }

  var stillCurrent =
    Number.isSafeInteger(currentSessionId) &&
    currentSessionId === targetSessionId;

  var plan = {
    kind: "apply", updateCache: false, session: null,
    clearBlock: false, clearUncertain: false, uncertainRecord: null,
    reloadProfiles: false, removeSession: false,
    showStatus: null, focus: null, syncVisibleUI: false,
  };

  // A valid authoritative session matching the target is required for
  // switched / not_changed to touch any protective state.
  function authoritativeSession(value) {
    if (!_isValidSessionResponse(value)) return null;
    if (value.id !== targetSessionId) return null;
    return value;
  }

  switch (outcome.status) {
    case "switched": {
      var swSession = authoritativeSession(outcome.session);
      if (swSession === null) return ignore();
      if (hasTarget) {
        plan.updateCache = true;
        plan.session = swSession;
      }
      plan.clearBlock = true;
      plan.clearUncertain = true;
      if (stillCurrent) {
        plan.syncVisibleUI = true;
        plan.focus = hasTarget ? "input" : null;
      }
      break;
    }

    case "not_changed": {
      var ncSession = authoritativeSession(outcome.session);
      if (ncSession === null) return ignore();
      if (hasTarget) {
        plan.updateCache = true;
        plan.session = ncSession;
      }
      plan.clearBlock = true;
      plan.clearUncertain = true;
      // target-scoped message, independent of visibility
      plan.showStatus =
        "The model switch did not complete. Your conversation still " +
        "uses its previous model.";
      if (stillCurrent) {
        plan.syncVisibleUI = true;
        plan.focus = "apply";
      }
      break;
    }

    case "cancelled":
      if (stillCurrent) {
        plan.focus = "apply";
      }
      break;

    case "not_found":
      plan.removeSession = true;
      plan.clearBlock = true;
      plan.clearUncertain = true;
      break;

    case "validation_error":
      plan.reloadProfiles = true;
      if (typeof outcome.message === "string" &&
          outcome.message.trim() !== "") {
        plan.showStatus = outcome.message;      // raw message, unchanged
      }
      if (stillCurrent) {
        plan.focus = "apply";
      }
      break;

    case "failed":
      if (typeof outcome.message === "string" &&
          outcome.message.trim() !== "") {
        plan.showStatus = outcome.message;      // raw message, unchanged
      }
      if (stillCurrent) {
        plan.focus = "apply";
      }
      break;

    case "uncertain":
      // Only a valid operation for THIS target session can produce a
      // record, and the outcome's requested profile must be an exact
      // triple match — late responses, wrong operations or assembly
      // mistakes must never write another session's request into the
      // current target.
      if (!_safeIsValidOperation(operation)) return ignore();
      if (operation.targetSessionId !== targetSessionId) return ignore();
      if (!_isValidRequestedProfile(outcome.requestedProfile)) return ignore();
      if (outcome.requestedProfile.id !== operation.requestedProfile.id ||
          outcome.requestedProfile.kind !== operation.requestedProfile.kind ||
          outcome.requestedProfile.model !== operation.requestedProfile.model) {
        return ignore();
      }
      plan.uncertainRecord = {
        generation: operation.generation,
        requestedProfile: _copyRequestedProfile(operation.requestedProfile),
        originalProfileId: operation.originalProfileId,
        originalModelSnapshot: operation.originalModelSnapshot,
      };
      // showStatus stays null — the persistent re-apply hint is
      // rendered exclusively from the uncertain record.
      if (stillCurrent) {
        plan.focus = "apply";
      }
      break;

    case "busy":
      break;

    default:
      return ignore();
  }

  return plan;
}

/* ------------------------------------------------------------------ */
/* Controller                                                         */
/* ------------------------------------------------------------------ */

/** Normalise a dependency error into the shared contract, or null. */
function _normalizeError(err) {
  if (err === null || typeof err !== "object" || Array.isArray(err)) {
    return null;
  }
  var kind = err.failureKind;
  if (kind !== "http" && kind !== "network" && kind !== "response_parse") {
    return null;
  }
  return {
    failureKind: kind,
    status: Number.isSafeInteger(err.status) ? err.status : 0,
    code: typeof err.code === "string" ? err.code : null,
    message: typeof err.message === "string" ? err.message : "",
  };
}

/**
 * Create the session profile switch controller.
 *
 * dependencies = {
 *   patchSwitch,              // (sessionId, profileId, ack) => Promise<parsed response>
 *   fetchOneSession,          // (sessionId) => Promise<parsed response>
 *   confirmRemoteHistory,     // (message) => Promise<boolean>
 *   validateSessionResponse,  // (value) => boolean
 * }
 *
 * ``apply(operation)`` is a TOTAL outcome API: apart from concurrent
 * ``busy``, it always resolves to one of the eight outcomes and never
 * rejects — malformed operations, dependency throws/rejections and
 * internal errors all map to ``failed`` (never to network-style
 * ambiguity, so no reconciliation GET runs for them).
 *
 * Per operation: at most ONE confirmation, at most TWO PATCH calls,
 * at most ONE reconciliation GET (this constraint covers the
 * controller only).  ``active`` is restored in every path.
 */
function createSessionProfileSwitchController(dependencies) {
  var patchSwitch = dependencies.patchSwitch;
  var fetchOneSession = dependencies.fetchOneSession;
  var confirmRemoteHistory = dependencies.confirmRemoteHistory;
  var validateSessionResponse = dependencies.validateSessionResponse;

  var active = false;

  function _matchesRequested(operation, session) {
    return session.llm_profile_id === operation.requestedProfile.id &&
      session.llm_model_snapshot === operation.requestedProfile.model &&
      session.llm_profile_status === "ready";
  }

  function _uncertainOutcome(operation) {
    return {
      status: "uncertain",
      message: UNCERTAIN_REAPPLY_TEXT,
      requestedProfile: _copyRequestedProfile(operation.requestedProfile),
    };
  }

  async function _reconcile(operation, source) {
    var fresh;
    try {
      fresh = await fetchOneSession(operation.targetSessionId);
    } catch (err) {
      var normFetch = _normalizeError(err);
      if (normFetch !== null) {
        if (normFetch.failureKind === "http" && normFetch.status === 404) {
          return { status: "not_found" };
        }
        if (normFetch.failureKind === "network" ||
            normFetch.failureKind === "response_parse") {
          return _uncertainOutcome(operation);
        }
        if (normFetch.failureKind === "http" &&
            normFetch.status >= 500 && normFetch.status < 600) {
          return _uncertainOutcome(operation);
        }
      }
      // Internal programming errors (plain Error / TypeError),
      // strings, nulls, malformed error objects and non-ambiguous HTTP
      // statuses must never masquerade as network uncertainty.
      return { status: "failed", message: "Unexpected error." };
    }

    var sessionValid;
    try {
      sessionValid = validateSessionResponse(fresh);
    } catch (_) {
      // A throwing validator is an internal programming error — never
      // masquerade as a network ambiguity.
      return { status: "failed", message: "Unexpected error." };
    }
    if (sessionValid !== true || fresh === null ||
        typeof fresh !== "object" || Array.isArray(fresh) ||
        fresh.id !== operation.targetSessionId) {
      return _uncertainOutcome(operation);
    }

    if (_matchesRequested(operation, fresh)) {
      return { status: "switched", session: fresh, reconciled: true };
    }

    if (source === "invalid_success_response") {
      return { status: "not_changed", session: fresh };
    }
    return _uncertainOutcome(operation);
  }

  async function _patch(operation, ack, confirmed) {
    var response;
    try {
      response = await patchSwitch(
        operation.targetSessionId,
        operation.requestedProfile.id,
        ack,
      );
    } catch (err) {
      var norm = _normalizeError(err);
      if (norm === null) {
        // Internal programming error — never masquerade as network.
        return { status: "failed", message: "Unexpected error." };
      }

      if (norm.failureKind === "http") {
        if (norm.status === 409 &&
            norm.code === "remote_history_ack_required") {
          if (confirmed) {
            return {
              status: "failed",
              message: norm.message || "The switch could not be applied.",
            };
          }
          var confirmedOk;
          try {
            confirmedOk = await confirmRemoteHistory(
              REMOTE_HISTORY_CONFIRM_TEXT,
            );
          } catch (_) {
            return { status: "failed", message: "Unexpected error." };
          }
          if (confirmedOk !== true) {
            return { status: "cancelled" };
          }
          return await _patch(operation, true, true);   // 2nd PATCH, same op
        }
        if (norm.status === 404) {
          return { status: "not_found" };
        }
        if (norm.status === 422) {
          return {
            status: "validation_error",
            message: norm.message || "Invalid request.",
          };
        }
        if (norm.status >= 400 && norm.status < 500) {
          return {
            status: "failed",
            message: norm.message || "The switch request was rejected.",
          };
        }
        if (norm.status >= 500 && norm.status < 600) {
          // 5xx → ambiguous
          return await _reconcile(operation, "ambiguous_failure");
        }
        // 1xx / 3xx / invalid statuses → definite failure, no GET.
        return {
          status: "failed",
          message: norm.message || "The switch request was rejected.",
        };
      }

      // network / response_parse → ambiguous
      return await _reconcile(operation, "ambiguous_failure");
    }

    // 2xx body received.
    var responseValid;
    try {
      responseValid = validateSessionResponse(response);
    } catch (_) {
      // A throwing validator is an internal programming error.
      return { status: "failed", message: "Unexpected error." };
    }
    if (responseValid !== true || response === null ||
        typeof response !== "object" || Array.isArray(response) ||
        response.id !== operation.targetSessionId) {
      return await _reconcile(operation, "invalid_success_response");
    }
    if (!_matchesRequested(operation, response)) {
      return await _reconcile(operation, "invalid_success_response");
    }
    return { status: "switched", session: response, reconciled: false };
  }

  async function apply(operation) {
    // Safe validation first: a getter/Proxy that throws while the
    // operation fields are read must resolve to failed, never reject.
    if (!_safeIsValidOperation(operation)) {
      return { status: "failed", message: "Unexpected error." };
    }
    if (active) {
      return { status: "busy" };
    }
    active = true;
    try {
      var ack = operation.needsConfirmHint;
      var confirmed = ack;
      if (ack) {
        var ok;
        try {
          ok = await confirmRemoteHistory(REMOTE_HISTORY_CONFIRM_TEXT);
        } catch (_) {
          return { status: "failed", message: "Unexpected error." };
        }
        if (ok !== true) {
          return { status: "cancelled" };
        }
        confirmed = true;
      }
      try {
        return await _patch(operation, ack, confirmed);
      } catch (_) {
        return { status: "failed", message: "Unexpected error." };
      }
    } finally {
      active = false;
    }
  }

  return {
    apply: apply,
    isActive: function () { return active; },
  };
}

/* ------------------------------------------------------------------ */
/* Confirmer                                                          */
/* ------------------------------------------------------------------ */

/**
 * Create a dependency-injected remote-history confirmation adapter.
 *
 * adapter = {
 *   showModal(), close(), setMessage(text), focusInitial(),
 *   isConnected(),
 *   onDialogCancel(cb), onCancelClick(cb), onContinueClick(cb)
 * } — the three subscription functions MUST return an unsubscribe
 * function.
 *
 * confirm(message) settles exactly once per call; a second call while
 * one is pending resolves false immediately.  Escape and Cancel map
 * to false, Continue to true.  Any adapter failure — setMessage,
 * subscription registration, showModal or focusInitial throwing, a
 * subscription not returning a function, or a disconnected dialog —
 * fails closed to false with full listener cleanup.  close() throwing
 * cannot prevent settlement or cleanup.
 */
function createRemoteHistoryConfirmer(adapter) {
  var pending = false;

  function _unsubscribeAll(subs) {
    for (var i = 0; i < subs.length; i++) {
      try { subs[i](); } catch (_) { /* best-effort */ }
    }
  }

  function confirm(message) {
    if (pending) {
      return Promise.resolve(false);
    }
    pending = true;

    return new Promise(function (resolve) {
      var settled = false;
      var unsubs = [];

      function settle(value) {
        if (settled) return;
        settled = true;
        pending = false;
        _unsubscribeAll(unsubs);
        try { adapter.close(); } catch (_) { /* must not block */ }
        resolve(value);
      }

      function register(subscribe, onEvent) {
        // subscribe is invoked with the adapter as its receiver so
        // method-style fakes see the right ``this``.  Returns false
        // when the adapter is unusable.
        var unsub;
        try {
          unsub = subscribe.call(adapter, onEvent);
        } catch (_) {
          settle(false);
          return false;
        }
        if (typeof unsub !== "function") {
          settle(false);
          return false;
        }
        if (settled) {
          // The callback fired synchronously during registration —
          // clean this one up manually so nothing leaks.
          try { unsub(); } catch (_) { /* best-effort */ }
          return false;
        }
        unsubs.push(unsub);
        return true;
      }

      try {
        adapter.setMessage(message);
      } catch (_) {
        settle(false);
        return;
      }

      if (!register(adapter.onDialogCancel,
            function () { settle(false); })) return;
      if (!register(adapter.onCancelClick,
            function () { settle(false); })) return;
      if (!register(adapter.onContinueClick,
            function () { settle(true); })) return;

      try {
        if (typeof adapter.isConnected === "function" &&
            !adapter.isConnected()) {
          settle(false);
          return;
        }
        if (typeof adapter.showModal !== "function") {
          settle(false);
          return;
        }
        adapter.showModal();
        adapter.focusInitial();
      } catch (_) {
        settle(false);
        return;
      }
    });
  }

  return { confirm: confirm };
}

// Dual export: browser globals, module.exports for node:test
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    historyStateFromValue: historyStateFromValue,
    captureRequestedProfile: captureRequestedProfile,
    planSessionCacheUpdate: planSessionCacheUpdate,
    preserveSelectedProfileWithoutFallback:
      preserveSelectedProfileWithoutFallback,
    classifyUncertainRefresh: classifyUncertainRefresh,
    planSwitchOutcomeEffects: planSwitchOutcomeEffects,
    createSessionProfileSwitchController: createSessionProfileSwitchController,
    createRemoteHistoryConfirmer: createRemoteHistoryConfirmer,
  };
}
