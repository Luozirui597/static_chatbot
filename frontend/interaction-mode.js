"use strict";

/**
 * Pure helpers for the per-session Interaction mode control.
 *
 * No DOM, fetch, or storage access.  Browser global + module.exports
 * dual export follows the existing classic-script test pattern.
 */

var RECEIVE_TEACHING_MODE = "receive_teaching";
var CORRECTIVE_MODE = "corrective";
var VALID_INTERACTION_MODES = [RECEIVE_TEACHING_MODE, CORRECTIVE_MODE];

/** Strict validator: strings only, exactly one of the two modes. */
function isValidInteractionMode(value) {
  return typeof value === "string" &&
    VALID_INTERACTION_MODES.indexOf(value) !== -1;
}

/** Normalise a session response mode.
 *  Missing/legacy values fall back to receive_teaching; invalid
 *  non-empty values return null so callers can fail closed. */
function resolveInteractionMode(value) {
  if (value === undefined || value === null) {
    return RECEIVE_TEACHING_MODE;
  }
  return isValidInteractionMode(value) ? value : null;
}


/** Build a strict PATCH payload.  Returns null for invalid input. */
function buildSwitchInteractionModePayload(mode) {
  if (isValidInteractionMode(mode) === false) return null;
  return { interaction_mode: mode };
}

function interactionModeLabel(mode) {
  if (mode === CORRECTIVE_MODE) return "Corrective";
  if (mode === RECEIVE_TEACHING_MODE) return "Receive teaching";
  return "";
}

function interactionModeBadgeText(mode) {
  if (mode === CORRECTIVE_MODE) return "Corrective";
  if (mode === RECEIVE_TEACHING_MODE) return "Receive teaching";
  return "";
}

function interactionModeHintText(mode) {
  if (mode === CORRECTIVE_MODE) {
    return "Corrective applies to future messages only. " +
      "If prior teaching history is available, you can start a " +
      "separate History Review from the review panel.";
  }
  return "The agent receives your teaching without proactively " +
    "correcting ordinary factual mistakes.";
}


var MODE_SWITCH_EVENT_FIELDS = [
  "id",
  "session_id",
  "from_mode",
  "to_mode",
  "created_at",
  "history_through_message_id",
  "history_boundary_version",
  "reviewable_user_message_count",
  "review_supported",
];

function _hasOwn(value, key) {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function _isPositiveSafeInteger(value) {
  return typeof value === "number" &&
    Number.isSafeInteger(value) && value > 0;
}

function _isNonNegativeSafeInteger(value) {
  return typeof value === "number" &&
    Number.isSafeInteger(value) && value >= 0;
}

function _isNullablePositiveSafeInteger(value) {
  return value === null || _isPositiveSafeInteger(value);
}

function _isNullableNonNegativeSafeInteger(value) {
  return value === null || _isNonNegativeSafeInteger(value);
}

function isValidModeSwitchEvent(value, validateTimestamp) {
  try {
    if (value === null || typeof value !== "object" ||
        Array.isArray(value)) {
      return false;
    }
    if (typeof validateTimestamp !== "function") {
      return false;
    }
    for (var i = 0; i < MODE_SWITCH_EVENT_FIELDS.length; i++) {
      if (_hasOwn(value, MODE_SWITCH_EVENT_FIELDS[i]) === false) {
        return false;
      }
    }

    var id = value.id;
    var sessionId = value.session_id;
    var fromMode = value.from_mode;
    var toMode = value.to_mode;
    var createdAt = value.created_at;
    var throughId = value.history_through_message_id;
    var boundaryVersion = value.history_boundary_version;
    var reviewableCount = value.reviewable_user_message_count;
    var reviewSupported = value.review_supported;

    if (_isPositiveSafeInteger(id) === false) return false;
    if (_isPositiveSafeInteger(sessionId) === false) return false;
    if (isValidInteractionMode(fromMode) === false) return false;
    if (isValidInteractionMode(toMode) === false) return false;
    if (validateTimestamp(createdAt) === false) return false;
    if (_isNullablePositiveSafeInteger(throughId) === false) return false;
    if (boundaryVersion !== null) {
      if (typeof boundaryVersion !== "string" ||
          boundaryVersion.length < 1 ||
          boundaryVersion.length > 50) {
        return false;
      }
    }
    if (_isNullableNonNegativeSafeInteger(reviewableCount) === false) {
      return false;
    }
    if (typeof reviewSupported !== "boolean") return false;

    if (reviewSupported === true) {
      if (fromMode !== RECEIVE_TEACHING_MODE) return false;
      if (toMode !== CORRECTIVE_MODE) return false;
      if (_isPositiveSafeInteger(throughId) === false) return false;
      if (typeof boundaryVersion !== "string" ||
          boundaryVersion.length < 1) {
        return false;
      }
      if (_isPositiveSafeInteger(reviewableCount) === false) return false;
    }
    return true;
  } catch (_) {
    return false;
  }
}

function copyModeSwitchEvent(value) {
  return {
    id: value.id,
    session_id: value.session_id,
    from_mode: value.from_mode,
    to_mode: value.to_mode,
    created_at: value.created_at,
    history_through_message_id: value.history_through_message_id,
    history_boundary_version: value.history_boundary_version,
    reviewable_user_message_count: value.reviewable_user_message_count,
    review_supported: value.review_supported,
  };
}


function isValidSwitchInteractionModeResponse(
  value, expectedSessionId, expectedMode, originalMode,
  validateSession, validateTimestamp
) {
  if (value === null || typeof value !== "object" ||
      Array.isArray(value)) {
    return false;
  }
  if (Number.isSafeInteger(expectedSessionId) === false ||
      expectedSessionId < 1) {
    return false;
  }
  if (isValidInteractionMode(expectedMode) === false) return false;
  if (isValidInteractionMode(originalMode) === false) return false;
  if (typeof validateSession !== "function") return false;
  if (typeof validateTimestamp !== "function") return false;
  if (validateSession(value.session) === false) return false;
  if (value.session.id !== expectedSessionId) return false;
  if (value.session.interaction_mode !== expectedMode) return false;

  if (expectedMode === originalMode) {
    return value.switch_event === undefined || value.switch_event === null;
  }

  if (value.switch_event === undefined || value.switch_event === null) {
    return false;
  }
  if (isValidModeSwitchEvent(value.switch_event, validateTimestamp) === false) {
    return false;
  }
  if (value.switch_event.session_id !== expectedSessionId) return false;
  if (value.switch_event.from_mode !== originalMode) return false;
  if (value.switch_event.to_mode !== expectedMode) return false;
  return true;
}

function interactionModeDraftForSession(session) {
  if (session === null || typeof session !== "object" ||
      Array.isArray(session)) {
    return RECEIVE_TEACHING_MODE;
  }
  return resolveInteractionMode(session.interaction_mode);
}

function canApplyInteractionMode(options) {
  if (options === null || typeof options !== "object") return false;
  if (options.isSwitching === true) return false;
  if (isValidInteractionMode(options.draftMode) === false) return false;
  if (options.session === null || typeof options.session !== "object" ||
      Array.isArray(options.session)) {
    return false;
  }
  if (options.hasUncertain === true) return true;
  return options.session.interaction_mode !== options.draftMode;
}


/** Resolve the server-authoritative mode without applying any draft. */
function interactionModeAuthoritativeMode(session) {
  if (session === null || typeof session !== "object" ||
      Array.isArray(session)) {
    return null;
  }
  return resolveInteractionMode(session.interaction_mode);
}

/** Safe per-session uncertain lookup for Object.create(null) maps. */
function hasInteractionModeUncertain(uncertainBySession, sessionId) {
  if (uncertainBySession === null ||
      typeof uncertainBySession !== "object" ||
      Array.isArray(uncertainBySession)) {
    return false;
  }
  if (Number.isSafeInteger(sessionId) === false || sessionId < 1) {
    return false;
  }
  var key = String(sessionId);
  if (Object.prototype.hasOwnProperty.call(uncertainBySession, key) === false) {
    return false;
  }
  var record = uncertainBySession[key];
  return record !== null && typeof record === "object" &&
    isValidInteractionMode(record.requestedMode) &&
    isValidInteractionMode(record.originalMode);
}

/** Stable UI text for an unresolved interaction-mode switch. */
function interactionModeUncertainText() {
  return "The server interaction mode could not be confirmed. " +
    "Use Recheck before sending messages or changing models.";
}

/**
 * Dependency-injected, DOM-free controller for mode-switch reconciliation.
 *
 * Outcomes:
 *   switched / not_changed / not_found / uncertain / failed / invalid_request
 */
function createInteractionModeSwitchController(dependencies) {
  if (dependencies === null || typeof dependencies !== "object" ||
      Array.isArray(dependencies) ||
      typeof dependencies.patchSwitch !== "function" ||
      typeof dependencies.fetchSession !== "function" ||
      typeof dependencies.validateSession !== "function" ||
      typeof dependencies.validateTimestamp !== "function") {
    throw new TypeError("Invalid interaction mode controller dependencies");
  }

  var patchSwitch = dependencies.patchSwitch;
  var fetchSession = dependencies.fetchSession;
  var validateSession = dependencies.validateSession;
  var validateTimestamp = dependencies.validateTimestamp;

  function _validOperation(operation) {
    if (operation === null || typeof operation !== "object" ||
        Array.isArray(operation)) return false;
    if (Number.isSafeInteger(operation.targetSessionId) === false ||
        operation.targetSessionId < 1) return false;
    if (isValidInteractionMode(operation.requestedMode) === false) return false;
    if (isValidInteractionMode(operation.originalMode) === false) return false;
    return true;
  }

  function _normaliseError(err) {
    var status = 0;
    var message = "Interaction mode switch could not be confirmed.";
    if (err !== null && typeof err === "object") {
      if (Number.isSafeInteger(err.status)) status = err.status;
      if (typeof err.message === "string" && err.message) message = err.message;
    }
    if (status >= 400 && status < 500 && status !== 404) {
      return {kind: "direct_failure", status: status, message: message};
    }
    return {kind: "reconcile", status: status, message: message};
  }

  async function _reconcile(operation) {
    var raw;
    try {
      raw = await fetchSession(operation.targetSessionId);
    } catch (err) {
      var failure = _normaliseError(err);
      if (failure.status === 404) {
        return {
          status: "not_found",
          message: failure.message,
          switchEvent: null,
        };
      }
      return {
        status: "uncertain",
        message: interactionModeUncertainText(),
        switchEvent: null,
      };
    }
    if (validateSession(raw) === false || raw === null ||
        typeof raw !== "object" || raw.id !== operation.targetSessionId) {
      return {
        status: "uncertain",
        message: interactionModeUncertainText(),
        switchEvent: null,
      };
    }
    if (Object.prototype.hasOwnProperty.call(
      raw, "interaction_mode",
    ) === false) {
      return {
        status: "uncertain",
        message: interactionModeUncertainText(),
        switchEvent: null,
      };
    }
    if (typeof raw.interaction_mode === "string" &&
        isValidInteractionMode(raw.interaction_mode)) {
      var freshMode = raw.interaction_mode;
      if (freshMode === operation.requestedMode) {
        return {
          status: "switched",
          session: raw,
          reconciled: true,
          switchEvent: null,
        };
      }
      if (freshMode === operation.originalMode) {
        return {
          status: "not_changed",
          session: raw,
          reconciled: true,
          switchEvent: null,
        };
      }
    }
    return {
      status: "uncertain",
      message: interactionModeUncertainText(),
      switchEvent: null,
    };
  }

  async function apply(operation) {
    if (_validOperation(operation) === false) {
      return {
        status: "invalid_request",
        message: "Invalid interaction mode switch request.",
      };
    }
    if (operation.requestedMode === operation.originalMode) {
      return {
        status: "not_changed",
        session: null,
        reconciled: false,
        switchEvent: null,
      };
    }

    var response;
    try {
      response = await patchSwitch(
        operation.targetSessionId, operation.requestedMode,
      );
    } catch (err) {
      var failure = _normaliseError(err);
      if (failure.kind === "direct_failure") {
        return {status: "failed", message: failure.message};
      }
      return _reconcile(operation);
    }

    if (isValidSwitchInteractionModeResponse(
      response, operation.targetSessionId, operation.requestedMode,
      operation.originalMode, validateSession, validateTimestamp,
    )) {
      return {
        status: "switched",
        session: response.session,
        reconciled: false,
        switchEvent: copyModeSwitchEvent(response.switch_event),
      };
    }
    return _reconcile(operation);
  }

  return {
    apply: apply,
    reconcile: function (operation) {
      if (_validOperation(operation) === false) {
        return Promise.resolve({
          status: "invalid_request",
          message: "Invalid interaction mode switch request.",
        });
      }
      return _reconcile(operation);
    },
  };
}

// Dual export: browser global for app.js, module.exports for node:test.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    RECEIVE_TEACHING_MODE: RECEIVE_TEACHING_MODE,
    CORRECTIVE_MODE: CORRECTIVE_MODE,
    VALID_INTERACTION_MODES: VALID_INTERACTION_MODES,
    isValidInteractionMode: isValidInteractionMode,
    resolveInteractionMode: resolveInteractionMode,
    buildSwitchInteractionModePayload: buildSwitchInteractionModePayload,
    interactionModeLabel: interactionModeLabel,
    interactionModeBadgeText: interactionModeBadgeText,
    interactionModeHintText: interactionModeHintText,
    isValidModeSwitchEvent: isValidModeSwitchEvent,
    copyModeSwitchEvent: copyModeSwitchEvent,
    isValidSwitchInteractionModeResponse:
      isValidSwitchInteractionModeResponse,
    interactionModeDraftForSession: interactionModeDraftForSession,
    interactionModeAuthoritativeMode: interactionModeAuthoritativeMode,
    hasInteractionModeUncertain: hasInteractionModeUncertain,
    interactionModeUncertainText: interactionModeUncertainText,
    canApplyInteractionMode: canApplyInteractionMode,
    createInteractionModeSwitchController:
      createInteractionModeSwitchController,
  };
}
