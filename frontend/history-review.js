"use strict";

/**
 * Pure History Review frontend logic.
 *
 * Classic-script + CommonJS dual export.  No DOM, fetch, storage or
 * timers are used here; controllers receive all I/O through injected
 * dependencies.
 */

var RECEIVE_TEACHING_MODE = "receive_teaching";
var CORRECTIVE_MODE = "corrective";

var HISTORY_REVIEW_STATUSES = [
  "pending",
  "running",
  "completed",
  "failed",
];

var HISTORY_REVIEW_VERDICTS = [
  "correct",
  "incorrect",
  "uncertain",
  "not_a_claim",
];

var SUMMARY_FIELDS = [
  "id",
  "session_id",
  "mode_switch_event_id",
  "status",
  "eligible_message_count",
  "source_message_count",
  "source_from_message_id",
  "source_through_message_id",
  "truncated",
  "summary",
  "coverage_note",
  "error_code",
  "error_message",
  "findings_count",
  "created_at",
  "started_at",
  "completed_at",
  "updated_at",
];

var FINDING_FIELDS = [
  "seq",
  "source_message_id",
  "verdict",
  "claim_text",
  "correction_text",
  "explanation_text",
];

var ACK_REQUIRED_CODE = "history_review_remote_ack_required";

var UNCERTAIN_MESSAGE =
  "The history review could not be confirmed. Use Recheck.";
var FAILED_MESSAGE = "The history review operation failed.";

function _hasOwn(value, key) {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function _isPlainObject(value) {
  return value !== null &&
    typeof value === "object" &&
    Array.isArray(value) === false;
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

function _isNullableString(value) {
  return value === null || typeof value === "string";
}

function _isStrictBoolean(value) {
  return typeof value === "boolean";
}

function _isValidTimestamp(validator, value) {
  try {
    return validator(value) === true;
  } catch (_) {
    return false;
  }
}

function _timestampMillis(value) {
  if (typeof value !== "string") return -Infinity;
  var hasTimezone =
    /Z$/i.test(value) || /[+-]\d{2}:\d{2}$/.test(value);
  var millis = Date.parse(hasTimezone ? value : value + "Z");
  return Number.isNaN(millis) ? -Infinity : millis;
}

function _isDefaultApiTimestamp(value) {
  if (typeof value !== "string" || value === "") return false;
  var hasTimezone =
    /Z$/i.test(value) || /[+-]\d{2}:\d{2}$/.test(value);
  var date = new Date(hasTimezone ? value : value + "Z");
  return Number.isNaN(date.getTime()) === false;
}

function isValidHistoryReviewStatus(value) {
  return typeof value === "string" &&
    HISTORY_REVIEW_STATUSES.indexOf(value) !== -1;
}

function isValidHistoryReviewVerdict(value) {
  return typeof value === "string" &&
    HISTORY_REVIEW_VERDICTS.indexOf(value) !== -1;
}

function isValidHistoryReviewFinding(value) {
  try {
    if (_isPlainObject(value) === false) return false;
    for (var i = 0; i < FINDING_FIELDS.length; i++) {
      if (_hasOwn(value, FINDING_FIELDS[i]) === false) return false;
    }
    if (_isPositiveSafeInteger(value.seq) === false) return false;
    if (_isPositiveSafeInteger(value.source_message_id) === false) {
      return false;
    }
    if (isValidHistoryReviewVerdict(value.verdict) === false) return false;
    if (typeof value.claim_text !== "string") return false;
    if (_isNullableString(value.correction_text) === false) return false;
    if (_isNullableString(value.explanation_text) === false) return false;
    return true;
  } catch (_) {
    return false;
  }
}

function isValidHistoryReviewSummary(
  value, expectedSessionId, validateTimestamp
) {
  try {
    if (_isPositiveSafeInteger(expectedSessionId) === false) return false;
    if (validateTimestamp !== undefined &&
        typeof validateTimestamp !== "function") {
      return false;
    }
    var timestampValidator = typeof validateTimestamp === "function"
      ? validateTimestamp
      : _isDefaultApiTimestamp;
    if (_isPlainObject(value) === false) return false;
    for (var i = 0; i < SUMMARY_FIELDS.length; i++) {
      if (_hasOwn(value, SUMMARY_FIELDS[i]) === false) return false;
    }

    if (_isPositiveSafeInteger(value.id) === false) return false;
    if (_isPositiveSafeInteger(value.session_id) === false) return false;
    if (value.session_id !== expectedSessionId) return false;
    if (_isPositiveSafeInteger(value.mode_switch_event_id) === false) {
      return false;
    }
    if (isValidHistoryReviewStatus(value.status) === false) return false;

    if (_isNonNegativeSafeInteger(value.eligible_message_count) === false) {
      return false;
    }
    if (_isNonNegativeSafeInteger(value.source_message_count) === false) {
      return false;
    }
    if (_isNullablePositiveSafeInteger(value.source_from_message_id) === false) {
      return false;
    }
    if (_isNullablePositiveSafeInteger(value.source_through_message_id) === false) {
      return false;
    }
    if (_isStrictBoolean(value.truncated) === false) return false;
    if (_isNullableString(value.summary) === false) return false;
    if (_isNullableString(value.coverage_note) === false) return false;
    if (_isNullableString(value.error_code) === false) return false;
    if (_isNullableString(value.error_message) === false) return false;
    if (_isNonNegativeSafeInteger(value.findings_count) === false) {
      return false;
    }
    if (value.status === "completed") {
      if (_isPositiveSafeInteger(value.findings_count) === false) {
        return false;
      }
    } else if (value.findings_count !== 0) {
      return false;
    }
    if (_isValidTimestamp(timestampValidator, value.created_at) === false) {
      return false;
    }
    if (_isValidTimestamp(timestampValidator, value.updated_at) === false) {
      return false;
    }
    if (value.started_at !== null &&
        _isValidTimestamp(timestampValidator, value.started_at) === false) {
      return false;
    }
    if (value.completed_at !== null &&
        _isValidTimestamp(timestampValidator, value.completed_at) === false) {
      return false;
    }
    return true;
  } catch (_) {
    return false;
  }
}

function isValidHistoryReviewSummaryList(
  value, expectedSessionId, validateTimestamp
) {
  try {
    if (_isPositiveSafeInteger(expectedSessionId) === false) return false;
    if (Array.isArray(value) === false) return false;

    var reviewIds = new Set();
    var eventIds = new Set();
    for (var i = 0; i < value.length; i++) {
      var summary = value[i];
      if (isValidHistoryReviewSummary(
        summary, expectedSessionId, validateTimestamp
      ) === false) {
        return false;
      }
      if (reviewIds.has(summary.id)) return false;
      if (eventIds.has(summary.mode_switch_event_id)) return false;
      reviewIds.add(summary.id);
      eventIds.add(summary.mode_switch_event_id);
    }
    return true;
  } catch (_) {
    return false;
  }
}

function isValidHistoryReviewDetail(
  value, expectedSessionId, expectedReviewId,
  expectedModeSwitchEventId, validateTimestamp
) {
  try {
    if (_isPositiveSafeInteger(expectedSessionId) === false) return false;
    if (_isPositiveSafeInteger(expectedReviewId) === false) return false;
    if (_isPositiveSafeInteger(expectedModeSwitchEventId) === false) {
      return false;
    }
    if (isValidHistoryReviewSummary(
      value, expectedSessionId, validateTimestamp
    ) === false) {
      return false;
    }
    if (value.id !== expectedReviewId) return false;
    if (value.mode_switch_event_id !== expectedModeSwitchEventId) {
      return false;
    }
    if (_hasOwn(value, "findings") === false) return false;
    if (Array.isArray(value.findings) === false) return false;
    if (value.findings_count !== value.findings.length) return false;

    if (value.status !== "completed") {
      if (value.findings_count !== 0) return false;
      if (value.findings.length !== 0) return false;
      return true;
    }
    if (value.findings.length < 1) return false;

    var previousSeq = 0;
    var seenSeq = new Set();
    for (var i = 0; i < value.findings.length; i++) {
      var finding = value.findings[i];
      if (isValidHistoryReviewFinding(finding) === false) return false;
      if (finding.seq <= previousSeq) return false;
      if (seenSeq.has(finding.seq)) return false;
      seenSeq.add(finding.seq);
      previousSeq = finding.seq;
    }
    return true;
  } catch (_) {
    return false;
  }
}

function _copyModeSwitchEvent(event) {
  return {
    id: event.id,
    session_id: event.session_id,
    from_mode: event.from_mode,
    to_mode: event.to_mode,
    created_at: event.created_at,
    history_through_message_id: event.history_through_message_id,
    history_boundary_version: event.history_boundary_version,
    reviewable_user_message_count: event.reviewable_user_message_count,
    review_supported: event.review_supported,
  };
}

function selectLatestHistoryReviewProposal(options) {
  try {
    if (_isPlainObject(options) === false) return null;
    if (_hasOwn(options, "events") === false) return null;
    if (_hasOwn(options, "reviews") === false) return null;
    if (_hasOwn(options, "sessionId") === false) return null;
    if (_hasOwn(options, "currentMode") === false) return null;
    if (_hasOwn(options, "validateTimestamp") === false) return null;
    if (_hasOwn(options, "validateModeSwitchEvent") === false) return null;

    var sessionId = options.sessionId;
    var currentMode = options.currentMode;
    var events = options.events;
    var reviews = options.reviews;
    var validateTimestamp = options.validateTimestamp;
    var validateModeSwitchEvent = options.validateModeSwitchEvent;

    if (_isPositiveSafeInteger(sessionId) === false) return null;
    if (currentMode !== CORRECTIVE_MODE) return null;
    if (Array.isArray(events) === false) return null;
    if (Array.isArray(reviews) === false) return null;
    if (typeof validateTimestamp !== "function") return null;
    if (typeof validateModeSwitchEvent !== "function") return null;
    if (isValidHistoryReviewSummaryList(
      reviews, sessionId, validateTimestamp
    ) === false) {
      return null;
    }

    var reviewedEventIds = new Set();
    for (var r = 0; r < reviews.length; r++) {
      reviewedEventIds.add(reviews[r].mode_switch_event_id);
    }

    var copiedEvents = [];
    for (var e = 0; e < events.length; e++) {
      var event = events[e];
      if (validateModeSwitchEvent(event, validateTimestamp) !== true) {
        return null;
      }
      if (event.session_id !== sessionId) return null;
      copiedEvents.push(_copyModeSwitchEvent(event));
    }

    copiedEvents.sort(function (left, right) {
      var leftMillis = _timestampMillis(left.created_at);
      var rightMillis = _timestampMillis(right.created_at);
      if (leftMillis !== rightMillis) return rightMillis - leftMillis;
      return right.id - left.id;
    });

    for (var i = 0; i < copiedEvents.length; i++) {
      var candidate = copiedEvents[i];
      if (candidate.from_mode !== RECEIVE_TEACHING_MODE) continue;
      if (candidate.to_mode !== CORRECTIVE_MODE) continue;
      if (candidate.review_supported !== true) continue;
      if (_isPositiveSafeInteger(
        candidate.reviewable_user_message_count
      ) === false) continue;
      if (_isPositiveSafeInteger(
        candidate.history_through_message_id
      ) === false) continue;
      if (reviewedEventIds.has(candidate.id)) continue;
      return {
        eventId: candidate.id,
        sessionId: candidate.session_id,
        reviewableUserMessageCount:
          candidate.reviewable_user_message_count,
        createdAt: candidate.created_at,
        event: candidate,
      };
    }
    return null;
  } catch (_) {
    return null;
  }
}

function buildHistoryReviewCreatePayload(
  modeSwitchEventId, acknowledgeRemoteHistory
) {
  if (_isPositiveSafeInteger(modeSwitchEventId) === false) return null;
  if (_isStrictBoolean(acknowledgeRemoteHistory) === false) return null;
  return {
    mode_switch_event_id: modeSwitchEventId,
    acknowledge_remote_history: acknowledgeRemoteHistory,
  };
}

function parseHistoryReviewAckRequired(value) {
  try {
    if (_isPlainObject(value) === false) return null;
    if (_hasOwn(value, "detail") === false) return null;
    var detail = value.detail;
    if (_isPlainObject(detail) === false) return null;

    var fields = [
      "code",
      "message",
      "source_message_count",
      "reviewer_profile_label",
      "reviewer_model",
      "truncated",
    ];
    for (var i = 0; i < fields.length; i++) {
      if (_hasOwn(detail, fields[i]) === false) return null;
    }

    if (detail.code !== ACK_REQUIRED_CODE) return null;
    if (typeof detail.message !== "string") return null;
    if (_isPositiveSafeInteger(detail.source_message_count) === false) {
      return null;
    }
    if (typeof detail.reviewer_profile_label !== "string" ||
        detail.reviewer_profile_label.trim() === "") {
      return null;
    }
    if (typeof detail.reviewer_model !== "string" ||
        detail.reviewer_model.trim() === "") {
      return null;
    }
    if (_isStrictBoolean(detail.truncated) === false) return null;

    return {
      code: ACK_REQUIRED_CODE,
      message: detail.message,
      sourceMessageCount: detail.source_message_count,
      reviewerProfileLabel: detail.reviewer_profile_label,
      reviewerModel: detail.reviewer_model,
      truncated: detail.truncated,
    };
  } catch (_) {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Controller                                                         */
/* ------------------------------------------------------------------ */

function _normaliseAdapterError(err) {
  try {
    if (_isPlainObject(err) === false) return null;
    var failureKind = err.failureKind;
    if (failureKind !== "http" &&
        failureKind !== "network" &&
        failureKind !== "response_parse") {
      return null;
    }
    var status = err.status;
    if (typeof status !== "number" ||
        Number.isSafeInteger(status) === false ||
        status < 0) {
      status = 0;
    }
    var code = null;
    if (typeof err.code === "string") {
      code = err.code;
    }
    var message = "";
    if (err.message !== null && err.message !== undefined) {
      if (typeof err.message !== "string") return null;
      message = err.message;
    }
    var body = null;
    if (_hasOwn(err, "body") && err.body !== undefined) {
      body = err.body;
    }
    return {
      failureKind: failureKind,
      status: status,
      code: code,
      message: message,
      body: body,
    };
  } catch (_) {
    return null;
  }
}

function _captureOperation(operation) {
  try {
    if (_isPlainObject(operation) === false) return null;
    if (_hasOwn(operation, "targetSessionId") === false) return null;
    if (_hasOwn(operation, "modeSwitchEventId") === false) return null;
    if (_hasOwn(operation, "generation") === false) return null;

    var targetSessionId = operation.targetSessionId;
    var modeSwitchEventId = operation.modeSwitchEventId;
    var generation = operation.generation;

    if (_isPositiveSafeInteger(targetSessionId) === false) return null;
    if (_isPositiveSafeInteger(modeSwitchEventId) === false) return null;
    if (_isNonNegativeSafeInteger(generation) === false) return null;
    return {
      targetSessionId: targetSessionId,
      modeSwitchEventId: modeSwitchEventId,
      generation: generation,
    };
  } catch (_) {
    return null;
  }
}

function _invalidRequestOutcome() {
  return {
    status: "invalid_request",
    targetSessionId: 0,
    modeSwitchEventId: 0,
    generation: 0,
  };
}

function _baseOutcome(operation, status) {
  return {
    status: status,
    targetSessionId: operation.targetSessionId,
    modeSwitchEventId: operation.modeSwitchEventId,
    generation: operation.generation,
  };
}

function _authoritativeOutcome(operation, detail, reconciled) {
  var outcome = _baseOutcome(operation, "authoritative");
  outcome.detail = detail;
  outcome.reconciled = reconciled === true;
  return outcome;
}

function _uncertainOutcome(operation) {
  var outcome = _baseOutcome(operation, "uncertain");
  outcome.message = UNCERTAIN_MESSAGE;
  return outcome;
}

function _failedOutcome(operation, code, message) {
  var outcome = _baseOutcome(operation, "failed");
  outcome.code = typeof code === "string" && code ? code : null;
  outcome.message = typeof message === "string" && message
    ? message
    : FAILED_MESSAGE;
  return outcome;
}

function _cancelledOutcome(operation) {
  return _baseOutcome(operation, "cancelled");
}

function _busyOutcome(operation) {
  return _baseOutcome(operation, "busy");
}

function _sessionNotFoundOutcome(operation) {
  return _baseOutcome(operation, "session_not_found");
}

function _safeMessageFromError(error) {
  if (error.message !== "" && error.message !== undefined) {
    return error.message;
  }
  return FAILED_MESSAGE;
}

function createHistoryReviewController(dependencies) {
  if (_isPlainObject(dependencies) === false) {
    throw new TypeError("Invalid history review controller dependencies");
  }
  var postReview = dependencies.postReview;
  var fetchReviewSummaries = dependencies.fetchReviewSummaries;
  var fetchReviewDetail = dependencies.fetchReviewDetail;
  var confirmRemoteHistory = dependencies.confirmRemoteHistory;
  var validateTimestamp = dependencies.validateTimestamp;

  if (typeof postReview !== "function" ||
      typeof fetchReviewSummaries !== "function" ||
      typeof fetchReviewDetail !== "function" ||
      typeof confirmRemoteHistory !== "function" ||
      typeof validateTimestamp !== "function") {
    throw new TypeError("Invalid history review controller dependencies");
  }

  var activeBySession = Object.create(null);

  function _sessionKey(operation) {
    return String(operation.targetSessionId);
  }

  function _validDetailForOperation(response, operation, expectedReviewId) {
    try {
      if (_isPlainObject(response) === false) return false;
      if (_hasOwn(response, "id") === false) return false;
      return isValidHistoryReviewDetail(
        response,
        operation.targetSessionId,
        expectedReviewId,
        operation.modeSwitchEventId,
        validateTimestamp,
      );
    } catch (_) {
      return false;
    }
  }

  async function _reconcile(operation) {
    var summaries;
    try {
      summaries = await fetchReviewSummaries(operation.targetSessionId);
    } catch (err) {
      var listError = _normaliseAdapterError(err);
      if (listError === null) {
        return _failedOutcome(operation, null, FAILED_MESSAGE);
      }
      if (listError.failureKind === "http" &&
          listError.status === 404 &&
          listError.code === "history_review_session_not_found") {
        return _sessionNotFoundOutcome(operation);
      }
      if (listError.failureKind === "network" ||
          listError.failureKind === "response_parse" ||
          listError.status >= 500) {
        return _uncertainOutcome(operation);
      }
      if (listError.failureKind === "http") {
        return _uncertainOutcome(operation);
      }
      return _failedOutcome(operation, null, FAILED_MESSAGE);
    }

    if (isValidHistoryReviewSummaryList(
      summaries, operation.targetSessionId, validateTimestamp
    ) === false) {
      return _uncertainOutcome(operation);
    }

    var matching = null;
    for (var i = 0; i < summaries.length; i++) {
      if (summaries[i].mode_switch_event_id ===
          operation.modeSwitchEventId) {
        matching = summaries[i];
        break;
      }
    }
    if (matching === null) return _uncertainOutcome(operation);

    var detail;
    try {
      detail = await fetchReviewDetail(
        operation.targetSessionId, matching.id
      );
    } catch (err) {
      var detailError = _normaliseAdapterError(err);
      if (detailError === null) {
        return _failedOutcome(operation, null, FAILED_MESSAGE);
      }
      if (detailError.failureKind === "network" ||
          detailError.failureKind === "response_parse" ||
          detailError.status >= 500 ||
          detailError.status === 404) {
        return _uncertainOutcome(operation);
      }
      return _uncertainOutcome(operation);
    }

    if (_validDetailForOperation(detail, operation, matching.id) === false) {
      return _uncertainOutcome(operation);
    }
    return _authoritativeOutcome(operation, detail, true);
  }

  async function _attemptPost(operation, acknowledge, alreadyConfirmed) {
    var payload = buildHistoryReviewCreatePayload(
      operation.modeSwitchEventId, acknowledge
    );
    if (payload === null) {
      return _failedOutcome(operation, null, FAILED_MESSAGE);
    }

    var response;
    try {
      response = await postReview(operation.targetSessionId, payload);
    } catch (err) {
      var error = _normaliseAdapterError(err);
      if (error === null) {
        return _failedOutcome(operation, null, FAILED_MESSAGE);
      }

      if (error.failureKind === "network" ||
          error.failureKind === "response_parse") {
        return await _reconcile(operation);
      }

      if (error.failureKind !== "http") {
        return _failedOutcome(operation, null, FAILED_MESSAGE);
      }

      if (error.status === 409) {
        if (error.code === ACK_REQUIRED_CODE) {
          var ackMetadata = parseHistoryReviewAckRequired(error.body);
          if (ackMetadata === null ||
              ackMetadata.code !== error.code) {
            return _failedOutcome(
              operation, error.code, _safeMessageFromError(error)
            );
          }
          if (alreadyConfirmed) {
            return _failedOutcome(
              operation,
              ACK_REQUIRED_CODE,
              "Remote history acknowledgement is still required.",
            );
          }
          var confirmed;
          try {
            confirmed = await confirmRemoteHistory(ackMetadata);
          } catch (_) {
            return _failedOutcome(operation, null, FAILED_MESSAGE);
          }
          if (confirmed !== true) {
            return _cancelledOutcome(operation);
          }
          return await _attemptPost(operation, true, true);
        }
        if (error.code === "history_review_execution_conflict") {
          return await _reconcile(operation);
        }
        return _failedOutcome(
          operation, error.code, _safeMessageFromError(error)
        );
      }

      if (error.status === 404 &&
          error.code === "history_review_session_not_found") {
        return _sessionNotFoundOutcome(operation);
      }

      if (error.status >= 400 && error.status < 500) {
        return _failedOutcome(
          operation, error.code, _safeMessageFromError(error)
        );
      }

      if (error.status >= 500 && error.status < 600) {
        return await _reconcile(operation);
      }

      return _failedOutcome(operation, null, FAILED_MESSAGE);
    }

    try {
      if (_isPlainObject(response) === false ||
          _hasOwn(response, "id") === false) {
        return await _reconcile(operation);
      }
      var reviewId = response.id;
      if (_isPositiveSafeInteger(reviewId) === false) {
        return await _reconcile(operation);
      }
      if (_validDetailForOperation(response, operation, reviewId) === true) {
        return _authoritativeOutcome(operation, response, false);
      }
      return await _reconcile(operation);
    } catch (_) {
      return await _reconcile(operation);
    }
  }

  async function start(operation) {
    var captured;
    try {
      captured = _captureOperation(operation);
    } catch (_) {
      return _invalidRequestOutcome();
    }
    if (captured === null) return _invalidRequestOutcome();

    var key = _sessionKey(captured);
    if (activeBySession[key] === true) {
      return _busyOutcome(captured);
    }
    activeBySession[key] = true;
    try {
      return await _attemptPost(captured, false, false);
    } catch (_) {
      return _failedOutcome(captured, null, FAILED_MESSAGE);
    } finally {
      delete activeBySession[key];
    }
  }

  async function recheck(operation) {
    var captured;
    try {
      captured = _captureOperation(operation);
    } catch (_) {
      return _invalidRequestOutcome();
    }
    if (captured === null) return _invalidRequestOutcome();

    var key = _sessionKey(captured);
    if (activeBySession[key] === true) {
      return _busyOutcome(captured);
    }
    activeBySession[key] = true;
    try {
      return await _reconcile(captured);
    } catch (_) {
      return _failedOutcome(captured, null, FAILED_MESSAGE);
    } finally {
      delete activeBySession[key];
    }
  }

  function isActive(sessionId) {
    if (_isPositiveSafeInteger(sessionId) === false) return false;
    return activeBySession[String(sessionId)] === true;
  }

  return {
    start: start,
    recheck: recheck,
    isActive: isActive,
  };
}

// Dual export: browser globals for app.js, CommonJS for node:test.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    RECEIVE_TEACHING_MODE: RECEIVE_TEACHING_MODE,
    CORRECTIVE_MODE: CORRECTIVE_MODE,
    HISTORY_REVIEW_STATUSES: HISTORY_REVIEW_STATUSES,
    HISTORY_REVIEW_VERDICTS: HISTORY_REVIEW_VERDICTS,
    isValidHistoryReviewStatus: isValidHistoryReviewStatus,
    isValidHistoryReviewVerdict: isValidHistoryReviewVerdict,
    isValidHistoryReviewFinding: isValidHistoryReviewFinding,
    isValidHistoryReviewSummary: isValidHistoryReviewSummary,
    isValidHistoryReviewSummaryList: isValidHistoryReviewSummaryList,
    isValidHistoryReviewDetail: isValidHistoryReviewDetail,
    selectLatestHistoryReviewProposal: selectLatestHistoryReviewProposal,
    buildHistoryReviewCreatePayload: buildHistoryReviewCreatePayload,
    parseHistoryReviewAckRequired: parseHistoryReviewAckRequired,
    createHistoryReviewController: createHistoryReviewController,
  };
}
