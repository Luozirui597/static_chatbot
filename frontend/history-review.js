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
var INTERRUPTED_ERROR_CODE = "history_review_execution_interrupted";

/**
 * Dialog copy shared by the browser bundle and the Node tests.
 *
 * The start dialog title is the confirmation question and the body
 * explains what the operation does; the two must never repeat the same
 * sentence.  None of this text may contain source messages, reviewer
 * prompts or other private material.
 */
var HISTORY_REVIEW_DIALOG_COPY = {
  startTitle: "Start a history review?",
  startBody:
    "A separate review is generated from the teaching history frozen " +
    "at this mode switch. Messages from this conversation only are " +
    "used, and the review never runs on its own — you have to start " +
    "it. The report appears here when it finishes.",
  continueTitle: "Continue this history review?",
  continueBody:
    "The pending review resumes with the teaching history that was " +
    "already frozen when it started. The completed report appears " +
    "here.",
  retryTitle: "Retry this history review?",
  retryBody:
    "The previous attempt was interrupted. It may already have " +
    "reached the model. Retrying sends the same frozen sources " +
    "again, and a remote model may produce a duplicate call or cost. " +
    "Continue only if you want to retry.",
};

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

function isValidHistoryReviewEventList(
  value, expectedSessionId, validateTimestamp, validateModeSwitchEvent
) {
  try {
    if (_isPositiveSafeInteger(expectedSessionId) === false) return false;
    if (Array.isArray(value) === false) return false;
    if (typeof validateTimestamp !== "function") return false;
    if (typeof validateModeSwitchEvent !== "function") return false;

    var seenIds = new Set();
    var previousMillis = Infinity;
    var previousId = Infinity;

    for (var i = 0; i < value.length; i++) {
      var event = value[i];
      if (validateModeSwitchEvent(event, validateTimestamp) !== true) {
        return false;
      }
      if (event.session_id !== expectedSessionId) return false;
      if (seenIds.has(event.id)) return false;
      seenIds.add(event.id);

      var millis = _timestampMillis(event.created_at);
      if (Number.isNaN(millis)) return false;
      if (millis > previousMillis) return false;
      if (millis === previousMillis && event.id > previousId) return false;
      previousMillis = millis;
      previousId = event.id;
    }
    return true;
  } catch (_) {
    return false;
  }
}

function _copyHistoryReviewSummary(value) {
  return {
    id: value.id,
    session_id: value.session_id,
    mode_switch_event_id: value.mode_switch_event_id,
    status: value.status,
    eligible_message_count: value.eligible_message_count,
    source_message_count: value.source_message_count,
    source_from_message_id: value.source_from_message_id,
    source_through_message_id: value.source_through_message_id,
    truncated: value.truncated,
    summary: value.summary,
    coverage_note: value.coverage_note,
    error_code: value.error_code,
    error_message: value.error_message,
    findings_count: value.findings_count,
    created_at: value.created_at,
    started_at: value.started_at,
    completed_at: value.completed_at,
    updated_at: value.updated_at,
  };
}

function _compareHistoryReviewSummaryDesc(left, right) {
  var leftMillis = _timestampMillis(left.created_at);
  var rightMillis = _timestampMillis(right.created_at);
  if (leftMillis !== rightMillis) return rightMillis - leftMillis;
  return right.id - left.id;
}

function isHistoryReviewDetailCurrent(
  detail, summary, expectedSessionId, validateTimestamp
) {
  try {
    if (isValidHistoryReviewSummary(
          summary, expectedSessionId, validateTimestamp) === false) {
      return false;
    }
    if (isValidHistoryReviewDetail(
          detail, expectedSessionId, summary.id,
          summary.mode_switch_event_id, validateTimestamp) === false) {
      return false;
    }
    if (detail.id !== summary.id) return false;
    if (detail.mode_switch_event_id !== summary.mode_switch_event_id) return false;
    if (detail.status !== summary.status) return false;
    var detailMillis = _timestampMillis(detail.updated_at);
    var summaryMillis = _timestampMillis(summary.updated_at);
    if (Number.isNaN(detailMillis) || Number.isNaN(summaryMillis)) return false;
    if (detailMillis < summaryMillis) return false;
    return true;
  } catch (_) {
    return false;
  }
}

function _historyReviewStatusRank(status) {
  if (status === "pending") return 0;
  if (status === "running") return 1;
  return 2;
}

function historyReviewDetailMayUpdateSummary(
  detail, summary, expectedSessionId, validateTimestamp
) {
  try {
    if (isValidHistoryReviewDetail(
          detail, expectedSessionId, summary.id,
          summary.mode_switch_event_id, validateTimestamp) === false) {
      return false;
    }
    if (isValidHistoryReviewSummary(
          summary, expectedSessionId, validateTimestamp) === false) {
      return false;
    }
    if (detail.id !== summary.id) return false;
    if (detail.mode_switch_event_id !== summary.mode_switch_event_id) return false;
    var detailMillis = _timestampMillis(detail.updated_at);
    var summaryMillis = _timestampMillis(summary.updated_at);
    if (Number.isNaN(detailMillis) || Number.isNaN(summaryMillis)) return false;
    if (detailMillis < summaryMillis) return false;
    if (detail.status === summary.status) return true;
    if (detailMillis === summaryMillis) return false;
    var detailRank = _historyReviewStatusRank(detail.status);
    var summaryRank = _historyReviewStatusRank(summary.status);
    if (detailRank <= summaryRank) return false;
    return true;
  } catch (_) {
    return false;
  }
}

function reconcileHistoryReviewCaches(
  detailsById, summaries, expectedSessionId, validateTimestamp
) {
  try {
    if (isValidHistoryReviewSummaryList(
          summaries, expectedSessionId, validateTimestamp) === false) {
      return null;
    }
    if (_isPlainObject(detailsById) === false) return null;

    var currentSummaries = summaries.map(_copyHistoryReviewSummary);
    var currentDetails = Object.create(null);
    var staleReviewIds = [];
    var keys = Object.keys(detailsById);

    for (var i = 0; i < keys.length; i++) {
      var key = keys[i];
      var detail = detailsById[key];
      if (isValidHistoryReviewDetail(
            detail, expectedSessionId, detail.id,
            detail.mode_switch_event_id, validateTimestamp) === false) {
        staleReviewIds.push(Number(key));
        continue;
      }

      var summary = null;
      for (var s = 0; s < currentSummaries.length; s++) {
        if (currentSummaries[s].id === detail.id) {
          summary = currentSummaries[s];
          break;
        }
      }
      if (summary === null) { staleReviewIds.push(Number(key)); continue; }

      if (isHistoryReviewDetailCurrent(
            detail, summary, expectedSessionId, validateTimestamp)) {
        currentDetails[key] = detail;
        continue;
      }

      if (historyReviewDetailMayUpdateSummary(
            detail, summary, expectedSessionId, validateTimestamp)) {
        var derived = historyReviewSummaryFromDetail(
          detail, expectedSessionId, validateTimestamp);
        if (derived === null) { staleReviewIds.push(Number(key)); continue; }
        var next = upsertHistoryReviewSummary(
          currentSummaries, derived, expectedSessionId, validateTimestamp);
        if (next === null) { staleReviewIds.push(Number(key)); continue; }
        currentSummaries = next;
        currentDetails[key] = detail;
        continue;
      }

      staleReviewIds.push(Number(key));
    }

    return {
      detailsById: currentDetails,
      summaries: currentSummaries,
      staleReviewIds: staleReviewIds,
    };
  } catch (_) {
    return null;
  }
}

function historyReviewSummaryFromDetail(
  detail, expectedSessionId, validateTimestamp
) {
  try {
    if (_isPlainObject(detail) === false) return null;
    if (_hasOwn(detail, "id") === false) return null;
    if (_hasOwn(detail, "mode_switch_event_id") === false) return null;
    var reviewId = detail.id;
    var eventId = detail.mode_switch_event_id;
    if (_isPositiveSafeInteger(reviewId) === false) return null;
    if (_isPositiveSafeInteger(eventId) === false) return null;
    if (isValidHistoryReviewDetail(
          detail, expectedSessionId, reviewId, eventId,
          validateTimestamp) === false) {
      return null;
    }
    return _copyHistoryReviewSummary(detail);
  } catch (_) {
    return null;
  }
}

function upsertHistoryReviewSummary(
  summaries, summary, expectedSessionId, validateTimestamp
) {
  try {
    if (isValidHistoryReviewSummary(
          summary, expectedSessionId, validateTimestamp) === false) {
      return null;
    }
    if (summaries !== null && Array.isArray(summaries) === false) {
      return null;
    }

    var existing = [];
    if (summaries !== null) {
      if (isValidHistoryReviewSummaryList(
            summaries, expectedSessionId, validateTimestamp) === false) {
        return null;
      }
      existing = summaries.map(_copyHistoryReviewSummary);
    }

    var next = existing.filter(function (item) {
      return item.id !== summary.id &&
        item.mode_switch_event_id !== summary.mode_switch_event_id;
    });
    next.push(_copyHistoryReviewSummary(summary));
    next.sort(_compareHistoryReviewSummaryDesc);

    if (isValidHistoryReviewSummaryList(
          next, expectedSessionId, validateTimestamp) === false) {
      return null;
    }
    return next;
  } catch (_) {
    return null;
  }
}

function isHistoryReviewTargetBusy(
  sessionId, state, validateTimestamp
) {
  try {
    if (_isPositiveSafeInteger(sessionId) === false) return false;
    if (_isPlainObject(state) === false) return false;

    if (state.operation !== null && state.operation !== undefined) return true;
    if (_isPositiveSafeInteger(state.uncertainEventId)) return true;

    var summariesUsable =
      Array.isArray(state.summaries) &&
      state.summariesStatus !== "error" &&
      isValidHistoryReviewSummaryList(
        state.summaries, sessionId, validateTimestamp);

    if (summariesUsable) {
      for (var i = 0; i < state.summaries.length; i++) {
        if (state.summaries[i].status === "running") return true;
      }
      return false;
    }

    if (_isPlainObject(state.detailsById)) {
      var keys = Object.keys(state.detailsById);
      for (var d = 0; d < keys.length; d++) {
        var detail = state.detailsById[keys[d]];
        if (isValidHistoryReviewDetail(
              detail, sessionId, detail.id, detail.mode_switch_event_id,
              validateTimestamp) &&
            detail.status === "running") {
          return true;
        }
      }
    }
    return false;
  } catch (_) {
    return false;
  }
}

function isHistoryReviewOperationVisiblyWorking(operation) {
  try {
    if (_isPlainObject(operation) === false) return false;
    var phase = operation.phase;
    // confirming_start is a concurrency reservation, not observable work.
    // Unknown/malformed phases stay non-working on purpose: concurrency is
    // still protected by operation !== null, but the UI must not claim work
    // that no known execution phase actually started.
    return phase === "posting" || phase === "rechecking";
  } catch (_) {
    return false;
  }
}

function _selectedSummaryForPanel(
  state, sessionId, validateTimestamp
) {
  if (Array.isArray(state.summaries) === false) return null;
  if (isValidHistoryReviewSummaryList(
        state.summaries, sessionId, validateTimestamp) === false) {
    return null;
  }
  for (var i = 0; i < state.summaries.length; i++) {
    if (state.summaries[i].id === state.selectedReviewId) {
      return state.summaries[i];
    }
  }
  return state.summaries.length > 0 ? state.summaries[0] : null;
}

function buildHistoryReviewPanelModel(options) {
  try {
    if (_isPlainObject(options) === false) return null;
    var sessionId = options.sessionId;
    var state = options.state;
    var currentMode = options.currentMode;
    var validateTimestamp = options.validateTimestamp;
    var validateModeSwitchEvent = options.validateModeSwitchEvent;

    if (_isPositiveSafeInteger(sessionId) === false) return null;
    if (_isPlainObject(state) === false) return null;
    if (typeof validateTimestamp !== "function") return null;
    if (typeof validateModeSwitchEvent !== "function") return null;

    var selectedSummary = _selectedSummaryForPanel(
      state, sessionId, validateTimestamp
    );
    var selectedDetail = null;
    if (selectedSummary !== null &&
        _isPlainObject(state.detailsById)) {
      var candidate = state.detailsById[selectedSummary.id];
      if (isHistoryReviewDetailCurrent(
            candidate, selectedSummary, sessionId, validateTimestamp)) {
        selectedDetail = candidate;
      }
    }

    var proposal = null;
    if (currentMode === CORRECTIVE_MODE &&
        Array.isArray(state.events) &&
        Array.isArray(state.summaries) &&
        isValidHistoryReviewEventList(
          state.events, sessionId, validateTimestamp,
          validateModeSwitchEvent) &&
        isValidHistoryReviewSummaryList(
          state.summaries, sessionId, validateTimestamp)) {
      proposal = selectLatestHistoryReviewProposal({
        events: state.events,
        reviews: state.summaries,
        sessionId: sessionId,
        currentMode: currentMode,
        validateTimestamp: validateTimestamp,
        validateModeSwitchEvent: validateModeSwitchEvent,
      });
      if (proposal !== null &&
          state.proposalDismissedEventId === proposal.eventId) {
        proposal = null;
      }
    }

    var actionKind = "none";
    var actionEventId = null;
    var actionReviewId = null;
    var selectedStatus = selectedSummary === null
      ? null
      : selectedSummary.status;
    var selectedErrorCode = selectedSummary === null
      ? null
      : selectedSummary.error_code;
    var selectedDetailMissing =
      selectedSummary !== null &&
      selectedSummary.status === "completed" &&
      selectedDetail === null;
    var selectedDetailErrorText = selectedSummary !== null &&
      _isPlainObject(state.detailErrorByReviewId) &&
      typeof state.detailErrorByReviewId[selectedSummary.id] === "string" &&
      state.detailErrorByReviewId[selectedSummary.id] !== ""
      ? state.detailErrorByReviewId[selectedSummary.id]
      : null;
    var selectedDetailError = selectedDetailErrorText !== null;

    if (_isPositiveSafeInteger(state.uncertainEventId)) {
      actionKind = "recheck";
      actionEventId = state.uncertainEventId;
    } else if (selectedSummary !== null && selectedStatus === "running") {
      actionKind = "recheck";
      actionEventId = selectedSummary.mode_switch_event_id;
    } else if (selectedSummary !== null && selectedStatus === "failed") {
      actionKind = selectedErrorCode === INTERRUPTED_ERROR_CODE
        ? "retry"
        : "recheck";
      actionEventId = selectedSummary.mode_switch_event_id;
    } else if (selectedDetailError || selectedDetailMissing) {
      actionKind = "reload";
      actionReviewId = selectedSummary.id;
    } else if (
      (state.eventsStatus === "error" || state.summariesStatus === "error") &&
      actionEventId === null
    ) {
      actionKind = "reload";
    }

    var operation = state.operation || null;
    var workingVisible =
      isHistoryReviewOperationVisiblyWorking(operation);

    var badgeText = "";
    if (workingVisible) {
      badgeText = "Working";
    } else if (proposal !== null) {
      badgeText = "Review available";
    } else if (selectedStatus === "pending") {
      badgeText = "Pending";
    } else if (selectedStatus === "running") {
      badgeText = "Running";
    } else if (selectedStatus === "completed") {
      badgeText = "Completed";
    } else if (selectedStatus === "failed") {
      badgeText = selectedErrorCode === INTERRUPTED_ERROR_CODE
        ? "Interrupted"
        : "Failed";
    } else if (state.uncertainEventId !== null &&
               state.uncertainEventId !== undefined) {
      badgeText = "Uncertain";
    } else if (state.eventsStatus === "error" ||
               state.summariesStatus === "error") {
      badgeText = "Error";
    } else if (state.eventsStatus === "loading" ||
               state.summariesStatus === "loading") {
      badgeText = "Loading";
    }

    var findings = [];
    var summaryText = null;
    var coverageText = null;
    var safeErrors = [];
    if (typeof state.panelError === "string" && state.panelError) {
      safeErrors.push(state.panelError);
    }
    if (state.eventsStatus === "error" &&
        typeof state.eventsError === "string" && state.eventsError) {
      safeErrors.push(state.eventsError);
    }
    if (state.summariesStatus === "error" &&
        typeof state.summariesError === "string" && state.summariesError) {
      safeErrors.push(state.summariesError);
    }
    if (selectedDetailErrorText !== null) {
      safeErrors.push(selectedDetailErrorText);
    }
    // Only surface the generic "not loaded" text when no specific detail
    // error was recorded; otherwise the same failure is reported twice.
    if (selectedDetailMissing && selectedDetailErrorText === null) {
      safeErrors.push("Review detail is not loaded.");
    }
    var errorText = safeErrors.length > 0 ? safeErrors.join("\n") : null;
    if (selectedDetail !== null) {
      if (selectedDetail.status === "completed") {
        summaryText = selectedDetail.summary;
        coverageText = selectedDetail.coverage_note;
        findings = selectedDetail.findings;
      } else if (selectedDetail.error_message || selectedDetail.error_code) {
        errorText = errorText || selectedDetail.error_message ||
          selectedDetail.error_code;
      }
    }
    // Runtime guard: a running/pending summary must never render completed
    // detail content, and completed content may only come from a detail
    // that is consistent with its authoritative summary.
    if (selectedStatus === "completed") {
      if (selectedDetail === null || selectedDetail.status !== "completed") {
        summaryText = null;
        coverageText = null;
        findings = [];
      }
    } else {
      summaryText = null;
      coverageText = null;
      findings = [];
    }

    var visible = proposal !== null ||
      selectedSummary !== null ||
      operation !== null ||
      state.uncertainEventId !== null ||
      state.eventsStatus === "error" ||
      state.summariesStatus === "error" ||
      state.eventsStatus === "loading" ||
      state.summariesStatus === "loading";

    return {
      visible: visible,
      badgeText: badgeText,
      workingVisible: workingVisible,
      workingLiveText: workingVisible
        ? "Working on history review..."
        : "",
      summaryText: summaryText,
      coverageText: coverageText,
      findings: findings,
      errorText: errorText,
      proposal: proposal,
      summaries: Array.isArray(state.summaries)
        ? state.summaries.slice()
        : [],
      selectedReviewId: state.selectedReviewId,
      selectedSummary: selectedSummary,
      detail: selectedDetail,
      selectorVisible: Array.isArray(state.summaries) &&
        state.summaries.length > 1,
      startVisible: proposal !== null &&
        operation === null,
      dismissVisible: proposal !== null &&
        operation === null,
      continueVisible: selectedStatus === "pending" &&
        operation === null,
      actionKind: actionKind,
      actionEventId: actionEventId,
      actionReviewId: actionReviewId,
      actionLabel: actionKind === "recheck"
        ? "Recheck"
        : (actionKind === "retry"
          ? "Retry"
          : (actionKind === "reload" ? "Reload" : null)),
      busy: operation !== null ||
        isHistoryReviewTargetBusy(sessionId, state, validateTimestamp),
      collapsed: state.collapsed === true,
    };
  } catch (_) {
    return null;
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
  modeSwitchEventId, acknowledgeRemoteHistory, retryFailed
) {
  if (_isPositiveSafeInteger(modeSwitchEventId) === false) return null;
  if (_isStrictBoolean(acknowledgeRemoteHistory) === false) return null;
  if (retryFailed !== undefined &&
      _isStrictBoolean(retryFailed) === false) {
    return null;
  }
  var payload = {
    mode_switch_event_id: modeSwitchEventId,
    acknowledge_remote_history: acknowledgeRemoteHistory,
  };
  if (retryFailed === true) {
    payload.retry_failed = true;
  }
  return payload;
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

function _operationSnapshotFrom(operation) {
  return {
    targetSessionId: operation.targetSessionId,
    modeSwitchEventId: operation.modeSwitchEventId,
    generation: operation.generation,
  };
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

function createCancelableReviewDialogAdapter(dom) {
  if (_isPlainObject(dom) === false) {
    throw new TypeError("Invalid dialog adapter DOM");
  }
  if (_isPlainObject(dom.dialog) === false ||
      _isPlainObject(dom.bodyEl) === false ||
      _isPlainObject(dom.cancelBtn) === false ||
      _isPlainObject(dom.continueBtn) === false) {
    throw new TypeError("Invalid dialog adapter DOM");
  }

  var pendingCancel = null;
  // Optional: the heading is only rewritten when the dialog exposes one.
  var titleEl = _isPlainObject(dom.titleEl) ? dom.titleEl : null;

  function clearIfSame(cb) {
    if (pendingCancel === cb) pendingCancel = null;
  }

  function addListener(target, name, handler) {
    target.addEventListener(name, handler);
    return function () {
      target.removeEventListener(name, handler);
    };
  }

  return {
    showModal: function () { dom.dialog.showModal(); },
    close: function () { if (dom.dialog.open) dom.dialog.close(); },
    setMessage: function (text) { dom.bodyEl.textContent = text; },
    setTitle: function (text) {
      if (titleEl !== null) titleEl.textContent = text;
    },
    focusInitial: function () { dom.cancelBtn.focus(); },
    isConnected: function () { return dom.dialog.isConnected; },
    onDialogCancel: function (cb) {
      pendingCancel = cb;
      var handler = function () {
        var current = pendingCancel;
        pendingCancel = null;
        if (typeof current === "function") current();
      };
      var unsub = addListener(dom.dialog, "cancel", handler);
      return function () { clearIfSame(cb); unsub(); };
    },
    onCancelClick: function (cb) {
      pendingCancel = cb;
      var handler = function () {
        var current = pendingCancel;
        pendingCancel = null;
        if (typeof current === "function") current();
      };
      var unsub = addListener(dom.cancelBtn, "click", handler);
      return function () { clearIfSame(cb); unsub(); };
    },
    onContinueClick: function (cb) {
      var handler = function () { cb(); };
      var unsub = addListener(dom.continueBtn, "click", handler);
      return function () { unsub(); };
    },
    cancelPending: function () {
      var current = pendingCancel;
      pendingCancel = null;
      if (typeof current === "function") {
        try { current(); } catch (_) { /* must not throw */ }
      }
    },
  };
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

  async function _attemptPost(
    operation, acknowledge, alreadyConfirmed, retryFailed
  ) {
    var payload = buildHistoryReviewCreatePayload(
      operation.modeSwitchEventId, acknowledge, retryFailed === true
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
            confirmed = await confirmRemoteHistory(
              ackMetadata,
              _operationSnapshotFrom(operation),
            );
          } catch (_) {
            return _failedOutcome(operation, null, FAILED_MESSAGE);
          }
          if (confirmed !== true) {
            return _cancelledOutcome(operation);
          }
          return await _attemptPost(
            operation, true, true, retryFailed === true
          );
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

  async function retry(operation) {
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
      return await _attemptPost(captured, false, false, true);
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
    retry: retry,
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
    HISTORY_REVIEW_DIALOG_COPY: HISTORY_REVIEW_DIALOG_COPY,
    INTERRUPTED_ERROR_CODE: INTERRUPTED_ERROR_CODE,
    isValidHistoryReviewStatus: isValidHistoryReviewStatus,
    isValidHistoryReviewVerdict: isValidHistoryReviewVerdict,
    isValidHistoryReviewFinding: isValidHistoryReviewFinding,
    isValidHistoryReviewSummary: isValidHistoryReviewSummary,
    isValidHistoryReviewSummaryList: isValidHistoryReviewSummaryList,
    isValidHistoryReviewDetail: isValidHistoryReviewDetail,
    isValidHistoryReviewEventList: isValidHistoryReviewEventList,
    createCancelableReviewDialogAdapter: createCancelableReviewDialogAdapter,
    historyReviewSummaryFromDetail: historyReviewSummaryFromDetail,
    isHistoryReviewDetailCurrent: isHistoryReviewDetailCurrent,
    historyReviewDetailMayUpdateSummary: historyReviewDetailMayUpdateSummary,
    reconcileHistoryReviewCaches: reconcileHistoryReviewCaches,
    upsertHistoryReviewSummary: upsertHistoryReviewSummary,
    isHistoryReviewTargetBusy: isHistoryReviewTargetBusy,
    isHistoryReviewOperationVisiblyWorking:
      isHistoryReviewOperationVisiblyWorking,
    buildHistoryReviewPanelModel: buildHistoryReviewPanelModel,
    selectLatestHistoryReviewProposal: selectLatestHistoryReviewProposal,
    buildHistoryReviewCreatePayload: buildHistoryReviewCreatePayload,
    parseHistoryReviewAckRequired: parseHistoryReviewAckRequired,
    createHistoryReviewController: createHistoryReviewController,
  };
}
