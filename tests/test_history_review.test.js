/**
 * Tests for frontend/history-review.js — pure History Review logic.
 *
 * Run: node --test tests/test_history_review.test.js
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const {
  isValidHistoryReviewStatus,
  isValidHistoryReviewVerdict,
  isValidHistoryReviewFinding,
  isValidHistoryReviewSummary,
  isValidHistoryReviewSummaryList,
  isValidHistoryReviewDetail,
  selectLatestHistoryReviewProposal,
  buildHistoryReviewCreatePayload,
  parseHistoryReviewAckRequired,
  createHistoryReviewController,
} = require(
  path.resolve(__dirname, "..", "frontend", "history-review.js"),
);

const {
  RECEIVE_TEACHING_MODE,
  CORRECTIVE_MODE,
  isValidModeSwitchEvent,
} = require(
  path.resolve(__dirname, "..", "frontend", "interaction-mode.js"),
);

const { isValidApiTimestamp } = require(
  path.resolve(__dirname, "..", "frontend", "model-selection.js"),
);

function makeSummary(overrides) {
  const value = {
    id: 10,
    session_id: 1,
    mode_switch_event_id: 5,
    status: "completed",
    eligible_message_count: 2,
    source_message_count: 2,
    source_from_message_id: 1,
    source_through_message_id: 2,
    truncated: false,
    summary: "summary text",
    coverage_note: null,
    error_code: null,
    error_message: null,
    findings_count: 1,
    created_at: "2026-08-06T12:00:00",
    started_at: "2026-08-06T12:00:01",
    completed_at: "2026-08-06T12:00:02",
    updated_at: "2026-08-06T12:00:02",
  };
  if (overrides) Object.assign(value, overrides);
  return value;
}

function makeFinding(overrides) {
  const value = {
    seq: 1,
    source_message_id: 1,
    verdict: "correct",
    claim_text: "claim",
    correction_text: null,
    explanation_text: null,
  };
  if (overrides) Object.assign(value, overrides);
  return value;
}

function makeDetail(overrides) {
  const value = Object.assign(makeSummary(), {
    findings_count: 1,
    findings: [makeFinding()],
  });
  if (overrides) Object.assign(value, overrides);
  return value;
}

function makeEvent(overrides) {
  const value = {
    id: 5,
    session_id: 1,
    from_mode: RECEIVE_TEACHING_MODE,
    to_mode: CORRECTIVE_MODE,
    created_at: "2026-08-06T12:00:00",
    history_through_message_id: 2,
    history_boundary_version: "history-boundary-v1",
    reviewable_user_message_count: 2,
    review_supported: true,
  };
  if (overrides) Object.assign(value, overrides);
  return value;
}

function makeAckEnvelope(overrides) {
  const detail = {
    code: "history_review_remote_ack_required",
    message: "Remote history acknowledgement is required.",
    source_message_count: 3,
    reviewer_profile_label: "API Model",
    reviewer_model: "api-model",
    truncated: false,
  };
  if (overrides) Object.assign(detail, overrides);
  return { detail: detail };
}

function proposalOptions(overrides) {
  const options = {
    events: [makeEvent()],
    reviews: [],
    sessionId: 1,
    currentMode: CORRECTIVE_MODE,
    validateTimestamp: isValidApiTimestamp,
    validateModeSwitchEvent: isValidModeSwitchEvent,
  };
  if (overrides) Object.assign(options, overrides);
  return options;
}

function makeControllerDeps(overrides) {
  const detail = makeDetail();
  const deps = {
    postReview: async function () { return detail; },
    fetchReviewSummaries: async function () { return [makeSummary()]; },
    fetchReviewDetail: async function () { return detail; },
    confirmRemoteHistory: async function () { return true; },
    validateTimestamp: isValidApiTimestamp,
  };
  if (overrides) Object.assign(deps, overrides);
  return deps;
}

function makeOperation(overrides) {
  const value = {
    targetSessionId: 1,
    modeSwitchEventId: 5,
    generation: 0,
  };
  if (overrides) Object.assign(value, overrides);
  return value;
}

function httpError(status, code, message, body) {
  return {
    failureKind: "http",
    status: status,
    code: code || null,
    message: message || "error",
    body: body || null,
  };
}

function record(value) {
  return JSON.parse(JSON.stringify(value));
}

/* ------------------------------------------------------------------ */
/* Validators                                                         */
/* ------------------------------------------------------------------ */

describe("history review validators", function () {
  it("accepts valid statuses and verdicts only", function () {
    assert.equal(isValidHistoryReviewStatus("pending"), true);
    assert.equal(isValidHistoryReviewStatus("running"), true);
    assert.equal(isValidHistoryReviewStatus("completed"), true);
    assert.equal(isValidHistoryReviewStatus("failed"), true);
    assert.equal(isValidHistoryReviewStatus("bogus"), false);
    assert.equal(isValidHistoryReviewVerdict("correct"), true);
    assert.equal(isValidHistoryReviewVerdict("incorrect"), true);
    assert.equal(isValidHistoryReviewVerdict("uncertain"), true);
    assert.equal(isValidHistoryReviewVerdict("not_a_claim"), true);
    assert.equal(isValidHistoryReviewVerdict("maybe"), false);
  });

  it("accepts a valid finding and rejects malformed findings", function () {
    assert.equal(isValidHistoryReviewFinding(makeFinding()), true);

    for (const field of [
      "seq", "source_message_id", "verdict",
      "claim_text", "correction_text", "explanation_text",
    ]) {
      const missing = makeFinding();
      delete missing[field];
      assert.equal(isValidHistoryReviewFinding(missing), false, field);
    }

    assert.equal(isValidHistoryReviewFinding(makeFinding({ seq: 0 })), false);
    assert.equal(isValidHistoryReviewFinding(makeFinding({ seq: 1.5 })), false);
    assert.equal(isValidHistoryReviewFinding(makeFinding({ seq: true })), false);
    assert.equal(isValidHistoryReviewFinding(makeFinding({ verdict: "maybe" })), false);
    assert.equal(isValidHistoryReviewFinding(makeFinding({ claim_text: 1 })), false);
    assert.equal(isValidHistoryReviewFinding(makeFinding({ correction_text: 1 })), false);
    assert.equal(isValidHistoryReviewFinding(makeFinding({ explanation_text: [] })), false);

    const inherited = Object.create(makeFinding());
    assert.equal(isValidHistoryReviewFinding(inherited), false);
  });

  it("accepts a valid summary and enforces required own fields", function () {
    assert.equal(
      isValidHistoryReviewSummary(makeSummary(), 1, isValidApiTimestamp),
      true,
    );

    for (const field of Object.keys(makeSummary())) {
      const missing = makeSummary();
      delete missing[field];
      assert.equal(
        isValidHistoryReviewSummary(missing, 1, isValidApiTimestamp),
        false,
        field,
      );
    }

    const inherited = Object.create(makeSummary());
    assert.equal(
      isValidHistoryReviewSummary(inherited, 1, isValidApiTimestamp),
      false,
    );
  });

  it("rejects bad summary field types and values", function () {
    const badValues = [
      { id: 0 },
      { session_id: 2 },
      { mode_switch_event_id: true },
      { status: "bogus" },
      { eligible_message_count: -1 },
      { source_message_count: 1.5 },
      { source_from_message_id: 0 },
      { source_through_message_id: Infinity },
      { truncated: 1 },
      { summary: 1 },
      { coverage_note: [] },
      { error_code: 1 },
      { error_message: {} },
      { findings_count: -1 },
      { created_at: "nope" },
      { updated_at: "" },
      { started_at: "nope" },
      { completed_at: 1 },
    ];
    for (const overrides of badValues) {
      assert.equal(
        isValidHistoryReviewSummary(
          makeSummary(overrides), 1, isValidApiTimestamp,
        ),
        false,
        JSON.stringify(overrides),
      );
    }
    assert.equal(
      isValidHistoryReviewSummary(makeSummary({ id: NaN }), 1),
      false,
    );
    assert.equal(
      isValidHistoryReviewSummary(
        makeSummary({ id: Number.MAX_SAFE_INTEGER + 1 }), 1,
      ),
      false,
    );
  });

  it("validates summary lists without mutating input", function () {
    const first = makeSummary({ id: 10, mode_switch_event_id: 5 });
    const second = makeSummary({ id: 11, mode_switch_event_id: 6 });
    const list = [second, first];
    const before = list.map(function (item) { return item.id; });

    assert.equal(
      isValidHistoryReviewSummaryList(list, 1, isValidApiTimestamp), true,
    );
    assert.deepEqual(list.map(function (item) { return item.id; }), before);
    assert.deepEqual(list.map(function (item) { return item.id; }), [11, 10]);

    assert.equal(
      isValidHistoryReviewSummaryList([first, first], 1), false,
    );
    assert.equal(
      isValidHistoryReviewSummaryList(
        [first, makeSummary({ id: 12, mode_switch_event_id: 5 })], 1,
      ),
      false,
    );
    assert.equal(
      isValidHistoryReviewSummaryList([first], 2), false,
    );
    assert.equal(
      isValidHistoryReviewSummaryList([first, null], 1), false,
    );
  });

  it("accepts a valid detail and rejects count/order mismatches", function () {
    const detail = makeDetail();
    assert.equal(
      isValidHistoryReviewDetail(detail, 1, 10, 5, isValidApiTimestamp),
      true,
    );
    assert.equal(
      isValidHistoryReviewDetail(detail, 1, 11, 5, isValidApiTimestamp),
      false,
    );
    assert.equal(
      isValidHistoryReviewDetail(detail, 1, 10, 6, isValidApiTimestamp),
      false,
    );
    assert.equal(
      isValidHistoryReviewDetail(
        makeDetail({ findings_count: 0 }), 1, 10, 5, isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidHistoryReviewDetail(
        makeDetail({
          findings_count: 2,
          findings: [makeFinding({ seq: 1 }), makeFinding({ seq: 1 })],
        }),
        1, 10, 5, isValidApiTimestamp,
      ),
      false,
    );
    const none = makeDetail({ findings_count: 0, findings: [] });
    assert.equal(
      isValidHistoryReviewDetail(none, 1, 10, 5, isValidApiTimestamp),
      false,
    );

    for (const status of ["pending", "running", "failed"]) {
      const legal = makeDetail({
        status: status,
        findings_count: 0,
        findings: [],
      });
      assert.equal(
        isValidHistoryReviewDetail(legal, 1, 10, 5, isValidApiTimestamp),
        true,
        status,
      );
      const illegal = makeDetail({
        status: status,
        findings_count: 1,
        findings: [makeFinding()],
      });
      assert.equal(
        isValidHistoryReviewDetail(
          illegal, 1, 10, 5, isValidApiTimestamp,
        ),
        false,
        status,
      );
    }
  });

  it("enforces summary findings_count invariants by status", function () {
    assert.equal(
      isValidHistoryReviewSummary(
        makeSummary({ status: "completed", findings_count: 0 }),
        1, isValidApiTimestamp,
      ),
      false,
    );
    for (const status of ["pending", "running", "failed"]) {
      assert.equal(
        isValidHistoryReviewSummary(
          makeSummary({ status: status, findings_count: 0 }),
          1, isValidApiTimestamp,
        ),
        true,
        status,
      );
      assert.equal(
        isValidHistoryReviewSummary(
          makeSummary({ status: status, findings_count: 1 }),
          1, isValidApiTimestamp,
        ),
        false,
        status,
      );
    }
  });

  it("does not mutate detail findings order", function () {
    const detail = makeDetail({
      findings_count: 2,
      findings: [makeFinding({ seq: 1 }), makeFinding({ seq: 2 })],
    });
    const before = detail.findings.map(function (f) { return f.seq; });
    assert.equal(
      isValidHistoryReviewDetail(detail, 1, 10, 5, isValidApiTimestamp),
      true,
    );
    assert.deepEqual(detail.findings.map(function (f) { return f.seq; }), before);
  });

  it("fails closed on throwing getters and proxies", function () {
    const throwing = {};
    Object.defineProperty(throwing, "id", {
      enumerable: true,
      get: function () { throw new Error("boom"); },
    });
    assert.doesNotThrow(function () {
      assert.equal(isValidHistoryReviewSummary(throwing, 1), false);
      assert.equal(isValidHistoryReviewFinding(throwing), false);
      assert.equal(isValidHistoryReviewDetail(throwing, 1, 10, 5), false);
      assert.equal(parseHistoryReviewAckRequired(throwing), null);
    });
  });
});

/* ------------------------------------------------------------------ */
/* Proposal selection                                                 */
/* ------------------------------------------------------------------ */

describe("history review proposal selection", function () {
  it("selects the latest valid unreviewed event and copies values", function () {
    const older = makeEvent({ id: 1, created_at: "2026-08-06T11:00:00" });
    const newer = makeEvent({ id: 2, created_at: "2026-08-06T12:00:00" });
    const events = [older, newer];
    const before = events.map(function (event) { return event.id; });

    const proposal = selectLatestHistoryReviewProposal(
      proposalOptions({ events: events }),
    );

    assert.notEqual(proposal, null);
    assert.equal(proposal.eventId, newer.id);
    assert.equal(proposal.sessionId, 1);
    assert.equal(proposal.reviewableUserMessageCount, 2);
    assert.deepEqual(proposal.event, newer);
    assert.notEqual(proposal.event, newer);
    assert.deepEqual(events.map(function (event) { return event.id; }), before);
  });

  it("uses id desc when created_at ties", function () {
    const first = makeEvent({ id: 1, created_at: "2026-08-06T12:00:00" });
    const second = makeEvent({ id: 2, created_at: "2026-08-06T12:00:00" });
    const proposal = selectLatestHistoryReviewProposal(
      proposalOptions({ events: [first, second] }),
    );
    assert.equal(proposal.eventId, 2);
  });

  it("rejects invalid global arguments and malformed lists", function () {
    assert.equal(selectLatestHistoryReviewProposal(null), null);
    assert.equal(
      selectLatestHistoryReviewProposal(
        proposalOptions({ sessionId: 0 }),
      ),
      null,
    );
    assert.equal(
      selectLatestHistoryReviewProposal(
        proposalOptions({ currentMode: RECEIVE_TEACHING_MODE }),
      ),
      null,
    );
    assert.equal(
      selectLatestHistoryReviewProposal(
        proposalOptions({ events: [makeEvent({ session_id: 2 })] }),
      ),
      null,
    );
    assert.equal(
      selectLatestHistoryReviewProposal(
        proposalOptions({ reviews: [makeSummary({ session_id: 2 })] }),
      ),
      null,
    );
    assert.equal(
      selectLatestHistoryReviewProposal(
        proposalOptions({ reviews: [makeSummary(), makeSummary()] }),
      ),
      null,
    );
  });

  it("skips unsupported, reversed, empty and reviewed events", function () {
    const cases = [
      makeEvent({ review_supported: false }),
      makeEvent({ reviewable_user_message_count: 0 }),
      makeEvent({ history_through_message_id: null }),
      makeEvent({ from_mode: CORRECTIVE_MODE, to_mode: RECEIVE_TEACHING_MODE }),
    ];
    for (const event of cases) {
      assert.equal(
        selectLatestHistoryReviewProposal(
          proposalOptions({ events: [event] }),
        ),
        null,
        JSON.stringify(event),
      );
    }

    const reviewedStatuses = ["pending", "running", "completed", "failed"];
    for (const status of reviewedStatuses) {
      const review = makeSummary({
        mode_switch_event_id: 5,
        status: status,
        findings_count: status === "completed" ? 1 : 0,
      });
      assert.equal(
        selectLatestHistoryReviewProposal(
          proposalOptions({ reviews: [review] }),
        ),
        null,
        status,
      );
    }
  });
});

/* ------------------------------------------------------------------ */
/* Payload and ack parsing                                            */
/* ------------------------------------------------------------------ */

describe("history review payload and ack parsing", function () {
  it("builds the exact create payload with strict inputs", function () {
    assert.deepEqual(
      buildHistoryReviewCreatePayload(5, false),
      {
        mode_switch_event_id: 5,
        acknowledge_remote_history: false,
      },
    );
    assert.deepEqual(
      Object.keys(buildHistoryReviewCreatePayload(5, true)).sort(),
      ["acknowledge_remote_history", "mode_switch_event_id"],
    );
    assert.equal(buildHistoryReviewCreatePayload(0, false), null);
    assert.equal(buildHistoryReviewCreatePayload(5, 1), null);
    assert.equal(buildHistoryReviewCreatePayload(5, "false"), null);
  });

  it("parses a valid ack-required envelope", function () {
    const parsed = parseHistoryReviewAckRequired(makeAckEnvelope());
    assert.deepEqual(parsed, {
      code: "history_review_remote_ack_required",
      message: "Remote history acknowledgement is required.",
      sourceMessageCount: 3,
      reviewerProfileLabel: "API Model",
      reviewerModel: "api-model",
      truncated: false,
    });
    assert.notEqual(parsed, makeAckEnvelope());
  });

  it("does not depend on the message wording", function () {
    const parsed = parseHistoryReviewAckRequired(
      makeAckEnvelope({ message: "另一个安全提示" }),
    );
    assert.notEqual(parsed, null);
    assert.equal(parsed.message, "另一个安全提示");
  });

  it("rejects malformed ack envelopes", function () {
    const badCases = [
      null,
      {},
      { detail: null },
      { detail: [] },
      { detail: { code: "other" } },
      makeAckEnvelope({ code: "history_review_other" }),
      makeAckEnvelope({ source_message_count: 0 }),
      makeAckEnvelope({ source_message_count: 1.5 }),
      makeAckEnvelope({ reviewer_profile_label: "" }),
      makeAckEnvelope({ reviewer_model: "   " }),
      makeAckEnvelope({ truncated: 1 }),
    ];
    for (const value of badCases) {
      assert.equal(parseHistoryReviewAckRequired(value), null);
    }

    const inherited = Object.create(makeAckEnvelope().detail);
    assert.equal(parseHistoryReviewAckRequired({ detail: inherited }), null);
  });
});

/* ------------------------------------------------------------------ */
/* Controller                                                         */
/* ------------------------------------------------------------------ */

describe("history review controller", function () {
  it("starts successfully with ack=false and returns authoritative detail", async function () {
    const posts = [];
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function (sessionId, payload) {
        posts.push({ sessionId: sessionId, payload: payload });
        return makeDetail();
      },
    }));

    const outcome = await controller.start(makeOperation());

    assert.equal(outcome.status, "authoritative");
    assert.equal(outcome.reconciled, false);
    assert.equal(outcome.targetSessionId, 1);
    assert.equal(outcome.modeSwitchEventId, 5);
    assert.equal(outcome.generation, 0);
    assert.deepEqual(outcome.detail, makeDetail());
    assert.equal(posts.length, 1);
    assert.deepEqual(posts[0].payload, {
      mode_switch_event_id: 5,
      acknowledge_remote_history: false,
    });
  });

  it("handles ack-required once, then succeeds with ack=true", async function () {
    const posts = [];
    let confirmCalls = 0;
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function (sessionId, payload) {
        posts.push(payload);
        if (posts.length === 1) {
          throw httpError(
            409,
            "history_review_remote_ack_required",
            "ack",
            makeAckEnvelope(),
          );
        }
        return makeDetail();
      },
      confirmRemoteHistory: async function () {
        confirmCalls += 1;
        return true;
      },
    }));

    const outcome = await controller.start(makeOperation());

    assert.equal(outcome.status, "authoritative");
    assert.equal(outcome.reconciled, false);
    assert.equal(confirmCalls, 1);
    assert.equal(posts.length, 2);
    assert.equal(posts[0].acknowledge_remote_history, false);
    assert.equal(posts[1].acknowledge_remote_history, true);
  });

  it("cancels without a second POST when ack is declined", async function () {
    let postCalls = 0;
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function () {
        postCalls += 1;
        throw httpError(
          409,
          "history_review_remote_ack_required",
          "ack",
          makeAckEnvelope(),
        );
      },
      confirmRemoteHistory: async function () { return false; },
    }));

    const outcome = await controller.start(makeOperation());

    assert.equal(outcome.status, "cancelled");
    assert.equal(postCalls, 1);
  });

  it("does not send a third POST after a second ack-required", async function () {
    let postCalls = 0;
    let confirmCalls = 0;
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function () {
        postCalls += 1;
        throw httpError(
          409,
          "history_review_remote_ack_required",
          "ack",
          makeAckEnvelope(),
        );
      },
      confirmRemoteHistory: async function () {
        confirmCalls += 1;
        return true;
      },
    }));

    const outcome = await controller.start(makeOperation());

    assert.equal(outcome.status, "failed");
    assert.equal(postCalls, 2);
    assert.equal(confirmCalls, 1);
  });

  it("reconciles ambiguous POST outcomes without a second POST", async function () {
    let postCalls = 0;
    let listCalls = 0;
    let detailCalls = 0;
    const cases = [
      { failureKind: "network", status: 0, code: null, message: "offline", body: null },
      { failureKind: "response_parse", status: 0, code: null, message: "bad", body: null },
      httpError(503, null, "upstream"),
      httpError(409, "history_review_execution_conflict", "conflict"),
    ];
    for (const error of cases) {
      postCalls = 0;
      listCalls = 0;
      detailCalls = 0;
      const controller = createHistoryReviewController(makeControllerDeps({
        postReview: async function () {
          postCalls += 1;
          throw error;
        },
        fetchReviewSummaries: async function () {
          listCalls += 1;
          return [makeSummary()];
        },
        fetchReviewDetail: async function () {
          detailCalls += 1;
          return makeDetail();
        },
      }));
      const outcome = await controller.start(makeOperation());
      assert.equal(outcome.status, "authoritative", JSON.stringify(error));
      assert.equal(outcome.reconciled, true);
      assert.equal(postCalls, 1);
      assert.equal(listCalls, 1);
      assert.equal(detailCalls, 1);
    }
  });

  it("reconciles an invalid 2xx body", async function () {
    let listCalls = 0;
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function () { return { id: 999 }; },
      fetchReviewSummaries: async function () {
        listCalls += 1;
        return [makeSummary()];
      },
      fetchReviewDetail: async function () { return makeDetail(); },
    }));

    const outcome = await controller.start(makeOperation());

    assert.equal(outcome.status, "authoritative");
    assert.equal(outcome.reconciled, true);
    assert.equal(listCalls, 1);
  });

  it("returns uncertain when reconciliation cannot find the review", async function () {
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function () {
        throw { failureKind: "network", status: 0, message: "offline" };
      },
      fetchReviewSummaries: async function () {
        return [makeSummary({ mode_switch_event_id: 6 })];
      },
    }));

    const outcome = await controller.start(makeOperation());
    assert.equal(outcome.status, "uncertain");
    assert.match(outcome.message, /Recheck/);
  });

  it("returns uncertain when reconciliation detail mismatches", async function () {
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function () {
        throw { failureKind: "network", status: 0, message: "offline" };
      },
      fetchReviewSummaries: async function () { return [makeSummary()]; },
      fetchReviewDetail: async function () {
        return makeDetail({ mode_switch_event_id: 6 });
      },
    }));

    const outcome = await controller.start(makeOperation());
    assert.equal(outcome.status, "uncertain");
  });

  it("handles deterministic HTTP 4xx without reconciliation", async function () {
    let listCalls = 0;
    const outcomes = [];
    const errors = [
      httpError(422, "history_review_source_message_too_large", "too large"),
      httpError(404, "history_review_session_not_found", "gone"),
    ];
    for (const error of errors) {
      listCalls = 0;
      const controller = createHistoryReviewController(makeControllerDeps({
        postReview: async function () { throw error; },
        fetchReviewSummaries: async function () {
          listCalls += 1;
          return [];
        },
      }));
      outcomes.push(await controller.start(makeOperation()));
      assert.equal(listCalls, 0);
    }
    assert.equal(outcomes[0].status, "failed");
    assert.equal(outcomes[0].code, "history_review_source_message_too_large");
    assert.equal(outcomes[1].status, "session_not_found");
  });

  it("recheck only performs GET and never POST", async function () {
    let postCalls = 0;
    let listCalls = 0;
    let detailCalls = 0;
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function () { postCalls += 1; return makeDetail(); },
      fetchReviewSummaries: async function () {
        listCalls += 1;
        return [makeSummary()];
      },
      fetchReviewDetail: async function () {
        detailCalls += 1;
        return makeDetail();
      },
    }));

    const outcome = await controller.recheck(makeOperation());

    assert.equal(outcome.status, "authoritative");
    assert.equal(outcome.reconciled, true);
    assert.equal(postCalls, 0);
    assert.equal(listCalls, 1);
    assert.equal(detailCalls, 1);
  });

  it("accepts each authoritative review status in recheck", async function () {
    for (const status of ["pending", "running", "completed", "failed"]) {
      const summary = makeSummary({
        status: status,
        findings_count: status === "completed" ? 1 : 0,
      });
      const detail = status === "completed"
        ? makeDetail()
        : makeDetail({
            status: status,
            findings_count: 0,
            findings: [],
          });
      const controller = createHistoryReviewController(makeControllerDeps({
        fetchReviewSummaries: async function () { return [summary]; },
        fetchReviewDetail: async function () { return detail; },
      }));
      const outcome = await controller.recheck(makeOperation());
      assert.equal(outcome.status, "authoritative", status);
      assert.equal(outcome.detail.status, status);
    }
  });

  it("isolates busy state per session on one controller", async function () {
    let releaseFirst;
    const firstGate = new Promise(function (resolve) {
      releaseFirst = resolve;
    });
    const posts = [];
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function (sessionId, payload) {
        posts.push({ sessionId: sessionId, payload: payload });
        if (sessionId === 1) {
          await firstGate;
        }
        return makeDetail({ session_id: sessionId });
      },
    }));

    const firstPromise = controller.start(makeOperation({
      targetSessionId: 1,
      modeSwitchEventId: 5,
      generation: 1,
    }));
    const second = await controller.start(makeOperation({
      targetSessionId: 2,
      modeSwitchEventId: 5,
      generation: 2,
    }));
    const duplicate = await controller.start(makeOperation({
      targetSessionId: 1,
      modeSwitchEventId: 5,
      generation: 3,
    }));

    assert.equal(second.status, "authoritative");
    assert.equal(second.targetSessionId, 2);
    assert.equal(second.modeSwitchEventId, 5);
    assert.equal(second.generation, 2);
    assert.equal(second.detail.session_id, 2);

    assert.equal(duplicate.status, "busy");
    assert.equal(duplicate.targetSessionId, 1);
    assert.equal(duplicate.modeSwitchEventId, 5);
    assert.equal(duplicate.generation, 3);
    assert.equal(controller.isActive(1), true);
    assert.equal(controller.isActive(2), false);

    releaseFirst();
    const first = await firstPromise;
    assert.equal(first.status, "authoritative");
    assert.equal(first.targetSessionId, 1);
    assert.equal(first.modeSwitchEventId, 5);
    assert.equal(first.generation, 1);
    assert.equal(first.detail.session_id, 1);
    assert.equal(controller.isActive(1), false);
    assert.equal(controller.isActive(2), false);
  });

  it("releases active state after dependency throws", async function () {
    let calls = 0;
    const controller = createHistoryReviewController(makeControllerDeps({
      postReview: async function () {
        calls += 1;
        if (calls === 1) throw new Error("programmer error");
        return makeDetail();
      },
    }));

    const first = await controller.start(makeOperation());
    assert.equal(first.status, "failed");
    assert.equal(controller.isActive(1), false);

    const second = await controller.start(makeOperation());
    assert.equal(second.status, "authoritative");
  });

  it("is total for hostile operations and never rejects", async function () {
    const controller = createHistoryReviewController(makeControllerDeps());
    const hostile = {};
    Object.defineProperty(hostile, "targetSessionId", {
      enumerable: true,
      get: function () { throw new Error("boom"); },
    });
    let outcome;
    await assert.doesNotReject(async function () {
      outcome = await controller.start(hostile);
    });
    assert.equal(outcome.status, "invalid_request");
  });

  it("does not mutate the operation object", async function () {
    const operation = makeOperation();
    const before = record(operation);
    const controller = createHistoryReviewController(makeControllerDeps());
    await controller.start(operation);
    assert.deepEqual(operation, before);
  });
});

/* ------------------------------------------------------------------ */
/* Recovery hardening and revoked-proxy fail-closed coverage          */
/* ------------------------------------------------------------------ */

describe("history review controller recovery hardening", function () {
  it("only the exact session_not_found 404 code clears the session", async function () {
    const cases = [
      { code: "history_review_session_not_found", expected: "session_not_found" },
      { code: "other_error", expected: "uncertain" },
      { code: null, expected: "uncertain" },
      { code: 123, expected: "uncertain" },
    ];

    for (const item of cases) {
      let postCalls = 0;
      let listCalls = 0;
      const controller = createHistoryReviewController(makeControllerDeps({
        postReview: async function () {
          postCalls += 1;
          throw {
            failureKind: "network",
            status: 0,
            code: null,
            message: "offline",
            body: null,
          };
        },
        fetchReviewSummaries: async function () {
          listCalls += 1;
          throw {
            failureKind: "http",
            status: 404,
            code: item.code,
            message: "not found",
            body: { secret: "must not leak" },
          };
        },
      }));

      const outcome = await controller.start(makeOperation());

      assert.equal(outcome.status, item.expected, JSON.stringify(item));
      assert.equal(postCalls, 1);
      assert.equal(listCalls, 1);
      assert.equal(JSON.stringify(outcome).indexOf("must not leak"), -1);
    }
  });

  it("never confirms when the top-level ack code is missing or inconsistent", async function () {
    const cases = [
      { code: null, body: makeAckEnvelope() },
      { code: "other_error", body: makeAckEnvelope() },
      {
        code: "history_review_remote_ack_required",
        body: {
          detail: Object.assign(makeAckEnvelope().detail, {
            code: "other_error",
          }),
        },
      },
      {
        code: "history_review_remote_ack_required",
        body: {},
      },
    ];

    for (const item of cases) {
      let postCalls = 0;
      let confirmCalls = 0;
      let listCalls = 0;
      const controller = createHistoryReviewController(makeControllerDeps({
        postReview: async function () {
          postCalls += 1;
          throw {
            failureKind: "http",
            status: 409,
            code: item.code,
            message: "ack",
            body: item.body,
          };
        },
        confirmRemoteHistory: async function () {
          confirmCalls += 1;
          return true;
        },
        fetchReviewSummaries: async function () {
          listCalls += 1;
          return [];
        },
      }));

      const outcome = await controller.start(makeOperation());

      assert.equal(outcome.status, "failed", JSON.stringify(item));
      assert.equal(postCalls, 1);
      assert.equal(confirmCalls, 0);
      assert.equal(listCalls, 0);
    }
  });

  it("reconciles ambiguous second POST with exactly two posts", async function () {
    const secondErrors = [
      {
        failureKind: "network",
        status: 0,
        code: null,
        message: "offline",
        body: null,
      },
      {
        failureKind: "response_parse",
        status: 0,
        code: null,
        message: "bad response",
        body: null,
      },
      httpError(503, null, "upstream"),
      httpError(409, "history_review_execution_conflict", "conflict"),
    ];

    for (const secondError of secondErrors) {
      let postCalls = 0;
      let confirmCalls = 0;
      let listCalls = 0;
      let detailCalls = 0;
      const controller = createHistoryReviewController(makeControllerDeps({
        postReview: async function () {
          postCalls += 1;
          if (postCalls === 1) {
            throw httpError(
              409,
              "history_review_remote_ack_required",
              "ack",
              makeAckEnvelope(),
            );
          }
          throw secondError;
        },
        confirmRemoteHistory: async function () {
          confirmCalls += 1;
          return true;
        },
        fetchReviewSummaries: async function () {
          listCalls += 1;
          return [makeSummary()];
        },
        fetchReviewDetail: async function () {
          detailCalls += 1;
          return makeDetail();
        },
      }));

      const outcome = await controller.start(makeOperation());

      assert.equal(
        outcome.status, "authoritative", JSON.stringify(secondError),
      );
      assert.equal(outcome.reconciled, true);
      assert.equal(postCalls, 2, JSON.stringify(secondError));
      assert.equal(confirmCalls, 1, JSON.stringify(secondError));
      assert.equal(listCalls, 1, JSON.stringify(secondError));
      assert.equal(detailCalls, 1, JSON.stringify(secondError));
    }
  });
});

describe("revoked Proxy hardening", function () {
  it("validators, proposal selector and ack parser fail closed", function () {
    const revocable = Proxy.revocable({}, {});
    revocable.revoke();
    const hostile = revocable.proxy;

    assert.doesNotThrow(function () {
      assert.equal(
        isValidHistoryReviewSummary(hostile, 1, isValidApiTimestamp),
        false,
      );
      assert.equal(isValidHistoryReviewFinding(hostile), false);
      assert.equal(
        isValidHistoryReviewDetail(
          hostile, 1, 10, 5, isValidApiTimestamp,
        ),
        false,
      );
      assert.equal(parseHistoryReviewAckRequired(hostile), null);
      assert.equal(
        selectLatestHistoryReviewProposal({
          events: [hostile],
          reviews: [],
          sessionId: 1,
          currentMode: CORRECTIVE_MODE,
          validateTimestamp: isValidApiTimestamp,
          validateModeSwitchEvent: isValidModeSwitchEvent,
        }),
        null,
      );
      assert.equal(
        selectLatestHistoryReviewProposal(hostile),
        null,
      );
    });
  });

  it("controller start and recheck never reject on revoked operations", async function () {
    const controller = createHistoryReviewController(makeControllerDeps());
    const revocable = Proxy.revocable({}, {});
    revocable.revoke();
    const hostile = revocable.proxy;

    let startOutcome;
    await assert.doesNotReject(async function () {
      startOutcome = await controller.start(hostile);
    });
    assert.equal(startOutcome.status, "invalid_request");

    let recheckOutcome;
    await assert.doesNotReject(async function () {
      recheckOutcome = await controller.recheck(hostile);
    });
    assert.equal(recheckOutcome.status, "invalid_request");
  });
});
