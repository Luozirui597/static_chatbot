/**
 * Tests for session-profile-switch.js — controller, confirmer and
 * pure planning helpers.
 *
 * Run:  node --test tests/test_session_profile_switch.test.js
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const {
  historyStateFromValue,
  captureRequestedProfile,
  planSessionCacheUpdate,
  preserveSelectedProfileWithoutFallback,
  classifyUncertainRefresh,
  planSwitchOutcomeEffects,
  createSessionProfileSwitchController,
  createRemoteHistoryConfirmer,
} = require(
  path.resolve(__dirname, "..", "frontend", "session-profile-switch.js"),
);

const {
  analyzeProfileRegistry,
  isValidSessionResponse,
} = require(
  path.resolve(__dirname, "..", "frontend", "model-selection.js"),
);

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

function makeProfile(id, extra) {
  var p = {
    id: id,
    label: "Label " + id,
    kind: "api",
    model: "model-" + id,
    is_default: false,
  };
  if (extra) {
    Object.keys(extra).forEach(function (k) { p[k] = extra[k]; });
  }
  return p;
}

function validRegistry() {
  return analyzeProfileRegistry([
    makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
    makeProfile("local", { kind: "local", model: "qwen" }),
  ]);
}

function makeSession(extra) {
  var s = {
    id: 1,
    title: "Test",
    created_at: "2026-08-06T12:00:00",
    updated_at: "2026-08-06T12:00:00",
    llm_profile_id: "default",
    llm_profile_label: "Default",
    llm_profile_status: "ready",
    llm_model_snapshot: "m1",
  };
  if (extra) {
    Object.keys(extra).forEach(function (k) { s[k] = extra[k]; });
  }
  return s;
}

function makeOperation(extra) {
  var op = {
    generation: 1,
    targetSessionId: 1,
    originalProfileId: "default",
    originalModelSnapshot: "m1",
    requestedProfile: { id: "local", kind: "local", model: "qwen" },
    needsConfirmHint: false,
  };
  if (extra) {
    Object.keys(extra).forEach(function (k) { op[k] = extra[k]; });
  }
  return op;
}

// Fake dependencies for the controller.
function makeControllerEnv(options) {
  options = options || {};
  var patchResults = options.patchResults || [];
  var getResults = options.getResults || [];
  var confirmResults = options.confirmResults || [];
  var validate = options.validate || isValidSessionResponse;

  var env = {
    patchCalls: [],
    getCalls: 0,
    confirmCalls: [],
  };

  async function patchSwitch(sessionId, profileId, ack) {
    env.patchCalls.push({ sessionId, profileId, ack });
    var entry = patchResults.shift();
    if (!entry) throw new Error("unexpected extra patch call");
    if (entry.type === "error") throw entry.err;
    return entry.value;
  }

  async function fetchOneSession(sessionId) {
    env.getCalls += 1;
    var entry = getResults.shift();
    if (!entry) throw new Error("unexpected extra GET call");
    if (entry.type === "error") throw entry.err;
    return entry.value;
  }

  async function confirmRemoteHistory(message) {
    env.confirmCalls.push(message);
    var entry = confirmResults.shift();
    if (entry === undefined) return false;
    if (typeof entry === "function") return entry();
    return entry;
  }

  env.controller = createSessionProfileSwitchController({
    patchSwitch: patchSwitch,
    fetchOneSession: fetchOneSession,
    confirmRemoteHistory: confirmRemoteHistory,
    validateSessionResponse: validate,
  });

  return env;
}

function httpError(status, code, message) {
  return {
    failureKind: "http",
    status: status,
    code: code || null,
    message: message || "error",
  };
}

function networkError() {
  return { failureKind: "network", status: 0, code: null, message: "down" };
}

function parseError() {
  return {
    failureKind: "response_parse", status: 0, code: null, message: "bad json",
  };
}

function makeFakeDialogAdapter(overrides) {
  var adapter = {
    open: false,
    message: null,
    focused: 0,
    closeCalls: 0,
    listeners: { cancel: [], cancelClick: [], continueClick: [] },
    listenerCount: function () {
      return this.listeners.cancel.length +
        this.listeners.cancelClick.length +
        this.listeners.continueClick.length;
    },
    showModal: function () { this.open = true; },
    close: function () { this.open = false; this.closeCalls += 1; },
    setMessage: function (text) { this.message = text; },
    focusInitial: function () { this.focused += 1; },
    isConnected: function () { return true; },
    onDialogCancel: function (cb) {
      this.listeners.cancel.push(cb);
      return function () { this.listeners.cancel = []; }.bind(this);
    },
    onCancelClick: function (cb) {
      this.listeners.cancelClick.push(cb);
      return function () { this.listeners.cancelClick = []; }.bind(this);
    },
    onContinueClick: function (cb) {
      this.listeners.continueClick.push(cb);
      return function () { this.listeners.continueClick = []; }.bind(this);
    },
  };
  if (overrides) {
    Object.keys(overrides).forEach(function (k) { adapter[k] = overrides[k]; });
  }
  return adapter;
}

function fireDialog(adapter, which) {
  var cbs = adapter.listeners[which];
  var cb = cbs[cbs.length - 1];
  cb();
}

// -----------------------------------------------------------------------
// historyStateFromValue
// -----------------------------------------------------------------------

describe("historyStateFromValue", function () {
  it("maps true/false/null/undefined", function () {
    assert.equal(historyStateFromValue(true), "present");
    assert.equal(historyStateFromValue(false), "empty");
    assert.equal(historyStateFromValue(null), "unknown");
    assert.equal(historyStateFromValue(undefined), "unknown");
  });

  it("fails closed to unknown for anything else", function () {
    ["present", "empty", 0, 1, "", {}, []].forEach(function (v) {
      assert.equal(historyStateFromValue(v), "unknown");
    });
  });
});

// -----------------------------------------------------------------------
// captureRequestedProfile
// -----------------------------------------------------------------------

describe("captureRequestedProfile", function () {
  it("returns a pure-value snapshot for a matching id", function () {
    var reg = validRegistry();
    var snap = captureRequestedProfile(reg, "local");
    assert.deepStrictEqual(snap, { id: "local", kind: "local", model: "qwen" });
  });

  it("returns null for missing ids and invalid registries", function () {
    assert.equal(captureRequestedProfile(validRegistry(), "gone"), null);
    assert.equal(captureRequestedProfile(null, "local"), null);
    assert.equal(captureRequestedProfile("x", "local"), null);
    assert.equal(
      captureRequestedProfile(
        analyzeProfileRegistry([makeProfile("a"), makeProfile("b")]),
        "local",
      ),
      null,
    );
  });

  it("snapshot is detached from later registry mutation", function () {
    var profiles = [
      makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
      makeProfile("local", { kind: "local", model: "qwen" }),
    ];
    var reg = analyzeProfileRegistry(profiles);
    var snap = captureRequestedProfile(reg, "local");
    profiles[1].model = "mutated";
    reg.profiles[1].model = "mutated";
    assert.deepStrictEqual(snap, { id: "local", kind: "local", model: "qwen" });
  });
});

// -----------------------------------------------------------------------
// preserveSelectedProfileWithoutFallback
// -----------------------------------------------------------------------

describe("preserveSelectedProfileWithoutFallback", function () {
  it("keeps a still-existing selection", function () {
    assert.equal(
      preserveSelectedProfileWithoutFallback(validRegistry(), "local"),
      "local",
    );
  });

  it("returns null when the selection disappeared (no fallback)", function () {
    assert.equal(
      preserveSelectedProfileWithoutFallback(validRegistry(), "gone"),
      null,
    );
    assert.equal(
      preserveSelectedProfileWithoutFallback(validRegistry(), null),
      null,
    );
    assert.equal(
      preserveSelectedProfileWithoutFallback("bad", "local"),
      null,
    );
  });
});

// -----------------------------------------------------------------------
// planSessionCacheUpdate
// -----------------------------------------------------------------------

describe("planSessionCacheUpdate", function () {
  var base = [
    makeSession({ id: 1, updated_at: "2026-08-06T10:00:00" }),
    makeSession({ id: 2, updated_at: "2026-08-06T12:00:00" }),
    makeSession({ id: 3, updated_at: "2026-08-06T11:00:00" }),
  ];

  it("replaces only the same id and sorts by updated_at desc, id desc", function () {
    var fresh = makeSession({
      id: 1, updated_at: "2026-08-06T09:00:00", title: "fresh",
    });
    var plan = planSessionCacheUpdate({
      sessions: base, requestedSessionId: 1, fresh: fresh,
    });
    assert.equal(plan.kind, "replace");
    assert.equal(plan.sessions.length, 3);
    assert.deepStrictEqual(
      plan.sessions.map(function (s) { return s.id; }),
      [2, 3, 1],
    );
    assert.equal(plan.newIndex, 2);
    assert.equal(plan.sessions[2].title, "fresh");
  });

  it("sorts equal timestamps by id desc", function () {
    var same = [
      makeSession({ id: 1, updated_at: "2026-08-06T10:00:00" }),
      makeSession({ id: 2, updated_at: "2026-08-06T10:00:00" }),
    ];
    var fresh = makeSession({ id: 1, updated_at: "2026-08-06T10:00:00" });
    var plan = planSessionCacheUpdate({
      sessions: same, requestedSessionId: 1, fresh: fresh,
    });
    assert.deepStrictEqual(
      plan.sessions.map(function (s) { return s.id; }),
      [2, 1],
    );
    assert.equal(plan.newIndex, 1);
  });

  it("never inserts a missing target", function () {
    var plan = planSessionCacheUpdate({
      sessions: base,
      requestedSessionId: 99,
      fresh: makeSession({ id: 99 }),
    });
    assert.deepStrictEqual(plan, { kind: "noop", reason: "missing_target" });
  });

  it("rejects an id mismatch", function () {
    var plan = planSessionCacheUpdate({
      sessions: base,
      requestedSessionId: 1,
      fresh: makeSession({ id: 2 }),
    });
    assert.deepStrictEqual(plan, { kind: "noop", reason: "id_mismatch" });
  });

  it("fails closed on any unparseable timestamp (invalid SessionResponse)", function () {
    // An unparseable updated_at makes the entry an invalid
    // SessionResponse, so the whole list is rejected.
    var bad = base.slice();
    bad.push(makeSession({ id: 4, updated_at: "not-a-date" }));
    var plan = planSessionCacheUpdate({
      sessions: bad,
      requestedSessionId: 1,
      fresh: makeSession({ id: 1 }),
    });
    assert.deepStrictEqual(plan, { kind: "noop", reason: "invalid_list" });
  });

  it("fails closed on malformed inputs (invalid_list)", function () {
    assert.deepStrictEqual(
      planSessionCacheUpdate(null),
      { kind: "noop", reason: "invalid_list" },
    );
    assert.deepStrictEqual(
      planSessionCacheUpdate({ sessions: "x", requestedSessionId: 1,
        fresh: makeSession({ id: 1 }) }),
      { kind: "noop", reason: "invalid_list" },
    );
  });

  it("rejects a list containing an incomplete SessionResponse", function () {
    var bad = [
      makeSession({ id: 1, updated_at: "2026-08-06T10:00:00" }),
      { id: 2, updated_at: "2026-08-06T12:00:00" },
    ];
    var plan = planSessionCacheUpdate({
      sessions: bad,
      requestedSessionId: 1,
      fresh: makeSession({ id: 1 }),
    });
    assert.deepStrictEqual(plan, { kind: "noop", reason: "invalid_list" });
  });

  it("rejects a list with duplicate session ids", function () {
    var dup = [
      makeSession({ id: 1, updated_at: "2026-08-06T10:00:00" }),
      makeSession({ id: 1, updated_at: "2026-08-06T11:00:00", title: "dup" }),
    ];
    var plan = planSessionCacheUpdate({
      sessions: dup,
      requestedSessionId: 1,
      fresh: makeSession({ id: 1 }),
    });
    assert.deepStrictEqual(plan, { kind: "noop", reason: "invalid_list" });
  });

  it("does not mutate the input array or elements", function () {
    var input = [
      makeSession({ id: 1, updated_at: "2026-08-06T10:00:00" }),
      makeSession({ id: 2, updated_at: "2026-08-06T12:00:00" }),
    ];
    var json = JSON.stringify(input);
    planSessionCacheUpdate({
      sessions: input,
      requestedSessionId: 1,
      fresh: makeSession({ id: 1, updated_at: "2026-08-06T09:00:00" }),
    });
    assert.equal(JSON.stringify(input), json);
  });
});

// -----------------------------------------------------------------------
// classifyUncertainRefresh
// -----------------------------------------------------------------------

describe("classifyUncertainRefresh", function () {
  var record = {
    generation: 7,
    requestedProfile: { id: "local", kind: "local", model: "qwen" },
    originalProfileId: "default",
    originalModelSnapshot: "m1",
  };

  function opts(overrides) {
    var base = {
      targetSessionId: 1,
      uncertainRecord: record,
      fresh: makeSession({
        llm_profile_id: "local", llm_model_snapshot: "qwen",
      }),
    };
    if (overrides) {
      Object.keys(overrides).forEach(function (k) { base[k] = overrides[k]; });
    }
    return base;
  }

  it("confirmed_target when the TARGET session is bound to the requested profile and ready", function () {
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts()),
      { status: "confirmed_target" },
    );
  });

  it("different_binding for the original binding", function () {
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts({
        fresh: makeSession({
          llm_profile_id: "default", llm_model_snapshot: "m1",
        }),
      })),
      { status: "different_binding" },
    );
  });

  it("different_binding for a third binding", function () {
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts({
        fresh: makeSession({
          llm_profile_id: "other", llm_model_snapshot: "m3",
        }),
      })),
      { status: "different_binding" },
    );
  });

  it("different_binding when id matches but model does not", function () {
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts({
        fresh: makeSession({
          llm_profile_id: "local", llm_model_snapshot: "stale",
        }),
      })),
      { status: "different_binding" },
    );
  });

  it("invalid when fresh belongs to a DIFFERENT session id", function () {
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts({
        fresh: makeSession({
          id: 2,
          llm_profile_id: "local", llm_model_snapshot: "qwen",
        }),
      })),
      { status: "invalid" },
    );
  });

  it("invalid for malformed records, ids or responses", function () {
    assert.deepStrictEqual(
      classifyUncertainRefresh(null),
      { status: "invalid" },
    );
    assert.deepStrictEqual(
      classifyUncertainRefresh({
        targetSessionId: 1, uncertainRecord: null, fresh: makeSession(),
      }),
      { status: "invalid" },
    );
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts({ fresh: null })),
      { status: "invalid" },
    );
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts({
        targetSessionId: "1",
      })),
      { status: "invalid" },
    );
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts({
        uncertainRecord: {
          generation: 1,
          requestedProfile: { id: "", kind: "local", model: "qwen" },
          originalProfileId: "default",
          originalModelSnapshot: "m1",
        },
      })),
      { status: "invalid" },
    );
    assert.deepStrictEqual(
      classifyUncertainRefresh(opts({
        uncertainRecord: {
          generation: 1,
          requestedProfile: { id: "local", kind: "local", model: "qwen" },
          originalProfileId: 42,
          originalModelSnapshot: "m1",
        },
      })),
      { status: "invalid" },
    );
  });
});

// -----------------------------------------------------------------------
// planSwitchOutcomeEffects
// -----------------------------------------------------------------------

describe("planSwitchOutcomeEffects", function () {
  var switched = {
    status: "switched",
    session: makeSession({ id: 1, llm_profile_id: "local",
      llm_model_snapshot: "qwen" }),
    reconciled: false,
  };

  function opts(overrides) {
    var base = {
      outcome: switched,
      operation: makeOperation(),
      targetSessionId: 1,
      currentSessionId: 1,
      hasTarget: true,
    };
    if (overrides) {
      Object.keys(overrides).forEach(function (k) { base[k] = overrides[k]; });
    }
    return base;
  }

  it("switched updates cache, clears block and uncertain, focuses input", function () {
    var plan = planSwitchOutcomeEffects(opts());
    assert.equal(plan.kind, "apply");
    assert.equal(plan.updateCache, true);
    assert.equal(plan.clearBlock, true);
    assert.equal(plan.clearUncertain, true);
    assert.equal(plan.focus, "input");
    assert.equal(plan.syncVisibleUI, true);
  });

  it("not_changed also clears the block and focuses apply", function () {
    var plan = planSwitchOutcomeEffects(opts({
      outcome: { status: "not_changed", session: makeSession({ id: 1 }) },
    }));
    assert.equal(plan.updateCache, true);
    assert.equal(plan.clearBlock, true);
    assert.equal(plan.clearUncertain, true);
    assert.equal(plan.focus, "apply");
  });

  it("malformed switched session clears NOTHING (fail closed)", function () {
    var plan = planSwitchOutcomeEffects(opts({
      outcome: { status: "switched", session: { broken: true }, reconciled: false },
    }));
    assert.equal(plan.kind, "ignore");
    assert.equal(plan.clearBlock, false);
    assert.equal(plan.clearUncertain, false);
    assert.equal(plan.updateCache, false);
    assert.equal(plan.focus, null);
    assert.equal(plan.syncVisibleUI, false);
  });

  it("switched with an id-mismatched session clears NOTHING", function () {
    var plan = planSwitchOutcomeEffects(opts({
      outcome: {
        status: "switched",
        session: makeSession({ id: 2 }),
        reconciled: false,
      },
    }));
    assert.equal(plan.kind, "ignore");
    assert.equal(plan.clearBlock, false);
    assert.equal(plan.clearUncertain, false);
  });

  it("malformed not_changed session clears NOTHING", function () {
    var plan = planSwitchOutcomeEffects(opts({
      outcome: { status: "not_changed", session: null },
    }));
    assert.equal(plan.kind, "ignore");
    assert.equal(plan.clearBlock, false);
    assert.equal(plan.clearUncertain, false);
  });

  it("cancelled keeps the block and focuses apply", function () {
    var plan = planSwitchOutcomeEffects(opts({
      outcome: { status: "cancelled" },
    }));
    assert.equal(plan.clearBlock, false);
    assert.equal(plan.focus, "apply");
  });

  it("not_found removes the session and cleans state", function () {
    var plan = planSwitchOutcomeEffects(opts({
      outcome: { status: "not_found" },
    }));
    assert.equal(plan.removeSession, true);
    assert.equal(plan.clearBlock, true);
    assert.equal(plan.clearUncertain, true);
  });

  it("validation_error is the only outcome with reloadProfiles", function () {
    var plan = planSwitchOutcomeEffects(opts({
      outcome: { status: "validation_error", message: "bad" },
    }));
    assert.equal(plan.reloadProfiles, true);
    assert.equal(plan.showStatus, "bad");

    ["failed", "uncertain", "cancelled", "busy", "not_found",
     "not_changed", "switched"].forEach(function (status) {
      var outcome = { status: status };
      if (status === "not_changed") outcome.session = makeSession({ id: 1 });
      if (status === "switched") {
        outcome.session = makeSession({ id: 1 });
        outcome.reconciled = false;
      }
      if (status === "uncertain") {
        outcome.message = "m";
        outcome.requestedProfile = { id: "local", kind: "local", model: "qwen" };
      }
      if (status === "failed") outcome.message = "m";
      var p = planSwitchOutcomeEffects(opts({ outcome: outcome }));
      assert.equal(p.reloadProfiles, false, status);
    });
  });

  it("uncertain produces a complete independent record and keeps the block", function () {
    var op = makeOperation();
    var plan = planSwitchOutcomeEffects(opts({
      operation: op,
      outcome: {
        status: "uncertain", message: "m",
        requestedProfile: { id: "local", kind: "local", model: "qwen" },
      },
    }));
    assert.deepStrictEqual(plan.uncertainRecord, {
      generation: op.generation,
      requestedProfile: { id: "local", kind: "local", model: "qwen" },
      originalProfileId: op.originalProfileId,
      originalModelSnapshot: op.originalModelSnapshot,
    });
    // independence: mutating the plan record must not touch the op
    plan.uncertainRecord.requestedProfile.model = "mutated";
    assert.equal(op.requestedProfile.model, "qwen");
    assert.equal(plan.clearBlock, false);
    assert.equal(plan.focus, "apply");
  });

  it("uncertain with an invalid operation is ignored", function () {
    var plan = planSwitchOutcomeEffects(opts({
      operation: null,
      outcome: {
        status: "uncertain", message: "m",
        requestedProfile: { id: "local", kind: "local", model: "qwen" },
      },
    }));
    assert.equal(plan.kind, "ignore");
  });

  it("failed and busy never clear the block", function () {
    assert.equal(
      planSwitchOutcomeEffects(opts({
        outcome: { status: "failed", message: "m" },
      })).clearBlock,
      false,
    );
    assert.equal(
      planSwitchOutcomeEffects(opts({ outcome: { status: "busy" } })).clearBlock,
      false,
    );
  });

  it("a different current session hides visible effects", function () {
    var plan = planSwitchOutcomeEffects(opts({
      currentSessionId: 2,
    }));
    assert.equal(plan.updateCache, true);          // cache still updated
    assert.equal(plan.showStatus, null);
    assert.equal(plan.focus, null);
    assert.equal(plan.syncVisibleUI, false);
  });

  it("hasTarget=false forbids cache writes", function () {
    var plan = planSwitchOutcomeEffects(opts({ hasTarget: false }));
    assert.equal(plan.updateCache, false);
    assert.equal(plan.focus, null);
    assert.equal(plan.showStatus, "The conversation no longer exists.");
  });

  it("unknown statuses and malformed inputs fail closed", function () {
    assert.equal(
      planSwitchOutcomeEffects(opts({ outcome: { status: "weird" } })).kind,
      "ignore",
    );
    assert.equal(planSwitchOutcomeEffects(null).kind, "ignore");
    assert.equal(planSwitchOutcomeEffects("x").kind, "ignore");
    assert.equal(
      planSwitchOutcomeEffects(opts({ outcome: null })).kind,
      "ignore",
    );
  });
});

// -----------------------------------------------------------------------
// Controller
// -----------------------------------------------------------------------

describe("createSessionProfileSwitchController", function () {
  function switchedResponse() {
    return makeSession({
      id: 1, llm_profile_id: "local", llm_model_snapshot: "qwen",
      llm_profile_status: "ready",
    });
  }

  it("returns busy when already active and restores active afterwards", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: switchedResponse() }],
    });
    var op = makeOperation();
    var first = env.controller.apply(op);
    var second = await env.controller.apply(op);
    assert.equal(second.status, "busy");
    assert.equal(env.controller.isActive(), true);
    var firstResult = await first;
    assert.equal(firstResult.status, "switched");
    assert.equal(env.controller.isActive(), false);
  });

  it("pre-confirm hint sends ack=true and one confirm call", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: switchedResponse() }],
      confirmResults: [true],
    });
    var outcome = await env.controller.apply(makeOperation({
      needsConfirmHint: true,
    }));
    assert.equal(outcome.status, "switched");
    assert.equal(env.patchCalls.length, 1);
    assert.equal(env.patchCalls[0].ack, true);
    assert.equal(env.confirmCalls.length, 1);
    assert.equal(env.controller.isActive(), false);
  });

  it("pre-confirm cancel produces cancelled with zero PATCH calls", async function () {
    var env = makeControllerEnv({
      patchResults: [],
      confirmResults: [false],
    });
    var outcome = await env.controller.apply(makeOperation({
      needsConfirmHint: true,
    }));
    assert.equal(outcome.status, "cancelled");
    assert.equal(env.patchCalls.length, 0);
    assert.equal(env.controller.isActive(), false);
  });

  it("no pre-confirm sends ack=false", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: switchedResponse() }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "switched");
    assert.equal(env.patchCalls[0].ack, false);
    assert.equal(env.confirmCalls.length, 0);
  });

  it("first 409 confirms once and retries with ack=true", async function () {
    var env = makeControllerEnv({
      patchResults: [
        { type: "error", err: httpError(409, "remote_history_ack_required") },
        { type: "ok", value: switchedResponse() },
      ],
      confirmResults: [true],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "switched");
    assert.equal(env.patchCalls.length, 2);
    assert.equal(env.patchCalls[0].ack, false);
    assert.equal(env.patchCalls[1].ack, true);
    assert.equal(env.confirmCalls.length, 1);
    assert.equal(env.controller.isActive(), false);
  });

  it("409 cancel after the first PATCH stops without retry", async function () {
    var env = makeControllerEnv({
      patchResults: [
        { type: "error", err: httpError(409, "remote_history_ack_required") },
      ],
      confirmResults: [false],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "cancelled");
    assert.equal(env.patchCalls.length, 1);
  });

  it("second 409 fails with no third PATCH and no second dialog", async function () {
    var env = makeControllerEnv({
      patchResults: [
        { type: "error", err: httpError(409, "remote_history_ack_required") },
        { type: "error", err: httpError(409, "remote_history_ack_required") },
      ],
      confirmResults: [true],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "failed");
    assert.equal(env.patchCalls.length, 2);
    assert.equal(env.confirmCalls.length, 1);
  });

  it("404 becomes not_found without a GET", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: httpError(404) }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "not_found");
    assert.equal(env.getCalls, 0);
  });

  it("422 becomes validation_error without a GET", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: httpError(422, null, "bad") }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "validation_error");
    assert.equal(outcome.message, "bad");
    assert.equal(env.getCalls, 0);
  });

  it("other 4xx becomes failed without a GET", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: httpError(400, null, "nope") }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "failed");
    assert.equal(env.getCalls, 0);
  });

  it("network error reconciles once and confirms the target binding", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: networkError() }],
      getResults: [{ type: "ok", value: switchedResponse() }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "switched");
    assert.equal(outcome.reconciled, true);
    assert.equal(env.getCalls, 1);
  });

  it("response_parse reconciles once", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: parseError() }],
      getResults: [{ type: "ok", value: switchedResponse() }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "switched");
    assert.equal(env.getCalls, 1);
  });

  it("5xx reconciles once", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: httpError(500) }],
      getResults: [{ type: "ok", value: switchedResponse() }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "switched");
    assert.equal(env.getCalls, 1);
  });

  it("invalid 2xx body reconciles; old binding → not_changed", async function () {
    var oldBinding = makeSession({
      id: 1, llm_profile_id: "default", llm_model_snapshot: "m1",
    });
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: { id: 1, broken: true } }],
      getResults: [{ type: "ok", value: oldBinding }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "not_changed");
    assert.equal(outcome.session.id, 1);
    assert.equal(env.getCalls, 1);
  });

  it("REAL 2xx id mismatch reconciles via GET", async function () {
    // PATCH returns a VALID SessionResponse for a DIFFERENT session id.
    var otherSession = makeSession({ id: 2 });
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: otherSession }],
      getResults: [{ type: "ok", value: switchedResponse() }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "switched");
    assert.equal(outcome.reconciled, true);
    assert.equal(env.patchCalls.length, 1);
    assert.equal(env.getCalls, 1);
  });

  it("REAL 2xx binding mismatch reconciles; old binding → not_changed", async function () {
    // PATCH returns a VALID response with the right id but still bound
    // to the original profile — an invalid_success_response.
    var oldBinding = makeSession({
      id: 1, llm_profile_id: "default", llm_model_snapshot: "m1",
    });
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: oldBinding }],
      getResults: [{ type: "ok", value: oldBinding }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "not_changed");
    assert.equal(outcome.session.id, 1);
    assert.equal(env.getCalls, 1);
  });

  it("network error with old binding after GET → uncertain", async function () {
    var oldBinding = makeSession({
      id: 1, llm_profile_id: "default", llm_model_snapshot: "m1",
    });
    var env = makeControllerEnv({
      patchResults: [
        { type: "error", err: networkError() },
      ],
      getResults: [{ type: "ok", value: oldBinding }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "uncertain");
    assert.deepStrictEqual(
      outcome.requestedProfile,
      { id: "local", kind: "local", model: "qwen" },
    );
    assert.equal(env.getCalls, 1);
  });

  it("reconciliation GET 404 → not_found", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: networkError() }],
      getResults: [{ type: "error", err: httpError(404) }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "not_found");
  });

  it("reconciliation GET failure or malformed body → uncertain", async function () {
    var env1 = makeControllerEnv({
      patchResults: [{ type: "error", err: networkError() }],
      getResults: [{ type: "error", err: networkError() }],
    });
    assert.equal((await env1.controller.apply(makeOperation())).status, "uncertain");

    var env2 = makeControllerEnv({
      patchResults: [{ type: "error", err: networkError() }],
      getResults: [{ type: "ok", value: { id: 1, broken: true } }],
    });
    assert.equal((await env2.controller.apply(makeOperation())).status, "uncertain");

    var env3 = makeControllerEnv({
      patchResults: [{ type: "error", err: networkError() }],
      getResults: [{ type: "ok", value: makeSession({ id: 2 }) }],
    });
    assert.equal((await env3.controller.apply(makeOperation())).status, "uncertain");
  });

  it("plain TypeError is failed and never triggers a GET", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: new TypeError("boom") }],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "failed");
    assert.equal(env.getCalls, 0);
  });

  it("malformed error objects are failed and never trigger a GET", async function () {
    var env = makeControllerEnv({
      patchResults: [
        { type: "error", err: { status: 0, message: "x" } },
        { type: "error", err: "string error" },
        { type: "error", err: null },
      ],
    });
    assert.equal((await env.controller.apply(makeOperation())).status, "failed");
    assert.equal((await env.controller.apply(makeOperation())).status, "failed");
    assert.equal((await env.controller.apply(makeOperation())).status, "failed");
    assert.equal(env.getCalls, 0);
  });

  it("does not mutate the operation or its requestedProfile", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: switchedResponse() }],
    });
    var op = makeOperation();
    var opJson = JSON.stringify(op);
    await env.controller.apply(op);
    assert.equal(JSON.stringify(op), opJson);
  });

  it("active is restored on every outcome path", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: httpError(422, null, "bad") }],
    });
    await env.controller.apply(makeOperation());
    assert.equal(env.controller.isActive(), false);
  });
});

  describe("controller total outcome API follow-ups", function () {
  function switchedResponse() {
    return makeSession({
      id: 1, llm_profile_id: "local", llm_model_snapshot: "qwen",
      llm_profile_status: "ready",
    });
  }

it("apply(null/undefined/array/broken operation) → failed, 0 PATCH, 0 GET", async function () {
    var cases = [null, undefined, [], "x", {}, {
      generation: 0, targetSessionId: 1,
      originalProfileId: "default", originalModelSnapshot: "m1",
      requestedProfile: { id: "local", kind: "local", model: "qwen" },
      needsConfirmHint: false,
    }, {
      generation: 1, targetSessionId: 1,
      originalProfileId: "default", originalModelSnapshot: "m1",
      requestedProfile: { id: "", kind: "local", model: "qwen" },
      needsConfirmHint: false,
    }, {
      generation: 1, targetSessionId: 1,
      originalProfileId: "default", originalModelSnapshot: "m1",
      requestedProfile: { id: "local", kind: "quantum", model: "qwen" },
      needsConfirmHint: false,
    }, {
      generation: 1, targetSessionId: 1,
      originalProfileId: "default", originalModelSnapshot: "m1",
      requestedProfile: { id: "local", kind: "local", model: "qwen" },
      needsConfirmHint: "yes",
    }, {
      generation: 1, targetSessionId: 1,
      originalProfileId: 42, originalModelSnapshot: "m1",
      requestedProfile: { id: "local", kind: "local", model: "qwen" },
      needsConfirmHint: false,
    }];
    for (var i = 0; i < cases.length; i++) {
      var env = makeControllerEnv({
        patchResults: [{ type: "ok", value: switchedResponse() }],
        getResults: [{ type: "ok", value: switchedResponse() }],
      });
      var outcome = await env.controller.apply(cases[i]);
      assert.equal(outcome.status, "failed", "case " + i);
      assert.equal(env.patchCalls.length, 0, "case " + i);
      assert.equal(env.getCalls, 0, "case " + i);
      assert.equal(env.controller.isActive(), false, "case " + i);
    }
  });

  it("pre-confirm confirmRemoteHistory throw/reject → failed, no PATCH/GET", async function () {
    var thrower = makeControllerEnv({
      patchResults: [{ type: "ok", value: switchedResponse() }],
      confirmResults: [
        function () { throw new Error("boom"); },
      ],
    });
    var o1 = await thrower.controller.apply(makeOperation({
      needsConfirmHint: true,
    }));
    assert.equal(o1.status, "failed");
    assert.equal(thrower.patchCalls.length, 0);
    assert.equal(thrower.getCalls, 0);
    assert.equal(thrower.controller.isActive(), false);

    var rejecter = makeControllerEnv({
      patchResults: [{ type: "ok", value: switchedResponse() }],
      confirmResults: [
        function () { return Promise.reject(new Error("nope")); },
      ],
    });
    var o2 = await rejecter.controller.apply(makeOperation({
      needsConfirmHint: true,
    }));
    assert.equal(o2.status, "failed");
    assert.equal(rejecter.patchCalls.length, 0);
    assert.equal(rejecter.getCalls, 0);
    assert.equal(rejecter.controller.isActive(), false);
  });

  it("409-stage confirmRemoteHistory throw → failed, no second PATCH", async function () {
    var env = makeControllerEnv({
      patchResults: [
        { type: "error", err: httpError(409, "remote_history_ack_required") },
      ],
      confirmResults: [
        function () { throw new Error("boom"); },
      ],
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "failed");
    assert.equal(env.patchCalls.length, 1);
    assert.equal(env.getCalls, 0);
    assert.equal(env.controller.isActive(), false);
  });

  it("validateSessionResponse throw → failed, no GET", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: switchedResponse() }],
      validate: function () { throw new Error("validator boom"); },
    });
    var outcome = await env.controller.apply(makeOperation());
    assert.equal(outcome.status, "failed");
    assert.equal(env.getCalls, 0);
    assert.equal(env.controller.isActive(), false);
  });

  it("uncertain requestedProfile shares no reference with the operation", async function () {
    var env = makeControllerEnv({
      patchResults: [{ type: "error", err: networkError() }],
      getResults: [
        { type: "ok", value: makeSession({ id: 1, llm_profile_id: "default",
          llm_model_snapshot: "m1" }) },
      ],
    });
    var op = makeOperation();
    var outcome = await env.controller.apply(op);
    assert.equal(outcome.status, "uncertain");
    assert.notEqual(outcome.requestedProfile, op.requestedProfile);
    // mutating the outcome must not affect the operation
    outcome.requestedProfile.model = "mutated";
    assert.equal(op.requestedProfile.model, "qwen");
  });

  it("HTTP 3xx / 1xx / invalid statuses → failed with 0 GET", async function () {
    var statuses = [302, 304, 100, 199, -1, 999];
    for (var i = 0; i < statuses.length; i++) {
      var env = makeControllerEnv({
        patchResults: [{ type: "error", err: httpError(statuses[i]) }],
      });
      var outcome = await env.controller.apply(makeOperation());
      assert.equal(outcome.status, "failed", "status " + statuses[i]);
      assert.equal(env.getCalls, 0, "status " + statuses[i]);
      assert.equal(env.controller.isActive(), false, "status " + statuses[i]);
    }
  });

  // -- reconciliation GET error classification --------------------------

  it("reconciliation GET classification (internal errors → failed)", async function () {
    var cases = [
      { name: "TypeError", err: new TypeError("boom") },
      { name: "plain Error", err: new Error("boom") },
      { name: "string", err: "boom" },
      { name: "null", err: null },
      { name: "malformed object", err: { failureKind: "weird", status: 0 } },
      { name: "http 422", err: httpError(422) },
      { name: "http 302", err: httpError(302) },
    ];
    for (var i = 0; i < cases.length; i++) {
      var env = makeControllerEnv({
        patchResults: [{ type: "error", err: networkError() }],
        getResults: [{ type: "error", err: cases[i].err }],
      });
      var outcome = await env.controller.apply(makeOperation());
      assert.equal(outcome.status, "failed", cases[i].name);
      assert.equal(env.getCalls, 1, cases[i].name);
      assert.equal(env.controller.isActive(), false, cases[i].name);
    }
  });

  it("reconciliation GET ambiguity (network/parse/5xx) → uncertain", async function () {
    var cases = [
      { name: "network", err: networkError() },
      { name: "response_parse", err: parseError() },
      { name: "http 500", err: httpError(500) },
    ];
    for (var i = 0; i < cases.length; i++) {
      var env = makeControllerEnv({
        patchResults: [{ type: "error", err: networkError() }],
        getResults: [{ type: "error", err: cases[i].err }],
      });
      var outcome = await env.controller.apply(makeOperation());
      assert.equal(outcome.status, "uncertain", cases[i].name);
      assert.equal(env.getCalls, 1, cases[i].name);
      assert.equal(env.controller.isActive(), false, cases[i].name);
    }
  });

  // -- uncertain record cross-session guards ---------------------------

  it("uncertain plan ignores an operation targeting another session", function () {
    var plan = planSwitchOutcomeEffects({
      outcome: {
        status: "uncertain", message: "m",
        requestedProfile: { id: "local", kind: "local", model: "qwen" },
      },
      operation: makeOperation({ targetSessionId: 2 }),
      targetSessionId: 1,
      currentSessionId: 1,
      hasTarget: true,
    });
    assert.equal(plan.kind, "ignore");
    assert.equal(plan.uncertainRecord, null);
    assert.equal(plan.focus, null);
  });

  it("uncertain plan ignores a mismatched outcome requestedProfile", function () {
    var mismatches = [
      { id: "other", kind: "local", model: "qwen" },
      { id: "local", kind: "api", model: "qwen" },
      { id: "local", kind: "local", model: "other" },
      null,
    ];
    mismatches.forEach(function (rp) {
      var plan = planSwitchOutcomeEffects({
        outcome: {
          status: "uncertain", message: "m",
          requestedProfile: rp,
        },
        operation: makeOperation(),
        targetSessionId: 1,
        currentSessionId: 1,
        hasTarget: true,
      });
      assert.equal(plan.kind, "ignore", JSON.stringify(rp));
      assert.equal(plan.uncertainRecord, null);
    });
  });

  // -- profile validation consistent with the backend -------------------

  it("rejects profile ids that violate the backend rule", async function () {
    var badIds = [" local", "local ", "Local", "local_model", "local\n",
      "a".repeat(51)];
    for (var i = 0; i < badIds.length; i++) {
      var env = makeControllerEnv({
        patchResults: [{ type: "ok", value: switchedResponse() }],
      });
      var outcome = await env.controller.apply(makeOperation({
        requestedProfile: { id: badIds[i], kind: "local", model: "qwen" },
      }));
      assert.equal(outcome.status, "failed", "id " + JSON.stringify(badIds[i]));
      assert.equal(env.patchCalls.length, 0);
    }
  });

  it("rejects un-normalised models and fake/non-fake mismatches", async function () {
    var cases = [
      { kind: "local", model: " qwen" },
      { kind: "local", model: "qwen " },
      { kind: "fake", model: "qwen" },
    ];
    for (var i = 0; i < cases.length; i++) {
      var env = makeControllerEnv({
        patchResults: [{ type: "ok", value: switchedResponse() }],
      });
      var outcome = await env.controller.apply(makeOperation({
        requestedProfile: cases[i],
      }));
      assert.equal(outcome.status, "failed", JSON.stringify(cases[i]));
      assert.equal(env.patchCalls.length, 0);
    }
  });

  it("accepts a well-formed fake profile (fake/fake)", async function () {
    var fakeBound = makeSession({
      id: 1, llm_profile_id: "default", llm_model_snapshot: "fake",
      llm_profile_status: "ready",
    });
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: fakeBound }],
    });
    var outcome = await env.controller.apply(makeOperation({
      requestedProfile: { id: "default", kind: "fake", model: "fake" },
    }));
    assert.equal(outcome.status, "switched");
  });

  it("rejects invalid originalProfileId / originalModelSnapshot", async function () {
    var cases = [
      { originalProfileId: "Local", originalModelSnapshot: "m1" },
      { originalProfileId: "default", originalModelSnapshot: " m1" },
      { originalProfileId: "default", originalModelSnapshot: "" },
    ];
    for (var i = 0; i < cases.length; i++) {
      var env = makeControllerEnv({
        patchResults: [{ type: "ok", value: switchedResponse() }],
      });
      var outcome = await env.controller.apply(makeOperation(cases[i]));
      assert.equal(outcome.status, "failed", JSON.stringify(cases[i]));
      assert.equal(env.patchCalls.length, 0);
    }
  });

  it("a getter-throwing operation resolves failed without rejecting", async function () {
    var evil = {};
    Object.defineProperty(evil, "generation", {
      get: function () { throw new Error("getter boom"); },
    });
    var env = makeControllerEnv({
      patchResults: [{ type: "ok", value: switchedResponse() }],
    });
    var outcome = await env.controller.apply(evil);
    assert.equal(outcome.status, "failed");
    assert.equal(env.patchCalls.length, 0);
    assert.equal(env.getCalls, 0);
    assert.equal(env.controller.isActive(), false);
  });

});

// -----------------------------------------------------------------------
// Confirmer
// -----------------------------------------------------------------------

describe("createRemoteHistoryConfirmer", function () {
  it("Continue resolves true and cleans listeners", async function () {
    var adapter = makeFakeDialogAdapter();
    var confirmer = createRemoteHistoryConfirmer(adapter);
    var promise = confirmer.confirm("msg");
    assert.equal(adapter.message, "msg");
    fireDialog(adapter, "continueClick");
    assert.equal(await promise, true);
    assert.equal(adapter.listenerCount(), 0);
    assert.equal(adapter.closeCalls, 1);
  });

  it("Cancel and Escape resolve false and clean listeners", async function () {
    var adapter = makeFakeDialogAdapter();
    var confirmer = createRemoteHistoryConfirmer(adapter);
    var p1 = confirmer.confirm("a");
    fireDialog(adapter, "cancelClick");
    assert.equal(await p1, false);
    assert.equal(adapter.listenerCount(), 0);

    var p2 = confirmer.confirm("b");
    fireDialog(adapter, "cancel");
    assert.equal(await p2, false);
    assert.equal(adapter.listenerCount(), 0);
  });

  it("settles exactly once per call", async function () {
    var adapter = makeFakeDialogAdapter();
    var confirmer = createRemoteHistoryConfirmer(adapter);
    var p = confirmer.confirm("a");
    // Capture the registered callbacks, then fire several of them —
    // only the first may settle the promise.
    var cancelCb = adapter.listeners.cancel[0];
    var cancelClickCb = adapter.listeners.cancelClick[0];
    var continueCb = adapter.listeners.continueClick[0];
    cancelClickCb();
    cancelCb();
    continueCb();
    assert.equal(await p, false);
    assert.equal(adapter.listenerCount(), 0);
  });

  it("a second pending confirmation resolves false immediately", async function () {
    var adapter = makeFakeDialogAdapter();
    var confirmer = createRemoteHistoryConfirmer(adapter);
    var p1 = confirmer.confirm("a");
    var p2 = await confirmer.confirm("b");
    assert.equal(p2, false);
    fireDialog(adapter, "continueClick");
    assert.equal(await p1, true);
  });

  it("showModal missing / throwing / disconnected fails closed", async function () {
    var noModal = makeFakeDialogAdapter({ showModal: undefined });
    assert.equal(await createRemoteHistoryConfirmer(noModal).confirm("a"), false);
    assert.equal(noModal.listenerCount(), 0);

    var throws = makeFakeDialogAdapter({
      showModal: function () { throw new Error("boom"); },
    });
    assert.equal(await createRemoteHistoryConfirmer(throws).confirm("a"), false);
    assert.equal(throws.listenerCount(), 0);

    var disconnected = makeFakeDialogAdapter({
      isConnected: function () { return false; },
    });
    assert.equal(
      await createRemoteHistoryConfirmer(disconnected).confirm("a"),
      false,
    );
    assert.equal(disconnected.listenerCount(), 0);
  });

  it("close() throwing cannot block settlement or cleanup", async function () {
    var adapter = makeFakeDialogAdapter({
      close: function () { throw new Error("close boom"); },
    });
    var confirmer = createRemoteHistoryConfirmer(adapter);
    var p = confirmer.confirm("a");
    fireDialog(adapter, "continueClick");
    assert.equal(await p, true);
    assert.equal(adapter.listenerCount(), 0);
  });

  it("setMessage / subscription / focusInitial failures fail closed with cleanup", async function () {
    var badSet = makeFakeDialogAdapter({
      setMessage: function () { throw new Error("boom"); },
    });
    assert.equal(await createRemoteHistoryConfirmer(badSet).confirm("a"), false);
    assert.equal(badSet.listenerCount(), 0);

    var badSub = makeFakeDialogAdapter({
      onCancelClick: function () { throw new Error("boom"); },
    });
    assert.equal(await createRemoteHistoryConfirmer(badSub).confirm("a"), false);
    assert.equal(badSub.listenerCount(), 0);

    var badFocus = makeFakeDialogAdapter({
      focusInitial: function () { throw new Error("boom"); },
    });
    // focusInitial throwing resolves false WITHOUT any user click,
    // and every listener is cleaned up.
    assert.equal(await createRemoteHistoryConfirmer(badFocus).confirm("a"), false);
    assert.equal(badFocus.listenerCount(), 0);
  });

  it("a subscription that returns a non-function fails closed with cleanup", async function () {
    var badUnsub = makeFakeDialogAdapter({
      onCancelClick: function () {
        // Broken adapter: registers nothing and returns garbage.
        return "not a function";
      },
    });
    assert.equal(
      await createRemoteHistoryConfirmer(badUnsub).confirm("a"),
      false,
    );
    assert.equal(badUnsub.listenerCount(), 0);
  });

  it("a subscription firing its callback synchronously leaves no listeners", async function () {
    var adapter = makeFakeDialogAdapter({
      onCancelClick: function (cb) {
        // Fire the callback synchronously during registration.
        cb();
        return function () { this.listeners.cancelClick = []; }.bind(this);
      },
    });
    var confirmer = createRemoteHistoryConfirmer(adapter);
    var result = await confirmer.confirm("a");
    assert.equal(result, false);
    assert.equal(adapter.listenerCount(), 0);
  });
});
