/**
 * Tests for frontend/interaction-mode.js — pure Iteration 1 helpers.
 *
 * Run: node --test tests/test_interaction_mode.test.js
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const {
  RECEIVE_TEACHING_MODE,
  CORRECTIVE_MODE,
  isValidInteractionMode,
  resolveInteractionMode,
  buildSwitchInteractionModePayload,
  interactionModeLabel,
  interactionModeBadgeText,
  interactionModeHintText,
  isValidModeSwitchEvent,
  isValidSwitchInteractionModeResponse,
  interactionModeDraftForSession,
  interactionModeAuthoritativeMode,
  hasInteractionModeUncertain,
  interactionModeUncertainText,
  canApplyInteractionMode,
  createInteractionModeSwitchController,
} = require(
  path.resolve(__dirname, "..", "frontend", "interaction-mode.js"),
);
const {
  isValidSessionResponse,
  isValidApiTimestamp,
} = require(
  path.resolve(__dirname, "..", "frontend", "model-selection.js"),
);

function makeSession(overrides) {
  const session = {
    id: 1,
    title: "New Chat",
    interaction_mode: RECEIVE_TEACHING_MODE,
    created_at: "2026-08-06T12:00:00",
    updated_at: "2026-08-06T12:00:00",
    llm_profile_id: "default",
    llm_profile_label: "Fake Model",
    llm_profile_status: "ready",
    llm_model_snapshot: "fake",
  };
  if (overrides) Object.assign(session, overrides);
  return session;
}

function makeSwitchEvent(overrides) {
  const event = {
    id: 1,
    session_id: 1,
    from_mode: RECEIVE_TEACHING_MODE,
    to_mode: CORRECTIVE_MODE,
    created_at: "2026-08-06T12:00:00",
    history_through_message_id: 1,
    history_boundary_version: "history-boundary-v1",
    reviewable_user_message_count: 1,
    review_supported: true,
  };
  if (overrides) Object.assign(event, overrides);
  return event;
}

describe("isValidInteractionMode", function () {
  it("accepts the two supported modes", function () {
    assert.equal(isValidInteractionMode(RECEIVE_TEACHING_MODE), true);
    assert.equal(isValidInteractionMode(CORRECTIVE_MODE), true);
  });

  it("rejects invalid, missing, and non-string values", function () {
    assert.equal(isValidInteractionMode("review"), false);
    assert.equal(isValidInteractionMode(""), false);
    assert.equal(isValidInteractionMode(null), false);
    assert.equal(isValidInteractionMode(undefined), false);
    assert.equal(isValidInteractionMode(1), false);
  });
});

describe("resolveInteractionMode legacy compatibility", function () {
  it("falls back to receive_teaching for missing legacy values", function () {
    assert.equal(resolveInteractionMode(undefined), RECEIVE_TEACHING_MODE);
    assert.equal(resolveInteractionMode(null), RECEIVE_TEACHING_MODE);
  });

  it("fails closed for malformed non-empty values", function () {
    assert.equal(resolveInteractionMode("review"), null);
  });
});

describe("buildSwitchInteractionModePayload", function () {
  it("contains only the requested mode", function () {
    assert.deepEqual(
      buildSwitchInteractionModePayload(CORRECTIVE_MODE),
      { interaction_mode: CORRECTIVE_MODE },
    );
  });

  it("never includes a model profile field", function () {
    const payload = buildSwitchInteractionModePayload(CORRECTIVE_MODE);
    assert.equal(payload.llm_profile_id, undefined);
  });

  it("returns null for invalid modes", function () {
    assert.equal(buildSwitchInteractionModePayload("review"), null);
  });
});

describe("labels and hints", function () {
  it("labels both modes and treats missing as receive", function () {
    assert.equal(interactionModeLabel(RECEIVE_TEACHING_MODE), "Receive teaching");
    assert.equal(interactionModeLabel(CORRECTIVE_MODE), "Corrective");
    assert.equal(interactionModeBadgeText(undefined), "");
  });

  it("corrective hint describes optional separate history review", function () {
    const hint = interactionModeHintText(CORRECTIVE_MODE);
    assert.match(hint, /future messages/);
    assert.match(hint, /History Review/);
    assert.doesNotMatch(hint, /not available/i);
    assert.doesNotMatch(hint, /automatically/i);
  });
});

describe("mode switch event validation", function () {
  it("accepts a complete valid event", function () {
    assert.equal(
      isValidModeSwitchEvent(makeSwitchEvent(), isValidApiTimestamp), true,
    );
  });

  it("requires every response field as an own property", function () {
    const inherited = Object.create(makeSwitchEvent());
    assert.equal(
      isValidModeSwitchEvent(inherited, isValidApiTimestamp), false,
    );

    const missing = makeSwitchEvent();
    delete missing.review_supported;
    assert.equal(
      isValidModeSwitchEvent(missing, isValidApiTimestamp), false,
    );

    const inheritedField = makeSwitchEvent();
    delete inheritedField.review_supported;
    Object.setPrototypeOf(inheritedField, { review_supported: true });
    assert.equal(
      isValidModeSwitchEvent(inheritedField, isValidApiTimestamp), false,
    );
  });

  it("validates optional field types strictly", function () {
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ history_through_message_id: 0 }),
        isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ history_boundary_version: "" }),
        isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ reviewable_user_message_count: -1 }),
        isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ review_supported: 1 }),
        isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ review_supported: false }),
        isValidApiTimestamp,
      ),
      true,
    );
  });

  it("enforces consistency when review_supported is true", function () {
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({
          from_mode: CORRECTIVE_MODE,
          review_supported: true,
        }),
        isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ history_through_message_id: null }),
        isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ reviewable_user_message_count: 0 }),
        isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ history_boundary_version: null }),
        isValidApiTimestamp,
      ),
      false,
    );
  });

  it("rejects invalid modes, malformed ids, and bad timestamps", function () {
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ from_mode: "review" }), isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ session_id: 0 }), isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidModeSwitchEvent(
        makeSwitchEvent({ created_at: "not-a-date" }), isValidApiTimestamp,
      ),
      false,
    );
  });
});

describe("switch response validation", function () {
  it("accepts a valid switched response when requested differs", function () {
    const value = {
      session: makeSession({ interaction_mode: CORRECTIVE_MODE }),
      switch_event: makeSwitchEvent(),
    };
    assert.equal(
      isValidSwitchInteractionModeResponse(
        value, 1, CORRECTIVE_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      true,
    );
  });

  it("rejects null event when requested differs from original", function () {
    const value = {
      session: makeSession({ interaction_mode: CORRECTIVE_MODE }),
      switch_event: null,
    };
    assert.equal(
      isValidSwitchInteractionModeResponse(
        value, 1, CORRECTIVE_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      false,
    );
  });

  it("rejects event.from_mode that does not equal original mode", function () {
    const value = {
      session: makeSession({ interaction_mode: CORRECTIVE_MODE }),
      switch_event: makeSwitchEvent({ from_mode: CORRECTIVE_MODE }),
    };
    assert.equal(
      isValidSwitchInteractionModeResponse(
        value, 1, CORRECTIVE_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      false,
    );
  });

  it("rejects event.session_id mismatch and bad event timestamp", function () {
    const wrongSession = {
      session: makeSession({ interaction_mode: CORRECTIVE_MODE }),
      switch_event: makeSwitchEvent({ session_id: 2 }),
    };
    assert.equal(
      isValidSwitchInteractionModeResponse(
        wrongSession, 1, CORRECTIVE_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      false,
    );

    const badTimestamp = {
      session: makeSession({ interaction_mode: CORRECTIVE_MODE }),
      switch_event: makeSwitchEvent({ created_at: "nope" }),
    };
    assert.equal(
      isValidSwitchInteractionModeResponse(
        badTimestamp, 1, CORRECTIVE_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      false,
    );
  });

  it("allows null event only for true same-mode no-op", function () {
    const value = {
      session: makeSession({ interaction_mode: RECEIVE_TEACHING_MODE }),
      switch_event: null,
    };
    assert.equal(
      isValidSwitchInteractionModeResponse(
        value, 1, RECEIVE_TEACHING_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      true,
    );
    assert.equal(
      isValidSwitchInteractionModeResponse(
        {
          session: makeSession({ interaction_mode: RECEIVE_TEACHING_MODE }),
          switch_event: makeSwitchEvent({
            from_mode: RECEIVE_TEACHING_MODE,
            to_mode: RECEIVE_TEACHING_MODE,
          }),
        },
        1, RECEIVE_TEACHING_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      false,
    );
  });

  it("rejects wrong session id, wrong target mode, and malformed session", function () {
    const value = {
      session: makeSession({ interaction_mode: CORRECTIVE_MODE }),
      switch_event: makeSwitchEvent(),
    };
    assert.equal(
      isValidSwitchInteractionModeResponse(
        value, 2, CORRECTIVE_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidSwitchInteractionModeResponse(
        value, 1, RECEIVE_TEACHING_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      false,
    );
    assert.equal(
      isValidSwitchInteractionModeResponse(
        { session: null, switch_event: null },
        1, CORRECTIVE_MODE, RECEIVE_TEACHING_MODE,
        isValidSessionResponse, isValidApiTimestamp,
      ),
      false,
    );
  });
});

describe("session/message response compatibility", function () {
  it("accepts a legacy session without interaction_mode", function () {
    const session = makeSession();
    delete session.interaction_mode;
    assert.equal(isValidSessionResponse(session), true);
  });

  it("rejects a session with a malformed interaction_mode", function () {
    assert.equal(
      isValidSessionResponse(makeSession({ interaction_mode: "review" })),
      false,
    );
  });
});

describe("draft and apply guards", function () {
  it("draft comes from the server session", function () {
    assert.equal(
      interactionModeDraftForSession(
        makeSession({ interaction_mode: CORRECTIVE_MODE }),
      ),
      CORRECTIVE_MODE,
    );
  });

  it("duplicate apply is blocked while a switch is in flight", function () {
    const session = makeSession();
    assert.equal(
      canApplyInteractionMode({
        session: session,
        draftMode: CORRECTIVE_MODE,
        isSwitching: false,
      }),
      true,
    );
    assert.equal(
      canApplyInteractionMode({
        session: session,
        draftMode: CORRECTIVE_MODE,
        isSwitching: true,
      }),
      false,
    );
  });

  it("same-mode apply is a no-op", function () {
    assert.equal(
      canApplyInteractionMode({
        session: makeSession(),
        draftMode: RECEIVE_TEACHING_MODE,
        isSwitching: false,
      }),
      false,
    );
  });

  it("failure rollback can restore the server mode", function () {
    const session = makeSession({ interaction_mode: CORRECTIVE_MODE });
    assert.equal(
      interactionModeDraftForSession(session),
      CORRECTIVE_MODE,
    );
  });
});


function makeController(overrides) {
  const deps = {
    patchSwitch: async function () {
      throw { failureKind: "network", status: 0, message: "offline" };
    },
    fetchSession: async function (sessionId) {
      return makeSession({
        id: sessionId,
        interaction_mode: CORRECTIVE_MODE,
      });
    },
    validateSession: isValidSessionResponse,
    validateTimestamp: isValidApiTimestamp,
  };
  if (overrides) Object.assign(deps, overrides);
  return createInteractionModeSwitchController(deps);
}

describe("interaction-mode reconciliation controller", function () {
  it("direct PATCH carries a copied validated switch event", async function () {
    const switchEvent = makeSwitchEvent();
    const controller = makeController({
      patchSwitch: async function () {
        return {
          session: makeSession({ interaction_mode: CORRECTIVE_MODE }),
          switch_event: switchEvent,
        };
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });

    assert.equal(outcome.status, "switched");
    assert.equal(outcome.reconciled, false);
    assert.deepEqual(outcome.switchEvent, switchEvent);
    assert.notEqual(outcome.switchEvent, switchEvent);
  });

  it("reconciled switch has no event", async function () {
    const controller = makeController({
      patchSwitch: async function () {
        throw { failureKind: "network", status: 0 };
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });

    assert.equal(outcome.status, "switched");
    assert.equal(outcome.reconciled, true);
    assert.equal(outcome.switchEvent, null);
  });

  it("PATCH response lost, GET confirms requested mode", async function () {
    const controller = makeController({
      patchSwitch: async function () {
        throw { failureKind: "response_parse", status: 0 };
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(outcome.status, "switched");
    assert.equal(outcome.reconciled, true);
    assert.equal(outcome.session.interaction_mode, CORRECTIVE_MODE);
  });

  it("PATCH fails, GET reports original mode", async function () {
    const controller = makeController({
      patchSwitch: async function () {
        throw { failureKind: "http", status: 503 };
      },
      fetchSession: async function () {
        return makeSession({ interaction_mode: RECEIVE_TEACHING_MODE });
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(outcome.status, "not_changed");
    assert.equal(outcome.reconciled, true);
  });

  it("PATCH and GET both fail, outcome is uncertain", async function () {
    const controller = makeController({
      patchSwitch: async function () {
        throw { failureKind: "network", status: 0 };
      },
      fetchSession: async function () {
        throw { failureKind: "network", status: 0 };
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(outcome.status, "uncertain");
    assert.match(outcome.message, /could not be confirmed/);
  });

  it("invalid 2xx response is reconciled instead of trusted", async function () {
    let fetchCalls = 0;
    const controller = makeController({
      patchSwitch: async function () {
        return {
          session: makeSession({ id: 999, interaction_mode: CORRECTIVE_MODE }),
          switch_event: null,
        };
      },
      fetchSession: async function () {
        fetchCalls += 1;
        return makeSession({
          id: 1,
          interaction_mode: RECEIVE_TEACHING_MODE,
        });
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(fetchCalls, 1);
    assert.equal(outcome.status, "not_changed");
    assert.equal(outcome.reconciled, true);
  });

  it("direct 4xx is a deterministic failure, not uncertainty", async function () {
    const controller = makeController({
      patchSwitch: async function () {
        throw { failureKind: "http", status: 422, message: "bad mode" };
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(outcome.status, "failed");
    assert.equal(outcome.message, "bad mode");
  });

  it("GET 404 during reconciliation is not_found", async function () {
    const controller = makeController({
      patchSwitch: async function () {
        throw { failureKind: "network", status: 0 };
      },
      fetchSession: async function () {
        throw { failureKind: "http", status: 404, message: "gone" };
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(outcome.status, "not_found");
  });

  it("same-mode operation is a no-op without PATCH", async function () {
    let patchCalls = 0;
    const controller = makeController({
      patchSwitch: async function () {
        patchCalls += 1;
        return {};
      },
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: RECEIVE_TEACHING_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(patchCalls, 0);
    assert.equal(outcome.status, "not_changed");
    assert.equal(outcome.switchEvent, null);
  });

  it("reconcile can be used as the Apply recheck convergence path", async function () {
    const controller = makeController({
      fetchSession: async function () {
        return makeSession({
          id: 1,
          interaction_mode: CORRECTIVE_MODE,
        });
      },
    });
    const outcome = await controller.reconcile({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(outcome.status, "switched");
    assert.equal(outcome.reconciled, true);
  });
});

describe("uncertainty, authoritative rendering, and write gates", function () {
  it("uncertain map is strictly target-session scoped", function () {
    const map = Object.create(null);
    map["2"] = {
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    };
    assert.equal(hasInteractionModeUncertain(map, 2), true);
    assert.equal(hasInteractionModeUncertain(map, 1), false);
  });

  it("draft changes never change the authoritative session mode", function () {
    const session = makeSession({ interaction_mode: RECEIVE_TEACHING_MODE });
    const draftMode = CORRECTIVE_MODE;
    assert.equal(interactionModeAuthoritativeMode(session),
      RECEIVE_TEACHING_MODE);
    assert.equal(interactionModeDraftForSession(session),
      RECEIVE_TEACHING_MODE);
    assert.notEqual(draftMode, interactionModeAuthoritativeMode(session));
  });

  it("switching in flight blocks Apply and write-like guards", function () {
    const session = makeSession();
    assert.equal(
      canApplyInteractionMode({
        session: session,
        draftMode: CORRECTIVE_MODE,
        isSwitching: true,
      }),
      false,
    );
  });

  it("uncertain enables only the recheck/Apply path, not a new switch", function () {
    const session = makeSession();
    assert.equal(
      canApplyInteractionMode({
        session: session,
        draftMode: CORRECTIVE_MODE,
        isSwitching: false,
        hasUncertain: true,
      }),
      true,
    );
    assert.equal(
      canApplyInteractionMode({
        session: session,
        draftMode: CORRECTIVE_MODE,
        isSwitching: true,
        hasUncertain: true,
      }),
      false,
    );
    assert.match(interactionModeUncertainText(), /could not be confirmed/);
  });
});


describe("strict reconciliation snapshot validation", function () {
  it("missing interaction_mode is uncertain for corrective to receive", async function () {
    const controller = createInteractionModeSwitchController({
      patchSwitch: async function () {
        throw { failureKind: "network", status: 0 };
      },
      fetchSession: async function () {
        const session = makeSession({ interaction_mode: RECEIVE_TEACHING_MODE });
        delete session.interaction_mode;
        return session;
      },
      validateSession: isValidSessionResponse,
      validateTimestamp: isValidApiTimestamp,
    });
    const outcome = await controller.apply({
      targetSessionId: 1,
      requestedMode: RECEIVE_TEACHING_MODE,
      originalMode: CORRECTIVE_MODE,
    });
    assert.equal(outcome.status, "uncertain");
    assert.notEqual(outcome.status, "switched");
  });

  it("null, undefined, non-string, and inherited values are uncertain", async function () {
    const values = [null, undefined, 123, "review"];
    for (const value of values) {
      const controller = createInteractionModeSwitchController({
        patchSwitch: async function () {
          throw { failureKind: "network", status: 0 };
        },
        fetchSession: async function () {
          return { id: 1, interaction_mode: value };
        },
        validateSession: function () { return true; },
        validateTimestamp: isValidApiTimestamp,
      });
      const outcome = await controller.reconcile({
        targetSessionId: 1,
        requestedMode: CORRECTIVE_MODE,
        originalMode: RECEIVE_TEACHING_MODE,
      });
      assert.equal(outcome.status, "uncertain");
    }

    const inheritedSession = Object.create({
      id: 1,
      interaction_mode: CORRECTIVE_MODE,
    });
    const controller = createInteractionModeSwitchController({
      patchSwitch: async function () {
        throw { failureKind: "network", status: 0 };
      },
      fetchSession: async function () { return inheritedSession; },
      validateSession: function () { return true; },
      validateTimestamp: isValidApiTimestamp,
    });
    const outcome = await controller.reconcile({
      targetSessionId: 1,
      requestedMode: CORRECTIVE_MODE,
      originalMode: RECEIVE_TEACHING_MODE,
    });
    assert.equal(outcome.status, "uncertain");
  });
});
