/**
 * Tests for model-selection.js — pure helpers for the model selector.
 *
 * Run:  node --test tests/test_model_selection.test.js
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const {
  analyzeProfileRegistry,
  resolveSelectedProfileId,
  isSessionWritable,
  readOnlyExplanation,
  temporaryBlockExplanation,
  profileRegistryStatusText,
  buildCreateSessionPayload,
  profileKindBadgeText,
  isValidApiTimestamp,
  isValidSessionResponse,
  isValidSessionList,
  isValidMessageResponse,
  isValidMessageList,
  isValidSendMessageResponse,
  profileKindFromRegistry,
  resolveNextSelectionId,
  findSessionButton,
  resolveSessionProfileDraft,
  planProfileDraftForSelection,
  canApplySessionProfile,
  canApplySessionProfileWithUncertain,
  buildSwitchSessionProfilePayload,
  needsRemoteHistoryConfirmation,
  parseRemoteHistoryAckRequired,
  messageProvenanceLabel,
  canApplySwitchResponse,
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

// -----------------------------------------------------------------------
// analyzeProfileRegistry
// -----------------------------------------------------------------------

describe("analyzeProfileRegistry", function () {
  it("accepts a valid single profile", function () {
    var r = analyzeProfileRegistry([makeProfile("default", { is_default: true })]);
    assert.equal(r.status, "valid");
    assert.equal(r.defaultProfileId, "default");
    assert.equal(r.profiles.length, 1);
  });

  it("accepts two valid profiles and finds the default", function () {
    var r = analyzeProfileRegistry([
      makeProfile("local", { kind: "local" }),
      makeProfile("default", { is_default: true, kind: "fake", model: "fake" }),
    ]);
    assert.equal(r.status, "valid");
    assert.equal(r.defaultProfileId, "default");
  });

  it("returns empty for an empty array", function () {
    var r = analyzeProfileRegistry([]);
    assert.equal(r.status, "empty");
    assert.equal(r.defaultProfileId, null);
  });

  it("returns invalid_response for null", function () {
    var r = analyzeProfileRegistry(null);
    assert.equal(r.status, "invalid_response");
  });

  it("returns invalid_response for undefined", function () {
    var r = analyzeProfileRegistry(undefined);
    assert.equal(r.status, "invalid_response");
  });

  it("returns invalid_response for a string", function () {
    var r = analyzeProfileRegistry("hello");
    assert.equal(r.status, "invalid_response");
  });

  it("returns invalid_response for a plain object", function () {
    var r = analyzeProfileRegistry({});
    assert.equal(r.status, "invalid_response");
  });

  it("returns invalid_response for an array containing null", function () {
    var r = analyzeProfileRegistry([null]);
    assert.equal(r.status, "invalid_response");
  });

  it("returns invalid_response for a malformed element", function () {
    // missing id, wrong types, unknown kind, non-boolean is_default
    var bad = [
      { label: "x", kind: "api", model: "m", is_default: true },      // no id
      makeProfile("a", { label: 42 }),                                 // label type
      makeProfile("b", { kind: "quantum" }),                           // kind
      makeProfile("c", { is_default: "true" }),                        // flag type
      makeProfile("d", { id: "   " }),                                 // blank id
      makeProfile("e", { label: "   " }),                              // blank label
      makeProfile("f", { model: "  " }),                               // blank model
    ];
    for (var i = 0; i < bad.length; i++) {
      var r = analyzeProfileRegistry([bad[i]]);
      assert.equal(r.status, "invalid_response", "case " + i);
    }
  });

  it("returns duplicate_ids for duplicated ids", function () {
    var r = analyzeProfileRegistry([
      makeProfile("default", { is_default: true }),
      makeProfile("default", { is_default: false }),
    ]);
    assert.equal(r.status, "duplicate_ids");
  });

  it("returns missing_default when no default exists", function () {
    var r = analyzeProfileRegistry([
      makeProfile("a"),
      makeProfile("b"),
    ]);
    assert.equal(r.status, "missing_default");
  });

  it("returns multiple_defaults when several defaults exist", function () {
    var r = analyzeProfileRegistry([
      makeProfile("a", { is_default: true }),
      makeProfile("b", { is_default: true }),
    ]);
    assert.equal(r.status, "multiple_defaults");
  });

  it("prefers duplicate_ids over multiple_defaults (deterministic priority)", function () {
    var r = analyzeProfileRegistry([
      makeProfile("a", { is_default: true }),
      makeProfile("a", { is_default: true }),
    ]);
    assert.equal(r.status, "duplicate_ids");
  });

  it("does not mutate the input array", function () {
    var input = [makeProfile("default", { is_default: true })];
    var snapshot = JSON.stringify(input);
    analyzeProfileRegistry(input);
    assert.equal(JSON.stringify(input), snapshot);
  });

  it("detects duplicate __proto__ ids (prototype-chain key)", function () {
    var r = analyzeProfileRegistry([
      makeProfile("__proto__", { is_default: true }),
      makeProfile("__proto__", { is_default: false }),
    ]);
    assert.equal(r.status, "duplicate_ids");
  });

  it("detects duplicate constructor ids", function () {
    var r = analyzeProfileRegistry([
      makeProfile("constructor", { is_default: true }),
      makeProfile("constructor", { is_default: false }),
    ]);
    assert.equal(r.status, "duplicate_ids");
  });
});

// -----------------------------------------------------------------------
// resolveSelectedProfileId
// -----------------------------------------------------------------------

describe("resolveSelectedProfileId", function () {
  var validRegistry = analyzeProfileRegistry([
    makeProfile("local", { kind: "local" }),
    makeProfile("default", { is_default: true }),
  ]);

  it("keeps a still-valid current selection", function () {
    assert.equal(resolveSelectedProfileId(validRegistry, "local"), "local");
  });

  it("falls back to the unique default when the selection disappeared", function () {
    assert.equal(resolveSelectedProfileId(validRegistry, "gone"), "default");
  });

  it("returns the default when current is null", function () {
    assert.equal(resolveSelectedProfileId(validRegistry, null), "default");
  });

  it("returns null for an invalid registry (never the first profile)", function () {
    var bad = analyzeProfileRegistry([makeProfile("a"), makeProfile("b")]);
    assert.equal(resolveSelectedProfileId(bad, "a"), null);

    var empty = analyzeProfileRegistry([]);
    assert.equal(resolveSelectedProfileId(empty, null), null);

    assert.equal(resolveSelectedProfileId(null, null), null);
    assert.equal(resolveSelectedProfileId("garbage", null), null);
  });
});

// -----------------------------------------------------------------------
// isSessionWritable
// -----------------------------------------------------------------------

describe("isSessionWritable", function () {
  it("allows typing with no current session (auto-create path)", function () {
    assert.equal(isSessionWritable(null, null), true);
    assert.equal(isSessionWritable(undefined, undefined), true);
  });

  it("allows a ready session", function () {
    assert.equal(
      isSessionWritable({ llm_profile_status: "ready" }, null),
      true,
    );
  });

  it("blocks the three known non-ready statuses", function () {
    assert.equal(
      isSessionWritable({ llm_profile_status: "profile_unavailable" }, null),
      false,
    );
    assert.equal(
      isSessionWritable({ llm_profile_status: "model_changed" }, null),
      false,
    );
    assert.equal(
      isSessionWritable({ llm_profile_status: "legacy_unknown" }, null),
      false,
    );
  });

  it("fails closed on an unknown status", function () {
    assert.equal(
      isSessionWritable({ llm_profile_status: "weird_status" }, null),
      false,
    );
  });

  it("fails closed when the status field is missing", function () {
    assert.equal(isSessionWritable({}, null), false);
  });

  it("a temporary block wins over a stale ready status", function () {
    assert.equal(
      isSessionWritable({ llm_profile_status: "ready" }, "conflict"),
      false,
    );
    assert.equal(
      isSessionWritable({ llm_profile_status: "ready" }, "profile_unavailable"),
      false,
    );
  });

  it("a present uncertain record blocks even a ready session", function () {
    assert.equal(
      isSessionWritable(
        { llm_profile_status: "ready" },
        null,
        { generation: 1 },
      ),
      false,
    );
  });

  it("no uncertain record leaves a ready session writable", function () {
    assert.equal(
      isSessionWritable({ llm_profile_status: "ready" }, null, undefined),
      true,
    );
    assert.equal(
      isSessionWritable({ llm_profile_status: "ready" }, null, null),
      true,
    );
  });

  it("the uncertain record only blocks its own session", function () {
    // The record is per-session state: with no record for THIS
    // session the answer is writable, and it flips as soon as the
    // record appears — other sessions' records never leak in.
    assert.equal(
      isSessionWritable({ llm_profile_status: "ready" }, null, undefined),
      true,
    );
    assert.equal(
      isSessionWritable(
        { llm_profile_status: "ready" }, null, { generation: 1 },
      ),
      false,
    );
  });

  it("writability returns once the uncertain argument is absent", function () {
    // Pure argument contract only: the same ready session flips from
    // blocked to writable when the third argument goes away.  Deleting
    // the per-session record is the app executor's job and is not
    // exercised here.
    assert.equal(
      isSessionWritable(
        { llm_profile_status: "ready" }, null, { generation: 1 },
      ),
      false,
    );
    assert.equal(
      isSessionWritable({ llm_profile_status: "ready" }, null, undefined),
      true,
    );
  });
});

// -----------------------------------------------------------------------
// readOnlyExplanation / temporaryBlockExplanation
// -----------------------------------------------------------------------

describe("readOnlyExplanation", function () {
  it("returns empty for ready", function () {
    assert.equal(readOnlyExplanation("ready"), "");
  });

  it("distinguishes the three known statuses", function () {
    var unavailable = readOnlyExplanation("profile_unavailable");
    var changed = readOnlyExplanation("model_changed");
    var legacy = readOnlyExplanation("legacy_unknown");

    assert.match(unavailable, /no longer available/i);
    assert.match(changed, /configuration has changed/i);
    assert.match(legacy, /before model tracking/i);

    // No cross-contamination of the distinguishing phrases
    assert.ok(!/no longer available/i.test(changed));
    assert.ok(!/configuration has changed/i.test(legacy));
    assert.ok(!/before model tracking/i.test(unavailable));
  });

  it("returns a generic read-only message for unknown statuses", function () {
    var text = readOnlyExplanation("weird_status");
    assert.match(text, /cannot accept new messages/i);
  });
});

describe("temporaryBlockExplanation", function () {
  it("uses the unavailable explanation for profile_unavailable", function () {
    var text = temporaryBlockExplanation("profile_unavailable");
    assert.match(text, /no longer available/i);
  });

  it("uses a generic conflict message that guesses nothing", function () {
    var text = temporaryBlockExplanation("conflict");
    assert.match(text, /compatibility issue/i);
    // Must not guess model_changed or legacy_unknown
    assert.ok(!/configuration has changed/i.test(text));
    assert.ok(!/before model tracking/i.test(text));
  });

  it("returns null for unknown block values", function () {
    assert.equal(temporaryBlockExplanation(null), null);
    assert.equal(temporaryBlockExplanation(undefined), null);
    assert.equal(temporaryBlockExplanation("weird"), null);
  });
});

// -----------------------------------------------------------------------
// profileRegistryStatusText
// -----------------------------------------------------------------------

describe("profileRegistryStatusText", function () {
  it("returns empty for a valid registry with no load error", function () {
    assert.equal(profileRegistryStatusText("valid", null), "");
  });

  it("prioritises a load error", function () {
    var text = profileRegistryStatusText("valid", "Network error.");
    assert.match(text, /Models unavailable\./);
    assert.match(text, /Network error\./);
  });

  it("maps every invalid status to accurate persistent text", function () {
    assert.match(profileRegistryStatusText("empty", null), /No models available/);
    assert.match(
      profileRegistryStatusText("invalid_response", null),
      /invalid model list/,
    );
    assert.match(
      profileRegistryStatusText("duplicate_ids", null),
      /duplicate model IDs/,
    );
    assert.match(
      profileRegistryStatusText("missing_default", null),
      /no default model/,
    );
    assert.match(
      profileRegistryStatusText("multiple_defaults", null),
      /multiple default models/,
    );
  });
});

// -----------------------------------------------------------------------
// buildCreateSessionPayload / profileKindBadgeText
// -----------------------------------------------------------------------

describe("buildCreateSessionPayload", function () {
  it("builds the exact payload shape", function () {
    assert.deepStrictEqual(
      buildCreateSessionPayload("local"),
      { llm_profile_id: "local" },
    );
    assert.deepStrictEqual(
      buildCreateSessionPayload("default"),
      { llm_profile_id: "default" },
    );
  });
});

describe("profileKindBadgeText", function () {
  it("maps the three known kinds", function () {
    assert.equal(profileKindBadgeText("api"), "API");
    assert.equal(profileKindBadgeText("local"), "Local");
    assert.equal(profileKindBadgeText("fake"), "Fake");
  });

  it("returns null for unknown kinds", function () {
    assert.equal(profileKindBadgeText("quantum"), null);
    assert.equal(profileKindBadgeText(undefined), null);
    assert.equal(profileKindBadgeText(42), null);
  });
});

// -----------------------------------------------------------------------
// isValidApiTimestamp
// -----------------------------------------------------------------------

describe("isValidApiTimestamp", function () {
  it("accepts ISO strings with and without a timezone suffix", function () {
    assert.equal(isValidApiTimestamp("2026-08-06T12:00:00"), true);
    assert.equal(isValidApiTimestamp("2026-08-06T12:00:00Z"), true);
    assert.equal(isValidApiTimestamp("2026-08-06T12:00:00+08:00"), true);
  });

  it("rejects non-strings, empty strings and unparseable values", function () {
    assert.equal(isValidApiTimestamp(null), false);
    assert.equal(isValidApiTimestamp(undefined), false);
    assert.equal(isValidApiTimestamp(42), false);
    assert.equal(isValidApiTimestamp(""), false);
    assert.equal(isValidApiTimestamp("not a date"), false);
  });
});

// -----------------------------------------------------------------------
// isValidSessionResponse
// -----------------------------------------------------------------------

function makeSession(extra) {
  var s = {
    id: 1,
    title: "Test",
    created_at: "2026-08-06T12:00:00",
    updated_at: "2026-08-06T12:00:00",
    llm_profile_id: "default",
    llm_profile_label: "Default",
    llm_profile_status: "ready",
    llm_model_snapshot: null,
  };
  if (extra) {
    Object.keys(extra).forEach(function (k) { s[k] = extra[k]; });
  }
  return s;
}

describe("isValidSessionResponse", function () {
  it("accepts a complete valid session", function () {
    assert.equal(isValidSessionResponse(makeSession()), true);
  });

  it("ignores extra fields", function () {
    assert.equal(
      isValidSessionResponse(makeSession({ something_extra: 42 })),
      true,
    );
  });

  it("rejects null, arrays and strings", function () {
    assert.equal(isValidSessionResponse(null), false);
    assert.equal(isValidSessionResponse([]), false);
    assert.equal(isValidSessionResponse("session"), false);
  });

  it("rejects missing id", function () {
    var s = makeSession();
    delete s.id;
    assert.equal(isValidSessionResponse(s), false);
  });

  it("rejects non-positive, fractional, NaN and Infinity ids", function () {
    var badIds = [0, -1, 1.5, NaN, Infinity, "1"];
    for (var i = 0; i < badIds.length; i++) {
      assert.equal(
        isValidSessionResponse(makeSession({ id: badIds[i] })),
        false,
        "id " + badIds[i],
      );
    }
  });

  it("rejects missing title", function () {
    var s = makeSession();
    delete s.title;
    assert.equal(isValidSessionResponse(s), false);
  });

  it("rejects missing or unparseable created_at", function () {
    var s1 = makeSession();
    delete s1.created_at;
    assert.equal(isValidSessionResponse(s1), false);

    assert.equal(
      isValidSessionResponse(makeSession({ created_at: "garbage" })),
      false,
    );
  });

  it("rejects missing or unparseable updated_at", function () {
    var s1 = makeSession();
    delete s1.updated_at;
    assert.equal(isValidSessionResponse(s1), false);

    assert.equal(
      isValidSessionResponse(makeSession({ updated_at: "" })),
      false,
    );
  });

  it("rejects missing llm_profile_label", function () {
    var s = makeSession();
    delete s.llm_profile_label;
    assert.equal(isValidSessionResponse(s), false);
  });

  it("rejects a blank or missing llm_profile_id", function () {
    assert.equal(
      isValidSessionResponse(makeSession({ llm_profile_id: "" })),
      false,
    );
    var s = makeSession();
    delete s.llm_profile_id;
    assert.equal(isValidSessionResponse(s), false);
  });

  it("accepts exactly the four authoritative statuses", function () {
    ["ready", "profile_unavailable", "model_changed", "legacy_unknown"]
      .forEach(function (status) {
        assert.equal(
          isValidSessionResponse(makeSession({ llm_profile_status: status })),
          true,
          status,
        );
      });
    assert.equal(
      isValidSessionResponse(makeSession({ llm_profile_status: "weird" })),
      false,
    );
  });

  it("accepts llm_model_snapshot as string or null only", function () {
    assert.equal(
      isValidSessionResponse(makeSession({ llm_model_snapshot: "fake" })),
      true,
    );
    assert.equal(
      isValidSessionResponse(makeSession({ llm_model_snapshot: 42 })),
      false,
    );
    assert.equal(
      isValidSessionResponse(makeSession({ llm_model_snapshot: undefined })),
      false,
    );
  });
});

// -----------------------------------------------------------------------
// isValidSessionList
// -----------------------------------------------------------------------

describe("isValidSessionList", function () {
  it("accepts a valid list", function () {
    assert.equal(
      isValidSessionList([makeSession({ id: 1 }), makeSession({ id: 2 })]),
      true,
    );
  });

  it("accepts an empty list", function () {
    assert.equal(isValidSessionList([]), true);
  });

  it("rejects non-arrays", function () {
    assert.equal(isValidSessionList(null), false);
    assert.equal(isValidSessionList({}), false);
    assert.equal(isValidSessionList("list"), false);
  });

  it("rejects a list containing an invalid element", function () {
    var bad = makeSession({ id: 1 });
    delete bad.title;
    assert.equal(
      isValidSessionList([makeSession({ id: 2 }), bad]),
      false,
    );
  });

  it("rejects a list with duplicate session ids", function () {
    assert.equal(
      isValidSessionList([makeSession({ id: 7 }), makeSession({ id: 7 })]),
      false,
    );
  });
});

// -----------------------------------------------------------------------
// isValidMessageResponse
// -----------------------------------------------------------------------

function makeMessage(extra) {
  var m = {
    id: 5,
    session_id: 1,
    role: "user",
    content: "hello",
    created_at: "2026-08-06T12:00:01",
    // Legacy messages carry explicit null snapshots — the backend
    // always returns the three fields.
    llm_profile_id_snapshot: null,
    llm_profile_kind_snapshot: null,
    llm_model_snapshot: null,
  };
  if (extra) {
    Object.keys(extra).forEach(function (k) { m[k] = extra[k]; });
  }
  return m;
}

function makeTrackedMessage(kind, model) {
  return makeMessage({
    llm_profile_id_snapshot: "default",
    llm_profile_kind_snapshot: kind,
    llm_model_snapshot: model,
  });
}

describe("isValidMessageResponse", function () {
  it("accepts valid user and assistant messages", function () {
    assert.equal(isValidMessageResponse(makeMessage()), true);
    assert.equal(
      isValidMessageResponse(makeMessage({ role: "assistant" })),
      true,
    );
  });

  it("rejects null, arrays and non-objects", function () {
    assert.equal(isValidMessageResponse(null), false);
    assert.equal(isValidMessageResponse([]), false);
    assert.equal(isValidMessageResponse("msg"), false);
  });

  it("rejects missing role, content or id", function () {
    var m1 = makeMessage();
    delete m1.role;
    assert.equal(isValidMessageResponse(m1), false);

    var m2 = makeMessage();
    delete m2.content;
    assert.equal(isValidMessageResponse(m2), false);

    var m3 = makeMessage();
    delete m3.id;
    assert.equal(isValidMessageResponse(m3), false);
  });

  it("rejects invalid roles", function () {
    assert.equal(
      isValidMessageResponse(makeMessage({ role: "system" })),
      false,
    );
  });

  it("rejects non-safe-integer ids and session_ids", function () {
    assert.equal(
      isValidMessageResponse(makeMessage({ id: 1.5 })),
      false,
    );
    assert.equal(
      isValidMessageResponse(makeMessage({ session_id: 0 })),
      false,
    );
  });

  it("rejects missing or unparseable created_at", function () {
    var m = makeMessage();
    delete m.created_at;
    assert.equal(isValidMessageResponse(m), false);
    assert.equal(
      isValidMessageResponse(makeMessage({ created_at: "nope" })),
      false,
    );
  });
});

// -----------------------------------------------------------------------
// isValidMessageList
// -----------------------------------------------------------------------

describe("isValidMessageList", function () {
  it("accepts a valid empty list", function () {
    assert.equal(isValidMessageList([], 1), true);
  });

  it("accepts a valid strictly increasing user/assistant list", function () {
    var list = [
      makeMessage({ id: 1, session_id: 1, role: "user" }),
      makeMessage({ id: 2, session_id: 1, role: "assistant" }),
      makeMessage({ id: 5, session_id: 1, role: "user" }),
    ];
    assert.equal(isValidMessageList(list, 1), true);
  });

  it("rejects non-arrays (null, object, string)", function () {
    assert.equal(isValidMessageList(null, 1), false);
    assert.equal(isValidMessageList({}, 1), false);
    assert.equal(isValidMessageList("messages", 1), false);
  });

  it("rejects a list containing null", function () {
    assert.equal(
      isValidMessageList([makeMessage({ id: 1 }), null], 1),
      false,
    );
  });

  it("rejects a list containing an invalid MessageResponse", function () {
    var bad = makeMessage({ id: 2 });
    delete bad.role;
    assert.equal(
      isValidMessageList([makeMessage({ id: 1 }), bad], 1),
      false,
    );
  });

  it("rejects messages belonging to another session", function () {
    assert.equal(
      isValidMessageList([makeMessage({ id: 1, session_id: 99 })], 1),
      false,
    );
  });

  it("rejects duplicate message ids", function () {
    assert.equal(
      isValidMessageList([
        makeMessage({ id: 3 }),
        makeMessage({ id: 3, role: "assistant" }),
      ], 1),
      false,
    );
  });

  it("rejects equal or decreasing message ids", function () {
    assert.equal(
      isValidMessageList([
        makeMessage({ id: 2 }),
        makeMessage({ id: 2, role: "assistant" }),
      ], 1),
      false,
    );
    assert.equal(
      isValidMessageList([
        makeMessage({ id: 5 }),
        makeMessage({ id: 3, role: "assistant" }),
      ], 1),
      false,
    );
  });

  it("rejects an invalid expectedSessionId", function () {
    [NaN, Infinity, 1.5, 0, -1, "1", null, undefined].forEach(function (bad) {
      assert.equal(
        isValidMessageList([makeMessage({ id: 1 })], bad),
        false,
        "expectedSessionId " + bad,
      );
    });
  });

  it("does not mutate the input list", function () {
    var list = [makeMessage({ id: 1 }), makeMessage({ id: 2, role: "assistant" })];
    var snapshot = JSON.stringify(list);
    isValidMessageList(list, 1);
    assert.equal(JSON.stringify(list), snapshot);
  });
});

// -----------------------------------------------------------------------
// isValidSendMessageResponse
// -----------------------------------------------------------------------

function makeSendResponse(extra) {
  var r = {
    user_message: makeMessage({ id: 1, session_id: 1, role: "user" }),
    assistant_message: makeMessage({
      id: 2, session_id: 1, role: "assistant",
    }),
  };
  if (extra) {
    Object.keys(extra).forEach(function (k) { r[k] = extra[k]; });
  }
  return r;
}

describe("isValidSendMessageResponse", function () {
  it("accepts a valid response for the expected session", function () {
    assert.equal(isValidSendMessageResponse(makeSendResponse(), 1), true);
  });

  it("rejects an invalid expectedSessionId", function () {
    [NaN, Infinity, 1.5, 0, -1, "1", null, undefined]
      .forEach(function (badId) {
        assert.equal(
          isValidSendMessageResponse(makeSendResponse(), badId),
          false,
          "expectedSessionId " + badId,
        );
      });
  });

  it("rejects non-object responses", function () {
    assert.equal(isValidSendMessageResponse(null, 1), false);
    assert.equal(isValidSendMessageResponse([], 1), false);
    assert.equal(isValidSendMessageResponse("ok", 1), false);
  });

  it("rejects a null user_message", function () {
    assert.equal(
      isValidSendMessageResponse(
        makeSendResponse({ user_message: null }),
        1,
      ),
      false,
    );
  });

  it("rejects swapped roles", function () {
    assert.equal(
      isValidSendMessageResponse(makeSendResponse({
        user_message: makeMessage({ id: 1, session_id: 1, role: "assistant" }),
        assistant_message: makeMessage({ id: 2, session_id: 1, role: "user" }),
      }), 1),
      false,
    );
  });

  it("rejects messages from a different session", function () {
    assert.equal(
      isValidSendMessageResponse(makeSendResponse({
        assistant_message: makeMessage({ id: 2, session_id: 99, role: "assistant" }),
      }), 1),
      false,
    );
    assert.equal(
      isValidSendMessageResponse(makeSendResponse({
        user_message: makeMessage({ id: 1, session_id: 99, role: "user" }),
      }), 1),
      false,
    );
  });
});

// -----------------------------------------------------------------------
// profileKindFromRegistry
// -----------------------------------------------------------------------

describe("profileKindFromRegistry", function () {
  var validRegistry = analyzeProfileRegistry([
    makeProfile("default", { is_default: true, kind: "fake", model: "fake" }),
    makeProfile("local", { kind: "local", model: "llama3" }),
  ]);

  it("returns the kind for a matching id in a valid registry", function () {
    assert.equal(profileKindFromRegistry(validRegistry, "local"), "local");
    assert.equal(profileKindFromRegistry(validRegistry, "default"), "fake");
  });

  it("returns null when the id does not match", function () {
    assert.equal(profileKindFromRegistry(validRegistry, "missing"), null);
  });

  it("returns null for every non-valid registry status", function () {
    var statuses = [
      analyzeProfileRegistry([]),                                     // empty
      analyzeProfileRegistry(null),                                   // invalid_response
      analyzeProfileRegistry([makeProfile("a"), makeProfile("a")]),   // duplicate_ids
      analyzeProfileRegistry([makeProfile("a")]),                     // missing_default
      analyzeProfileRegistry([
        makeProfile("a", { is_default: true }),
        makeProfile("b", { is_default: true }),
      ]),                                                             // multiple_defaults
    ];
    for (var i = 0; i < statuses.length; i++) {
      // Even a raw matching id must not produce a kind.
      assert.equal(profileKindFromRegistry(statuses[i], "default"), null);
      assert.equal(profileKindFromRegistry(statuses[i], "a"), null);
    }
  });

  it("returns null for null or non-object registry values", function () {
    assert.equal(profileKindFromRegistry(null, "default"), null);
    assert.equal(profileKindFromRegistry("valid", "default"), null);
    assert.equal(profileKindFromRegistry({ status: "valid" }, "default"), null);
  });

  it("a forged valid registry with a malformed element yields null", function () {
    var forged = {
      status: "valid",
      profiles: [
        null,                       // malformed element BEFORE a match
        makeProfile("local", { kind: "local" }),
      ],
    };
    assert.equal(profileKindFromRegistry(forged, "local"), null);

    var forged2 = {
      status: "valid",
      profiles: [
        makeProfile("local", { kind: "local" }),
        null,                       // malformed element AFTER a match
      ],
    };
    assert.equal(profileKindFromRegistry(forged2, "local"), null);
  });

  it("non-valid statuses with otherwise valid profiles yield null", function () {
    var validProfiles = [
      makeProfile("default", { is_default: true, kind: "fake", model: "fake" }),
      makeProfile("local", { kind: "local" }),
    ];
    var statuses = [
      "invalid_response", "empty", "duplicate_ids",
      "missing_default", "multiple_defaults",
    ];
    for (var i = 0; i < statuses.length; i++) {
      var forged = { status: statuses[i], profiles: validProfiles };
      assert.equal(
        profileKindFromRegistry(forged, "local"),
        null,
        "status " + statuses[i],
      );
    }
  });

  it("an array as registry yields null", function () {
    assert.equal(
      profileKindFromRegistry([{ status: "valid", profiles: [] }], "local"),
      null,
    );
  });

  it("a valid registry still matches normally (regression)", function () {
    assert.equal(profileKindFromRegistry(validRegistry, "local"), "local");
    assert.equal(profileKindFromRegistry(validRegistry, "default"), "fake");
  });
});

// -----------------------------------------------------------------------
// resolveNextSelectionId
// -----------------------------------------------------------------------

describe("resolveNextSelectionId", function () {
  var list = [makeSession({ id: 5 }), makeSession({ id: 9 })];

  it("returns null with changed=true for an empty list after a selection", function () {
    var r = resolveNextSelectionId([], 5, true);
    assert.deepStrictEqual(r, { selectionId: null, changed: true });
  });

  it("returns null with changed=false for an empty list with no selection", function () {
    var r = resolveNextSelectionId([], null, true);
    assert.deepStrictEqual(r, { selectionId: null, changed: false });
  });

  it("keeps the current selection when preserved and present", function () {
    var r = resolveNextSelectionId(list, 9, true);
    assert.deepStrictEqual(r, { selectionId: 9, changed: false });
  });

  it("falls back to the first session when the selection disappeared", function () {
    var r = resolveNextSelectionId(list, 42, true);
    assert.deepStrictEqual(r, { selectionId: 5, changed: true });
  });

  it("always selects the first session when preserveSelection is false", function () {
    var r = resolveNextSelectionId(list, 9, false);
    assert.deepStrictEqual(r, { selectionId: 5, changed: true });
  });

  it("selects the first session when there is no current selection", function () {
    var r = resolveNextSelectionId(list, null, true);
    assert.deepStrictEqual(r, { selectionId: 5, changed: true });
  });

  it("never throws on a non-array input", function () {
    var r = resolveNextSelectionId(null, 5, true);
    assert.deepStrictEqual(r, { selectionId: null, changed: false });
  });

  it("never throws when the list contains malformed elements", function () {
    var badList = [null, makeSession({ id: 7 })];
    var r = resolveNextSelectionId(badList, null, true);
    // first element is malformed → defensive null selection
    assert.equal(r.selectionId, null);
  });
});

// -----------------------------------------------------------------------
// findSessionButton
// -----------------------------------------------------------------------

function makeFakeButton(sessionId) {
  return {
    disabled: false,
    dataset: sessionId === undefined
      ? undefined
      : { sessionId: String(sessionId) },
  };
}

describe("findSessionButton", function () {
  var buttons = [makeFakeButton(3), makeFakeButton(7), makeFakeButton(9)];

  it("finds the matching button", function () {
    assert.equal(findSessionButton(buttons, 7), buttons[1]);
  });

  it("compares numeric session ids against string dataset values", function () {
    assert.equal(findSessionButton(buttons, "9"), buttons[2]);
  });

  it("returns null when nothing matches", function () {
    assert.equal(findSessionButton(buttons, 42), null);
  });

  it("returns null for null, undefined and non-array-like input", function () {
    assert.equal(findSessionButton(null, 7), null);
    assert.equal(findSessionButton(undefined, 7), null);
    assert.equal(findSessionButton("buttons", 7), null);
    assert.equal(findSessionButton(buttons, null), null);
    assert.equal(findSessionButton(buttons, undefined), null);
  });

  it("skips elements without a dataset safely", function () {
    var mixed = [makeFakeButton(), makeFakeButton(7)];
    assert.equal(findSessionButton(mixed, 7), mixed[1]);
  });
});

// -----------------------------------------------------------------------
// resolveSessionProfileDraft
// -----------------------------------------------------------------------

describe("resolveSessionProfileDraft", function () {
  var apiLocalRegistry = analyzeProfileRegistry([
    makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
    makeProfile("local", { kind: "local", model: "qwen" }),
  ]);

  it("ready session returns its binding id", function () {
    assert.equal(
      resolveSessionProfileDraft(
        makeSession({ llm_profile_id: "local", llm_model_snapshot: "qwen" }),
        apiLocalRegistry,
      ),
      "local",
    );
  });

  it("model_changed returns the same id for repair", function () {
    assert.equal(
      resolveSessionProfileDraft(
        makeSession({
          llm_profile_status: "model_changed",
          llm_model_snapshot: "stale",
        }),
        apiLocalRegistry,
      ),
      "default",
    );
  });

  it("legacy_unknown returns the same id for repair", function () {
    assert.equal(
      resolveSessionProfileDraft(
        makeSession({ llm_profile_status: "legacy_unknown" }),
        apiLocalRegistry,
      ),
      "default",
    );
  });

  it("profile_unavailable with absent id returns null (no fallback)", function () {
    assert.equal(
      resolveSessionProfileDraft(
        makeSession({
          llm_profile_status: "profile_unavailable",
          llm_profile_id: "gone",
        }),
        apiLocalRegistry,
      ),
      null,
    );
  });

  it("profile_unavailable with a reappeared id returns that id", function () {
    assert.equal(
      resolveSessionProfileDraft(
        makeSession({ llm_profile_status: "profile_unavailable" }),
        apiLocalRegistry,
      ),
      "default",
    );
  });

  it("invalid registry or invalid session yields null", function () {
    var bad = analyzeProfileRegistry([makeProfile("a"), makeProfile("b")]);
    assert.equal(
      resolveSessionProfileDraft(
        makeSession(),
        bad,
      ),
      null,
    );
    assert.equal(
      resolveSessionProfileDraft(null, apiLocalRegistry),
      null,
    );
    assert.equal(resolveSessionProfileDraft(undefined, apiLocalRegistry), null);
  });
});

// -----------------------------------------------------------------------
// planProfileDraftForSelection
// -----------------------------------------------------------------------

describe("planProfileDraftForSelection", function () {
  var registry = analyzeProfileRegistry([
    makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
    makeProfile("local", { kind: "local", model: "qwen" }),
  ]);

  function twoSessions() {
    return [
      makeSession({ id: 1, llm_profile_id: "default" }),
      makeSession({ id: 2, llm_profile_id: "default" }),
    ];
  }

  it("A → B derives the draft from B and never inherits A's draft", function () {
    // A (id 1) is bound to "default" but the user picked "local"
    // without applying; after the programmatic switch the draft must
    // come from B's server binding, not from A's un-applied choice.
    var plan = planProfileDraftForSelection({
      sessions: twoSessions(),
      previousSessionId: 1,
      nextSessionId: 2,
      previousDraftId: "local",
      registry: registry,
    });
    assert.equal(plan.selectionChanged, true);
    assert.equal(plan.draftId, "default");
    assert.equal(plan.draftChanged, true);
  });

  it("A → B resolves B's own binding id", function () {
    var plan = planProfileDraftForSelection({
      sessions: [
        makeSession({ id: 1, llm_profile_id: "default" }),
        makeSession({ id: 2, llm_profile_id: "local",
          llm_model_snapshot: "qwen" }),
      ],
      previousSessionId: 1,
      nextSessionId: 2,
      previousDraftId: "default",
      registry: registry,
    });
    assert.equal(plan.selectionChanged, true);
    assert.equal(plan.draftId, "local");
  });

  it("A → null clears the draft and reports the change", function () {
    var plan = planProfileDraftForSelection({
      sessions: [],
      previousSessionId: 1,
      nextSessionId: null,
      previousDraftId: "local",
      registry: registry,
    });
    assert.equal(plan.selectionChanged, true);
    assert.equal(plan.draftId, null);
    assert.equal(plan.draftChanged, true);
  });

  it("an unchanged selection keeps the un-applied draft", function () {
    var plan = planProfileDraftForSelection({
      sessions: [makeSession({ id: 1, llm_profile_id: "default" })],
      previousSessionId: 1,
      nextSessionId: 1,
      previousDraftId: "local",
      registry: registry,
    });
    assert.equal(plan.selectionChanged, false);
    assert.equal(plan.draftChanged, false);
    assert.equal(plan.draftId, "local");
  });

  it("an unchanged selection keeps the draft even when the server binding differs", function () {
    // An ordinary list refresh must never overwrite the user's
    // un-applied choice, even if the fresh SessionResponse carries a
    // different binding.
    var plan = planProfileDraftForSelection({
      sessions: [makeSession({ id: 1, llm_profile_id: "default" })],
      previousSessionId: 1,
      nextSessionId: 1,
      previousDraftId: "local",
      registry: registry,
    });
    assert.equal(plan.draftId, "local");
  });

  it("deleting a NON-current session leaves the current draft alone", function () {
    // removeSessionLocally re-plans only when the deleted session was
    // current; the unchanged-selection rule keeps A's draft intact.
    var plan = planProfileDraftForSelection({
      sessions: [makeSession({ id: 1, llm_profile_id: "default" })],
      previousSessionId: 1,
      nextSessionId: 1,
      previousDraftId: "local",
      registry: registry,
    });
    assert.equal(plan.selectionChanged, false);
    assert.equal(plan.draftId, "local");
  });

  it("deleting the current session re-derives the draft from the first remaining session", function () {
    var plan = planProfileDraftForSelection({
      sessions: [
        makeSession({ id: 2, llm_profile_id: "local", llm_model_snapshot: "qwen" }),
        makeSession({ id: 3, llm_profile_id: "default" }),
      ],
      previousSessionId: 1,      // deleted
      nextSessionId: 2,          // first remaining
      previousDraftId: "default",// deleted session's draft
      registry: registry,
    });
    assert.equal(plan.selectionChanged, true);
    assert.equal(plan.draftId, "local");
  });

  it("a next id absent from the list fails closed to null", function () {
    var plan = planProfileDraftForSelection({
      sessions: twoSessions(),
      previousSessionId: 1,
      nextSessionId: 99,
      previousDraftId: "default",
      registry: registry,
    });
    assert.equal(plan.selectionChanged, true);
    assert.equal(plan.draftId, null);
  });

  it("malformed options fail closed without throwing", function () {
    var failClosed = {
      draftId: null, selectionChanged: false, draftChanged: false,
    };
    var cases = [
      null,
      undefined,
      "x",
      42,
      [],
      {},                       // plain object with no fields at all
      { previousSessionId: 1 }, // missing nextSessionId / previousDraftId
      { nextSessionId: 1 },
      { previousSessionId: 1, nextSessionId: 2 }, // missing draft
    ];
    cases.forEach(function (bad) {
      assert.deepStrictEqual(
        planProfileDraftForSelection(bad),
        failClosed,
        JSON.stringify(bad),
      );
    });
  });

  it("two invalid but equal ids never reach the unchanged branch", function () {
    // "1" === "1" would take the keep-the-draft branch without
    // validation; invalid ids must fail closed first.  Every other
    // field is complete and valid so the id check is what fires.
    var failClosed = {
      draftId: null, selectionChanged: false, draftChanged: false,
    };
    ["1", 0, -1, 1.5, true].forEach(function (badId) {
      assert.deepStrictEqual(
        planProfileDraftForSelection({
          sessions: [],
          registry: registry,
          previousSessionId: badId,
          nextSessionId: badId,
          previousDraftId: "local",
        }),
        failClosed,
        "id " + String(badId),
      );
    });
  });

  it("an invalid previousDraftId fails closed", function () {
    var failClosed = {
      draftId: null, selectionChanged: false, draftChanged: false,
    };
    ["", 42, true, {}, []].forEach(function (badDraft) {
      assert.deepStrictEqual(
        planProfileDraftForSelection({
          sessions: [],
          registry: registry,
          previousSessionId: 1,
          nextSessionId: 1,
          previousDraftId: badDraft,
        }),
        failClosed,
        "draft " + JSON.stringify(badDraft),
      );
    });
  });

  it("missing sessions fails closed even with valid equal ids", function () {
    // The buggy shape: both ids valid and equal would take the
    // keep-the-draft branch — the missing own sessions field must
    // fail closed first.
    assert.deepStrictEqual(
      planProfileDraftForSelection({
        registry: registry,
        previousSessionId: 1,
        nextSessionId: 1,
        previousDraftId: "local",
      }),
      { draftId: null, selectionChanged: false, draftChanged: false },
    );
  });

  it("non-array sessions fails closed", function () {
    var failClosed = {
      draftId: null, selectionChanged: false, draftChanged: false,
    };
    [null, {}, "x", 42].forEach(function (badSessions) {
      assert.deepStrictEqual(
        planProfileDraftForSelection({
          sessions: badSessions,
          registry: registry,
          previousSessionId: 1,
          nextSessionId: 1,
          previousDraftId: "local",
        }),
        failClosed,
        "sessions " + JSON.stringify(badSessions),
      );
    });
  });

  it("missing registry fails closed even when the rest is valid", function () {
    assert.deepStrictEqual(
      planProfileDraftForSelection({
        sessions: [],
        previousSessionId: 1,
        nextSessionId: 1,
        previousDraftId: "local",
      }),
      { draftId: null, selectionChanged: false, draftChanged: false },
    );
  });

  it("required fields inherited from the prototype are rejected", function () {
    // ``in`` would find each inherited field; only own properties may
    // satisfy the contract.
    var failClosed = {
      draftId: null, selectionChanged: false, draftChanged: false,
    };
    var base = {
      sessions: [],
      registry: registry,
      previousSessionId: 1,
      nextSessionId: 1,
      previousDraftId: "local",
    };
    Object.keys(base).forEach(function (field) {
      var proto = {};
      proto[field] = base[field];
      var options = Object.create(proto);
      Object.keys(base).forEach(function (k) {
        if (k !== field) options[k] = base[k];
      });
      assert.deepStrictEqual(
        planProfileDraftForSelection(options),
        failClosed,
        "inherited field " + field,
      );
    });
  });

  it("whitespace-only previousDraftId fails closed", function () {
    var failClosed = {
      draftId: null, selectionChanged: false, draftChanged: false,
    };
    [" ", "\t", "\n", " \t\n "].forEach(function (blankDraft) {
      assert.deepStrictEqual(
        planProfileDraftForSelection({
          sessions: [],
          registry: registry,
          previousSessionId: 1,
          nextSessionId: 1,
          previousDraftId: blankDraft,
        }),
        failClosed,
        "draft " + JSON.stringify(blankDraft),
      );
    });
  });

  it("a draft with surrounding whitespace is valid and kept verbatim", function () {
    // Non-blank after trim: the unchanged branch returns the draft
    // exactly as given — the function never trims or rewrites it.
    var plan = planProfileDraftForSelection({
      sessions: [],
      registry: registry,
      previousSessionId: 1,
      nextSessionId: 1,
      previousDraftId: "  local  ",
    });
    assert.deepStrictEqual(plan, {
      draftId: "  local  ",
      selectionChanged: false,
      draftChanged: false,
    });
  });

  it("does not mutate the options object on the fail-closed paths", function () {
    var cases = [
      {
        previousSessionId: "1",
        nextSessionId: "1",
        previousDraftId: "local",
      },
      {
        sessions: [],
        registry: registry,
        previousSessionId: 1,
        nextSessionId: 1,
        previousDraftId: "   ",
      },
    ];
    cases.forEach(function (options) {
      var json = JSON.stringify(options);
      planProfileDraftForSelection(options);
      assert.equal(JSON.stringify(options), json, JSON.stringify(options));
    });
  });

  it("does not mutate the sessions array or its elements", function () {
    var sessions = twoSessions();
    var json = JSON.stringify(sessions);
    planProfileDraftForSelection({
      sessions: sessions,
      previousSessionId: 1,
      nextSessionId: 2,
      previousDraftId: "local",
      registry: registry,
    });
    assert.equal(JSON.stringify(sessions), json);
  });

  it("a throwing getter on any required field fails closed without throwing", function () {
    var failClosed = {
      draftId: null, selectionChanged: false, draftChanged: false,
    };
    var base = {
      sessions: [],
      registry: registry,
      previousSessionId: 1,
      nextSessionId: 1,
      previousDraftId: "local",
    };
    Object.keys(base).forEach(function (field) {
      var options = {};
      Object.keys(base).forEach(function (k) {
        if (k === field) {
          Object.defineProperty(options, k, {
            enumerable: true,
            configurable: true,
            get: function () { throw new Error("getter boom"); },
          });
        } else {
          options[k] = base[k];
        }
      });
      assert.deepStrictEqual(
        planProfileDraftForSelection(options),
        failClosed,
        "field " + field,
      );
    });
  });

  it("a session element with a throwing id getter fails closed without throwing", function () {
    var hostileSession = {};
    Object.defineProperty(hostileSession, "id", {
      enumerable: true,
      configurable: true,
      get: function () { throw new Error("id getter boom"); },
    });
    assert.deepStrictEqual(
      planProfileDraftForSelection({
        sessions: [hostileSession],
        registry: registry,
        previousSessionId: 1,
        nextSessionId: 2,
        previousDraftId: "local",
      }),
      { draftId: null, selectionChanged: false, draftChanged: false },
    );
  });

  it("a hostile session inside resolveSessionProfileDraft fails closed without throwing", function () {
    // The id matches the target, so resolveSessionProfileDraft runs
    // and its field reads throw — the wrapper must still fail closed.
    var hostile = makeSession({ id: 2 });
    Object.defineProperty(hostile, "llm_profile_id", {
      enumerable: true,
      configurable: true,
      get: function () { throw new Error("profile getter boom"); },
    });
    assert.deepStrictEqual(
      planProfileDraftForSelection({
        sessions: [hostile],
        registry: registry,
        previousSessionId: 1,
        nextSessionId: 2,
        previousDraftId: "local",
      }),
      { draftId: null, selectionChanged: false, draftChanged: false },
    );
  });

  it("a hostile registry Proxy inside resolveSessionProfileDraft fails closed without throwing", function () {
    var registryProxy = new Proxy({}, {
      get: function () { throw new Error("registry trap boom"); },
    });
    assert.deepStrictEqual(
      planProfileDraftForSelection({
        sessions: [makeSession({ id: 2 })],
        registry: registryProxy,
        previousSessionId: 1,
        nextSessionId: 2,
        previousDraftId: "local",
      }),
      { draftId: null, selectionChanged: false, draftChanged: false },
    );
  });

  it("a Proxy options whose getOwnPropertyDescriptor trap throws fails closed", function () {
    var proxy = new Proxy({}, {
      getOwnPropertyDescriptor: function () {
        throw new Error("descriptor trap boom");
      },
    });
    assert.deepStrictEqual(
      planProfileDraftForSelection(proxy),
      { draftId: null, selectionChanged: false, draftChanged: false },
    );
  });

  it("a Proxy options whose field-read get trap throws fails closed", function () {
    // All five own fields exist on the target so the own-property
    // checks pass; the first field READ then hits the throwing trap.
    var target = {
      sessions: [],
      registry: registry,
      previousSessionId: 1,
      nextSessionId: 1,
      previousDraftId: "local",
    };
    var proxy = new Proxy(target, {
      get: function (t, prop) {
        if (prop === "sessions") { throw new Error("get trap boom"); }
        return t[prop];
      },
    });
    assert.deepStrictEqual(
      planProfileDraftForSelection(proxy),
      { draftId: null, selectionChanged: false, draftChanged: false },
    );
  });

  it("a Proxy sessions array with a throwing get trap fails closed", function () {
    var sessionsProxy = new Proxy([], {
      get: function () { throw new Error("sessions trap boom"); },
    });
    assert.deepStrictEqual(
      planProfileDraftForSelection({
        sessions: sessionsProxy,
        registry: registry,
        previousSessionId: 1,
        nextSessionId: 2,
        previousDraftId: "local",
      }),
      { draftId: null, selectionChanged: false, draftChanged: false },
    );
  });
});

// -----------------------------------------------------------------------
// canApplySessionProfile
// -----------------------------------------------------------------------

describe("canApplySessionProfile", function () {
  var registry = analyzeProfileRegistry([
    makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
    makeProfile("local", { kind: "local", model: "qwen" }),
  ]);

  function opts(overrides) {
    var base = {
      session: makeSession({ llm_model_snapshot: "m1" }),
      registry: registry,
      draftProfileId: "default",
      isSwitching: false,
    };
    if (overrides) {
      Object.keys(overrides).forEach(function (k) { base[k] = overrides[k]; });
    }
    return base;
  }

  it("ready + same draft is strictly idempotent (false)", function () {
    assert.equal(canApplySessionProfile(opts()), false);
  });

  it("ready + different draft is true", function () {
    assert.equal(
      canApplySessionProfile(opts({ draftProfileId: "local" })),
      true,
    );
  });

  it("non-ready statuses allow the same id for repair", function () {
    ["model_changed", "legacy_unknown", "profile_unavailable"]
      .forEach(function (status) {
        assert.equal(
          canApplySessionProfile(opts({
            session: makeSession({
              llm_profile_status: status,
              llm_model_snapshot: null,
            }),
          })),
          true,
          status,
        );
      });
  });

  it("isSwitching must be exactly false", function () {
    [true, undefined, null, 0, "", "no"].forEach(function (bad) {
      assert.equal(
        canApplySessionProfile(opts({ isSwitching: bad, draftProfileId: "local" })),
        false,
        "isSwitching " + bad,
      );
    });
  });

  it("invalid draft / registry / session yield false", function () {
    assert.equal(canApplySessionProfile(opts({ draftProfileId: "gone" })), false);
    assert.equal(
      canApplySessionProfile(opts({ registry: analyzeProfileRegistry([]) })),
      false,
    );
    assert.equal(canApplySessionProfile(opts({ session: null })), false);
  });

  it("null / undefined / array / string options never throw", function () {
    [null, undefined, [], "x", 42].forEach(function (bad) {
      assert.equal(canApplySessionProfile(bad), false);
    });
  });
});

// -----------------------------------------------------------------------
// canApplySessionProfileWithUncertain
// -----------------------------------------------------------------------

describe("canApplySessionProfileWithUncertain", function () {
  var registry = analyzeProfileRegistry([
    makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
    makeProfile("local", { kind: "local", model: "qwen" }),
  ]);

  function opts(overrides) {
    var base = {
      session: makeSession({ llm_model_snapshot: "m1" }),
      registry: registry,
      draftProfileId: "default",
      uncertain: false,
    };
    if (overrides) {
      Object.keys(overrides).forEach(function (k) { base[k] = overrides[k]; });
    }
    return base;
  }

  it("ready + same draft + uncertain record keeps Apply available", function () {
    // The ordinary contract rejects the idempotent same-binding case,
    // but the convergence re-apply must stay possible.
    assert.equal(
      canApplySessionProfileWithUncertain(opts({ uncertain: true })),
      true,
    );
  });

  it("ready + same draft without the record stays idempotent", function () {
    assert.equal(canApplySessionProfileWithUncertain(opts()), false);
  });

  it("uncertain with a draft outside the registry is false", function () {
    assert.equal(
      canApplySessionProfileWithUncertain(
        opts({ uncertain: true, draftProfileId: "gone" }),
      ),
      false,
    );
  });

  it("uncertain with an invalid session fails closed", function () {
    assert.equal(
      canApplySessionProfileWithUncertain(opts({ uncertain: true, session: null })),
      false,
    );
  });

  it("the ordinary rule is untouched when uncertain is false", function () {
    assert.equal(
      canApplySessionProfileWithUncertain(opts({ draftProfileId: "local" })),
      true,
    );
  });

  it("uncertain must be exactly true", function () {
    ["yes", 1, null, undefined].forEach(function (bad) {
      assert.equal(
        canApplySessionProfileWithUncertain(opts({ uncertain: bad })),
        false,
        "uncertain " + bad,
      );
    });
  });

  it("null / undefined / array / string options never throw", function () {
    [null, undefined, [], "x", 42].forEach(function (bad) {
      assert.equal(canApplySessionProfileWithUncertain(bad), false);
    });
  });

  it("does not mutate the session or registry", function () {
    var session = makeSession({ llm_model_snapshot: "m1" });
    var sessionJson = JSON.stringify(session);
    var registryJson = JSON.stringify(registry);
    canApplySessionProfileWithUncertain({
      session: session,
      registry: registry,
      draftProfileId: "default",
      uncertain: true,
    });
    assert.equal(JSON.stringify(session), sessionJson);
    assert.equal(JSON.stringify(registry), registryJson);
  });
});

// -----------------------------------------------------------------------
// buildSwitchSessionProfilePayload
// -----------------------------------------------------------------------

describe("buildSwitchSessionProfilePayload", function () {
  it("builds the exact two-key payload with ack true", function () {
    assert.deepStrictEqual(
      buildSwitchSessionProfilePayload("local", true),
      { llm_profile_id: "local", acknowledge_remote_history: true },
    );
  });

  it("builds the exact two-key payload with ack false", function () {
    assert.deepStrictEqual(
      buildSwitchSessionProfilePayload("default", false),
      { llm_profile_id: "default", acknowledge_remote_history: false },
    );
  });

  it("normalises non-strict truthiness to a strict boolean", function () {
    assert.deepStrictEqual(
      buildSwitchSessionProfilePayload("local", "yes"),
      { llm_profile_id: "local", acknowledge_remote_history: false },
    );
  });
});

// -----------------------------------------------------------------------
// needsRemoteHistoryConfirmation
// -----------------------------------------------------------------------

describe("needsRemoteHistoryConfirmation", function () {
  var registry = analyzeProfileRegistry([
    makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
    makeProfile("api-b", { kind: "api", model: "m2" }),
    makeProfile("local", { kind: "local", model: "qwen" }),
  ]);

  function opts(overrides) {
    var base = {
      session: makeSession({
        llm_profile_id: "local",
        llm_model_snapshot: "qwen",
      }),
      targetProfileId: "default",
      registry: registry,
      historyState: "present",
    };
    if (overrides) {
      Object.keys(overrides).forEach(function (k) { base[k] = overrides[k]; });
    }
    return base;
  }

  it("Local → API: present/unknown need confirmation, empty does not", function () {
    assert.equal(needsRemoteHistoryConfirmation(opts({ historyState: "present" })), true);
    assert.equal(needsRemoteHistoryConfirmation(opts({ historyState: "unknown" })), true);
    assert.equal(needsRemoteHistoryConfirmation(opts({ historyState: "empty" })), false);
  });

  it("API → Local never needs the confirmation", function () {
    assert.equal(
      needsRemoteHistoryConfirmation(opts({
        session: makeSession({ llm_model_snapshot: "m1" }),
        targetProfileId: "local",
      })),
      false,
    );
  });

  it("API A → API B with history needs the confirmation", function () {
    assert.equal(
      needsRemoteHistoryConfirmation(opts({
        session: makeSession({ llm_model_snapshot: "m1" }),
        targetProfileId: "api-b",
      })),
      true,
    );
  });

  it("same reliable API binding never needs it", function () {
    assert.equal(
      needsRemoteHistoryConfirmation(opts({
        session: makeSession({ llm_model_snapshot: "m1" }),
        targetProfileId: "default",
      })),
      false,
    );
  });

  it("stale snapshot on the same API id still needs it", function () {
    assert.equal(
      needsRemoteHistoryConfirmation(opts({
        session: makeSession({
          llm_profile_id: "default",
          llm_model_snapshot: "old",
        }),
        targetProfileId: "default",
      })),
      true,
    );
  });

  it("invalid historyState for an API target fails closed (true)", function () {
    ["none", null, undefined, 123, {}].forEach(function (bad) {
      assert.equal(
        needsRemoteHistoryConfirmation(opts({ historyState: bad })),
        true,
        "historyState " + bad,
      );
    });
  });

  it("null / undefined / array / string options never throw (fail closed true)", function () {
    [null, undefined, [], "x"].forEach(function (bad) {
      assert.equal(needsRemoteHistoryConfirmation(bad), true);
    });
  });
});

// -----------------------------------------------------------------------
// parseRemoteHistoryAckRequired
// -----------------------------------------------------------------------

describe("parseRemoteHistoryAckRequired", function () {
  var valid = {
    detail: {
      code: "remote_history_ack_required",
      message: "please confirm",
    },
  };

  it("parses the exact structure", function () {
    assert.deepStrictEqual(
      parseRemoteHistoryAckRequired(valid),
      { code: "remote_history_ack_required", message: "please confirm" },
    );
  });

  it("ignores extra fields on both levels", function () {
    assert.deepStrictEqual(
      parseRemoteHistoryAckRequired({
        request_id: "abc",
        detail: {
          code: "remote_history_ack_required",
          message: "please confirm",
          metadata: { x: 1 },
        },
      }),
      { code: "remote_history_ack_required", message: "please confirm" },
    );
  });

  it("returns null for malformed shapes", function () {
    var cases = [
      null,
      undefined,
      [],
      "text",
      {},
      { detail: "string detail" },
      { detail: null },
      { detail: [] },
      { detail: { code: "wrong_code", message: "x" } },
      { detail: { code: "remote_history_ack_required" } },
      { detail: { code: "remote_history_ack_required", message: "" } },
      { detail: { code: "remote_history_ack_required", message: "   " } },
      { detail: { code: "remote_history_ack_required", message: 42 } },
    ];
    cases.forEach(function (c) {
      assert.equal(parseRemoteHistoryAckRequired(c), null);
    });
  });
});

// -----------------------------------------------------------------------
// Message snapshot rules (triple)
// -----------------------------------------------------------------------

describe("isValidMessageResponse snapshot triple", function () {
  it("accepts legacy explicit nulls", function () {
    assert.equal(isValidMessageResponse(makeMessage()), true);
  });

  it("accepts tracked triples for every kind", function () {
    ["api", "local", "fake"].forEach(function (kind) {
      assert.equal(
        isValidMessageResponse(makeTrackedMessage(kind, "m")),
        true,
        kind,
      );
    });
  });

  it("rejects partial nulls", function () {
    assert.equal(
      isValidMessageResponse(makeMessage({
        llm_profile_id_snapshot: "default",
      })),
      false,
    );
    assert.equal(
      isValidMessageResponse(makeMessage({
        llm_profile_id_snapshot: "default",
        llm_profile_kind_snapshot: "api",
      })),
      false,
    );
    assert.equal(
      isValidMessageResponse(makeMessage({
        llm_profile_kind_snapshot: "api",
      })),
      false,
    );
  });

  it("rejects missing fields", function () {
    var m = makeMessage();
    delete m.llm_profile_id_snapshot;
    assert.equal(isValidMessageResponse(m), false);

    var m2 = makeMessage();
    delete m2.llm_profile_kind_snapshot;
    assert.equal(isValidMessageResponse(m2), false);

    var m3 = makeMessage();
    delete m3.llm_model_snapshot;
    assert.equal(isValidMessageResponse(m3), false);
  });

  it("rejects inherited (prototype) snapshot fields", function () {
    var proto = {
      llm_profile_id_snapshot: null,
      llm_profile_kind_snapshot: null,
      llm_model_snapshot: null,
    };
    var m = makeMessage({});
    delete m.llm_profile_id_snapshot;
    delete m.llm_profile_kind_snapshot;
    delete m.llm_model_snapshot;
    Object.setPrototypeOf(m, proto);
    assert.equal(isValidMessageResponse(m), false);
  });

  it("rejects blank, over-long and invalid-kind values", function () {
    assert.equal(
      isValidMessageResponse(makeTrackedMessage("api", "   ")),
      false,
    );
    assert.equal(
      isValidMessageResponse(makeTrackedMessage("quantum", "m")),
      false,
    );
    assert.equal(
      isValidMessageResponse(makeMessage({
        llm_profile_id_snapshot: "a".repeat(51),
        llm_profile_kind_snapshot: "api",
        llm_model_snapshot: "m",
      })),
      false,
    );
    assert.equal(
      isValidMessageResponse(makeMessage({
        llm_profile_id_snapshot: "default",
        llm_profile_kind_snapshot: "api",
        llm_model_snapshot: "m".repeat(256),
      })),
      false,
    );
    assert.equal(
      isValidMessageResponse(makeMessage({
        llm_profile_id_snapshot: "",
        llm_profile_kind_snapshot: "api",
        llm_model_snapshot: "m",
      })),
      false,
    );
    assert.equal(
      isValidMessageResponse(makeMessage({
        llm_profile_id_snapshot: 42,
        llm_profile_kind_snapshot: "api",
        llm_model_snapshot: "m",
      })),
      false,
    );
  });

  it("does not mutate the message", function () {
    var m = makeTrackedMessage("api", "m1");
    var snapshot = JSON.stringify(m);
    isValidMessageResponse(m);
    messageProvenanceLabel(m);
    assert.equal(JSON.stringify(m), snapshot);
  });
});

// -----------------------------------------------------------------------
// messageProvenanceLabel
// -----------------------------------------------------------------------

describe("messageProvenanceLabel", function () {
  it("labels the three kinds from the message's own snapshot", function () {
    assert.equal(messageProvenanceLabel(makeTrackedMessage("api", "m1")), "API · m1");
    assert.equal(messageProvenanceLabel(makeTrackedMessage("local", "qwen")), "Local · qwen");
    assert.equal(messageProvenanceLabel(makeTrackedMessage("fake", "fake")), "Fake · fake");
  });

  it("legacy and invalid messages yield null", function () {
    assert.equal(messageProvenanceLabel(makeMessage()), null);
    assert.equal(
      messageProvenanceLabel(makeMessage({ llm_profile_id_snapshot: "x" })),
      null,
    );
    assert.equal(messageProvenanceLabel(null), null);
    assert.equal(messageProvenanceLabel("msg"), null);
  });

  it("does not depend on the current registry or session", function () {
    var msg = makeTrackedMessage("api", "old-model");
    // Any registry / session state is irrelevant — label comes from the
    // message itself.
    assert.equal(messageProvenanceLabel(msg), "API · old-model");
  });
});

// -----------------------------------------------------------------------
// canApplySwitchResponse
// -----------------------------------------------------------------------

describe("canApplySwitchResponse", function () {
  var validResponse = makeSession({ llm_model_snapshot: "qwen" });

  it("trustworthy response on the current session", function () {
    assert.deepStrictEqual(
      canApplySwitchResponse({
        requestedSessionId: 1,
        currentSessionId: 1,
        response: validResponse,
      }),
      { validForCache: true, stillCurrent: true },
    );
  });

  it("trustworthy response after the user switched away", function () {
    assert.deepStrictEqual(
      canApplySwitchResponse({
        requestedSessionId: 1,
        currentSessionId: 2,
        response: validResponse,
      }),
      { validForCache: true, stillCurrent: false },
    );
  });

  it("stillCurrent is false for invalid current ids", function () {
    [null, undefined, "1", -1, 0, 1.5].forEach(function (bad) {
      var r = canApplySwitchResponse({
        requestedSessionId: 1,
        currentSessionId: bad,
        response: validResponse,
      });
      assert.equal(r.validForCache, true);
      assert.equal(r.stillCurrent, false, "current " + bad);
    });
  });

  it("id mismatch or malformed response is not cacheable", function () {
    assert.deepStrictEqual(
      canApplySwitchResponse({
        requestedSessionId: 1,
        currentSessionId: 1,
        response: makeSession({ id: 2 }),
      }),
      { validForCache: false, stillCurrent: false },
    );
    assert.deepStrictEqual(
      canApplySwitchResponse({
        requestedSessionId: 1,
        currentSessionId: 1,
        response: null,
      }),
      { validForCache: false, stillCurrent: false },
    );
    assert.deepStrictEqual(
      canApplySwitchResponse({
        requestedSessionId: "1",
        currentSessionId: 1,
        response: validResponse,
      }),
      { validForCache: false, stillCurrent: false },
    );
    assert.deepStrictEqual(
      canApplySwitchResponse({
        requestedSessionId: 1.5,
        currentSessionId: 1,
        response: validResponse,
      }),
      { validForCache: false, stillCurrent: false },
    );
  });

  it("null / undefined / array / string options never throw", function () {
    [null, undefined, [], "x", 42].forEach(function (bad) {
      assert.deepStrictEqual(
        canApplySwitchResponse(bad),
        { validForCache: false, stillCurrent: false },
      );
    });
  });
});

// -----------------------------------------------------------------------
// inputs never mutated (new public functions)
// -----------------------------------------------------------------------

describe("new switch helpers do not mutate inputs", function () {
  var registry = analyzeProfileRegistry([
    makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
    makeProfile("local", { kind: "local", model: "qwen" }),
  ]);

  it("resolveSessionProfileDraft / canApplySessionProfile", function () {
    var session = makeSession({ llm_model_snapshot: "m1" });
    var sessionJson = JSON.stringify(session);
    var registryJson = JSON.stringify(registry);

    resolveSessionProfileDraft(session, registry);
    canApplySessionProfile({
      session: session, registry: registry,
      draftProfileId: "local", isSwitching: false,
    });
    needsRemoteHistoryConfirmation({
      session: session, registry: registry,
      targetProfileId: "local", historyState: "present",
    });

    assert.equal(JSON.stringify(session), sessionJson);
    assert.equal(JSON.stringify(registry), registryJson);
  });

  it("parseRemoteHistoryAckRequired leaves the body untouched", function () {
    var body = {
      detail: { code: "remote_history_ack_required", message: "x" },
    };
    var json = JSON.stringify(body);
    parseRemoteHistoryAckRequired(body);
    assert.equal(JSON.stringify(body), json);
  });
});

// -----------------------------------------------------------------------
// 4A review follow-ups — five direct behaviour tests
// -----------------------------------------------------------------------

describe("4A review follow-ups", function () {
  it("fake target never needs the remote-history confirmation", function () {
    var registry = analyzeProfileRegistry([
      makeProfile("default", { is_default: true, kind: "api", model: "m1" }),
      makeProfile("fake", { kind: "fake", model: "fake" }),
    ]);
    var session = makeSession({ llm_model_snapshot: "m1" });
    var sessionJson = JSON.stringify(session);
    var registryJson = JSON.stringify(registry);

    var result = needsRemoteHistoryConfirmation({
      session: session,
      registry: registry,
      targetProfileId: "fake",
      historyState: "present",
    });

    assert.equal(result, false);
    assert.equal(JSON.stringify(session), sessionJson);
    assert.equal(JSON.stringify(registry), registryJson);
  });

  it("provenance label works from the snapshot triple alone", function () {
    // Deliberately no id / session_id / role / content / created_at —
    // the label must depend only on the three snapshot fields.
    var bareSnapshot = {
      llm_profile_id_snapshot: "local",
      llm_profile_kind_snapshot: "local",
      llm_model_snapshot: "qwen3.5:4b",
    };
    assert.equal(
      messageProvenanceLabel(bareSnapshot),
      "Local · qwen3.5:4b",
    );
  });

  it("409 message is returned verbatim (trim only for blank check)", function () {
    var body = {
      detail: {
        code: "remote_history_ack_required",
        message: "  keep original spacing  ",
      },
    };
    var parsed = parseRemoteHistoryAckRequired(body);
    assert.deepStrictEqual(parsed, {
      code: "remote_history_ack_required",
      message: "  keep original spacing  ",
    });
  });

  it("internal snapshot helpers are not exported", function () {
    var selectionModule = require(
      path.resolve(__dirname, "..", "frontend", "model-selection.js"),
    );
    assert.equal(
      Object.prototype.hasOwnProperty.call(selectionModule, "analyzeMessageSnapshot"),
      false,
    );
    assert.equal(
      Object.prototype.hasOwnProperty.call(selectionModule, "profileFromRegistry"),
      false,
    );
  });

  it("canApplySwitchResponse does not mutate options or response", function () {
    var response = makeSession({ llm_model_snapshot: "qwen" });
    var options = {
      requestedSessionId: 1,
      currentSessionId: 2,
      response: response,
    };
    var optionsJson = JSON.stringify(options);
    var responseJson = JSON.stringify(response);

    var result = canApplySwitchResponse(options);

    assert.deepStrictEqual(result, { validForCache: true, stillCurrent: false });
    assert.equal(JSON.stringify(options), optionsJson);
    assert.equal(JSON.stringify(response), responseJson);
  });
});
