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
  };
  if (extra) {
    Object.keys(extra).forEach(function (k) { m[k] = extra[k]; });
  }
  return m;
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
