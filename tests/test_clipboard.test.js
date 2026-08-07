/**
 * Tests for clipboard.js and copy-controller.js
 *
 * Run:  node --test tests/test_clipboard.test.js
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const { copyWithAPI, copyWithFallback, copyToClipboard } = require(
    path.resolve(__dirname, "..", "frontend", "clipboard.js"),
);
const { createCopyController } = require(
    path.resolve(__dirname, "..", "frontend", "copy-controller.js"),
);

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

function makeFakeDoc(activeElement, execCommandResult) {
    var removed = [];
    var bodyChildren = [];
    var focusCalls = 0;
    var selectionCalls = { removeAllRanges: 0, addRange: 0 };

    function makeTextarea() {
        var el = {
            value: "",
            style: {},
            parentNode: null,
            setAttribute: function () {},
            focus: function () {},
            select: function () {},
        };
        return el;
    }

    return {
        activeElement: activeElement || null,
        focusCalls: function () { return focusCalls; },
        selectionCalls: function () { return selectionCalls; },
        body: {
            appendChild: function (el) {
                el.parentNode = { removeChild: function () {
                    var idx = bodyChildren.indexOf(el);
                    if (idx >= 0) bodyChildren.splice(idx, 1);
                    removed.push(el);
                }};
                bodyChildren.push(el);
            },
            removeChild: function (el) {
                var idx = bodyChildren.indexOf(el);
                if (idx >= 0) bodyChildren.splice(idx, 1);
                removed.push(el);
            },
            get children() { return bodyChildren; },
        },
        createElement: function (tag) {
            if (tag === "textarea") return makeTextarea();
            throw new Error("unsupported tag: " + tag);
        },
        getSelection: function () {
            return {
                rangeCount: 1,
                getRangeAt: function () {
                    return { cloneRange: function () { return {}; } };
                },
                removeAllRanges: function () { selectionCalls.removeAllRanges++; },
                addRange: function () { selectionCalls.addRange++; },
            };
        },
        execCommand: function (cmd) {
            if (typeof execCommandResult === "function") {
                return execCommandResult(cmd);
            }
            return execCommandResult;
        },
    };
}

/** Fake doc where createElement throws */
function makeFailingCreateDoc() {
    return {
        activeElement: null,
        body: {
            appendChild: function () {},
            removeChild: function () {},
            children: [],
        },
        createElement: function () { throw new Error("createElement failed"); },
        getSelection: function () { return { rangeCount: 0 }; },
        execCommand: function () { return true; },
    };
}

/** Fake doc where focus throws */
function makeFailingFocusDoc() {
    return {
        activeElement: { focus: function () { throw new Error("focus failed"); } },
        body: {
            appendChild: function (el) { el.parentNode = { removeChild: function () {} }; },
            removeChild: function () {},
            children: [],
        },
        createElement: function () {
            return {
                value: "", style: {},
                setAttribute: function () {},
                focus: function () { throw new Error("focus failed"); },
                select: function () {},
                parentNode: null,
            };
        },
        getSelection: function () { return { rangeCount: 0 }; },
        execCommand: function () { return true; },
    };
}

/** Fake doc where select throws */
function makeFailingSelectDoc() {
    return {
        activeElement: null,
        body: {
            appendChild: function (el) { el.parentNode = { removeChild: function () {} }; },
            removeChild: function () {},
            children: [],
        },
        createElement: function () {
            return {
                value: "", style: {},
                setAttribute: function () {},
                focus: function () {},
                select: function () { throw new Error("select failed"); },
                parentNode: null,
            };
        },
        getSelection: function () { return { rangeCount: 0 }; },
        execCommand: function () { return true; },
    };
}

function makeFakeTimer() {
    var currentId = 0;
    var pending = new Map();
    var cleared = [];

    function setTimer(fn, ms) {
        var id = ++currentId;
        pending.set(id, { fn: fn, ms: ms });
        return id;
    }
    function clearTimer(id) {
        cleared.push(id);
        pending.delete(id);
    }
    function fireAll() {
        var callbacks = Array.from(pending.values());
        pending.clear();
        callbacks.forEach(function (t) { t.fn(); });
    }
    return {
        setTimer: setTimer,
        clearTimer: clearTimer,
        fireAll: fireAll,
        pendingCount: function () { return pending.size; },
        pendingIds: function () { return Array.from(pending.keys()); },
        wasCleared: function (id) { return cleared.includes(id); },
        clearedIds: cleared,
    };
}

// -----------------------------------------------------------------------
// copyWithAPI
// -----------------------------------------------------------------------

describe("copyWithAPI", function () {
    it("resolves true on success", async function () {
        var api = { writeText: function () { return Promise.resolve(); } };
        var ok = await copyWithAPI("hello", api);
        assert.equal(ok, true);
    });

    it("resolves false on reject", async function () {
        var api = { writeText: function () { return Promise.reject(new Error("nope")); } };
        var ok = await copyWithAPI("hello", api);
        assert.equal(ok, false);
    });

    it("resolves false when clipboard is null", async function () {
        var ok = await copyWithAPI("hello", null);
        assert.equal(ok, false);
    });

    it("resolves false when writeText is not a function", async function () {
        var ok = await copyWithAPI("hello", {});
        assert.equal(ok, false);
    });

    it("resolves false when writeText throws synchronously", async function () {
        var api = { writeText: function () { throw new Error("sync fail"); } };
        var ok = await copyWithAPI("hello", api);
        assert.equal(ok, false);
    });

    it("resolves false when writeText returns non-Promise", async function () {
        var api = { writeText: function () { return "not a promise"; } };
        var ok = await copyWithAPI("hello", api);
        assert.equal(ok, false);
    });

    it("passes exact text to writeText", async function () {
        var received = null;
        var api = { writeText: function (t) { received = t; return Promise.resolve(); } };
        await copyWithAPI("hello\nworld\t!", api);
        assert.equal(received, "hello\nworld\t!");
    });
});

// -----------------------------------------------------------------------
// copyWithFallback
// -----------------------------------------------------------------------

describe("copyWithFallback", function () {
    it("returns true when execCommand succeeds", function () {
        var doc = makeFakeDoc(null, true);
        var ok = copyWithFallback("hello", doc);
        assert.equal(ok, true);
    });

    it("returns false when execCommand returns false", function () {
        var doc = makeFakeDoc(null, false);
        var ok = copyWithFallback("hello", doc);
        assert.equal(ok, false);
    });

    it("returns false when execCommand throws", function () {
        var doc = makeFakeDoc(null, function () { throw new Error("fail"); });
        var ok = copyWithFallback("hello", doc);
        assert.equal(ok, false);
    });

    it("sets textarea value to exact text with newlines and Unicode", function () {
        var capturedValue = null;
        var doc = makeFakeDoc(null, true);
        var origCreate = doc.createElement;
        doc.createElement = function (tag) {
            var el = origCreate(tag);
            var origSetAttr = el.setAttribute;
            el.setAttribute = function (name, val) {
                if (name === "value") capturedValue = val;
                origSetAttr(name, val);
            };
            return el;
        };
        // Override to track value after setting
        var realCreate = doc.createElement;
        doc.createElement = function (tag) {
            var el = realCreate(tag);
            el._value = null;
            var desc = Object.getOwnPropertyDescriptor(el, "value");
            if (!desc || !desc.set) {
                var stored = "";
                Object.defineProperty(el, "value", {
                    get: function () { return stored; },
                    set: function (v) { stored = v; capturedValue = v; },
                    configurable: true,
                });
            }
            return el;
        };

        copyWithFallback("hello\nworld\t\"quoted\" ☃", doc);
        assert.equal(capturedValue, "hello\nworld\t\"quoted\" ☃");
    });

    it("textarea is removed on success", function () {
        var doc = makeFakeDoc(null, true);
        copyWithFallback("hello", doc);
        assert.equal(doc.body.children.length, 0);
    });

    it("textarea is removed on execCommand failure", function () {
        var doc = makeFakeDoc(null, false);
        copyWithFallback("hello", doc);
        assert.equal(doc.body.children.length, 0);
    });

    it("restores focus to activeElement", function () {
        var focused = false;
        var el = { focus: function () { focused = true; } };
        var doc = makeFakeDoc(el, true);
        copyWithFallback("hello", doc);
        assert.equal(focused, true);
    });

    it("calls selection restore methods", function () {
        var doc = makeFakeDoc(null, true);
        copyWithFallback("hello", doc);
        var calls = doc.selectionCalls();
        assert.ok(calls.removeAllRanges >= 0);
    });

    it("returns false when createElement throws", function () {
        var doc = makeFailingCreateDoc();
        var ok = copyWithFallback("hello", doc);
        assert.equal(ok, false);
    });

    it("returns false when focus throws", function () {
        var doc = makeFailingFocusDoc();
        var ok = copyWithFallback("hello", doc);
        assert.equal(ok, false);
    });

    it("returns false when select throws", function () {
        var doc = makeFailingSelectDoc();
        var ok = copyWithFallback("hello", doc);
        assert.equal(ok, false);
    });

    it("textarea removed even when focus throws", function () {
        var doc = makeFailingFocusDoc();
        copyWithFallback("hello", doc);
        assert.equal(doc.body.children.length, 0);
    });

    it("textarea removed even when select throws", function () {
        var doc = makeFailingSelectDoc();
        copyWithFallback("hello", doc);
        assert.equal(doc.body.children.length, 0);
    });
});

// -----------------------------------------------------------------------
// copyToClipboard orchestration
// -----------------------------------------------------------------------

describe("copyToClipboard", function () {
    it("API success skips fallback", async function () {
        var fallbackCalled = false;
        var ok = await copyToClipboard(
            "hello",
            { writeText: function () { return Promise.resolve(); } },
            { execCommand: function () { fallbackCalled = true; return true; } }
        );
        assert.equal(ok, true);
        assert.equal(fallbackCalled, false);
    });

    it("API null falls back", async function () {
        var ok = await copyToClipboard(
            "hello", null, makeFakeDoc(null, true)
        );
        assert.equal(ok, true);
    });

    it("API reject falls back", async function () {
        var api = { writeText: function () { return Promise.reject(new Error("nope")); } };
        var ok = await copyToClipboard("hello", api, makeFakeDoc(null, true));
        assert.equal(ok, true);
    });

    it("API null + fallback false returns false", async function () {
        var ok = await copyToClipboard("hello", null, makeFakeDoc(null, false));
        assert.equal(ok, false);
    });

    it("passes exact text to API writeText", async function () {
        var received = null;
        var api = { writeText: function (t) { received = t; return Promise.resolve(); } };
        await copyToClipboard("x\ny\tz", api, makeFakeDoc(null, true));
        assert.equal(received, "x\ny\tz");
    });

    it("API fail + fallback throws resolves false", async function () {
        var api = { writeText: function () { return Promise.reject(new Error("nope")); } };
        var badDoc = makeFailingCreateDoc();
        var ok = await copyToClipboard("hello", api, badDoc);
        assert.equal(ok, false);
    });

    it("API null + fallback throws resolves false", async function () {
        var ok = await copyToClipboard("hello", null, makeFailingCreateDoc());
        assert.equal(ok, false);
    });
});

// -----------------------------------------------------------------------
// createCopyController state machine
// -----------------------------------------------------------------------

describe("createCopyController", function () {
    function makeEnv(doCopyImpl) {
        var states = [];
        var timer = makeFakeTimer();
        var active = true;
        var ctrl = createCopyController(
            doCopyImpl,
            timer.setTimer,
            timer.clearTimer,
            function (s) { states.push(s); },
            function () { return active; }
        );
        return {
            ctrl: ctrl,
            states: states,
            timer: timer,
            disconnect: function () { active = false; },
        };
    }

    it("idle → copying → copied → timer → idle", async function () {
        var env = makeEnv(function () { return Promise.resolve(true); });
        env.ctrl.handleClick("hello");
        assert.equal(env.ctrl.getState(), "copying");
        assert.deepStrictEqual(env.states, ["copying"]);

        // let microtasks flush
        await new Promise(function (r) { setTimeout(r, 5); });
        assert.equal(env.ctrl.getState(), "copied");
        assert.deepStrictEqual(env.states, ["copying", "copied"]);

        env.timer.fireAll();
        assert.equal(env.ctrl.getState(), "idle");
        assert.deepStrictEqual(env.states, ["copying", "copied", "idle"]);
    });

    it("idle → copying → failed → timer → idle", async function () {
        var env = makeEnv(function () { return Promise.resolve(false); });
        env.ctrl.handleClick("hello");
        await new Promise(function (r) { setTimeout(r, 5); });
        assert.equal(env.ctrl.getState(), "failed");

        env.timer.fireAll();
        assert.equal(env.ctrl.getState(), "idle");
    });

    it("ignores clicks during copying", async function () {
        var callCount = 0;
        // Use a never-resolving promise so state stays "copying"
        var never = new Promise(function () {});
        var env = makeEnv(function () { callCount++; return never; });
        env.ctrl.handleClick("a");
        // Flush microtasks so doCopy is actually called
        await new Promise(function (r) { setTimeout(r, 5); });
        assert.equal(callCount, 1);
        assert.equal(env.ctrl.getState(), "copying");
        // Additional clicks must be ignored
        env.ctrl.handleClick("b");
        env.ctrl.handleClick("c");
        assert.equal(callCount, 1);
    });

    it("failed → click → copying → copied (retry)", async function () {
        var doCopyCalls = [];

        function doCopy(text) {
            doCopyCalls.push(text);
            // First call fails, second call (retry) succeeds
            return Promise.resolve(doCopyCalls.length >= 2);
        }

        var timer = makeFakeTimer();
        var states = [];
        var ctrl = createCopyController(
            doCopy,
            timer.setTimer,
            timer.clearTimer,
            function (s) { states.push(s); },
            function () { return true; },
        );

        // --- Phase 1: trigger first failure ---
        ctrl.handleClick("first-text");

        // Sync: state is copying, doCopy not yet called
        assert.equal(ctrl.getState(), "copying");
        assert.deepStrictEqual(states, ["copying"]);
        assert.equal(doCopyCalls.length, 0);

        await new Promise(function (r) { setTimeout(r, 5); });

        // Async: first doCopy executed, returned false → failed
        assert.equal(ctrl.getState(), "failed");
        assert.deepStrictEqual(states, ["copying", "failed"]);
        assert.equal(doCopyCalls.length, 1);
        assert.equal(doCopyCalls[0], "first-text");
        assert.equal(timer.pendingCount(), 1);          // failed timer set

        // --- Phase 2: retry ---
        ctrl.handleClick("retry-text");

        // === Sync assertions ===
        // Old failed timer cleared, state → copying,
        // but Promise.resolve().then(() => doCopy("retry-text")) NOT yet executed
        assert.equal(ctrl.getState(), "copying");
        assert.ok(timer.clearedIds.length >= 1);         // old failed timer cleared
        assert.equal(timer.pendingCount(), 0);           // old cleared, new not yet set
        assert.equal(doCopyCalls.length, 1);             // still 1 — second doCopy queued, not called
        assert.deepStrictEqual(
            states,
            ["copying", "failed", "copying"],
        );

        await new Promise(function (r) { setTimeout(r, 5); });

        // === Async assertions ===
        // Second doCopy executed, returned true → copied
        assert.equal(doCopyCalls.length, 2);             // now 2
        assert.equal(doCopyCalls[1], "retry-text");
        assert.equal(ctrl.getState(), "copied");
        assert.deepStrictEqual(
            states,
            ["copying", "failed", "copying", "copied"],
        );
        assert.equal(timer.pendingCount(), 1);           // new copied timer set

        // --- Phase 3: timer fires → idle ---
        timer.fireAll();

        assert.equal(ctrl.getState(), "idle");
        assert.deepStrictEqual(
            states,
            ["copying", "failed", "copying", "copied", "idle"],
        );
        assert.equal(timer.pendingCount(), 0);
    });

    it("two controllers are independent", function () {
        var s1 = [], s2 = [];
        var c1 = createCopyController(
            function () { return new Promise(function () {}); },
            function () { return 1; }, function () {},
            function (s) { s1.push(s); }, function () { return true; }
        );
        var c2 = createCopyController(
            function () { return new Promise(function () {}); },
            function () { return 2; }, function () {},
            function (s) { s2.push(s); }, function () { return true; }
        );
        c1.handleClick("a");
        c2.handleClick("b");
        assert.deepStrictEqual(s1, ["copying"]);
        assert.deepStrictEqual(s2, ["copying"]);
    });

    it("old timer cancelled on retry during copied", async function () {
        var timer = makeFakeTimer();
        var states = [];
        var ctrl = createCopyController(
            function () { return Promise.resolve(true); },
            timer.setTimer,
            timer.clearTimer,
            function (s) { states.push(s); },
            function () { return true; },
        );

        // --- Phase 1: first copy succeeds ---
        ctrl.handleClick("hello");
        await new Promise(function (r) { setTimeout(r, 5); });
        assert.equal(ctrl.getState(), "copied");

        // === Time point 1: before retry ===
        assert.equal(timer.pendingCount(), 1);            // copied timer exists
        var oldTimerId = timer.pendingIds()[0];           // capture old timer ID

        // --- Phase 2: retry ---
        ctrl.handleClick("hello");

        // === Time point 2: after retry click, before microtask ===
        // clearExistingTimer() has run synchronously — old timer cleared
        // Promise.resolve().then(() => doCopy(...)) queued but NOT executed
        assert.ok(timer.clearedIds.length >= 1);          // old timer was cleared
        assert.ok(timer.wasCleared(oldTimerId));          // confirmed it was the OLD timer
        assert.equal(timer.pendingCount(), 0);            // old cleared, new not yet set
        assert.equal(ctrl.getState(), "copying");         // retry is in progress

        await new Promise(function (r) { setTimeout(r, 5); });

        // === Time point 3: after second copy completes ===
        assert.equal(ctrl.getState(), "copied");          // second copy succeeded
        assert.equal(timer.pendingCount(), 1);            // new copied timer set
        assert.ok(!timer.wasCleared(timer.pendingIds()[0])); // new timer NOT cleared
        assert.deepStrictEqual(
            states,
            ["copying", "copied", "copying", "copied"],
        );

        // --- Phase 3: new timer fires → idle ---
        timer.fireAll();

        assert.equal(ctrl.getState(), "idle");
        assert.deepStrictEqual(
            states,
            ["copying", "copied", "copying", "copied", "idle"],
        );
        assert.equal(timer.pendingCount(), 0);
    });

    it("doCopy synchronous throw → failed", async function () {
        var env = makeEnv(function () { throw new Error("sync fail"); });
        env.ctrl.handleClick("hello");
        await new Promise(function (r) { setTimeout(r, 5); });
        assert.equal(env.ctrl.getState(), "failed");
    });

    it("doCopy Promise reject → failed", async function () {
        var env = makeEnv(function () { return Promise.reject(new Error("async fail")); });
        env.ctrl.handleClick("hello");
        await new Promise(function (r) { setTimeout(r, 5); });
        assert.equal(env.ctrl.getState(), "failed");
    });

    it("timer callback sets timerId to null then transitions", async function () {
        var timer = makeFakeTimer();
        var states = [];
        var ctrl = createCopyController(
            function () { return Promise.resolve(false); },
            timer.setTimer, timer.clearTimer,
            function (s) { states.push(s); }, function () { return true; }
        );
        ctrl.handleClick("x");
        await new Promise(function (r) { setTimeout(r, 5); });
        assert.equal(ctrl.getState(), "failed");

        timer.fireAll();
        assert.equal(ctrl.getState(), "idle");
        assert.deepStrictEqual(states, ["copying", "failed", "idle"]);
    });

    it("disconnected button does not receive onState", async function () {
        var states = [];
        var active = true;
        var ctrl = createCopyController(
            function () { return Promise.resolve(true); },
            function (fn) { fn(); },
            function () {},
            function (s) { states.push(s); },
            function () { return active; }
        );
        active = false;  // simulate disconnected
        ctrl.handleClick("hello");
        // transition("copying") should not call onState
        assert.deepStrictEqual(states, []);
    });
});
