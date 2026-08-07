"use strict";

/**
 * Factory that creates a per-button copy state controller.
 * All side effects (DOM updates, timers) are injected so Node
 * can test every state transition without a browser.
 *
 * @param {function(string): Promise<boolean>} doCopy
 *   Copy executor — receives text, returns Promise<boolean>.
 * @param {function(function, number): *} setTimer
 *   setTimeout-like — (fn, delayMs) → timerId
 * @param {function(*): void} clearTimerFn
 *   clearTimeout-like — (timerId) → void
 * @param {function(string): void} onState
 *   Called on state transition with new state name.
 * @param {function(): boolean} isActive
 *   Returns true when the target button is still connected.
 * @returns {{ handleClick: function(string): void, getState: function(): string }}
 */
function createCopyController(doCopy, setTimer, clearTimerFn, onState, isActive) {
    var state = "idle";       // idle | copying | copied | failed
    var timerId = null;

    function transition(newState) {
        state = newState;
        if (isActive()) {
            onState(newState);
        }
    }

    function clearExistingTimer() {
        if (timerId !== null) {
            clearTimerFn(timerId);
            timerId = null;
        }
    }

    function handleClick(text) {
        if (state === "copying") return;

        clearExistingTimer();
        transition("copying");

        Promise.resolve()
            .then(function () { return doCopy(text); })
            .then(
                function (ok) {
                    if (state !== "copying") return;
                    clearExistingTimer();
                    if (ok) {
                        transition("copied");
                        timerId = setTimer(function () {
                            timerId = null;
                            if (state === "copied") transition("idle");
                        }, 1800);
                    } else {
                        transition("failed");
                        timerId = setTimer(function () {
                            timerId = null;
                            if (state === "failed") transition("idle");
                        }, 1800);
                    }
                },
                function (_err) {
                    // doCopy reject or synchronous throw
                    if (state !== "copying") return;
                    clearExistingTimer();
                    transition("failed");
                    timerId = setTimer(function () {
                        timerId = null;
                        if (state === "failed") transition("idle");
                    }, 1800);
                }
            );
    }

    return {
        handleClick: handleClick,
        getState: function () { return state; },
    };
}

// Dual export
if (typeof module !== "undefined" && module.exports) {
    module.exports = { createCopyController: createCopyController };
}
