"use strict";

/**
 * Attempt to copy *text* using the provided clipboard-like object.
 *
 * Returns Promise<boolean>.  Never throws, never rejects.
 *
 * @param {string} text
 * @param {{ writeText: (t: string) => Promise<void> }|null} clipboardAPI
 * @returns {Promise<boolean>}
 */
function copyWithAPI(text, clipboardAPI) {
    try {
        if (!clipboardAPI || typeof clipboardAPI.writeText !== "function") {
            return Promise.resolve(false);
        }
        var result = clipboardAPI.writeText(text);
        if (!result || typeof result.then !== "function") {
            return Promise.resolve(false);
        }
        return Promise.resolve(result).then(
            function () { return true; },
            function () { return false; }
        );
    } catch (_) {
        return Promise.resolve(false);
    }
}

/**
 * Fallback copy using hidden textarea + execCommand("copy").
 *
 * Restores focus and selection on both success and failure paths.
 * The temporary textarea is always removed.  Never throws.
 *
 * @param {string} text
 * @param {Document|Object} doc
 * @returns {boolean}
 */
function copyWithFallback(text, doc) {
    var active = null;
    var sel = null;
    var clonedRange = null;
    var ta = null;
    var ok = false;

    // -- save state --
    try {
        active = doc.activeElement;
        sel = doc.getSelection();
        if (sel && sel.rangeCount > 0) {
            clonedRange = sel.getRangeAt(0).cloneRange();
        }
    } catch (_) { /* best-effort */ }

    // -- copy --
    try {
        ta = doc.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        ta.style.top = "-9999px";
        ta.setAttribute("readonly", "");
        ta.setAttribute("aria-hidden", "true");
        doc.body.appendChild(ta);
        ta.focus();
        ta.select();
        try {
            ok = doc.execCommand("copy") === true;
        } catch (_) {
            ok = false;
        }
    } catch (_) {
        ok = false;
    } finally {
        // -- always remove textarea --
        if (ta && ta.parentNode) {
            try { ta.parentNode.removeChild(ta); } catch (_) {}
        }
        // -- best-effort restore focus --
        if (active && typeof active.focus === "function") {
            try { active.focus(); } catch (_) {}
        }
        // -- best-effort restore selection --
        if (sel && clonedRange) {
            try {
                sel.removeAllRanges();
                sel.addRange(clonedRange);
            } catch (_) {}
        }
    }

    return ok;
}

/**
 * Copy *text* to the system clipboard.  Tries Clipboard API first,
 * falls back to execCommand.  Never throws, never rejects.
 *
 * @param {string} text
 * @param {{ writeText: (t: string) => Promise<void> }|null} clipboardAPI
 * @param {Document|Object} doc
 * @returns {Promise<boolean>}
 */
function copyToClipboard(text, clipboardAPI, doc) {
    return copyWithAPI(text, clipboardAPI).then(function (ok) {
        if (ok) {
            return true;
        }
        try {
            return copyWithFallback(text, doc);
        } catch (_) {
            return false;
        }
    }, function (_err) {
        // copyWithAPI unexpectedly rejected — fallback
        try {
            return copyWithFallback(text, doc);
        } catch (_) {
            return false;
        }
    });
}

// Dual export: browser global, module.exports for node:test
if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        copyWithAPI: copyWithAPI,
        copyWithFallback: copyWithFallback,
        copyToClipboard: copyToClipboard,
    };
}
