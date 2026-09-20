/* ------------------------------------------------------------------ */
/* Static Chatbot — multi-session frontend logic                      */
/* ------------------------------------------------------------------ */

(function () {
  "use strict";

  /**
   * History Review dialog copy, shared with history-review.js so the
   * heading is the confirmation question and the body explains what the
   * operation does.  The literals are a fallback for the case where
   * history-review.js failed to load; neither may ever contain source
   * messages, reviewer prompts or other private material.
   */
  const historyReviewDialogCopy = (function () {
    const fallback = {
      startTitle: "Start a history review?",
      startBody:
        "A separate review is generated from the teaching history " +
        "frozen at this mode switch. Messages from this conversation " +
        "only are used, and the review never runs on its own — you " +
        "have to start it. The report appears here when it finishes.",
      continueTitle: "Continue this history review?",
      continueBody:
        "The pending review resumes with the teaching history that was " +
        "already frozen when it started. The completed report appears " +
        "here.",
      retryTitle: "Retry this history review?",
      retryBody:
        "The previous attempt was interrupted. It may already have " +
        "reached the model. Retrying sends the same frozen sources " +
        "again, and a remote model may produce a duplicate call or " +
        "cost. Continue only if you want to retry.",
    };
    try {
      if (typeof HISTORY_REVIEW_DIALOG_COPY === "object" &&
          HISTORY_REVIEW_DIALOG_COPY !== null) {
        return HISTORY_REVIEW_DIALOG_COPY;
      }
    } catch (_) { /* history-review.js did not load */ }
    return fallback;
  })();

  /* ---- DOM references ---------------------------------------------- */

  const sidebarEl = document.getElementById("sidebar");
  const sidebarToggleEl = document.getElementById("sidebarToggle");
  const sessionListEl = document.getElementById("sessionList");
  const sessionListStatusEl = document.getElementById("sessionListStatus");
  const newChatBtn = document.getElementById("newChatButton");
  const profileSelectEl = document.getElementById("profileSelect");
  const profileSelectorStatusEl =
    document.getElementById("profileSelectorStatus");
  const sessionHeaderEl = document.getElementById("sessionHeader");
  const sessionTitleEl = document.getElementById("sessionTitle");
  const sessionModelBadgeEl = document.getElementById("sessionModelBadge");
  const interactionModeBadgeEl =
    document.getElementById("interactionModeBadge");
  const sessionCompatibilityNoticeEl =
    document.getElementById("sessionCompatibilityNotice");
  const currentProfileBarEl = document.getElementById("currentProfileBar");
  const currentProfileSelectEl =
    document.getElementById("currentProfileSelect");
  const applyProfileBtn = document.getElementById("applyProfileBtn");
  const currentProfileStatusEl =
    document.getElementById("currentProfileStatus");
  const interactionModeBarEl =
    document.getElementById("interactionModeBar");
  const interactionModeSelectEl =
    document.getElementById("interactionModeSelect");
  const applyInteractionModeBtn =
    document.getElementById("applyInteractionModeBtn");
  const interactionModeStatusEl =
    document.getElementById("interactionModeStatus");
  const profileSwitchDialogEl =
    document.getElementById("profileSwitchDialog");

  const historyReviewPanelEl =
    document.getElementById("historyReviewPanel");
  const historyReviewTitleEl =
    document.getElementById("historyReviewTitle");
  const historyReviewStatusBadgeEl =
    document.getElementById("historyReviewStatusBadge");
  const historyReviewToggleBtn =
    document.getElementById("historyReviewToggleBtn");
  const historyReviewBodyEl =
    document.getElementById("historyReviewBody");
  const historyReviewLiveEl =
    document.getElementById("historyReviewLive");
  const historyReviewProposalEl =
    document.getElementById("historyReviewProposal");
  const historyReviewProposalTextEl =
    document.getElementById("historyReviewProposalText");
  const historyReviewStartBtn =
    document.getElementById("historyReviewStartBtn");
  const historyReviewDismissBtn =
    document.getElementById("historyReviewDismissBtn");
  const historyReviewDetailEl =
    document.getElementById("historyReviewDetail");
  const historyReviewSelectEl =
    document.getElementById("historyReviewSelect");
  const historyReviewSummaryEl =
    document.getElementById("historyReviewSummary");
  const historyReviewCoverageEl =
    document.getElementById("historyReviewCoverage");
  const historyReviewFindingsEl =
    document.getElementById("historyReviewFindings");
  const historyReviewErrorEl =
    document.getElementById("historyReviewError");
  const historyReviewRecheckBtn =
    document.getElementById("historyReviewRecheckBtn");
  const historyReviewContinueBtn =
    document.getElementById("historyReviewContinueBtn");

  const historyReviewStartDialogEl =
    document.getElementById("historyReviewStartDialog");
  const historyReviewStartTitleEl =
    document.getElementById("hrs-title");
  const historyReviewStartBodyEl =
    document.getElementById("hrs-body");
  const historyReviewStartCancelBtn =
    document.getElementById("hrs-cancel");
  const historyReviewStartConfirmBtn =
    document.getElementById("hrs-start");

  const historyReviewConsentDialogEl =
    document.getElementById("historyReviewConsentDialog");
  const historyReviewConsentBodyEl =
    document.getElementById("hrc-body");
  const historyReviewConsentCancelBtn =
    document.getElementById("hrc-cancel");
  const historyReviewConsentConfirmBtn =
    document.getElementById("hrc-continue");

  const historyReviewSelectorEl =
    document.getElementById("historyReviewSelector");

  const messagesEl = document.getElementById("messages");
  const statusEl = document.getElementById("status");
  const inputEl = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendButton");

  /* ---- State ------------------------------------------------------- */

  let sessions = [];            // [{id, title, ..., llm_profile_*}, ...]
  let currentSessionId = null;  // int | null
  let isSending = false;        // prevent double-submit
  let isCreatingSession = false; // prevent double-create of New Chat
  let isRenaming = false;       // editing mode active
  let renamingSessionId = null; // which session is being edited
  let isRenameSaving = false;   // saving in progress (prevent double-submit)
  let isInitializing = true;    // block user actions during page init
  let sessionLoadRequestId = 0;  // race-condition guard for message loads
  let sessionLastMessageId = {}; // sessionId -> int (last known message id)

  // Model selector state
  let profiles = [];               // raw profile list from the server
  let selectedProfileId = null;    // string | null (new chats only)
  let profilesLoadState = "loading"; // "loading" | "ready" | "error"
  let profilesLoadError = null;    // persistent — #profileSelectorStatus

  // Session list state
  let sessionsLoadState = "loading"; // "loading" | "ready" | "error"
  let sessionsEverLoaded = false;    // ever loaded the full list
  let sessionsLoadError = null;      // persistent — #sessionListStatus

  // Temporary per-session send blocks set after 409/503 responses.
  // sessionId -> "conflict" | "profile_unavailable"
  let sessionSendBlocks = {};

  // Current-session model switcher state (fully independent from the
  // new-chat selector above).
  let currentProfileDraftId = null;   // string | null (current chat only)
  let isProfileSwitching = false;     // busy — covers outcome application
  let isDeletingSession = false;      // delete busy (programmatic guard)
  let deletingSessionId = null;
  let profileSwitchGeneration = 0;    // monotonic token / generation

  // Current-session interaction-mode state (fully independent from the
  // model profile controls above).
  let currentInteractionModeDraft = RECEIVE_TEACHING_MODE;
  let isInteractionModeSwitching = false;
  let interactionModeSwitchGeneration = 0;
  let interactionModeController = null;
  let interactionModeInitializationError = null;
  const interactionModeUncertainBySession = Object.create(null);

  // Per-session records.  Keys are positive safe integer ids.
  const sessionHasMessages = Object.create(null);      // true | false | undefined
  const sessionSwitchUncertain = Object.create(null);  // uncertain records
  const profileSwitchStatusBySession = Object.create(null); // {text, isError}
  const interactionModeStatusBySession = Object.create(null); // {text, isError}

  // History Review state — per-session, never a single global result.
  const reviewStateBySession = Object.create(null);
  const reviewLifecycleEpochBySession = Object.create(null);
  const reviewDialogContextBySession = Object.create(null);
  const reviewBatchGenerationBySession = Object.create(null);

  let historyReviewController = null;
  let historyReviewInitializationError = null;
  let historyReviewStartConfirmer = null;
  let historyReviewConsentConfirmer = null;
  let historyReviewStartDialogAdapter = null;
  let historyReviewConsentDialogAdapter = null;
  function getReviewStateIfExists(sessionId) {
    if (isValidSessionIdKey(sessionId) === false) return null;
    const key = String(sessionId);
    if (Object.hasOwn(reviewStateBySession, key) === false) return null;
    return reviewStateBySession[key];
  }

  function ensureReviewState(sessionId) {
    if (isValidSessionIdKey(sessionId) === false) return null;
    if (findSessionInList(sessionId) === null) return null;
    const key = String(sessionId);
    if (Object.hasOwn(reviewStateBySession, key) === false) {
      reviewStateBySession[key] = {
        events: null, eventsStatus: "idle", eventsError: null,
        summaries: null, summariesStatus: "idle", summariesError: null,
        detailsById: Object.create(null),
        detailStatusByReviewId: Object.create(null),
        detailErrorByReviewId: Object.create(null),
        detailGenerationByReviewId: Object.create(null),
        selectedReviewId: null, proposalEvent: null,
        proposalDismissedEventId: null, operation: null,
        operationGeneration: 0, uncertainEventId: null,
        uncertainMessage: "", sessionLoadGeneration: 0,
        collapsed: false, panelError: null,
      };
    }
    return reviewStateBySession[key];
  }

  function reviewEpoch(sessionId) {
    return reviewLifecycleEpochBySession[String(sessionId)] || 0;
  }

  function bumpReviewEpoch(sessionId) {
    const key = String(sessionId);
    reviewLifecycleEpochBySession[key] = reviewEpoch(sessionId) + 1;
  }

  function safeReviewError(err) {
    if (err && typeof err.message === "string" && err.message) return err.message;
    return "History review request failed.";
  }

  function isExactSessionNotFound(err) {
    return err !== null && typeof err === "object" &&
      err.failureKind === "http" && err.status === 404 &&
      err.code === "history_review_session_not_found";
  }

  function canApplyCaptured(captured, kind) {
    if (findSessionInList(captured.targetSessionId) === null) return false;
    const current = getReviewStateIfExists(captured.targetSessionId);
    if (current === null || current !== captured.stateIdentity) return false;
    if (reviewEpoch(captured.targetSessionId) !== captured.epoch) return false;
    if (kind === "batch") return current.sessionLoadGeneration === captured.generation;
    if (kind === "detail") {
      return current.detailGenerationByReviewId[captured.reviewId] === captured.generation;
    }
    if (kind === "operation") {
      return current.operation !== null && current.operation.generation === captured.generation;
    }
    return false;
  }

  function clearReviewSessionState(sessionId) {
    if (isValidSessionIdKey(sessionId) === false) return;
    const key = String(sessionId);
    const context = reviewDialogContextBySession[key];
    if (context && context.pendingAdapter &&
        typeof context.pendingAdapter.cancelPending === "function") {
      try { context.pendingAdapter.cancelPending(); } catch (_) {}
    }
    delete reviewDialogContextBySession[key];
    bumpReviewEpoch(sessionId);
    delete reviewStateBySession[key];
    if (currentSessionId === sessionId) {
      renderHistoryReviewPanel();
      updateControlStates();
    }
  }

  function reviewTargetBusy(sessionId) {
    const state = getReviewStateIfExists(sessionId);
    if (state === null) return false;
    return isHistoryReviewTargetBusy(sessionId, state, isValidApiTimestamp);
  }

  async function historyReviewRequestJson(url, options = {}) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (_) {
      throw { failureKind: "network", status: 0, code: null,
        message: "Network error. Please check your connection.", body: null };
    }
    let body = null;
    let parseFailed = false;
    try { body = await response.json(); } catch (_) { parseFailed = true; }

    if (response.ok === false) {
      let code = null;
      let message = "Something went wrong. Please try again.";
      if (body && typeof body === "object" && Array.isArray(body) === false &&
          body.detail && typeof body.detail === "object" &&
          Array.isArray(body.detail) === false) {
        if (typeof body.detail.code === "string") code = body.detail.code;
        if (typeof body.detail.message === "string") message = body.detail.message;
      } else if (body && typeof body === "object" &&
                 Array.isArray(body.detail) && body.detail.length > 0 &&
                 body.detail[0] && typeof body.detail[0].msg === "string") {
        message = body.detail[0].msg;
      } else if (body && typeof body === "object" &&
                 typeof body.detail === "string") {
        message = body.detail;
      }
      throw { failureKind: "http", status: response.status,
        code: code, message: message, body: body };
    }
    if (parseFailed) {
      throw { failureKind: "response_parse", status: 0, code: null,
        message: "The server returned an unreadable response.", body: null };
    }
    return body;
  }

  function fetchModeSwitchEventsRequest(sessionId) {
    return historyReviewRequestJson("/api/sessions/" + sessionId + "/mode-switch-events");
  }
  function fetchReviewSummariesRequest(sessionId) {
    return historyReviewRequestJson("/api/sessions/" + sessionId + "/history-reviews");
  }
  function fetchReviewDetailRequest(sessionId, reviewId) {
    return historyReviewRequestJson("/api/sessions/" + sessionId + "/history-reviews/" + reviewId);
  }
  function createHistoryReviewRequest(sessionId, payload) {
    return historyReviewRequestJson("/api/sessions/" + sessionId + "/history-reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  function createCancelableDialogAdapter(
    dialog, bodyEl, cancelBtn, continueBtn, titleEl
  ) {
    const heading = titleEl === undefined ? null : titleEl;
    if (typeof createCancelableReviewDialogAdapter === "function") {
      return createCancelableReviewDialogAdapter({
        dialog: dialog, bodyEl: bodyEl,
        cancelBtn: cancelBtn, continueBtn: continueBtn,
        titleEl: heading,
      });
    }
    let pendingCancel = null;
    function clearIfSame(cb) { if (pendingCancel === cb) pendingCancel = null; }
    function addListener(target, name, handler) {
      target.addEventListener(name, handler);
      return function () { target.removeEventListener(name, handler); };
    }
    return {
      showModal: function () { dialog.showModal(); },
      close: function () { if (dialog.open) dialog.close(); },
      setMessage: function (text) { bodyEl.textContent = text; },
      setTitle: function (text) {
        if (heading !== null) heading.textContent = text;
      },
      focusInitial: function () { cancelBtn.focus(); },
      isConnected: function () { return dialog.isConnected; },
      onDialogCancel: function (cb) {
        pendingCancel = cb;
        const handler = function () {
          const current = pendingCancel;
          pendingCancel = null;
          if (typeof current === "function") current();
        };
        const unsub = addListener(dialog, "cancel", handler);
        return function () { clearIfSame(cb); unsub(); };
      },
      onCancelClick: function (cb) {
        pendingCancel = cb;
        const handler = function () {
          const current = pendingCancel;
          pendingCancel = null;
          if (typeof current === "function") current();
        };
        const unsub = addListener(cancelBtn, "click", handler);
        return function () { clearIfSame(cb); unsub(); };
      },
      onContinueClick: function (cb) {
        const handler = function () { cb(); };
        const unsub = addListener(continueBtn, "click", handler);
        return function () { unsub(); };
      },
      cancelPending: function () {
        const current = pendingCancel;
        pendingCancel = null;
        if (typeof current === "function") {
          try { current(); } catch (_) {}
        }
      },
    };
  }

  async function confirmWithReviewFocus(confirmFn, targetSessionId, triggerEl) {
    let confirmed = false;
    try { confirmed = await confirmFn(); } catch (_) { confirmed = false; }
    try {
      if (triggerEl && triggerEl.isConnected &&
          currentSessionId === targetSessionId &&
          findSessionInList(targetSessionId) !== null) {
        triggerEl.focus();
      }
    } catch (_) { /* focus failure must not leak */ }
    return confirmed === true;
  }

  function initializeHistoryReview() {
    try {
      historyReviewStartDialogAdapter = createCancelableDialogAdapter(
        historyReviewStartDialogEl, historyReviewStartBodyEl,
        historyReviewStartCancelBtn, historyReviewStartConfirmBtn,
        historyReviewStartTitleEl);
      historyReviewConsentDialogAdapter = createCancelableDialogAdapter(
        historyReviewConsentDialogEl, historyReviewConsentBodyEl,
        historyReviewConsentCancelBtn, historyReviewConsentConfirmBtn);
      historyReviewStartConfirmer = createRemoteHistoryConfirmer(historyReviewStartDialogAdapter);
      historyReviewConsentConfirmer = createRemoteHistoryConfirmer(historyReviewConsentDialogAdapter);
      historyReviewController = createHistoryReviewController({
        postReview: createHistoryReviewRequest,
        fetchReviewSummaries: fetchReviewSummariesRequest,
        fetchReviewDetail: fetchReviewDetailRequest,
        confirmRemoteHistory: function (metadata, operationSnapshot, title) {
          const key = String(operationSnapshot.targetSessionId);
          const context = reviewDialogContextBySession[key];
          if (context === null || context === undefined) return Promise.resolve(false);
          if (findSessionInList(operationSnapshot.targetSessionId) === null) return Promise.resolve(false);
          if (currentSessionId !== operationSnapshot.targetSessionId) return Promise.resolve(false);
          if (context.stateIdentity !== getReviewStateIfExists(operationSnapshot.targetSessionId)) return Promise.resolve(false);
          if (context.lifecycleEpoch !== reviewEpoch(operationSnapshot.targetSessionId)) return Promise.resolve(false);
          if (context.operationGeneration !== operationSnapshot.generation) return Promise.resolve(false);
          context.pendingAdapter = historyReviewConsentDialogAdapter;
          const text = metadata.sourceMessageCount + " messages\n" +
            metadata.reviewerProfileLabel + " / " + metadata.reviewerModel + "\n" +
            "truncated: " + (metadata.truncated ? "yes" : "no");
          return confirmWithReviewFocus(
            function () {
              return historyReviewConsentConfirmer.confirm(text, title);
            },
            operationSnapshot.targetSessionId,
            context.triggerEl,
          );
        },
        validateTimestamp: isValidApiTimestamp,
      });
      historyReviewInitializationError = null;
      return true;
    } catch (_) {
      historyReviewController = null;
      historyReviewStartConfirmer = null;
      historyReviewConsentConfirmer = null;
      historyReviewInitializationError = "History review is unavailable.";
      return false;
    }
  }

  function renderHistoryReviewFindings(findings) {
    historyReviewFindingsEl.replaceChildren();
    for (const finding of findings) {
      if (isValidHistoryReviewFinding(finding) === false) continue;
      const li = document.createElement("li");
      li.className = "history-review-finding";
      const verdict = document.createElement("span");
      verdict.className = "finding-verdict";
      verdict.textContent =
        finding.verdict === "correct" ? "Correct" :
        finding.verdict === "incorrect" ? "Incorrect" :
        finding.verdict === "uncertain" ? "Uncertain" :
        finding.verdict === "not_a_claim" ? "Not a claim" : "";
      li.appendChild(verdict);
      const claim = document.createElement("div");
      claim.textContent = "Source " + finding.source_message_id + ": " + finding.claim_text;
      li.appendChild(claim);
      if (typeof finding.correction_text === "string") {
        const correction = document.createElement("div");
        correction.textContent = "Correction: " + finding.correction_text;
        li.appendChild(correction);
      }
      if (typeof finding.explanation_text === "string") {
        const explanation = document.createElement("div");
        explanation.textContent = "Explanation: " + finding.explanation_text;
        li.appendChild(explanation);
      }
      historyReviewFindingsEl.appendChild(li);
    }
  }

  function renderHistoryReviewPanel() {
    if (currentSessionId === null) { historyReviewPanelEl.hidden = true; return; }
    const state = getReviewStateIfExists(currentSessionId);
    if (state === null) { historyReviewPanelEl.hidden = true; return; }
    const model = buildHistoryReviewPanelModel({
      sessionId: currentSessionId,
      state: state,
      currentMode: currentSessionInteractionMode(),
      validateTimestamp: isValidApiTimestamp,
      validateModeSwitchEvent: isValidModeSwitchEvent,
    });
    if (model === null || model.visible === false) {
      historyReviewPanelEl.hidden = true;
      return;
    }
    historyReviewPanelEl.hidden = false;
    historyReviewPanelEl.classList.toggle("collapsed", model.collapsed === true);
    historyReviewToggleBtn.textContent = model.collapsed ? "Expand" : "Collapse";
    historyReviewToggleBtn.setAttribute("aria-expanded", model.collapsed ? "false" : "true");
    historyReviewStatusBadgeEl.textContent = model.badgeText;
    historyReviewLiveEl.textContent = model.workingVisible
      ? model.workingLiveText
      : (model.errorText || "");

    if (model.proposal !== null) {
      historyReviewProposalEl.hidden = false;
      historyReviewProposalTextEl.textContent =
        model.proposal.reviewableUserMessageCount + " previous teaching messages are available for review.";
    } else {
      historyReviewProposalEl.hidden = true;
      historyReviewProposalTextEl.textContent = "";
    }

    const hasDetail = model.detail !== null || model.selectedSummary !== null;
    historyReviewDetailEl.hidden = hasDetail ? false : true;

    historyReviewSelectEl.replaceChildren();
    historyReviewSelectEl.hidden = model.selectorVisible ? false : true;
    // The "Review" label is hidden with the control so a single-review
    // session shows neither an empty dropdown nor an orphaned label.  The
    // wrapper owns the layout; the label's own hidden attribute keeps the
    // intent explicit in the DOM (and in tests).
    if (historyReviewSelectorEl !== null) {
      historyReviewSelectorEl.hidden = model.selectorVisible ? false : true;
    }
    const selectorLabel = document.querySelector(
      'label[for="historyReviewSelect"]');
    if (selectorLabel !== null) {
      selectorLabel.hidden = model.selectorVisible ? false : true;
    }
    if (model.selectorVisible) {
      for (const summary of model.summaries) {
        const option = document.createElement("option");
        option.value = String(summary.id);
        option.textContent = "Review #" + summary.id + " · " + summary.status;
        historyReviewSelectEl.appendChild(option);
      }
      if (model.selectedReviewId !== null) {
        historyReviewSelectEl.value = String(model.selectedReviewId);
      }
    }

    historyReviewSummaryEl.textContent = model.summaryText || "";
    historyReviewCoverageEl.textContent = model.coverageText || "";
    historyReviewSummaryEl.hidden = model.summaryText ? false : true;
    historyReviewCoverageEl.hidden = model.coverageText ? false : true;
    renderHistoryReviewFindings(model.findings || []);

    const errorText = model.errorText || "";
    historyReviewErrorEl.textContent = errorText;
    historyReviewErrorEl.hidden = errorText ? false : true;

    const operationActive = state.operation !== null;
    const writeActionBlocked = operationActive || model.busy === true;
    const retryAction = model.actionKind === "retry";
    historyReviewStartBtn.disabled = writeActionBlocked;
    historyReviewDismissBtn.disabled = operationActive;
    historyReviewContinueBtn.disabled = writeActionBlocked;
    historyReviewRecheckBtn.disabled = operationActive ||
      model.actionKind === "none" ||
      (retryAction && model.busy === true);
    historyReviewStartBtn.hidden = model.startVisible ? false : true;
    historyReviewDismissBtn.hidden = model.dismissVisible ? false : true;
    historyReviewContinueBtn.hidden = model.continueVisible ? false : true;
    historyReviewRecheckBtn.hidden = model.actionKind === "none";
    historyReviewRecheckBtn.textContent = model.actionLabel || "Recheck";
  }

  function renderHistoryReviewPanelIfCurrent(sessionId) {
    if (currentSessionId === sessionId) renderHistoryReviewPanel();
  }

  async function loadHistoryReviewDetail(sessionId, reviewId) {
    const state = ensureReviewState(sessionId);
    if (state === null) return;
    const summary = Array.isArray(state.summaries)
      ? state.summaries.find(function (item) { return item.id === reviewId; })
      : null;
    if (summary === undefined || summary === null) return;
    const generation = (state.detailGenerationByReviewId[reviewId] || 0) + 1;
    state.detailGenerationByReviewId[reviewId] = generation;
    const captured = {
      targetSessionId: sessionId, stateIdentity: state,
      epoch: reviewEpoch(sessionId), reviewId: reviewId, generation: generation,
    };
    try {
      const detail = await fetchReviewDetailRequest(sessionId, reviewId);
      if (canApplyCaptured(captured, "detail") === false) return;
      const currentSummary = Array.isArray(state.summaries)
        ? state.summaries.find(function (item) { return item.id === reviewId; })
        : null;
      if (currentSummary === undefined || currentSummary === null) {
        state.detailErrorByReviewId[reviewId] = "Review detail is out of date.";
        renderHistoryReviewPanelIfCurrent(sessionId);
        updateControlStates();
        return;
      }
      if (isValidHistoryReviewDetail(
            detail, sessionId, reviewId,
            currentSummary.mode_switch_event_id, isValidApiTimestamp) === false) {
        state.detailErrorByReviewId[reviewId] = "Invalid review detail.";
        renderHistoryReviewPanelIfCurrent(sessionId);
        updateControlStates();
        return;
      }

      if (isHistoryReviewDetailCurrent(
            detail, currentSummary, sessionId, isValidApiTimestamp)) {
        state.detailsById[reviewId] = detail;
        state.detailStatusByReviewId[reviewId] = "ready";
        state.detailErrorByReviewId[reviewId] = null;
      } else if (historyReviewDetailMayUpdateSummary(
                   detail, currentSummary, sessionId, isValidApiTimestamp)) {
        const derived = historyReviewSummaryFromDetail(
          detail, sessionId, isValidApiTimestamp);
        const next = derived === null ? null : upsertHistoryReviewSummary(
          state.summaries, derived, sessionId, isValidApiTimestamp);
        if (next === null) {
          state.summariesStatus = "error";
          state.summariesError = "Could not update review list.";
          state.detailErrorByReviewId[reviewId] = "Could not update review list.";
        } else {
          state.summaries = next;
          state.summariesStatus = "ready";
          state.summariesError = null;
          state.detailsById[reviewId] = detail;
          state.detailStatusByReviewId[reviewId] = "ready";
          state.detailErrorByReviewId[reviewId] = null;
        }
      } else {
        state.detailErrorByReviewId[reviewId] = "Review detail is out of date.";
      }
      renderHistoryReviewPanelIfCurrent(sessionId);
      updateControlStates();
    } catch (err) {
      if (canApplyCaptured(captured, "detail") === false) return;
      if (isExactSessionNotFound(err)) { removeSessionLocally(sessionId); return; }
      state.detailStatusByReviewId[reviewId] = "error";
      state.detailErrorByReviewId[reviewId] = safeReviewError(err);
      renderHistoryReviewPanelIfCurrent(sessionId);
      updateControlStates();
    }
  }

  async function loadHistoryReviewSession(sessionId, options) {
    if (findSessionInList(sessionId) === null) { clearReviewSessionState(sessionId); return; }
    const state = ensureReviewState(sessionId);
    if (state === null) return;
    const silent = options && options.silent === true;
    const captured = {
      targetSessionId: sessionId, stateIdentity: state,
      epoch: reviewEpoch(sessionId),
      generation: state.sessionLoadGeneration + 1,
    };
    state.sessionLoadGeneration = captured.generation;
    if (silent === false) {
      state.eventsStatus = "loading";
      state.summariesStatus = "loading";
      state.panelError = null;
      renderHistoryReviewPanelIfCurrent(sessionId);
    }
    const settled = await Promise.allSettled([
      fetchModeSwitchEventsRequest(sessionId),
      fetchReviewSummariesRequest(sessionId),
    ]);
    const eventsResult = settled[0];
    const summariesResult = settled[1];
    if (eventsResult.status === "rejected" && isExactSessionNotFound(eventsResult.reason)) {
      removeSessionLocally(sessionId); return;
    }
    if (summariesResult.status === "rejected" && isExactSessionNotFound(summariesResult.reason)) {
      removeSessionLocally(sessionId); return;
    }
    if (canApplyCaptured(captured, "batch") === false) return;
    if (eventsResult.status === "fulfilled" &&
        isValidHistoryReviewEventList(eventsResult.value, sessionId,
          isValidApiTimestamp, isValidModeSwitchEvent)) {
      state.events = eventsResult.value;
      state.eventsStatus = "ready";
      state.eventsError = null;
    } else {
      state.eventsStatus = "error";
      state.eventsError = eventsResult.status === "rejected"
        ? safeReviewError(eventsResult.reason) : "Invalid event list.";
    }
    if (summariesResult.status === "fulfilled" &&
        isValidHistoryReviewSummaryList(summariesResult.value, sessionId,
          isValidApiTimestamp)) {
      state.summaries = summariesResult.value;
      state.summariesStatus = "ready";
      state.summariesError = null;
      const reconciled = reconcileHistoryReviewCaches(
        state.detailsById, state.summaries, sessionId, isValidApiTimestamp);
      if (reconciled !== null) {
        state.summaries = reconciled.summaries;
        state.detailsById = reconciled.detailsById;
        for (const staleId of reconciled.staleReviewIds) {
          state.detailErrorByReviewId[staleId] = "Review detail is out of date.";
        }
      }
    } else {
      state.summariesStatus = "error";
      state.summariesError = summariesResult.status === "rejected"
        ? safeReviewError(summariesResult.reason) : "Invalid review list.";
    }
    if (state.summariesStatus === "ready" && Array.isArray(state.summaries) &&
        state.summaries.length > 0) {
      const selectedExists = state.summaries.some(function (item) {
        return item.id === state.selectedReviewId;
      });
      if (selectedExists === false) state.selectedReviewId = state.summaries[0].id;
      const selectedSummary = state.summaries.find(function (item) {
        return item.id === state.selectedReviewId;
      });
      const cachedDetail = state.detailsById[state.selectedReviewId];
      if (selectedSummary && isHistoryReviewDetailCurrent(
            cachedDetail, selectedSummary, sessionId, isValidApiTimestamp) === false) {
        delete state.detailsById[selectedSummary.id];
        state.detailErrorByReviewId[selectedSummary.id] = null;
        await loadHistoryReviewDetail(sessionId, selectedSummary.id);
      }
    }
    if (canApplyCaptured(captured, "batch") === false) return;
    renderHistoryReviewPanelIfCurrent(sessionId);
    updateControlStates();
  }

  function applyAuthoritativeReviewDetail(state, sessionId, detail) {
    state.detailsById[detail.id] = detail;
    state.selectedReviewId = detail.id;
    state.uncertainEventId = null;
    state.uncertainMessage = "";
    state.proposalEvent = null;
    state.panelError = null;
    const summary = historyReviewSummaryFromDetail(detail, sessionId, isValidApiTimestamp);
    if (summary !== null) {
      const next = upsertHistoryReviewSummary(
        state.summaries, summary, sessionId, isValidApiTimestamp);
      if (next !== null) {
        state.summaries = next;
        state.summariesStatus = "ready";
        state.summariesError = null;
      } else {
        state.summariesStatus = "error";
        state.summariesError = "Could not update review list.";
      }
    }
    renderHistoryReviewPanelIfCurrent(sessionId);
    loadHistoryReviewSession(sessionId, { silent: true });
  }

  async function startHistoryReview(kind, eventId, triggerEl) {
    if (historyReviewController === null) {
      showStatus(historyReviewInitializationError || "History review is unavailable.", true);
      return;
    }
    if (currentSessionId === null) return;
    const sessionId = currentSessionId;
    const state = ensureReviewState(sessionId);
    if (state === null) return;
    if (state.operation !== null) return;
    if (kind !== "recheck" && reviewTargetBusy(sessionId)) return;
    const generation = state.operationGeneration + 1;
    state.operationGeneration = generation;
    state.operation = { kind: kind, eventId: eventId, generation: generation,
      phase: kind === "recheck" ? "rechecking" : "confirming_start" };
    reviewDialogContextBySession[String(sessionId)] = {
      stateIdentity: state, lifecycleEpoch: reviewEpoch(sessionId),
      operationGeneration: generation, triggerEl: triggerEl || null,
      pendingAdapter: null,
    };
    renderHistoryReviewPanelIfCurrent(sessionId);
    updateControlStates();
    try {
      if (kind !== "recheck") {
        const dialogContext = reviewDialogContextBySession[String(sessionId)];
        if (dialogContext) dialogContext.pendingAdapter = historyReviewStartDialogAdapter;
        // The heading is the confirmation question and the body explains
        // the operation; the two never repeat the same sentence.
        const isContinue = kind === "continue";
        const isRetry = kind === "retry";
        const dialogTitle = isRetry
          ? historyReviewDialogCopy.retryTitle
          : (isContinue
            ? historyReviewDialogCopy.continueTitle
            : historyReviewDialogCopy.startTitle);
        const message = isRetry
          ? historyReviewDialogCopy.retryBody
          : (isContinue
            ? historyReviewDialogCopy.continueBody
            : historyReviewDialogCopy.startBody);
        const confirmed = historyReviewStartConfirmer === null
          ? false
          : await confirmWithReviewFocus(
              function () {
                return historyReviewStartConfirmer.confirm(message, dialogTitle);
              },
              sessionId,
              triggerEl,
            );
        if (confirmed !== true) {
          state.operation = null;
          delete reviewDialogContextBySession[String(sessionId)];
          renderHistoryReviewPanelIfCurrent(sessionId);
          return;
        }
        state.operation.phase = "posting";
        renderHistoryReviewPanelIfCurrent(sessionId);
      }
      const operation = {
        targetSessionId: sessionId,
        modeSwitchEventId: eventId,
        generation: generation,
      };
      const outcome = kind === "recheck"
        ? await historyReviewController.recheck(operation)
        : (kind === "retry"
          ? await historyReviewController.retry(operation)
          : await historyReviewController.start(operation));
      const captured = {
        targetSessionId: sessionId, stateIdentity: state,
        epoch: reviewEpoch(sessionId), generation: generation,
      };
      if (canApplyCaptured(captured, "operation") === false) return;
      if (outcome.status === "authoritative") {
        applyAuthoritativeReviewDetail(state, sessionId, outcome.detail);
      } else if (outcome.status === "uncertain") {
        state.uncertainEventId = eventId;
        state.uncertainMessage = outcome.message || "";
        state.panelError = outcome.message || "";
      } else if (outcome.status === "cancelled") {
        state.panelError = "History review was cancelled.";
      } else if (outcome.status === "session_not_found") {
        removeSessionLocally(sessionId);
        return;
      } else if (outcome.status === "busy") {
        state.panelError = "Another history review operation is already running.";
      } else if (outcome.status === "invalid_request") {
        state.panelError = "Invalid history review request.";
      } else if (outcome.status === "failed") {
        state.panelError = outcome.message || "History review failed.";
      }
      renderHistoryReviewPanelIfCurrent(sessionId);
    } catch (_) {
      state.panelError = "History review request failed.";
      renderHistoryReviewPanelIfCurrent(sessionId);
    } finally {
      if (state.operation != null && state.operation.generation === generation) {
        state.operation = null;
      }
      const context = reviewDialogContextBySession[String(sessionId)];
      if (context && context.operationGeneration === generation) {
        delete reviewDialogContextBySession[String(sessionId)];
      }
      renderHistoryReviewPanelIfCurrent(sessionId);
      updateControlStates();
    }
  }

  function handleInteractionModeReviewOutcome(targetSessionId, outcome) {
    if (outcome.status != "switched") return;
    const state = ensureReviewState(targetSessionId);
    if (state === null) return;
    if (outcome.switchEvent && isValidModeSwitchEvent(outcome.switchEvent, isValidApiTimestamp)) {
      const existing = Array.isArray(state.events) ? state.events : [];
      state.events = existing
        .filter(function (event) { return event.id === outcome.switchEvent.id ? false : true; })
        .concat([outcome.switchEvent]);
    }
    loadHistoryReviewSession(targetSessionId, { silent: true });
  }

  function handleHistoryReviewSelectChange() {
    if (currentSessionId === null) return;
    const state = getReviewStateIfExists(currentSessionId);
    if (state === null) return;
    const reviewId = Number(historyReviewSelectEl.value);
    if (!Number.isSafeInteger(reviewId) || reviewId < 1) return;
    state.selectedReviewId = reviewId;
    if (state.detailsById[reviewId] === undefined) {
      loadHistoryReviewDetail(currentSessionId, reviewId);
    }
    renderHistoryReviewPanel();
  }

  function handleHistoryReviewToggle() {
    if (currentSessionId === null) return;
    const state = getReviewStateIfExists(currentSessionId);
    if (state === null) return;
    state.collapsed = state.collapsed === true ? false : true;
    renderHistoryReviewPanel();
  }

  function handleHistoryReviewRecheckAction() {
    if (currentSessionId === null) return;
    const state = getReviewStateIfExists(currentSessionId);
    if (state === null) return;
    const model = buildHistoryReviewPanelModel({
      sessionId: currentSessionId, state: state,
      currentMode: currentSessionInteractionMode(),
      validateTimestamp: isValidApiTimestamp,
      validateModeSwitchEvent: isValidModeSwitchEvent,
    });
    if (model === null) return;
    if (model.actionKind === "reload") {
      if (Number.isSafeInteger(model.actionReviewId) && model.actionReviewId > 0) {
        loadHistoryReviewDetail(currentSessionId, model.actionReviewId);
      } else {
        loadHistoryReviewSession(currentSessionId);
      }
      return;
    }
    if (model.actionKind === "retry" &&
        Number.isSafeInteger(model.actionEventId)) {
      startHistoryReview("retry", model.actionEventId, historyReviewRecheckBtn);
    } else if (model.actionKind === "recheck" &&
               Number.isSafeInteger(model.actionEventId)) {
      startHistoryReview("recheck", model.actionEventId, historyReviewRecheckBtn);
    }
  }

  // Controller + confirmer, initialised once in init().
  let switchController = null;
  let profileSwitchInitializationError = null;   // string | null

  const UNCERTAIN_REAPPLY_TEXT =
    "The previous model switch could not be confirmed. Apply again to " +
    "check the current binding before retrying.";

  /* ---- Derived helpers --------------------------------------------- */

  /** The registry analysis of the current profile list. */
  function registry() {
    return analyzeProfileRegistry(profiles);
  }

  /** The SessionResponse for currentSessionId, or null. */
  function currentSession() {
    if (currentSessionId === null) return null;
    for (let i = 0; i < sessions.length; i++) {
      if (sessions[i].id === currentSessionId) return sessions[i];
    }
    return null;
  }

  /** Whether creating new sessions is currently possible. */
  function registryUsable() {
    return profilesLoadState === "ready" && registry().status === "valid";
  }

  /** Whether the current session (or none) can accept a message.
   *  An uncertain-switch record for the current session always blocks
   *  sending — even when the cached llm_profile_status is "ready". */
  function currentSessionWritable() {
    if (hasInteractionModeUncertain(
      interactionModeUncertainBySession, currentSessionId,
    )) {
      return false;
    }
    return isSessionWritable(
      currentSession(),
      sessionSendBlocks[currentSessionId] || null,
      sessionSwitchUncertain[currentSessionId],
    );
  }

  /** The current session authoritative interaction mode, or null when unknown. */
  function currentSessionInteractionMode() {
    return interactionModeAuthoritativeMode(currentSession());
  }

  /** Whether the current mode draft can be applied/rechecked right now. */
  function interactionModeApplyEnabled() {
    if (interactionModeController === null ||
        interactionModeInitializationError !== null ||
        isInitializing || isSending || isCreatingSession || isRenaming ||
        isRenameSaving || isDeletingSession || isProfileSwitching ||
        isInteractionModeSwitching) {
      return false;
    }
    const session = currentSession();
    if (session === null) return false;
    return canApplyInteractionMode({
      session: session,
      draftMode: currentInteractionModeDraft,
      isSwitching: isInteractionModeSwitching,
      hasUncertain: hasInteractionModeUncertain(
        interactionModeUncertainBySession, session.id,
      ),
    });
  }

  /** kind of a profile — only from a structurally valid registry. */
  function lookupProfileKind(profileId) {
    return profileKindFromRegistry(registry(), profileId);
  }

  /** Find a session in the local list by id (no insertion). */
  function findSessionInList(sessionId) {
    if (!Number.isSafeInteger(sessionId)) return null;
    for (let i = 0; i < sessions.length; i++) {
      if (sessions[i].id === sessionId) return sessions[i];
    }
    return null;
  }

  /** Per-session record keys must be positive safe integers. */
  function isValidSessionIdKey(id) {
    return Number.isSafeInteger(id) && id >= 1;
  }

  /** Set the ordinary switch status for ONE session (target-scoped). */
  function setProfileSwitchStatus(sessionId, text, isError) {
    if (!isValidSessionIdKey(sessionId)) return;
    profileSwitchStatusBySession[sessionId] = {
      text: typeof text === "string" ? text : "",
      isError: isError === true,
    };
    renderCurrentProfileStatus();
  }

  /** Clear the ordinary switch status for ONE session. */
  function clearProfileSwitchStatus(sessionId) {
    if (!isValidSessionIdKey(sessionId)) return;
    delete profileSwitchStatusBySession[sessionId];
    renderCurrentProfileStatus();
  }

  /** Render the status area for the CURRENT session only.  The
   * uncertain record has the highest display priority; ordinary
   * per-session status comes second; otherwise the DOM is cleared. */
  function renderCurrentProfileStatus() {
    let text = "";
    let isError = false;

    if (profileSwitchInitializationError !== null &&
        currentSessionId !== null) {
      text = profileSwitchInitializationError;
      isError = true;
    } else if (currentSessionId !== null &&
               sessionSwitchUncertain[currentSessionId] !== undefined) {
      text = UNCERTAIN_REAPPLY_TEXT;
      isError = true;
    } else if (currentSessionId !== null &&
               profileSwitchStatusBySession[currentSessionId] !== undefined) {
      text = profileSwitchStatusBySession[currentSessionId].text;
      isError = profileSwitchStatusBySession[currentSessionId].isError;
    }

    currentProfileStatusEl.textContent = text;
    currentProfileStatusEl.className =
      "current-profile-status" + (isError ? " error" : "");
  }

  /** Apply the outcome's ordinary-status effect to the TARGET session
   * only — independent of which session is currently visible. */
  function applyTargetProfileSwitchStatus(targetSessionId, outcome, plan) {
    if (!isValidSessionIdKey(targetSessionId)) return;
    switch (outcome.status) {
      case "switched":
      case "cancelled":
        clearProfileSwitchStatus(targetSessionId);
        break;
      case "not_changed":
      case "validation_error":
      case "failed":
        if (typeof plan.showStatus === "string" &&
            plan.showStatus.trim() !== "") {
          setProfileSwitchStatus(targetSessionId, plan.showStatus, true);
        }
        break;
      case "uncertain":
        // The uncertain record lives in sessionSwitchUncertain; the
        // persistent hint is rendered from it (single source).
        break;
      case "busy":
      default:
        break;
    }
  }

  /** Raw history state → enum for needsRemoteHistoryConfirmation. */
  function historyStateFor(sessionId) {
    return historyStateFromValue(sessionHasMessages[sessionId]);
  }

  /** Write the has-messages state for a session (undefined = unknown). */
  function setHistoryState(sessionId, state) {
    if (!isValidSessionIdKey(sessionId)) return;
    if (state === undefined) {
      delete sessionHasMessages[sessionId];
    } else {
      sessionHasMessages[sessionId] = state === true;
    }
  }

  /* ---- Custom error ------------------------------------------------ */

  class ApiError extends Error {
    constructor(message, status = 0, code = null) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
    }
  }

  /* ---- Unified API request ----------------------------------------- */

  /**
   * Thin wrapper around fetch() that normalises errors.
   *
   * Returns the Response on success.  Throws ApiError with a
   * human-readable English message on any failure.
   */
  async function apiRequest(url, options = {}) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (_) {
      throw new ApiError(
        "Network error. Please check your connection.",
        0,
      );
    }

    if (!response.ok) {
      let detail = "Something went wrong. Please try again.";
      let code = null;
      try {
        const body = await response.json();
        if (typeof body.detail === "string") {
          detail = body.detail;
        } else if (
          body !== null && typeof body === "object" &&
          !Array.isArray(body) &&
          typeof body.detail === "object" && body.detail !== null &&
          !Array.isArray(body.detail) &&
          typeof body.detail.code === "string" &&
          typeof body.detail.message === "string"
        ) {
          // Structured error body (e.g. the model-switch 409) — the
          // stable code is preserved; callers never match on text.
          code = body.detail.code;
          detail = body.detail.message;
        } else if (
          Array.isArray(body.detail) &&
          body.detail.length > 0 &&
          body.detail[0].msg
        ) {
          detail = body.detail[0].msg;
        }
      } catch (_) {
        /* non-JSON body — use default detail */
      }
      throw new ApiError(detail, response.status, code);
    }

    return response;
  }

  /* ---- API calls --------------------------------------------------- */

  async function fetchSessions() {
    const resp = await apiRequest("/api/sessions");
    return resp.json();
  }

  async function fetchSession(sessionId) {
    const resp = await apiRequest("/api/sessions/" + sessionId);
    return resp.json();
  }

  async function fetchProfiles() {
    const resp = await apiRequest("/api/llm/profiles");
    return resp.json();
  }

  async function createSessionRequest(profileId) {
    const resp = await apiRequest("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildCreateSessionPayload(profileId)),
    });
    return resp.json();
  }

  async function fetchMessages(sessionId) {
    const resp = await apiRequest(
      "/api/sessions/" + sessionId + "/messages",
    );
    return resp.json();
  }

  async function sendSessionMessageRequest(sessionId, text) {
    const resp = await apiRequest(
      "/api/sessions/" + sessionId + "/messages",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      },
    );
    return resp.json();
  }

  async function deleteSessionRequest(sessionId) {
    await apiRequest("/api/sessions/" + sessionId, { method: "DELETE" });
  }

  async function renameSessionRequest(sessionId, title) {
    const resp = await apiRequest(
      "/api/sessions/" + sessionId,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title }),
      },
    );
    return resp.json();
  }

  async function switchInteractionModeRequest(sessionId, mode) {
    const payload = buildSwitchInteractionModePayload(mode);
    if (payload === null) {
      throw {
        failureKind: "validation", status: 0,
        message: "Choose a valid interaction mode.",
      };
    }
    let response;
    try {
      response = await fetch(
        "/api/sessions/" + sessionId + "/interaction-mode",
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
    } catch (_) {
      throw {
        failureKind: "network", status: 0,
        message: "Network error. Please check your connection.",
      };
    }
    if (response.ok === false) {
      let message = "Something went wrong. Please try again.";
      try {
        const body = await response.json();
        if (body !== null && typeof body === "object" &&
            !Array.isArray(body) && typeof body.detail === "string") {
          message = body.detail;
        } else if (body !== null && typeof body === "object" &&
                   !Array.isArray(body) && Array.isArray(body.detail) &&
                   body.detail.length > 0 && body.detail[0] &&
                   typeof body.detail[0].msg === "string") {
          message = body.detail[0].msg;
        }
      } catch (_) {
        // keep default message
      }
      throw {
        failureKind: "http", status: response.status,
        message: message,
      };
    }
    try {
      return await response.json();
    } catch (_) {
      throw {
        failureKind: "response_parse", status: 0,
        message: "The server returned an unreadable response.",
      };
    }
  }

  /* ---- Model switch API wrappers ----------------------------------- */

  /** PATCH the switch endpoint.  Success returns parsed JSON (validated
   * by the controller); failures throw normalised error objects with
   * failureKind "http" | "network" | "response_parse". */
  async function switchProfileRequest(sessionId, profileId, ack) {
    let response;
    try {
      response = await fetch("/api/sessions/" + sessionId + "/llm-profile", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSwitchSessionProfilePayload(profileId, ack)),
      });
    } catch (_) {
      throw {
        failureKind: "network", status: 0, code: null,
        message: "Network error. Please check your connection.",
      };
    }

    if (!response.ok) {
      let code = null;
      let message = "Something went wrong. Please try again.";
      let body = null;
      try { body = await response.json(); } catch (_) { body = null; }
      const structured = parseRemoteHistoryAckRequired(body);
      if (structured !== null) {
        code = structured.code;
        message = structured.message;
      } else if (body !== null && typeof body === "object" &&
                 !Array.isArray(body) && typeof body.detail === "string") {
        message = body.detail;
      } else if (body !== null && typeof body === "object" &&
                 !Array.isArray(body) && Array.isArray(body.detail) &&
                 body.detail.length > 0 && body.detail[0] &&
                 typeof body.detail[0].msg === "string") {
        message = body.detail[0].msg;
      }
      throw {
        failureKind: "http", status: response.status, code: code,
        message: message,
      };
    }

    try {
      return await response.json();
    } catch (_) {
      throw {
        failureKind: "response_parse", status: 0, code: null,
        message: "The server returned an unreadable response.",
      };
    }
  }

  /** GET a single session as parsed JSON; errors follow the same
   * normalised contract as switchProfileRequest. */
  async function fetchOneSessionRaw(sessionId) {
    let response;
    try {
      response = await fetch("/api/sessions/" + sessionId);
    } catch (_) {
      throw {
        failureKind: "network", status: 0, code: null,
        message: "Network error. Please check your connection.",
      };
    }
    if (!response.ok) {
      let body = null;
      try { body = await response.json(); } catch (_) { body = null; }
      let message = "Something went wrong. Please try again.";
      if (body !== null && typeof body === "object" && !Array.isArray(body) &&
          typeof body.detail === "string") {
        message = body.detail;
      }
      throw {
        failureKind: "http", status: response.status, code: null,
        message: message,
      };
    }
    try {
      return await response.json();
    } catch (_) {
      throw {
        failureKind: "response_parse", status: 0, code: null,
        message: "The server returned an unreadable response.",
      };
    }
  }

  /** Structured single-session fetch with NO cache side effects. */
  async function fetchAndValidateOneSession(sessionId) {
    let raw;
    try {
      raw = await fetchOneSessionRaw(sessionId);
    } catch (err) {
      if (err && typeof err === "object" && err.failureKind === "http" &&
          err.status === 404) {
        return { status: "not_found" };
      }
      return { status: "failed", message: err && err.message ? err.message : "" };
    }
    if (!isValidSessionResponse(raw) || raw.id !== sessionId) {
      return { status: "invalid_response", message: "Invalid session response." };
    }
    return { status: "ok", session: raw };
  }

  /* ---- Rendering: profile selector ---------------------------------- */

  /**
   * Rebuild the selector options and its persistent status text.
   *
   * Called ONLY when profile content or the resolved selection changes
   * (initial load, 422 reload).  Never called from routine control
   * synchronisation — rebuilding options would churn the DOM and drop
   * the select's focus.
   */
  function appendPlaceholderOption(selectEl) {
    const option = document.createElement("option");
    option.value = "";
    option.disabled = true;
    option.selected = true;
    option.textContent = "Select a model";
    selectEl.appendChild(option);
  }

  function renderProfileSelectorContent() {
    const hadFocus = document.activeElement === profileSelectEl;
    const reg = registry();

    profileSelectEl.replaceChildren();

    if (reg.status === "valid") {
      for (const p of reg.profiles) {
        const option = document.createElement("option");
        option.value = p.id;
        option.textContent = p.label;
        profileSelectEl.appendChild(option);
      }
      if (selectedProfileId !== null) {
        profileSelectEl.value = selectedProfileId;
      } else {
        // Explicit placeholder — never silently fall back to the
        // first profile.
        appendPlaceholderOption(profileSelectEl);
        profileSelectEl.value = "";
      }
    } else {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No models available";
      profileSelectEl.appendChild(option);
      profileSelectEl.value = "";
    }

    profileSelectorStatusEl.textContent = profileRegistryStatusText(
      reg.status,
      profilesLoadError,
    );

    // Minimal, explainable focus restoration: only when the select
    // held focus before the rebuild and is still usable.
    if (hadFocus && !profileSelectEl.disabled) {
      profileSelectEl.focus();
    }
  }

  /* ---- Rendering: session list -------------------------------------- */

  /**
   * Render the session list and its persistent status area.
   *
   * Handles list content, the empty-list message, initial load errors
   * and stale-list warnings in one place so the list is never hidden
   * or faked when an error exists.
   */
  function renderSessionListOrError() {
    if (
      sessionsLoadState === "error" &&
      sessions.length === 0 &&
      !sessionsEverLoaded
    ) {
      // Initial load failed — show an error, never "No conversations
      // yet.".
      sessionListEl.replaceChildren();
      const errDiv = document.createElement("div");
      errDiv.className = "sidebar-empty";
      errDiv.textContent = "Couldn't load conversations.";
      sessionListEl.appendChild(errDiv);

      sessionListStatusEl.textContent =
        sessionsLoadError || "Couldn't load conversations.";
      sessionListStatusEl.classList.remove("warning");
      sessionListStatusEl.hidden = false;
      return;
    }

    renderSessionList();

    if (sessionsLoadState === "error") {
      // Keep the cached/known list visible with a warning.
      if (sessionsEverLoaded) {
        sessionListStatusEl.textContent =
          "Conversation list may be out of date.";
      } else {
        sessionListStatusEl.textContent =
          "Some conversations may not have loaded.";
      }
      sessionListStatusEl.classList.add("warning");
      sessionListStatusEl.hidden = false;
    } else {
      sessionListStatusEl.textContent = "";
      sessionListStatusEl.classList.remove("warning");
      sessionListStatusEl.hidden = true;
    }
  }

  /* ---- Rendering: current session meta ------------------------------ */

  /**
   * Title, model badge, and compatibility notice for the current
   * session.  Everything comes from the SessionResponse itself; the
   * kind chip only when the current registry is valid.
   */
  function renderCurrentSessionMeta() {
    const session = currentSession();

    if (session === null) {
      sessionTitleEl.textContent = "";
      sessionModelBadgeEl.replaceChildren();
      sessionModelBadgeEl.hidden = true;
      interactionModeBadgeEl.replaceChildren();
      interactionModeBadgeEl.className = "interaction-mode-badge";
      interactionModeBadgeEl.hidden = true;
      sessionCompatibilityNoticeEl.textContent = "";
      sessionCompatibilityNoticeEl.hidden = true;
      return;
    }

    // -- title ----------------------------------------------------------
    sessionTitleEl.textContent = session.title;

    // -- badge: kind chip (valid registry only) + session's own label ---
    sessionModelBadgeEl.replaceChildren();
    const kindText = profileKindBadgeText(
      lookupProfileKind(session.llm_profile_id),
    );
    if (kindText !== null) {
      const kindSpan = document.createElement("span");
      kindSpan.className = "badge-kind";
      kindSpan.textContent = kindText;
      sessionModelBadgeEl.appendChild(kindSpan);
    }
    const labelSpan = document.createElement("span");
    labelSpan.className = "badge-label";
    labelSpan.textContent = session.llm_profile_label || "";
    sessionModelBadgeEl.appendChild(labelSpan);
    sessionModelBadgeEl.hidden = false;

    interactionModeBadgeEl.replaceChildren();
    if (hasInteractionModeUncertain(
      interactionModeUncertainBySession, session.id,
    )) {
      interactionModeBadgeEl.textContent = "Mode uncertain";
      interactionModeBadgeEl.className =
        "interaction-mode-badge uncertain";
    } else {
      const authoritativeMode = currentSessionInteractionMode();
      const modeBadgeText = interactionModeBadgeText(authoritativeMode);
      interactionModeBadgeEl.textContent =
        modeBadgeText || "Mode unavailable";
      interactionModeBadgeEl.className = "interaction-mode-badge";
    }
    interactionModeBadgeEl.hidden = false;

    // -- compatibility notice: temporary block first, then server status
    const block = sessionSendBlocks[session.id] || null;
    const blockText = temporaryBlockExplanation(block);
    const noticeText =
      blockText !== null
        ? blockText
        : readOnlyExplanation(session.llm_profile_status);

    if (noticeText) {
      sessionCompatibilityNoticeEl.textContent = noticeText;
      sessionCompatibilityNoticeEl.hidden = false;
    } else {
      sessionCompatibilityNoticeEl.textContent = "";
      sessionCompatibilityNoticeEl.hidden = true;
    }
  }

  /* ---- Control state helpers --------------------------------------- */

  /**
   * Sync disabled states and placeholders only.  Never touches focus
   * and never rebuilds selector options.
   *
   * Tiered blocks:
   * - blockSessionSelection — session switching disabled
   * - blockSessionActions  — rename/delete disabled
   * - blockCreate          — New Chat disabled
   * - blockProfileSelect   — selector disabled
   * - blockSend            — input/Send disabled
   *
   * Session switching stays available while a send is in progress
   * (sendingSessionId guards stale renders); everything else is
   * locked during send/create.
   */
  function updateControlStates() {
    const blockBase = isInitializing || isRenaming || isRenameSaving;
    const blockSessionSelection = blockBase || isCreatingSession;
    const blockSessionActions =
      blockBase || isCreatingSession || isSending ||
      isProfileSwitching || isDeletingSession ||
      isInteractionModeSwitching;
    const usable = registryUsable();
    const writable = currentSessionWritable();
    const reviewBusy = currentSessionId === null
      ? false
      : reviewTargetBusy(currentSessionId);

    const blockCreate =
      blockSessionActions || !usable || selectedProfileId === null;
    const blockProfileSelect =
      blockSessionActions || !usable || profiles.length <= 1;
    const blockSend =
      blockBase || isCreatingSession || isSending ||
      isProfileSwitching || isDeletingSession ||
      isInteractionModeSwitching || !writable || reviewBusy ||
      (currentSessionId === null && !usable);

    const switchInitialized =
      switchController !== null && profileSwitchInitializationError === null;
    const interactionModeUncertain = hasInteractionModeUncertain(
      interactionModeUncertainBySession, currentSessionId,
    );
    const authoritativeInteractionMode =
      currentSessionInteractionMode();
    const blockCurrentProfile = blockSessionActions || !usable ||
      !switchInitialized || currentSessionId === null ||
      interactionModeUncertain || reviewBusy;
    const blockInteractionMode = blockSessionActions ||
      currentSessionId === null || reviewBusy;

    sendBtn.disabled = blockSend;
    inputEl.disabled = blockSend;
    newChatBtn.disabled = blockCreate;
    profileSelectEl.disabled = blockProfileSelect;
    currentProfileSelectEl.disabled = blockCurrentProfile;
    applyProfileBtn.disabled = blockCurrentProfile || !applyEnabled();
    interactionModeSelectEl.disabled = blockInteractionMode ||
      interactionModeUncertain;
    applyInteractionModeBtn.disabled = blockInteractionMode ? true :
      (interactionModeUncertain ? false :
        interactionModeApplyEnabled() === false);
    applyInteractionModeBtn.textContent =
      interactionModeUncertain ? "Recheck" : "Apply";

    if (interactionModeUncertain) {
      inputEl.placeholder =
        "Interaction mode uncertain. Use Recheck before sending.";
    } else if (blockSend && currentSessionId !== null && !writable) {
      inputEl.placeholder =
        "This conversation is read-only. Start a new chat to continue.";
    } else if (currentSessionId !== null &&
               authoritativeInteractionMode === CORRECTIVE_MODE) {
      inputEl.placeholder = "Continue; the agent will check your claims...";
    } else if (currentSessionId !== null) {
      inputEl.placeholder = "Teach your learner...";
    } else {
      inputEl.placeholder = "Type a message...";
    }

    const deleteBtns = document.querySelectorAll(".delete-session-btn");
    deleteBtns.forEach(function (btn) {
      const sid = Number(btn.dataset.sessionId);
      const targetBusy = Number.isSafeInteger(sid) && sid > 0 && reviewTargetBusy(sid);
      btn.disabled = blockSessionActions || targetBusy;
    });

    const renameBtns = document.querySelectorAll(".rename-session-btn");
    renameBtns.forEach(function (btn) {
      const sid = Number(btn.dataset.sessionId);
      const targetBusy = Number.isSafeInteger(sid) && sid > 0 && reviewTargetBusy(sid);
      btn.disabled = blockSessionActions || targetBusy;
    });

    const selectBtns = document.querySelectorAll(".session-select-btn");
    selectBtns.forEach(function (btn) {
      btn.disabled = blockSessionSelection;
    });

    // Rename input / Save / Cancel are only controlled by isRenameSaving:
    // - Editing (isRenaming=true, isRenameSaving=false): enabled
    // - Saving  (isRenaming=true, isRenameSaving=true):  disabled
    if (isRenaming) {
      const input = document.querySelector(".session-rename-input");
      const saveBtn = document.querySelector(".session-rename-save-btn");
      const cancelBtn = document.querySelector(".session-rename-cancel-btn");
      if (input) input.disabled = isRenameSaving;
      if (saveBtn) saveBtn.disabled = isRenameSaving;
      if (cancelBtn) cancelBtn.disabled = isRenameSaving;
    }
  }

  /**
   * Synchronise the current-session UI.  Only renders meta and
   * controls — never rebuilds selector options.
   */
  function syncCurrentSessionUI() {
    renderCurrentSessionMeta();
    updateControlStates();
  }

  /** Set the mode draft from a session without rebuilding options. */
  function setInteractionModeDraftForSession(session) {
    if (session !== null && typeof session === "object" &&
        hasInteractionModeUncertain(
          interactionModeUncertainBySession, session.id,
        )) {
      currentInteractionModeDraft =
        interactionModeUncertainBySession[String(session.id)].requestedMode;
    } else {
      currentInteractionModeDraft =
        interactionModeDraftForSession(session);
    }
    renderCurrentInteractionModeStatus();
  }

  /**
   * Apply a profile-draft plan to the current-session model controls.
   *
   * ``plan`` comes from ``planProfileDraftForSelection``.  An
   * unchanged selection keeps the draft untouched and never rebuilds
   * the dropdown (no option churn, no focus loss); a changed
   * selection re-renders the bar — the ONLY place that shows or hides
   * it — which also re-renders the status area.  With no current
   * session the bar hides and the status text clears, while the
   * per-session records of other sessions stay untouched.
   */
  function applySelectionDraftPlan(plan) {
    currentProfileDraftId = plan.draftId;
    if (plan.selectionChanged) {
      renderCurrentProfileBar();
    } else {
      renderCurrentProfileStatus();
    }
  }

  function setSendingState(on) {
    isSending = on;
    updateControlStates();
  }

  /* ---- Status helpers ---------------------------------------------- */

  function showStatus(text, isError) {
    statusEl.textContent = text;
    statusEl.className = isError ? "chat-status error" : "chat-status";
  }

  function clearStatus() {
    statusEl.textContent = "";
    statusEl.className = "chat-status";
  }

  /* ---- Messages ---------------------------------------------------- */

  function scrollMessagesToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function clearMessages() {
    messagesEl.replaceChildren();
  }

  /** Remove any .welcome-state node from the messages container. */
  function removeEmptyState() {
    const el = messagesEl.querySelector(".welcome-state");
    if (el) {
      el.remove();
    }
  }

  /**
   * Append a single message bubble using textContent (XSS-safe).
   *
   * Removes the welcome / empty-state placeholder on the first real
   * message so it never coexists with message bubbles.
   */
  function appendMessage(role, content) {
    removeEmptyState();

    var wrapper = document.createElement("div");
    wrapper.className = "message " + role;

    var contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = content;

    wrapper.appendChild(contentDiv);

    // Assistant messages get a copy button
    if (role === "assistant") {
      var actionsWrapper = document.createElement("div");
      actionsWrapper.className = "message-actions";

      var copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "copy-btn";
      copyBtn.setAttribute("aria-label", "Copy response");
      copyBtn.textContent = "Copy";
      copyBtn.dataset.state = "idle";

      var liveRegion = document.createElement("span");
      liveRegion.className = "sr-only";
      liveRegion.setAttribute("aria-live", "polite");
      liveRegion.setAttribute("aria-atomic", "true");

      var ctrl = createCopyController(
        function (t) {
          return copyToClipboard(t, navigator.clipboard, document);
        },
        function (fn, ms) { return setTimeout(fn, ms); },
        function (id) { clearTimeout(id); },
        function (newState) {
          if (!copyBtn.isConnected) return;

          copyBtn.dataset.state = newState;

          switch (newState) {
            case "copying":
              copyBtn.textContent = "Copying…";
              copyBtn.disabled = true;
              liveRegion.textContent = "Copying response to clipboard.";
              break;
            case "copied":
              copyBtn.textContent = "Copied";
              copyBtn.disabled = false;
              liveRegion.textContent = "Response copied to clipboard.";
              break;
            case "failed":
              copyBtn.textContent = "Failed";
              copyBtn.disabled = false;
              liveRegion.textContent = "Failed to copy response. Press to retry.";
              break;
            default:
              copyBtn.textContent = "Copy";
              copyBtn.disabled = false;
              liveRegion.textContent = "";
              break;
          }
        },
        function () { return copyBtn.isConnected; }
      );

      copyBtn.addEventListener("click", function () {
        ctrl.handleClick(content);
      });

      actionsWrapper.appendChild(copyBtn);
      wrapper.appendChild(actionsWrapper);
      wrapper.appendChild(liveRegion);
    }

    messagesEl.appendChild(wrapper);
    scrollMessagesToBottom();
  }

  /**
   * Replace the messages area with a full history render.
   *
   * Skips unknown roles silently (XSS-safe by design — only
   * "user" and "assistant" are rendered).
   */
  function renderMessages(messages) {
    clearMessages();
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role === "user" || msg.role === "assistant") {
        appendMessage(msg.role, msg.content);
      }
    }
    // Track the last known message id for this session so network-error
    // recovery can use an ID boundary instead of text matching.
    if (messages.length > 0) {
      sessionLastMessageId[currentSessionId] = messages[messages.length - 1].id;
    } else {
      sessionLastMessageId[currentSessionId] = 0;
    }
  }

  /* ---- Welcome / empty states -------------------------------------- */

  function renderWelcome() {
    clearMessages();
    const div = document.createElement("div");
    div.className = "welcome-state";

    const h2 = document.createElement("h2");
    h2.textContent = "Welcome to Static Chatbot";

    const p = document.createElement("p");
    p.textContent =
      "Click + New Chat to start a conversation, " +
      "or type a message below to begin.";

    div.appendChild(h2);
    div.appendChild(p);
    messagesEl.appendChild(div);
  }

  function renderEmptyChat() {
    clearMessages();
    const div = document.createElement("div");
    div.className = "welcome-state";

    const p = document.createElement("p");
    p.textContent = "Start a conversation...";

    div.appendChild(p);
    messagesEl.appendChild(div);
  }

  /* ---- Time formatting --------------------------------------------- */

  /**
   * Parse an ISO-8601 string from the API.
   *
   * Backend timestamps are UTC but may lack a timezone suffix
   * (e.g. "2026-08-06T14:00:00").  Browsers parse such strings as
   * local time, causing an offset equal to the local timezone.
   *
   * This helper detects the missing suffix and appends "Z" so the
   * string is always interpreted as UTC.  Strings that already carry
   * a timezone ("Z", "+HH:MM", "-HH:MM") are left unchanged.
   *
   * Returns a Date, or null when the input is not a valid string.
   */
  function parseApiDate(isoString) {
    if (typeof isoString !== "string" || !isoString) {
      return null;
    }

    const hasTimezone =
      /Z$/i.test(isoString) ||
      /[+-]\d{2}:\d{2}$/.test(isoString);

    const date = new Date(hasTimezone ? isoString : isoString + "Z");

    return Number.isNaN(date.getTime()) ? null : date;
  }

  function formatSessionTime(isoString) {
    const now = new Date();
    const date = parseApiDate(isoString);

    // Fallback for unparseable timestamps
    if (date === null) {
      return "";
    }

    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);

    if (diffMin < 1) return "Just now";
    if (diffMin < 60) return diffMin + "m ago";
    if (diffHour < 24) return diffHour + "h ago";
    if (diffDay < 7) return diffDay + "d ago";

    const months = [
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    return months[date.getMonth()] + " " + date.getDate();
  }

  /* ---- Session list rendering -------------------------------------- */

  function renderSessionList() {
    sessionListEl.replaceChildren();

    if (sessions.length === 0) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "sidebar-empty";
      emptyDiv.textContent = "No conversations yet.";
      sessionListEl.appendChild(emptyDiv);
      return;
    }

    // Use for…of with const — each iteration gets a fresh binding
    // so event listeners capture the correct session id.
    for (const session of sessions) {
      const sid = session.id;

      // Layout container — no interactive role
      const item = document.createElement("div");
      item.className = "session-item";
      if (sid === currentSessionId) {
        item.classList.add("active");
      }

      // -- Inline-edit mode for this session -----------------------------
      if (isRenaming && renamingSessionId === sid) {
        item.classList.add("editing");

        const input = document.createElement("input");
        input.type = "text";
        input.className = "session-rename-input";
        input.value = session.title;
        input.setAttribute("aria-label", "Rename session");

        const saveBtn = document.createElement("button");
        saveBtn.type = "button";
        saveBtn.className = "session-rename-save-btn";
        saveBtn.textContent = "Save";

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "session-rename-cancel-btn";
        cancelBtn.textContent = "Cancel";

        input.addEventListener("keydown", function (event) {
          if (event.key === "Enter" && !event.isComposing) {
            event.preventDefault();
            saveRename(sid, input);
          } else if (event.key === "Escape") {
            event.preventDefault();
            cancelRename();
          }
        });

        saveBtn.addEventListener("click", function () {
          saveRename(sid, input);
        });
        cancelBtn.addEventListener("click", function () {
          cancelRename();
        });

        item.appendChild(input);
        item.appendChild(saveBtn);
        item.appendChild(cancelBtn);

        // Auto-focus the input on next render frame
        setTimeout(function () { input.focus(); input.select(); }, 0);
      } else {
        // -- Select button (native <button> — keyboard/ARIA for free) -----
        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "session-select-btn";
        selectBtn.dataset.sessionId = String(sid);
        if (sid === currentSessionId) {
          selectBtn.classList.add("active");
        }

        const label = document.createElement("span");
        label.className = "session-item-label";
        label.textContent = session.title;

        const time = document.createElement("span");
        time.className = "session-time";
        time.textContent = formatSessionTime(session.updated_at);

        selectBtn.appendChild(label);
        selectBtn.appendChild(time);
        selectBtn.addEventListener("click", function () {
          selectSession(sid);
        });

        // -- Rename button -----------------------------------------------
        const renameBtn = document.createElement("button");
        renameBtn.type = "button";
        renameBtn.className = "rename-session-btn";
        renameBtn.dataset.sessionId = String(sid);
        renameBtn.setAttribute("aria-label", "Rename session " + sid);
        renameBtn.title = "Rename";
        renameBtn.textContent = "✎";  // U+270E
        renameBtn.addEventListener("click", function (event) {
          event.stopPropagation();
          startRename(sid);
        });

        // -- Delete button -----------------------------------------------
        const delBtn = document.createElement("button");
        delBtn.className = "delete-session-btn";
        delBtn.dataset.sessionId = String(sid);
        delBtn.type = "button";
        delBtn.setAttribute("aria-label", "Delete session " + sid);
        delBtn.title = "Delete";
        delBtn.textContent = "×";
        delBtn.addEventListener("click", function (event) {
          event.stopPropagation();
          handleDeleteSession(sid, event);
        });

        item.appendChild(selectBtn);
        item.appendChild(renameBtn);
        item.appendChild(delBtn);
      }

      sessionListEl.appendChild(item);
    }
  }

  /* ---- Session operations ------------------------------------------ */

  /**
   * Refresh the full session list.
   *
   * Returns true on success, false on failure.  On failure the
   * existing sessions array, current selection and rendered messages
   * are kept; a persistent stale warning is shown.  On success the
   * server list is authoritative: temporary send blocks for sessions
   * it contains are cleared (their real llm_profile_status now
   * governs writability) and blocks for sessions no longer present
   * are dropped.
   */
  async function refreshSessions(preserveSelection) {
    if (preserveSelection === undefined) preserveSelection = true;

    let data;
    try {
      data = await fetchSessions();
    } catch (err) {
      sessionsLoadState = "error";
      sessionsLoadError = err.message;
      renderSessionListOrError();
      showStatus(err.message, true);
      return false;
    }

    // Validate BEFORE touching any local state: a malformed response
    // must never replace the cached list, the current selection, or
    // any temporary send block.
    if (!isValidSessionList(data)) {
      sessionsLoadState = "error";
      sessionsLoadError = "The server returned an invalid response.";
      renderSessionListOrError();
      showStatus(sessionsLoadError, true);
      return false;
    }

    sessions = data;
    sessionsLoadState = "ready";
    sessionsEverLoaded = true;
    sessionsLoadError = null;

    // Server list is authoritative for temporary blocks and other
    // per-session records.
    const presentIds = {};
    for (const s of sessions) {
      presentIds[s.id] = true;
      delete sessionSendBlocks[s.id];
      delete interactionModeStatusBySession[s.id];
    }
    for (const sid of Object.keys(sessionSendBlocks)) {
      if (!presentIds[sid]) delete sessionSendBlocks[sid];
    }
    for (const sid of Object.keys(sessionSwitchUncertain)) {
      if (!presentIds[sid]) delete sessionSwitchUncertain[sid];
    }
    for (const sid of Object.keys(sessionHasMessages)) {
      if (!presentIds[sid]) delete sessionHasMessages[sid];
    }
    for (const sid of Object.keys(profileSwitchStatusBySession)) {
      if (!presentIds[sid]) delete profileSwitchStatusBySession[sid];
    }
    for (const sid of Object.keys(interactionModeStatusBySession)) {
      if (!presentIds[sid]) delete interactionModeStatusBySession[sid];
    }
    for (const sid of Object.keys(interactionModeUncertainBySession)) {
      if (!presentIds[sid]) delete interactionModeUncertainBySession[sid];
    }
    for (const sid of Object.keys(reviewStateBySession)) {
      if (!presentIds[sid]) clearReviewSessionState(Number(sid));
    }

    // Decide the final selection BEFORE rendering the list so the
    // highlight, title, badge and message area all point at the same
    // session.
    const previousSelectionId = currentSessionId;
    const next = resolveNextSelectionId(
      sessions,
      currentSessionId,
      preserveSelection,
    );
    const selectionChanged = next.changed;
    currentSessionId = next.selectionId;

    // The model controls must follow the selection: a changed
    // selection re-derives the draft from the NEW session's server
    // binding (an old session's un-applied draft is never inherited),
    // while an unchanged selection keeps any un-applied draft — an
    // ordinary refresh never overwrites a user choice and never
    // rebuilds the dropdown.
    applySelectionDraftPlan(planProfileDraftForSelection({
      sessions: sessions,
      previousSessionId: previousSelectionId,
      nextSessionId: currentSessionId,
      previousDraftId: currentProfileDraftId,
      registry: registry(),
    }));
    setInteractionModeDraftForSession(findSessionInList(currentSessionId));
    if (selectionChanged) renderInteractionModeBar();

    // Only the null-selection case bumps the request guard manually:
    // it must invalidate an in-flight load without starting a new one.
    // A real switch delegates the bump to loadMessages().
    if (next.selectionId === null) {
      sessionLoadRequestId++;
    }

    renderSessionListOrError();
    syncCurrentSessionUI();

    if (next.selectionId === null) {
      renderInteractionModeBar();
      renderWelcome();
      return true;
    }

    if (selectionChanged) {
      loadMessages(next.selectionId);
      loadHistoryReviewSession(next.selectionId);
    }

    return true;
  }

  /**
   * Precisely refresh one session after a 409/503 response.
   *
   * Clears the temporary block only when the GET succeeded, the
   * response is an object with the expected id, and it was written
   * back into the local sessions array.  On 404 the session is
   * removed locally.  Any other failure keeps the block, the draft
   * and the read-only notice.  Never touches sessionsLoadState.
   */
  async function refreshOneSessionCompatibility(sessionId) {
    // Side-effect-free structured fetch first; the cache is only ever
    // replaced when the session still exists locally — a deleted
    // session is never re-inserted (no ghost sessions).
    const result = await fetchAndValidateOneSession(sessionId);
    if (result.status === "not_found") {
      removeSessionLocally(sessionId);
      return false;
    }
    if (result.status !== "ok") {
      // failed / invalid_response — keep the block and the cached
      // session, never overwrite anything.
      return false;
    }

    const cachePlan = planSessionCacheUpdate({
      sessions: sessions,
      requestedSessionId: sessionId,
      fresh: result.session,
    });
    if (cachePlan.kind !== "replace") {
      return false;
    }
    sessions = cachePlan.sessions;

    delete sessionSendBlocks[sessionId];
    renderSessionListOrError();
    syncCurrentSessionUI();
    return true;
  }

  /**
   * Remove a session from the local list and handle UI transitions
   * if it was the currently-selected session.
   */
  function removeSessionLocally(sessionId) {
    clearReviewSessionState(sessionId);
    sessions = sessions.filter(function (s) { return s.id !== sessionId; });
    delete sessionLastMessageId[sessionId];
    delete sessionSendBlocks[sessionId];
    delete sessionSwitchUncertain[sessionId];
    delete sessionHasMessages[sessionId];
    delete profileSwitchStatusBySession[sessionId];
    delete interactionModeStatusBySession[sessionId];
    delete interactionModeUncertainBySession[sessionId];

    if (sessionId === currentSessionId) {
      sessionLoadRequestId++;
      clearMessages();
      clearStatus();

      // The model controls follow the new selection — the deleted
      // session's draft is never inherited; with no session left the
      // bar hides and the status text clears.  Deleting a NON-current
      // session never reaches this branch, so its draft and focus
      // stay untouched.
      const previousSelectionId = currentSessionId;
      currentSessionId = sessions.length > 0 ? sessions[0].id : null;
      applySelectionDraftPlan(planProfileDraftForSelection({
        sessions: sessions,
        previousSessionId: previousSelectionId,
        nextSessionId: currentSessionId,
        previousDraftId: currentProfileDraftId,
        registry: registry(),
      }));
      setInteractionModeDraftForSession(findSessionInList(currentSessionId));
      renderInteractionModeBar();
      syncCurrentSessionUI();

      if (currentSessionId !== null) {
        loadMessages(currentSessionId);
        loadHistoryReviewSession(currentSessionId);
      } else {
        renderWelcome();
      }
    }

    renderSessionListOrError();
  }

  async function loadMessages(sessionId) {
    sessionLoadRequestId++;
    const requestId = sessionLoadRequestId;

    clearMessages();
    showStatus("Loading...", false);

    let data;
    try {
      data = await fetchMessages(sessionId);
    } catch (err) {
      // On 404 the session no longer exists — remove it locally
      if (err instanceof ApiError && err.status === 404) {
        if (requestId === sessionLoadRequestId) {
          removeSessionLocally(sessionId);
        }
        return;
      }
      // Other errors — show if still current
      if (requestId === sessionLoadRequestId &&
          currentSessionId === sessionId) {
        renderEmptyChat();
        showStatus(err.message, true);
      }
      return;
    }

    // Discard stale responses
    if (requestId !== sessionLoadRequestId) return;
    if (currentSessionId !== sessionId) return;

    // A malformed list must never reach renderMessages or touch
    // sessionLastMessageId.
    if (!isValidMessageList(data, sessionId)) {
      renderEmptyChat();
      showStatus("The server returned an invalid response.", true);
      return;
    }

    setHistoryState(sessionId, data.length > 0);
    if (data.length === 0) {
      renderEmptyChat();
    } else {
      renderMessages(data);
    }
    clearStatus();
  }

  /**
   * Switch to a different session.  Allowed during send — the
   * sendingSessionId guard in sendMessage prevents the old response
   * from rendering into the new session.
   *
   * Focus rule: on desktop the focus stays on the clicked session
   * button.  On mobile (sidebar closes) focus moves to the input when
   * the session is writable, or to the compatibility notice when it
   * is read-only.
   */
  function selectSession(sessionId) {
    if (sessionId === currentSessionId) return;

    currentSessionId = sessionId;
    // Recompute the draft from the server binding of the newly visible
    // session; never carry over another session's un-applied draft.
    const session = findSessionInList(sessionId);
    currentProfileDraftId = session !== null
      ? resolveSessionProfileDraft(session, registry())
      : null;
    setInteractionModeDraftForSession(session);
    renderSessionListOrError();
    syncCurrentSessionUI();
    renderCurrentProfileBar();
    renderCurrentProfileStatus();
    renderInteractionModeBar();
    loadMessages(sessionId);
    loadHistoryReviewSession(sessionId);

    if (isMobile()) {
      closeSidebarOnMobile();
      const session = currentSession();
      const writable = isSessionWritable(
        session,
        sessionSendBlocks[sessionId] || null,
        sessionSwitchUncertain[sessionId],
      );
      if (session !== null && writable) {
        inputEl.focus();
      } else if (session !== null) {
        sessionCompatibilityNoticeEl.focus();
      }
    } else {
      // The list was rebuilt above — restore focus to the freshly
      // created button for this session (user-initiated switches
      // only; programmatic flows never call selectSession).
      const btn = findSessionButton(
        sessionListEl.querySelectorAll(".session-select-btn"),
        sessionId,
      );
      if (btn !== null && !btn.disabled) {
        btn.focus();
      }
    }
  }

  /** Handle a failed create-session request. */
  async function handleCreateSessionFailure(err) {
    if (err instanceof ApiError && err.status === 422) {
      // The profile no longer exists — no fallback, no local session.
      showStatus(err.message, true);
      await reloadProfiles();
      return;
    }
    showStatus(err.message, true);
  }

  /**
   * Reload the profile list (after a 422, or retry).  Blocks creation
   * while loading; on failure keeps the persistent selector error and
   * disables creation.
   */
  async function reloadProfiles({ allowDefaultFallback = true } = {}) {
    profilesLoadState = "loading";
    profilesLoadError = null;
    updateControlStates();

    try {
      const data = await fetchProfiles();
      profiles = data;
      profilesLoadState = "ready";
      profilesLoadError = null;
    } catch (err) {
      profiles = [];
      profilesLoadState = "error";
      profilesLoadError = err && err.message ? err.message : "Failed to load models.";
      // On failure keep BOTH selectors' current values; the controls
      // are disabled via profilesLoadState === "error".
      renderProfileSelectorContent();
      renderCurrentProfileBar();
      renderCurrentProfileStatus();
      syncCurrentSessionUI();
      return;
    }

    // Read the truly visible state AFTER the await — never capture
    // drafts before the request.
    const visibleSessionIdAtApply = currentSessionId;
    const visibleDraftIdAtApply = currentProfileDraftId;
    const newRegistry = registry();

    if (allowDefaultFallback) {
      selectedProfileId = resolveSelectedProfileId(
        newRegistry, selectedProfileId,
      );
    } else {
      selectedProfileId = preserveSelectedProfileWithoutFallback(
        newRegistry, selectedProfileId,
      );
    }

    const draftStillValid =
      typeof visibleDraftIdAtApply === "string" &&
      captureRequestedProfile(newRegistry, visibleDraftIdAtApply) !== null;
    if (!draftStillValid) {
      const visibleAtApply = findSessionInList(visibleSessionIdAtApply);
      currentProfileDraftId = visibleAtApply !== null
        ? resolveSessionProfileDraft(visibleAtApply, newRegistry)
        : null;
    }
    // draftStillValid → keep the visible draft untouched (e.g. an
    // un-applied choice made on another session while reloading).

    renderProfileSelectorContent();
    renderCurrentProfileBar();
    renderCurrentProfileStatus();
    syncCurrentSessionUI();
  }

  /** Create a new session (called by "New Chat" button). */
  async function newChat() {
    if (isCreatingSession || isSending || isInitializing ||
        isProfileSwitching || isDeletingSession ||
        isInteractionModeSwitching) return;

    // Capture the profile id at request start so async changes cannot
    // make the UI and the request disagree.
    const profileId = selectedProfileId;
    if (profileId === null) {
      showStatus("No model available. New chats are disabled.", true);
      return;
    }

    let created = false;
    let createdReadOnly = false;

    isCreatingSession = true;
    updateControlStates();

    try {
      showStatus("Creating session...", false);
      const session = await createSessionRequest(profileId);

      // Validate BEFORE modifying local state — an invalid response
      // must never create a local ghost session.
      if (!isValidSessionResponse(session)) {
        showStatus("The server returned an invalid response.", true);
        return;
      }
      for (let i = 0; i < sessions.length; i++) {
        if (sessions[i].id === session.id) {
          showStatus("The server returned an invalid response.", true);
          return;
        }
      }

      clearStatus();

      // The server SessionResponse is the source of truth for model
      // fields — never the selector's current guess.
      sessions.unshift(session);
      currentSessionId = session.id;
      sessionLastMessageId[session.id] = 0;
      setHistoryState(session.id, false);        // brand-new session
      currentProfileDraftId = session.llm_profile_id;
      setInteractionModeDraftForSession(session);
      sessionLoadRequestId++;
      renderSessionListOrError();
      syncCurrentSessionUI();
      renderCurrentProfileBar();
      renderInteractionModeBar();
      renderHistoryReviewPanel();
      renderEmptyChat();
      clearStatus();
      inputEl.value = "";
      inputEl.style.height = "";
      created = true;
      createdReadOnly = session.llm_profile_status !== "ready";
      closeSidebarOnMobile();
    } catch (err) {
      await handleCreateSessionFailure(err);
    } finally {
      isCreatingSession = false;
      updateControlStates();
      // Focus only after the controls are re-enabled, and only on
      // focusable, visible elements.
      if (created) {
        if (!inputEl.disabled) {
          inputEl.focus();
        } else if (createdReadOnly) {
          if (isMobile() && !sessionCompatibilityNoticeEl.hidden) {
            // Mobile: the sidebar is closed, so focus moves to the
            // visible compatibility notice.
            sessionCompatibilityNoticeEl.focus();
          } else if (!newChatBtn.disabled) {
            // Desktop: explicitly return focus to the re-enabled
            // New Chat button (never rely on natural focus, since it
            // was disabled during the async request).
            newChatBtn.focus();
          }
        }
      }
    }
  }

  /** Ensure a session exists, creating one if necessary. */
  async function ensureSession() {
    if (currentSessionId !== null) return true;

    if (isCreatingSession || isSending || isInitializing ||
        isProfileSwitching || isDeletingSession ||
        isInteractionModeSwitching) return false;

    const profileId = selectedProfileId;
    if (profileId === null) {
      showStatus("No model available. New chats are disabled.", true);
      return false;
    }

    isCreatingSession = true;
    updateControlStates();

    try {
      showStatus("Creating session...", false);
      const session = await createSessionRequest(profileId);

      // Validate BEFORE modifying local state.
      if (!isValidSessionResponse(session)) {
        showStatus("The server returned an invalid response.", true);
        return false;
      }
      for (let i = 0; i < sessions.length; i++) {
        if (sessions[i].id === session.id) {
          showStatus("The server returned an invalid response.", true);
          return false;
        }
      }

      sessions.unshift(session);
      currentSessionId = session.id;
      sessionLastMessageId[session.id] = 0;
      setHistoryState(session.id, false);
      currentProfileDraftId = session.llm_profile_id;
      setInteractionModeDraftForSession(session);
      sessionLoadRequestId++;
      renderSessionListOrError();
      syncCurrentSessionUI();
      renderCurrentProfileBar();
      renderInteractionModeBar();
      renderHistoryReviewPanel();
      clearStatus();
      return true;
    } catch (err) {
      await handleCreateSessionFailure(err);
      return false;
    } finally {
      isCreatingSession = false;
      updateControlStates();
    }
  }

  async function handleDeleteSession(sessionId, event) {
    event.stopPropagation();
    if (reviewTargetBusy(sessionId)) return;
    if (isSending || isProfileSwitching || isDeletingSession ||
        isCreatingSession || isInteractionModeSwitching) return;

    if (!confirm("Delete this conversation?")) return;

    // Enter the delete busy state after confirmation and BEFORE the
    // first await — double clicks can only produce one DELETE.
    isDeletingSession = true;
    deletingSessionId = sessionId;
    updateControlStates();

    try {
      try {
        await deleteSessionRequest(sessionId);
      } catch (err) {
        // 404 → session already gone on server, remove locally
        if (err instanceof ApiError && err.status === 404) {
          removeSessionLocally(sessionId);
          return;
        }
        // Other errors — keep list, show error
        showStatus(err.message, true);
        return;
      }

      removeSessionLocally(sessionId);
    } finally {
      isDeletingSession = false;
      deletingSessionId = null;
      updateControlStates();
    }
  }

  /* ---- Rename session ---------------------------------------------- */

  function startRename(sessionId) {
    if (isSending || isRenaming || isInitializing ||
        isProfileSwitching || isDeletingSession ||
        isInteractionModeSwitching) return;
    if (reviewTargetBusy(sessionId)) return;

    isRenaming = true;
    renamingSessionId = sessionId;
    updateControlStates();
    renderSessionListOrError();
  }

  async function saveRename(sessionId, renameInputEl) {
    if (isRenameSaving) return;  // prevent double-submit
    if (isProfileSwitching || isDeletingSession ||
        isInteractionModeSwitching) return;
    if (reviewTargetBusy(sessionId)) return;

    var rawTitle = renameInputEl.value;
    if (!rawTitle.trim()) {
      showStatus("Title must not be blank.", true);
      renameInputEl.focus();
      return;
    }

    isRenameSaving = true;
    updateControlStates();  // disable input, Save, Cancel

    try {
      var updated = await renameSessionRequest(sessionId, rawTitle);

      // Validate BEFORE overwriting anything: a malformed response or
      // an id mismatch must not replace a local session, exit editing
      // or drop the draft.
      if (!isValidSessionResponse(updated) || updated.id !== sessionId) {
        isRenameSaving = false;
        updateControlStates();  // re-enable editing controls
        showStatus("The server returned an invalid response.", true);
        // Stay in Editing state — rename input is still connected.
        renameInputEl.focus();
        return;
      }

      // Use server-normalised title from the full SessionResponse
      for (var i = 0; i < sessions.length; i++) {
        if (sessions[i].id === sessionId) {
          sessions[i] = updated;
          break;
        }
      }
      // Move renamed session to top (updated_at has changed)
      sessions = [updated].concat(
        sessions.filter(function (s) { return s.id !== sessionId; })
      );
      clearStatus();
      isRenaming = false;
      renamingSessionId = null;
      isRenameSaving = false;
      updateControlStates();
      renderSessionListOrError();
      syncCurrentSessionUI();
      inputEl.focus();  // explicitly focus chat message input
    } catch (err) {
      // sessions was never modified — no rollback needed
      isRenameSaving = false;
      updateControlStates();  // re-enable input, Save, Cancel
      showStatus(err.message, true);
      // Stay in Editing state — do NOT call renderSessionListOrError()
      renameInputEl.focus();
    }
  }

  function cancelRename() {
    // No optimistic update was made, no restore needed
    isRenaming = false;
    renamingSessionId = null;
    isRenameSaving = false;
    updateControlStates();
    renderSessionListOrError();
    inputEl.focus();  // explicitly focus chat message input
  }

  /* ---- Send message ------------------------------------------------ */

  async function sendMessage(text) {
    if (isSending || isInitializing || isCreatingSession ||
        isProfileSwitching || isDeletingSession ||
        isInteractionModeSwitching) return;
    if (currentSessionId !== null && reviewTargetBusy(currentSessionId)) return;

    // Auto-create session on first send
    if (currentSessionId === null) {
      if (!registryUsable()) return;  // defensive — UI already blocked
      const ok = await ensureSession();
      if (!ok) return;  // create failed or already in progress
    } else {
      // Defensive writability check — the UI is already disabled for
      // read-only sessions.
      if (!currentSessionWritable()) return;
    }

    const sendingSessionId = currentSessionId;
    const lastMessageIdBeforeSend = sessionLastMessageId[sendingSessionId] || 0;

    setSendingState(true);
    showStatus("Generating...", false);

    try {
      const data = await sendSessionMessageRequest(sendingSessionId, text);

      // Validate BEFORE touching any UI or state.  A malformed 2xx
      // body means the message may already be saved on the server —
      // route it through the same uncertain-send inspection as a
      // network error, never through the plain success path.
      if (!isValidSendMessageResponse(data, sendingSessionId)) {
        await handleUncertainSendOutcome(
          sendingSessionId, text, lastMessageIdBeforeSend,
        );
        return;
      }

      // --- Success --------------------------------------------------
      clearStatus();
      inputEl.value = "";
      inputEl.style.height = "";

      if (currentSessionId === sendingSessionId) {
        appendMessage(data.user_message.role, data.user_message.content);
        appendMessage(data.assistant_message.role, data.assistant_message.content);
        scrollMessagesToBottom();
      }

      sessionLastMessageId[sendingSessionId] = data.assistant_message.id;
      setHistoryState(sendingSessionId, true);   // a user message is saved

      // Refresh session list (sending session moves to top).  On
      // failure the rendered messages stay and a stale warning shows.
      await refreshSessions(true);
    } catch (err) {
      await handleSendFailure(err, sendingSessionId, text, lastMessageIdBeforeSend);
      return;
    } finally {
      setSendingState(false);
    }
  }

  /**
   * Re-fetch the messages for a session and analyse what actually
   * happened to an uncertain send.
   *
   * Pure analysis: no DOM, no status text, no input clearing, no
   * sessionLastMessageId writes, no refreshSessions call.
   *
   * @returns {{status: string}} One of:
   *   {status: "succeeded", messages, lastMessageId}
   *   {status: "user_saved", messages, userMessageId}
   *   {status: "unknown"}
   */
  async function inspectUncertainSend(
    sendingSessionId,
    originalText,
    lastMessageIdBeforeSend,
  ) {
    let msgs;
    try {
      msgs = await fetchMessages(sendingSessionId);
    } catch (_) {
      return { status: "unknown" };
    }

    // A malformed list must never reach findSentMessages — fail closed.
    if (!isValidMessageList(msgs, sendingSessionId)) {
      return { status: "unknown" };
    }

    const result = findSentMessages(
      msgs, lastMessageIdBeforeSend, originalText,
    );
    if (result.userIdx < 0) {
      return { status: "unknown" };
    }

    if (result.hasAssistant) {
      // The list is validated and strictly increasing, so the last
      // element carries the highest known message id.
      return {
        status: "succeeded",
        messages: msgs,
        lastMessageId: msgs[msgs.length - 1].id,
      };
    }

    return {
      status: "user_saved",
      messages: msgs,
      userMessageId: result.userMessageId,
    };
  }

  /**
   * Apply the outcome of an uncertain send to the UI.  Shared by
   * network errors (status 0) and malformed 2xx send responses — one
   * recovery implementation, no duplicated logic.
   *
   * - succeeded  → render history on the sending session, clear the
   *                (now saved) draft, update the last id, refresh.
   * - user_saved → clear the saved draft, update the user id, show
   *                the saved-but-unanswered state.
   * - unknown    → keep the draft and show an uncertain warning;
   *                never assume success or failure.
   *
   * Rendering still guards on ``currentSessionId === sendingSessionId``
   * because switching sessions is allowed during a send.
   */
  async function handleUncertainSendOutcome(
    sendingSessionId,
    originalText,
    lastMessageIdBeforeSend,
  ) {
    const outcome = await inspectUncertainSend(
      sendingSessionId, originalText, lastMessageIdBeforeSend,
    );

    if (outcome.status === "succeeded") {
      setHistoryState(sendingSessionId, true);
      if (currentSessionId === sendingSessionId) {
        renderMessages(outcome.messages);
        scrollMessagesToBottom();
        clearStatus();
      }
      inputEl.value = "";
      inputEl.style.height = "";
      if (outcome.lastMessageId > 0) {
        sessionLastMessageId[sendingSessionId] = outcome.lastMessageId;
      }
      await refreshSessions(true);
      return;
    }

    if (outcome.status === "user_saved") {
      setHistoryState(sendingSessionId, true);
      sessionLastMessageId[sendingSessionId] = outcome.userMessageId;
      inputEl.value = "";
      inputEl.style.height = "";
      if (currentSessionId === sendingSessionId) {
        showStatus(
          "Your message was saved, but the assistant could not respond.",
          true,
        );
        // outcome.messages was already validated by
        // inspectUncertainSend — no second network request.
        if (outcome.messages.length === 0) {
          renderEmptyChat();
        } else {
          renderMessages(outcome.messages);
        }
      } else {
        showStatus(
          "Message saved in session #" + sendingSessionId +
          ", but the assistant could not respond.",
          true,
        );
      }
      return;
    }

    // unknown — keep the draft; never append, never update the last
    // id, never claim the request failed.
    showStatus(
      "Request status is uncertain. Review the conversation before resending.",
      true,
    );
  }

  /**
   * Handle a failed send — strategy depends on the error type.
   *
   * 409 / 503 → model compatibility conflict.  Handled as an early
   *              branch: a temporary block is set immediately, the
   *              session becomes read-only, the draft is kept, and
   *              only that session is refreshed precisely.  Never
   *              routed through network-recovery logic.
   * 502 / 504  → user message was saved (Phase 1 committed).
   *               Clear input, re-sync messages from DB.
   * 404 / 422  → user message was NOT saved.
   *               Keep input, show error.
   * Network (0) → uncertain.  Inspect the message log and decide by
   *               the message-ID boundary.
   *
   * *lastMessageIdBeforeSend* is the highest known message id in the
   * session right before the API call.  After a network error we only
   * look at messages with id > that boundary — this handles the case
   * where the user sends the same text twice consecutively.
   */
  async function handleSendFailure(
    err,
    sendingSessionId,
    originalText,
    lastMessageIdBeforeSend,
  ) {
    const errStatus = (err instanceof ApiError) ? err.status : 0;

    // --- 409/503: model compatibility — early branch --------------------
    if (errStatus === 409 || errStatus === 503) {
      // Never guess the exact 409 status from the English detail; use
      // a generic conflict block.  503 maps to profile_unavailable.
      sessionSendBlocks[sendingSessionId] =
        errStatus === 503 ? "profile_unavailable" : "conflict";

      // Immediately make the session read-only — do not wait for the
      // refresh to succeed.
      syncCurrentSessionUI();
      showStatus(err.message, true);
      // Draft stays in the textarea.

      // Precisely refresh only the affected session.  If the user has
      // switched to another session, the block is still recorded for
      // sendingSessionId and the notice is not rendered elsewhere.
      await refreshOneSessionCompatibility(sendingSessionId);
      return;
    }

    let isSaved = false;

    // --- Determine save status ----------------------------------------
    if (errStatus === 502 || errStatus === 504) {
      isSaved = true;
    } else if (errStatus === 0) {
      // Network error — inspect the log by ID boundary.  The shared
      // recovery handles succeeded / user-saved / unknown.
      await handleUncertainSendOutcome(
        sendingSessionId, originalText, lastMessageIdBeforeSend,
      );
      return;
    }
    // 404, 422, other HTTP → isSaved stays false

    // --- Act based on save status -------------------------------------
    if (currentSessionId === sendingSessionId) {
      if (isSaved) {
        setHistoryState(sendingSessionId, true);
        showStatus(
          "Your message was saved, but the assistant could not respond.",
          true,
        );
        inputEl.value = "";
        inputEl.style.height = "";

        // Reload history to show the saved user message.  Only a
        // validated list may reach renderMessages; an invalid body
        // must not replace the current message area and must not be
        // mistaken for a plain send failure.
        try {
          const history = await fetchMessages(sendingSessionId);
          if (currentSessionId === sendingSessionId) {
            if (isValidMessageList(history, sendingSessionId)) {
              if (history.length === 0) {
                renderEmptyChat();
              } else {
                renderMessages(history);
              }
            } else {
              showStatus(
                "Your message was saved, but the conversation history " +
                "could not be refreshed.",
                true,
              );
            }
          }
        } catch (_) {
          /* best-effort sync */
        }
      } else {
        // 404, 422, or other non-saved errors
        if (errStatus === 404) {
          await refreshSessions(true);
        }
        showStatus(err.message || "Something went wrong.", true);
        // Keep input for retry
      }
    } else {
      // User switched away — just show a brief warning
      showStatus(
        "Failed to send message in session #" + sendingSessionId + ".",
        true,
      );
    }
  }

  /* ---- Current-session interaction-mode switcher ------------------- */

  function setInteractionModeStatus(sessionId, text, isError) {
    if (isValidSessionIdKey(sessionId) === false) return;
    interactionModeStatusBySession[sessionId] = {
      text: typeof text === "string" ? text : "",
      isError: isError === true,
    };
    renderCurrentInteractionModeStatus();
  }

  function clearInteractionModeStatus(sessionId) {
    if (isValidSessionIdKey(sessionId) === false) return;
    delete interactionModeStatusBySession[sessionId];
    renderCurrentInteractionModeStatus();
  }

  function renderCurrentInteractionModeStatus() {
    let text = "";
    let isError = false;
    if (currentSessionId !== null &&
        interactionModeStatusBySession[currentSessionId] !== undefined) {
      text = interactionModeStatusBySession[currentSessionId].text;
      isError = interactionModeStatusBySession[currentSessionId].isError;
    } else if (currentSessionId !== null &&
               hasInteractionModeUncertain(
                 interactionModeUncertainBySession, currentSessionId,
               )) {
      text = interactionModeUncertainText();
      isError = true;
    } else if (currentSessionId !== null) {
      const authoritativeMode = currentSessionInteractionMode();
      if (authoritativeMode !== null &&
          currentInteractionModeDraft !== authoritativeMode) {
        text = "Click Apply to switch to " +
          interactionModeLabel(currentInteractionModeDraft) + ".";
      } else if (authoritativeMode !== null) {
        text = interactionModeHintText(authoritativeMode);
      }
    }
    interactionModeStatusEl.textContent = text;
    interactionModeStatusEl.className =
      "interaction-mode-status" + (isError ? " error" : "");
  }

  function renderInteractionModeBar() {
    if (currentSessionId === null) {
      interactionModeBarEl.hidden = true;
      renderCurrentInteractionModeStatus();
      return;
    }
    interactionModeBarEl.hidden = false;
    interactionModeSelectEl.replaceChildren();
    for (let i = 0; i < VALID_INTERACTION_MODES.length; i++) {
      const mode = VALID_INTERACTION_MODES[i];
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = interactionModeLabel(mode);
      interactionModeSelectEl.appendChild(option);
    }
    if (currentInteractionModeDraft !== null) {
      interactionModeSelectEl.value = currentInteractionModeDraft;
    }
    renderCurrentInteractionModeStatus();
  }

  function updateInteractionModeSessionCache(fresh) {
    const cachePlan = planSessionCacheUpdate({
      sessions: sessions,
      requestedSessionId: fresh.id,
      fresh: fresh,
    });
    if (cachePlan.kind === "replace") {
      sessions = cachePlan.sessions;
      renderSessionListOrError();
    }
  }

  function applyInteractionModeOutcome(outcome, operation) {
    const targetSessionId = operation.targetSessionId;
    handleInteractionModeReviewOutcome(targetSessionId, outcome);

    if (outcome.status === "not_found") {
      removeSessionLocally(targetSessionId);
      return;
    }

    if (outcome.session) {
      updateInteractionModeSessionCache(outcome.session);
    }

    const visible = currentSessionId === targetSessionId;

    if (outcome.status === "switched") {
      delete interactionModeUncertainBySession[targetSessionId];
      clearInteractionModeStatus(targetSessionId);
      if (visible) {
        currentInteractionModeDraft = outcome.session !== null &&
          outcome.session !== undefined
          ? outcome.session.interaction_mode
          : operation.requestedMode;
        renderInteractionModeBar();
        syncCurrentSessionUI();
      } else {
        renderCurrentInteractionModeStatus();
      }
      return;
    }

    if (outcome.status === "not_changed") {
      delete interactionModeUncertainBySession[targetSessionId];
      if (visible) {
        currentInteractionModeDraft = operation.requestedMode;
        setInteractionModeStatus(
          targetSessionId,
          "Server still uses " + interactionModeLabel(operation.originalMode) +
            ". Click Apply to try again.",
          false,
        );
        renderInteractionModeBar();
        syncCurrentSessionUI();
      } else {
        clearInteractionModeStatus(targetSessionId);
      }
      return;
    }

    if (outcome.status === "uncertain") {
      delete interactionModeStatusBySession[targetSessionId];
      interactionModeUncertainBySession[targetSessionId] = {
        requestedMode: operation.requestedMode,
        originalMode: operation.originalMode,
      };
      if (visible) {
        currentInteractionModeDraft = operation.requestedMode;
        renderInteractionModeBar();
        syncCurrentSessionUI();
      }
      return;
    }

    // failed / invalid_request / defensive unknown outcome
    delete interactionModeUncertainBySession[targetSessionId];
    if (visible) {
      currentInteractionModeDraft = operation.requestedMode;
      setInteractionModeStatus(
        targetSessionId,
        outcome.message || "Interaction mode switch failed.",
        true,
      );
      renderInteractionModeBar();
      syncCurrentSessionUI();
    }
  }

  async function applyInteractionModeSwitch() {
    if (interactionModeController === null) {
      if (currentSessionId !== null) {
        setInteractionModeStatus(
          currentSessionId,
          interactionModeInitializationError ||
            "Interaction mode switching is unavailable.",
          true,
        );
      }
      return;
    }
    if (isInitializing || isSending || isCreatingSession || isRenaming ||
        isRenameSaving || isDeletingSession || isProfileSwitching ||
        isInteractionModeSwitching) return;
    if (currentSessionId === null) return;
    if (reviewTargetBusy(currentSessionId)) return;

    const targetSessionId = currentSessionId;
    const session = findSessionInList(targetSessionId);
    if (session === null) return;

    const uncertainRecord =
      interactionModeUncertainBySession[targetSessionId];
    let requestedMode;
    let originalMode;

    if (uncertainRecord !== undefined &&
        uncertainRecord !== null) {
      requestedMode = uncertainRecord.requestedMode;
      originalMode = uncertainRecord.originalMode;
    } else {
      requestedMode = currentInteractionModeDraft;
      originalMode = interactionModeAuthoritativeMode(session);
      if (originalMode === null ||
          requestedMode === originalMode) return;
    }

    if (isValidInteractionMode(requestedMode) === false ||
        isValidInteractionMode(originalMode) === false) {
      return;
    }

    const generation = ++interactionModeSwitchGeneration;
    isInteractionModeSwitching = true;
    updateControlStates();
    setInteractionModeStatus(
      targetSessionId,
      uncertainRecord !== undefined && uncertainRecord !== null
        ? "Checking server interaction mode..."
        : "Switching interaction mode...",
      false,
    );

    try {
      const operation = {
        targetSessionId: targetSessionId,
        requestedMode: requestedMode,
        originalMode: originalMode,
      };
      const outcome =
        uncertainRecord !== undefined && uncertainRecord !== null
          ? await interactionModeController.reconcile(operation)
          : await interactionModeController.apply(operation);

      if (generation !== interactionModeSwitchGeneration) return;
      applyInteractionModeOutcome(outcome, operation);
    } catch (_) {
      if (generation !== interactionModeSwitchGeneration) return;
      delete interactionModeStatusBySession[targetSessionId];
      interactionModeUncertainBySession[targetSessionId] = {
        requestedMode: requestedMode,
        originalMode: originalMode,
      };
      if (currentSessionId === targetSessionId) {
        renderInteractionModeBar();
        syncCurrentSessionUI();
      }
    } finally {
      if (generation === interactionModeSwitchGeneration) {
        isInteractionModeSwitching = false;
        updateControlStates();
      }
    }
  }

  /* ---- Current-session model switcher ------------------------------ */

  /** Render the options and draft of the current-session control bar. */
  function renderCurrentProfileBar() {
    const reg = registry();

    if (currentSessionId === null) {
      currentProfileBarEl.hidden = true;
      // Clear the status area too — the null-selection render must
      // leave no stale visible text or error class behind.  The
      // shared renderer only reads the per-session maps, so other
      // sessions' records stay untouched and the next selection
      // restores the right hint from them.
      renderCurrentProfileStatus();
      return;
    }
    currentProfileBarEl.hidden = false;

    currentProfileSelectEl.replaceChildren();
    if (reg.status === "valid") {
      for (const p of reg.profiles) {
        const option = document.createElement("option");
        option.value = p.id;
        option.textContent = p.label;
        currentProfileSelectEl.appendChild(option);
      }
      if (currentProfileDraftId !== null) {
        currentProfileSelectEl.value = currentProfileDraftId;
      } else {
        // Explicit placeholder — no silent fallback to the first
        // profile for a missing/unavailable binding.
        appendPlaceholderOption(currentProfileSelectEl);
        currentProfileSelectEl.value = "";
      }
    } else {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No models available";
      currentProfileSelectEl.appendChild(option);
      currentProfileSelectEl.value = "";
    }

    renderCurrentProfileStatus();
  }

  /** Whether Apply is available for the current session/draft.  An
   *  uncertain-switch record keeps Apply available for convergence
   *  even when the draft equals the cached binding. */
  function applyEnabled() {
    if (switchController === null ||
        profileSwitchInitializationError !== null) return false;
    if (isProfileSwitching || isDeletingSession || isSending ||
        isCreatingSession || isRenaming || isRenameSaving ||
        isInitializing || isInteractionModeSwitching) return false;
    if (registry().status !== "valid") return false;
    const session = currentSession();
    if (session === null) return false;
    if (hasInteractionModeUncertain(
      interactionModeUncertainBySession, session.id,
    )) return false;
    return canApplySessionProfileWithUncertain({
      session: session,
      registry: registry(),
      draftProfileId: currentProfileDraftId,
      uncertain: sessionSwitchUncertain[session.id] !== undefined,
    });
  }

  /** Real <dialog> adapter matching createRemoteHistoryConfirmer. */
  function createProfileSwitchDialogAdapter() {
    const dlg = profileSwitchDialogEl;
    const cancelBtn = document.getElementById("psd-cancel");
    const continueBtn = document.getElementById("psd-continue");
    const bodyEl = document.getElementById("psd-body");

    function on(target, event, cb) {
      function handler(e) {
        if (event === "cancel") e.preventDefault();
        cb();
      }
      target.addEventListener(event, handler);
      return function () { target.removeEventListener(event, handler); };
    }

    return {
      showModal: function () { dlg.showModal(); },
      close: function () { if (dlg.open) dlg.close(); },
      setMessage: function (text) { bodyEl.textContent = text; },
      focusInitial: function () { cancelBtn.focus(); },
      isConnected: function () { return dlg.isConnected; },
      onDialogCancel: function (cb) { return on(dlg, "cancel", cb); },
      onCancelClick: function (cb) { return on(cancelBtn, "click", cb); },
      onContinueClick: function (cb) { return on(continueBtn, "click", cb); },
    };
  }

  /** One-time, fail-closed controller/confirmer initialisation. */
  function initializeProfileSwitching() {
    try {
      const dialogAdapter = createProfileSwitchDialogAdapter();
      const historyConfirmer = createRemoteHistoryConfirmer(dialogAdapter);
      switchController = createSessionProfileSwitchController({
        patchSwitch: switchProfileRequest,
        fetchOneSession: fetchOneSessionRaw,
        confirmRemoteHistory: historyConfirmer.confirm,
        validateSessionResponse: isValidSessionResponse,
      });
      profileSwitchInitializationError = null;
      return true;
    } catch (_) {
      switchController = null;
      profileSwitchInitializationError = "模型切换功能暂不可用。";
      return false;
    }
  }

  /** One-time, fail-closed interaction-mode controller initialization. */
  function initializeInteractionModeSwitching() {
    try {
      interactionModeController = createInteractionModeSwitchController({
        patchSwitch: switchInteractionModeRequest,
        fetchSession: fetchOneSessionRaw,
        validateSession: isValidSessionResponse,
        validateTimestamp: isValidApiTimestamp,
      });
      interactionModeInitializationError = null;
      return true;
    } catch (_) {
      interactionModeController = null;
      interactionModeInitializationError =
        "Interaction mode switching is unavailable.";
      return false;
    }
  }

  /** Apply the current draft to the current session. */
  async function applyProfileSwitch() {
    if (switchController === null) {
      if (currentSessionId !== null) {
        setProfileSwitchStatus(
          currentSessionId,
          profileSwitchInitializationError || "模型切换功能暂不可用。",
          true,
        );
      }
      return;
    }
    if (isInitializing || isSending || isCreatingSession || isRenaming ||
        isRenameSaving || isDeletingSession || isProfileSwitching ||
        isInteractionModeSwitching) return;
    if (currentSessionId === null) return;
    if (hasInteractionModeUncertain(
      interactionModeUncertainBySession, currentSessionId,
    )) return;
    if (!applyEnabled()) return;
    if (reviewTargetBusy(currentSessionId)) return;

    const targetSessionId = currentSessionId;
    const requestedDraftId = currentProfileDraftId;
    const generation = ++profileSwitchGeneration;
    isProfileSwitching = true;
    updateControlStates();

    let focusIntent = null;
    try {
      let effectiveSession = findSessionInList(targetSessionId);
      if (effectiveSession === null) {
        setProfileSwitchStatus(targetSessionId,
          "The conversation no longer exists.", true);
        return;
      }

      // -- uncertain convergence: refresh the authoritative session
      //    BEFORE building any operation.
      if (sessionSwitchUncertain[targetSessionId] !== undefined) {
        const result = await fetchAndValidateOneSession(targetSessionId);
        if (result.status === "ok") {
          const cachePlan = planSessionCacheUpdate({
            sessions: sessions,
            requestedSessionId: targetSessionId,
            fresh: result.session,
          });
          if (cachePlan.kind !== "replace") {
            setProfileSwitchStatus(targetSessionId,
              "The conversation could not be refreshed.", true);
            return;
          }
          sessions = cachePlan.sessions;
          renderSessionListOrError();
          effectiveSession = findSessionInList(targetSessionId);
          if (effectiveSession === null) {
            setProfileSwitchStatus(targetSessionId,
              "The conversation no longer exists.", true);
            return;
          }
          const classified = classifyUncertainRefresh({
            targetSessionId: targetSessionId,
            uncertainRecord: sessionSwitchUncertain[targetSessionId],
            fresh: result.session,
          });
          if (classified.status === "invalid") {
            setProfileSwitchStatus(targetSessionId, UNCERTAIN_REAPPLY_TEXT, true);
            return;
          }
          if (classified.status === "confirmed_target") {
            delete sessionSendBlocks[targetSessionId];
            delete sessionSwitchUncertain[targetSessionId];
            clearProfileSwitchStatus(targetSessionId);
            if (effectiveSession.llm_profile_id === requestedDraftId) {
              if (currentSessionId === targetSessionId) {
                currentProfileDraftId = requestedDraftId;
                renderCurrentProfileBar();
                syncCurrentSessionUI();
              }
              return;
            }
            // Draft differs — continue with a fresh switch below.
          }
          // different_binding → keep uncertain, continue with a fresh
          // PATCH for the current draft (same lock on the server
          // serialises the new request after the old one).
        } else if (result.status === "not_found") {
          removeSessionLocally(targetSessionId);
          return;
        } else {
          setProfileSwitchStatus(targetSessionId, UNCERTAIN_REAPPLY_TEXT, true);
          return;
        }
      }

      const requestedProfile = captureRequestedProfile(
        registry(), requestedDraftId,
      );
      if (requestedProfile === null) {
        setProfileSwitchStatus(targetSessionId,
          "Choose an available model first.", true);
        focusIntent = "apply";
        return;
      }

      const operation = {
        generation: generation,
        targetSessionId: targetSessionId,
        originalProfileId: effectiveSession.llm_profile_id,
        originalModelSnapshot: effectiveSession.llm_model_snapshot,
        requestedProfile: requestedProfile,
        needsConfirmHint: needsRemoteHistoryConfirmation({
          session: effectiveSession,
          registry: registry(),
          targetProfileId: requestedDraftId,
          historyState: historyStateFor(targetSessionId),
        }),
      };

      let outcome;
      try {
        outcome = await switchController.apply(operation);
      } catch (_) {
        outcome = { status: "failed", message: "Unexpected error." };
      }

      const plan = planSwitchOutcomeEffects({
        outcome: outcome,
        operation: operation,
        targetSessionId: targetSessionId,
        currentSessionId: currentSessionId,
        hasTarget: findSessionInList(targetSessionId) !== null,
      });

      focusIntent = await executeSwitchEffects(plan, outcome, operation);
    } finally {
      isProfileSwitching = false;
      updateControlStates();
      applyFocusIntent(focusIntent, targetSessionId, generation);
    }
  }

  /** Execute the effect plan.  Target-scoped state is always applied;
   * visible effects require the target to still be the current
   * session after every await. */
  async function executeSwitchEffects(plan, outcome, operation) {
    const targetSessionId = operation.targetSessionId;

    if (plan.kind === "ignore") return null;
    if (plan.removeSession) { removeSessionLocally(targetSessionId); return null; }

    // 1) cache: only a successful replace may unlock protective state
    if (plan.updateCache) {
      const cachePlan = planSessionCacheUpdate({
        sessions: sessions,
        requestedSessionId: targetSessionId,
        fresh: plan.session,
      });
      if (cachePlan.kind !== "replace") {
        setProfileSwitchStatus(targetSessionId,
          "The conversation could not be updated safely.", true);
        return null;
      }
      sessions = cachePlan.sessions;
    }

    // 2) protective state (target-scoped)
    if (plan.clearBlock) delete sessionSendBlocks[targetSessionId];
    if (plan.clearUncertain) delete sessionSwitchUncertain[targetSessionId];
    if (plan.uncertainRecord !== null) {
      sessionSwitchUncertain[targetSessionId] = plan.uncertainRecord;
    }

    // 3) ordinary target status BEFORE any await / visibility check
    applyTargetProfileSwitchStatus(targetSessionId, outcome, plan);

    // 4) profiles reload
    if (plan.reloadProfiles) {
      await reloadProfiles({ allowDefaultFallback: false });
    }

    // 5) re-read visibility after awaits
    renderSessionListOrError();
    if (currentSessionId !== targetSessionId) {
      renderCurrentProfileStatus();
      return null;
    }

    // 6) visible effects — target is still current
    switch (outcome.status) {
      case "switched":
        currentProfileDraftId = plan.session.llm_profile_id;
        break;
      case "not_changed":
        currentProfileDraftId = plan.session.llm_profile_id;
        break;
      case "cancelled":
      case "failed":
      case "validation_error": {
        const current = findSessionInList(targetSessionId);
        currentProfileDraftId = current !== null
          ? resolveSessionProfileDraft(current, registry())
          : null;
        break;
      }
      case "uncertain":
        // keep the user's draft; the record holds the persistent hint
        break;
      default:
        break;
    }

    renderCurrentProfileBar();
    renderCurrentProfileStatus();
    if (plan.syncVisibleUI) syncCurrentSessionUI();
    return plan.focus;
  }

  /** Focus application — triple-checked, never steals focus. */
  function applyFocusIntent(intent, targetSessionId, generation) {
    if (intent === null || intent === undefined) return;
    if (generation !== profileSwitchGeneration) return;
    if (currentSessionId !== targetSessionId) return;
    const el = intent === "input" ? inputEl : applyProfileBtn;
    if (el && el.isConnected && !el.disabled && !el.hidden) el.focus();
  }

  /* ---- Mobile sidebar ---------------------------------------------- */

  function isMobile() {
    return window.innerWidth <= 767;
  }

  function toggleSidebar() {
    sidebarEl.classList.toggle("collapsed");
    const expanded = !sidebarEl.classList.contains("collapsed");
    sidebarToggleEl.setAttribute("aria-expanded", String(expanded));
  }

  function closeSidebarOnMobile() {
    if (isMobile()) {
      sidebarEl.classList.add("collapsed");
      sidebarToggleEl.setAttribute("aria-expanded", "false");
    }
  }

  /* ---- Event bindings ---------------------------------------------- */

  // Send button
  sendBtn.addEventListener("click", function () {
    if (isSending || isInitializing || isCreatingSession) return;
    const text = inputEl.value.trim();
    if (!text) return;
    sendMessage(text);
  });

  // Textarea: Enter to send, Shift+Enter for newline
  inputEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendBtn.click();
    }
  });

  // Auto-resize textarea (clamped at CSS max-height)
  inputEl.addEventListener("input", function () {
    inputEl.style.height = "";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + "px";
  });

  // New Chat button
  newChatBtn.addEventListener("click", function () {
    newChat();
  });

  // Model selector: only records the choice for the NEXT new chat.
  // No option rebuild, no focus move, no current-session change.
  profileSelectEl.addEventListener("change", function () {
    selectedProfileId = profileSelectEl.value;
  });

  // Current-session selector: records the draft only — Apply is the
  // only action that sends a request.
  currentProfileSelectEl.addEventListener("change", function () {
    currentProfileDraftId = currentProfileSelectEl.value;
    updateControlStates();
  });

  applyProfileBtn.addEventListener("click", function () {
    applyProfileSwitch();
  });

  // Interaction mode selector: records the draft only — Apply is the
  // only action that sends a request.
  interactionModeSelectEl.addEventListener("change", function () {
    currentInteractionModeDraft = interactionModeSelectEl.value;
    updateControlStates();
    renderCurrentInteractionModeStatus();
  });

  applyInteractionModeBtn.addEventListener("click", function () {
    applyInteractionModeSwitch();
  });

  historyReviewToggleBtn.addEventListener("click", function () {
    handleHistoryReviewToggle();
  });
  historyReviewStartBtn.addEventListener("click", function () {
    if (currentSessionId === null) return;
    const state = getReviewStateIfExists(currentSessionId);
    if (state === null) return;
    const model = buildHistoryReviewPanelModel({
      sessionId: currentSessionId, state: state,
      currentMode: currentSessionInteractionMode(),
      validateTimestamp: isValidApiTimestamp,
      validateModeSwitchEvent: isValidModeSwitchEvent,
    });
    if (model && model.proposal) {
      startHistoryReview("start", model.proposal.eventId, historyReviewStartBtn);
    }
  });
  historyReviewDismissBtn.addEventListener("click", function () {
    if (currentSessionId === null) return;
    const state = getReviewStateIfExists(currentSessionId);
    if (state === null) return;
    const model = buildHistoryReviewPanelModel({
      sessionId: currentSessionId, state: state,
      currentMode: currentSessionInteractionMode(),
      validateTimestamp: isValidApiTimestamp,
      validateModeSwitchEvent: isValidModeSwitchEvent,
    });
    if (model && model.proposal) {
      state.proposalDismissedEventId = model.proposal.eventId;
      renderHistoryReviewPanel();
    }
  });
  historyReviewContinueBtn.addEventListener("click", function () {
    if (currentSessionId === null) return;
    const state = getReviewStateIfExists(currentSessionId);
    if (state === null) return;
    const model = buildHistoryReviewPanelModel({
      sessionId: currentSessionId, state: state,
      currentMode: currentSessionInteractionMode(),
      validateTimestamp: isValidApiTimestamp,
      validateModeSwitchEvent: isValidModeSwitchEvent,
    });
    if (model && model.selectedSummary) {
      startHistoryReview("continue", model.selectedSummary.mode_switch_event_id, historyReviewContinueBtn);
    }
  });
  historyReviewRecheckBtn.addEventListener("click", function () {
    handleHistoryReviewRecheckAction();
  });
  historyReviewSelectEl.addEventListener("change", function () {
    handleHistoryReviewSelectChange();
  });

  // Mobile sidebar toggle
  sidebarToggleEl.addEventListener("click", function () {
    toggleSidebar();
  });

  // Close sidebar when clicking chat area on mobile
  document.getElementById("chatArea").addEventListener("click", function (e) {
    if (isMobile() && !sidebarEl.classList.contains("collapsed")) {
      if (e.target !== sidebarToggleEl &&
          !sidebarToggleEl.contains(e.target)) {
        closeSidebarOnMobile();
      }
    }
  });

  // Keep aria-expanded in sync when the window is resized
  window.addEventListener("resize", function () {
    if (isMobile()) {
      var expanded = !sidebarEl.classList.contains("collapsed");
      sidebarToggleEl.setAttribute("aria-expanded", String(expanded));
    } else {
      // Keep the latent mobile state collapsed while the desktop CSS
      // displays the sidebar normally.  If the viewport later crosses
      // back into mobile, the drawer is already off-screen instead of
      // jumping over the chat area at the breakpoint.
      sidebarEl.classList.add("collapsed");
      sidebarToggleEl.setAttribute("aria-expanded", "true");
    }
  });

  /* ---- Init -------------------------------------------------------- */

  async function init() {
    isInitializing = true;
    updateControlStates();

    // Keep the latent drawer state collapsed on every viewport.  The
    // class only has a visual effect inside the mobile media query, so
    // the desktop sidebar remains visible while future desktop-to-mobile
    // transitions start in the correct closed state.
    sidebarEl.classList.add("collapsed");
    if (isMobile()) {
      sidebarToggleEl.setAttribute("aria-expanded", "false");
    } else {
      sidebarToggleEl.setAttribute("aria-expanded", "true");
    }

    renderSessionListOrError();
    renderWelcome();
    showStatus("Loading...", false);

    // One-time, fail-closed switch controller/confirmer setup — runs
    // synchronously before the first await.  On failure the rest of
    // the app keeps working; only the model switch bar is disabled.
    initializeProfileSwitching();
    initializeInteractionModeSwitching();
    initializeHistoryReview();

    // Load profiles and sessions in parallel — results are handled
    // independently so one failure never blocks the other.
    const [profilesResult, sessionsResult] = await Promise.allSettled([
      fetchProfiles(),
      fetchSessions(),
    ]);

    if (profilesResult.status === "fulfilled") {
      profiles = profilesResult.value;
      profilesLoadState = "ready";
      profilesLoadError = null;
    } else {
      profiles = [];
      profilesLoadState = "error";
      profilesLoadError =
        profilesResult.reason && profilesResult.reason.message
          ? profilesResult.reason.message
          : "Failed to load models.";
    }
    selectedProfileId = resolveSelectedProfileId(
      registry(),
      selectedProfileId,
    );
    renderProfileSelectorContent();

    if (sessionsResult.status === "fulfilled") {
      if (isValidSessionList(sessionsResult.value)) {
        sessions = sessionsResult.value;
        sessionsLoadState = "ready";
        sessionsEverLoaded = true;
        sessionsLoadError = null;
      } else {
        sessions = [];
        sessionsLoadState = "error";
        sessionsLoadError = "The server returned an invalid response.";
      }
    } else {
      sessions = [];
      sessionsLoadState = "error";
      sessionsLoadError =
        sessionsResult.reason && sessionsResult.reason.message
          ? sessionsResult.reason.message
          : "Failed to load conversations.";
    }

    // Decide the initial selection BEFORE rendering the list so the
    // first highlight, title, badge and message area are consistent.
    if (sessionsLoadState === "ready" && sessions.length > 0) {
      currentSessionId = sessions[0].id;
    }

    renderSessionListOrError();

    if (sessionsLoadState === "ready") {
      if (sessions.length > 0) {
        const first = findSessionInList(currentSessionId);
        currentProfileDraftId = first !== null
          ? resolveSessionProfileDraft(first, registry())
          : null;
        setInteractionModeDraftForSession(first);
        syncCurrentSessionUI();
        renderCurrentProfileBar();
        renderCurrentProfileStatus();
        renderInteractionModeBar();
        await loadMessages(currentSessionId);
        await loadHistoryReviewSession(currentSessionId);
      } else {
        renderWelcome();
        clearStatus();
      }
    } else {
      renderWelcome();
      if (sessionsLoadError) showStatus(sessionsLoadError, true);
    }

    renderCurrentProfileBar();
    renderCurrentProfileStatus();
    renderInteractionModeBar();

    isInitializing = false;
    updateControlStates();

    // Focus rule: only when the page has no explicit user focus and
    // typing is actually possible.
    const canType = currentSessionId === null
      ? registryUsable()
      : currentSessionWritable();
    if (document.activeElement === document.body && canType) {
      inputEl.focus();
    }
  }

  init();
})();
