"use strict";

var TEACHING_STRATEGIES = ["baseline", "a", "b", "ab"];
function isValidTeachingStrategy(value) {
  return typeof value === "string" && TEACHING_STRATEGIES.indexOf(value) !== -1;
}
function resolveTeachingStrategy(value) {
  return value === undefined ? "baseline" : (isValidTeachingStrategy(value) ? value : null);
}
function teachingStrategyLabel(value) {
  return {baseline: "Standard", a: "A — Limited inference", b: "B — Ask for teaching", ab: "A+B — Inference and questions"}[value] || "Unavailable";
}
function teachingStrategyHint(value) {
  return {
    baseline: "Uses the teaching provided without deriving new conclusions.",
    a: "May apply an explicitly taught rule once. Missing teaching is reported without asking for it.",
    b: "Reports direct teaching. May ask one focused question; does not infer new conclusions.",
    ab: "May apply an explicitly taught rule once, or ask one focused question about missing teaching.",
  }[value] || "Teaching strategy could not be confirmed.";
}

function createTeachingStrategySwitchController(deps) {
  function valid(session, id) {
    return deps.validateSession(session) && session.id === id &&
      isValidTeachingStrategy(session.teaching_strategy);
  }
  async function recheck(id) {
    try {
      var fresh = await deps.fetchSession(id);
      if (valid(fresh, id)) return {kind: "confirmed", session: fresh};
    } catch (_) { /* A failed read cannot confirm a previous write. */ }
    return {kind: "uncertain"};
  }
  return {
    recheck: recheck,
    apply: async function (id, strategy) {
      if (!Number.isSafeInteger(id) || id < 1 || !isValidTeachingStrategy(strategy)) {
        return {kind: "invalid"};
      }
      try {
        var body = await deps.patchSwitch(id, strategy);
        if (body && valid(body.session, id) &&
            body.session.teaching_strategy === strategy &&
            body.session.interaction_mode === "receive_teaching") {
          return {kind: "confirmed", session: body.session};
        }
      } catch (_) { /* Reconcile all writes with an unknown outcome. */ }
      return recheck(id);
    },
  };
}

function createTeachingStrategyControl(options) {
  var bar = options.bar;
  if (!bar) return null;
  var select = options.select, apply = options.apply, status = options.status;
  var drafts = Object.create(null), uncertain = Object.create(null), notes = Object.create(null);
  var controller = createTeachingStrategySwitchController(options);
  TEACHING_STRATEGIES.forEach(function (value) {
    var option = document.createElement("option");
    option.value = value; option.textContent = teachingStrategyLabel(value);
    select.appendChild(option);
  });
  function render() {
    var session = options.getSession();
    bar.hidden = !session || session.interaction_mode !== "receive_teaching" ||
      session.teaching_strategy_switchable === false;
    if (options.badge) {
      options.badge.hidden = bar.hidden;
      options.badge.textContent = session && uncertain[session.id]
        ? "Strategy uncertain" : (session ? teachingStrategyLabel(resolveTeachingStrategy(session.teaching_strategy)) : "");
    }
    if (!session) return;
    var id = session.id, value = resolveTeachingStrategy(session.teaching_strategy);
    if (drafts[id] === undefined) drafts[id] = value;
    select.value = drafts[id] || "";
    var busy = options.getBusy();
    select.disabled = busy || !!uncertain[id] || value === null;
    apply.disabled = busy || (!uncertain[id] && value !== null && drafts[id] === value);
    apply.textContent = uncertain[id] || value === null ? "Recheck" : "Apply";
    status.textContent = uncertain[id] || value === null
      ? "Strategy could not be confirmed. Recheck before sending."
      : (notes[id] || (drafts[id] !== value ? "Pending: " + teachingStrategyLabel(drafts[id]) : teachingStrategyHint(value)));
  }
  select.addEventListener("change", function () {
    var session = options.getSession();
    if (!session) return;
    drafts[session.id] = select.value; delete notes[session.id]; render();
  });
  apply.addEventListener("click", async function () {
    var session = options.getSession();
    if (!session || options.getBusy() || session.interaction_mode !== "receive_teaching" ||
        session.teaching_strategy_switchable === false) return;
    var id = session.id, desired = drafts[id];
    options.setBusy(true);
    try {
      var result = uncertain[id] || resolveTeachingStrategy(session.teaching_strategy) === null
        ? await controller.recheck(id) : await controller.apply(id, desired);
      if (result.kind === "confirmed") {
        delete uncertain[id];
        drafts[id] = result.session.teaching_strategy;
        notes[id] = desired === drafts[id] ? "Teaching strategy saved for future messages." : "Server strategy confirmed: " + teachingStrategyLabel(drafts[id]) + ".";
        options.setUncertain(id, false);
        options.onSession(result.session);
      } else {
        uncertain[id] = true; options.setUncertain(id, true);
      }
    } finally {
      options.setBusy(false); render();
    }
  });
  return {render: render};
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {TEACHING_STRATEGIES: TEACHING_STRATEGIES,
    isValidTeachingStrategy: isValidTeachingStrategy, resolveTeachingStrategy: resolveTeachingStrategy,
    teachingStrategyLabel: teachingStrategyLabel, createTeachingStrategySwitchController: createTeachingStrategySwitchController};
}
