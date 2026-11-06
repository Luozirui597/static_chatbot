'use strict';

const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(process.argv[2], 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
const source = scripts.find((item) => item.includes('refreshState'));
assert(source, 'page refresh script not found');

function makeElement(id, tag = 'div') {
  const el = {
    id,
    tagName: String(tag).toUpperCase(),
    hidden: false,
    disabled: false,
    _text: '',
    _children: [],
    _classes: new Set(),
    style: {
      _values: {},
      setProperty(name, value) { this._values[name] = String(value); },
      getPropertyValue(name) { return this._values[name] || ''; },
    },
    classList: {
      add(name) { el._classes.add(name); },
      remove(name) { el._classes.delete(name); },
      contains(name) { return el._classes.has(name); },
      toggle(name, force) {
        const next = force === undefined ? !el._classes.has(name) : Boolean(force);
        if (next) el._classes.add(name); else el._classes.delete(name);
        return next;
      },
    },
    appendChild(child) { el._children.push(child); child.parent = el; return child; },
    get children() { return el._children; },
    get textContent() { return el._text; },
    set textContent(value) {
      el._text = String(value);
      el._children.length = 0;
    },
    get offsetHeight() { return el.hidden ? 0 : (el.id === 'option-b-test-banner' ? 64 : 180); },
    onclick: null,
  };
  return el;
}

const ids = {
  'option-b-test-banner': makeElement('option-b-test-banner'),
  'option-b-state-line': makeElement('option-b-state-line'),
  'option-b-branch-panel': makeElement('option-b-branch-panel'),
  'option-b-branch-list': makeElement('option-b-branch-list', 'div'),
  'option-b-t2-wrap': makeElement('option-b-t2-wrap'),
  'option-b-t2': makeElement('option-b-t2', 'pre'),
};
const documentElement = makeElement('html');
const body = makeElement('body');
const document = {
  getElementById(id) { return ids[id] || null; },
  createElement(tag) { return makeElement('', tag); },
  documentElement,
  body,
};

let stateQueue = [];
let branchResponse = { t2_content: null };
let stateCalls = 0;
let branchCalls = 0;
let intervalCallback = null;

async function fetchStub(url) {
  if (url === '/__option_b__/state') {
    stateCalls += 1;
    assert(stateQueue.length, 'unexpected extra state refresh');
    const state = stateQueue.shift();
    return { ok: true, json: async () => state };
  }
  if (url === '/__option_b__/branch') {
    branchCalls += 1;
    return { ok: true, json: async () => branchResponse };
  }
  throw new Error(`unexpected fetch ${url}`);
}

const sandbox = {
  document,
  window: { addEventListener() {} },
  fetch: fetchStub,
  setInterval(callback) { intervalCallback = callback; return 1; },
  clearInterval() {},
  console,
};
vm.createContext(sandbox);

async function flush() {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

function script(id, phase, selectedT2 = null) {
  return {
    id,
    phase,
    session_id: id === 'I2' ? 2 : id === 'I3' ? 3 : 1,
    q1_assistant_message_id: id === 'I2' ? 102 : 202,
    turns_expected: 3,
    branches: [
      { id: 'bc', label: 'B links to C', is_none: false },
      { id: 'none', label: 'No additional request', is_none: true },
    ],
    selected_t2_content: selectedT2,
  };
}

function state(status, current, scriptItem, paused = false) {
  return {
    status,
    current_script: current,
    paused_human_classification: paused,
    scripts: [scriptItem],
    total_consumed: 20,
    total_max: 24,
  };
}

async function load(initial) {
  stateQueue = [initial];
  branchCalls = 0;
  vm.runInContext(source, sandbox);
  await flush();
}

async function refresh(next) {
  stateQueue.push(next);
  await intervalCallback();
  await flush();
}

function assertHidden(label) {
  assert.strictEqual(ids['option-b-branch-panel'].hidden, true, `${label}: panel hidden`);
  assert.strictEqual(ids['option-b-t2'].textContent, '', `${label}: t2 text empty`);
  assert.strictEqual(ids['option-b-t2-wrap'].hidden, true, `${label}: t2 wrapper hidden`);
  assert.strictEqual(ids['option-b-branch-list'].children.length, 0, `${label}: branch list empty`);
  assert.strictEqual(
    documentElement.style.getPropertyValue('--option-b-panel-height'),
    '0px',
    `${label}: panel layout height restored`,
  );
  assert.strictEqual(body._classes.has('option-b-panel-open'), false, `${label}: layout class cleared`);
}

async function main() {
  ids['option-b-branch-panel'].hidden = true;

  // running/paused -> failed
  await load(state('running', 'I2', script('I2', 'paused_human_classification'), true));
  assert.strictEqual(ids['option-b-branch-panel'].hidden, false);
  assert.strictEqual(ids['option-b-t2'].textContent, '');
  await refresh(state('failed', 'I2', script('I2', 'paused_human_classification'), true));
  assertHidden('paused -> failed');

  // running/branch_selected -> failed
  await load(state('running', 'I2', script('I2', 'branch_selected', 'old I2 T2')));
  assert.strictEqual(ids['option-b-t2'].textContent, 'old I2 T2');
  await refresh(state('failed', 'I2', script('I2', 'branch_selected', 'old I2 T2')));
  assertHidden('branch_selected -> failed');

  // running/branch_selected -> cancelled
  await load(state('running', 'I2', script('I2', 'branch_selected', 'old I2 T2')));
  await refresh(state('cancelled', 'I2', script('I2', 'branch_selected', 'old I2 T2')));
  assertHidden('branch_selected -> cancelled');

  // normal branch selection, then I3 pause clears old T2, then none completes.
  await load(state('running', 'I2', script('I2', 'paused_human_classification'), true));
  branchResponse = { t2_content: 'the frozen I2 T2' };
  const buttons = ids['option-b-branch-list'].children;
  assert(buttons.length >= 1, 'branch buttons rendered');
  stateQueue.push(state('running', 'I2', script('I2', 'branch_selected', 'the frozen I2 T2')));
  await buttons[0].onclick();
  await flush();
  assert.strictEqual(ids['option-b-t2'].textContent, 'the frozen I2 T2');
  assert.strictEqual(ids['option-b-t2-wrap'].hidden, false);

  await refresh(state('running', 'I3', script('I3', 'paused_human_classification'), true));
  assert.strictEqual(ids['option-b-branch-panel'].hidden, false);
  assert.strictEqual(ids['option-b-t2'].textContent, '');
  assert.strictEqual(ids['option-b-t2-wrap'].hidden, true);
  const i3Buttons = ids['option-b-branch-list'].children;
  const noneButton = i3Buttons.find((item) => item.textContent.includes('No additional'));
  assert(noneButton, 'none branch button rendered');
  branchResponse = { t2_content: null };
  stateQueue.push(state('completed', null, script('I3', 'completed'), false));
  await noneButton.onclick();
  await flush();
  assertHidden('none -> completed');

  assert(branchCalls >= 2, 'branch endpoint called for manual selection');
  assert(stateCalls >= 1, 'state endpoint used');
  console.log('page script state sequences ok');
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
