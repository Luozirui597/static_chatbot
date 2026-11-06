"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {createTeachingStrategySwitchController, isValidTeachingStrategy, resolveTeachingStrategy} = require("../frontend/teaching-strategy.js");
function session(strategy, id=1) { return {id, teaching_strategy:strategy, interaction_mode:"receive_teaching"}; }
function valid(value) { return !!value && Number.isInteger(value.id); }
test("strict strategies and legacy default", () => {
  for (const value of ["baseline","a","b","ab"]) assert.equal(isValidTeachingStrategy(value),true);
  for (const value of ["A","a+b","a\n",null,1,{}]) assert.equal(isValidTeachingStrategy(value),false);
  assert.equal(resolveTeachingStrategy(undefined),"baseline");
  assert.equal(resolveTeachingStrategy(null),null);
});
test("confirmed patch requires matching identity and desired strategy", async () => {
  let reads=0;
  const c=createTeachingStrategySwitchController({validateSession:valid,patchSwitch:async()=>({session:session("ab")}),fetchSession:async()=>{reads++;}});
  assert.equal((await c.apply(1,"ab")).session.teaching_strategy,"ab");
  assert.equal(reads,0);
});
test("lost write response is reconciled without repeating patch", async () => {
  let patches=0;
  const c=createTeachingStrategySwitchController({validateSession:valid,patchSwitch:async()=>{patches++;throw Error("lost");},fetchSession:async()=>session("a")});
  assert.equal((await c.apply(1,"a")).kind,"confirmed");
  assert.equal(patches,1);
});
test("unconfirmed write remains uncertain and Recheck performs no new write", async () => {
  let patches=0, readable=false;
  const c=createTeachingStrategySwitchController({validateSession:valid,patchSwitch:async()=>{patches++;throw Error();},fetchSession:async()=>{if(!readable)throw Error();return session("b");}});
  assert.equal((await c.apply(1,"b")).kind,"uncertain");
  readable=true;
  assert.equal((await c.recheck(1)).session.teaching_strategy,"b");
  assert.equal(patches,1);
});
test("wrong-session and invalid strategy responses cannot confirm a switch", async () => {
  const c=createTeachingStrategySwitchController({validateSession:valid,patchSwitch:async()=>({session:session("a",2)}),fetchSession:async()=>session("wrong")});
  assert.equal((await c.apply(1,"a")).kind,"uncertain");
});
test("server state is authoritative when requested strategy was not applied", async () => {
  const c=createTeachingStrategySwitchController({validateSession:valid,patchSwitch:async()=>({session:session("b")}),fetchSession:async()=>session("b")});
  assert.equal((await c.apply(1,"ab")).session.teaching_strategy,"b");
});
