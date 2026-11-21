import { test } from "node:test";
import { strict as assert } from "node:assert";
import { scenarios, simulate } from "./simulation.js";

const inputs = { savings: 20000, income: 48000, expenses: 2500, incomeGrowth: 0, inflation: 0, returnRate: 0, years: 5 };

test("baseline accumulates monthly income less expenses", () => {
  const result = simulate(inputs);
  assert.equal(result.final, 110000);
  assert.equal(result.minimum, 20000);
  assert.equal(result.monthsBelowZero, 0);
  assert.equal(result.points.length, 6);
});

test("scenarios recompute from the same starting state", () => {
  assert.equal(simulate(inputs, "masters").final, 59600);
  assert.equal(simulate(inputs, "relocate").final, 102500);
  assert.equal(simulate(inputs, "business").final, 147800);
  assert.equal(simulate(inputs, "home").final, 56000);
  assert.equal(simulate(inputs, "break").final, 62000);
  assert.equal(simulate(inputs, "shock").final, 81000);
  assert.equal(Object.keys(scenarios).length, 6);
});

test("tracks shortfalls even when the ending balance recovers", () => {
  const result = simulate({ ...inputs, savings: 0 }, "break");
  assert.ok(result.minimum < 0);
  assert.ok(result.final > 0);
  assert.ok(result.monthsBelowZero > 0);
});

test("annual rates compound over the horizon", () => {
  const result = simulate({ ...inputs, savings: 10000, income: 0, expenses: 0, returnRate: 10, years: 1 });
  assert.ok(Math.abs(result.final - 11000) < 0.000001);
});
