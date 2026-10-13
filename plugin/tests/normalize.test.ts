import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeAnswers } from "../src/normalize.ts";
import type { DecideRequest } from "../src/types.ts";

const req: DecideRequest = {
  state: "deploy?",
  questions: {
    go: { type: "noul" },
    size: { type: "choice", criteria: { small: "under 1k", big: "over 1k" } },
    risk: { type: "score", criteria: ["low", "medium", "high"] },
  },
};

const full = (): Record<string, unknown> => ({
  go: { value: true, probability: 0.9 },
  size: { choice: "small", confidence: 0.8, probabilities: { small: 0.8, big: 0.2 } },
  risk: { confidence: 0.5, probabilities: [0.25, 0.5, 0.25], legend: ["low", "medium", "high"] },
});

test("choice answers keep the chosen option and its probabilities", () => {
  const out = normalizeAnswers(full(), req);
  assert.deepEqual(out.size, {
    type: "choice",
    choice: "small",
    confidence: 0.8,
    probabilities: { small: 0.8, big: 0.2 },
  });
});

test("score answers take the argmax as level and the expected value when score is missing", () => {
  const out = normalizeAnswers(full(), req);
  assert.deepEqual(out.risk, {
    type: "score",
    score: 1,
    level: 1,
    confidence: 0.5,
    probabilities: [0.25, 0.5, 0.25],
  });
});

test("score answers use a given score as is", () => {
  const raw = { ...full(), risk: { score: 2.5, confidence: 0.9, probabilities: [0.1, 0.2, 0.7] } };
  const out = normalizeAnswers(raw, req);
  assert.equal(out.risk.type === "score" && out.risk.score, 2.5);
  assert.equal(out.risk.type === "score" && out.risk.level, 2);
});

test("noul accepts a bare number, {probability}, or {value, probability}", () => {
  const bare = normalizeAnswers({ ...full(), go: 0.75 }, req);
  assert.deepEqual(bare.go, { type: "noul", value: true, probability: 0.75, confidence: 0.75 });

  const onlyProbability = normalizeAnswers({ ...full(), go: { probability: 0.25 } }, req);
  assert.deepEqual(onlyProbability.go, { type: "noul", value: false, probability: 0.25, confidence: 0.75 });

  const withValue = normalizeAnswers({ ...full(), go: { value: true, probability: 0.9 } }, req);
  assert.deepEqual(withValue.go, { type: "noul", value: true, probability: 0.9, confidence: 0.9 });
});

test("the official server's shapes: noul as {noul: p}, score probabilities keyed by level", () => {
  // As joint_schema_model.systemone_answer writes them.
  const official = {
    go: { type: "noul", noul: 0.2 },
    size: { type: "choice", choice: "big", confidence: 0.7, probabilities: { small: 0.3, big: 0.7 } },
    risk: { type: "score", score: 1.2, confidence: 0.6, legend: { "0": "low", "1": "medium", "2": "high" },
      probabilities: { "0": 0.1, "1": 0.6, "2": 0.3 } },
  };
  const out = normalizeAnswers(official, req);
  assert.deepEqual(out.go, { type: "noul", value: false, probability: 0.2, confidence: 0.8 });
  assert.deepEqual(out.risk, { type: "score", score: 1.2, level: 1, confidence: 0.6, probabilities: [0.1, 0.6, 0.3] });
});

test("a missing answer throws and names the question", () => {
  const { size: _size, ...raw } = full();
  assert.throws(() => normalizeAnswers(raw, req), /answer for "size" is missing/);
});

test("malformed answers throw and name the question", () => {
  assert.throws(
    () => normalizeAnswers({ ...full(), size: { choice: "medium", confidence: 0.8, probabilities: { small: 0.8, big: 0.2 } } }, req),
    /answer for "size" is malformed/,
  );
  assert.throws(() => normalizeAnswers({ ...full(), go: 1.5 }, req), /answer for "go" is malformed/);
  assert.throws(
    () => normalizeAnswers({ ...full(), risk: { confidence: 0.5, probabilities: [0.5, 0.5] } }, req),
    /answer for "risk" is malformed/,
  );
});

test("answers that are not an object throw", () => {
  assert.throws(() => normalizeAnswers(["go"], req), /answers must be an object/);
  assert.throws(() => normalizeAnswers(null, req), /answers must be an object/);
});

test("answers for questions that were not requested are ignored", () => {
  const out = normalizeAnswers({ ...full(), extra: { probability: 0.1 } }, req);
  assert.deepEqual(Object.keys(out), ["go", "size", "risk"]);
});
