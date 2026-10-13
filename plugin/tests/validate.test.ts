import { test } from "node:test";
import assert from "node:assert/strict";
import { validateRequest } from "../src/validate.ts";

const valid = () => ({
  model: "clef-flash",
  state: { text: "hello" },
  questions: {
    go: { type: "noul", instructions: "Should we proceed?", criteria: { true: "proceed", false: "stop" } },
    size: { type: "choice", criteria: { small: "under 1k", big: "over 1k" } },
    risk: { type: "score", criteria: ["low", "medium", "high"] },
  },
  images: ["data:image/png;base64,AAAA", "https://example.com/a.png"],
});

const withQuestions = (questions: unknown) => ({ state: 1, questions });

function errorOf(req: unknown): string {
  const result = validateRequest(req);
  if (result.ok) assert.fail("expected validation to fail");
  return result.error;
}

test("accepts a well-formed request and returns it typed", () => {
  const result = validateRequest(valid());
  assert.equal(result.ok, true);
  if (result.ok) assert.equal(result.req.model, "clef-flash");
});

test("rejects a request that is not an object", () => {
  assert.equal(errorOf(null), "request: must be an object");
  assert.equal(errorOf([]), "request: must be an object");
});

test("requires state", () => {
  const { state: _state, ...rest } = valid();
  assert.equal(errorOf(rest), "state: is required");
});

test("questions must be a non-empty object of at most 32 entries", () => {
  assert.equal(errorOf(withQuestions({})), "questions: must be a non-empty object");
  assert.equal(errorOf(withQuestions([])), "questions: must be a non-empty object");
  const noul = { type: "noul" };
  const tooMany = Object.fromEntries(Array.from({ length: 33 }, (_, i) => [`q${i}`, noul]));
  assert.equal(errorOf(withQuestions(tooMany)), "questions: at most 32 entries");
  const max = Object.fromEntries(Array.from({ length: 32 }, (_, i) => [`q${i}`, noul]));
  assert.equal(validateRequest(withQuestions(max)).ok, true);
});

test("question ids match [A-Za-z0-9_.-] and 1-64 chars", () => {
  const noul = { type: "noul" };
  assert.equal(errorOf(withQuestions({ "bad id": noul })), "questions.bad id: id must be 1-64 chars of A-Za-z0-9_.-");
  const long = "x".repeat(65);
  assert.equal(errorOf(withQuestions({ [long]: noul })), `questions.${long}: id must be 1-64 chars of A-Za-z0-9_.-`);
  assert.equal(validateRequest(withQuestions({ ["x".repeat(64)]: noul, "a.b-c_1": noul })).ok, true);
});

test("question type is noul, choice or score", () => {
  assert.equal(errorOf(withQuestions({ q: { type: "float" } })), "questions.q.type: must be noul, choice or score");
});

test("a question must be an object", () => {
  assert.equal(errorOf(withQuestions({ q: "noul" })), "questions.q: must be an object");
});

test("choice needs 2-64 options with string descriptions", () => {
  const choice = (criteria: unknown) => withQuestions({ size: { type: "choice", criteria } });
  assert.equal(errorOf(choice({ only: "one" })), "questions.size.criteria: choice needs 2-64 options");
  assert.equal(errorOf(choice(undefined)), "questions.size.criteria: choice needs 2-64 options");
  assert.equal(errorOf(choice(["a", "b"])), "questions.size.criteria: choice needs 2-64 options");
  const many = Object.fromEntries(Array.from({ length: 65 }, (_, i) => [`o${i}`, "d"]));
  assert.equal(errorOf(choice(many)), "questions.size.criteria: choice needs 2-64 options");
  assert.equal(errorOf(choice({ small: "a", big: 3 })), "questions.size.criteria.big: option description must be a string");
});

test("score needs 2-16 string levels", () => {
  const score = (criteria: unknown) => withQuestions({ risk: { type: "score", criteria } });
  assert.equal(errorOf(score(["only"])), "questions.risk.criteria: score needs 2-16 level strings");
  assert.equal(errorOf(score("low,high")), "questions.risk.criteria: score needs 2-16 level strings");
  const seventeen = Array.from({ length: 17 }, () => "level");
  assert.equal(errorOf(score(seventeen)), "questions.risk.criteria: score needs 2-16 level strings");
  assert.equal(errorOf(score(["low", 2, "high"])), "questions.risk.criteria[1]: must be a string");
});

test("noul criteria is optional, and true/false are optional strings", () => {
  const noul = (criteria: unknown) => withQuestions({ go: { type: "noul", criteria } });
  assert.equal(validateRequest(noul(undefined)).ok, true);
  assert.equal(validateRequest(noul({ true: "yes" })).ok, true);
  assert.equal(errorOf(noul("yes")), "questions.go.criteria: must be an object with optional true and false strings");
  assert.equal(errorOf(noul({ true: 1 })), "questions.go.criteria.true: must be a string");
});

test("instructions are an optional string of at most 2000 chars", () => {
  const withInstructions = (instructions: unknown) => withQuestions({ go: { type: "noul", instructions } });
  assert.equal(validateRequest(withInstructions("x".repeat(2000))).ok, true);
  assert.equal(
    errorOf(withInstructions("x".repeat(2001))),
    "questions.go.instructions: must be a string of at most 2000 chars",
  );
  assert.equal(errorOf(withInstructions(7)), "questions.go.instructions: must be a string of at most 2000 chars");
});

test("images are an array of data:image/ or https:// strings", () => {
  assert.equal(errorOf({ ...valid(), images: "data:image/png;base64,AAAA" }), "images: must be an array of strings");
  assert.equal(
    errorOf({ ...valid(), images: ["data:image/png;base64,AAAA", "http://example.com/a.png"] }),
    "images[1]: must start with data:image/ or https://",
  );
  assert.equal(errorOf({ ...valid(), images: [42] }), "images[0]: must start with data:image/ or https://");
});

test("model is an optional string", () => {
  assert.equal(errorOf({ ...valid(), model: 5 }), "model: must be a string");
});
