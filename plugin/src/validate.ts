import type { DecideRequest } from "./types.ts";

const ID_PATTERN = /^[A-Za-z0-9_.-]{1,64}$/;
const MAX_QUESTIONS = 32;
const MAX_INSTRUCTIONS = 2000;
const IMAGE_PREFIXES = ["data:image/", "https://"];

class ValidationError extends Error {}

function fail(path: string, message: string): never {
  throw new ValidationError(`${path}: ${message}`);
}

/** True for a plain JSON object: not null, not an array. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Checks an untrusted decide request. On success returns it typed as a DecideRequest; on failure returns the
 * first offending path and what is wrong with it, e.g. `questions.size.criteria: choice needs 2-64 options`.
 */
export function validateRequest(req: unknown): { ok: true; req: DecideRequest } | { ok: false; error: string } {
  try {
    checkRequest(req);
    return { ok: true, req: req as DecideRequest };
  } catch (err) {
    if (err instanceof ValidationError) return { ok: false, error: err.message };
    throw err;
  }
}

function checkRequest(req: unknown): void {
  if (!isRecord(req)) fail("request", "must be an object");
  if (req.state === undefined) fail("state", "is required");
  checkQuestions(req.questions);
  checkImages(req.images);
  if (req.model !== undefined && typeof req.model !== "string") fail("model", "must be a string");
}

function checkQuestions(questions: unknown): void {
  if (!isRecord(questions) || Object.keys(questions).length === 0) {
    fail("questions", "must be a non-empty object");
  }
  const ids = Object.keys(questions);
  if (ids.length > MAX_QUESTIONS) fail("questions", `at most ${MAX_QUESTIONS} entries`);
  for (const id of ids) {
    if (!ID_PATTERN.test(id)) fail(`questions.${id}`, "id must be 1-64 chars of A-Za-z0-9_.-");
    checkQuestion(`questions.${id}`, questions[id]);
  }
}

function checkQuestion(path: string, question: unknown): void {
  if (!isRecord(question)) fail(path, "must be an object");
  if (question.type !== "noul" && question.type !== "choice" && question.type !== "score") {
    fail(`${path}.type`, "must be noul, choice or score");
  }
  checkInstructions(path, question.instructions);
  if (question.type === "noul") checkNoulCriteria(path, question.criteria);
  else if (question.type === "choice") checkChoiceCriteria(path, question.criteria);
  else checkScoreCriteria(path, question.criteria);
}

function checkInstructions(path: string, instructions: unknown): void {
  if (instructions !== undefined && (typeof instructions !== "string" || instructions.length > MAX_INSTRUCTIONS)) {
    fail(`${path}.instructions`, `must be a string of at most ${MAX_INSTRUCTIONS} chars`);
  }
}

function checkNoulCriteria(path: string, criteria: unknown): void {
  if (criteria === undefined) return;
  const p = `${path}.criteria`;
  if (!isRecord(criteria)) fail(p, "must be an object with optional true and false strings");
  for (const key of ["true", "false"]) {
    if (criteria[key] !== undefined && typeof criteria[key] !== "string") fail(`${p}.${key}`, "must be a string");
  }
}

function checkChoiceCriteria(path: string, criteria: unknown): void {
  const p = `${path}.criteria`;
  if (!isRecord(criteria)) fail(p, "choice needs 2-64 options");
  const options = Object.keys(criteria);
  if (options.length < 2 || options.length > 64) fail(p, "choice needs 2-64 options");
  for (const option of options) {
    if (typeof criteria[option] !== "string") fail(`${p}.${option}`, "option description must be a string");
  }
}

function checkScoreCriteria(path: string, criteria: unknown): void {
  const p = `${path}.criteria`;
  if (!Array.isArray(criteria) || criteria.length < 2 || criteria.length > 16) {
    fail(p, "score needs 2-16 level strings");
  }
  for (let i = 0; i < criteria.length; i++) {
    if (typeof criteria[i] !== "string") fail(`${p}[${i}]`, "must be a string");
  }
}

function checkImages(images: unknown): void {
  if (images === undefined) return;
  if (!Array.isArray(images)) fail("images", "must be an array of strings");
  for (let i = 0; i < images.length; i++) {
    const image = images[i];
    if (typeof image !== "string" || !IMAGE_PREFIXES.some((prefix) => image.startsWith(prefix))) {
      fail(`images[${i}]`, "must start with data:image/ or https://");
    }
  }
}
