import type {
  Answer,
  ChoiceAnswer,
  ChoiceQuestion,
  DecideRequest,
  NoulAnswer,
  Question,
  ScoreAnswer,
  ScoreQuestion,
} from "./types.ts";
import { isRecord } from "./validate.ts";

/**
 * Turns a SystemOne `answers` object into the contract's Answer types, one per requested question.
 * Throws when an answer is missing or malformed. Answers for questions that were not requested are ignored.
 */
export function normalizeAnswers(raw: unknown, req: DecideRequest): Record<string, Answer> {
  if (!isRecord(raw)) throw new Error("answers must be an object");
  const out: Record<string, Answer> = {};
  for (const [id, question] of Object.entries(req.questions)) {
    const answer = Object.hasOwn(raw, id) ? raw[id] : undefined;
    if (answer === undefined || answer === null) throw new Error(`answer for "${id}" is missing`);
    out[id] = normalizeOne(id, question, answer);
  }
  return out;
}

function normalizeOne(id: string, question: Question, answer: unknown): Answer {
  switch (question.type) {
    case "noul":
      return noulAnswer(id, answer);
    case "choice":
      return choiceAnswer(id, question, answer);
    case "score":
      return scoreAnswer(id, question, answer);
  }
}

/**
 * Accepts the official `{type: "noul", noul: p}`, a bare probability of true, or an object carrying `probability`
 * (any `value` is recomputed from it).
 */
function noulAnswer(id: string, answer: unknown): NoulAnswer {
  const probability = typeof answer === "number" ? answer
    : isRecord(answer) ? (answer.noul ?? answer.probability) : undefined;
  if (!isProbability(probability)) malformed(id, "expected a probability between 0 and 1");
  return {
    type: "noul",
    value: probability >= 0.5,
    probability,
    confidence: Math.max(probability, 1 - probability),
  };
}

function choiceAnswer(id: string, question: ChoiceQuestion, answer: unknown): ChoiceAnswer {
  if (!isRecord(answer)) malformed(id, "expected an object");
  const { choice, confidence, probabilities } = answer;
  if (typeof choice !== "string" || !Object.hasOwn(question.criteria, choice)) {
    malformed(id, "choice must be one of the options");
  }
  if (!isNumber(confidence)) malformed(id, "confidence must be a number");
  if (!isRecord(probabilities)) malformed(id, "probabilities must be an object");
  const byOption: Record<string, number> = {};
  for (const option of Object.keys(question.criteria)) {
    const p = probabilities[option];
    if (!isProbability(p)) malformed(id, `probability for "${option}" must be between 0 and 1`);
    byOption[option] = p;
  }
  return { type: "choice", choice, confidence, probabilities: byOption };
}

/** The level is the argmax of the probabilities; a missing score is the expected level, sum of i * p. */
function scoreAnswer(id: string, question: ScoreQuestion, answer: unknown): ScoreAnswer {
  if (!isRecord(answer)) malformed(id, "expected an object");
  const { score, confidence } = answer;
  // The official server keys probabilities by level ("0", "1", ...); others send a list.
  const probabilities = isRecord(answer.probabilities)
    ? question.criteria.map((_, i) => (answer.probabilities as Record<string, unknown>)[String(i)])
    : answer.probabilities;
  if (
    !Array.isArray(probabilities) ||
    probabilities.length !== question.criteria.length ||
    !probabilities.every(isProbability)
  ) {
    malformed(id, `probabilities must be ${question.criteria.length} numbers between 0 and 1`);
  }
  let level = 0;
  let expected = 0;
  for (let i = 0; i < probabilities.length; i++) {
    expected += i * probabilities[i];
    if (probabilities[i] > probabilities[level]) level = i;
  }
  const value = score === undefined || score === null ? expected : score;
  if (!isNumber(value)) malformed(id, "score must be a number");
  if (!isNumber(confidence)) malformed(id, "confidence must be a number");
  return { type: "score", score: value, level, confidence, probabilities: probabilities as number[] };
}

function malformed(id: string, why: string): never {
  throw new Error(`answer for "${id}" is malformed: ${why}`);
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isProbability(value: unknown): value is number {
  return isNumber(value) && value >= 0 && value <= 1;
}
