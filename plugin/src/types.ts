/**
 * The shared contract. A decision request follows the SystemOne / Jev request body (Clef's API): a `state` to decide
 * on and typed `questions`. Every backend returns the same normalised response, so callers never see backend quirks.
 */

/** A yes/no question. `criteria` optionally describes what true and false mean. */
export type NoulQuestion = { type: "noul"; instructions?: string; criteria?: { true?: string; false?: string } };
/** Pick one named option: `criteria` maps option id to its description. */
export type ChoiceQuestion = { type: "choice"; instructions?: string; criteria: Record<string, string> };
/** An ordered scale: `criteria[i]` describes level i, from 0. */
export type ScoreQuestion = { type: "score"; instructions?: string; criteria: string[] };
export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;

export interface DecideRequest {
  /** Model id passed to the backend; each backend has a default. */
  model?: string;
  /** Any string or JSON value describing the situation. */
  state: unknown;
  /** Question id to question. Ids are 1-64 chars of [A-Za-z0-9_.-]. */
  questions: Record<string, Question>;
  /** Images as data: URLs or https URLs; backends that cannot read images refuse the request. */
  images?: string[];
}

export type NoulAnswer = { type: "noul"; value: boolean; probability: number; confidence: number };
export type ChoiceAnswer = { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> };
export type ScoreAnswer = { type: "score"; score: number; level: number; confidence: number; probabilities: number[] };
export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export interface DecideResponse {
  ok: true;
  model: string;
  answers: Record<string, Answer>;
  /** Name of the backend that answered. */
  backend: string;
  latency_ms: number;
  usage?: { input_tokens?: number };
}

export interface DecideFailure {
  ok: false;
  error: string;
  /** Each backend tried, in order, with why it failed. */
  tried: { backend: string; error: string }[];
}

export type DecideResult = DecideResponse | DecideFailure;

/** What a backend returns before decide() stamps backend and latency on it. */
export type BackendResult = { model: string; answers: Record<string, Answer>; usage?: { input_tokens?: number } };

export interface Backend {
  name: string;
  /** Quick reachability check, no decision made. */
  status(signal: AbortSignal): Promise<{ ok: boolean; detail: string }>;
  decide(req: DecideRequest, signal: AbortSignal): Promise<BackendResult>;
}

/** Any server speaking SystemOne: POST `${url}/v1/systemone` (the local Clefkit server, Jev, others). */
export type HttpBackendConfig = { kind: "http"; name?: string; url: string; model?: string; headers?: Record<string, string> };
/** Cloudflare Workers AI. Credentials come from environment variables, never from the config file. */
export type WorkersAiBackendConfig = {
  kind: "workers-ai"; name?: string; model?: string; account_id_env?: string; token_env?: string;
};
export type BackendConfig = HttpBackendConfig | WorkersAiBackendConfig;

export interface Config {
  /** Tried in order until one answers. */
  backends: BackendConfig[];
  /** Per-backend timeout. */
  timeout_ms: number;
  /** Optional JSONL file: one line per decision (request, result, time) for later calibration. */
  log?: string;
}

export const DEFAULT_CONFIG: Config = {
  backends: [{ kind: "http", name: "local", url: "http://127.0.0.1:8790" }],
  timeout_ms: 3000,
};
