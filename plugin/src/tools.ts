/**
 * Operations shared by the MCP server and the CLI: preset loading, turning caller input into a decide request, backend
 * status checks, and plain-text formatting of their results.
 */
import { readdir, readFile } from "node:fs/promises";
import { makeBackend } from "./backends/index.ts";
import type { BackendConfig, Config, DecideRequest, Question } from "./types.ts";

const PRESET_DIR = new URL("../presets/", import.meta.url);
const PRESET_NAME = /^[a-z0-9-]+$/;
const STATUS_TIMEOUT_MS = 2000;

/** A bundled question set from plugin/presets/<name>.json. */
export interface Preset {
  name: string;
  description: string;
  questions: Record<string, Question>;
}

/** One backend's status check result. */
export interface BackendStatus {
  name: string;
  ok: boolean;
  detail: string;
}

/** A caller mistake (bad preset, malformed request). Front ends report it as bad input, not as a failed decision. */
export class InputError extends Error {}

/** True for a non-null, non-array object. */
export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The message of an error, or its string form when it is not an Error. */
export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Loads plugin/presets/<name>.json. The name is checked against a strict pattern before it reaches the filesystem. */
export async function loadPreset(name: string): Promise<Preset> {
  if (!PRESET_NAME.test(name)) throw new InputError(`invalid preset name: ${name}`);
  let text: string;
  try {
    text = await readFile(new URL(`${name}.json`, PRESET_DIR), "utf8");
  } catch (err) {
    if ((err as { code?: string }).code === "ENOENT") throw new InputError(`unknown preset: ${name}`);
    throw err;
  }
  const body: unknown = JSON.parse(text);
  if (!isObject(body) || !isObject(body.questions)) throw new Error(`preset ${name} has no questions object`);
  return {
    name,
    description: typeof body.description === "string" ? body.description : "",
    questions: body.questions as Record<string, Question>,
  };
}

/** Lists every bundled preset, sorted by name. */
export async function listPresets(): Promise<Preset[]> {
  const names = (await readdir(PRESET_DIR))
    .filter((file) => file.endsWith(".json"))
    .map((file) => file.slice(0, -".json".length))
    .filter((name) => PRESET_NAME.test(name))
    .sort();
  return Promise.all(names.map((name) => loadPreset(name)));
}

/**
 * Turns caller input (`state`, plus `questions` or `preset`, and optional `model` and `images`) into a DecideRequest.
 * Explicit `questions` win over a preset's questions; one of the two is required.
 */
export async function resolveRequest(input: unknown): Promise<DecideRequest> {
  if (!isObject(input)) throw new InputError("request must be a JSON object");
  if (!("state" in input)) throw new InputError("state is required");
  const { model, preset, questions, images } = input;
  if (model !== undefined && typeof model !== "string") throw new InputError("model must be a string");
  if (preset !== undefined && typeof preset !== "string") throw new InputError("preset must be a string");
  if (images !== undefined && !(Array.isArray(images) && images.every((image) => typeof image === "string"))) {
    throw new InputError("images must be an array of strings");
  }
  let chosen: Record<string, Question> | undefined;
  if (typeof preset === "string") chosen = (await loadPreset(preset)).questions;
  if (questions !== undefined) {
    if (!isObject(questions)) throw new InputError("questions must be an object");
    chosen = questions as Record<string, Question>;
  }
  if (chosen === undefined) throw new InputError("give questions or a preset");
  const request: DecideRequest = { state: input.state, questions: chosen };
  if (model !== undefined) request.model = model;
  if (images !== undefined) request.images = images as string[];
  return request;
}

/** Checks one backend. The check is raced against a timeout, so a backend that ignores its abort signal still reports. */
async function checkOne(config: BackendConfig): Promise<BackendStatus> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new Error(`no answer within ${STATUS_TIMEOUT_MS} ms`));
    }, STATUS_TIMEOUT_MS);
  });
  let name = config.name ?? config.kind;
  try {
    const backend = makeBackend(config);
    name = backend.name;
    const { ok, detail } = await Promise.race([backend.status(controller.signal), timeout]);
    return { name, ok, detail };
  } catch (err) {
    return { name, ok: false, detail: errorMessage(err) };
  } finally {
    clearTimeout(timer);
  }
}

/** Checks every configured backend in parallel. */
export async function checkBackends(config: Config): Promise<BackendStatus[]> {
  return Promise.all(config.backends.map((backend) => checkOne(backend)));
}

/** One line per backend: `name: ok=true, detail`. */
export function formatStatuses(statuses: BackendStatus[]): string {
  if (statuses.length === 0) return "no backends configured";
  return statuses.map((status) => `${status.name}: ok=${status.ok}, ${status.detail}`).join("\n");
}

/** One line per preset: `name: description`. */
export function formatPresets(presets: Preset[]): string {
  return presets.map((preset) => `${preset.name}: ${preset.description}`).join("\n");
}
