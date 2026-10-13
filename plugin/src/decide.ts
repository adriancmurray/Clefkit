import { appendFile } from "node:fs/promises";
import { makeBackend } from "./backends/index.ts";
import type { Backend, Config, DecideFailure, DecideRequest, DecideResult } from "./types.ts";
import { validateRequest } from "./validate.ts";

/**
 * Validates a decide request and asks the backends in order until one answers. Backends default to the ones in
 * `config`. Never throws: every failure comes back as a DecideFailure that lists what each backend said.
 */
export async function decide(input: unknown, config: Config, backends?: Backend[]): Promise<DecideResult> {
  const checked = validateRequest(input);
  if (!checked.ok) return { ok: false, error: checked.error, tried: [] };
  const result = await firstAnswer(checked.req, config, backends).catch(
    (err: unknown): DecideFailure => ({ ok: false, error: errorMessage(err), tried: [] }),
  );
  await writeLog(config.log, checked.req, result);
  return result;
}

async function firstAnswer(req: DecideRequest, config: Config, backends?: Backend[]): Promise<DecideResult> {
  const started = Date.now();
  const tried: DecideFailure["tried"] = [];
  for (const backend of backends ?? config.backends.map(makeBackend)) {
    try {
      const result = await backend.decide(req, AbortSignal.timeout(config.timeout_ms));
      return { ok: true, ...result, backend: backend.name, latency_ms: Date.now() - started };
    } catch (err) {
      tried.push({ backend: backend.name, error: errorMessage(err) });
    }
  }
  return { ok: false, error: "no backend answered", tried };
}

// Images are logged as a count so data URLs stay out of the file.
async function writeLog(path: string | undefined, req: DecideRequest, result: DecideResult): Promise<void> {
  if (!path) return;
  try {
    const line = JSON.stringify({ ts: new Date().toISOString(), request: { ...req, images: req.images?.length }, result });
    // The log holds task text, so it is created readable by its owner only.
    await appendFile(path, `${line}\n`, { mode: 0o600 });
  } catch {
    // A logging failure must not turn a decision into an error.
  }
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
