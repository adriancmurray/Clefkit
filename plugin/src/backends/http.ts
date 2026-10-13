import { normalizeAnswers } from "../normalize.ts";
import type { Backend, BackendResult, DecideRequest, HttpBackendConfig } from "../types.ts";
import { isRecord } from "../validate.ts";

const DEFAULT_MODEL = "clef-flash";

/** Header values may contain ${NAME}, replaced from the environment when the request is made; a missing variable is an error. */
export function expandHeaders(headers?: Record<string, string>): Record<string, string> | undefined {
  if (!headers) return undefined;
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    out[key] = value.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_, name: string) => {
      const found = process.env[name];
      if (!found) throw new Error(`header ${key} needs the environment variable ${name}, which is not set`);
      return found;
    });
  }
  return out;
}

/** A backend that speaks SystemOne over HTTP: POST `${url}/v1/systemone` to decide, GET `${url}/health` to check. */
export function httpBackend(cfg: HttpBackendConfig): Backend {
  const base = cfg.url.replace(/\/+$/, "");
  return {
    name: cfg.name ?? "http",
    async status(signal) {
      try {
        const res = await fetch(`${base}/health`, { headers: expandHeaders(cfg.headers), signal });
        const text = await res.text();
        return { ok: res.ok, detail: text.slice(0, 100) };
      } catch (err) {
        return { ok: false, detail: err instanceof Error ? err.message : String(err) };
      }
    },
    async decide(req, signal) {
      const model = req.model ?? cfg.model ?? DEFAULT_MODEL;
      const res = await fetch(`${base}/v1/systemone`, {
        method: "POST",
        headers: { "content-type": "application/json", ...expandHeaders(cfg.headers) },
        body: JSON.stringify({ model, state: req.state, questions: req.questions, images: req.images }),
        signal,
      });
      const text = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
      return toResult(JSON.parse(text), model, req);
    },
  };
}

function toResult(body: unknown, requestedModel: string, req: DecideRequest): BackendResult {
  if (!isRecord(body)) throw new Error("response body must be a JSON object");
  const usage = isRecord(body.usage) && typeof body.usage.input_tokens === "number"
    ? { input_tokens: body.usage.input_tokens }
    : undefined;
  return {
    model: typeof body.model === "string" ? body.model : requestedModel,
    answers: normalizeAnswers(body.answers, req),
    usage,
  };
}
