import type { Backend, BackendResult, DecideRequest, WorkersAiBackendConfig } from "../types.ts";
import { normalizeAnswers } from "../normalize.ts";

const API = "https://api.cloudflare.com/client/v4/accounts";
const MODEL_IDS: Record<string, string> = {
  "clef": "@cf/cloudflare/clef",
  "clef-flash": "@cf/cloudflare/clef-flash",
};
const DEFAULT_MODEL = "@cf/cloudflare/clef-flash";
// A model id goes into the URL path next to the account's token, so only plain @cf ids are allowed through.
const MODEL_ID_RE = /^@cf\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

type Envelope = { success?: boolean; errors?: { message?: string }[]; result?: unknown };

/** Cloudflare Workers AI (Clef models) over the REST API. Credentials are read from env vars on each call. */
export function workersAiBackend(cfg: WorkersAiBackendConfig): Backend {
  const accountVar = cfg.account_id_env ?? "CLOUDFLARE_ACCOUNT_ID";
  const tokenVar = cfg.token_env ?? "CLOUDFLARE_API_TOKEN";
  const name = cfg.name ?? "workers-ai";

  const creds = () => {
    const account = process.env[accountVar]?.trim();
    const token = process.env[tokenVar]?.trim();
    if (!account || !token) return undefined;
    return { account, token };
  };
  const missing = () => new Error(`Workers AI credentials not set: ${accountVar} / ${tokenVar}`);

  const failure = (env: Envelope, status: number): Error => {
    const msgs = (env.errors ?? []).map((e) => e?.message).filter((m): m is string => typeof m === "string");
    return new Error(`Workers AI error (HTTP ${status}): ${(msgs.join("; ") || "request failed").slice(0, 200)}`);
  };

  return {
    name,
    async status(signal) {
      const c = creds();
      if (!c) return { ok: false, detail: `credentials not set: ${accountVar} / ${tokenVar}` };
      try {
        // Token verification does not run inference.
        const res = await fetch(`${API}/${encodeURIComponent(c.account)}/tokens/verify`, {
          headers: { Authorization: `Bearer ${c.token}` }, signal,
        });
        const env = (await res.json().catch(() => ({}))) as Envelope;
        if (res.ok && env.success) return { ok: true, detail: "credentials verified" };
        return { ok: false, detail: failure(env, res.status).message };
      } catch (err) {
        return { ok: false, detail: `credentials set, Cloudflare unreachable: ${err instanceof Error ? err.message : String(err)}` };
      }
    },

    async decide(req: DecideRequest, signal): Promise<BackendResult> {
      const c = creds();
      if (!c) throw missing();
      const requested = (req.model ?? cfg.model ?? DEFAULT_MODEL).trim();
      const modelId = MODEL_IDS[requested] ?? requested;
      if (!MODEL_ID_RE.test(modelId) || modelId.includes("..")) throw new Error(`unsupported Workers AI model id: ${requested.slice(0, 80)}`);
      const shortName = modelId.split("/").pop() ?? requested;
      const body: Record<string, unknown> = {
        model: MODEL_IDS[shortName] ? shortName : "clef-flash",
        state: req.state,
        questions: req.questions,
      };
      if (req.images?.length) {
        // Workers AI accepts embedded images only, no remote URLs.
        if (req.images.some((i) => !i.startsWith("data:"))) throw new Error("Workers AI accepts data: URL images only");
        body.images = req.images;
      }
      const res = await fetch(`${API}/${encodeURIComponent(c.account)}/ai/run/${modelId}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${c.token}`, "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });
      const env = (await res.json().catch(() => ({}))) as Envelope;
      if (!res.ok || env.success === false) throw failure(env, res.status);
      const result = (env.result ?? env) as { model?: string; answers?: unknown; usage?: { input_tokens?: number; prompt_tokens?: number } };
      if (!result || typeof result !== "object" || result.answers === undefined) throw new Error("Workers AI response has no answers");
      const input = result.usage?.input_tokens ?? result.usage?.prompt_tokens;
      return {
        model: modelId,
        answers: normalizeAnswers(result.answers, req),
        ...(typeof input === "number" ? { usage: { input_tokens: input } } : {}),
      };
    },
  };
}
