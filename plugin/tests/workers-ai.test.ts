import { test } from "node:test";
import assert from "node:assert/strict";
import { workersAiBackend } from "../src/backends/workers-ai.ts";

const TOKEN = "secret-token-value-123";
const req = { state: "down", questions: { urgent: { type: "noul" as const, instructions: "urgent?" } } };
const sig = new AbortController().signal;
const realFetch = globalThis.fetch;
let calls: { url: string; init: RequestInit }[] = [];

function fake(status: number, json: unknown) {
  calls = [];
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    calls.push({ url: String(url), init });
    return new Response(JSON.stringify(json), { status });
  }) as typeof fetch;
}
function env(set: boolean) {
  if (set) { process.env.CLOUDFLARE_ACCOUNT_ID = "acct1"; process.env.CLOUDFLARE_API_TOKEN = TOKEN; }
  else { delete process.env.CLOUDFLARE_ACCOUNT_ID; delete process.env.CLOUDFLARE_API_TOKEN; }
}
test.afterEach(() => { globalThis.fetch = realFetch; });

test("request URL, auth and body", async () => {
  env(true);
  fake(200, { success: true, errors: [], result: { model: "clef-flash", answers: { urgent: { type: "noul", value: true, probability: 0.9, confidence: 0.8 } }, usage: { input_tokens: 12 } } });
  const out = await workersAiBackend({ kind: "workers-ai" }).decide({ ...req, model: "clef" }, sig);
  assert.equal(calls[0].url, "https://api.cloudflare.com/client/v4/accounts/acct1/ai/run/@cf/cloudflare/clef");
  assert.equal((calls[0].init.headers as Record<string, string>).Authorization, `Bearer ${TOKEN}`);
  const body = JSON.parse(String(calls[0].init.body));
  assert.equal(body.model, "clef");
  assert.deepEqual(body.questions, req.questions);
  assert.equal(body.state, "down");
  assert.ok(out.answers.urgent);
  assert.equal(out.usage?.input_tokens, 12);
});

test("default model is clef-flash", async () => {
  env(true);
  fake(200, { success: true, result: { answers: {} } });
  await workersAiBackend({ kind: "workers-ai" }).decide(req, sig).catch(() => {});
  assert.ok(calls[0].url.endsWith("/ai/run/@cf/cloudflare/clef-flash"));
  assert.equal(JSON.parse(String(calls[0].init.body)).model, "clef-flash");
});

test("remote image URLs are refused", async () => {
  env(true);
  fake(200, {});
  await assert.rejects(workersAiBackend({ kind: "workers-ai" }).decide({ ...req, images: ["https://x/y.png"] }, sig), /data: URL/);
  assert.equal(calls.length, 0);
});

test("model ids that could leave the ai/run path are refused before any request", async () => {
  env(true);
  fake(200, {});
  for (const model of ["../../user/tokens", "@cf/../../x", "@cf/cloudflare/clef?x=1", "x/y"]) {
    await assert.rejects(workersAiBackend({ kind: "workers-ai" }).decide({ ...req, model }, sig), /unsupported Workers AI model id/);
  }
  assert.equal(calls.length, 0);
});

test("status reports an unreachable Cloudflare as not ok", async () => {
  env(true);
  globalThis.fetch = (async () => { throw new Error("getaddrinfo ENOTFOUND"); }) as typeof fetch;
  const s = await workersAiBackend({ kind: "workers-ai" }).status(sig);
  assert.equal(s.ok, false);
  assert.ok(!s.detail.includes(TOKEN));
});

test("error envelope throws Cloudflare messages, never the token", async () => {
  env(true);
  fake(403, { success: false, errors: [{ message: "Authentication error" }], result: null });
  await assert.rejects(workersAiBackend({ kind: "workers-ai" }).decide(req, sig), (e: Error) => {
    assert.match(e.message, /Authentication error/);
    assert.ok(!e.message.includes(TOKEN));
    return true;
  });
});

test("missing credentials", async () => {
  env(false);
  fake(200, {});
  const b = workersAiBackend({ kind: "workers-ai" });
  await assert.rejects(b.decide(req, sig), (e: Error) => {
    assert.equal(e.message, "Workers AI credentials not set: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN");
    return true;
  });
  assert.equal((await b.status(sig)).ok, false);
  assert.equal(calls.length, 0);
});

test("status verifies the token without inference", async () => {
  env(true);
  fake(200, { success: true, result: { status: "active" } });
  const s = await workersAiBackend({ kind: "workers-ai" }).status(sig);
  assert.equal(s.ok, true);
  assert.ok(calls[0].url.endsWith("/accounts/acct1/tokens/verify"));
  assert.ok(!s.detail.includes(TOKEN));
});
