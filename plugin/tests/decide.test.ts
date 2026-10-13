import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { decide } from "../src/decide.ts";
import { httpBackend } from "../src/backends/http.ts";
import type { Backend, Config, DecideRequest } from "../src/types.ts";

type Handler = (req: IncomingMessage, res: ServerResponse, body: string) => void;

async function serve(handler: Handler): Promise<{ url: string; server: Server }> {
  const server = createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk: string) => {
      body += chunk;
    });
    req.on("end", () => handler(req, res, body));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const { port } = server.address() as AddressInfo;
  return { url: `http://127.0.0.1:${port}`, server };
}

function stop(server: Server): void {
  server.close();
  server.closeAllConnections();
}

function send(res: ServerResponse, status: number, body: string): void {
  res.writeHead(status, { "content-type": "text/plain" });
  res.end(body);
}

/** A SystemOne stand-in: answers /health and /v1/systemone, recording each decide body in `seen`. */
function systemOne(seen: Record<string, unknown>[]): Handler {
  return (req, res, body) => {
    if (req.url === "/health") return send(res, 200, "ok");
    if (req.url !== "/v1/systemone") return send(res, 404, "not found");
    const parsed = JSON.parse(body) as Record<string, unknown>;
    seen.push(parsed);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ model: parsed.model, answers: { go: { probability: 0.9 } }, usage: { input_tokens: 12 } }));
  };
}

const sample: DecideRequest = { state: { situation: "deploy?" }, questions: { go: { type: "noul" } } };
const config: Config = { backends: [], timeout_ms: 2000 };

test("answers from a responding backend", async () => {
  const seen: Record<string, unknown>[] = [];
  const { url, server } = await serve(systemOne(seen));
  try {
    const result = await decide(sample, config, [httpBackend({ kind: "http", name: "local", url })]);
    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.backend, "local");
    assert.equal(result.model, "clef-flash");
    assert.deepEqual(result.answers.go, { type: "noul", value: true, probability: 0.9, confidence: 0.9 });
    assert.equal(result.usage?.input_tokens, 12);
    assert.equal(typeof result.latency_ms, "number");
    assert.deepEqual(seen[0], { model: "clef-flash", state: sample.state, questions: sample.questions });
  } finally {
    stop(server);
  }
});

test("falls back from a failing backend to the next one", async () => {
  const seen: Record<string, unknown>[] = [];
  const bad = await serve((_req, res) => send(res, 500, "boom"));
  const good = await serve(systemOne(seen));
  try {
    const backends = [
      httpBackend({ kind: "http", name: "bad", url: bad.url }),
      httpBackend({ kind: "http", name: "good", url: good.url }),
    ];
    const result = await decide(sample, config, backends);
    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.backend, "good");
    assert.equal(seen.length, 1);
  } finally {
    stop(bad.server);
    stop(good.server);
  }
});

test("a failing backend is recorded in tried with its HTTP status and body", async () => {
  const bad = await serve((_req, res) => send(res, 500, "boom"));
  try {
    const result = await decide(sample, config, [httpBackend({ kind: "http", name: "bad", url: bad.url })]);
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.deepEqual(result.tried, [{ backend: "bad", error: "HTTP 500: boom" }]);
  } finally {
    stop(bad.server);
  }
});

test("a backend that does not answer in time is recorded in tried", async () => {
  const { url, server } = await serve(() => {});
  try {
    const result = await decide(sample, { ...config, timeout_ms: 50 }, [
      httpBackend({ kind: "http", name: "slow", url }),
    ]);
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.equal(result.tried.length, 1);
    assert.equal(result.tried[0].backend, "slow");
    assert.match(result.tried[0].error, /timeout/i);
  } finally {
    stop(server);
  }
});

test("an invalid request fails before any backend is called", async () => {
  let calls = 0;
  const counting: Backend = {
    name: "counting",
    status: async () => ({ ok: true, detail: "" }),
    decide: async () => {
      calls++;
      throw new Error("should not be called");
    },
  };
  const result = await decide({ state: 1, questions: {} }, config, [counting]);
  assert.deepEqual(result, { ok: false, error: "questions: must be a non-empty object", tried: [] });
  assert.equal(calls, 0);
});

test("each decision appends one JSON line, with images logged as a count", async () => {
  const dir = await mkdtemp(join(tmpdir(), "clefkit-decide-"));
  const logPath = join(dir, "decisions.jsonl");
  const { url, server } = await serve(systemOne([]));
  try {
    const request = { ...sample, images: ["data:image/png;base64,QUJDRA=="] };
    const result = await decide(request, { ...config, log: logPath }, [httpBackend({ kind: "http", url })]);
    assert.equal(result.ok, true);
    const lines = (await readFile(logPath, "utf8")).trim().split("\n");
    assert.equal(lines.length, 1);
    assert.equal(lines[0].includes("QUJDRA"), false);
    const entry = JSON.parse(lines[0]) as { ts: string; request: { images: number }; result: { ok: boolean } };
    assert.equal(entry.request.images, 1);
    assert.equal(entry.result.ok, true);
    assert.equal(Number.isNaN(Date.parse(entry.ts)), false);
  } finally {
    stop(server);
    await rm(dir, { recursive: true, force: true });
  }
});

test("status reports health, and an unreachable backend as not ok", async () => {
  const { url, server } = await serve(systemOne([]));
  try {
    const up = await httpBackend({ kind: "http", url }).status(AbortSignal.timeout(2000));
    assert.deepEqual(up, { ok: true, detail: "ok" });
  } finally {
    stop(server);
  }
  const down = await httpBackend({ kind: "http", url }).status(AbortSignal.timeout(2000));
  assert.equal(down.ok, false);
});
