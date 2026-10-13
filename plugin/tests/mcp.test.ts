import assert from "node:assert/strict";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { after, before, describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { startFakeSystemOne } from "./fake-systemone.ts";

const SERVER = fileURLToPath(new URL("../bin/clefkit-mcp.ts", import.meta.url));
const MODEL_ROUTING = JSON.parse(readFileSync(new URL("../presets/model-routing.json", import.meta.url), "utf8")) as {
  questions: Record<string, unknown>;
};

type Reply = { id: number | null; result?: any; error?: { code: number; message: string } };
type ToolResult = { content: { type: string; text: string }[]; isError?: boolean };

describe("clefkit MCP server over stdio", () => {
  let fake: Awaited<ReturnType<typeof startFakeSystemOne>>;
  let child: ChildProcessWithoutNullStreams;
  const inbox: Reply[] = [];
  let wake: (() => void) | undefined;
  let nextId = 1;

  async function nextReply(): Promise<Reply> {
    while (inbox.length === 0) await new Promise<void>((resolve) => (wake = resolve));
    return inbox.shift() as Reply;
  }

  function send(message: object): void {
    child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  async function call(method: string, params?: object): Promise<Reply> {
    const id = nextId++;
    send({ jsonrpc: "2.0", id, method, params });
    const reply = await nextReply();
    assert.equal(reply.id, id);
    return reply;
  }

  async function callTool(name: string, args: object): Promise<ToolResult> {
    const reply = await call("tools/call", { name, arguments: args });
    assert.equal(reply.error, undefined);
    return reply.result as ToolResult;
  }

  before(async () => {
    fake = await startFakeSystemOne();
    child = spawn(process.execPath, [SERVER], {
      env: {
        ...process.env,
        CLEFKIT_URL: fake.url,
        CLEFKIT_CONFIG: join(tmpdir(), `clefkit-missing-${process.pid}.json`),
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    child.stderr.resume();
    createInterface({ input: child.stdout }).on("line", (line) => {
      inbox.push(JSON.parse(line) as Reply);
      wake?.();
      wake = undefined;
    });
  });

  after(async () => {
    child.kill();
    await fake.close();
  });

  it("echoes the client protocol version and reports the tools capability", async () => {
    const reply = await call("initialize", {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: { name: "test", version: "0" },
    });
    assert.equal(reply.result.protocolVersion, "2025-03-26");
    assert.deepEqual(reply.result.capabilities, { tools: {} });
    assert.deepEqual(reply.result.serverInfo, { name: "clefkit", version: "0.1.0" });
  });

  it("sends no reply to notifications/initialized", async () => {
    send({ jsonrpc: "2.0", method: "notifications/initialized" });
    const reply = await call("ping");
    assert.deepEqual(reply.result, {});
  });

  it("lists the three tools", async () => {
    const reply = await call("tools/list");
    assert.deepEqual(reply.result.tools.map((tool: { name: string }) => tool.name).sort(), [
      "backends_status",
      "decide",
      "presets",
    ]);
  });

  it("returns -32601 for an unknown method", async () => {
    const reply = await call("no/such/method");
    assert.equal(reply.error?.code, -32601);
  });

  it("decides with the model-routing preset and a state string", async () => {
    const result = await callTool("decide", { preset: "model-routing", state: "Rename a helper in one file." });
    assert.notEqual(result.isError, true);
    const decision = JSON.parse(result.content[0].text);
    assert.equal(decision.ok, true);
    assert.deepEqual(Object.keys(decision.answers).sort(), Object.keys(MODEL_ROUTING.questions).sort());
    assert.deepEqual(Object.keys(fake.requests.at(-1)?.questions ?? {}).sort(), Object.keys(MODEL_ROUTING.questions).sort());
  });

  it("reports an invalid preset name as a tool error", async () => {
    const result = await callTool("decide", { preset: "../secrets", state: "x" });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /invalid preset name/);
  });

  it("lists the bundled presets", async () => {
    const result = await callTool("presets", {});
    for (const name of ["model-routing", "command-risk", "triage"]) assert.match(result.content[0].text, new RegExp(name));
  });

  it("reports each backend's status", async () => {
    const result = await callTool("backends_status", {});
    assert.notEqual(result.isError, true);
    assert.match(result.content[0].text, /ok=true/);
  });

  it("returns -32602 for an unknown tool", async () => {
    const reply = await call("tools/call", { name: "no_such_tool", arguments: {} });
    assert.equal(reply.error?.code, -32602);
  });
});
