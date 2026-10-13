import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { startFakeSystemOne } from "./fake-systemone.ts";

const CLI = fileURLToPath(new URL("../bin/clefkit.ts", import.meta.url));
const MODEL_ROUTING_QUESTIONS = (
  JSON.parse(readFileSync(new URL("../presets/model-routing.json", import.meta.url), "utf8")) as { questions: unknown }
).questions;

function runCli(
  args: string[],
  stdin: string,
  env: Record<string, string | undefined>,
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [CLI, ...args], { env });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk: string) => (stdout += chunk));
    child.stderr.setEncoding("utf8").on("data", (chunk: string) => (stderr += chunk));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.end(stdin);
  });
}

describe("clefkit CLI", () => {
  let fake: Awaited<ReturnType<typeof startFakeSystemOne>>;
  let env: Record<string, string | undefined>;

  before(async () => {
    fake = await startFakeSystemOne();
    env = {
      ...process.env,
      CLEFKIT_URL: fake.url,
      CLEFKIT_CONFIG: join(tmpdir(), `clefkit-missing-${process.pid}.json`),
    };
  });

  after(() => fake.close());

  it("decide - reads a request from stdin and exits 0 with a decision", async () => {
    const request = JSON.stringify({ state: "Rename a helper in one file.", questions: MODEL_ROUTING_QUESTIONS });
    const result = await runCli(["decide", "-"], request, env);
    assert.equal(result.code, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).ok, true);
  });

  it("exits 2 when the request is not valid JSON", async () => {
    const result = await runCli(["decide", "-"], "{not json", env);
    assert.equal(result.code, 2);
    assert.match(result.stderr, /not valid JSON/);
  });
});
