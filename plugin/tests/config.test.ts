import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";
import { DEFAULT_CONFIG } from "../src/types.ts";

async function withDir(run: (dir: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "clefkit-config-"));
  try {
    await run(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test("a missing file yields the defaults", async () => {
  await withDir(async (dir) => {
    assert.deepEqual(await loadConfig(join(dir, "absent.json")), DEFAULT_CONFIG);
  });
});

test("file values override the defaults; invalid JSON names the file", async () => {
  await withDir(async (dir) => {
    const path = join(dir, "config.json");
    const log = join(dir, "decisions.jsonl");
    await writeFile(path, JSON.stringify({ timeout_ms: 500, log }));
    const config = await loadConfig(path);
    assert.equal(config.timeout_ms, 500);
    assert.equal(config.log, log);
    assert.deepEqual(config.backends, DEFAULT_CONFIG.backends);

    await writeFile(path, "{ not json");
    await assert.rejects(loadConfig(path), (err: Error) => err.message.includes(path));
  });
});

test("CLEFKIT_URL replaces the backends with one HTTP backend", async () => {
  await withDir(async (dir) => {
    const previous = process.env.CLEFKIT_URL;
    process.env.CLEFKIT_URL = "http://127.0.0.1:9999";
    try {
      const config = await loadConfig(join(dir, "absent.json"));
      assert.deepEqual(config.backends, [{ kind: "http", url: "http://127.0.0.1:9999" }]);
    } finally {
      if (previous === undefined) delete process.env.CLEFKIT_URL;
      else process.env.CLEFKIT_URL = previous;
    }
  });
});
