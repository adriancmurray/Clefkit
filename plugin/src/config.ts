import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import type { Config } from "./types.ts";
import { DEFAULT_CONFIG } from "./types.ts";

/**
 * Loads the config from `path`, else $CLEFKIT_CONFIG, else ~/.config/clefkit/config.json. A missing file means the
 * defaults. Values in the file override the defaults. $CLEFKIT_URL, when set, replaces the backends with one HTTP
 * backend at that URL.
 */
export async function loadConfig(path?: string): Promise<Config> {
  const file = path || process.env.CLEFKIT_CONFIG || `${homedir()}/.config/clefkit/config.json`;
  const config: Config = { ...structuredClone(DEFAULT_CONFIG), ...(await readConfigFile(file)) };
  const url = process.env.CLEFKIT_URL;
  if (url) config.backends = [{ kind: "http", url }];
  return config;
}

async function readConfigFile(file: string): Promise<Partial<Config>> {
  let text: string;
  try {
    text = await readFile(file, "utf8");
  } catch (err) {
    if ((err as { code?: unknown }).code === "ENOENT") return {};
    throw err;
  }
  try {
    return JSON.parse(text) as Partial<Config>;
  } catch (err) {
    throw new Error(`invalid JSON in ${file}: ${err instanceof Error ? err.message : String(err)}`);
  }
}
