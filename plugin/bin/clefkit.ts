#!/usr/bin/env node
/**
 * clefkit CLI: `decide` (from a request file, stdin, or a preset plus state), `status` and `presets`.
 * Exit codes: 0 decision or check succeeded, 1 decision failed or no backend answered, 2 bad input.
 */
import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { loadConfig } from "../src/config.ts";
import { decide } from "../src/decide.ts";
import {
  InputError,
  checkBackends,
  errorMessage,
  formatPresets,
  formatStatuses,
  listPresets,
  resolveRequest,
} from "../src/tools.ts";

const USAGE = `usage:
  clefkit decide <request.json | ->
  clefkit decide --preset <name> --state <text | @file>
  clefkit status
  clefkit presets
  clefkit --help`;

const OPTIONS = {
  help: { type: "boolean", short: "h" },
  preset: { type: "string" },
  state: { type: "string" },
} as const;

type Flags = { help?: boolean; preset?: string; state?: string };

function print(text: string): void {
  process.stdout.write(`${text}\n`);
}

async function readText(path: string): Promise<string> {
  try {
    return await readFile(path, "utf8");
  } catch (err) {
    throw new InputError(`cannot read ${path}: ${errorMessage(err)}`);
  }
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

/** Input for `decide --preset <name> --state <text | @file>`. */
async function presetInput(flags: Flags, args: string[]): Promise<unknown> {
  if (args.length > 0) throw new InputError("use either a request file or --preset, not both");
  if (flags.state === undefined) throw new InputError("--state is required with --preset");
  const state = flags.state.startsWith("@") ? await readText(flags.state.slice(1)) : flags.state;
  return { preset: flags.preset, state };
}

/** Input for `decide <request.json | ->`: a request object, read from a file or stdin. */
async function fileInput(flags: Flags, args: string[]): Promise<unknown> {
  if (flags.state !== undefined) throw new InputError("--state needs --preset");
  const [source, ...extra] = args;
  if (source === undefined || extra.length > 0) throw new InputError("decide needs one request file, or - for stdin");
  const text = source === "-" ? await readStdin() : await readText(source);
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new InputError(`request is not valid JSON: ${errorMessage(err)}`);
  }
}

async function decideCommand(flags: Flags, args: string[]): Promise<number> {
  const input = flags.preset !== undefined ? await presetInput(flags, args) : await fileInput(flags, args);
  const result = await decide(await resolveRequest(input), await loadConfig());
  print(JSON.stringify(result, null, 2));
  return result.ok ? 0 : 1;
}

async function statusCommand(): Promise<number> {
  const statuses = await checkBackends(await loadConfig());
  print(formatStatuses(statuses));
  return statuses.some((status) => status.ok) ? 0 : 1;
}

async function presetsCommand(): Promise<number> {
  print(formatPresets(await listPresets()));
  return 0;
}

async function main(argv: string[]): Promise<number> {
  let parsed: ReturnType<typeof parseArgs<{ options: typeof OPTIONS; allowPositionals: true }>>;
  try {
    parsed = parseArgs({ args: argv, options: OPTIONS, allowPositionals: true, strict: true });
  } catch (err) {
    throw new InputError(errorMessage(err));
  }
  const flags = parsed.values as Flags;
  const [command, ...args] = parsed.positionals;
  if (flags.help) {
    print(USAGE);
    return 0;
  }
  switch (command) {
    case "decide":
      return decideCommand(flags, args);
    case "status":
      return statusCommand();
    case "presets":
      return presetsCommand();
    case undefined:
      throw new InputError("no command given");
    default:
      throw new InputError(`unknown command: ${command}`);
  }
}

async function run(argv: string[]): Promise<number> {
  try {
    return await main(argv);
  } catch (err) {
    if (err instanceof InputError) {
      process.stderr.write(`clefkit: ${err.message}\n${USAGE}\n`);
      return 2;
    }
    process.stderr.write(`clefkit: ${errorMessage(err)}\n`);
    return 1;
  }
}

run(process.argv.slice(2)).then((code) => {
  process.exitCode = code;
});
