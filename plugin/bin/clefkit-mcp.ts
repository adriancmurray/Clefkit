#!/usr/bin/env node
/**
 * Clefkit MCP server: the MCP stdio transport, which is newline-delimited JSON-RPC 2.0. It serves the `decide`,
 * `backends_status` and `presets` tools. stdout carries protocol messages only; diagnostics go to stderr.
 */
import { createInterface } from "node:readline";
import { loadConfig } from "../src/config.ts";
import { decide } from "../src/decide.ts";
import {
  checkBackends,
  errorMessage,
  formatPresets,
  formatStatuses,
  isObject,
  listPresets,
  resolveRequest,
} from "../src/tools.ts";

type JsonObject = Record<string, unknown>;
interface ToolOutput {
  content: { type: "text"; text: string }[];
  isError: boolean;
}

const PROTOCOL_VERSION = "2025-06-18";
const SERVER_INFO = { name: "clefkit", version: "0.1.0" };
const NO_ARGUMENTS = { type: "object", properties: {} };

/** The tools/list result. */
const TOOLS = [
  {
    name: "decide",
    description:
      "Make a typed decision about a state from a Clef decision model. Each question returns a yes/no (noul), one " +
      "choice (choice) or a level on a scale (score), with a confidence. Give `questions`, or a `preset` from the " +
      "presets tool. Treat a confidence below 0.6 as unsure.",
    inputSchema: {
      type: "object",
      properties: {
        state: { description: "What to decide on: a string or any JSON value." },
        questions: {
          type: "object",
          description: "Question id to question. Ids are 1-64 characters of [A-Za-z0-9_.-].",
          additionalProperties: {
            type: "object",
            properties: { type: { enum: ["noul", "choice", "score"] } },
            required: ["type"],
          },
        },
        model: { type: "string", description: "Model id for the backend. Omit it for the backend default." },
        preset: {
          type: "string",
          pattern: "^[a-z0-9-]+$",
          description: "Preset name from the presets tool. Supplies the questions when `questions` is omitted.",
        },
      },
      required: ["state"],
    },
  },
  {
    name: "backends_status",
    description: "Check each configured decision backend (2 s timeout each) and report whether it is reachable.",
    inputSchema: NO_ARGUMENTS,
  },
  {
    name: "presets",
    description: "List the bundled question presets with their descriptions. Pass a name as decide's `preset`.",
    inputSchema: NO_ARGUMENTS,
  },
];
const TOOL_NAMES = new Set(TOOLS.map((tool) => tool.name));

function send(message: JsonObject): void {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function respond(id: unknown, result: unknown): void {
  send({ jsonrpc: "2.0", id, result });
}

function fail(id: unknown, code: number, message: string): void {
  send({ jsonrpc: "2.0", id, error: { code, message } });
}

function toolText(body: string, isError: boolean): ToolOutput {
  return { content: [{ type: "text", text: body }], isError };
}

/** Runs one tool. The caller has already checked that the name is known. */
async function callTool(name: string, args: JsonObject): Promise<ToolOutput> {
  switch (name) {
    case "decide": {
      const result = await decide(await resolveRequest(args), await loadConfig());
      return toolText(JSON.stringify(result, null, 2), !result.ok);
    }
    case "backends_status":
      return toolText(formatStatuses(await checkBackends(await loadConfig())), false);
    case "presets":
      return toolText(formatPresets(await listPresets()), false);
    default:
      throw new Error(`no such tool: ${name}`);
  }
}

/** tools/call. Unknown tools and malformed arguments are protocol errors; failures inside a tool are tool results. */
async function handleCall(id: unknown, params: JsonObject): Promise<void> {
  const name = params.name;
  if (typeof name !== "string" || !TOOL_NAMES.has(name)) {
    return fail(id, -32602, `Unknown tool: ${String(name)}`);
  }
  const args = params.arguments ?? {};
  if (!isObject(args)) return fail(id, -32602, "arguments must be an object");
  try {
    respond(id, await callTool(name, args));
  } catch (err) {
    console.error(`clefkit-mcp: tool ${name} failed: ${errorMessage(err)}`);
    respond(id, toolText(errorMessage(err), true));
  }
}

/** Handles one JSON-RPC message. Notifications carry no id and never get a reply. */
async function handle(line: string): Promise<void> {
  let msg: unknown;
  try {
    msg = JSON.parse(line);
  } catch {
    return fail(null, -32700, "Parse error");
  }
  if (!isObject(msg)) return fail(null, -32600, "Invalid Request");
  if (typeof msg.method !== "string") {
    // A reply to a request this server never sent needs no answer.
    if (!("result" in msg) && !("error" in msg)) fail(msg.id ?? null, -32600, "Invalid Request");
    return;
  }
  if (!("id" in msg)) return;
  const id = msg.id;
  const params = isObject(msg.params) ? msg.params : {};
  switch (msg.method) {
    case "initialize":
      return respond(id, {
        protocolVersion: typeof params.protocolVersion === "string" ? params.protocolVersion : PROTOCOL_VERSION,
        capabilities: { tools: {} },
        serverInfo: SERVER_INFO,
      });
    case "ping":
      return respond(id, {});
    case "tools/list":
      return respond(id, { tools: TOOLS });
    case "tools/call":
      return handleCall(id, params);
    default:
      return fail(id, -32601, `Method not found: ${msg.method}`);
  }
}

createInterface({ input: process.stdin }).on("line", (line) => {
  if (line.trim() === "") return;
  handle(line).catch((err: unknown) => console.error(`clefkit-mcp: ${errorMessage(err)}`));
});
