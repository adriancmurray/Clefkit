# Clefkit

TLDR: typed decisions (yes/no, pick one, a level on a scale) from Clef-style decision models, through one small
interface. Use it from Claude Code (an MCP tool and a skill), from the command line, or from your own code. It talks
to a local server running Cloudflare's Clef-flash on your machine (PyTorch, or MLX on Apple Silicon), to Cloudflare
Workers AI, or to any server that speaks the SystemOne / Jev request format.

## What a decision looks like

You send a `state` (text or JSON describing the situation) and typed `questions`:

```json
{
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "team": { "type": "choice", "criteria": { "billing": "Payments or invoices", "technical": "Bugs or outages" } },
    "urgency": { "type": "score", "criteria": ["Can wait", "This week", "Today"] },
    "outage": { "type": "noul", "instructions": "Is a service down?" }
  }
}
```

Every answer comes back with probabilities and a confidence, in the same shape whichever backend answered. Treat a
confidence below about 0.6 as "unsure" and fall back to your own rule.

## Parts

| Part | Where | What it does |
|---|---|---|
| Core | `plugin/src/` | Validation, backends tried in order with a timeout each, answer normalisation, an optional JSONL decision log |
| Backends | `plugin/src/backends/` | `http` (any SystemOne server: the local one, Jev, others) and `workers-ai` (Cloudflare) |
| MCP server | `plugin/bin/clefkit-mcp.ts` | Tools `decide`, `backends_status`, `presets` |
| CLI | `plugin/bin/clefkit.ts` | `clefkit decide`, `clefkit status`, `clefkit presets` |
| Presets | `plugin/presets/` | Reusable question sets: `model-routing`, `command-risk`, `triage` |
| Local server | `server/clefkit_server/` | SystemOne over HTTP on 127.0.0.1:8790, engines `torch`, `mlx`, `fake` |
| MLX port | `server/mlx/clefkit_mlx/` | Clef-flash on Apple Silicon, with an 8/4-bit quantizer that leaves the decision head at full precision |
| Parity | `server/mlx/parity/` | Measures whether a port or quant still makes the same decisions as the original |

## Install

Requirements: Claude Code 2.1.275 or later for the plugin, and Node 22.18 or later for the CLI (22.18 is the target;
only Node 25.9 has been tested). There are no npm packages.

```text
/plugin marketplace add adriancmurray/Clefkit
/plugin install clefkit@clefkit
```

HaiKrew users who already added this marketplace can run `/plugin marketplace update haikrew` and then
`/plugin install clefkit@haikrew` instead. To try it from a clone for one session:
`claude --plugin-dir /path/to/Clefkit/plugin`. The plugin adds the MCP tools `decide`, `backends_status` and `presets`,
and the `decide` skill. The CLI is `node plugin/bin/clefkit.ts`. The plugin install does not include `worker/`, so the
wrangler proxy in option A needs a clone.

The plugin needs a backend. Pick one of the three below.

### A. Workers AI through the included wrangler proxy (recommended)

No GPU needed. The plugin install brings the MCP tools, the skill and the CLI, but not `worker/`, so clone the
repository first. `worker/` is a loopback-only SystemOne endpoint on `127.0.0.1:8788` that forwards to Cloudflare
Workers AI under your wrangler login. It has no auth, so do not expose or deploy it as is.

```sh
npm install -g wrangler && wrangler login    # or any install that puts wrangler on PATH
git clone https://github.com/adriancmurray/Clefkit.git ~/Clefkit
~/Clefkit/worker/launchd/install.sh          # macOS: runs it at login as a LaunchAgent (undo: ~/Clefkit/worker/launchd/uninstall.sh)
# any other system, or to run it by hand:
cd ~/Clefkit/worker && wrangler dev --port 8788 --ip 127.0.0.1
curl http://127.0.0.1:8788/health
```

Then point Clefkit at it, in `~/.config/clefkit/config.json` (or the file named by `CLEFKIT_CONFIG`), or set
`CLEFKIT_URL=http://127.0.0.1:8788`:

```json
{
  "backends": [{ "kind": "http", "name": "workers-ai-proxy", "url": "http://127.0.0.1:8788" }],
  "timeout_ms": 3000
}
```

Each decision uses a little of the Workers AI daily allocation; see `docs/workers-ai.md` and `worker/README.md`. The
`workers-ai` backend kind calls the Cloudflare REST API directly instead, with `CLOUDFLARE_ACCOUNT_ID` and
`CLOUDFLARE_API_TOKEN` from the environment (never from the config file).

### B. Any SystemOne / Jev endpoint by URL

Any server that answers `POST <url>/v1/systemone` and `GET <url>/health` in the SystemOne format works. Header values
may use `${NAME}`, which is read from the environment when a request is made, so a token never sits in the file:

```json
{
  "backends": [
    {
      "kind": "http",
      "name": "jev",
      "url": "https://your-endpoint.example",
      "headers": { "Authorization": "Bearer ${JEV_API_KEY}" }
    }
  ]
}
```

The hosted Jev API's URL, auth scheme and model names come from TypeSafe's documentation. They have not been tested here.
Backends are tried in order, each with `timeout_ms`, so you can list a local one first and a hosted one after it.

### C. Local: the `server/` folder

Runs Cloudflare's clef-flash on your machine, so no state leaves it. See `server/README.md`.

- Weights: the official Cloudflare clef-flash release from Hugging Face (the folder holds `joint_schema_model.py` and the
  weights). Set `CLEF_MODEL_DIR` to it.
- `torch` engine: `python3 -m clefkit_server --engine torch --model-dir "$CLEF_MODEL_DIR" --device mps` (or `cpu`, `cuda`).
  Needs torch and transformers. The BF16 release is about 19 GB on disk and the server refuses to start with less than
  14 GiB free (`--min-free-gib`).
- `mlx` engine: Apple Silicon only. Needs `mlx`, `mlx-lm`, `transformers` and numpy. Quantize once, into a new folder; the
  quantizer leaves the decision head at full precision:

  ```sh
  cd server/mlx
  "${CLEFKIT_PYTHON:-python3}" -m clefkit_mlx.quantize --source "$CLEF_MODEL_DIR" --dest ./clef-flash-mlx-q8 --bits 8 --group-size 64
  cd .. && python3 -m clefkit_server --engine mlx --model-dir ./mlx/clef-flash-mlx-q8
  ```

  Memory as measured by the author: the 8-bit build peaks at about 11.6 GiB. That figure is not reproduced by any test in
  this repository.
- Config: `{ "kind": "http", "name": "local", "url": "http://127.0.0.1:8790" }`.

Environment variables used by the Python tooling, each with an error when unset and needed: `CLEF_MODEL_DIR` (model
folder), `CLEFKIT_PYTHON` (the Python for the MLX steps in the parity guide), `CLEF_CASES` (cases file for the parity
scripts).

## Using it with HaiKrew

[HaiKrew](https://github.com/adriancmurray/HaiKrew) can ask Clef which model to use for a task. Start any backend above
that serves SystemOne over HTTP, then in HaiKrew set `decider.url` to its base URL and `decider.mode` to `shadow`
(HaiKrew posts to `<url>/v1/systemone`, and in shadow mode it records the pick without acting on it). For example
`http://127.0.0.1:8788` for the wrangler proxy or `http://127.0.0.1:8790` for the local server.

## Parts and config

The config file is `~/.config/clefkit/config.json`, or the file named by `CLEFKIT_CONFIG`. Optional keys: `timeout_ms`
and `log` (a path to a JSONL decision log, which holds task text). The server and MLX port are described in
`server/README.md` and `server/mlx/parity/README.md`.

## Privacy

The state you send goes to whichever backend answers. The local server keeps it on your machine; Workers AI and Jev
send it to those services. The decision log, when enabled, holds task text and is created readable by its owner only.

## Tests

```sh
node --test 'plugin/tests/*.test.ts'
python3 -m unittest discover -s server/tests
```

## What has NOT been verified

- The accuracy of 8-bit and 4-bit MLX builds against the official PyTorch model, and the MLX and torch latency, are not
  documented here: the parity outputs are git-ignored (`server/mlx/parity/results/`), so a clone cannot reproduce a
  claim about them without running the parity guide.
- The Workers AI backend kind (direct REST) has not been called against Cloudflare in this repository's tests; its
  response shape is assumed to match SystemOne. The wrangler proxy was run by the author, not in CI.
- Loading the MCP server inside Claude Code as a plugin, and the skill triggering, are untested.
- Only Node 25.9 and Python 3.14 were used for the tests. Linux, Windows and Intel Macs are untested.
- The marketplace install is untested: `/plugin marketplace add`, `/plugin install clefkit@clefkit`, and the HaiKrew
  `/plugin marketplace update haikrew` and `clefkit@haikrew` path. `claude plugin validate` passes on the manifests only.
- The wrangler proxy has run only on the author's machine. Its install on other systems, including non-macOS systems,
  is untested, and so is running `install.sh` from a fresh clone at `~/Clefkit`.
- The Jev hosted API and `${NAME}` header expansion against a real server are untested. The 11.6 GiB figure is the
  author's single measurement.
