# clefkit server

A small local HTTP server that answers SystemOne requests. The server uses only the Python standard library. The torch engine needs torch and transformers, which the Clef model folder's environment provides.

## Engines

- `torch`: loads a Clef release from `--model-dir` (the folder must contain `joint_schema_model.py`) with `load_release_model`, onto `--device` (`mps`, `cpu` or `cuda`). Weights load on the first request, not at startup.
- `mlx`: Clef-flash on Apple Silicon through `server/mlx/clefkit_mlx`. Needs `mlx`, `mlx-lm`, `transformers` and numpy, and `--model-dir` (a BF16 release folder or a folder made by `clefkit_mlx.quantize`).
- `fake`: deterministic answers for tests. Needs nothing installed.

## Running

```sh
cd server
python3 -m clefkit_server --engine fake
python3 -m clefkit_server --engine torch --model-dir "$CLEF_MODEL_DIR" --device mps
```

Options: `--port` (default 8790), `--host` (default 127.0.0.1), `--min-free-gib` (default 14), `--allow-remote`.

The server binds only loopback addresses. A non-loopback `--host` is refused unless `--allow-remote` is given. One decision runs at a time, because the model is not thread-safe. `/health` does not wait for a decision in progress.

The torch engine adds the model folder to `sys.path` and sets `sys.dont_write_bytecode`, so nothing is written into the model folder.

## Memory guard

Before the torch engine is created, the server runs `vm_stat` and sums free, inactive and speculative pages times the page size. It refuses to start when the total is below `--min-free-gib`. The check runs on macOS only. Elsewhere it is skipped and a line says so. The fake engine skips the check. The check runs once at startup, so memory can drop before the first request loads the weights.

## Endpoints

`GET /health` returns 200:

```json
{"ok": true, "backend": "torch", "model": "model", "loaded": false}
```

`POST /v1/systemone` takes a JSON object body of at most 4 MiB, which is the SystemOne request (`model`, `state`, `questions`). It returns:

- 200 with the engine's SystemOne response (`model`, `answers`, `usage`). The fake engine omits `usage`.
- 400 `{"error": ...}` for invalid JSON, a body that is not an object, or a `ValueError` from the engine.
- 413 `{"error": ...}` when the declared `Content-Length` is above 4 MiB. The body is not read.
- 500 `{"error": "internal error"}` for any other exception. The traceback goes to stderr.

Any other path returns 404 `{"error": "not found"}`.

## Tests

```sh
cd server
python3 -m unittest discover -s tests
```

The tests use the fake engine and do not load a model.
