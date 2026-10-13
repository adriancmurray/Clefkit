# Clef-flash MLX parity harness

Measures the MLX port (`clefkit_mlx`) against the official PyTorch Clef-flash release. Everything here runs on synthetic inputs or the synthetic cases in the file named by `CLEF_CASES`. [SIMULATION: SYNTHETIC WORKLOAD]

## Files

- `make_head_fixtures.py`: PyTorch env. Loads only `joint_head.safetensors` through the official `JointSchemaHead`, runs it in float32 on six seeded batches, writes `../fixtures/head-00..05.npz`, `head-shared.npz` (synthetic 512-row embedding table) and `head-meta.json` (record structure and provenance). Total about 44 MB.
- `compare_head.py`: MLX env, read-only. Loads `clefkit_mlx.head.load_head` and compares logits, per-question softmax and top choice with the fixtures.
- `run_cases.py`: runs the torch engine (official `load_release_model` + `systemone`) or the MLX engine (`clefkit_mlx.engine.MlxEngine(model_path, head_dir=...).decide`) over `cases.json` and writes JSONL plus a `.meta.json` sidecar.
- `compare_decisions.py`: compares two JSONL files (agreement, probability drift, KL, changed decisions, accuracy against labels). numpy only.
- `tests/test_compare.py`: unit tests for `compare_decisions` on tiny synthetic files.

## Environments

- PyTorch: any Python with torch, transformers and safetensors that can load the official release. Use `python -B` so no bytecode is written into the model folder.
- MLX: any Python with `mlx`, `mlx-lm`, `transformers` and numpy. Nothing is installed by these scripts.

```sh
export CLEF_MODEL_DIR=/path/to/clef-flash/model     # official release folder (joint_schema_model.py, weights)
export CLEF_CASES=/path/to/cases.json               # synthetic cases file
export CLEFKIT_PYTHON=python3                       # the Python you run the MLX steps with
cd server/mlx
```

## Sequence

Run the steps in order. Never run two models at the same time.

### 1. Head fixtures (PyTorch env)

```sh
python -B parity/make_head_fixtures.py
```

Memory: reads the 244 MB head file only. The backbone is never loaded.

### 2. Head parity (MLX env)

```sh
"$CLEFKIT_PYTHON" -B parity/compare_head.py; echo "exit=$?"
```

Exit 0: all six fixtures pass (top-choice 100%, probability difference at most 1e-3, finite outputs). Exit 1: a fixture fails. Exit 2: `clefkit_mlx.head` is unavailable or the fixtures are missing.

### 3. BF16 torch reference (PyTorch env)

```sh
python -B parity/run_cases.py --engine torch \
    --model-path "$CLEF_MODEL_DIR" --device mps --min-free-gib 22 \
    --out parity/out/bf16-torch.jsonl
```

Memory: the BF16 release is 19.1 GB on disk. The runtime README's start guard for BF16 is 22 GiB free; `--min-free-gib 22` matches it. The port's own BF16 floor is 24 GiB.

### 4. MLX run (MLX env)

Create the MLX q8 folder once, into a new destination (the quantizer refuses an existing folder):

```sh
"$CLEFKIT_PYTHON" -B -m clefkit_mlx.quantize --source "$CLEF_MODEL_DIR" --dest <NEW_MLX_Q8_DIR> --bits 8 --group-size 64
```

Then run the cases:

```sh
"$CLEFKIT_PYTHON" -B parity/run_cases.py --engine mlx --model-path <NEW_MLX_Q8_DIR> \
    --head-dir "$CLEF_MODEL_DIR" --min-free-gib 14 --out parity/out/q8-mlx.jsonl
```

Use `--head-dir` only if the quantized folder lacks the official head files. Memory: 14 GiB free is the floor, matching the port's quantized floor.

### 5. Compare (MLX env)

```sh
"$CLEFKIT_PYTHON" -B parity/compare_decisions.py parity/out/bf16-torch.jsonl parity/out/q8-mlx.jsonl
```

### Checks that need no model

```sh
"$CLEFKIT_PYTHON" -B -m unittest discover -s parity/tests -p 'test_*.py'
"$CLEFKIT_PYTHON" -B parity/run_cases.py --engine torch --dry-run
```

`parity/out/` is created by the runs and should stay out of git.

## Run guards

- `run_cases.py` reads free plus inactive pages from `vm_stat`. It refuses to load below `--min-free-gib` (default 14) and stops between cases below `--stop-free-gib` (default 2). If `vm_stat` cannot be parsed, it refuses. It never overwrites an existing `--out`.
- The guard is a floor, not a guarantee that the run fits. Check `vm_stat` and swap during a BF16 run.

## What has NOT been verified

- No real model has been run. `run_cases.py` was checked only with `--help`, `--dry-run` and a guard refusal (`--min-free-gib 999`, which loads nothing).
- Head parity is checked on synthetic hidden states and a synthetic 512-row embedding table, not the real vocabulary table. Reference compute is float32 from bf16 weights; the runtime runs bf16.
- Decision accuracy, KL and probability drift on the real model are unmeasured. SystemOne answers are rounded to 4 decimals, so drift values are quantised at that level.
- No swap watchdog in `run_cases.py`. The runtime's `supervise.py` has one; `run_cases.py` does not.
- The MLX q8 folder has not been created. The `clefkit_mlx.quantize` step and the `MlxEngine` decide path are untested end to end here.
- Unit tests cover `compare_decisions` only. `compare_head.py` was run once against the current port, and its pass depends on the port's current state.
