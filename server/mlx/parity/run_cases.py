"""Run Clef decisions over runtime/cases.json and write one JSON line per case.

Engines: "torch" loads the release model with the official load_release_model and answers with
the official systemone (PyTorch environment). "mlx" constructs clefkit_mlx.engine.MlxEngine(model_path)
and calls decide(request) (MLX environment). Both return SystemOne response dicts.

Memory guard: before loading, free memory (vm_stat free plus inactive pages, the figure psutil
reports as available on macOS) must be at least --min-free-gib. While running, the run stops if
it falls below --stop-free-gib. An existing --out file is never overwritten.

[SIMULATION: SYNTHETIC WORKLOAD] All cases are synthetic.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

NOTICE = "[SIMULATION: SYNTHETIC WORKLOAD]"
MODEL_NAME = "clef-flash"
CASES_ENV = os.environ.get("CLEF_CASES")  # the cases file; --cases overrides it
MLX_SERVER_DIR = Path(__file__).resolve().parents[1]
QUESTION_TYPES = ("noul", "choice", "score")
EXIT_GUARD = 3


class GuardError(RuntimeError):
    """Free memory is below a configured floor, or vm_stat cannot be read."""


def build_request(case, schemas=None):
    """SystemOne request body for one case. A cases100 row carries its own questions; a cases.json case names
    one task whose schema is looked up in schemas."""
    if "questions" in case:
        questions = copy.deepcopy(case["questions"])
    else:
        questions = {case["task"]: copy.deepcopy(schemas[case["task"]])}
    return {"model": MODEL_NAME, "state": case["state"], "questions": questions}


def load_cases(path):
    """(schemas, cases): a cases.json file gives its schemas and cases; a cases100 JSONL file gives one case per line."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return None, [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    return data["schemas"], data["cases"]


def validate_request(request):
    """Apply the same checks the official systemone applies, so a dry run catches bad cases."""
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("model and state are required")
    for question_id, question in request["questions"].items():
        if question.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")


def available_gib():
    """Free plus inactive memory from vm_stat, in GiB."""
    text = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page_size = int(re.search(r"page size of (\d+) bytes", text).group(1))
    pages = {name: int(count) for name, count in re.findall(r"Pages (free|inactive):\s+(\d+)\.", text)}
    return (pages["free"] + pages["inactive"]) * page_size / 2**30


def read_free_gib():
    """available_gib, raising GuardError when vm_stat is unavailable or unparseable (fail closed)."""
    try:
        return available_gib()
    except (OSError, subprocess.SubprocessError, AttributeError, KeyError) as error:
        raise GuardError(f"cannot read vm_stat ({error})") from error


def require_free(floor_gib, label):
    """Return the free GiB, or raise GuardError when it is below the floor."""
    free = read_free_gib()
    if free < floor_gib:
        raise GuardError(f"{label}: {free:.2f} GiB available, floor {floor_gib:g} GiB")
    return free


def open_torch(model_path, device):
    """Load the release model with the official loader; returns request -> SystemOne response."""
    sys.dont_write_bytecode = True  # keeps the model directory free of __pycache__
    sys.path.insert(0, str(model_path))
    import torch
    from joint_schema_model import load_release_model, systemone

    torch.set_num_threads(4)
    model, processor = load_release_model(
        model_path,
        device=device,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    )
    return lambda request: systemone(model, processor, request)


def open_mlx(model_path, head_dir):
    """Construct the MLX engine; returns its decide method. head_dir None means the model directory."""
    sys.path.insert(0, str(MLX_SERVER_DIR))
    from clefkit_mlx.engine import MlxEngine

    return MlxEngine(model_path, head_dir=head_dir).decide


def sha256(path):
    """Hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, requests):
    """Load the engine, run every request in order and write a JSONL line per case."""
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    free_before = require_free(args.min_free_gib, "before load")
    if args.engine == "torch":
        decide = open_torch(args.model_path, args.device)
    else:
        decide = open_mlx(args.model_path, args.head_dir)
    meta_path = args.out.with_name(args.out.name + ".meta.json")
    meta = {
        "notice": NOTICE,
        "engine": args.engine,
        "model_path": str(args.model_path),
        "device": args.device if args.engine == "torch" else None,
        "cases": str(args.cases),
        "cases_sha256": sha256(args.cases),
        "case_count": len(requests),
        "min_free_gib": args.min_free_gib,
        "stop_free_gib": args.stop_free_gib,
        "free_gib_before_load": round(free_before, 3),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    written = 0
    status = "failed"
    try:
        with args.out.open("w") as output:
            for case, request in requests:
                require_free(args.stop_free_gib, f"before case {case['id']}")
                started = time.perf_counter()
                response = decide(request)
                seconds = time.perf_counter() - started
                row = {
                    "id": case["id"],
                    "answers": response["answers"],
                    "input_tokens": response.get("usage", {}).get("input_tokens"),
                    "latency_seconds": seconds,
                    "latency_ms": round(seconds * 1000.0, 3),
                }
                if case.get("source", "synthetic") == "synthetic":
                    row = {"notice": NOTICE, **row}
                if "task" in case:
                    row["task"] = case["task"]
                output.write(json.dumps(row) + "\n")
                output.flush()
                written += 1
        status = "completed"
    finally:
        meta.update(status=status, cases_written=written, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return 0


def parse_args(argv):
    """Command-line options for the runner."""
    parser = argparse.ArgumentParser(description="Run Clef decisions over cases.json.")
    parser.add_argument("--engine", required=True, choices=["torch", "mlx"])
    parser.add_argument("--model-path", type=Path, help="model directory (required unless --dry-run)")
    parser.add_argument("--device", default="mps", help="torch only; default mps")
    parser.add_argument("--head-dir", type=Path, help="mlx only; directory with the official head files if not the model directory")
    parser.add_argument("--out", type=Path, help="JSONL output; <out>.meta.json is written beside it")
    parser.add_argument("--cases", type=Path, default=Path(CASES_ENV) if CASES_ENV else None, help="cases file (default: $CLEF_CASES)")
    parser.add_argument("--min-free-gib", type=float, default=14.0, help="refuse to load below this (default 14)")
    parser.add_argument("--stop-free-gib", type=float, default=2.0, help="stop between cases below this (default 2)")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases; 0 means all")
    parser.add_argument("--dry-run", action="store_true", help="build and validate requests; load nothing")
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point: build and validate requests, then either dry-run or run the engine."""
    args = parse_args(argv)
    if args.cases is None:
        raise SystemExit("no cases file: pass --cases or set CLEF_CASES to a cases.json or cases JSONL file")
    schemas, cases = load_cases(args.cases)
    cases = cases[: args.limit] if args.limit else cases
    requests = [(case, build_request(case, schemas)) for case in cases]
    for _, request in requests:
        validate_request(request)
    if args.dry_run:
        try:
            free = f"{read_free_gib():.2f} GiB"
        except GuardError as error:
            free = f"unavailable ({error})"
        print(NOTICE)
        print(f"dry run: {len(requests)} requests validated from {args.cases}; no model loaded")
        print(f"questions: {dict(Counter(q for _, request in requests for q in request['questions']))}")
        print(f"available memory (informational): {free}")
        print(f"first request: {json.dumps(requests[0][1], ensure_ascii=False)}")
        return 0
    if args.model_path is None or args.out is None:
        raise SystemExit("--model-path and --out are required unless --dry-run")
    try:
        return run(args, requests)
    except GuardError as error:
        print(f"STOPPED: {error}", file=sys.stderr)
        return EXIT_GUARD


if __name__ == "__main__":
    sys.exit(main())
