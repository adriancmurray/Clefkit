"""Run cases100.jsonl against the Workers AI proxy and write one JSONL row per answered case.

Calls are sequential. Each case gets a 30 s timeout and one retry (two attempts). A row holds
{"id", "answers", "latency_ms"}, where latency_ms is the wall time of the attempt that succeeded.
Cases that fail both attempts are not written; they are listed under "failures" in the sidecar
<out>.meta.json. An existing --out file is never overwritten.

[SIMULATION: SYNTHETIC WORKLOAD] applies to the synthetic cases only; the real prompts are not simulated.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PARITY_DIR / "results"
DEFAULT_URL = "http://127.0.0.1:8788/v1/systemone"
MODEL_NAME = "clef-flash"
TIMEOUT_SECONDS = 30
ATTEMPTS = 2


def build_body(case):
    """SystemOne request body for one case."""
    return {"model": MODEL_NAME, "state": case["state"], "questions": case["questions"]}


def post_once(url, body, timeout):
    """POST the body as JSON and return the decoded response object."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def decide_with_retry(url, body, attempts=ATTEMPTS, timeout=TIMEOUT_SECONDS):
    """(answers, latency_ms, error): the first attempt that returns an answers object, else the last error."""
    error = None
    for _ in range(attempts):
        started = time.perf_counter()
        try:
            response = post_once(url, body, timeout)
        except (OSError, ValueError) as exc:  # URLError and timeouts are OSError; bad JSON is ValueError
            error = f"{type(exc).__name__}: {exc}"
            continue
        answers = response.get("answers") if isinstance(response, dict) else None
        if isinstance(answers, dict):
            return answers, (time.perf_counter() - started) * 1000.0, None
        error = "response has no answers object"
    return None, None, error


def load_cases(path):
    """Cases from a cases100 JSONL file, one object per line."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run(cases, url, out, timeout, attempts):
    """Call the proxy for each case in order and write the rows and the sidecar meta."""
    meta_path = out.with_name(out.name + ".meta.json")
    failures = []
    written = 0
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with out.open("w") as handle:
        for case in cases:
            answers, latency_ms, error = decide_with_retry(url, build_body(case), attempts, timeout)
            if answers is None:
                failures.append({"id": case["id"], "error": error})
                continue
            row = {"id": case["id"], "answers": answers, "latency_ms": round(latency_ms, 3)}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            written += 1
    meta = {
        "url": url,
        "timeout_seconds": timeout,
        "attempts": attempts,
        "cases_requested": len(cases),
        "cases_written": written,
        "failures": failures,
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None):
    """Command-line entry: run the proxy over the cases and print the counts."""
    parser = argparse.ArgumentParser(description="Run cases against the Workers AI SystemOne proxy.")
    parser.add_argument("--cases", type=Path, default=RESULTS_DIR / "cases100.jsonl")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "workers-ai.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases; 0 means all")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = run(cases, args.url, args.out, TIMEOUT_SECONDS, ATTEMPTS)
    print(f"requested {meta['cases_requested']}, written {meta['cases_written']}, "
          f"failed {len(meta['failures'])} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
