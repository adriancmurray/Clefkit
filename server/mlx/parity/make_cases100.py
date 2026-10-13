"""Build results/cases100.jsonl: the 30 synthetic cases plus 70 real routing prompts.

Synthetic cases come from runtime/cases.json. Each keeps its schema, its expected labels, and the
acceptable-label set of an ambiguous case. Real cases are the first user message of HaiKrew subagent
transcripts, newest first. Duplicates and prompts under 80 characters are skipped, and each prompt is
cut to 3000 characters. Their questions are the four in plugin/presets/model-routing.json.

Labels come from the newest HaiKrew ledger. A run on a haiku model with verdict "pass" labels the case
model={"haiku"}; other cases carry no label. Such a label only shows that Haiku was sufficient, so
accuracy on real cases is one-sided.

Stdlib only. Refuses to overwrite an existing output file.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PARITY_DIR / "results"
CASES_ENV = os.environ.get("CLEF_CASES")  # a cases.json with the synthetic cases
CASES_PATH = Path(CASES_ENV) if CASES_ENV else None
ROUTING_PATH = PARITY_DIR.parents[2] / "plugin" / "presets" / "model-routing.json"
PROJECTS_DIR = Path.home() / ".claude" / "projects"
LEDGER_DIR = Path.home() / ".claude" / "plugins" / "store"
# Glob under ~/.claude/projects for subagent transcripts, e.g. "-Users-you-myproject*/*/subagents/agent-*.jsonl".
TRANSCRIPT_GLOB = os.environ.get("CLEF_TRANSCRIPT_GLOB")
REAL_COUNT = 70
MIN_PROMPT_CHARS = 80
MAX_PROMPT_CHARS = 3000


def first_user_message(path):
    """(text, timestamp) of a transcript's first user message; text is None when it is not plain text."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "user":
                content = row.get("message", {}).get("content")
                return (content if isinstance(content, str) else None), row.get("timestamp", "")
    return None, ""


def select_prompts(found):
    """Unique usable prompts from (timestamp, agent_id, text) tuples, newest first.

    Returns (prompts, skipped). Duplicates are matched on the full text; the cut to 3000 characters
    happens after the length check, so a long prompt is never skipped for its length.
    """
    prompts, seen = [], set()
    skipped = {"no_text": 0, "short": 0, "duplicate": 0}
    for stamp, agent_id, text in sorted(found, key=lambda item: (item[0], item[1]), reverse=True):
        if text is None:
            skipped["no_text"] += 1
        elif len(text) < MIN_PROMPT_CHARS:
            skipped["short"] += 1
        elif text in seen:
            skipped["duplicate"] += 1
        else:
            seen.add(text)
            prompts.append({"agent_id": agent_id, "timestamp": stamp, "state": text[:MAX_PROMPT_CHARS]})
    return prompts, skipped


def newest_ledger(store_dir):
    """(path, runs) from the newest haikrew_inline-*.json ledger; runs live under 'runs' or 'ledger.runs'."""
    files = sorted(store_dir.glob("haikrew_inline-*.json"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise SystemExit(f"no haikrew_inline ledger under {store_dir}")
    path = files[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    return path, data.get("runs") or data.get("ledger", {}).get("runs") or []


def runs_by_agent(runs):
    """Map each ledger agent_id to its list of runs."""
    grouped = {}
    for run in runs:
        grouped.setdefault(run.get("agent_id"), []).append(run)
    return grouped


def label_for(agent_runs):
    """The model label for one agent's ledger runs, or None when no run passed on Haiku."""
    if any("haiku" in run.get("model", "") and run.get("verdict") == "pass" for run in agent_runs):
        return {"task": "model", "expected": ["haiku"], "ambiguous": False}
    return None


def synthetic_rows(data):
    """One row per runtime/cases.json case, with its schema and labels."""
    rows = []
    for case in data["cases"]:
        task = case["task"]
        rows.append({
            "id": case["id"],
            "source": "synthetic",
            "state": case["state"],
            "questions": {task: data["schemas"][task]},
            "labels": {"task": task, "expected": case["expected"], "ambiguous": bool(case.get("ambiguous", False))},
        })
    return rows


def real_rows(prompts, questions, by_agent):
    """One row per real prompt, with the routing questions and a label when the ledger supports one."""
    rows = []
    for number, prompt in enumerate(prompts, start=1):
        row = {
            "id": f"r{number:02d}",
            "source": "real",
            "agent_id": prompt["agent_id"],
            "state": prompt["state"],
            "questions": questions,
        }
        label = label_for(by_agent.get(prompt["agent_id"], []))
        if label is not None:
            row["labels"] = label
        rows.append(row)
    return rows


def main(argv=None):
    """Write cases100.jsonl and its sidecar meta file."""
    parser = argparse.ArgumentParser(description="Build the 100-case set for the backend comparison.")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "cases100.jsonl")
    args = parser.parse_args(argv)
    if CASES_PATH is None or not TRANSCRIPT_GLOB:
        raise SystemExit("set CLEF_CASES (a cases.json) and CLEF_TRANSCRIPT_GLOB (transcripts under ~/.claude/projects)")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    routing_questions = json.loads(ROUTING_PATH.read_text(encoding="utf-8"))["questions"]
    transcripts = sorted(PROJECTS_DIR.glob(TRANSCRIPT_GLOB))
    found = []
    for path in transcripts:
        text, stamp = first_user_message(path)
        found.append((stamp, path.stem.removeprefix("agent-"), text))
    prompts, skipped = select_prompts(found)
    if len(prompts) < REAL_COUNT:
        raise SystemExit(f"only {len(prompts)} usable prompts, need {REAL_COUNT}")
    chosen = prompts[:REAL_COUNT]

    ledger_path, runs = newest_ledger(LEDGER_DIR)
    grouped = runs_by_agent(runs)
    real = real_rows(chosen, routing_questions, grouped)
    rows = synthetic_rows(data) + real

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta = {
        "synthetic_count": len(rows) - len(real),
        "real_count": len(real),
        "real_labelled": sum(1 for row in real if "labels" in row),
        "ledger": str(ledger_path),
        "ledger_runs_matched": sum(len(grouped.get(prompt["agent_id"], [])) for prompt in chosen),
        "transcripts_scanned": len(transcripts),
        "prompts_available": len(prompts),
        "skipped": skipped,
        "cases_source_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "routing_questions": sorted(routing_questions),
    }
    args.out.with_name(args.out.stem + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(rows)} cases to {args.out}: {meta['synthetic_count']} synthetic, "
          f"{meta['real_count']} real, {meta['real_labelled']} real labelled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
