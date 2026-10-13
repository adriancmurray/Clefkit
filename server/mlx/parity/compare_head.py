"""Compare clefkit_mlx.head with the PyTorch reference fixtures.

Run in the MLX environment. Read-only: reads fixtures/ and
joint_head.safetensors and writes nothing. Per fixture it prints the maximum absolute logit
difference, the maximum absolute per-question softmax difference and top-choice agreement.

Exit codes: 0 every fixture passes (top-choice agreement 100%, probability difference at most
1e-3, all outputs finite); 1 a fixture fails; 2 clefkit_mlx.head is unavailable or fixtures are missing.

[SIMULATION: SYNTHETIC WORKLOAD] Inputs are seeded random fixtures.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SERVER_MLX_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = SERVER_MLX_DIR / "fixtures"
MODEL_DIR = os.environ.get("CLEF_MODEL_DIR")
MAX_PROBABILITY_DIFFERENCE = 1e-3
NOTICE = "[SIMULATION: SYNTHETIC WORKLOAD]"


class HeadUnavailable(RuntimeError):
    """clefkit_mlx.head cannot be imported or does not expose the expected interface."""


@dataclass(frozen=True)
class Question:
    """Question structure read by the head; attribute names mirror the PyTorch EncodedQuestion."""

    question_id: str
    question_type: int
    question_span: tuple
    option_spans: tuple
    option_ids: tuple


@dataclass(frozen=True)
class Record:
    """Record structure read by the head; attribute names mirror the PyTorch EncodedRecord."""

    record_id: str
    questions: tuple


def records_from_meta(entries):
    """Rebuild head-input record structures from head-meta.json entries."""
    return [
        Record(
            record_id=entry["record_id"],
            questions=tuple(
                Question(
                    question_id=q["question_id"],
                    question_type=q["question_type"],
                    question_span=tuple(q["question_span"]),
                    option_spans=tuple(tuple(span) for span in q["option_spans"]),
                    option_ids=tuple(q["option_ids"]),
                )
                for q in entry["questions"]
            ),
        )
        for entry in entries
    ]


def load_mlx_head():
    """Load clefkit_mlx.head.load_head(model_dir) in float32, matching the fixture reference dtype."""
    try:
        import mlx.core as mx
        from clefkit_mlx.head import load_head
    except ImportError as error:
        raise HeadUnavailable(f"clefkit_mlx.head is not importable ({error})") from error
    try:
        if not MODEL_DIR:
            raise SystemExit("CLEF_MODEL_DIR is not set: set it to the Clef-flash model folder")
        return load_head(Path(MODEL_DIR), dtype=mx.float32)
    except (OSError, ValueError, KeyError) as error:
        raise HeadUnavailable(f"clefkit_mlx.head could not load joint_head weights ({error})") from error


def run_mlx_head(module, fixture, npz, shared_table):
    """Run the MLX head on one fixture; returns per-record lists of per-question logit vectors."""
    import mlx.core as mx

    outputs = module(
        mx.array(npz["hidden_states"].astype(np.float32)),
        mx.array(npz["input_ids"].astype(np.int32)),
        mx.array(npz["attention_mask"].astype(np.int32)),
        records_from_meta(fixture["records"]),
        mx.array(shared_table.astype(np.float32)),
    )
    return [[np.array(vector, dtype=np.float32) for vector in per_record] for per_record in outputs]


def reference_vectors(padded_logits, records_meta):
    """Unpad the stored reference logits into the same nested layout as the MLX output."""
    records = records_from_meta(records_meta)
    return [
        [
            padded_logits[batch_index, question_index, : len(question.option_spans)].astype(np.float32)
            for question_index, question in enumerate(record.questions)
        ]
        for batch_index, record in enumerate(records)
    ]


def softmax(values):
    """Numerically stable softmax in float64."""
    shifted = values.astype(np.float64) - np.max(values)
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def compare_fixture(reference, candidate):
    """Summarise agreement between nested reference and candidate logit lists."""
    summary = {"logit_diff": 0.0, "probability_diff": 0.0, "matches": 0, "questions": 0, "finite": True}
    for ref_record, cand_record in zip(reference, candidate):
        if len(ref_record) != len(cand_record):
            raise ValueError(f"question count {len(cand_record)} != reference {len(ref_record)}")
        for ref, cand in zip(ref_record, cand_record):
            if ref.shape != cand.shape:
                raise ValueError(f"option count {cand.shape} != reference {ref.shape}")
            summary["finite"] = summary["finite"] and bool(np.all(np.isfinite(cand)))
            summary["logit_diff"] = max(summary["logit_diff"], float(np.max(np.abs(ref - cand))))
            probability_gap = np.abs(softmax(ref) - softmax(cand))
            summary["probability_diff"] = max(summary["probability_diff"], float(np.max(probability_gap)))
            summary["matches"] += int(np.argmax(ref) == np.argmax(cand))
            summary["questions"] += 1
    return summary


def fixture_passes(summary):
    """True when agreement is complete, probabilities are within tolerance and outputs are finite."""
    return (
        summary["finite"]
        and summary["matches"] == summary["questions"]
        and summary["probability_diff"] <= MAX_PROBABILITY_DIFFERENCE
    )


def main():
    """Compare every fixture; print one line each and a verdict; return the exit code."""
    meta_path = FIXTURE_DIR / "head-meta.json"
    if not meta_path.exists():
        print(f"fixtures missing: {meta_path}; run parity/make_head_fixtures.py first", file=sys.stderr)
        return 2
    sys.path.insert(0, str(SERVER_MLX_DIR))
    meta = json.loads(meta_path.read_text())
    try:
        module = load_mlx_head()
    except HeadUnavailable as error:
        print(f"HEAD UNAVAILABLE: {error}", file=sys.stderr)
        return 2
    shared_table = np.load(FIXTURE_DIR / meta["shared_file"])["output_embedding_weight"]
    print(NOTICE)
    all_passed = True
    for name, fixture in meta["fixtures"].items():
        npz = np.load(FIXTURE_DIR / fixture["file"])
        reference = reference_vectors(npz["logits"], fixture["records"])
        try:
            candidate = run_mlx_head(module, fixture, npz, shared_table)
            summary = compare_fixture(reference, candidate)
        except Exception as error:
            print(f"{name}: run error {type(error).__name__}: {error}")
            all_passed = False
            continue
        passed = fixture_passes(summary)
        all_passed = all_passed and passed
        print(
            f"{name}: max|dlogit|={summary['logit_diff']:.3e} max|dprob|={summary['probability_diff']:.3e} "
            f"top1={summary['matches']}/{summary['questions']} finite={summary['finite']} "
            f"{'PASS' if passed else 'FAIL'}"
        )
    print(f"VERDICT: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
