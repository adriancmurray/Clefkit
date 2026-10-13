"""Build PyTorch reference fixtures for the MLX JointSchemaHead port.

Run in the Clef PyTorch environment (runtime/.venv). Loads only joint_head.safetensors through
the official JointSchemaHead class, runs it in float32 on six seeded synthetic batches, and writes
into server/mlx/fixtures/: head-00.npz .. head-05.npz (hidden states, token ids, attention mask,
reference logits), head-shared.npz (a synthetic output-embedding table) and head-meta.json
(record and option structure, provenance). The 9B backbone is never loaded.

[SIMULATION: SYNTHETIC WORKLOAD] Hidden states, token ids and the embedding table are random.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(os.environ["CLEF_MODEL_DIR"]) if os.environ.get("CLEF_MODEL_DIR") else None
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
NOTICE = "[SIMULATION: SYNTHETIC WORKLOAD]"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
HIDDEN_SIZE = 4096
VOCAB_SIZE = 512
SEED = 20261012
SIZE_BUDGET_BYTES = 50 * 1024 * 1024
OUTLIER_DIMENSIONS = 8
PREFIX_TOKENS = 40
STATE_TOKENS = (8, 120)
SCHEMA_HEADER_TOKENS = 5
FIELD_HEADER_TOKENS = 6
INSTRUCTION_TOKENS = (3, 12)
ALLOWED_HEADER_TOKENS = 3
OPTION_HEADER_TOKENS = 3
OPTION_TOKENS = (4, 14)
END_FIELD_TOKENS = 3
SUFFIX_TOKENS = 12
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}

# Each record is a list of (question type, option count); noul always has two options.
PLANS = {
    "head-00": [[("choice", 3), ("noul", 2), ("score", 5)]],
    "head-01": [[("choice", 2), ("choice", 8)], [("noul", 2), ("score", 7), ("choice", 4)]],
    "head-02": [
        [("score", 2)],
        [("choice", 6), ("choice", 3), ("noul", 2), ("score", 8)],
        [("choice", 5)],
    ],
    "head-03": [[("noul", 2)]],
    "head-04": [[("choice", 8), ("score", 8), ("noul", 2), ("choice", 2)], [("score", 4), ("choice", 3)]],
    "head-05": [
        [("choice", 2), ("choice", 2)],
        [("choice", 7)],
        [("score", 6), ("noul", 2), ("choice", 8), ("choice", 5)],
    ],
}


def _option_ids(question_type, count):
    """Option ids in the order the official encoder lists them."""
    if question_type == "noul":
        return ["true", "false"]
    if question_type == "choice":
        return [chr(ord("a") + index) for index in range(count)]
    return [str(index) for index in range(count)]


def _lay_out_record(rng, plan):
    """Token spans for one record laid out as prefix, state, schema fields and suffix.

    Returns the record length and one dict per question with its spans and option ids.
    """
    position = PREFIX_TOKENS + int(rng.integers(*STATE_TOKENS)) + SCHEMA_HEADER_TOKENS
    fields = []
    for field_index, (question_type, count) in enumerate(plan):
        position += FIELD_HEADER_TOKENS
        question_span = [position, position + int(rng.integers(*INSTRUCTION_TOKENS))]
        position = question_span[1] + ALLOWED_HEADER_TOKENS
        option_spans = []
        for _ in range(count):
            position += OPTION_HEADER_TOKENS
            start = position
            position += int(rng.integers(*OPTION_TOKENS))
            option_spans.append([start, position])
            position += 1
        position += END_FIELD_TOKENS
        fields.append(
            {
                "question_id": f"field_{field_index}",
                "question_type": QUESTION_TYPES[question_type],
                "question_span": question_span,
                "option_spans": option_spans,
                "option_ids": _option_ids(question_type, count),
            }
        )
    return position + SUFFIX_TOKENS, fields


def _build_records(name, plan, rng):
    """Metadata for every record of one fixture; lengths follow from the layout."""
    records = []
    for index, record_plan in enumerate(plan):
        length, fields = _lay_out_record(rng, record_plan)
        records.append({"record_id": f"{name}-r{index}", "length": length, "questions": fields})
    return records


def _batch_arrays(records, rng):
    """Right-padded hidden states (float16 storage), token ids and attention mask."""
    batch = len(records)
    longest = max(record["length"] for record in records)
    hidden = rng.standard_normal((batch, longest, HIDDEN_SIZE), dtype=np.float32)
    hidden[..., rng.choice(HIDDEN_SIZE, size=OUTLIER_DIMENSIONS, replace=False)] *= 20.0
    input_ids = rng.integers(1, VOCAB_SIZE, size=(batch, longest), dtype=np.int64)
    mask = np.zeros((batch, longest), dtype=np.int64)
    for index, record in enumerate(records):
        mask[index, : record["length"]] = 1
        input_ids[index, record["length"] :] = 0
    return hidden.astype(np.float16), input_ids, mask


def _encode_records(records, input_ids, encoded_record, encoded_question):
    """Official EncodedRecord objects, the structure the head's forward reads."""
    return [
        encoded_record(
            input_ids=tuple(int(token) for token in input_ids[batch_index, : record["length"]]),
            questions=tuple(
                encoded_question(
                    question_id=q["question_id"],
                    question_type=q["question_type"],
                    question_span=tuple(q["question_span"]),
                    option_spans=tuple(tuple(span) for span in q["option_spans"]),
                    option_ids=tuple(q["option_ids"]),
                )
                for q in record["questions"]
            ),
            record_id=record["record_id"],
        )
        for batch_index, record in enumerate(records)
    ]


def _pack_logits(outputs, records):
    """Pad per-question logits into (records, questions, options) with NaN for absent entries."""
    questions = max(len(record["questions"]) for record in records)
    options = max(len(q["option_spans"]) for record in records for q in record["questions"])
    packed = np.full((len(records), questions, options), np.nan, dtype=np.float32)
    for batch_index, per_question in enumerate(outputs):
        for question_index, logits in enumerate(per_question):
            values = logits.detach().cpu().numpy()
            packed[batch_index, question_index, : values.shape[0]] = values
    return packed


def _run_fixture(head, torch, classes, index, name, plan, table_tensor):
    """Build one fixture, run the reference head, write its npz and return its meta entry."""
    rng = np.random.default_rng(SEED + index + 1)
    records = _build_records(name, plan, rng)
    hidden, input_ids, mask = _batch_arrays(records, rng)
    encoded = _encode_records(records, input_ids, *classes)
    with torch.inference_mode():
        outputs = head(
            torch.from_numpy(hidden.astype(np.float32)),
            torch.from_numpy(input_ids),
            torch.from_numpy(mask),
            encoded,
            table_tensor,
        )
    logits = _pack_logits(outputs, records)
    np.savez(FIXTURE_DIR / f"{name}.npz", hidden_states=hidden, input_ids=input_ids, attention_mask=mask, logits=logits)
    expected = sum(len(q["option_spans"]) for record in records for q in record["questions"])
    finite = int(np.isfinite(logits).sum())
    print(
        f"{name}: records={len(records)} questions={sum(len(r['questions']) for r in records)} "
        f"padded_length={hidden.shape[1]} finite_logits={finite}/{expected} "
        f"max|logit|={float(np.nanmax(np.abs(logits))):.3f}"
    )
    if finite != expected:
        raise SystemExit(f"{name}: non-finite reference logits")
    return {"file": f"{name}.npz", "records": records}


def _sha256(path):
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_meta(config, weights_path, torch, fixtures):
    """Write head-meta.json with record structure and provenance."""
    meta = {
        "notice": NOTICE,
        "source": f"Cloudflare/clef-flash joint_head.safetensors, revision {REVISION}",
        "head_sha256": _sha256(weights_path),
        "head_config": config,
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "seed": SEED,
        "reference_dtype": "float32 compute; bf16 weights upcast exactly; hidden states stored float16",
        "shared_file": "head-shared.npz",
        "embedding_note": "synthetic table; only rows gathered by option token ids are read by the head",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "fixtures": fixtures,
    }
    (FIXTURE_DIR / "head-meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _check_budget():
    """Fail when the fixture files exceed the size budget."""
    total = sum(path.stat().st_size for path in FIXTURE_DIR.glob("head-*.npz"))
    print(f"fixture bytes: {total} (budget {SIZE_BUDGET_BYTES})")
    if total > SIZE_BUDGET_BYTES:
        raise SystemExit("fixtures exceed the 50 MB budget")


def main():
    """Write every fixture file and head-meta.json; print a per-fixture summary."""
    if MODEL_DIR is None:
        raise SystemExit("CLEF_MODEL_DIR is not set: set it to the Clef-flash model folder")
    sys.dont_write_bytecode = True  # keeps the model directory free of __pycache__
    sys.path.insert(0, str(MODEL_DIR))
    import torch
    from joint_schema_model import EncodedQuestion, EncodedRecord, JointSchemaHead
    from safetensors.torch import load_file

    weights_path = MODEL_DIR / "joint_head.safetensors"
    config = json.loads((MODEL_DIR / "joint_head_config.json").read_text())
    head = JointSchemaHead(**config)
    head.load_state_dict(load_file(weights_path), strict=True)
    head.eval()
    parameters = sum(parameter.numel() for parameter in head.parameters())
    print(NOTICE)
    print(f"loaded JointSchemaHead with {parameters} parameters from joint_head.safetensors (float32 compute)")

    table = (np.random.default_rng(SEED).standard_normal((VOCAB_SIZE, HIDDEN_SIZE)) * 0.02).astype(np.float16)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE_DIR / "head-shared.npz", output_embedding_weight=table)
    table_tensor = torch.from_numpy(table.astype(np.float32))

    fixtures = {
        name: _run_fixture(head, torch, (EncodedRecord, EncodedQuestion), index, name, plan, table_tensor)
        for index, (name, plan) in enumerate(PLANS.items())
    }
    _write_meta(config, weights_path, torch, fixtures)
    _check_budget()


if __name__ == "__main__":
    main()
