"""Encoding parity: clefkit_mlx.encode vs the full official joint_schema_model module.

The official module imports torch; this venv has none, so it is imported here
with a minimal stand-in torch that only satisfies class/decorator definitions.
encode_record itself never touches torch for text-only records.
"""

import importlib.util
import json
import sys
import types

import pytest
from conftest import INVOICE, MODEL_DIR, SUPPORT, needs_release

from clefkit_mlx.encode import collate_records, encode_record

RECORDS = [
    INVOICE,
    SUPPORT,
    {"state": [1, {"b": "ü", "a": None}], "questions": {"q": {"type": "noul", "criteria": {"true": "yes!"}}}},
    {"state": "x" * 50, "questions": {"z": {"type": "choice", "instructions": {"k": 1}, "criteria": {"b": None, "a": "A"}}}},
]


def _official_module():
    fake = types.ModuleType("torch")
    fake.nn = types.ModuleType("torch.nn")
    fake.nn.Module = object
    fake.nn.functional = types.ModuleType("torch.nn.functional")
    fake.inference_mode = lambda *a, **k: (lambda f: f)
    fake.__getattr__ = lambda name: object()  # dtype defaults such as torch.bfloat16
    saved = {k: sys.modules.get(k) for k in ("torch", "torch.nn", "torch.nn.functional")}
    sys.modules.update({"torch": fake, "torch.nn": fake.nn, "torch.nn.functional": fake.nn.functional})
    try:
        spec = importlib.util.spec_from_file_location("official_jsm", MODEL_DIR / "joint_schema_model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["official_jsm"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@needs_release
@pytest.mark.parametrize("index", range(len(RECORDS)))
def test_matches_official(tokenizer, index):
    official = _official_module()
    record = RECORDS[index]
    ours = encode_record(tokenizer, record)
    theirs = official.encode_record(tokenizer, record)
    assert ours.input_ids == theirs.input_ids
    assert [(q.question_id, q.question_type, q.question_span, q.option_spans, q.option_ids) for q in ours.questions] == [
        (q.question_id, q.question_type, q.question_span, q.option_spans, q.option_ids) for q in theirs.questions
    ]


@needs_release
def test_spans_decode_to_schema_text(tokenizer):
    encoded = encode_record(tokenizer, INVOICE)
    status, large = encoded.questions
    decode = lambda span: tokenizer.decode(list(encoded.input_ids[span[0]:span[1]]))  # noqa: E731
    assert decode(status.question_span) == "What is the invoice status?"
    assert status.option_ids == ("draft", "overdue", "paid")
    assert json.loads(decode(status.option_spans[1])) == {"option_id": "overdue", "description": "Invoice is past due."}
    assert large.option_ids == ("true", "false")
    text = tokenizer.decode(list(encoded.input_ids))
    assert text.startswith("<|im_start|>system\n") and text.endswith("JOINT SCHEMA DECISIONS:")


@needs_release
def test_text_only_and_collate(tokenizer):
    with pytest.raises(NotImplementedError):
        encode_record(tokenizer, {**INVOICE, "images": [object()]})
    a, b = encode_record(tokenizer, INVOICE), encode_record(tokenizer, SUPPORT)
    batch = collate_records([a, b], tokenizer.pad_token_id)
    longest = max(len(a.input_ids), len(b.input_ids))
    assert batch["input_ids"].shape == (2, longest)
    assert batch["attention_mask"].sum(axis=1).tolist() == [len(a.input_ids), len(b.input_ids)]
