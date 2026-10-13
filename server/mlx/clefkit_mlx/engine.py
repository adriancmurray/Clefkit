"""MlxEngine: SystemOne-compatible decisions from Clef-flash on MLX."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import resolve_model_dir
from .backbone import hidden_states, load_backbone, output_embedding_weight
from .encode import collate_records, encode_record, load_tokenizer, official
from .head import load_head


def validate_request(request: dict[str, Any], question_types: dict[str, int]) -> None:
    """Same checks as the official systemone()."""
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for question_id, question in questions.items():
        if question.get("type") not in question_types:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")
    if request.get("images") or request.get("videos"):
        raise NotImplementedError("clefkit_mlx is text-only; images/videos are not supported yet")


class MlxEngine:
    name = "mlx"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        head_dir: str | Path | None = None,
        model_id: str = "clef-flash",
        head_dtype: mx.Dtype = mx.float32,
        max_length: int = 16384,
        min_available_gib: float | None = None,
        tokenizer: Any = None,
        backbone: Any = None,
        head: Any = None,
    ) -> None:
        self.model_dir = resolve_model_dir(model_dir)
        self.head_dir = Path(head_dir) if head_dir else self.model_dir
        self.model_id = model_id
        self.head_dtype = head_dtype
        self.max_length = max_length
        self.min_available_gib = min_available_gib
        self._tokenizer = tokenizer
        self._backbone = backbone
        self._head = head
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._backbone is not None and self._head is not None and self._tokenizer is not None

    def load(self) -> None:
        with self._lock:
            if self._tokenizer is None:
                self._tokenizer = load_tokenizer(self.model_dir)
            if self._head is None:
                self._head = load_head(self.head_dir, dtype=self.head_dtype)
            if self._backbone is None:
                self._backbone = load_backbone(self.model_dir, self.min_available_gib)

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        spec = official(self.head_dir)
        validate_request(request, spec.QUESTION_TYPES)
        if not self.loaded:
            self.load()
        encoded = encode_record(
            self._tokenizer, request, max_length=self.max_length, model_dir=self.head_dir
        )
        batch = collate_records([encoded], self._tokenizer.pad_token_id)
        hidden = hidden_states(self._backbone, batch["input_ids"])
        logits = self._head(
            hidden,
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            output_embedding_weight(self._backbone),
        )[0]
        questions = request["questions"]
        answers = {}
        for question, question_logits in zip(encoded.questions, logits):
            probabilities = mx.softmax(question_logits.astype(mx.float32), axis=-1).tolist()
            answers[question.question_id] = spec.systemone_answer(
                questions[question.question_id], dict(zip(question.option_ids, probabilities))
            )
        return {
            "model": request["model"],
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }


INVOICE_EXAMPLE = {
    "model": "clef-flash",
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "What is the invoice status?",
            "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
        },
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    },
}


def main(argv: list[str] | None = None) -> int:
    """Run the official invoice example twice (cold, warm) and print latency and peak memory."""
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(prog="python -m clefkit_mlx.engine")
    parser.add_argument("--model-dir", default=None, help="model folder (default: $CLEF_MODEL_DIR)")
    parser.add_argument("--min-available-gib", type=float, default=None)
    args = parser.parse_args(argv)
    engine = MlxEngine(model_dir=args.model_dir, min_available_gib=args.min_available_gib)
    for label in ("cold", "warm"):
        start = time.perf_counter()
        response = engine.decide(INVOICE_EXAMPLE)
        elapsed = time.perf_counter() - start
        print(json.dumps({"run": label, "seconds": round(elapsed, 3),
                          "peak_memory_bytes": mx.get_peak_memory(), "response": response}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
