"""Qwen3.5 text backbone via mlx_lm's qwen3_5 model class, returning final hidden states.

The release checkpoint is Qwen3_5ForConditionalGeneration (text + vision).
mlx_lm's qwen3_5.Model.sanitize drops the vision tower, renames
``model.language_model.*`` -> ``language_model.model.*``, transposes conv1d
and applies the (1 + weight) RMSNorm shift. The head needs the post-norm
hidden states (``language_model.model(...)``) and the lm_head rows (the
official ``get_output_embeddings().weight``), not logits.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model

from . import resolve_model_dir

BF16_MIN_GIB = 24.0
QUANTIZED_MIN_GIB = 14.0


class MemoryGuardError(RuntimeError):
    pass


def available_gib() -> float:
    """free + inactive + speculative pages from vm_stat, in GiB."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"{label}:\s+(\d+)\.", out)
        pages += int(match.group(1)) if match else 0
    return pages * page / 2**30


def is_quantized(model_dir: str | Path) -> bool:
    config = json.loads((Path(model_dir) / "config.json").read_text())
    return "quantization" in config


def require_memory(minimum_gib: float, what: str) -> float:
    available = available_gib()
    if available < minimum_gib:
        raise MemoryGuardError(
            f"{what}: {available:.2f} GiB available (free+inactive+speculative), "
            f"guard requires {minimum_gib:.0f} GiB. Not loading."
        )
    return available


def check_weights(model_dir: str | Path | None = None) -> dict[str, Any]:
    """Key/shape validation without reading tensor data.

    mx.load is lazy and load_weights(strict=True) compares keys and shapes, so
    this builds the full model graph but evaluates nothing.
    """
    before = mx.get_active_memory()
    model, _ = load_model(resolve_model_dir(model_dir), lazy=True, strict=True)
    params = dict(_flatten(model.parameters()))
    active_delta = mx.get_active_memory() - before
    if active_delta > 64 * 2**20:
        raise RuntimeError(f"lazy check materialized {active_delta} bytes")
    return {
        "parameters": len(params),
        "elements": sum(v.size for v in params.values()),
        "active_bytes_delta": active_delta,
        "lm_head": tuple(model.language_model.lm_head.weight.shape)
        if hasattr(model.language_model, "lm_head")
        else None,
    }


def _flatten(tree: Any) -> list[tuple[str, mx.array]]:
    from mlx.utils import tree_flatten

    return tree_flatten(tree)


def load_backbone(
    model_dir: str | Path | None = None, min_available_gib: float | None = None
) -> nn.Module:
    """Load the text backbone (BF16 release or an MLX-quantized folder), after the memory guard."""
    model_dir = resolve_model_dir(model_dir)
    if min_available_gib is None:
        min_available_gib = QUANTIZED_MIN_GIB if is_quantized(model_dir) else BF16_MIN_GIB
    require_memory(min_available_gib, f"load {model_dir}")
    model, _ = load_model(Path(model_dir), lazy=False, strict=True)
    model.eval()
    return model


def hidden_states(model: nn.Module, input_ids: mx.array) -> mx.array:
    """Final (post-RMSNorm) hidden states, shape (batch, length, hidden).

    No attention mask is needed for right-padded batches: attention is causal
    and the linear-attention recurrence runs forward, so pad tokens after a
    sequence never influence its valid positions.
    """
    return model.language_model.model(input_ids)


def output_embedding_weight(model: nn.Module) -> mx.array:
    text = model.language_model
    if text.args.tie_word_embeddings:
        layer = text.model.embed_tokens
    else:
        layer = text.lm_head
    if isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        raise ValueError("output embedding is quantized; the joint head needs it unquantized")
    return layer.weight
