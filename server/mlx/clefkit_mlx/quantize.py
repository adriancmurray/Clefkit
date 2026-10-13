"""Quantize the Clef-flash backbone Linear layers with MLX affine quantization.

    python -m clefkit_mlx.quantize --source <model dir> --dest <new dir> --bits 8|4 --group-size 64

Only nn.Linear layers inside the decoder blocks (language_model.model.layers.*)
are quantized. Embeddings, lm_head (the head's lexical table), norms, conv1d,
A_log/dt_bias and the joint head stay at their release precision. The source
is loaded lazily and written shard by shard; the destination must not exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model, save_config, save_model

from .backbone import BF16_MIN_GIB, MemoryGuardError, require_memory

COPY_FILES = (
    "joint_head.safetensors",
    "joint_head_config.json",
    "joint_schema_model.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "generation_config.json",
    "LICENSE",
)
LAYER_PREFIX = "language_model.model.layers."


def backbone_linear_predicate(group_size: int):
    def predicate(path: str, module: nn.Module) -> bool:
        return (
            type(module) is nn.Linear
            and path.startswith(LAYER_PREFIX)
            and module.weight.shape[-1] % group_size == 0
        )

    return predicate


def quantize(
    source: Path, dest: Path, bits: int, group_size: int, min_available_gib: float = BF16_MIN_GIB
) -> dict:
    if dest.exists():
        raise FileExistsError(f"{dest} exists; quantize writes only to a new folder")
    if source.resolve() == dest.resolve():
        raise ValueError("source and dest must differ")
    require_memory(min_available_gib, f"quantize {source}")
    model, config = load_model(source, lazy=True, strict=True)
    quantized_paths: list[str] = []
    base = backbone_linear_predicate(group_size)

    def predicate(path: str, module: nn.Module) -> bool:
        keep = base(path, module)
        if keep:
            quantized_paths.append(path)
        return keep

    nn.quantize(model, group_size=group_size, bits=bits, mode="affine", class_predicate=predicate)
    config = dict(config)
    config["quantization"] = {"group_size": group_size, "bits": bits, "mode": "affine"}
    dest.mkdir(parents=True)
    save_model(dest, model, donate_model=True)
    save_config(config, dest / "config.json")
    for name in COPY_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, dest / name)
    summary = {
        "source": str(source),
        "bits": bits,
        "group_size": group_size,
        "quantized_linear_layers": len(quantized_paths),
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    (dest / "clefkit_quantization.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m clefkit_mlx.quantize")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--min-available-gib", type=float, default=BF16_MIN_GIB)
    args = parser.parse_args(argv)
    try:
        summary = quantize(args.source, args.dest, args.bits, args.group_size, args.min_available_gib)
    except MemoryGuardError as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
