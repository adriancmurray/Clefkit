"""Record encoding reused verbatim from the official joint_schema_model.py.

joint_schema_model.py imports torch at module level, and the MLX venv has no
torch. Instead of re-typing the encoder, this module parses the official file
and executes only the torch-free definitions (constants, render,
question_options, the Encoded* dataclasses, _tokens, encode_record,
systemone_answer). Token ids are therefore produced by the official code.
Media encoding (_encode_media, which needs torch tensors) is replaced by a stub
that rejects images/videos: this port is text-only.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from . import resolve_model_dir

OFFICIAL_NAMES = frozenset(
    {
        "SYSTEM_PROMPT",
        "IMAGE_PLACEHOLDER",
        "VIDEO_PLACEHOLDER",
        "MEDIA_BATCH_KEYS",
        "MEDIA_TOKEN_KEYS",
        "QUESTION_TYPES",
        "render",
        "question_options",
        "EncodedQuestion",
        "EncodedRecord",
        "_tokens",
        "encode_record",
        "systemone_answer",
    }
)
SAFE_IMPORTS = frozenset({"__future__", "json", "math", "dataclasses", "pathlib", "typing"})


def _text_only_media(processor: Any, record: dict[str, Any]) -> tuple[list[int], None]:
    if record.get("images") or record.get("videos"):
        raise NotImplementedError(
            "clefkit_mlx is text-only: images/videos need the Qwen3.5 vision encoder, "
            "which is not ported yet"
        )
    return [], None


def load_official(model_dir: str | Path | None = None) -> SimpleNamespace:
    """Execute the torch-free parts of <model_dir>/joint_schema_model.py."""
    source_path = resolve_model_dir(model_dir) / "joint_schema_model.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    kept: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "") in SAFE_IMPORTS:
            kept.append(node)
        elif isinstance(node, ast.Import) and all(a.name in SAFE_IMPORTS for a in node.names):
            kept.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in OFFICIAL_NAMES:
            kept.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) and t.id in OFFICIAL_NAMES for t in node.targets
        ):
            kept.append(node)
    # dataclasses resolve string annotations through sys.modules[cls.__module__].
    module = types.ModuleType(f"clefkit_mlx._official_{abs(hash(str(source_path)))}")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(source_path), "exec"), namespace)
    missing = OFFICIAL_NAMES - namespace.keys()
    if missing:
        raise RuntimeError(f"official encoder is missing {sorted(missing)} in {source_path}")
    namespace["_encode_media"] = _text_only_media
    return SimpleNamespace(**{name: namespace[name] for name in OFFICIAL_NAMES | {"_encode_media"}})


_cache: dict[str, SimpleNamespace] = {}


def official(model_dir: str | Path | None = None) -> SimpleNamespace:
    model_dir = resolve_model_dir(model_dir)
    key = str(model_dir.resolve())
    if key not in _cache:
        _cache[key] = load_official(model_dir)
    return _cache[key]


def load_tokenizer(model_dir: str | Path | None = None) -> Any:
    """The same HF tokenizer the official processor wraps (AutoProcessor(...).tokenizer)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(resolve_model_dir(model_dir)))


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 16384,
    max_state_tokens: int | None = None,
    model_dir: str | Path | None = None,
) -> Any:
    """Official encode_record, text-only (processor=None)."""
    return official(model_dir).encode_record(
        tokenizer, record, max_length=max_length, max_state_tokens=max_state_tokens, processor=None
    )


def collate_records(records: list[Any], pad_token_id: int) -> dict[str, Any]:
    """MLX version of the official collate_records (right padding, text-only)."""
    if any(record.media for record in records):
        raise NotImplementedError("clefkit_mlx is text-only")
    maximum_length = max(len(record.input_ids) for record in records)
    ids = [list(r.input_ids) + [pad_token_id] * (maximum_length - len(r.input_ids)) for r in records]
    mask = [[1] * len(r.input_ids) + [0] * (maximum_length - len(r.input_ids)) for r in records]
    return {
        "input_ids": mx.array(ids, dtype=mx.int32),
        "attention_mask": mx.array(mask, dtype=mx.int32),
        "records": records,
        "media": {},
    }
