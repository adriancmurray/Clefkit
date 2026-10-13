"""MLX port of Cloudflare Clef-flash (Qwen3.5 backbone + JointSchemaHead)."""

import os
from pathlib import Path

# The model folder comes from the CLEF_MODEL_DIR environment variable or an explicit argument; there is no default.
DEFAULT_MODEL_DIR = os.environ.get("CLEF_MODEL_DIR") or None


def resolve_model_dir(model_dir=None) -> Path:
    """The model folder: the argument, else CLEF_MODEL_DIR, else a clear error."""
    chosen = model_dir or DEFAULT_MODEL_DIR
    if not chosen:
        raise RuntimeError("no model folder: pass model_dir (--model-dir) or set CLEF_MODEL_DIR to the Clef-flash model folder")
    return Path(chosen)
