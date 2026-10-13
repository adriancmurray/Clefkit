import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clefkit_mlx import DEFAULT_MODEL_DIR  # noqa: E402

MODEL_DIR = Path(DEFAULT_MODEL_DIR or "/nonexistent")
needs_release = pytest.mark.skipif(
    not (MODEL_DIR / "joint_schema_model.py").exists(), reason="CLEF_MODEL_DIR not set or release missing"
)

TINY_HEAD = {"hidden_size": 64, "width": 32, "routing_layers": 1, "layers": 2, "heads": 4, "feedforward": 64}
TINY_TEXT = {
    "model_type": "qwen3_5_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,  # gated-delta Metal kernel needs Dk >= 32
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "rms_norm_eps": 1e-6,
    "vocab_size": 248320,
    "tie_word_embeddings": False,
    "attn_output_gate": True,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
}

INVOICE = {
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
SUPPORT = {
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}


@pytest.fixture(scope="session")
def tokenizer():
    if not (MODEL_DIR / "tokenizer.json").exists():
        pytest.skip("Clef-flash release not mounted")
    from clefkit_mlx.encode import load_tokenizer

    return load_tokenizer(MODEL_DIR)


def _hf_format(weights: dict) -> dict:
    """Turn mlx-format qwen3_5 weights back into the HF release layout."""
    out = {}
    norm_suffixes = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight")
    for key, value in weights.items():
        if key == "language_model.lm_head.weight":
            new = "lm_head.weight"
        else:
            new = key.replace("language_model.model.", "model.language_model.")
        if "conv1d.weight" in key:
            value = value.moveaxis(1, 2)  # [C, K, 1] -> [C, 1, K]
        if key.endswith(norm_suffixes) or key == "language_model.model.norm.weight":
            value = value - 1.0
        out[new] = value
    out["model.visual.patch_embed.proj.weight"] = mx.zeros((4, 4))
    return out


@pytest.fixture(scope="session")
def tiny_release(tmp_path_factory):
    """A tiny random Clef release folder in the HF layout, plus the mlx reference model."""
    if not (MODEL_DIR / "joint_schema_model.py").exists():
        pytest.skip("Clef-flash release not mounted")
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5

    from clefkit_mlx.head import JointSchemaHead

    mx.random.seed(0)
    root = tmp_path_factory.mktemp("tiny_release")
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": dict(TINY_TEXT)}))
    # Give norms non-trivial values so the (1 + w) shift is exercised.
    params = dict(tree_flatten(model.parameters()))
    for key, value in params.items():
        if key.endswith("norm.weight"):
            params[key] = 1.0 + 0.1 * mx.random.normal(value.shape)
        elif key.endswith("weight") and value.ndim == 2:
            params[key] = 0.05 * mx.random.normal(value.shape)
    model.load_weights(list(params.items()))
    mx.eval(model.parameters())
    mx.save_safetensors(str(root / "model.safetensors"), _hf_format(params))
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "text_config": TINY_TEXT, "vision_config": {"depth": 1}})
    )
    head = JointSchemaHead(**TINY_HEAD)
    head_params = {k: 0.1 * mx.random.normal(v.shape) for k, v in tree_flatten(head.parameters())}
    mx.save_safetensors(str(root / "joint_head.safetensors"), head_params)
    (root / "joint_head_config.json").write_text(json.dumps(TINY_HEAD))
    for name in ("joint_schema_model.py", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy2(MODEL_DIR / name, root / name)
    return root, model
