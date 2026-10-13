import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import INVOICE, SUPPORT

from clefkit_mlx.backbone import check_weights, hidden_states, load_backbone, output_embedding_weight
from clefkit_mlx.engine import MlxEngine
from clefkit_mlx.quantize import main as quantize_main


def test_hf_layout_maps_to_same_hidden_states(tiny_release):
    root, reference = tiny_release
    info = check_weights(root)
    assert info["lm_head"] == (248320, 64)
    loaded = load_backbone(root, min_available_gib=0)
    ids = mx.array([[1, 50, 9000, 248000, 7, 7, 3, 12]])
    assert mx.allclose(hidden_states(loaded, ids), hidden_states(reference, ids), atol=1e-4).item()


def _check_response(response, request):
    assert response["model"] == "clef-flash"
    assert set(response["answers"]) == set(request["questions"])
    assert response["usage"]["output_tokens"] == 0 and response["usage"]["input_tokens"] > 50
    for qid, answer in response["answers"].items():
        question = request["questions"][qid]
        assert answer["type"] == question["type"]
        if question["type"] == "noul":
            assert 0.0 <= answer["noul"] <= 1.0
        elif question["type"] == "choice":
            assert set(answer["probabilities"]) == set(question["criteria"])
            assert answer["choice"] in question["criteria"]
            assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-3
            assert answer["confidence"] == max(answer["probabilities"].values())
        else:
            levels = [str(i) for i in range(len(question["criteria"]))]
            assert list(answer["probabilities"]) == levels
            assert answer["legend"] == dict(zip(levels, question["criteria"]))
            assert 0.0 <= answer["score"] <= len(levels) - 1


def test_engine_lazy_decide(tiny_release):
    root, _ = tiny_release
    engine = MlxEngine(model_dir=root, min_available_gib=0)
    assert engine.name == "mlx" and engine.model_id == "clef-flash" and not engine.loaded
    for request in (INVOICE, SUPPORT):
        _check_response(engine.decide(request), request)
    assert engine.loaded
    with pytest.raises(ValueError):
        engine.decide({"model": "clef-flash", "state": "x", "questions": {"a": {"type": "choice"}}})
    with pytest.raises(NotImplementedError):
        engine.decide({**INVOICE, "images": ["img"]})


def test_quantize_then_decide(tiny_release, tmp_path):
    root, _ = tiny_release
    dest = tmp_path / "q8"
    assert quantize_main(["--source", str(root), "--dest", str(dest), "--bits", "8",
                          "--group-size", "64", "--min-available-gib", "0"]) == 0
    with pytest.raises(FileExistsError):
        quantize_main(["--source", str(root), "--dest", str(dest), "--bits", "8", "--min-available-gib", "0"])
    config = json.loads((dest / "config.json").read_text())
    assert config["quantization"]["bits"] == 8
    model = load_backbone(dest, min_available_gib=0)
    kinds = {name: type(m) for name, m in model.named_modules()}
    quantized = [n for n, k in kinds.items() if k is nn.QuantizedLinear]
    assert quantized and all(n.startswith("language_model.model.layers.") for n in quantized)
    assert kinds["language_model.lm_head"] is nn.Linear
    assert kinds["language_model.model.embed_tokens"] is nn.Embedding
    assert output_embedding_weight(model).dtype != mx.uint32
    engine = MlxEngine(model_dir=dest, min_available_gib=0)
    _check_response(engine.decide(SUPPORT), SUPPORT)


def test_memory_guard_refuses(tiny_release):
    from clefkit_mlx.backbone import MemoryGuardError

    root, _ = tiny_release
    with pytest.raises(MemoryGuardError):
        load_backbone(root, min_available_gib=10_000)
