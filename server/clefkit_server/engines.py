"""SystemOne engines the HTTP server can wrap: torch (Clef release), an MLX placeholder, and a fake."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

QUESTION_TYPES = ("noul", "choice", "score")
GIB = 1024**3
_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_PAGE_COUNT_RE = re.compile(r"^Pages (free|inactive|speculative):\s+(\d+)\.", re.MULTILINE)


class Engine(Protocol):
    """A SystemOne backend. ``decide`` is not thread-safe; the server calls it under a lock."""

    name: str
    model_id: str
    loaded: bool

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer one SystemOne request body; raise ValueError when the request is invalid."""
        ...


class TorchEngine:
    """Runs a Clef release through Cloudflare's ``systemone`` on torch, loading on first use."""

    name = "torch"

    def __init__(self, model_dir: str | Path, device: str) -> None:
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "joint_schema_model.py").is_file():
            raise RuntimeError(f"{self.model_dir} has no joint_schema_model.py")
        self.device = device
        self.model_id = self.model_dir.name
        self.loaded = False
        self._module: Any = None
        self._model: Any = None
        self._processor: Any = None

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer with ``systemone``; the first call loads the model onto the device."""
        if not self.loaded:
            self._load()
        return self._module.systemone(self._model, self._processor, request)

    def _load(self) -> None:
        """Import the Clef model code from ``model_dir`` and load the release weights."""
        path = str(self.model_dir)
        if path not in sys.path:
            sys.path.insert(0, path)
        # The model folder may be read-only; skip writing __pycache__ into it.
        sys.dont_write_bytecode = True
        self._module = importlib.import_module("joint_schema_model")
        self._model, self._processor = self._module.load_release_model(path, self.device)
        self.loaded = True


class MlxEngine:
    """Placeholder for the MLX port: checks that ``clefkit_mlx`` imports, but has no decide adapter yet."""

    name = "mlx"
    loaded = False

    def __init__(self, model_dir: str | Path) -> None:
        try:
            importlib.import_module("clefkit_mlx")
        except ImportError as exc:
            raise RuntimeError(f"MLX engine unavailable: cannot import clefkit_mlx ({exc})") from exc
        self.model_dir = Path(model_dir)
        self.model_id = self.model_dir.name

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        """Always fails until the MLX port is wired in."""
        # ponytail: placeholder until clefkit_mlx exposes a systemone-style call; wire it here.
        raise RuntimeError("MLX engine has no decide() adapter yet")


class FakeEngine:
    """Deterministic SystemOne stand-in for tests: no weights, fixed answers."""

    name = "fake"
    model_id = "fake-systemone"
    loaded = True

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer each question with a fixed value: first option by id at 0.9, middle level, noul 0.8.

        ``usage`` is omitted: the fake counts no tokens, and a made-up count would look real.
        """
        questions = validate_request(request)
        return {
            "model": request["model"],
            "answers": {qid: _fake_answer(question) for qid, question in questions.items()},
        }


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    """Return the request's questions after the shape checks ``systemone`` makes; ValueError otherwise."""
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for qid, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{qid}: type must be noul, choice, or score")
        _check_criteria(qid, question)
    return questions


def _check_criteria(qid: str, question: dict[str, Any]) -> None:
    """Raise ValueError unless the criteria have the shape the official model reads for the type."""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria and not isinstance(criteria, dict):
            raise ValueError(f"{qid}: noul criteria must be an object")
    elif kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"{qid}: choice criteria must be a non-empty object of option id to description")
    elif not isinstance(criteria, list) or not criteria:
        raise ValueError(f"{qid}: score criteria must be a non-empty list")


def _fake_answer(question: dict[str, Any]) -> dict[str, Any]:
    """Build the fixed answer for one question, shaped like ``systemone_answer``."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": 0.8}
    if kind == "choice":
        return _fake_choice(question["criteria"])
    return _fake_score(question["criteria"])


def _fake_choice(criteria: dict[str, Any]) -> dict[str, Any]:
    """Pick the first option by id, as the official model sorts them; probabilities sum to 1."""
    options = sorted(criteria)
    others = len(options) - 1
    confidence = 0.9 if others else 1.0
    probabilities = {
        option: confidence if index == 0 else (1.0 - confidence) / others
        for index, option in enumerate(options)
    }
    return {
        "type": "choice",
        "choice": options[0],
        "confidence": confidence,
        "probabilities": probabilities,
    }


def _fake_score(criteria: list[Any]) -> dict[str, Any]:
    """Put all probability on the middle level, so the score is that level's index."""
    middle = (len(criteria) - 1) // 2
    levels = [str(index) for index in range(len(criteria))]
    return {
        "type": "score",
        "score": float(middle),
        "confidence": 1.0,
        "legend": dict(zip(levels, criteria)),
        "probabilities": {level: 1.0 if index == middle else 0.0 for index, level in enumerate(levels)},
    }


def free_gib_from_vm_stat(text: str) -> float:
    """Return free + inactive + speculative memory in GiB from ``vm_stat`` output."""
    page = _PAGE_SIZE_RE.search(text)
    if page is None:
        raise ValueError("vm_stat output has no page size line")
    counts = {name: int(count) for name, count in _PAGE_COUNT_RE.findall(text)}
    missing = {"free", "inactive", "speculative"} - set(counts)
    if missing:
        raise ValueError(f"vm_stat output lacks: {', '.join(sorted(missing))}")
    return sum(counts.values()) * int(page.group(1)) / GIB


# ponytail: ceiling is one check at startup, but the torch load is lazy (first decide), so memory can
# fall in between. Upgrade: re-run this check at the start of TorchEngine._load.
def check_torch_memory(min_free_gib: float) -> float | None:
    """Raise RuntimeError when free memory is under ``min_free_gib``; return GiB free, or None off macOS."""
    if sys.platform != "darwin":
        return None
    try:
        output = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True, timeout=10
        ).stdout
        free = free_gib_from_vm_stat(output)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"cannot measure free memory with vm_stat: {exc}") from exc
    if free < min_free_gib:
        raise RuntimeError(
            f"only {free:.1f} GiB free (free + inactive + speculative); "
            f"the torch engine needs {min_free_gib:g} GiB (--min-free-gib)"
        )
    return free
