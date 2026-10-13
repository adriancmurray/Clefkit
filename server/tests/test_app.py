"""Unit tests for the SystemOne HTTP server, run against FakeEngine (no model, no packages)."""

import copy
import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clefkit_server.app import MAX_BODY_BYTES, make_server, require_loopback  # noqa: E402
from clefkit_server.engines import FakeEngine, free_gib_from_vm_stat  # noqa: E402

# Question shapes follow joint_schema_model.py: choice criteria are an object, score criteria a list.
REQUEST = {
    "model": "clef-flash",
    "state": {"text": "hello"},
    "questions": {
        "friendly": {"type": "noul", "instructions": "Is it friendly?"},
        "mood": {
            "type": "choice",
            "instructions": "Pick the mood.",
            "criteria": {"happy": "Upbeat.", "sad": "Down.", "angry": "Heated."},
        },
        "rating": {"type": "score", "instructions": "Rate it.", "criteria": ["low", "mid", "high"]},
    },
}


def _start(engine, host="127.0.0.1"):
    """Serve ``engine`` on an ephemeral port of ``host`` in a daemon thread."""
    server = make_server(engine, host, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _call(port, method, path, body=None, headers=None, host="127.0.0.1"):
    """Send one request and return (status, parsed JSON body)."""
    conn = HTTPConnection(host, port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


class ServerTests(unittest.TestCase):
    """Routes and decisions, served by FakeEngine."""

    @classmethod
    def setUpClass(cls):
        cls.server = _start(FakeEngine())
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_health(self):
        status, body = _call(self.port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(
            body, {"ok": True, "backend": "fake", "model": "fake-systemone", "loaded": True}
        )

    def test_valid_decision(self):
        status, body = _call(
            self.port,
            "POST",
            "/v1/systemone",
            body=json.dumps(REQUEST).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "clef-flash")
        answers = body["answers"]
        self.assertEqual(answers["friendly"], {"type": "noul", "noul": 0.8})
        # Choice options are ordered by id, so "angry" comes first.
        self.assertEqual(answers["mood"]["choice"], "angry")
        self.assertEqual(answers["mood"]["confidence"], 0.9)
        self.assertAlmostEqual(sum(answers["mood"]["probabilities"].values()), 1.0)
        self.assertEqual(answers["rating"]["score"], 1.0)
        self.assertEqual(answers["rating"]["legend"], {"0": "low", "1": "mid", "2": "high"})

    def test_invalid_json_is_400(self):
        status, body = _call(self.port, "POST", "/v1/systemone", body=b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("invalid JSON", body["error"])

    def test_non_object_body_is_400(self):
        status, body = _call(self.port, "POST", "/v1/systemone", body=b"[1, 2]")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "body must be a JSON object")

    def test_deeply_nested_json_is_400_and_server_survives(self):
        # About 2 MB: under the body limit, but deep enough to overflow the parser's stack.
        status, body = _call(self.port, "POST", "/v1/systemone", body=b"[" * 1_000_000 + b"]" * 1_000_000)
        self.assertEqual(status, 400)
        self.assertIn("invalid JSON", body["error"])
        self.assertEqual(_call(self.port, "GET", "/health")[0], 200)

    def test_list_choice_criteria_is_400(self):
        request = copy.deepcopy(REQUEST)
        request["questions"]["mood"]["criteria"] = ["happy", "sad"]
        status, body = _call(
            self.port, "POST", "/v1/systemone", body=json.dumps(request).encode("utf-8")
        )
        self.assertEqual(status, 400)
        self.assertIn("mood", body["error"])

    def test_engine_value_error_is_400(self):
        status, body = _call(
            self.port, "POST", "/v1/systemone", body=json.dumps({"model": "m"}).encode("utf-8")
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "model and state are required")

    def test_oversize_is_413(self):
        # Only the declared length is sent; the server must refuse before reading a body.
        status, body = _call(
            self.port,
            "POST",
            "/v1/systemone",
            headers={"Content-Length": str(MAX_BODY_BYTES + 1)},
        )
        self.assertEqual(status, 413)
        self.assertIn("error", body)

    def test_unknown_path_is_404(self):
        self.assertEqual(_call(self.port, "GET", "/nope")[0], 404)
        self.assertEqual(_call(self.port, "POST", "/v1/other")[0], 404)


class Ipv6Tests(unittest.TestCase):
    """The server binds IPv6 loopback when the host is an IPv6 literal."""

    def test_ipv6_loopback_serves_health(self):
        try:
            server = _start(FakeEngine(), "::1")
        except OSError as exc:
            self.skipTest(f"no IPv6 loopback here: {exc}")
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        status, body = _call(server.server_address[1], "GET", "/health", host="::1")
        self.assertEqual(status, 200)
        self.assertEqual(body["backend"], "fake")


class FakeAnswerTests(unittest.TestCase):
    """Fixed answers keep the official criteria shapes and sum to one."""

    def _decide(self, question):
        request = {"model": "m", "state": {}, "questions": {"q": question}}
        return FakeEngine().decide(request)["answers"]["q"]

    def test_single_option_choice_has_confidence_one(self):
        answer = self._decide({"type": "choice", "criteria": {"only": "The only option."}})
        self.assertEqual(answer["choice"], "only")
        self.assertEqual(answer["confidence"], 1.0)
        self.assertEqual(answer["probabilities"], {"only": 1.0})

    def test_choice_probabilities_sum_to_one(self):
        for size in (2, 3, 4, 7):
            with self.subTest(size=size):
                criteria = {f"opt{index}": "d" for index in range(size)}
                answer = self._decide({"type": "choice", "criteria": criteria})
                self.assertEqual(answer["choice"], "opt0")
                self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)

    def test_score_puts_all_probability_on_middle_level(self):
        answer = self._decide({"type": "score", "criteria": ["a", "b", "c", "d"]})
        self.assertEqual(answer["score"], 1.0)
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)

    def test_wrong_criteria_shapes_raise_value_error(self):
        bad_questions = [
            {"type": "choice", "criteria": ["happy", "sad"]},
            {"type": "choice", "criteria": {}},
            {"type": "score", "criteria": {"low": "x"}},
            {"type": "noul", "criteria": ["true"]},
        ]
        for question in bad_questions:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    self._decide(question)


class EngineFailureTests(unittest.TestCase):
    """An unexpected engine exception becomes a generic 500."""

    class _BrokenEngine(FakeEngine):
        def decide(self, request):
            raise RuntimeError("boom")

    def test_unexpected_error_is_500(self):
        server = _start(self._BrokenEngine())
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        status, body = _call(
            server.server_address[1],
            "POST",
            "/v1/systemone",
            body=json.dumps(REQUEST).encode("utf-8"),
        )
        self.assertEqual(status, 500)
        self.assertEqual(body, {"error": "internal error"})


class GuardTests(unittest.TestCase):
    """Bind guard and vm_stat parsing, called directly."""

    def test_non_loopback_refused_without_allow_remote(self):
        for host in ("0.0.0.0", "192.168.1.5", "::", "example.com", ""):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    require_loopback(host, allow_remote=False)

    def test_loopback_and_allow_remote_pass(self):
        for host in ("127.0.0.1", "localhost", "::1"):
            with self.subTest(host=host):
                require_loopback(host, allow_remote=False)
        require_loopback("0.0.0.0", allow_remote=True)

    def test_vm_stat_free_memory(self):
        sample = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                               1000.\n"
            "Pages active:                             5000.\n"
            "Pages inactive:                           2000.\n"
            "Pages speculative:                         500.\n"
        )
        self.assertAlmostEqual(free_gib_from_vm_stat(sample), 3500 * 16384 / 1024**3)

    def test_vm_stat_missing_counter_raises(self):
        with self.assertRaises(ValueError):
            free_gib_from_vm_stat("Mach Virtual Memory Statistics: (page size of 4096 bytes)\n")


if __name__ == "__main__":
    unittest.main()
