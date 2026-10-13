"""Unit tests for the cases100 additions: noul values, source-split labels and accuracy, routing counts,
latency percentiles, prompt selection, ledger labels, the HTTP retry, and the request builder.

Run: python -B -m unittest discover -s parity/tests -p 'test_*.py'
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PARITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARITY_DIR))

import compare_decisions as cd  # noqa: E402
import make_cases100 as mc  # noqa: E402
import run_cases  # noqa: E402
import run_http  # noqa: E402

MODEL_OPTIONS = ["haiku", "sonnet", "opus"]


def peaked(winner, options):
    """Probability map with 0.6 on the winner and 0.2 split between the others."""
    rest = [o for o in options if o != winner]
    probabilities = {winner: 0.6}
    probabilities.update({o: 0.2 / len(rest) for o in rest})
    return probabilities


def choice(winner, options):
    """A SystemOne choice answer."""
    return {"type": "choice", "choice": winner, "probabilities": peaked(winner, options)}


class NoulAndDecisionTests(unittest.TestCase):
    def test_bare_number_is_a_noul_probability(self):
        self.assertEqual(cd.distribution(0.3), {"true": 0.3, "false": 0.7})
        self.assertEqual(cd.decision(0.5), "true")
        self.assertEqual(cd.decision(0.1), "false")

    def test_noul_dict_without_type_is_accepted(self):
        self.assertEqual(cd.decision({"noul": 0.9}), "true")

    def test_noul_without_probability_raises(self):
        with self.assertRaises(ValueError):
            cd.noul_probability({"noul": "high"})
        with self.assertRaises(ValueError):
            cd.noul_probability({"noul": True})


class LabelAndSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = [
            {"id": "c01", "source": "synthetic", "state": "x", "questions": {}, "labels": {"task": "topic", "expected": ["software"], "ambiguous": False}},
            {"id": "c02", "source": "synthetic", "state": "y", "questions": {}, "labels": {"task": "topic", "expected": ["finance", "travel"], "ambiguous": True}},
            {"id": "r01", "source": "real", "state": "z", "questions": {}, "labels": {"task": "model", "expected": ["haiku"], "ambiguous": False}},
            {"id": "r02", "source": "real", "state": "w", "questions": {}},
        ]
        self.cases_path = self.root / "cases100.jsonl"
        self.cases_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_labels_keeps_only_labelled_rows_with_source(self):
        labels = cd.load_labels(self.cases_path)
        self.assertEqual(set(labels), {"c01", "c02", "r01"})
        self.assertEqual(labels["r01"], {"task": "model", "expected": ["haiku"], "ambiguous": False, "source": "real"})

    def test_load_sources_covers_every_row(self):
        self.assertEqual(cd.load_sources(self.cases_path), {"c01": "synthetic", "c02": "synthetic", "r01": "real", "r02": "real"})

    def test_accuracy_splits_by_source(self):
        labels = cd.load_labels(self.cases_path)
        results = {
            "c01": {"topic": choice("software", ["software", "finance"])},
            "c02": {"topic": choice("mixed", ["software", "finance", "travel", "mixed"])},
            "r01": {"model": choice("sonnet", MODEL_OPTIONS)},
        }
        block = cd.accuracy(results, labels)
        self.assertEqual(block["groups"]["synthetic"], {"correct": 1, "total": 2})
        self.assertEqual(block["groups"]["synthetic/clear"], {"correct": 1, "total": 1})
        self.assertEqual(block["groups"]["synthetic/ambiguous"], {"correct": 0, "total": 1})
        self.assertEqual(block["groups"]["real"], {"correct": 0, "total": 1})
        self.assertEqual(block["groups"]["all"], {"correct": 1, "total": 3})
        self.assertEqual(block["not_run"], 0)

    def test_accuracy_counts_unrun_labelled_cases(self):
        block = cd.accuracy({"c01": {"topic": choice("software", ["software"])}}, cd.load_labels(self.cases_path))
        self.assertEqual(block["not_run"], 2)


class RoutingAndLatencyTests(unittest.TestCase):
    def test_route_share_counts_choices_and_missing(self):
        results = {
            "r01": {"model": choice("haiku", MODEL_OPTIONS)},
            "r02": {"model": choice("opus", MODEL_OPTIONS)},
            "r03": {"model": choice("haiku", MODEL_OPTIONS)},
            "r04": {"effort": {"type": "score", "probabilities": {"0": 1.0}}},
        }
        counts, missing = cd.route_share(results, ["r01", "r02", "r03", "r04", "r05"], "model")
        self.assertEqual(counts, {"haiku": 2, "opus": 1})
        self.assertEqual(missing, 2)

    def test_latency_summary_interpolates(self):
        summary = cd.latency_summary([float(v) for v in range(1, 21)])
        self.assertEqual(summary["n"], 20)
        self.assertAlmostEqual(summary["p50"], 10.5)
        self.assertAlmostEqual(summary["p95"], 19.05)

    def test_latency_summary_empty(self):
        self.assertEqual(cd.latency_summary([]), {"n": 0, "p50": None, "p95": None})

    def test_load_latency_ms_accepts_both_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(json.dumps({"id": "a", "answers": {}, "latency_ms": 250.0}) + "\n"
                            + json.dumps({"id": "b", "answers": {}, "latency_seconds": 0.5}) + "\n")
            self.assertEqual(cd.load_latency_ms(path), {"a": 250.0, "b": 500.0})


class PromptSelectionTests(unittest.TestCase):
    def test_skips_short_duplicate_and_missing_text_and_orders_newest_first(self):
        long_text = "x" * 80
        found = [
            ("2026-10-01T00:00:00Z", "old", long_text),
            ("2026-10-03T00:00:00Z", "new", long_text + " new"),
            ("2026-10-02T00:00:00Z", "dupe", long_text),
            ("2026-10-04T00:00:00Z", "short", "too short"),
            ("2026-10-05T00:00:00Z", "none", None),
        ]
        prompts, skipped = mc.select_prompts(found)
        self.assertEqual([p["agent_id"] for p in prompts], ["new", "dupe"])
        self.assertEqual(skipped, {"no_text": 1, "short": 1, "duplicate": 1})

    def test_prompt_is_cut_to_limit_after_length_check(self):
        prompts, _ = mc.select_prompts([("t", "a", "y" * 5000)])
        self.assertEqual(len(prompts[0]["state"]), mc.MAX_PROMPT_CHARS)

    def test_ledger_label_needs_a_haiku_pass(self):
        self.assertEqual(mc.label_for([{"model": "claude-haiku-5-5", "verdict": "pass"}]),
                         {"task": "model", "expected": ["haiku"], "ambiguous": False})
        self.assertIsNone(mc.label_for([{"model": "claude-haiku-5-5", "verdict": "fail"}]))
        self.assertIsNone(mc.label_for([{"model": "claude-sonnet-5-5", "verdict": "pass"}]))
        self.assertIsNone(mc.label_for([]))


class RunnerTests(unittest.TestCase):
    def test_retry_recovers_after_one_failure(self):
        answers = {"model": {"type": "choice", "choice": "haiku"}}
        responses = [OSError("timeout"), {"answers": answers}]
        with mock.patch.object(run_http, "post_once", side_effect=responses) as post:
            got, latency_ms, error = run_http.decide_with_retry("http://x", {}, attempts=2, timeout=1)
        self.assertEqual(got, answers)
        self.assertIsNone(error)
        self.assertEqual(post.call_count, 2)
        self.assertGreaterEqual(latency_ms, 0.0)

    def test_gives_up_after_the_last_attempt(self):
        with mock.patch.object(run_http, "post_once", side_effect=[OSError("refused"), {"no": "answers"}]):
            got, latency_ms, error = run_http.decide_with_retry("http://x", {}, attempts=2, timeout=1)
        self.assertIsNone(got)
        self.assertIsNone(latency_ms)
        self.assertEqual(error, "response has no answers object")

    def test_cases100_row_uses_its_own_questions(self):
        questions = {"model": {"type": "choice", "criteria": {"haiku": "a"}}}
        request = run_cases.build_request({"id": "r01", "state": "s", "questions": questions})
        self.assertEqual(request, {"model": "clef-flash", "state": "s", "questions": questions})
        self.assertIsNot(request["questions"], questions)

    def test_legacy_case_looks_up_its_task_schema(self):
        schemas = {"tool": {"type": "choice", "criteria": {"needed": "n"}}}
        request = run_cases.build_request({"id": "u01", "task": "tool", "state": "s"}, schemas)
        self.assertEqual(request["questions"], {"tool": schemas["tool"]})


if __name__ == "__main__":
    unittest.main()
