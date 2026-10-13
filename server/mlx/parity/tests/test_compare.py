"""Unit tests for compare_decisions using tiny synthetic result files.

Run: python -B -m unittest discover -s parity/tests -p 'test_*.py'
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARITY_DIR))

import compare_decisions as cd  # noqa: E402

CASES = {
    "schemas": {},
    "cases": [
        {"id": "a", "task": "topic", "state": "x", "expected": ["software"]},
        {"id": "b", "task": "topic", "state": "y", "expected": ["finance", "travel"], "ambiguous": True},
        {"id": "c", "task": "tool", "state": "z", "expected": ["not_needed"]},
    ],
}
TOPIC_OPTIONS = ["software", "finance", "travel", "mixed", "unclear"]
TOOL_OPTIONS = ["needed", "not_needed", "clarify"]


def choice(winner, probabilities):
    """A SystemOne choice answer for the given winner and probability map."""
    return {"type": "choice", "choice": winner, "confidence": probabilities[winner], "probabilities": probabilities}


def peaked(winner, options):
    """Probability map with 0.6 on the winner and 0.1 on each other option."""
    return {key: (0.6 if key == winner else 0.1) for key in options}


def write_jsonl(path, rows):
    """Write one JSON object per line."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CompareDecisionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cases_path = self.root / "cases.json"
        self.cases_path.write_text(json.dumps(CASES))
        self.labels = cd.load_labels(self.cases_path)

    def tearDown(self):
        self.tmp.cleanup()

    def compare_rows(self, reference_rows, candidate_rows):
        """Write both files and compare them through the file loaders."""
        ref_path = self.root / "ref.jsonl"
        cand_path = self.root / "cand.jsonl"
        write_jsonl(ref_path, reference_rows)
        write_jsonl(cand_path, candidate_rows)
        return cd.compare(cd.load_results(ref_path), cd.load_results(cand_path), self.labels)

    def base_rows(self):
        """Three cases: a and b answer topic, c answers tool."""
        return [
            {"id": "a", "answers": {"topic": choice("software", peaked("software", TOPIC_OPTIONS))}, "latency_seconds": 0.5},
            {"id": "b", "answers": {"topic": choice("travel", peaked("travel", TOPIC_OPTIONS))}},
            {"id": "c", "answers": {"tool": choice("not_needed", peaked("not_needed", TOOL_OPTIONS))}},
        ]

    def test_identical_results_agree_fully(self):
        report = self.compare_rows(self.base_rows(), self.base_rows())
        self.assertEqual(report["changed"], [])
        self.assertEqual(report["by_question"]["topic"]["agree"], 2)
        self.assertEqual(report["by_question"]["topic"]["n"], 2)
        self.assertTrue(all(value == 0.0 for value in report["by_question"]["topic"]["diffs"]))
        self.assertAlmostEqual(report["by_question"]["topic"]["kls"][0], 0.0, places=9)
        self.assertEqual(report["accuracy"]["candidate"]["groups"]["all"], {"correct": 3, "total": 3})

    def test_flipped_decision_is_listed_and_lowers_agreement(self):
        candidate = self.base_rows()
        candidate[0] = {"id": "a", "answers": {"topic": choice("finance", peaked("finance", TOPIC_OPTIONS))}}
        report = self.compare_rows(self.base_rows(), candidate)
        self.assertEqual(report["changed"], [("a", "topic", "software", "finance")])
        self.assertEqual(report["by_question"]["topic"]["agree"], 1)
        self.assertGreater(report["by_question"]["topic"]["kls"][0], 0.0)
        self.assertEqual(report["accuracy"]["reference"]["groups"]["clear"], {"correct": 2, "total": 2})
        self.assertEqual(report["accuracy"]["candidate"]["groups"]["clear"], {"correct": 1, "total": 2})

    def test_ambiguous_label_accepts_any_listed_option(self):
        report = self.compare_rows(self.base_rows(), self.base_rows())
        self.assertEqual(report["accuracy"]["reference"]["groups"]["ambiguous"], {"correct": 1, "total": 1})
        self.assertEqual(report["accuracy"]["reference"]["not_run"], 0)

    def test_noul_tie_resolves_to_true(self):
        self.assertEqual(cd.decision({"type": "noul", "noul": 0.5}), "true")
        self.assertEqual(cd.decision({"type": "noul", "noul": 0.2}), "false")
        self.assertEqual(cd.distribution({"type": "noul", "noul": 0.2}), {"true": 0.2, "false": 0.8})

    def test_noul_flip_is_detected(self):
        reference = [{"id": "c", "answers": {"tool": {"type": "noul", "noul": 0.9}}}]
        candidate = [{"id": "c", "answers": {"tool": {"type": "noul", "noul": 0.1}}}]
        report = self.compare_rows(reference, candidate)
        self.assertEqual(report["changed"], [("c", "tool", "true", "false")])

    def test_mismatched_option_sets_raise(self):
        candidate = self.base_rows()
        candidate[1] = {"id": "b", "answers": {"topic": choice("travel", {"travel": 0.7, "finance": 0.3})}}
        with self.assertRaises(ValueError):
            self.compare_rows(self.base_rows(), candidate)

    def test_duplicate_case_id_raises(self):
        path = self.root / "dup.jsonl"
        write_jsonl(path, [{"id": "a", "answers": {}}, {"id": "a", "answers": {}}])
        with self.assertRaises(ValueError):
            cd.load_results(path)


if __name__ == "__main__":
    unittest.main()
