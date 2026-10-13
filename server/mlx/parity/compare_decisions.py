"""Compare per-case decisions from two backends (for example Workers AI and MLX q8).

Input: JSONL result files with one object per case, {"id": str, "answers": {question_id: answer}}.
Each answer is a SystemOne answer (choice, score or noul). Latency fields are read only for the
latency summary. Labels and sources come from results/cases100.jsonl; a legacy runtime/cases.json
is also accepted, and its cases count as synthetic.

Reports per-question top-choice agreement, mean and max absolute probability drift, KL divergence
KL(reference || candidate) with probabilities floored at 1e-12 and renormalised, the cases whose
decision changed, accuracy against the labels per source, the routing choices over real cases,
and latency percentiles. The Markdown report is written with --out.

[SIMULATION: SYNTHETIC WORKLOAD] applies to the synthetic cases only.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

NOTICE = "[SIMULATION: SYNTHETIC WORKLOAD]"
FLOOR = 1e-12
PARITY_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = PARITY_DIR / "results" / "cases100.jsonl"
ROUTE_ORDER = ("haiku", "sonnet", "opus")


def read_jsonl(path):
    """Parsed objects of a JSONL file, skipping blank lines."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_results(path):
    """Map each case id to its answers from a results JSONL file."""
    results = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row["id"])
        if case_id in results:
            raise ValueError(f"{path}:{number}: duplicate case id {case_id!r}")
        if not isinstance(row.get("answers"), dict):
            raise ValueError(f"{path}:{number}: 'answers' must be an object")
        results[case_id] = row["answers"]
    return results


def load_labels(path):
    """Map each labelled case id to {"task", "expected", "ambiguous", "source"}.

    A cases100 JSONL file gives labels on the rows that have them; a legacy cases.json gives them on every case.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        return {
            str(row["id"]): {**row["labels"], "source": row.get("source", "synthetic")}
            for row in read_jsonl(path)
            if "labels" in row
        }
    data = json.loads(path.read_text())
    return {
        str(case["id"]): {
            "task": case["task"],
            "expected": case["expected"],
            "ambiguous": bool(case.get("ambiguous", False)),
            "source": "synthetic",
        }
        for case in data["cases"]
    }


def load_sources(path):
    """Map each case id to its source, "synthetic" or "real"; a legacy cases.json is all synthetic."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return {str(row["id"]): row.get("source", "synthetic") for row in read_jsonl(path)}
    return {str(case["id"]): "synthetic" for case in json.loads(path.read_text())["cases"]}


def load_latency_ms(path):
    """Map each case id to its latency in ms: latency_ms, or latency_seconds converted for older rows."""
    latencies = {}
    for row in read_jsonl(path):
        if "latency_ms" in row:
            latencies[str(row["id"])] = float(row["latency_ms"])
        elif "latency_seconds" in row:
            latencies[str(row["id"])] = 1000.0 * float(row["latency_seconds"])
    return latencies


def is_noul(answer):
    """True for a noul answer: {"type": "noul"} or {"noul": p}, or a bare probability."""
    if isinstance(answer, dict):
        return answer.get("type") == "noul" or "noul" in answer
    return isinstance(answer, (int, float))


def noul_probability(answer):
    """P(true) from a noul answer, given as {"noul": p} or as a bare number."""
    value = answer["noul"] if isinstance(answer, dict) else answer
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"noul answer carries no probability: {answer!r}")
    return float(value)


def distribution(answer):
    """Per-option probabilities of one answer: the probabilities map of a choice or score, or true/false for noul."""
    if isinstance(answer, dict) and "probabilities" in answer:
        return {str(key): float(value) for key, value in answer["probabilities"].items()}
    if is_noul(answer):
        probability = noul_probability(answer)
        return {"true": probability, "false": 1.0 - probability}
    raise ValueError(f"answer carries no probabilities: {answer!r}")


def decision(answer):
    """The option a SystemOne answer selects; a tie resolves to the first listed option."""
    if isinstance(answer, dict) and "choice" in answer:
        return str(answer["choice"])
    probabilities = distribution(answer)
    return max(probabilities, key=probabilities.__getitem__)


def kl_divergence(reference, candidate):
    """KL(reference || candidate) over aligned probability vectors."""
    p = np.maximum(reference, FLOOR)
    q = np.maximum(candidate, FLOOR)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def accuracy(results, labels):
    """Correct and total counts over labelled cases that were run.

    Groups: all, clear, ambiguous, then per source ("synthetic", "real") and per source with the clear/ambiguous split.
    """
    totals = {}
    not_run = 0
    for case_id, label in labels.items():
        if case_id not in results:
            not_run += 1
            continue
        answers = results[case_id]
        correct = label["task"] in answers and decision(answers[label["task"]]) in label["expected"]
        source = label.get("source", "synthetic")
        kind = "ambiguous" if label.get("ambiguous") else "clear"
        for key in ("all", kind, source, f"{source}/{kind}"):
            count, total = totals.get(key, (0, 0))
            totals[key] = (count + int(correct), total + 1)
    return {"groups": {key: {"correct": c, "total": t} for key, (c, t) in totals.items()}, "not_run": not_run}


def route_share(results, case_ids, question_id):
    """Counts of the option each run chose for question_id over case_ids, and how many cases had no answer."""
    counts, missing = Counter(), 0
    for case_id in case_ids:
        answers = results.get(case_id, {})
        if question_id not in answers:
            missing += 1
            continue
        counts[decision(answers[question_id])] += 1
    return counts, missing


def latency_summary(values):
    """Count, p50 and p95 (linear interpolation) of latencies in ms; the percentiles are None when empty."""
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    p50, p95 = np.percentile(np.asarray(values, dtype=np.float64), [50, 95])
    return {"n": len(values), "p50": float(p50), "p95": float(p95)}


def compare(reference, candidate, labels):
    """Compare two {case_id: answers} maps and return a plain dict for reporting."""
    shared_ids = sorted(set(reference) & set(candidate))
    by_question = {}
    changed = []
    unmatched = []
    for case_id in shared_ids:
        ref_answers, cand_answers = reference[case_id], candidate[case_id]
        for question_id in sorted(set(ref_answers) | set(cand_answers)):
            if question_id not in ref_answers or question_id not in cand_answers:
                unmatched.append(f"{case_id}/{question_id}")
                continue
            ref_dist = distribution(ref_answers[question_id])
            cand_dist = distribution(cand_answers[question_id])
            if set(ref_dist) != set(cand_dist):
                raise ValueError(f"{case_id}/{question_id}: option sets differ")
            keys = sorted(ref_dist)
            p = np.array([ref_dist[key] for key in keys])
            q = np.array([cand_dist[key] for key in keys])
            stats = by_question.setdefault(question_id, {"n": 0, "agree": 0, "diffs": [], "kls": []})
            stats["n"] += 1
            ref_decision = decision(ref_answers[question_id])
            cand_decision = decision(cand_answers[question_id])
            if ref_decision == cand_decision:
                stats["agree"] += 1
            else:
                changed.append((case_id, question_id, ref_decision, cand_decision))
            stats["diffs"].extend(np.abs(p - q).tolist())
            stats["kls"].append(kl_divergence(p, q))
    return {
        "shared": len(shared_ids),
        "only_reference": sorted(set(reference) - set(candidate)),
        "only_candidate": sorted(set(candidate) - set(reference)),
        "unmatched": unmatched,
        "by_question": by_question,
        "changed": changed,
        "accuracy": {"reference": accuracy(reference, labels), "candidate": accuracy(candidate, labels)},
    }


def build_summary(results, latencies, sources, question_id):
    """One backend's answered count, routing counts over real cases, and latency percentiles."""
    real_ids = sorted(case_id for case_id, source in sources.items() if source == "real")
    counts, missing = route_share(results, real_ids, question_id)
    return {
        "answered": len(results),
        "real_total": len(real_ids),
        "choices": counts,
        "missing": missing,
        "latency": latency_summary(list(latencies.values())),
    }


def _mean_max(values):
    """Mean and maximum of a list, or zeros when empty."""
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.max())


def _fraction(block, key):
    """'correct/total' for one accuracy group, or 0/0 when the group has no cases."""
    group = block["groups"].get(key, {"correct": 0, "total": 0})
    return f"{group['correct']}/{group['total']}"


def _percent(count, total):
    """Share of total as a percentage string, or n/a when total is zero."""
    return f"{100.0 * count / total:.1f}%" if total else "n/a"


def _ms(value):
    """A latency in ms to one decimal, or n/a."""
    return "n/a" if value is None else f"{value:.1f}"


def _agreement_rows(report):
    """Markdown rows for the per-question agreement and drift table, with an overall row."""
    rows, all_diffs, all_kls, total, agreed = [], [], [], 0, 0
    for question_id, stats in sorted(report["by_question"].items()):
        mean_diff, max_diff = _mean_max(stats["diffs"])
        mean_kl, max_kl = _mean_max(stats["kls"])
        all_diffs.extend(stats["diffs"])
        all_kls.extend(stats["kls"])
        total += stats["n"]
        agreed += stats["agree"]
        rows.append(
            f"| {question_id} | {stats['n']} | {100.0 * stats['agree'] / stats['n']:.2f}% | "
            f"{mean_diff:.6f} | {max_diff:.6f} | {mean_kl:.6f} | {max_kl:.6f} |"
        )
    mean_diff, max_diff = _mean_max(all_diffs)
    mean_kl, max_kl = _mean_max(all_kls)
    if total:
        rows.append(
            f"| all | {total} | {100.0 * agreed / total:.2f}% | {mean_diff:.6f} | {max_diff:.6f} | "
            f"{mean_kl:.6f} | {max_kl:.6f} |"
        )
    return rows


def _change_rows(report):
    """One Markdown bullet per changed decision, or a single 'None.' line."""
    if not report["changed"]:
        return ["None."]
    return [f"- {case_id}/{question_id}: {ref} -> {cand}" for case_id, question_id, ref, cand in report["changed"]]


def _choice_rows(summary):
    """Markdown rows for the routing choices: haiku, sonnet, opus, then any other option seen."""
    reference, candidate = summary["reference"], summary["candidate"]
    options = list(ROUTE_ORDER) + sorted((set(reference["choices"]) | set(candidate["choices"])) - set(ROUTE_ORDER))
    rows = []
    for option in options:
        ref_count, cand_count = reference["choices"][option], candidate["choices"][option]
        rows.append(
            f"| {option} | {ref_count} ({_percent(ref_count, sum(reference['choices'].values()))}) | "
            f"{cand_count} ({_percent(cand_count, sum(candidate['choices'].values()))}) |"
        )
    rows.append(f"| no answer | {reference['missing']} | {candidate['missing']} |")
    return rows


NOT_VERIFIED = (
    "- One run per backend. No repeat runs, so no variance estimate for any figure.",
    "- Real-case labels exist only where the ledger has a Haiku run with verdict pass. They show that Haiku "
    "was sufficient, not that the decision was right. Real-case accuracy is one-sided.",
    "- The routing choices show what each backend decided, not whether the decision was right. Unlabelled "
    "real cases carry no accuracy figure at all.",
    "- Probabilities from the Workers AI proxy are rounded to 4 decimals, so drift and KL are quantised at that level.",
    "- Cases a backend failed to answer are left out of drift, agreement and accuracy for that pair; "
    "their count is in the answered lines above and in the sidecar meta for workers-ai.jsonl.",
    "- The MLX first case includes the lazy model load, because MlxEngine loads on its first decide call. "
    "No warm-up was run, so the MLX p95 may include that cost.",
    "- Latency covers this machine only, for the Workers AI proxy path (local proxy to a remote backend) and "
    "the local MLX path. It is not a throughput measure, and no end-to-end timing was taken.",
    "- The MLX run was under memory pressure: available memory fell to about 5 GiB during it, with swap in use "
    "at one check (1.5 GB used). That may slow the MLX latency, so the latency gap is not a clean engine measure.",
    "- Effort is sent as a score question with list criteria. Whether both backends render the legend the same way was not checked.",
    "- Only the 100 cases in cases100.jsonl were run. Nothing beyond them was measured.",
)


def format_report(report, summary, reference_name, candidate_name):
    """Render the comparison as Markdown; summary holds the question name and both build_summary blocks."""
    reference, candidate = summary["reference"], summary["candidate"]
    question = summary["question"]
    lines = [
        "# Clef-flash parity: Workers AI proxy vs MLX q8",
        "",
        f"{NOTICE} applies to the 30 synthetic cases only. The 70 real cases are first messages from HaiKrew "
        "subagent transcripts and are not simulated.",
        "",
        f"- Reference: `{reference_name}`, answered {reference['answered']}.",
        f"- Candidate: `{candidate_name}`, answered {candidate['answered']}.",
        f"- Shared cases compared: {report['shared']}.",
    ]
    if report["only_reference"] or report["only_candidate"]:
        lines.append(f"- Only on one side: reference {report['only_reference']}, candidate {report['only_candidate']}.")
    lines += [
        "",
        "## Agreement and drift",
        "",
        "| question | n | top-choice agreement | mean abs dp | max abs dp | mean KL | max KL |",
        "|---|---|---|---|---|---|---|",
        *_agreement_rows(report),
        "",
        f"## Decision changes ({len(report['changed'])})",
        "",
        *_change_rows(report),
        "",
        "## Accuracy against labels (correct/total)",
        "",
        "| group | reference | candidate |",
        "|---|---|---|",
        f"| synthetic, all | {_fraction(report['accuracy']['reference'], 'synthetic')} | "
        f"{_fraction(report['accuracy']['candidate'], 'synthetic')} |",
        f"| synthetic, clear | {_fraction(report['accuracy']['reference'], 'synthetic/clear')} | "
        f"{_fraction(report['accuracy']['candidate'], 'synthetic/clear')} |",
        f"| synthetic, ambiguous (any listed label) | {_fraction(report['accuracy']['reference'], 'synthetic/ambiguous')} | "
        f"{_fraction(report['accuracy']['candidate'], 'synthetic/ambiguous')} |",
        f"| real, ledger Haiku-pass labels (one-sided) | {_fraction(report['accuracy']['reference'], 'real')} | "
        f"{_fraction(report['accuracy']['candidate'], 'real')} |",
        f"| all labelled | {_fraction(report['accuracy']['reference'], 'all')} | "
        f"{_fraction(report['accuracy']['candidate'], 'all')} |",
        f"Labelled cases not run: {report['accuracy']['reference']['not_run']} (reference), "
        f"{report['accuracy']['candidate']['not_run']} (candidate).",
        "",
        f"## Routing choices over real cases (question `{question}`, {reference['real_total']} real cases)",
        "",
        "| choice | reference | candidate |",
        "|---|---|---|",
        *_choice_rows(summary),
        "",
        "## Latency (all answered cases, ms)",
        "",
        "| backend | n | p50 | p95 |",
        "|---|---|---|---|",
        f"| reference | {reference['latency']['n']} | {_ms(reference['latency']['p50'])} | {_ms(reference['latency']['p95'])} |",
        f"| candidate | {candidate['latency']['n']} | {_ms(candidate['latency']['p50'])} | {_ms(candidate['latency']['p95'])} |",
        "",
        "## What has NOT been verified",
        "",
        *NOT_VERIFIED,
    ]
    return "\n".join(lines)


def main(argv=None):
    """Command-line entry: compare two results files and print or write the Markdown report."""
    parser = argparse.ArgumentParser(description="Compare per-case decisions from two backends.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="cases100.jsonl (labels, sources) or cases.json")
    parser.add_argument("--route-question", default="model", help="question whose choices are tallied over real cases")
    parser.add_argument("--out", type=Path, help="write the Markdown report here instead of printing it")
    args = parser.parse_args(argv)
    try:
        reference, candidate = load_results(args.reference), load_results(args.candidate)
        report = compare(reference, candidate, load_labels(args.cases))
        sources = load_sources(args.cases)
        summary = {
            "question": args.route_question,
            "reference": build_summary(reference, load_latency_ms(args.reference), sources, args.route_question),
            "candidate": build_summary(candidate, load_latency_ms(args.candidate), sources, args.route_question),
        }
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = format_report(report, summary, str(args.reference), str(args.candidate))
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
