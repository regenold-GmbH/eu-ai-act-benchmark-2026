"""Print the eight benchmark metrics and geometric-mean overall score."""

import argparse
import json
from math import exp, log
from pathlib import Path
from statistics import fmean

from latency_scoring import compute_response_speed_score

LABELS = [
    "Answer correctness (loose)",
    "Answer correctness (strict)",
    "Answer conciseness",
    "Reference correctness (loose)",
    "Reference correctness (strict)",
    "Reference conciseness",
    "Regulatory tone",
    "Response speed",
]


def mean_or_none(values):
    return fmean(values) if values else None


def overall_score(values):
    available = [value for value in values if value is not None]
    if not available:
        return None
    if 0 in available:
        return 0.0
    return exp(fmean(log(value) for value in available))


def compute_run_metrics(entries, run_index):
    criteria = []
    strict = []
    tone = []
    for entry in entries:
        judgements = entry["judgements_correctness"]
        if run_index < len(judgements):
            flags = [criterion["passed"] for criterion in judgements[run_index]["criteria"]]
            if flags:
                criteria.extend(100.0 * flag for flag in flags)
                strict.append(100.0 * all(flags))
        judgements = entry["judgements_tone"]
        if run_index < len(judgements):
            judgement = judgements[run_index]
            tone.append(100.0 * (
                judgement["appropriateness_passed"] and judgement["clarity_passed"]
            ))

    def ratios(key, score):
        return mean_or_none([
            score(entry[key]) for entry in entries if entry.get(key) is not None
        ])

    return [
        mean_or_none(criteria),
        mean_or_none(strict),
        ratios("answer_conciseness_score", lambda x: 100 / max(1, x)),
        ratios("loose_found_pct", lambda x: 100 * x),
        ratios("strict_found_pct", lambda x: 100 * x),
        ratios("strict_extra_pct", lambda x: 100 / (1 + x)),
        mean_or_none(tone),
        ratios("latency_seconds", compute_response_speed_score),
    ]


def summarize(entries):
    if not entries:
        raise ValueError("The result file is empty")
    runs = max(
        max(len(entry["judgements_correctness"]), len(entry["judgements_tone"]))
        for entry in entries
    )
    if runs < 1:
        raise ValueError("The result file contains no judge runs")
    per_run = [compute_run_metrics(entries, index) for index in range(runs)]
    metrics = []
    for index in range(len(LABELS)):
        values = [row[index] for row in per_run if row[index] is not None]
        metrics.append({
            "mean": mean_or_none(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        })
    overall_runs = [overall_score(row) for row in per_run]
    overall_runs = [score for score in overall_runs if score is not None]
    return {
        "rows": len(entries),
        "judge_runs": runs,
        "metrics": dict(zip(LABELS, metrics)),
        "overall": {
            "mean": overall_score([metric["mean"] for metric in metrics]),
            "min": min(overall_runs) if overall_runs else None,
            "max": max(overall_runs) if overall_runs else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="Result JSON files")
    args = parser.parse_args()
    for path in args.results:
        report = summarize(json.loads(path.read_text(encoding="utf-8")))
        print(f"\n{path.name}: {report['rows']} rows, {report['judge_runs']} judge runs")
        for label, metric in [*report["metrics"].items(), ("Overall", report["overall"])]:
            if metric["mean"] is None:
                print(f"  {label:32s} n/a")
            else:
                print(
                    f"  {label:32s} {metric['mean']:6.2f}"
                    f"  [{metric['min']:.2f}, {metric['max']:.2f}]"
                )


if __name__ == "__main__":
    main()
