from copy import deepcopy

import pytest

from summarize_results import LABELS, overall_score, summarize


def entry(runs, *, ratio=1, references=True, latency=10):
    return {
        "judgements_correctness": [
            {"criteria": [{"passed": passed} for passed in flags]} for flags in runs
        ],
        "judgements_tone": [
            {"appropriateness_passed": True, "clarity_passed": True} for _ in runs
        ],
        "answer_conciseness_score": ratio,
        "loose_found_pct": 1 if references else None,
        "strict_found_pct": 1 if references else None,
        "strict_extra_pct": 0 if references else None,
        "latency_seconds": latency,
    }


def test_strict_correctness_is_averaged_per_judge_run():
    # Strict passes in run 1 and fails in run 2: mean must be 50%, not 0%.
    report = summarize([entry([[True, True], [True, False]], ratio=2)])
    metrics = report["metrics"]
    assert metrics[LABELS[0]] == {"mean": 75, "min": 50, "max": 100}
    assert metrics[LABELS[1]] == {"mean": 50, "min": 0, "max": 100}
    assert metrics[LABELS[2]]["mean"] == 50
    assert metrics[LABELS[7]]["mean"] == 90
    assert report["overall"]["mean"] == pytest.approx(
        (75 * 50 * 50 * 100 * 100 * 100 * 100 * 90) ** (1 / 8)
    )
    assert report["overall"]["min"] == 0


def test_missing_references_are_excluded_and_zero_metrics_zero_overall():
    report = summarize([
        entry([[True]], references=False, latency=150),
        entry([[True]], references=True, latency=100),
    ])
    assert report["metrics"][LABELS[3]]["mean"] == 100
    assert report["metrics"][LABELS[7]]["mean"] == 0
    assert report["overall"]["mean"] == 0
    missing = summarize([entry([[True]], references=False)])
    assert missing["metrics"][LABELS[3]]["mean"] is None
    assert missing["overall"]["mean"] == pytest.approx((100**4 * 90) ** (1 / 5))
    assert overall_score([None, None]) is None


def test_extra_references_and_tone_failures():
    row = entry([[True]])
    row["strict_extra_pct"] = 2
    row["judgements_tone"][0]["clarity_passed"] = False
    report = summarize([row])
    assert report["metrics"][LABELS[5]]["mean"] == pytest.approx(100 / 3)
    assert report["metrics"][LABELS[6]]["mean"] == 0
    assert report["overall"]["mean"] == 0


def test_summary_does_not_modify_input():
    rows = [entry([[True, False], [False, True]])]
    before = deepcopy(rows)
    summarize(rows)
    assert rows == before


def test_empty_results_fail_clearly():
    with pytest.raises(ValueError, match="empty"):
        summarize([])
