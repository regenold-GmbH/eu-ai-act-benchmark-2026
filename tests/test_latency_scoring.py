"""Tests for deterministic response-speed scoring."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from latency_scoring import compute_response_speed_score


@pytest.mark.parametrize(
    ("latency_seconds", "expected_score"),
    [(0.0, 100.0), (1.0, 99.0), (99.5, 0.5), (100.0, 0.0), (150.0, 0.0)],
)
def test_response_speed_loses_one_point_per_second_and_clips_at_zero(
    latency_seconds, expected_score
):
    assert compute_response_speed_score(latency_seconds) == expected_score
