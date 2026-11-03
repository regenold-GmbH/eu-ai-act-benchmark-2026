"""Tests for compute_ref_scores."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ref_scoring import compute_ref_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scores(found, expected):
    """Thin wrapper so tests read clearly."""
    return compute_ref_scores(found, expected)


# ---------------------------------------------------------------------------
# Single-reference cases (from the bugfix session)
# ---------------------------------------------------------------------------

class TestSingleRef:
    def test_annex_iv_found_against_annex_iv1_expected(self):
        """Candidate cites Annex IV; expected is Annex IV.1.
        Loose: top-level matches → found=1, extra=0.
        Strict: IV ≠ IV.1 → found=0.
          Extra: "annex iv" is a less-specific variant of the expected
          "annex iv.1", so it is NOT counted as an unrelated extra → extra=0.
        """
        r = scores(["Annex IV"], ["Annex IV.1"])
        assert r["loose_found_pct"] == 1.0
        assert r["loose_extra_pct"] == 0.0
        assert r["strict_found_pct"] == 0.0
        assert r["strict_extra_pct"] == 0.0

    def test_annex_iv1a_found_against_annex_iv1_expected(self):
        """Candidate cites Annex IV.1.a; expected is Annex IV.1.
        Both loose and strict should match (the .a suffix is discarded).
        """
        r = scores(["Annex IV.1.a"], ["Annex IV.1"])
        assert r["loose_found_pct"] == 1.0
        assert r["loose_extra_pct"] == 0.0
        assert r["strict_found_pct"] == 1.0
        assert r["strict_extra_pct"] == 0.0

    def test_article_3_found_against_article_31_expected(self):
        """Candidate cites Article 3; expected is Article 3.1.
        Loose: top-level matches. Strict: does not.
          Extra: "article 3" is a less-specific variant of the expected
          "article 3.1", so it is NOT counted as an unrelated extra → extra=0.
        """
        r = scores(["Article 3"], ["Article 3.1"])
        assert r["loose_found_pct"] == 1.0
        assert r["loose_extra_pct"] == 0.0
        assert r["strict_found_pct"] == 0.0
        assert r["strict_extra_pct"] == 0.0

    def test_article_31a_found_against_article_31_expected(self):
        """Candidate cites Article 3.1.a; expected is Article 3.1.
        Both loose and strict should match.
        """
        r = scores(["Article 3.1.a"], ["Article 3.1"])
        assert r["loose_found_pct"] == 1.0
        assert r["loose_extra_pct"] == 0.0
        assert r["strict_found_pct"] == 1.0
        assert r["strict_extra_pct"] == 0.0

    def test_exact_match_article(self):
        """Exact same reference: full match on both loose and strict."""
        r = scores(["Article 50.4"], ["Article 50.4"])
        assert r["loose_found_pct"] == 1.0
        assert r["loose_extra_pct"] == 0.0
        assert r["strict_found_pct"] == 1.0
        assert r["strict_extra_pct"] == 0.0

    def test_exact_match_annex(self):
        r = scores(["Annex IV"], ["Annex IV"])
        assert r["loose_found_pct"] == 1.0
        assert r["loose_extra_pct"] == 0.0
        assert r["strict_found_pct"] == 1.0
        assert r["strict_extra_pct"] == 0.0

    def test_wrong_article(self):
        """Completely different article: nothing found."""
        r = scores(["Article 6"], ["Article 10"])
        assert r["loose_found_pct"] == 0.0
        assert r["strict_found_pct"] == 0.0

    def test_unrelated_annex_is_penalized_as_extra(self):
      """An unrelated Annex reference should count as an extra in both modes.

      Candidate cites Annex IV.1.a, expected is Article 3.2.
      There is no loose or strict match, and the candidate reference is not a
      less-specific variant of the expected one, so it should be penalized as
      an extra citation in both loose and strict scoring.
      """
      r = scores(["Annex IV.1.a"], ["Article 3.2"])
      assert r["loose_found_pct"] == 0.0
      assert r["loose_extra_pct"] == 1.0
      assert r["strict_found_pct"] == 0.0
      assert r["strict_extra_pct"] == 1.0

    def test_wrong_sibling_sublevels_are_strict_extras(self):
        r = scores(
            ["Article 50.1", "Article 50.2", "Article 50.4"],
            ["Article 50.4"],
        )

        assert r["strict_found_pct"] == 1.0
        assert r["strict_extra_pct"] == 2.0

    def test_more_specific_reference_satisfies_expected_parent(self):
        r = scores(["Article 50.4"], ["Article 50"])

        assert r["loose_found_pct"] == 1.0
        assert r["strict_found_pct"] == 1.0
        assert r["strict_extra_pct"] == 0.0

    def test_only_one_more_specific_reference_is_needed_for_expected_parent(self):
        r = scores(
            ["Article 11.2", "Article 11.3", "Article 11.5"],
            ["Article 11"],
        )

        assert r["strict_found_pct"] == 1.0
        assert r["strict_extra_pct"] == 2.0

    def test_missing_expected_references_excludes_all_reference_scores(self):
        r = scores(["Article 6"], [])

        assert r["loose_found_pct"] is None
        assert r["strict_found_pct"] is None
        assert r["loose_extra_pct"] is None
        assert r["strict_extra_pct"] is None

    def test_no_candidate_references_has_zero_excess(self):
        r = scores([], ["Article 6"])

        assert r["loose_found_pct"] == 0.0
        assert r["strict_found_pct"] == 0.0
        assert r["loose_extra_pct"] == 0.0
        assert r["strict_extra_pct"] == 0.0


# ---------------------------------------------------------------------------
# Multi-reference candidate set collapsing
# ---------------------------------------------------------------------------

class TestMultiRefCollapsing:
    def test_candidate_annex_iv_iv1_iv1a_against_annex_iv1(self):
        """Candidate cites ["Annex IV", "Annex IV.1", "Annex IV.1.a"];
        expected is ["Annex IV.1"].

        Loose:
          - All three candidate refs normalise to "annex iv" (top-level only)
            and collapse to a single-element set {"annex iv"}.
          - Expected normalises to {"annex iv"}.
          - Intersection = 1 out of 1 expected → found_pct = 1.0, extra_pct = 0.0.

        Strict:
          - "Annex IV" → "annex iv", "Annex IV.1" → "annex iv.1",
            "Annex IV.1.a" → "annex iv.1" (suffix .a dropped).
          - Candidate strict set = {"annex iv", "annex iv.1"}.
          - Expected strict set = {"annex iv.1"}.
          - Intersection = {"annex iv.1"} → found_pct = 1.0.
          - Raw extra = {"annex iv"}, but its loose form ("annex iv") matches the
            loose form of the expected ref ("annex iv.1" → "annex iv"), so it is
            treated as a less-specific variant of an already-covered reference,
            not a genuinely unrelated extra → extra_pct = 0.0.
        """
        r = scores(["Annex IV", "Annex IV.1", "Annex IV.1.a"], ["Annex IV.1"])
        # Loose
        assert r["loose_found_pct"] == 1.0, f"loose_found_pct: {r}"
        assert r["loose_extra_pct"] == 0.0, f"loose_extra_pct: {r}"
        # Strict
        assert r["strict_found_pct"] == 1.0, f"strict_found_pct: {r}"
        assert r["strict_extra_pct"] == 0.0, f"strict_extra_pct: {r}"
