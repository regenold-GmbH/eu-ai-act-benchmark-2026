"""Pure reference-scoring helpers with no heavy dependencies.

Kept separate so tests can import them without pulling in pandas / litellm.
"""

import re

REFERENCE_RE = re.compile(
    r"^(?P<kind>Article|Annex)\s+(?P<base>[^\s.]+)(?:\.(?P<sub>[^\s.]+))?(?:\.[^\s.]+)*$",
    flags=re.IGNORECASE,
)


def _normalize_reference(reference: str, *, keep_sublevel: bool) -> str:
    reference = reference.strip().lower()
    match = REFERENCE_RE.match(reference)
    if not match:
        return reference

    normalized = f"{match.group('kind').lower()} {match.group('base').lower()}"
    sublevel = match.group("sub")
    if keep_sublevel and sublevel:
        normalized += f".{sublevel.lower()}"
    return normalized


def compute_ref_scores(found: list[str], expected: list[str]) -> dict:
    found_refs = [ref.strip().lower() for ref in found if ref.strip()]
    expected_refs = [ref.strip().lower() for ref in expected if ref.strip()]

    # Two steps:
    # 1) "Loose": compare only the top-level Article/Annex reference.
    # 2) "Strict": compare the top-level reference plus one sublevel, if present.
    # This keeps Annex IV.1 aligned with Annex IV for loose matching while still
    # distinguishing Annex IV.1 from Annex IV in strict matching.
    loose_expected = [_normalize_reference(ref, keep_sublevel=False) for ref in expected_refs]
    loose_found = [_normalize_reference(ref, keep_sublevel=False) for ref in found_refs]

    strict_expected = [_normalize_reference(ref, keep_sublevel=True) for ref in expected_refs]
    strict_found = [_normalize_reference(ref, keep_sublevel=True) for ref in found_refs]

    loose_expected_set = set(loose_expected)
    loose_found_set = set(loose_found)
    strict_expected_set = set(strict_expected)
    strict_found_set = set(strict_found)

    result = dict()
    if loose_expected_set:
        result["loose_found_pct"] = len(
            loose_expected_set.intersection(loose_found_set)
        ) / len(loose_expected_set)
        result["loose_extra_pct"] = len(
            loose_found_set - loose_expected_set
        ) / len(loose_expected_set)
    else:
        result["loose_found_pct"] = None
        result["loose_extra_pct"] = None

    if strict_expected_set:
        matched_expected = strict_expected_set.intersection(strict_found_set)
        matched_found = set(matched_expected)

        # A specific paragraph satisfies an expected top-level Article/Annex.
        # Consume one such candidate reference; any siblings remain extras.
        for expected_ref in sorted(strict_expected_set - matched_expected):
            if expected_ref != _normalize_reference(
                expected_ref, keep_sublevel=False
            ):
                continue
            for found_ref in sorted(strict_found_set - matched_found):
                if _normalize_reference(
                    found_ref, keep_sublevel=False
                ) == expected_ref:
                    matched_expected.add(expected_ref)
                    matched_found.add(found_ref)
                    break

        result["strict_found_pct"] = len(matched_expected) / len(
            strict_expected_set
        )
        raw_extra = strict_found_set - matched_found
        # Exempt only a top-level parent of an expected sublevel.
        raw_extra = {
            ref for ref in raw_extra
            if not (
                ref == _normalize_reference(ref, keep_sublevel=False)
                and ref in loose_expected_set
            )
        }
        result["strict_extra_pct"] = len(raw_extra) / len(
            strict_expected_set
        )
    else:
        # Questions without annotated expected references cannot distinguish
        # relevant citations from extras, so exclude them from these metrics.
        result["strict_found_pct"] = None
        result["strict_extra_pct"] = None

    return result
