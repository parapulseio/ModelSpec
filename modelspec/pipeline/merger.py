"""Merge flat claims from all extractors and resolve conflicts.

Group by field path; the highest-confidence claim wins. Losing claims with a
different value are recorded as conflicts for human review. A few list-valued
fields (e.g. architecture.tags) are unioned instead of overwritten.
See docs/pipeline.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modelspec.extractors.base import CONFIDENCE_RANK, FieldClaim

# Fields whose list values are accumulated (set-union) across sources rather
# than resolved by a single winner.
_ACCUMULATE_FIELDS = {"architecture.tags"}


@dataclass
class MergedField:
    value: Any
    source: str
    confidence: str


@dataclass
class MergeResult:
    fields: dict[str, MergedField] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)


def merge_claims(claims: list[FieldClaim]) -> MergeResult:
    grouped: dict[str, list[FieldClaim]] = {}
    for c in claims:
        grouped.setdefault(c.field_path, []).append(c)

    result = MergeResult()
    for path, group in grouped.items():
        if path in _ACCUMULATE_FIELDS:
            result.fields[path] = _accumulate(group)
            continue

        # Highest confidence wins; ties keep first-seen order.
        winner = max(group, key=lambda c: CONFIDENCE_RANK.get(c.confidence, 0))
        result.fields[path] = MergedField(winner.value, winner.source, winner.confidence)

        for c in group:
            if c is winner:
                continue
            if c.value != winner.value:
                result.conflicts.append(
                    {
                        "field_path": path,
                        "value": c.value,
                        "source": c.source,
                        "confidence": c.confidence,
                        "winner_source": winner.source,
                        "winner_value": winner.value,
                    }
                )
    return result


def _accumulate(group: list[FieldClaim]) -> MergedField:
    """Union list-valued claims, preserving first-seen order."""
    seen: list[Any] = []
    best_conf = "low"
    for c in group:
        values = c.value if isinstance(c.value, list) else [c.value]
        for v in values:
            if v not in seen:
                seen.append(v)
        if CONFIDENCE_RANK.get(c.confidence, 0) > CONFIDENCE_RANK.get(best_conf, 0):
            best_conf = c.confidence
    return MergedField(seen, "merged", best_conf)
