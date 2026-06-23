"""Coverage report — the evolution feedback loop, aggregated over a corpus.

Three signals (see docs/extractors.md "覆盖率 sanity check"):
  - unknown_fields frequency  -> what to start extracting (promotion candidates)
  - canonical fill rates      -> what the extractors are missing (alias gaps)
  - per-family fill rates     -> N/A vs missing nuance (e.g. DeepSeek MLA has no kv_heads)

Inputs come straight from per-model provenance, so this needs no re-download.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelspec.analytics.batch import BatchResult
    from modelspec.schema import ModelSpec

# The curated canonical fields whose fill rate we track. Deliberately small —
# canonical is a strictly controlled set (see AGENTS.md).
CANONICAL_FIELDS: list[str] = [
    "architecture.family",
    "architecture.num_layers",
    "architecture.hidden_size",
    "attention.type",
    "attention.num_heads",
    "attention.num_kv_heads",
    "parameters.total",
    "parameters.dtype_native",
    "context.declared",
    "tokenizer.vocab_size",
    "license.spdx_id",
]

# raw → passthrough promotion threshold: a field seen in >10% of models with a
# clear meaning is worth buffering (see docs/extractors.md field promotion).
DEFAULT_PROMOTION_THRESHOLD = 0.10


def _filled_paths(spec: "ModelSpec") -> set[str]:
    """Field paths a source actually claimed (per_field is the winners' record)."""
    return set(spec.provenance.per_field)


def _pct(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


@dataclass
class CoverageReport:
    total: int
    succeeded: int
    failed: int
    failures: list[tuple[str, str]] = field(default_factory=list)  # (target, error)
    unknown_field_frequency: list[tuple[str, int, float]] = field(default_factory=list)
    promotion_candidates: list[tuple[str, int, float]] = field(default_factory=list)
    canonical_fill_rates: dict[str, float] = field(default_factory=dict)
    # Per canonical field: how many models flagged it N/A (excluded from the
    # fill-rate denominator). See provenance.not_applicable.
    na_counts: dict[str, int] = field(default_factory=dict)
    per_family_fill_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    models_with_warnings: int = 0
    models_with_conflicts: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "failures": [{"target": t, "error": e} for t, e in self.failures],
            "unknown_field_frequency": [
                {"field": f, "count": c, "pct": p}
                for f, c, p in self.unknown_field_frequency
            ],
            "promotion_candidates": [
                {"field": f, "count": c, "pct": p} for f, c, p in self.promotion_candidates
            ],
            "canonical_fill_rates": self.canonical_fill_rates,
            "na_counts": self.na_counts,
            "per_family_fill_rates": self.per_family_fill_rates,
            "family_counts": self.family_counts,
            "models_with_warnings": self.models_with_warnings,
            "models_with_conflicts": self.models_with_conflicts,
        }

    def render(self, *, full: bool = True, top_n: int = 20) -> str:
        """Human-readable dashboard."""
        lines: list[str] = []
        lines.append(
            f"models: {self.total}  ok: {self.succeeded}  failed: {self.failed}"
        )
        lines.append(
            f"  warnings in {self.models_with_warnings} models, "
            f"conflicts in {self.models_with_conflicts} models"
        )

        lines.append("")
        lines.append(f"unknown_fields frequency (top {top_n}):")
        if self.unknown_field_frequency:
            for f, c, p in self.unknown_field_frequency[:top_n]:
                lines.append(f"  {p:6.1%}  {c:>6}  {f}")
        else:
            lines.append("  (none)")

        if self.promotion_candidates:
            lines.append("")
            lines.append(
                f"promotion candidates (>= {DEFAULT_PROMOTION_THRESHOLD:.0%} of models):"
            )
            for f, c, p in self.promotion_candidates:
                lines.append(f"  {p:6.1%}  {c:>6}  {f}")

        if full:
            lines.append("")
            lines.append("canonical fill rates (over applicable models):")
            for path in CANONICAL_FIELDS:
                na = self.na_counts.get(path, 0)
                applicable = self.succeeded - na
                if applicable <= 0:
                    # Every model marked this field N/A — no gap, just not applicable.
                    lines.append(f"     n/a  {path}  (all {na} N/A)")
                    continue
                rate = self.canonical_fill_rates.get(path, 0.0)
                flag = "  <-- low" if rate < 0.5 else ""
                suffix = f"  (n/a: {na})" if na else ""
                lines.append(f"  {rate:6.1%}  {path}{flag}{suffix}")

            if self.per_family_fill_rates:
                lines.append("")
                lines.append("per-family fill rates (family: n):")
                for fam, rates in sorted(
                    self.per_family_fill_rates.items(),
                    key=lambda kv: self.family_counts.get(kv[0], 0),
                    reverse=True,
                ):
                    n = self.family_counts.get(fam, 0)
                    cells = "  ".join(
                        f"{p.split('.')[-1]}={rates.get(p, 0.0):.0%}" for p in CANONICAL_FIELDS
                    )
                    lines.append(f"  {fam} (n={n}): {cells}")

        if self.failures:
            lines.append("")
            lines.append("failures:")
            for t, e in self.failures[:top_n]:
                lines.append(f"  {t}: {e}")

        return "\n".join(lines)


def build_coverage_report(
    result: "BatchResult",
    *,
    promotion_threshold: float = DEFAULT_PROMOTION_THRESHOLD,
    top_n: int = 20,
) -> CoverageReport:
    specs = result.specs
    n = len(specs)

    # --- unknown_fields frequency ---
    unknown_counter: Counter[str] = Counter()
    for spec in specs:
        unknown_counter.update(set(spec.provenance.unknown_fields))
    unknown_freq = [
        (fieldname, count, _pct(count, n))
        for fieldname, count in unknown_counter.most_common()
    ]
    promotion = [
        (f, c, p) for f, c, p in unknown_freq if p >= promotion_threshold
    ]

    # --- canonical fill rates (overall + per family) ---
    # Fill rate is computed over *applicable* models: models that flagged a field
    # as N/A (provenance.not_applicable) are excluded from the denominator, so a
    # field that is legitimately absent (e.g. num_kv_heads under MLA) doesn't read
    # as a low-coverage gap.
    filled_counter: Counter[str] = Counter()
    na_counter: Counter[str] = Counter()
    family_filled: dict[str, Counter[str]] = {}
    family_na: dict[str, Counter[str]] = {}
    family_counts: Counter[str] = Counter()
    for spec in specs:
        paths = _filled_paths(spec)
        na = set(spec.provenance.not_applicable)
        fam = spec.architecture.family or "unknown"
        family_counts[fam] += 1
        family_filled.setdefault(fam, Counter())
        family_na.setdefault(fam, Counter())
        for path in CANONICAL_FIELDS:
            if path in paths:
                filled_counter[path] += 1
                family_filled[fam][path] += 1
            if path in na:
                na_counter[path] += 1
                family_na[fam][path] += 1

    canonical_fill = {
        p: _pct(filled_counter.get(p, 0), n - na_counter.get(p, 0)) for p in CANONICAL_FIELDS
    }
    na_counts = {p: na_counter.get(p, 0) for p in CANONICAL_FIELDS}
    per_family = {
        fam: {
            p: _pct(counter.get(p, 0), family_counts[fam] - family_na[fam].get(p, 0))
            for p in CANONICAL_FIELDS
        }
        for fam, counter in family_filled.items()
    }

    # --- conflict / warning prevalence ---
    with_warnings = sum(1 for s in specs if s.provenance.warnings)
    with_conflicts = sum(1 for s in specs if s.provenance.conflicts)

    return CoverageReport(
        total=result.total,
        succeeded=result.succeeded,
        failed=result.failed,
        failures=[(it.target, it.error or "") for it in result.failures],
        unknown_field_frequency=unknown_freq,
        promotion_candidates=promotion,
        canonical_fill_rates=canonical_fill,
        na_counts=na_counts,
        per_family_fill_rates=per_family,
        family_counts=dict(family_counts),
        models_with_warnings=with_warnings,
        models_with_conflicts=with_conflicts,
    )
