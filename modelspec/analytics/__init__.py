"""Analytics package (M4) — batch extraction and the evolution feedback loop.

Turns the per-model provenance accumulated by M1–M3 (unknown_fields, per_field,
conflicts, warnings) into cross-corpus signals that drive schema / extractor
evolution. See docs/roadmap.md (M4) and docs/analytics.md.
"""

from modelspec.analytics.batch import BatchItem, BatchResult, read_targets, run_batch
from modelspec.analytics.report import CoverageReport, build_coverage_report

__all__ = [
    "BatchItem",
    "BatchResult",
    "run_batch",
    "read_targets",
    "CoverageReport",
    "build_coverage_report",
]
