"""Pipeline package — orchestration, merging and conflict resolution."""

from modelspec.pipeline.merger import MergeResult, merge_claims
from modelspec.pipeline.orchestrator import (
    detect_source_format,
    extract,
    extract_from_source,
    reshape,
)

__all__ = [
    "extract",
    "extract_from_source",
    "detect_source_format",
    "reshape",
    "merge_claims",
    "MergeResult",
]
