"""Extractor protocol and the data contracts shared by every extractor.

Extractors are plain Python — they never import Pydantic. Each one reads one
kind of source file and returns flat ``FieldClaim`` tuples plus the two lower
extraction layers (passthrough, raw). See docs/extractors.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

# Confidence ordering used by the merger when sources disagree.
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


class FieldClaim(NamedTuple):
    """The connection point between extractors and the schema.

    Extractors emit flat dotted-path claims (not nested dicts) so the merger
    can resolve conflicts field-by-field.
    """

    field_path: str  # e.g. "architecture.num_layers"
    value: Any
    source: str  # "config" | "tensors" | "inferred" | "fingerprint" | ...
    confidence: str  # "high" | "medium" | "low"


@dataclass
class ExtractorResult:
    """The three-layer output of a single extractor.

    - ``claims``: canonical layer — fields the schema cares about, normalized.
    - ``passthrough``: recognized-but-unmapped values, kept verbatim.
    - ``raw``: complete original blob, lossless insurance.
    - ``unknown_fields``: raw keys covered by neither canonical nor passthrough.
    - ``not_applicable``: dotted paths this model legitimately lacks (e.g.
      attention.num_kv_heads under MLA) — distinct from merely missing.
    """

    claims: list[FieldClaim] = field(default_factory=list)
    passthrough: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    unknown_fields: list[str] = field(default_factory=list)
    not_applicable: list[str] = field(default_factory=list)


@dataclass
class ExtractionSource:
    """A local directory holding the (metadata-only) files of one model.

    The fetcher downloads metadata into ``root``; the orchestrator runs every
    extractor over this directory. ``repo_files`` is the listing of file names
    available in the repo (used by ``can_handle``).
    """

    root: Path
    repo_files: list[str]
    repo_id: str | None = None
    source_format: str = "unknown"

    def path(self, name: str) -> Path:
        return self.root / name

    def has(self, name: str) -> bool:
        return self.path(name).is_file()


@runtime_checkable
class Extractor(Protocol):
    """The protocol the orchestrator depends on — nothing more."""

    name: str

    def can_handle(self, source: ExtractionSource) -> bool: ...

    def extract(self, source: ExtractionSource) -> ExtractorResult: ...
