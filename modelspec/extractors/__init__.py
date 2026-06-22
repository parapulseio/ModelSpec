"""Extractor registry.

The orchestrator pulls every extractor from ``ALL_EXTRACTORS``; disabling one
is just removing it from this list. New sources implement the ``Extractor``
protocol and register here — the orchestrator never changes.
"""

from modelspec.extractors.base import (
    Extractor,
    ExtractionSource,
    ExtractorResult,
    FieldClaim,
)
from modelspec.extractors.config_json import ConfigJsonExtractor
from modelspec.extractors.safetensors import SafetensorsExtractor

# M1 ships the two highest-coverage, lowest-dependency sources.
# GGUF / license / tokenizer extractors land in M2 (see docs/roadmap.md).
ALL_EXTRACTORS: list[Extractor] = [
    ConfigJsonExtractor(),
    SafetensorsExtractor(),
]

__all__ = [
    "Extractor",
    "ExtractionSource",
    "ExtractorResult",
    "FieldClaim",
    "ConfigJsonExtractor",
    "SafetensorsExtractor",
    "ALL_EXTRACTORS",
]
