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
from modelspec.extractors.gguf import GGUFExtractor
from modelspec.extractors.license import LicenseExtractor
from modelspec.extractors.merge import MergeExtractor
from modelspec.extractors.safetensors import SafetensorsExtractor
from modelspec.extractors.tokenizer import TokenizerExtractor

# M1: config + safetensors. M2 adds gguf / tokenizer / license. M3 adds merge
# (and quantization claims emitted by config_json + gguf). The orchestrator runs
# every extractor whose can_handle() returns True, so the GGUF extractor without
# the optional gguf package simply opts out.
ALL_EXTRACTORS: list[Extractor] = [
    ConfigJsonExtractor(),
    SafetensorsExtractor(),
    GGUFExtractor(),
    TokenizerExtractor(),
    LicenseExtractor(),
    MergeExtractor(),
]

__all__ = [
    "Extractor",
    "ExtractionSource",
    "ExtractorResult",
    "FieldClaim",
    "ConfigJsonExtractor",
    "SafetensorsExtractor",
    "GGUFExtractor",
    "TokenizerExtractor",
    "LicenseExtractor",
    "MergeExtractor",
    "ALL_EXTRACTORS",
]
