"""Merger — conflicting claims resolve by confidence; tags are unioned."""

from __future__ import annotations

from modelspec.extractors.base import FieldClaim
from modelspec.pipeline.merger import merge_claims


def test_highest_confidence_wins():
    claims = [
        FieldClaim("context.declared", 4096, "gguf", "low"),
        FieldClaim("context.declared", 131072, "config", "high"),
    ]
    result = merge_claims(claims)
    assert result.fields["context.declared"].value == 131072
    assert result.fields["context.declared"].source == "config"
    # The loser is recorded for review.
    assert len(result.conflicts) == 1
    assert result.conflicts[0]["value"] == 4096
    assert result.conflicts[0]["winner_value"] == 131072


def test_no_conflict_when_values_agree():
    claims = [
        FieldClaim("architecture.num_layers", 32, "config", "high"),
        FieldClaim("architecture.num_layers", 32, "tensors", "medium"),
    ]
    result = merge_claims(claims)
    assert result.fields["architecture.num_layers"].value == 32
    assert result.conflicts == []


def test_tags_are_unioned():
    claims = [
        FieldClaim("architecture.tags", ["decoder-only", "gqa"], "inferred", "medium"),
        FieldClaim("architecture.tags", ["moe"], "heuristic", "high"),
        FieldClaim("architecture.tags", ["gqa"], "heuristic", "high"),  # dup
    ]
    result = merge_claims(claims)
    tags = result.fields["architecture.tags"].value
    assert tags == ["decoder-only", "gqa", "moe"]
    assert result.conflicts == []
