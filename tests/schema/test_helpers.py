"""ModelSpec convenience accessors (M5) — orthogonal structures + provenance."""

from __future__ import annotations

from modelspec.schema import ModelSpec


def test_orthogonal_predicates_default_false():
    spec = ModelSpec.model_validate({})
    assert not spec.is_quantized()
    assert not spec.is_merged()
    assert not spec.is_moe()
    assert not spec.is_adapter()
    assert not spec.is_derived()
    assert spec.quant_format is None
    assert spec.bits_per_weight is None


def test_quant_helpers_gguf():
    spec = ModelSpec.model_validate(
        {"quantization": {"format": "gguf", "bits_per_weight_avg": 4.83}}
    )
    assert spec.is_quantized()
    assert spec.quant_format == "gguf"
    assert spec.bits_per_weight == 4.83


def test_quant_helpers_awq_bits_coerced_to_float():
    spec = ModelSpec.model_validate({"quantization": {"format": "awq", "bits": 4}})
    assert spec.quant_format == "awq"
    assert spec.bits_per_weight == 4.0


def test_modality_helpers():
    llm = ModelSpec.model_validate({"architecture": {"tags": ["decoder-only", "gqa"]}})
    assert llm.is_decoder_only() and not llm.is_multimodal()
    assert llm.modality == "decoder-only"

    vlm = ModelSpec.model_validate(
        {"architecture": {"tags": ["decoder-only", "multimodal", "gqa"]}}
    )
    assert vlm.is_multimodal() and vlm.is_decoder_only()
    assert vlm.modality == "multimodal"  # most specific wins

    audio = ModelSpec.model_validate({"architecture": {"tags": ["audio", "mha"]}})
    assert not audio.is_decoder_only()
    assert audio.modality == "audio"

    assert ModelSpec.model_validate({}).modality == "unknown"


def test_is_moe_and_merged_and_derived():
    spec = ModelSpec.model_validate(
        {
            "moe": {"num_experts": 8, "top_k": 2},
            "merge": {"detection_signal": "hf_tag", "confidence": "high"},
            "identity": {"lineage": {"base_models": ["meta-llama/Llama-3.1-8B"]}},
        }
    )
    assert spec.is_moe()
    assert spec.is_merged()
    assert spec.is_derived()


def test_effective_context_fallback_chain():
    assert ModelSpec.model_validate({"context": {"effective": 32768, "declared": 8192}}).effective_context == 32768
    assert ModelSpec.model_validate({"context": {"declared": 8192}}).effective_context == 8192
    assert ModelSpec.model_validate({"context": {"trained": 4096}}).effective_context == 4096
    assert ModelSpec.model_validate({}).effective_context is None


def test_provenance_accessors():
    spec = ModelSpec.model_validate(
        {
            "provenance": {
                "per_field": {"architecture.family": {"source": "config", "confidence": "high"}},
                "not_applicable": ["attention.num_kv_heads"],
                "conflicts": [
                    {
                        "field_path": "parameters.total",
                        "value": 1,
                        "source": "gguf",
                        "confidence": "low",
                        "winner_source": "tensors",
                        "winner_value": 2,
                    }
                ],
            }
        }
    )
    assert spec.source_of("architecture.family") == "config"
    assert spec.confidence_of("architecture.family") == "high"
    assert spec.source_of("missing.path") is None
    assert spec.is_not_applicable("attention.num_kv_heads")
    assert not spec.is_not_applicable("architecture.family")
    conflicts = spec.conflicts_for("parameters.total")
    assert len(conflicts) == 1 and conflicts[0].winner_value == 2
    assert spec.conflicts_for("nope") == []
