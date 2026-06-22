"""Cross-validation — warnings for param mismatch, context, and MoE signals."""

from __future__ import annotations

from modelspec.pipeline.cross_validate import cross_validate
from modelspec.schema import ModelSpec


def _spec(data: dict) -> ModelSpec:
    return ModelSpec.model_validate(data)


def test_parameter_double_path_agreement():
    # vocab*hidden*2 (tied? no) + layers*(attn+ffn). Build a config whose
    # estimate matches the tensor total within tolerance.
    cfg = {"vocab_size": 100, "intermediate_size": 32}
    spec = _spec(
        {
            "architecture": {"hidden_size": 16, "num_layers": 2, "tied_embeddings": False},
            "attention": {"num_heads": 4, "num_kv_heads": 4},
            "tokenizer": {"vocab_size": 100},
            "provenance": {"raw_config_json": cfg},
        }
    )
    # Compute the exact estimate and feed it back as the tensor total.
    from modelspec.pipeline.cross_validate import _estimate_params_from_config

    spec.parameters.total = _estimate_params_from_config(spec, cfg)
    cross_validate(spec)
    assert not any("parameter count mismatch" in w for w in spec.provenance.warnings)


def test_parameter_mismatch_warns():
    cfg = {"vocab_size": 100, "intermediate_size": 32}
    spec = _spec(
        {
            "architecture": {"hidden_size": 16, "num_layers": 2, "tied_embeddings": False},
            "attention": {"num_heads": 4, "num_kv_heads": 4},
            "parameters": {"total": 999999},
            "tokenizer": {"vocab_size": 100},
            "provenance": {"raw_config_json": cfg},
        }
    )
    cross_validate(spec)
    assert any("parameter count mismatch" in w for w in spec.provenance.warnings)


def test_context_effective_inconsistency_warns():
    spec = _spec(
        {
            "context": {"declared": 4096, "effective": 9999},
            "provenance": {"raw_config_json": {"rope_scaling": {"factor": 8.0}}},
        }
    )
    cross_validate(spec)
    assert any("context.effective" in w for w in spec.provenance.warnings)


def test_moe_signal_disagreement_warns():
    # config says MoE (expert count) but no tensor "moe" tag.
    spec = _spec({"moe": {"num_experts": 8}, "architecture": {"tags": ["decoder-only"]}})
    cross_validate(spec)
    assert any("MoE signal disagreement" in w for w in spec.provenance.warnings)
