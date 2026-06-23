"""Composable predicate / filter library (M5)."""

from __future__ import annotations

from modelspec import query as q
from modelspec.schema import ModelSpec


def _spec(**overrides) -> ModelSpec:
    base: dict = {}
    base.update(overrides)
    return ModelSpec.model_validate(base)


LLAMA_Q = _spec(
    architecture={"family": "llama"},
    parameters={"total": 8_000_000_000},
    quantization={"format": "gguf", "bits_per_weight_avg": 4.83},
    context={"declared": 131072},
    license={"spdx_id": "apache-2.0", "commercial_use": True},
)
QWEN_FP = _spec(
    architecture={"family": "qwen2"},
    parameters={"total": 500_000_000},
    context={"declared": 32768},
    license={"spdx_id": "other", "commercial_use": False},
)
MERGE = _spec(merge={"detection_signal": "hf_tag", "confidence": "high"})

ALL = [LLAMA_Q, QWEN_FP, MERGE]


def test_bare_predicates():
    assert q.is_quantized(LLAMA_Q) and not q.is_quantized(QWEN_FP)
    assert q.is_merged(MERGE) and not q.is_merged(LLAMA_Q)
    assert q.is_dense(LLAMA_Q)
    assert q.commercial_use_allowed(LLAMA_Q) and not q.commercial_use_allowed(QWEN_FP)


def test_factory_predicates():
    assert q.family_is("LLAMA")(LLAMA_Q)  # case-insensitive
    assert not q.family_is("mistral")(LLAMA_Q)
    assert q.quant_format_in("gguf", "awq")(LLAMA_Q)
    assert q.min_params(7e9)(LLAMA_Q) and not q.min_params(7e9)(QWEN_FP)
    assert q.max_params(1e9)(QWEN_FP)
    assert q.min_context(100000)(LLAMA_Q) and not q.min_context(100000)(QWEN_FP)
    assert q.license_is("apache-2.0")(LLAMA_Q)


def test_unknown_values_excluded_by_numeric_predicates():
    assert not q.min_params(1)(MERGE)  # no parameter count -> excluded
    assert not q.min_context(1)(MERGE)


def test_combinators():
    big_quant = q.all_of(q.is_quantized, q.min_params(7e9))
    assert big_quant(LLAMA_Q) and not big_quant(QWEN_FP)
    either = q.any_of(q.is_merged, q.is_quantized)
    assert either(MERGE) and either(LLAMA_Q) and not either(QWEN_FP)
    assert q.negate(q.is_quantized)(QWEN_FP)


def test_filter_specs_implicit_and():
    out = list(q.filter_specs(ALL, q.is_quantized, q.family_is("llama")))
    assert out == [LLAMA_Q]
    # no predicates -> everything passes
    assert list(q.filter_specs(ALL)) == ALL
