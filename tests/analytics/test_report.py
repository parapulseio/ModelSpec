"""Coverage report — unknown_fields frequency, fill rates, promotion candidates."""

from __future__ import annotations

from modelspec.analytics.batch import BatchItem, BatchResult
from modelspec.analytics.report import build_coverage_report
from modelspec.schema import ModelSpec


def _spec(family: str, filled: list[str], unknown: list[str]) -> ModelSpec:
    """Build a spec with a chosen family, filled canonical fields and unknowns."""
    per_field = {p: {"source": "config", "confidence": "high"} for p in filled}
    return ModelSpec.model_validate(
        {
            "architecture": {"family": family},
            "provenance": {"per_field": per_field, "unknown_fields": unknown},
        }
    )


def _result(specs) -> BatchResult:
    return BatchResult(items=[BatchItem(target=f"m{i}", spec=s) for i, s in enumerate(specs)])


def test_unknown_field_frequency_and_promotion():
    specs = [
        _spec("llama", ["architecture.family"], ["decoder_sparse_step", "rare_a"]),
        _spec("llama", ["architecture.family"], ["decoder_sparse_step"]),
        _spec("qwen2", ["architecture.family"], ["decoder_sparse_step", "rare_b"]),
    ]
    report = build_coverage_report(_result(specs), promotion_threshold=0.5)

    freq = dict((f, (c, p)) for f, c, p in report.unknown_field_frequency)
    assert freq["decoder_sparse_step"] == (3, 1.0)  # in all 3 models
    # promotion: only fields >= 50% of models
    promoted = [f for f, _, _ in report.promotion_candidates]
    assert "decoder_sparse_step" in promoted
    assert "rare_a" not in promoted


def test_canonical_fill_rates():
    specs = [
        _spec("llama", ["architecture.family", "attention.num_kv_heads"], []),
        _spec("llama", ["architecture.family"], []),  # missing num_kv_heads
    ]
    report = build_coverage_report(_result(specs))
    assert report.canonical_fill_rates["architecture.family"] == 1.0
    assert report.canonical_fill_rates["attention.num_kv_heads"] == 0.5


def test_per_family_breakdown():
    specs = [
        _spec("llama", ["attention.num_kv_heads"], []),
        _spec("deepseek_v3", [], []),  # MLA: kv_heads legitimately absent
    ]
    report = build_coverage_report(_result(specs))
    assert report.family_counts == {"llama": 1, "deepseek_v3": 1}
    assert report.per_family_fill_rates["llama"]["attention.num_kv_heads"] == 1.0
    assert report.per_family_fill_rates["deepseek_v3"]["attention.num_kv_heads"] == 0.0


def test_failures_included():
    result = BatchResult(
        items=[
            BatchItem(target="ok", spec=_spec("llama", ["architecture.family"], [])),
            BatchItem(target="bad", error="ValueError: boom"),
        ]
    )
    report = build_coverage_report(result)
    assert report.succeeded == 1
    assert report.failed == 1
    assert report.failures == [("bad", "ValueError: boom")]
    # render must not raise
    assert "unknown_fields frequency" in report.render()
