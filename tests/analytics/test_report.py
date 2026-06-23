"""Coverage report — unknown_fields frequency, fill rates, promotion candidates."""

from __future__ import annotations

from modelspec.analytics.batch import BatchItem, BatchResult
from modelspec.analytics.report import build_coverage_report
from modelspec.schema import ModelSpec


def _spec(
    family: str, filled: list[str], unknown: list[str], na: list[str] | None = None
) -> ModelSpec:
    """Build a spec with a chosen family, filled canonical fields and unknowns."""
    per_field = {p: {"source": "config", "confidence": "high"} for p in filled}
    return ModelSpec.model_validate(
        {
            "architecture": {"family": family},
            "provenance": {
                "per_field": per_field,
                "unknown_fields": unknown,
                "not_applicable": na or [],
            },
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


def test_na_excluded_from_fill_denominator():
    # 1 GQA model with num_kv_heads filled + 1 MLA model where it is N/A.
    specs = [
        _spec("llama", ["attention.num_kv_heads"], []),
        _spec("deepseek_v3", [], [], na=["attention.num_kv_heads"]),
    ]
    report = build_coverage_report(_result(specs))
    # Naive denom would be 1/2 = 50%; over applicable models it is 1/1 = 100%.
    assert report.canonical_fill_rates["attention.num_kv_heads"] == 1.0
    assert report.na_counts["attention.num_kv_heads"] == 1
    # The MLA family's denominator is also reduced -> not counted as a 0% gap.
    assert report.per_family_fill_rates["deepseek_v3"]["attention.num_kv_heads"] == 0.0
    assert report.per_family_fill_rates["llama"]["attention.num_kv_heads"] == 1.0


def test_all_na_field_renders_without_low_flag():
    specs = [_spec("deepseek_v3", [], [], na=["attention.num_kv_heads"])]
    report = build_coverage_report(_result(specs))
    out = report.render()
    # Whole corpus N/A for this field -> shown as n/a, no "<-- low" alarm.
    assert "attention.num_kv_heads  (all 1 N/A)" in out
    assert "num_kv_heads  <-- low" not in out


def test_per_family_breakdown():
    specs = [
        _spec("llama", ["attention.num_kv_heads"], []),
        _spec("deepseek_v3", [], []),  # MLA: kv_heads legitimately absent
    ]
    report = build_coverage_report(_result(specs))
    assert report.family_counts == {"llama": 1, "deepseek_v3": 1}
    assert report.per_family_fill_rates["llama"]["attention.num_kv_heads"] == 1.0
    assert report.per_family_fill_rates["deepseek_v3"]["attention.num_kv_heads"] == 0.0


def test_conflict_field_histogram():
    def with_conflicts(conflicts):
        return ModelSpec.model_validate({"provenance": {"conflicts": conflicts}})

    vocab_conflict = {
        "field_path": "tokenizer.vocab_size", "value": 1000, "source": "config",
        "confidence": "high", "winner_source": "tensors", "winner_value": 1001,
    }
    ctx_conflict = {
        "field_path": "context.declared", "value": 4096, "source": "gguf",
        "confidence": "low", "winner_source": "config", "winner_value": 131072,
    }
    specs = [
        with_conflicts([vocab_conflict]),
        with_conflicts([vocab_conflict]),
        with_conflicts([ctx_conflict]),
    ]
    report = build_coverage_report(_result(specs))

    freq = {f: (c, p) for f, c, p in report.conflict_field_frequency}
    assert freq["tokenizer.vocab_size"] == (2, round(2 / 3, 4))
    assert freq["context.declared"][0] == 1
    # the dominant disagreeing source pair is surfaced
    assert report.conflict_sources["tokenizer.vocab_size"] == "tensors vs config"
    out = report.render()
    assert "conflict fields" in out
    assert "tokenizer.vocab_size" in out and "tensors vs config" in out


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
