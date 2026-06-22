"""Schema — feed dicts, assert validation behaviour."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modelspec.schema import ModelSpec


def test_minimal_dict_validates():
    spec = ModelSpec.model_validate({})
    assert spec.spec_version == "1.0"
    assert spec.identity.source_format == "unknown"
    assert spec.architecture.tags == []
    assert spec.moe is None


def test_type_error_is_raised():
    with pytest.raises(ValidationError):
        ModelSpec.model_validate({"parameters": {"total": "7B"}})


def test_bad_enum_rejected():
    with pytest.raises(ValidationError):
        ModelSpec.model_validate({"attention": {"type": "weird"}})


def test_cross_field_warning_for_indivisible_heads():
    spec = ModelSpec.model_validate(
        {"attention": {"num_heads": 7, "num_kv_heads": 2}}
    )
    assert any("divisible" in w for w in spec.provenance.warnings)


def test_orthogonal_structures_default_none():
    spec = ModelSpec.model_validate({})
    assert spec.quantization is None
    assert spec.merge is None
    assert spec.adapter is None


def test_json_schema_exports():
    schema = ModelSpec.model_json_schema()
    assert schema["title"] == "ModelSpec"
    assert "identity" in schema["properties"]
