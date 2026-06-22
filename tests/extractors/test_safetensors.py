"""safetensors extractor — build tiny header-only files, assert param counts."""

from __future__ import annotations

import json
from pathlib import Path

from modelspec.extractors.base import ExtractionSource
from modelspec.extractors.safetensors import SafetensorsExtractor
from tests.conftest import write_safetensors_header


def _claims(src: ExtractionSource) -> dict:
    result = SafetensorsExtractor().extract(src)
    return {c.field_path: c.value for c in result.claims}, result


def test_single_file_param_count(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": {"dtype": "BF16", "shape": [100, 16]},  # 1600
            "lm_head.weight": {"dtype": "BF16", "shape": [100, 16]},  # 1600
        },
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model.safetensors"])
    claims, _ = _claims(src)
    assert claims["parameters.total"] == 3200
    assert claims["parameters.dtype_native"] == "BF16"
    assert claims["architecture.tied_embeddings"] is False
    assert claims["identity.file_layout"] == "single"


def test_tied_embeddings_when_no_lm_head(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "F32", "shape": [10, 4]}},
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model.safetensors"])
    claims, _ = _claims(src)
    assert claims["architecture.tied_embeddings"] is True


def test_sharded_aggregates_all_shards(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model-00001-of-00002.safetensors",
        {"a.weight": {"dtype": "BF16", "shape": [10, 10]}},  # 100
    )
    write_safetensors_header(
        tmp_path / "model-00002-of-00002.safetensors",
        {"b.weight": {"dtype": "BF16", "shape": [10, 10]}},  # 100
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    src = ExtractionSource(
        root=tmp_path,
        repo_files=[
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    )
    claims, _ = _claims(src)
    # Must aggregate both shards — not half.
    assert claims["parameters.total"] == 200
    assert claims["identity.file_layout"] == "sharded"


def test_moe_tensor_pattern(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {"model.layers.0.block_sparse_moe.experts.0.w1.weight": {"dtype": "BF16", "shape": [4, 4]}},
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model.safetensors"])
    _, result = _claims(src)
    tag_claims = [c.value for c in result.claims if c.field_path == "architecture.tags"]
    assert ["moe"] in tag_claims
