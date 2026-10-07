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


def _packed_claims(tmp_path: Path, tensors: dict) -> dict:
    write_safetensors_header(tmp_path / "model.safetensors", tensors)
    src = ExtractionSource(root=tmp_path, repo_files=["model.safetensors"])
    return _claims(src)[0]


def test_awq_packed_param_count(tmp_path: Path):
    # in=256, out=128, 4-bit (pack=8), group=128 -> AWQ GEMM layout
    claims = _packed_claims(
        tmp_path,
        {
            "l.qweight": {"dtype": "I32", "shape": [256, 16]},
            "l.qzeros": {"dtype": "I32", "shape": [2, 16]},
            "l.scales": {"dtype": "F16", "shape": [2, 128]},
            "norm.weight": {"dtype": "F16", "shape": [256]},
        },
    )
    assert claims["parameters.total"] == 256 * 128 + 256
    assert claims["parameters.dtype_native"] == "F16"


def test_gptq_packed_param_count_with_and_without_g_idx(tmp_path: Path):
    base = {
        "l.qweight": {"dtype": "I32", "shape": [32, 128]},  # in/pack=32 -> in=256
        "l.qzeros": {"dtype": "I32", "shape": [2, 16]},
        "l.scales": {"dtype": "F16", "shape": [2, 128]},
    }
    assert _packed_claims(tmp_path, base)["parameters.total"] == 256 * 128
    with_g = {**base, "l.g_idx": {"dtype": "I32", "shape": [256]}}
    assert _packed_claims(tmp_path, with_g)["parameters.total"] == 256 * 128


def test_awq_without_qzeros(tmp_path: Path):
    claims = _packed_claims(
        tmp_path,
        {
            "l.qweight": {"dtype": "I32", "shape": [256, 16]},
            "l.scales": {"dtype": "F16", "shape": [2, 128]},
        },
    )
    assert claims["parameters.total"] == 256 * 128
    assert claims["parameters.dtype_native"] == "F16"


def test_gptq_g_idx_without_qzeros(tmp_path: Path):
    claims = _packed_claims(
        tmp_path,
        {
            "l.qweight": {"dtype": "I32", "shape": [32, 128]},
            "l.scales": {"dtype": "F16", "shape": [2, 128]},
            "l.g_idx": {"dtype": "I32", "shape": [256]},
        },
    )
    assert claims["parameters.total"] == 256 * 128


def test_underivable_gptq_lowers_confidence(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {
            "l.qweight": {"dtype": "I32", "shape": [32, 128]},
            "l.scales": {"dtype": "F16", "shape": [2, 128]},
        },
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model.safetensors"])
    result = SafetensorsExtractor().extract(src)
    total = next(c for c in result.claims if c.field_path == "parameters.total")
    assert total.confidence == "medium"


def test_packed_tensors_aggregate_across_shards(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model-00001-of-00002.safetensors",
        {
            "a.qweight": {"dtype": "I32", "shape": [256, 16]},
            "a.qzeros": {"dtype": "I32", "shape": [2, 16]},
            "a.scales": {"dtype": "F16", "shape": [2, 128]},
        },
    )
    write_safetensors_header(
        tmp_path / "model-00002-of-00002.safetensors",
        {
            "b.qweight": {"dtype": "I32", "shape": [32, 128]},
            "b.qzeros": {"dtype": "I32", "shape": [2, 16]},
            "b.scales": {"dtype": "F16", "shape": [2, 128]},
            "b.g_idx": {"dtype": "I32", "shape": [256]},
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.qweight": "model-00001-of-00002.safetensors",
                    "b.qweight": "model-00002-of-00002.safetensors",
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
    assert _claims(src)[0]["parameters.total"] == 2 * 256 * 128


def test_quantized_modules_passthrough(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": {"dtype": "F16", "shape": [10, 256]},
            "model.layers.0.mlp.down_proj.qweight": {"dtype": "I32", "shape": [256, 16]},
            "model.layers.0.mlp.down_proj.qzeros": {"dtype": "I32", "shape": [2, 16]},
            "model.layers.0.mlp.down_proj.scales": {"dtype": "F16", "shape": [2, 128]},
            "model.layers.0.input_layernorm.weight": {"dtype": "F16", "shape": [256]},
            "lm_head.weight": {"dtype": "F16", "shape": [10, 256]},
        },
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model.safetensors"])
    result = SafetensorsExtractor().extract(src)
    assert result.passthrough["quantized_modules"] == {
        "quantized": ["model.layers.0.mlp.down_proj"],
        "unquantized": ["lm_head.weight", "model.embed_tokens.weight"],
    }


def test_no_quantized_modules_passthrough_for_plain_checkpoint(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model.safetensors", {"w.weight": {"dtype": "F16", "shape": [4, 4]}}
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model.safetensors"])
    assert "quantized_modules" not in SafetensorsExtractor().extract(src).passthrough


def test_gpt2_style_embeddings_reported_unquantized(tmp_path: Path):
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {
            "transformer.wte.weight": {"dtype": "F16", "shape": [10, 256]},
            "transformer.wpe.weight": {"dtype": "F16", "shape": [8, 256]},
            "transformer.h.0.mlp.c_proj.qweight": {"dtype": "I32", "shape": [256, 16]},
            "transformer.h.0.mlp.c_proj.scales": {"dtype": "F16", "shape": [2, 128]},
        },
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model.safetensors"])
    qm = SafetensorsExtractor().extract(src).passthrough["quantized_modules"]
    assert qm["unquantized"] == ["transformer.wpe.weight", "transformer.wte.weight"]
