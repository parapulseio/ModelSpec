"""Quantization claims — AWQ / GPTQ from config.json, GGUF from tensor types."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelspec.extractors.base import ExtractionSource
from modelspec.extractors.config_json import ConfigJsonExtractor
from tests.conftest import write_config


def _config_claims(tmp_path: Path, config: dict) -> dict:
    write_config(tmp_path / "config.json", config)
    src = ExtractionSource(root=tmp_path, repo_files=["config.json"])
    result = ConfigJsonExtractor().extract(src)
    return {c.field_path: c.value for c in result.claims}


def test_awq_from_config(tmp_path: Path):
    claims = _config_claims(
        tmp_path,
        {
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            },
        },
    )
    assert claims["quantization.format"] == "awq"
    assert claims["quantization.bits"] == 4
    assert claims["quantization.group_size"] == 128
    assert claims["quantization.zero_point"] is True


def test_gptq_from_config(tmp_path: Path):
    claims = _config_claims(
        tmp_path,
        {
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
            },
        },
    )
    assert claims["quantization.format"] == "gptq"
    assert claims["quantization.desc_act"] is False


def test_unknown_quant_method_emits_nothing(tmp_path: Path):
    claims = _config_claims(
        tmp_path,
        {"architectures": ["LlamaForCausalLM"], "quantization_config": {"quant_method": "exotic"}},
    )
    assert "quantization.format" not in claims


# --- GGUF quantization ---

pytest.importorskip("gguf")
from modelspec.extractors.gguf import (  # noqa: E402
    GGUFExtractor,
    _avg_bits_per_weight,
    _filetype_name,
)
from tests.conftest import write_gguf  # noqa: E402


def test_filetype_name():
    assert _filetype_name(15) == "Q4_K_M"
    assert _filetype_name(99999) is None


def test_avg_bits_per_weight_block_quant():
    from gguf.constants import GGMLQuantizationType

    # All weights Q4_K: 144 bytes / 256 elems * 8 = 4.5 bpw.
    bpw = _avg_bits_per_weight({GGMLQuantizationType.Q4_K: 256}, 256)
    assert bpw == 4.5


def test_gguf_quantization_end_to_end(tmp_path: Path):
    write_gguf(
        tmp_path / "model-Q4_K_M.gguf",
        kv={
            "general.architecture": "llama",
            "general.file_type": 15,  # MOSTLY_Q4_K_M
            "llama.block_count": 1,
            "llama.attention.head_count": 8,
            "llama.attention.head_count_kv": 8,
        },
        tensors={
            "blk.0.attn_q.weight": ([4, 256], "Q4_K"),  # 1024 elems quantized
            "output_norm.weight": ([4], "F32"),  # tiny full-precision tensor
        },
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model-Q4_K_M.gguf"])
    claims = {c.field_path: c.value for c in GGUFExtractor().extract(src).claims}
    assert claims["quantization.format"] == "gguf"
    assert claims["quantization.file_type"] == "Q4_K_M"
    assert claims["quantization.bits_per_weight_avg"] is not None
    assert "Q4_K" in claims["quantization.tensor_types"]


def test_unquantized_gguf_has_no_quantization(tmp_path: Path):
    write_gguf(
        tmp_path / "model-f16.gguf",
        kv={"general.architecture": "llama", "general.file_type": 1},
        tensors={"token_embd.weight": ([8, 8], "F16")},
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model-f16.gguf"])
    claims = {c.field_path: c.value for c in GGUFExtractor().extract(src).claims}
    assert "quantization.format" not in claims


def _imatrix_claims(tmp_path: Path, filename: str, extra_kv: dict):
    write_gguf(
        tmp_path / filename,
        kv={"general.architecture": "llama", "general.file_type": 15, **extra_kv},
        tensors={"blk.0.attn_q.weight": ([4, 256], "Q4_K")},
    )
    src = ExtractionSource(root=tmp_path, repo_files=[filename])
    return GGUFExtractor().extract(src)


def test_imatrix_kv_gives_high_confidence(tmp_path: Path):
    res = _imatrix_claims(
        tmp_path,
        "model-Q4_K_M.gguf",
        {"quantize.imatrix.file": "imatrix.dat", "quantize.imatrix.dataset": "wiki.train.raw"},
    )
    claim = next(c for c in res.claims if c.field_path == "quantization.has_imatrix")
    assert (claim.value, claim.source, claim.confidence) == (True, "gguf", "high")
    assert res.passthrough["quantize.imatrix.dataset"] == "wiki.train.raw"
    assert res.passthrough["quantize.imatrix.file"] == "imatrix.dat"


def test_imatrix_kv_beats_filename_heuristic(tmp_path: Path):
    res = _imatrix_claims(tmp_path, "model-imat-Q4_K_M.gguf", {"quantize.imatrix.entries_count": 3})
    claims = [c for c in res.claims if c.field_path == "quantization.has_imatrix"]
    assert len(claims) == 1 and claims[0].confidence == "high"


def test_imatrix_filename_only_is_low_confidence(tmp_path: Path):
    res = _imatrix_claims(tmp_path, "model-imat-Q4_K_M.gguf", {})
    claim = next(c for c in res.claims if c.field_path == "quantization.has_imatrix")
    assert (claim.value, claim.source, claim.confidence) == (True, "heuristic", "low")


def test_no_imatrix_evidence_emits_no_claim(tmp_path: Path):
    res = _imatrix_claims(tmp_path, "model-Q4_K_M.gguf", {})
    assert all(c.field_path != "quantization.has_imatrix" for c in res.claims)
    assert not any(k.startswith("quantize.imatrix") for k in res.passthrough)


def test_mixed_precision_layout_in_passthrough(tmp_path: Path):
    types = {0: "Q6_K", 1: "Q4_K", 2: "Q4_K", 3: "Q6_K"}
    tensors = {f"blk.{i}.ffn_down.weight": ([4, 256], t) for i, t in types.items()}
    tensors["token_embd.weight"] = ([4, 256], "Q4_K")
    write_gguf(
        tmp_path / "model-Q4_K_M.gguf",
        kv={"general.architecture": "llama", "general.file_type": 15},
        tensors=tensors,
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model-Q4_K_M.gguf"])
    res = GGUFExtractor().extract(src)
    layout = res.passthrough["tensor_type_layout"]
    assert layout["ffn_down.weight"] == {"Q6_K": [0, 3], "Q4_K": [1, 2]}
    assert layout["token_embd.weight"] == {"Q4_K": []}
    assert all(c.field_path != "quantization.tensor_type_layout" for c in res.claims)


def test_unquantized_gguf_has_no_layout(tmp_path: Path):
    write_gguf(
        tmp_path / "model-f16.gguf",
        kv={"general.architecture": "llama", "general.file_type": 1},
        tensors={"token_embd.weight": ([8, 8], "F16")},
    )
    src = ExtractionSource(root=tmp_path, repo_files=["model-f16.gguf"])
    assert "tensor_type_layout" not in GGUFExtractor().extract(src).passthrough
