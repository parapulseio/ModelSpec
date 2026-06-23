"""GGUF extractor — build a tiny real GGUF, assert the FieldClaim list."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gguf")

from modelspec.extractors.base import ExtractionSource
from modelspec.extractors.gguf import GGUFExtractor
from tests.conftest import write_gguf


def _claims(tmp_path: Path, kv: dict, tensors: dict):
    write_gguf(tmp_path / "model.gguf", kv, tensors)
    src = ExtractionSource(root=tmp_path, repo_files=["model.gguf"])
    result = GGUFExtractor().extract(src)
    return {c.field_path: c.value for c in result.claims}, result


def test_truncated_gguf_prefix_still_parses(tmp_path: Path):
    # The remote path downloads only a ~24MB prefix (header + tensor infos, no
    # data). GGUFReader chokes on that ("cannot reshape ..."); our own parser
    # must succeed because it never touches tensor data.
    full = tmp_path / "model.gguf"
    write_gguf(
        full,
        kv={"general.architecture": "llama", "llama.block_count": 4},
        # one big F32 tensor so the data section dominates the file size
        tensors={"token_embd.weight": ([512, 8], "F32")},
    )
    raw = full.read_bytes()
    truncated = tmp_path / "trunc.gguf"
    truncated.write_bytes(raw[:400])  # keep header + tensor info, cut the data

    # The old approach (GGUFReader) would raise on this truncated file.
    from gguf import GGUFReader

    with pytest.raises(Exception):
        GGUFReader(truncated)

    src = ExtractionSource(root=tmp_path, repo_files=["trunc.gguf"])
    claims = {c.field_path: c.value for c in GGUFExtractor().extract(src).claims}
    assert claims["architecture.family"] == "llama"
    assert claims["architecture.num_layers"] == 4
    assert claims["parameters.total"] == 512 * 8  # tensor info parsed, data skipped


def test_llama_gguf_basic(tmp_path: Path):
    claims, result = _claims(
        tmp_path,
        kv={
            "general.architecture": "llama",
            "general.file_type": 15,  # Q4_K_M enum
            "llama.block_count": 4,
            "llama.embedding_length": 16,
            "llama.context_length": 4096,
            "llama.attention.head_count": 8,
            "llama.attention.head_count_kv": 2,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": ["a", "b", "c", "d", "e"],
        },
        tensors={"token_embd.weight": ([16, 8], "F32"), "output.weight": ([16, 8], "F32")},
    )
    assert claims["architecture.family"] == "llama"
    assert claims["architecture.num_layers"] == 4
    assert claims["context.declared"] == 4096
    assert claims["attention.type"] == "gqa"
    assert claims["attention.num_kv_heads"] == 2
    assert claims["tokenizer.type"] == "BPE"
    assert claims["tokenizer.vocab_size"] == 5
    assert claims["parameters.total"] == 256  # 16*8 + 16*8
    assert claims["parameters.dtype_native"] == "F32"
    # file_type is recognized-but-unmapped -> passthrough (consumed in M3).
    assert result.passthrough["general.file_type"] == 15
    # Large arrays are reduced to a length marker in the raw KV dump.
    assert result.raw["tokenizer.ggml.tokens"] == {"_array_len": 5}


def test_gguf_moe(tmp_path: Path):
    claims, _ = _claims(
        tmp_path,
        kv={
            "general.architecture": "qwen2moe",
            "qwen2moe.block_count": 2,
            "qwen2moe.attention.head_count": 8,
            "qwen2moe.attention.head_count_kv": 8,
            "qwen2moe.expert_count": 60,
            "qwen2moe.expert_used_count": 4,
        },
        tensors={"token_embd.weight": ([8, 8], "F32")},
    )
    assert claims["moe.num_experts"] == 60
    assert claims["moe.top_k"] == 4
    assert claims["attention.type"] == "mha"
