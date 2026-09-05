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


def test_gguf_single_file_layout(tmp_path: Path):
    claims, _ = _claims(
        tmp_path,
        kv={"general.architecture": "llama", "llama.block_count": 4},
        tensors={"token_embd.weight": ([8, 8], "F32")},
    )
    assert claims["identity.file_layout"] == "single"


def test_gguf_sharded_via_split_count_kv(tmp_path: Path):
    # A split-aware writer (llama.cpp's gguf-split) embeds split.count in every
    # part's own header — this is enough to detect sharding without reading
    # the other parts.
    claims, _ = _claims(
        tmp_path,
        kv={
            "general.architecture": "llama",
            "llama.block_count": 4,
            "split.count": 3,
        },
        tensors={"token_embd.weight": ([8, 8], "F32")},
    )
    assert claims["identity.file_layout"] == "sharded"


def test_gguf_sharded_via_filename_pattern(tmp_path: Path):
    # Fallback signal when split.count is absent: the "-NNNNN-of-MMMMM.gguf"
    # naming convention gguf-split uses.
    path = tmp_path / "model-00001-of-00003.gguf"
    write_gguf(
        path,
        kv={"general.architecture": "llama", "llama.block_count": 4},
        tensors={"token_embd.weight": ([8, 8], "F32")},
    )
    src = ExtractionSource(root=tmp_path, repo_files=[path.name])
    claims = {c.field_path: c.value for c in GGUFExtractor().extract(src).claims}
    assert claims["identity.file_layout"] == "sharded"


def test_gguf_single_part_split_naming_not_sharded(tmp_path: Path):
    # "-00001-of-00001" is a single-part file that merely went through
    # gguf-split's naming; must not be flagged as sharded.
    path = tmp_path / "model-00001-of-00001.gguf"
    write_gguf(
        path,
        kv={"general.architecture": "llama", "llama.block_count": 4},
        tensors={"token_embd.weight": ([8, 8], "F32")},
    )
    src = ExtractionSource(root=tmp_path, repo_files=[path.name])
    claims = {c.field_path: c.value for c in GGUFExtractor().extract(src).claims}
    assert claims["identity.file_layout"] == "single"


def _write_shard(tmp_path: Path, part: int, total: int, kv: dict, tensors: dict) -> str:
    name = f"model-{part:05d}-of-{total:05d}.gguf"
    write_gguf(tmp_path / name, kv, tensors)
    return name


def test_gguf_shard_aggregation(tmp_path: Path):
    # Only part 1 carries the full architecture metadata (the realistic case
    # for older gguf-split output); parts 2 and 3 add tensors only.
    part1 = _write_shard(
        tmp_path,
        1,
        3,
        kv={
            "general.architecture": "llama",
            "llama.block_count": 4,
            "llama.embedding_length": 16,
            "split.no": 0,
            "split.count": 3,
        },
        tensors={"token_embd.weight": ([16, 8], "F32")},
    )
    part2 = _write_shard(
        tmp_path,
        2,
        3,
        kv={"general.architecture": "llama", "split.no": 1, "split.count": 3},
        tensors={"blk.0.attn_q.weight": ([16, 8], "F32")},
    )
    part3 = _write_shard(
        tmp_path,
        3,
        3,
        kv={"general.architecture": "llama", "split.no": 2, "split.count": 3},
        tensors={"output.weight": ([16, 8], "F32")},
    )

    # repo_files order should not matter — sibling lookup sorts by part number.
    src = ExtractionSource(root=tmp_path, repo_files=[part3, part1, part2])
    claims = {c.field_path: c.value for c in GGUFExtractor().extract(src).claims}

    assert claims["identity.file_layout"] == "sharded"
    assert claims["architecture.num_layers"] == 4  # only on part 1
    assert claims["architecture.hidden_size"] == 16  # only on part 1
    assert claims["parameters.total"] == 16 * 8 * 3  # summed across all 3 parts


def test_gguf_shard_aggregation_missing_part_is_partial(tmp_path: Path):
    # If a sibling part isn't downloaded/present locally, aggregation can only
    # use what's there; layout must still say "sharded" (via the split.count
    # KV fallback), not silently look complete.
    part1 = _write_shard(
        tmp_path,
        1,
        2,
        kv={"general.architecture": "llama", "llama.block_count": 4, "split.count": 2},
        tensors={"token_embd.weight": ([16, 8], "F32")},
    )
    src = ExtractionSource(root=tmp_path, repo_files=[part1])
    claims = {c.field_path: c.value for c in GGUFExtractor().extract(src).claims}

    assert claims["identity.file_layout"] == "sharded"
    assert claims["parameters.total"] == 16 * 8  # only the one local part


def test_gguf_distinct_quant_variants_not_merged(tmp_path: Path):
    # Two independent quantizations of the same model living in one repo must
    # not be mistaken for split parts of each other.
    write_gguf(
        tmp_path / "model-Q4_K_M.gguf",
        kv={"general.architecture": "llama", "llama.block_count": 4},
        tensors={"token_embd.weight": ([16, 8], "F32")},
    )
    write_gguf(
        tmp_path / "model-Q8_0.gguf",
        kv={"general.architecture": "llama", "llama.block_count": 4},
        tensors={"token_embd.weight": ([16, 8], "F32"), "output.weight": ([16, 8], "F32")},
    )
    src = ExtractionSource(
        root=tmp_path, repo_files=["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
    )
    claims = {c.field_path: c.value for c in GGUFExtractor().extract(src).claims}

    assert claims["identity.file_layout"] == "single"
    assert claims["parameters.total"] == 16 * 8  # only the anchor file's own tensor
