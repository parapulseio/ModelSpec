"""End-to-end: a local fixture dir -> validated ModelSpec."""

from __future__ import annotations

from pathlib import Path

from modelspec.pipeline import detect_source_format, extract
from tests.conftest import write_config, write_safetensors_header


def test_not_applicable_plumbed_end_to_end(tmp_path: Path):
    write_config(
        tmp_path / "config.json",
        {
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
        },
    )
    spec = extract(str(tmp_path), offline=True)
    assert spec.attention.type == "mla"
    assert spec.attention.num_kv_heads is None  # suppressed, not a GQA grouping
    assert "attention.num_kv_heads" in spec.provenance.not_applicable


def test_detect_source_format():
    assert detect_source_format(["config.json", "model.safetensors"]) == "hf"
    assert detect_source_format(["model.gguf"]) == "gguf"
    assert detect_source_format(["adapter_config.json"]) == "adapter"
    assert detect_source_format(["model.safetensors"]) == "raw"


def test_local_hf_model_end_to_end(tmp_path: Path):
    write_config(
        tmp_path / "config.json",
        {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
        },
    )
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 16]},  # 512
            "lm_head.weight": {"dtype": "BF16", "shape": [32, 16]},  # 512
        },
    )
    spec = extract(str(tmp_path), offline=True)

    assert spec.identity.source_format == "hf"
    assert spec.architecture.family == "llama"
    assert spec.architecture.num_layers == 2
    assert spec.attention.type == "gqa"
    assert spec.parameters.total == 1024
    assert spec.parameters.dtype_native == "BF16"
    assert spec.architecture.tied_embeddings is False
    # tags unioned across config + tensors
    assert "decoder-only" in spec.architecture.tags
    assert "gqa" in spec.architecture.tags
    # provenance populated
    assert "architecture.num_layers" in spec.provenance.per_field


def test_offline_rejects_nonexistent(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        extract("meta-llama/Llama-3.1-8B", offline=True)


def test_merged_quantized_model_end_to_end(tmp_path: Path):
    # A merge (mergekit_config.yml) + AWQ quantization (config.json) — the two
    # orthogonal structures must coexist on one spec.
    write_config(
        tmp_path / "config.json",
        {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
        },
    )
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 16]}},
    )
    (tmp_path / "mergekit_config.yml").write_text(
        "merge_method: dare-ties\nmodels:\n  - model: org/a\n  - model: org/b\n"
    )
    spec = extract(str(tmp_path), offline=True)

    # quantization branch
    assert spec.quantization is not None
    assert spec.quantization.format == "awq"
    assert spec.quantization.bits == 4
    # merge branch (orthogonal, coexists)
    assert spec.merge is not None
    assert spec.merge.method == "dare_ties"  # alias-normalized
    assert {c.model_id for c in spec.merge.components} == {"org/a", "org/b"}
    # lineage chain
    assert spec.identity.lineage is not None
    assert spec.identity.lineage.relation == "merge"


def test_local_gguf_model_end_to_end(tmp_path: Path):
    import pytest

    pytest.importorskip("gguf")
    from tests.conftest import write_gguf

    write_gguf(
        tmp_path / "model-Q4_K_M.gguf",
        kv={
            "general.architecture": "llama",
            "general.file_type": 15,
            "llama.block_count": 4,
            "llama.embedding_length": 16,
            "llama.context_length": 8192,
            "llama.attention.head_count": 8,
            "llama.attention.head_count_kv": 2,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": ["x"] * 32,
        },
        tensors={"token_embd.weight": ([16, 8], "F32")},
    )
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n")

    spec = extract(str(tmp_path), offline=True)
    assert spec.identity.source_format == "gguf"
    assert spec.architecture.family == "llama"
    assert spec.attention.type == "gqa"
    assert spec.tokenizer.vocab_size == 32
    assert spec.license.spdx_id == "apache-2.0"
    # raw GGUF KV dump is archived under provenance.
    assert spec.provenance.raw_gguf_kv is not None
