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


def test_local_dir_recovers_repo_id_from_download_manifest(tmp_path: Path):
    # Simulates a directory previously produced by `--download-only`: the local
    # path is meaningless, but the manifest records the real HF repo_id.
    write_config(
        tmp_path / "config.json",
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2},
    )
    (tmp_path / "MODELSPEC_MANIFEST.md").write_text(
        "# ModelSpec download manifest\n\n- repo_id: org/some-model\n- revision: main\n",
        encoding="utf-8",
    )
    spec = extract(str(tmp_path), offline=True)
    assert spec.identity.repo_id == "org/some-model"
    # the manifest itself must not leak into the extracted file listing
    assert "MODELSPEC_MANIFEST.md" not in spec.provenance.unknown_fields


def test_local_dir_without_manifest_uses_path_as_repo_id(tmp_path: Path):
    write_config(
        tmp_path / "config.json",
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2},
    )
    spec = extract(str(tmp_path), offline=True)
    assert spec.identity.repo_id == str(tmp_path)


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


def test_quantization_claims_without_format_are_dropped(tmp_path: Path):
    from modelspec.extractors.base import FieldClaim
    from modelspec.pipeline.merger import merge_claims
    from modelspec.pipeline.orchestrator import reshape
    from modelspec.schema import ModelSpec

    merged = merge_claims(
        [
            FieldClaim("quantization.bits_per_weight_avg", 5.7, "tensors", "high"),
            FieldClaim("architecture.num_layers", 2, "config", "high"),
        ]
    )
    tree = reshape(
        merged,
        repo_id="x",
        source_format="hf",
        raw_config=None,
        raw_gguf=None,
        unknown_fields=[],
        not_applicable=[],
    )
    assert "quantization" not in tree
    assert "quantization.bits_per_weight_avg" not in tree["provenance"]["per_field"]
    spec = ModelSpec.model_validate(tree)
    assert spec.quantization is None
    assert spec.architecture.num_layers == 2
    assert any("5.7" in w for w in spec.provenance.warnings)
    assert spec.provenance.passthrough == {"quantization": {"bits_per_weight_avg": 5.7}}


def test_quantization_with_format_is_kept():
    from modelspec.extractors.base import FieldClaim
    from modelspec.pipeline.merger import merge_claims
    from modelspec.pipeline.orchestrator import reshape

    merged = merge_claims(
        [
            FieldClaim("quantization.format", "awq", "config", "high"),
            FieldClaim("quantization.bits", 4, "config", "high"),
        ]
    )
    tree = reshape(
        merged, repo_id=None, source_format="hf", raw_config=None, raw_gguf=None,
        unknown_fields=[], not_applicable=[],
    )
    assert tree["quantization"] == {"format": "awq", "bits": 4}
    assert tree["provenance"]["warnings"] == []


def test_packed_awq_param_count_passes_double_path_check(tmp_path: Path):
    # hidden=16, 1 layer, vocab=32, intermediate=32, MHA, tied embeddings.
    write_config(
        tmp_path / "config.json",
        {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "vocab_size": 32,
            "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 16},
        },
    )

    def awq(i: int, o: int) -> dict:
        return {
            "qweight": {"dtype": "I32", "shape": [i, o // 8]},
            "qzeros": {"dtype": "I32", "shape": [i // 16, o // 8]},
            "scales": {"dtype": "F16", "shape": [i // 16, o]},
        }

    tensors = {"model.embed_tokens.weight": {"dtype": "F16", "shape": [32, 16]}}
    for proj, (i, o) in {
        "self_attn.q_proj": (16, 16),
        "self_attn.k_proj": (16, 16),
        "self_attn.v_proj": (16, 16),
        "self_attn.o_proj": (16, 16),
        "mlp.gate_proj": (16, 32),
        "mlp.up_proj": (16, 32),
        "mlp.down_proj": (32, 16),
    }.items():
        for leaf, info in awq(i, o).items():
            tensors[f"model.layers.0.{proj}.{leaf}"] = info
    write_safetensors_header(tmp_path / "model.safetensors", tensors)

    spec = extract(str(tmp_path), offline=True)
    assert spec.parameters.total == 32 * 16 + 4 * 256 + 3 * 512
    assert not any("parameter count mismatch" in w for w in spec.provenance.warnings)


def test_quantized_modules_exposed_in_provenance_passthrough(tmp_path: Path):
    write_config(
        tmp_path / "config.json",
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    )
    write_safetensors_header(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": {"dtype": "F16", "shape": [10, 256]},
            "model.layers.0.mlp.down_proj.qweight": {"dtype": "I32", "shape": [256, 16]},
            "model.layers.0.mlp.down_proj.scales": {"dtype": "F16", "shape": [2, 128]},
        },
    )
    spec = extract(str(tmp_path), offline=True)
    qm = spec.provenance.passthrough["safetensors"]["quantized_modules"]
    assert qm["quantized"] == ["model.layers.0.mlp.down_proj"]
    assert qm["unquantized"] == ["model.embed_tokens.weight"]
