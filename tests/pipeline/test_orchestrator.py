"""End-to-end: a local fixture dir -> validated ModelSpec."""

from __future__ import annotations

from pathlib import Path

from modelspec.pipeline import detect_source_format, extract
from tests.conftest import write_config, write_safetensors_header


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
