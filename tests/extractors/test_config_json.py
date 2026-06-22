"""config.json extractor — feed a fixture, assert the FieldClaim list."""

from __future__ import annotations

from pathlib import Path

from modelspec.extractors.base import ExtractionSource
from modelspec.extractors.config_json import ConfigJsonExtractor
from tests.conftest import write_config


def _claims(tmp_path: Path, config: dict) -> dict:
    write_config(tmp_path / "config.json", config)
    src = ExtractionSource(root=tmp_path, repo_files=["config.json"])
    result = ConfigJsonExtractor().extract(src)
    # Flatten to {field_path: value} for easy assertions (last write wins).
    return {c.field_path: c.value for c in result.claims}, result


def test_llama_gqa(tmp_path: Path):
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "vocab_size": 128256,
    }
    claims, _ = _claims(tmp_path, cfg)
    assert claims["architecture.family"] == "llama"
    assert claims["architecture.num_layers"] == 32
    assert claims["attention.type"] == "gqa"
    assert claims["attention.num_kv_heads"] == 8
    assert claims["context.declared"] == 131072


def test_alias_gpt2_style(tmp_path: Path):
    cfg = {"architectures": ["GPT2LMHeadModel"], "n_layer": 12, "n_embd": 768, "n_positions": 1024}
    claims, _ = _claims(tmp_path, cfg)
    assert claims["architecture.num_layers"] == 12
    assert claims["architecture.hidden_size"] == 768
    assert claims["context.declared"] == 1024


def test_moe_detection(tmp_path: Path):
    cfg = {
        "architectures": ["MixtralForCausalLM"],
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }
    claims, result = _claims(tmp_path, cfg)
    assert claims["moe.num_experts"] == 8
    assert claims["moe.top_k"] == 2
    assert "moe" in claims["architecture.tags"]


def test_mla_overrides(tmp_path: Path):
    cfg = {"architectures": ["DeepseekV3ForCausalLM"], "kv_lora_rank": 512, "q_lora_rank": 1536}
    claims, _ = _claims(tmp_path, cfg)
    assert claims["attention.type"] == "mla"


def test_unknown_fields_reported(tmp_path: Path):
    cfg = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2, "enable_my_custom_thing": True}
    _, result = _claims(tmp_path, cfg)
    assert "enable_my_custom_thing" in result.unknown_fields
    assert result.raw["enable_my_custom_thing"] is True
