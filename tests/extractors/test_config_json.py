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


def test_vlm_reads_text_config(tmp_path: Path):
    # Multimodal configs nest the LM fields under text_config; we must read them.
    cfg = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "vision_config": {"depth": 24, "hidden_size": 1024},
        "text_config": {
            "num_hidden_layers": 36,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 128000,
            "head_dim": 128,
        },
    }
    claims, _ = _claims(tmp_path, cfg)
    assert claims["architecture.num_layers"] == 36  # from text_config
    assert claims["attention.type"] == "gqa"
    assert claims["attention.num_kv_heads"] == 8
    assert claims["context.declared"] == 128000
    assert claims["architecture.head_dim"] == 128
    assert "multimodal" in claims["architecture.tags"]


def test_special_tokens_and_head_dim(tmp_path: Path):
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "bos_token_id": 1,
        "eos_token_id": [128001, 128009],  # Llama-3 style list
        "pad_token_id": 0,
    }
    claims, _ = _claims(tmp_path, cfg)
    assert claims["architecture.head_dim"] == 128
    assert claims["tokenizer.bos_token_id"] == 1
    assert claims["tokenizer.eos_token_id"] == [128001, 128009]
    assert claims["tokenizer.pad_token_id"] == 0


def test_mha_inferred_when_kv_heads_absent(tmp_path: Path):
    # Many configs omit num_key_value_heads (== MHA). We must still infer the
    # attention type and fill num_kv_heads from the query head count.
    cfg = {"architectures": ["GPTNeoXForCausalLM"], "num_attention_heads": 16}
    claims, _ = _claims(tmp_path, cfg)
    assert claims["attention.type"] == "mha"
    assert claims["attention.num_kv_heads"] == 16  # inferred = query heads


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


def test_mla_wins_over_gqa_and_marks_kv_heads_na(tmp_path: Path):
    # A realistic DeepSeek config has heads AND lora ranks: MLA must win, and
    # num_kv_heads must be flagged N/A rather than carried as a GQA grouping.
    cfg = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
    }
    write_config(tmp_path / "config.json", cfg)
    src = ExtractionSource(root=tmp_path, repo_files=["config.json"])
    result = ConfigJsonExtractor().extract(src)
    by_path = {c.field_path: c.value for c in result.claims}
    assert by_path["attention.type"] == "mla"  # not gqa
    assert "attention.num_kv_heads" not in by_path  # value suppressed
    assert "attention.num_kv_heads" in result.not_applicable


def test_vision_model_decoder_fields_marked_na(tmp_path: Path):
    # A pure vision model (no LM fields) must not count missing decoder fields
    # as gaps: they're flagged N/A, and it isn't tagged "decoder-only".
    cfg = {"architectures": ["CLIPModel"], "vision_config": {"num_hidden_layers": 24}}
    write_config(tmp_path / "config.json", cfg)
    src = ExtractionSource(root=tmp_path, repo_files=["config.json"])
    result = ConfigJsonExtractor().extract(src)
    tags = next(c.value for c in result.claims if c.field_path == "architecture.tags")
    assert "decoder-only" not in tags
    assert "vision" in tags
    assert "attention.type" in result.not_applicable
    assert "context.declared" in result.not_applicable


def test_encoder_keeps_filled_fields_not_na(tmp_path: Path):
    # BERT (encoder) does have layers/heads -> they stay filled, not N/A.
    cfg = {
        "architectures": ["BertForSequenceClassification"],
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_position_embeddings": 512,
    }
    write_config(tmp_path / "config.json", cfg)
    src = ExtractionSource(root=tmp_path, repo_files=["config.json"])
    result = ConfigJsonExtractor().extract(src)
    claims = {c.field_path: c.value for c in result.claims}
    tags = claims["architecture.tags"]
    assert "encoder" in tags and "decoder-only" not in tags
    assert claims["architecture.num_layers"] == 12  # filled, real signal
    assert "architecture.num_layers" not in result.not_applicable  # not wrongly N/A'd
    assert claims["attention.type"] == "mha"  # inferred from heads


def test_unknown_fields_reported(tmp_path: Path):
    cfg = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2, "enable_my_custom_thing": True}
    _, result = _claims(tmp_path, cfg)
    assert "enable_my_custom_thing" in result.unknown_fields
    assert result.raw["enable_my_custom_thing"] is True
