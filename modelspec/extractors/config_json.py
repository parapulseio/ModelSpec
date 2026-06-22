"""config.json extractor — three-layer extraction with alias normalization.

We do NOT try to cover every config field. The canonical layer is a small,
strictly-controlled set; everything recognized-but-unmapped goes to passthrough;
the full original dict is kept as raw. See docs/extractors.md.
"""

from __future__ import annotations

import json

from modelspec.extractors.base import ExtractionSource, ExtractorResult, FieldClaim

# Same canonical field may appear under several historical HF names
# (GPT-2 style vs Llama style). First match wins.
ALIASES: dict[str, list[str]] = {
    "architecture.num_layers": ["num_hidden_layers", "n_layer", "n_layers"],
    "architecture.hidden_size": ["hidden_size", "n_embd", "d_model"],
    "context.declared": ["max_position_embeddings", "n_positions", "seq_length"],
    "attention.num_heads": ["num_attention_heads", "n_head", "num_heads"],
    "attention.num_kv_heads": ["num_key_value_heads", "num_kv_heads"],
    "attention.sliding_window": ["sliding_window"],
    "tokenizer.vocab_size": ["vocab_size"],
}

# Map common architectures[0] class names to a normalized family label.
_FAMILY_REGISTRY: dict[str, str] = {
    "llamaforcausallm": "llama",
    "mistralforcausallm": "mistral",
    "mixtralforcausallm": "mixtral",
    "qwen2forcausallm": "qwen2",
    "qwen2moeforcausallm": "qwen2_moe",
    "qwen3forcausallm": "qwen3",
    "qwen3moeforcausallm": "qwen3_moe",
    "gemmaforcausallm": "gemma",
    "gemma2forcausallm": "gemma2",
    "phi3forcausallm": "phi3",
    "deepseekv2forcausallm": "deepseek_v2",
    "deepseekv3forcausallm": "deepseek_v3",
}

# Recognized but not yet mapped to canonical — kept verbatim as a buffer.
KNOWN_PASSTHROUGH = {
    "rope_scaling",
    "rope_theta",
    "quantization_config",
    "pretraining_tp",
    "attention_bias",
    "attention_dropout",
    "torch_dtype",
    "tie_word_embeddings",
    "intermediate_size",
    "num_local_experts",
    "num_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "q_lora_rank",
    "kv_lora_rank",
    "model_type",
    "architectures",
}


def _infer_family(raw: dict) -> str | None:
    """Infer the family from architectures[0].

    NOTE: under trust_remote_code this is a custom class name and may not be in
    the registry — we never execute modeling_*.py; tensor-name fallback (in the
    safetensors extractor) handles those cases.
    """
    archs = raw.get("architectures")
    if isinstance(archs, list) and archs:
        key = str(archs[0]).lower()
        if key in _FAMILY_REGISTRY:
            return _FAMILY_REGISTRY[key]
        # Heuristic: strip the common task suffix.
        stripped = key.replace("forcausallm", "").replace("model", "")
        return stripped or None
    mt = raw.get("model_type")
    return str(mt) if mt else None


def _covered_keys() -> set[str]:
    keys = set(KNOWN_PASSTHROUGH)
    for candidates in ALIASES.values():
        keys.update(candidates)
    return keys


def _extract_quantization(raw: dict) -> list[FieldClaim]:
    """Emit quantization claims from config.json ``quantization_config``.

    M3 recognizes AWQ and GPTQ; an unknown quant_method emits nothing (the field
    stays None and the discriminated union never sees an unknown discriminator),
    while the original quantization_config is still archived via passthrough/raw.
    """
    qc = raw.get("quantization_config")
    if not isinstance(qc, dict):
        return []
    method = str(qc.get("quant_method", "")).lower()
    claims: list[FieldClaim] = []
    if method == "awq":
        claims.append(FieldClaim("quantization.format", "awq", "config", "high"))
        if "bits" in qc:
            claims.append(FieldClaim("quantization.bits", qc["bits"], "config", "high"))
        if "group_size" in qc:
            claims.append(
                FieldClaim("quantization.group_size", qc["group_size"], "config", "high")
            )
        if "zero_point" in qc:
            claims.append(
                FieldClaim("quantization.zero_point", qc["zero_point"], "config", "high")
            )
    elif method == "gptq":
        claims.append(FieldClaim("quantization.format", "gptq", "config", "high"))
        if "bits" in qc:
            claims.append(FieldClaim("quantization.bits", qc["bits"], "config", "high"))
        if "group_size" in qc:
            claims.append(
                FieldClaim("quantization.group_size", qc["group_size"], "config", "high")
            )
        if "desc_act" in qc:
            claims.append(
                FieldClaim("quantization.desc_act", qc["desc_act"], "config", "high")
            )
    return claims


class ConfigJsonExtractor:
    name = "config_json"

    def can_handle(self, source: ExtractionSource) -> bool:
        return source.has("config.json")

    def extract(self, source: ExtractionSource) -> ExtractorResult:
        raw = json.loads(source.path("config.json").read_text(encoding="utf-8"))
        claims: list[FieldClaim] = []

        # --- canonical layer: alias normalization ---
        for canonical_path, candidates in ALIASES.items():
            for name in candidates:
                if name in raw:
                    claims.append(FieldClaim(canonical_path, raw[name], "config", "high"))
                    break

        family = _infer_family(raw)
        if family:
            claims.append(FieldClaim("architecture.family", family, "config", "high"))

        # --- feature inference -> architecture tags & attention type ---
        tags: list[str] = ["decoder-only"]

        n_heads = raw.get("num_attention_heads")
        n_kv = raw.get("num_key_value_heads")
        if n_kv is not None and n_heads is not None:
            if n_kv == 1:
                attn_type, tag = "mqa", "mqa"
            elif n_kv != n_heads:
                attn_type, tag = "gqa", "gqa"
            else:
                attn_type, tag = "mha", "mha"
            claims.append(FieldClaim("attention.type", attn_type, "inferred", "high"))
            tags.append(tag)

        # MLA (DeepSeek style) overrides the head-count heuristic.
        if "kv_lora_rank" in raw or "q_lora_rank" in raw:
            claims.append(FieldClaim("attention.type", "mla", "inferred", "high"))
            tags.append("mla")

        # MoE detection.
        n_experts = raw.get("num_local_experts") or raw.get("num_experts")
        if n_experts:
            claims.append(FieldClaim("moe.num_experts", n_experts, "config", "high"))
            top_k = raw.get("num_experts_per_tok")
            if top_k is not None:
                claims.append(FieldClaim("moe.top_k", top_k, "config", "high"))
            shared = raw.get("n_shared_experts")
            if shared is not None:
                claims.append(FieldClaim("moe.shared_experts", shared, "config", "high"))
            tags.append("moe")

        # RoPE scaling -> tag + effective context.
        rope = raw.get("rope_scaling")
        if isinstance(rope, dict):
            rope_type = rope.get("type") or rope.get("rope_type")
            if rope_type:
                tags.append(f"rope-{rope_type}")
            factor = rope.get("factor")
            declared = raw.get("max_position_embeddings")
            if isinstance(factor, (int, float)) and isinstance(declared, int):
                claims.append(
                    FieldClaim("context.effective", int(declared * factor), "inferred", "medium")
                )

        if raw.get("sliding_window"):
            tags.append("sliding-window")

        # tie_word_embeddings is a declared hint; tensor presence is authoritative.
        if raw.get("tie_word_embeddings") is True:
            tags.append("tied-embed")
            claims.append(
                FieldClaim("architecture.tied_embeddings", True, "config", "medium")
            )

        # Quantization (AWQ / GPTQ branch of the discriminated union).
        claims.extend(_extract_quantization(raw))

        claims.append(FieldClaim("architecture.tags", tags, "inferred", "medium"))

        # --- passthrough + raw + unknown ---
        passthrough = {k: raw[k] for k in KNOWN_PASSTHROUGH if k in raw}
        unknown = sorted(set(raw) - _covered_keys())

        return ExtractorResult(
            claims=claims,
            passthrough=passthrough,
            raw=raw,
            unknown_fields=unknown,
        )
