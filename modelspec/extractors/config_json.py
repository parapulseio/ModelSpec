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
    "architecture.num_layers": ["num_hidden_layers", "n_layer", "n_layers", "num_layers"],
    "architecture.hidden_size": ["hidden_size", "n_embd", "d_model"],
    "architecture.head_dim": ["head_dim"],
    "context.declared": ["max_position_embeddings", "n_positions", "seq_length"],
    "attention.num_heads": ["num_attention_heads", "n_head", "num_heads"],
    "attention.num_kv_heads": ["num_key_value_heads", "num_kv_heads"],
    "attention.sliding_window": ["sliding_window"],
    "tokenizer.vocab_size": ["vocab_size"],
    "tokenizer.bos_token_id": ["bos_token_id"],
    "tokenizer.eos_token_id": ["eos_token_id"],
    "tokenizer.pad_token_id": ["pad_token_id"],
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
    # Multimodal nested configs — the language-model fields are read from
    # text_config (see _llm_config); the rest is recognized but unmapped.
    "text_config",
    "vision_config",
    "audio_config",
    "llm_config",
    # Recognized high-frequency fields (from the M4 promotion-candidate report);
    # buffered in passthrough until a downstream consumer warrants canonical.
    "hidden_act",
    "rms_norm_eps",
    "layer_norm_eps",
    "layer_norm_epsilon",
    "use_sliding_window",
    "max_window_layers",
    "position_embedding_type",
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


# Causal-LM-specific canonical fields. For models that are NOT decoder LLMs
# (audio / vision / encoder / seq2seq), any of these we couldn't fill is marked
# not_applicable rather than counted as a coverage gap.
_DECODER_FIELDS = (
    "architecture.num_layers",
    "architecture.hidden_size",
    "architecture.head_dim",
    "attention.type",
    "attention.num_heads",
    "attention.num_kv_heads",
    "context.declared",
)


def _classify_kind(raw: dict) -> str:
    """Classify the model kind from architectures[0] / model_type.

    Used to keep non-decoder models (audio / vision / encoder / seq2seq) from
    polluting the decoder-centric coverage statistics.
    """
    archs = raw.get("architectures")
    cls = str(archs[0]).lower() if isinstance(archs, list) and archs else ""
    mt = str(raw.get("model_type", "")).lower()
    if cls.endswith("forcausallm"):
        return "causal_lm"
    if isinstance(raw.get("text_config"), dict) or isinstance(raw.get("llm_config"), dict):
        return "multimodal"  # VLM whose LM fields we read from text_config
    if "ctc" in cls or "wav2vec2" in cls or "audio" in cls or "hubert" in cls or mt in (
        "whisper",
        "wav2vec2",
    ):
        return "audio"
    if "clip" in cls or "vision" in cls or "image" in cls or cls.endswith("vitmodel"):
        return "vision"
    if cls.endswith("forconditionalgeneration") or mt in ("t5", "bart", "mt5", "mbart"):
        return "seq2seq"
    if any(
        s in cls
        for s in ("formaskedlm", "forsequenceclassification", "fortokenclassification",
                  "forquestionanswering", "formultiplechoice")
    ):
        return "encoder"
    return "unknown"


def _llm_config(raw: dict) -> dict:
    """Effective config for language-model fields, with multimodal fallback.

    VLM / multimodal configs (e.g. *ForConditionalGeneration) nest the LM fields
    under ``text_config`` (vision under ``vision_config``). We read top-level
    first, falling back to the text sub-config so those models aren't left empty.
    """
    sub = raw.get("text_config") or raw.get("llm_config")
    if isinstance(sub, dict):
        return {**sub, **raw}  # top-level wins; the text sub-config fills gaps
    return raw


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
        not_applicable: list[str] = []

        # LM fields are read from `eff`: top-level, falling back to text_config for
        # multimodal models. `raw` stays the source for family / quantization /
        # passthrough.
        eff = _llm_config(raw)

        # Model kind drives scope: non-decoder models (audio/vision/encoder/seq2seq)
        # shouldn't count missing decoder fields as gaps.
        kind = _classify_kind(raw)
        decoder_scope = kind in ("causal_lm", "multimodal")

        # MLA (DeepSeek style) — detected up front so it takes precedence over the
        # head-count heuristic and suppresses the (inoperative) kv-head count.
        is_mla = "kv_lora_rank" in eff or "q_lora_rank" in eff

        # --- canonical layer: alias normalization ---
        for canonical_path, candidates in ALIASES.items():
            if is_mla and canonical_path == "attention.num_kv_heads":
                continue  # GQA kv-head count is not the operative quantity under MLA
            for name in candidates:
                if name in eff:
                    claims.append(FieldClaim(canonical_path, eff[name], "config", "high"))
                    break

        family = _infer_family(raw)
        if family:
            claims.append(FieldClaim("architecture.family", family, "config", "high"))

        # --- feature inference -> architecture tags & attention type ---
        tags: list[str] = []
        if decoder_scope:
            tags.append("decoder-only")
        if kind == "multimodal" or isinstance(raw.get("vision_config"), dict) or isinstance(
            raw.get("audio_config"), dict
        ):
            tags.append("multimodal")
        if kind in ("encoder", "audio", "vision", "seq2seq"):
            tags.append(kind)

        n_heads = eff.get("num_attention_heads")
        n_kv = eff.get("num_key_value_heads")
        if is_mla:
            claims.append(FieldClaim("attention.type", "mla", "inferred", "high"))
            tags.append("mla")
            not_applicable.append("attention.num_kv_heads")
        elif n_heads is not None:
            # A missing num_key_value_heads means MHA (kv heads == query heads):
            # the HF default. Infer it so MHA models aren't left without an
            # attention.type / num_kv_heads.
            kv = n_kv if n_kv is not None else n_heads
            if kv == 1:
                attn_type, tag = "mqa", "mqa"
            elif kv != n_heads:
                attn_type, tag = "gqa", "gqa"
            else:
                attn_type, tag = "mha", "mha"
            claims.append(FieldClaim("attention.type", attn_type, "inferred", "high"))
            tags.append(tag)
            if n_kv is None:
                claims.append(FieldClaim("attention.num_kv_heads", n_heads, "inferred", "high"))

        # MoE detection.
        n_experts = eff.get("num_local_experts") or eff.get("num_experts")
        if n_experts:
            claims.append(FieldClaim("moe.num_experts", n_experts, "config", "high"))
            top_k = eff.get("num_experts_per_tok")
            if top_k is not None:
                claims.append(FieldClaim("moe.top_k", top_k, "config", "high"))
            shared = eff.get("n_shared_experts")
            if shared is not None:
                claims.append(FieldClaim("moe.shared_experts", shared, "config", "high"))
            tags.append("moe")

        # RoPE scaling -> tag + effective context.
        rope = eff.get("rope_scaling")
        if isinstance(rope, dict):
            rope_type = rope.get("type") or rope.get("rope_type")
            if rope_type:
                tags.append(f"rope-{rope_type}")
            factor = rope.get("factor")
            declared = eff.get("max_position_embeddings")
            if isinstance(factor, (int, float)) and isinstance(declared, int):
                claims.append(
                    FieldClaim("context.effective", int(declared * factor), "inferred", "medium")
                )

        if eff.get("sliding_window"):
            tags.append("sliding-window")

        # tie_word_embeddings is a declared hint; tensor presence is authoritative.
        if eff.get("tie_word_embeddings") is True:
            tags.append("tied-embed")
            claims.append(
                FieldClaim("architecture.tied_embeddings", True, "config", "medium")
            )

        # Quantization (AWQ / GPTQ branch of the discriminated union).
        claims.extend(_extract_quantization(raw))

        claims.append(FieldClaim("architecture.tags", tags, "inferred", "medium"))

        # Scope: for non-decoder models, decoder fields we couldn't fill are N/A
        # (not gaps). Self-correcting — only fields actually missing are flagged,
        # so an encoder (BERT) that does have layers/heads keeps them filled.
        if not decoder_scope:
            claimed = {c.field_path for c in claims}
            for path_ in _DECODER_FIELDS:
                if path_ not in claimed and path_ not in not_applicable:
                    not_applicable.append(path_)

        # --- passthrough + raw + unknown ---
        passthrough = {k: raw[k] for k in KNOWN_PASSTHROUGH if k in raw}
        unknown = sorted(set(raw) - _covered_keys())

        return ExtractorResult(
            claims=claims,
            passthrough=passthrough,
            raw=raw,
            unknown_fields=unknown,
            not_applicable=not_applicable,
        )
