"""Cross-source validation — multi-source sanity checks (see docs/pipeline.md).

Runs on the validated ModelSpec and appends findings to provenance.warnings. It
never raises: noisy / partial inputs are expected, and warnings are the product.

The config raw blob (provenance.raw_config_json) is used to reach fields that are
not promoted to canonical (e.g. intermediate_size) for the parameter estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelspec.schema import ModelSpec

# Relative tolerance for the parameter double-path check.
_PARAM_TOLERANCE = 0.01


def _estimate_params_from_config(spec: "ModelSpec", cfg: dict) -> int | None:
    """Path B: rough parameter estimate from config geometry.

    embedding + layers * (attention + ffn) + lm_head (unless tied). This is a
    sanity check only — it ignores biases, norms and MoE expert routing, so a
    small mismatch is normal; a large one usually means a missing component.
    """
    hidden = spec.architecture.hidden_size
    layers = spec.architecture.num_layers
    vocab = spec.tokenizer.vocab_size or cfg.get("vocab_size")
    inter = cfg.get("intermediate_size")
    if not (hidden and layers and vocab and inter):
        return None

    n_heads = spec.attention.num_heads or 0
    n_kv = spec.attention.num_kv_heads or n_heads
    head_dim = hidden // n_heads if n_heads else 0

    # Attention: q (hidden*hidden) + k,v (hidden * kv_dim) + o (hidden*hidden).
    kv_dim = n_kv * head_dim if head_dim else hidden
    attn = hidden * hidden + 2 * hidden * kv_dim + hidden * hidden
    # FFN: gate + up + down for a SwiGLU MLP (3 matrices).
    ffn = 3 * hidden * inter

    embedding = vocab * hidden
    lm_head = 0 if spec.architecture.tied_embeddings else vocab * hidden
    return embedding + layers * (attn + ffn) + lm_head


def cross_validate(spec: "ModelSpec") -> None:
    """Append cross-source warnings to ``spec.provenance.warnings`` in place."""
    warnings = spec.provenance.warnings
    cfg = spec.provenance.raw_config_json or {}

    # --- 1. parameter double-path check ---
    actual = spec.parameters.total
    estimate = _estimate_params_from_config(spec, cfg)
    if actual and estimate:
        diff = abs(actual - estimate) / actual
        if diff > _PARAM_TOLERANCE:
            warnings.append(
                f"parameter count mismatch: tensors={actual:,} vs "
                f"config-estimate={estimate:,} ({diff:.1%} off)"
            )

    # --- 2. context three-layer consistency ---
    ctx = spec.context
    if ctx.trained is not None and ctx.declared is not None and ctx.trained > ctx.declared:
        warnings.append(
            f"context.trained ({ctx.trained}) exceeds context.declared ({ctx.declared})"
        )
    rope = cfg.get("rope_scaling")
    if isinstance(rope, dict) and ctx.declared is not None and ctx.effective is not None:
        factor = rope.get("factor")
        if isinstance(factor, (int, float)):
            expected = int(ctx.declared * factor)
            if expected != ctx.effective:
                warnings.append(
                    f"context.effective ({ctx.effective}) != declared*factor ({expected})"
                )

    # --- 3. MoE flag cross-check (config vs tensor patterns) ---
    has_moe_field = spec.moe is not None and (spec.moe.num_experts or 0) > 0
    has_moe_tag = "moe" in spec.architecture.tags
    if has_moe_field != has_moe_tag:
        warnings.append(
            f"MoE signal disagreement: config/expert_count={has_moe_field}, "
            f"tensor-pattern={has_moe_tag}"
        )

    # --- 4. merge architecture consistency ---
    # All merge components should share the architecture family. We can only
    # compare the recipe's declared base_architecture against the resolved
    # family here (component archs aren't fetched); a mismatch usually means a
    # frankenmerge/passthrough or a wrong detection.
    if spec.merge is not None:
        base_arch = spec.merge.base_architecture
        family = spec.architecture.family
        if base_arch and family and base_arch != family:
            warnings.append(
                f"merge base_architecture ({base_arch}) != resolved family ({family})"
            )
