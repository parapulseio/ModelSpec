"""The unified ``ModelSpec`` Pydantic v2 schema — the foundation of the system.

Design notes (see docs/schema.md):
    - Every scalar field is Optional with a ``None`` default. A model may only
      ship a subset of sources (e.g. raw weights only), so most fields are
      genuinely unknown and must validate cleanly when missing.
    - The required top-level sub-models default to an empty instance via
      ``default_factory`` so a minimal reshape dict always validates.
    - ``quantization`` / ``merge`` / ``adapter`` are orthogonal optional
      structures (see docs/quantization-and-merge.md); never collapse them into
      a ``model_type`` enum. ``quantization`` + ``merge`` are implemented;
      ``adapter`` is still reserved.
    - Every field carries a ``description`` so ``model_json_schema()`` is
      self-documenting (UI tooltips, generated forms). See docs/schema-review.md.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Source labels attached to every claim / provenance entry.
# "merged" is emitted by the merger for accumulated (unioned) list fields.
SourceLabel = Literal[
    "config", "tensors", "gguf", "inferred", "heuristic", "fingerprint", "keyword", "llm", "merged"
]
Confidence = Literal["high", "medium", "low"]


class _Model(BaseModel):
    """Shared base config for all schema models."""

    model_config = ConfigDict(extra="ignore", validate_assignment=False)


class Lineage(_Model):
    """Unified source relationship (the base_model chain).

    Quantized, merged and adapter models all carry a base_model link, so it
    lives here rather than inside each orthogonal sub-structure.
    """

    base_models: list[str] = Field(
        default_factory=list, description="Upstream model ids this model derives from."
    )
    relation: Optional[str] = Field(
        default=None,
        description='How this model relates to its base: "quantized" / "merge" / "adapter" / "finetune".',
    )


class Identity(_Model):
    repo_id: Optional[str] = Field(default=None, description="HF repo id or local path of the model.")
    source_format: Literal["hf", "gguf", "adapter", "raw", "unknown"] = Field(
        default="unknown", description="Detected repo layout: hf / gguf / adapter / raw / unknown."
    )
    file_layout: Optional[str] = Field(
        default=None, description='Weight file layout: "single" or "sharded".'
    )
    lineage: Optional[Lineage] = Field(
        default=None, description="Unified base_model lineage (see Lineage)."
    )


class Architecture(_Model):
    family: Optional[str] = Field(
        default=None, description='Normalized architecture family, e.g. "llama" / "qwen2" / "deepseek".'
    )
    variant: Optional[str] = Field(default=None, description="Sub-variant within the family, if known.")
    num_layers: Optional[int] = Field(default=None, description="Number of transformer blocks.")
    hidden_size: Optional[int] = Field(default=None, description="Model hidden / embedding dimension.")
    tied_embeddings: Optional[bool] = Field(
        default=None,
        description="True if input embeddings are tied to the output head (no separate lm_head.weight).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description='Multi-tag descriptor set, e.g. ["decoder-only", "moe", "gqa", "rope-yarn"].',
    )


class Attention(_Model):
    type: Optional[Literal["mha", "gqa", "mqa", "mla"]] = Field(
        default=None, description="Attention scheme: mha / gqa / mqa / mla."
    )
    num_heads: Optional[int] = Field(default=None, description="Number of attention (query) heads.")
    num_kv_heads: Optional[int] = Field(
        default=None,
        description="Number of key/value heads (GQA grouping). Not applicable under MLA.",
    )
    sliding_window: Optional[int] = Field(
        default=None, description="Sliding-window size in tokens; None means global attention."
    )


class Parameters(_Model):
    total: Optional[int] = Field(
        default=None, description="Total parameter count, summed from tensor shapes (authoritative)."
    )
    active: Optional[int] = Field(
        default=None,
        description="Active parameters per token (MoE only). Shared experts are always active.",
    )
    by_component: Optional[dict[str, int]] = Field(
        default=None, description="Parameter counts broken down by component, if computed."
    )
    dtype_native: Optional[str] = Field(
        default=None, description='Native storage dtype, e.g. "BF16" / "F16" / "Q4_K".'
    )


class Context(_Model):
    trained: Optional[int] = Field(
        default=None, description="Actual training context window (often only in the model card)."
    )
    declared: Optional[int] = Field(
        default=None, description="Declared context: config max_position_embeddings."
    )
    effective: Optional[int] = Field(
        default=None, description="Effective context after RoPE scaling (declared * factor)."
    )
    sliding_window: Optional[int] = Field(
        default=None, description="Context-level sliding window, if distinct from attention's."
    )
    rope_scaling: Optional[dict[str, Any]] = Field(
        default=None, description="Raw RoPE scaling config (type/factor), passed through."
    )


class Tokenizer(_Model):
    type: Optional[str] = Field(default=None, description='Tokenizer model type, e.g. "BPE" / "Unigram".')
    vocab_size: Optional[int] = Field(default=None, description="Vocabulary size (incl. added tokens).")
    chat_template_present: Optional[bool] = Field(
        default=None, description="True if a chat template is shipped (a weak instruct-model proxy)."
    )


class License(_Model):
    spdx_id: Optional[str] = Field(
        default=None, description='Identified license id, e.g. "apache-2.0" / "llama3.1".'
    )
    commercial_use: Optional[bool] = Field(default=None, description="Whether commercial use is permitted.")
    redistribution: Optional[bool] = Field(default=None, description="Whether redistribution is permitted.")
    attribution_required: Optional[bool] = Field(
        default=None, description="Whether attribution is required."
    )
    confidence_tier: Optional[Literal["fingerprint", "keyword", "llm"]] = Field(
        default=None, description="How the license was identified: fingerprint / keyword / llm."
    )


class MoE(_Model):
    num_experts: Optional[int] = Field(default=None, description="Total number of experts.")
    top_k: Optional[int] = Field(default=None, description="Experts routed per token (top-k).")
    shared_experts: Optional[int] = Field(
        default=None, description="Number of always-on shared experts."
    )


# --- Quantization: an orthogonal optional structure, modeled as a discriminated
# union keyed by ``format``. M3 ships GGUF / AWQ / GPTQ; BnB / FP8 / MLX follow.
# An unrecognized quant method emits no quantization claim (stays None), so the
# union never has to validate an unknown discriminator.


class GGUFQuant(_Model):
    format: Literal["gguf"] = Field(description="Discriminator: GGUF container quantization.")
    file_type: Optional[str] = Field(default=None, description='GGUF file type, e.g. "Q4_K_M" / "IQ3_XS".')
    bits_per_weight_avg: Optional[float] = Field(
        default=None, description="Measured average bits-per-weight (not the nominal value)."
    )
    tensor_types: dict[str, int] = Field(
        default_factory=dict, description='Element count per ggml type, e.g. {"Q4_K": 200, "F32": 5}.'
    )
    has_imatrix: Optional[bool] = Field(
        default=None, description="Importance-matrix quantization heuristic; None when undetermined."
    )


class AWQQuant(_Model):
    format: Literal["awq"] = Field(description="Discriminator: AWQ quantization.")
    bits: Optional[int] = Field(default=None, description="Weight bit width.")
    group_size: Optional[int] = Field(default=None, description="Quantization group size.")
    zero_point: Optional[bool] = Field(default=None, description="Whether zero-point quantization is used.")


class GPTQQuant(_Model):
    format: Literal["gptq"] = Field(description="Discriminator: GPTQ quantization.")
    bits: Optional[int] = Field(default=None, description="Weight bit width.")
    group_size: Optional[int] = Field(default=None, description="Quantization group size.")
    desc_act: Optional[bool] = Field(default=None, description="Whether activation-order (desc_act) is used.")


Quantization = Annotated[
    Union[GGUFQuant, AWQQuant, GPTQQuant], Field(discriminator="format")
]


# --- Merge: another orthogonal optional structure. Components carry whatever the
# data source exposes; HF-tag detections often only know the model ids.


class MergeComponent(_Model):
    model_id: str = Field(description="A model participating in the merge.")
    weight: Optional[float] = Field(default=None, description="Merge weight, if the method exposes it.")
    density: Optional[float] = Field(default=None, description="DARE / TIES density parameter, if any.")
    role: Optional[Literal["base", "ingredient", "donor"]] = Field(
        default=None, description="Component role in the merge."
    )


class MergeSpec(_Model):
    detection_signal: Literal[
        "hf_tag", "config_file", "card_relation", "base_model_array", "readme_yaml"
    ] = Field(description="Highest-priority signal that identified this as a merge.")
    confidence: Confidence = Field(description="Confidence of the merge detection.")
    method: Optional[str] = Field(
        default=None, description='Normalized merge method, e.g. "slerp" / "dare_ties".'
    )
    components: list[MergeComponent] = Field(
        default_factory=list, description="Participating models with weights/roles where known."
    )
    base_architecture: Optional[str] = Field(
        default=None, description="Shared architecture of the components (mismatch -> warning)."
    )
    tokenizer_source: Optional[str] = Field(
        default=None, description="Which component provided the tokenizer, if known."
    )
    mergekit_version: Optional[str] = Field(default=None, description="mergekit version, if recorded.")
    has_config_yml: bool = Field(
        default=False, description="True if a mergekit_config.yml recipe was found."
    )
    raw_recipe: Optional[dict[str, Any]] = Field(
        default=None, description="The full parsed mergekit recipe, for reproduction."
    )


class FieldProvenance(_Model):
    source: SourceLabel = Field(description="Which source produced the winning value for a field.")
    confidence: Confidence = Field(description="Confidence of that value.")


class Conflict(_Model):
    """A losing claim kept for human review when sources disagree."""

    field_path: str = Field(description="Dotted path of the conflicting field.")
    value: Any = Field(description="The losing value.")
    source: SourceLabel = Field(description="Source of the losing value.")
    confidence: Confidence = Field(description="Confidence of the losing value.")
    winner_source: SourceLabel = Field(description="Source of the value that won.")
    winner_value: Any = Field(description="The value that won.")


class Provenance(_Model):
    per_field: dict[str, FieldProvenance] = Field(
        default_factory=dict, description="Source + confidence for each filled field path."
    )
    conflicts: list[Conflict] = Field(
        default_factory=list, description="Losing claims when sources disagreed."
    )
    warnings: list[str] = Field(
        default_factory=list, description="Cross-validation warnings (non-fatal)."
    )
    not_applicable: list[str] = Field(
        default_factory=list,
        description=(
            "Dotted paths that are legitimately N/A for this model (e.g. "
            "attention.num_kv_heads under MLA) — distinct from merely missing."
        ),
    )
    # Information-loss insurance: full original blobs, hashed-archive friendly.
    raw_config_json: Optional[dict[str, Any]] = Field(
        default=None, description="The full original config.json, archived losslessly."
    )
    raw_gguf_kv: Optional[dict[str, Any]] = Field(
        default=None, description="The full GGUF KV dump (large arrays folded to length markers)."
    )
    unknown_fields: list[str] = Field(
        default_factory=list,
        description="Raw keys covered by neither canonical nor passthrough (drives field promotion).",
    )


class ModelSpec(_Model):
    spec_version: str = Field(default="1.0", description="Schema version of this document.")
    identity: Identity = Field(default_factory=Identity, description="Who this model is.")
    architecture: Architecture = Field(default_factory=Architecture, description="Architectural shape.")
    attention: Attention = Field(default_factory=Attention, description="Attention configuration.")
    parameters: Parameters = Field(default_factory=Parameters, description="Parameter counts and dtype.")
    context: Context = Field(default_factory=Context, description="Context-length information.")
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, description="Tokenizer information.")
    license: License = Field(default_factory=License, description="License and usage flags.")
    moe: Optional[MoE] = Field(default=None, description="Mixture-of-experts config (None if dense).")
    # Orthogonal optional structures — independent, non-exclusive.
    quantization: Optional[Quantization] = Field(
        default=None, description="Quantization (discriminated union); None if not quantized."
    )
    merge: Optional[MergeSpec] = Field(default=None, description="Merge info; None if not a merge.")
    adapter: Optional[dict[str, Any]] = Field(
        default=None, description="Adapter (PEFT/LoRA) info; reserved, currently always null."
    )
    provenance: Provenance = Field(
        default_factory=Provenance, description="Field-level sources, conflicts, warnings, raw blobs."
    )

    @model_validator(mode="after")
    def _check_cross_field(self) -> "ModelSpec":
        """Light cross-field consistency checks.

        Violations are recorded as warnings (non-fatal) rather than raised,
        because partial / noisy inputs are expected. Hard structural errors
        (wrong types, bad enums) are already caught by field validation.
        """
        a = self.attention
        if (
            a.num_heads is not None
            and a.num_kv_heads is not None
            and a.num_kv_heads > 0
            and a.num_heads % a.num_kv_heads != 0
        ):
            self.provenance.warnings.append(
                f"num_heads ({a.num_heads}) not divisible by num_kv_heads ({a.num_kv_heads})"
            )
        if self.moe is not None and self.parameters.active is None:
            self.provenance.warnings.append(
                "MoE model is missing parameters.active"
            )
        return self
