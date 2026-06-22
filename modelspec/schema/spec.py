"""The unified ``ModelSpec`` Pydantic v2 schema — the foundation of the system.

Design notes (see docs/schema.md):
    - Every scalar field is Optional with a ``None`` default. A model may only
      ship a subset of sources (e.g. raw weights only), so most fields are
      genuinely unknown and must validate cleanly when missing.
    - The eight required top-level sub-models default to an empty instance via
      ``default_factory`` so a minimal reshape dict always validates.
    - ``quantization`` / ``merge`` / ``adapter`` are orthogonal optional
      structures. They are reserved here and populated in M3 (see
      docs/quantization-and-merge.md); never collapse them into a ``model_type``
      enum.
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

    base_models: list[str] = Field(default_factory=list)
    relation: Optional[str] = None  # "quantized" / "merge" / "adapter" / "finetune"


class Identity(_Model):
    repo_id: Optional[str] = None
    source_format: Literal["hf", "gguf", "adapter", "raw", "unknown"] = "unknown"
    file_layout: Optional[str] = None  # "single" / "sharded"
    lineage: Optional[Lineage] = None


class Architecture(_Model):
    family: Optional[str] = None  # "llama" / "qwen2" / "deepseek"
    variant: Optional[str] = None
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    tied_embeddings: Optional[bool] = None
    # Tag set rather than a single string — better for downstream filtering.
    tags: list[str] = Field(default_factory=list)


class Attention(_Model):
    type: Optional[Literal["mha", "gqa", "mqa", "mla"]] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    sliding_window: Optional[int] = None


class Parameters(_Model):
    total: Optional[int] = None  # authoritative: summed from tensor shapes
    active: Optional[int] = None  # MoE only; shared experts are always active
    by_component: Optional[dict[str, int]] = None
    dtype_native: Optional[str] = None


class Context(_Model):
    trained: Optional[int] = None  # actual training window (often from model card)
    declared: Optional[int] = None  # config max_position_embeddings
    effective: Optional[int] = None  # declared * rope_scaling.factor
    sliding_window: Optional[int] = None
    rope_scaling: Optional[dict[str, Any]] = None


class Tokenizer(_Model):
    type: Optional[str] = None  # "BPE" / "Unigram"
    vocab_size: Optional[int] = None
    chat_template_present: Optional[bool] = None


class License(_Model):
    spdx_id: Optional[str] = None
    commercial_use: Optional[bool] = None
    redistribution: Optional[bool] = None
    attribution_required: Optional[bool] = None
    confidence_tier: Optional[Literal["fingerprint", "keyword", "llm"]] = None


class MoE(_Model):
    num_experts: Optional[int] = None
    top_k: Optional[int] = None
    shared_experts: Optional[int] = None


# --- Quantization: an orthogonal optional structure, modeled as a discriminated
# union keyed by ``format``. M3 ships GGUF / AWQ / GPTQ; BnB / FP8 / MLX follow.
# An unrecognized quant method emits no quantization claim (stays None), so the
# union never has to validate an unknown discriminator.


class GGUFQuant(_Model):
    format: Literal["gguf"]
    file_type: Optional[str] = None  # "Q4_K_M" / "IQ3_XS"
    bits_per_weight_avg: Optional[float] = None  # measured, not the nominal value
    tensor_types: dict[str, int] = Field(default_factory=dict)  # {"Q4_K": 200, ...}
    has_imatrix: Optional[bool] = None  # heuristic; None when undetermined


class AWQQuant(_Model):
    format: Literal["awq"]
    bits: Optional[int] = None
    group_size: Optional[int] = None
    zero_point: Optional[bool] = None


class GPTQQuant(_Model):
    format: Literal["gptq"]
    bits: Optional[int] = None
    group_size: Optional[int] = None
    desc_act: Optional[bool] = None


Quantization = Annotated[
    Union[GGUFQuant, AWQQuant, GPTQQuant], Field(discriminator="format")
]


# --- Merge: another orthogonal optional structure. Components carry whatever the
# data source exposes; HF-tag detections often only know the model ids.


class MergeComponent(_Model):
    model_id: str
    weight: Optional[float] = None
    density: Optional[float] = None  # DARE / TIES parameter
    role: Optional[Literal["base", "ingredient", "donor"]] = None


class MergeSpec(_Model):
    detection_signal: Literal[
        "hf_tag", "config_file", "card_relation", "base_model_array", "readme_yaml"
    ]
    confidence: Confidence
    method: Optional[str] = None  # normalized: "slerp" / "dare_ties" / ...
    components: list[MergeComponent] = Field(default_factory=list)
    base_architecture: Optional[str] = None
    tokenizer_source: Optional[str] = None
    mergekit_version: Optional[str] = None
    has_config_yml: bool = False
    raw_recipe: Optional[dict[str, Any]] = None


class FieldProvenance(_Model):
    source: SourceLabel
    confidence: Confidence


class Conflict(_Model):
    """A losing claim kept for human review when sources disagree."""

    field_path: str
    value: Any
    source: SourceLabel
    confidence: Confidence
    winner_source: SourceLabel
    winner_value: Any


class Provenance(_Model):
    per_field: dict[str, FieldProvenance] = Field(default_factory=dict)
    conflicts: list[Conflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Information-loss insurance: full original blobs, hashed-archive friendly.
    raw_config_json: Optional[dict[str, Any]] = None
    raw_gguf_kv: Optional[dict[str, Any]] = None
    # raw keys present but covered by neither canonical nor passthrough —
    # this drives the auto-feedback loop for field promotion.
    unknown_fields: list[str] = Field(default_factory=list)


class ModelSpec(_Model):
    spec_version: str = "1.0"
    identity: Identity = Field(default_factory=Identity)
    architecture: Architecture = Field(default_factory=Architecture)
    attention: Attention = Field(default_factory=Attention)
    parameters: Parameters = Field(default_factory=Parameters)
    context: Context = Field(default_factory=Context)
    tokenizer: Tokenizer = Field(default_factory=Tokenizer)
    license: License = Field(default_factory=License)
    moe: Optional[MoE] = None
    # Orthogonal optional structures — independent, non-exclusive (see
    # docs/quantization-and-merge.md). quantization + merge land in M3; adapter
    # stays reserved (typed loosely) until its own milestone.
    quantization: Optional[Quantization] = None
    merge: Optional[MergeSpec] = None
    adapter: Optional[dict[str, Any]] = None
    provenance: Provenance = Field(default_factory=Provenance)

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
