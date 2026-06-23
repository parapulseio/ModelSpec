# ModelSpec Schema Design

> **Status**: the schema described here is implemented in `modelspec/schema/spec.py`, including the M3 `Quantization` discriminated union (GGUF/AWQ/GPTQ) and `MergeSpec`. `adapter` remains a reserved field (`dict | None`, always `null`). Field-level details of quantization/merge are in [quantization-and-merge.md](quantization-and-merge.md).
>
> For an evaluation of how well this schema serves downstream tasks (model comparison, production selection, further development, inference-engine config, visualization) and a prioritized list of proposed additions, see [schema-review.md](schema-review.md).

## Technology choice: Pydantic v2

We use Pydantic v2 (a Rust core, pydantic-core, 5–50× faster than v1). **There is no v3 and no public roadmap for one**; the current stable line is 2.x.

Pydantic does four jobs here, all unrelated to the source files on the left:

1. **Schema as documentation** — the `ModelSpec` class is itself the system's spec definition; a newcomer understands it by reading the model.
2. **Validation** — an extractor that fills a wrong type errors immediately (`parameters.total = "7B"` is rejected), so bugs don't seep into the downstream consumers.
3. **Serialization / deserialization** — `model_dump_json()` produces JSON, `model_validate()` restores from JSON, no hand-written conversion code.
4. **JSON Schema export** — `model_json_schema()` exports a standard JSON Schema in one line, for API docs / frontend forms / external integration.

Key v2 features (the ones useful here):

- **Discriminated unions**: `Field(discriminator="format")` fits the multi-branch quantization structure.
- **Strict mode**: enable `strict=True` globally or per-field to disable implicit coercion, suited to a spec that demands precision.
- **`@field_validator` / `@model_validator`**: cross-field consistency checks.

## Top-level structure

A model can be **quantized, merged, and carry an adapter at the same time** (e.g. SLERP merge → GGUF Q4_K_M quantization → further LoRA training). So you **cannot** use a `model_type: base|merged|quantized|adapter` enum; each dimension must be an independent optional sub-structure.

```
ModelSpec
  ├── spec_version        : str = "1.0"   (required)
  ├── identity            : Identity       (required)  repo_id / source_format / file_layout / lineage
  ├── architecture        : Architecture   (required)  family / variant / layers / heads / tags
  ├── attention           : Attention      (required)  mha/gqa/mqa/mla / window / kv_heads
  ├── parameters          : Parameters     (required)  total / active / by_component / dtype
  ├── context             : Context        (required)  trained / declared / effective / sliding_window
  ├── tokenizer           : Tokenizer      (required)  type / vocab_size / chat_template
  ├── license             : License        (required)  spdx / capability flags
  ├── moe                 : MoE | None      (optional)  num_experts / top_k / shared_experts
  ├── quantization        : Quant | None    (optional)  discriminated union (see below)
  ├── merge               : MergeSpec | None (optional) detection / method / components
  ├── adapter             : Adapter | None   (optional) peft_type / base_model / rank
  └── provenance          : Provenance      (required)  per_field / conflicts / raw_* / unknown_fields
```

`quantization`, `merge`, and `adapter` are all `Optional`, mutually non-exclusive, and never reference each other. Downstream filtering for "all AWQ-quantized merged models" is just a predicate: `spec.quantization.format == "awq" and spec.merge is not None`.

## Sub-model fields (skeleton)

> This is the target for all later extractor code. Pin down the field list first; you don't have to fill it all at once.

### Identity
- `repo_id: str`
- `source_format: Literal["hf", "gguf", "adapter", "raw"]`
- `file_layout: str | None` — single / sharded
- `lineage: Lineage | None` — the **unified source relationship** (the base_model chain). Quantization, merge, and adapter all have a base_model; keep it here, not inside each sub-structure.

### Architecture
- `family: str | None` — `"llama"` / `"qwen"` / `"deepseek"` (tiered fallback inference, see below)
- `variant: str | None`
- `num_layers: int`
- `hidden_size: int`
- `tied_embeddings: bool` — inferred from whether the tensor header has `lm_head.weight`
- `tags: list[str]` — a **tag set rather than a single string**, e.g. `["decoder-only", "moe", "gqa", "rope-yarn", "tied-embed"]`, better suited to downstream search/filtering

### Attention
- `type: Literal["mha", "gqa", "mqa", "mla"]`
- `num_heads: int`
- `num_kv_heads: int | None`
- `sliding_window: int | None`

> Semantic-drift pitfall: `num_key_value_heads` is the GQA grouping in the Llama family, but in DeepSeek-V2/V3 (MLA) the field is semantically hollow (KV is actually low-rank compressed). Do not hard-normalize it; for the DeepSeek family this field should be `N/A` rather than `missing`.

### Parameters
- `total: int` — summed tensor element counts (authoritative)
- `active: int | None` — MoE only; `total - (num_experts - top_k) × per_expert_params`, where **shared experts are always active and must not be subtracted from active**
- `by_component: dict[str, int] | None`
- `dtype_native: str` — `BF16` / `INT4` …

### Context (three layers + sliding window)
- `trained: int | None` — the actual training window, usually pulled from the model card
- `declared: int` — the config's `max_position_embeddings`
- `effective: int | None` — `declared × rope_scaling.factor` (YARN / NTK / linear)
- `sliding_window: int | None` — a separate field (the Gemma family, early Mistral)
- `rope_scaling: dict | None`

### Tokenizer
- `type: str` — BPE / Unigram …
- `vocab_size: int`
- `chat_template_present: bool`

### License (three-tier identification, see extractors.md)
- `spdx_id: str | None`
- `commercial_use: bool | None`
- `redistribution: bool | None`
- `attribution_required: bool | None`
- `confidence_tier: Literal["fingerprint", "keyword", "llm"]`

### Provenance (the core of losslessness)
- `per_field: dict[str, FieldProvenance]` — each field's `{source, confidence}`
- `conflicts: list[Conflict]` — multi-source conflicts archived for human review
- `warnings: list[str]` — cross-validation warnings (e.g. parameter double-path diff >1%)
- `raw_config_json: dict | None` — the full original config, hash-archived, never lost
- `raw_gguf_kv: dict | None`
- `unknown_fields: list[str]` — fields present in raw but covered by neither canonical nor passthrough (the auto feedback loop)

## FieldClaim — the connection point between extractor and schema

An extractor does not return a nested dict directly; it returns a flat list of four-tuples, so the merger can resolve conflicts easily:

```python
class FieldClaim(NamedTuple):
    field_path: str          # "architecture.num_layers"
    value: Any
    source: str              # "config" | "tensors" | "inferred" | "fingerprint" | ...
    confidence: str          # "high" | "medium" | "low"
```

## Custom validators (cross-field consistency)

Run during `model_validate`, for example:

- `num_kv_heads` must divide `num_heads`.
- If `moe.num_experts > 0` then `parameters.active` is required.
- `attention.type` must be one of `mha/gqa/mqa/mla`.
- All merge components should share an architecture; a mismatch goes to `provenance.warnings` (frankenmerge excepted).

## Architecture identification: tiered fallback

Try in descending order of trust, and **output a tag set rather than a single string**:

1. `config.architectures[0]` → look up the family in a registry (but under `trust_remote_code=True` it is a custom class name — **do not execute `modeling_*.py`**)
2. config feature detection (`num_local_experts` → MoE; `q_lora_rank`/`kv_lora_rank` → MLA; `num_key_value_heads != num_attention_heads` → GQA; `rope_scaling.type` → extension method; `mamba_*` → hybrid SSM)
3. tensor-name pattern matching as the last resort (`block_sparse_moe.experts.{j}.w1` → Mixtral MoE; `mlp.experts.{j}.gate_proj` → Qwen MoE; `self_attn.kv_a_proj_with_mqa` → DeepSeek MLA; `lora_A`/`lora_B` → LoRA)
