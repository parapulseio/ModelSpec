# Schema Review — Fitness for Downstream Tasks

This document evaluates the current `ModelSpec` schema (`modelspec/schema/spec.py`)
against five intended downstream uses, identifies gaps, and proposes concrete
additions. It is a **design review**, not an implementation; nothing here is
wired up yet.

> **Implemented since this review**: finding **D** (per-field `Field(description=...)`,
> so JSON Schema is self-documenting) and finding **A** (`provenance.not_applicable`
> for the N/A-vs-missing distinction; the config extractor flags
> `attention.num_kv_heads` under MLA, which also fixed an MLA-vs-GQA precedence
> bug). The remaining items below are still proposals.

The five target tasks:

1. **Model comparison** — diff two specs.
2. **Production selection** — help choose a model for a deployment.
3. **Further development** — inform fine-tuning / merging / continued training.
4. **Inference-engine config** — provide a basis for vLLM-style launch settings.
5. **Visualization** — render a spec in a UI.

## Scorecard

| # | Task | Rating | Summary |
| --- | --- | --- | --- |
| 1 | Model comparison | **B+ (good)** | Strong scalar coverage + provenance. Missing: revision/sha, "N/A vs missing" distinction, dtype/tag normalization. |
| 2 | Production selection | **C+ (partial)** | License + size + quantization present. Missing: memory footprint, modality, instruct-vs-base, capability flags. |
| 3 | Further development | **C (partial)** | Lineage is good. FFN size / head_dim / rope_theta / norm / activation / special tokens / adapter details exist only in raw, not canonical. |
| 4 | Inference-engine config | **C+ (partial)** | Most inputs present or derivable. Missing: head_dim, trust_remote_code flag, engine-normalized dtype/quant vocab, raw architecture class name. |
| 5 | Visualization | **B (good)** | Nested grouping + per-field provenance + JSON Schema fit a UI naturally. Missing: field descriptions, warning severity. |

## Cross-cutting findings (affect multiple tasks)

### A. "N/A" vs "missing" are indistinguishable (tasks 1, 2, 5) ✅ done

Every absent value is `None`, whether it is *genuinely not applicable* (DeepSeek
MLA has no `num_kv_heads`) or *we failed to extract it*. A diff tool, a UI, and
the coverage report all conflate the two.

**Recommendation:** add an explicit applicability concept. Options:
- a `provenance.not_applicable: list[str]` (dotted paths the model legitimately
  lacks), set by extractors that *know* a field doesn't apply (e.g. the MLA
  branch marks `attention.num_kv_heads` N/A); or
- a sentinel value distinct from `None`.

The provenance-list option is the least invasive and reuses the existing
provenance home. **Implemented:** `provenance.not_applicable` (the MLA branch
flags `attention.num_kv_heads`), and the coverage report now excludes N/A models
from each field's fill-rate denominator so legitimately-absent fields are no
longer reported as low-coverage gaps (see [analytics.md](analytics.md)).

### B. Extracted facts vs derived values — keep them separate (tasks 2, 4)

The most valuable outputs for production selection and inference config —
**VRAM footprint, KV-cache size, parameter "size class", tensor-parallel
divisors, a ready-made vLLM arg dict** — are all *computed* from extracted
fields. They have no single `source`, so storing them inside `ModelSpec` would
pollute `provenance.per_field`.

**Recommendation:** do **not** add them to `ModelSpec`. Add a thin derivation
layer (e.g. `modelspec/derive.py` producing a `DerivedView`, or helper functions
/ computed properties) that takes a validated `ModelSpec` and returns derived
values on demand. This is exactly the shape of the proposed **M5 "help
function"** milestone.

### C. Many development/inference-critical fields live only in raw (tasks 3, 4) — partly done

> **Promoted (M4-driven):** `architecture.head_dim` and `tokenizer.{bos,eos,pad}_token_id`
> are now canonical (they ranked high in the coverage `promotion_candidates`).
> Also: VLM nested `text_config` is now read, so multimodal models fill the LM
> fields. Remaining (`intermediate_size`, `rope_theta`, norm/activation) stay in
> passthrough for now.


`intermediate_size`, `rope_theta`, `head_dim`, `norm_type`, `activation_fn`,
attention bias, and tokenizer special tokens are present in
`provenance.raw_config_json` (passthrough) but are not canonical. These are the
fields tasks 3 and 4 need most.

**Recommendation:** these will surface naturally at the top of the **M4
`unknown_fields` frequency** list; promote them through the normal
raw → passthrough → canonical workflow (see [analytics.md](analytics.md)). The
review's wishlist and the M4 feedback loop converge here.

### D. Fields carry no machine-readable descriptions (task 5) ✅ done

Fields are documented with code comments, so `model_json_schema()` export has no
`description`. Adding `Field(..., description="...")` gives UI tooltips,
self-documenting JSON Schema, and auto-generated forms in one move. Low effort,
high payoff. (No new data, no extractor changes.)

## Per-task detail and proposed additions

### 1. Model comparison

**Works:** `architecture.family`, `parameters.total/active`,
`context.declared/effective`, `attention.type`, `quantization.format`,
`license.*` flags, `architecture.tags` (set-comparable), and
`provenance.per_field` (so a diff can show *why* a value differs).

**Gaps / proposals:**

| Proposed | Where | Why |
| --- | --- | --- |
| `identity.revision` / `identity.commit_sha` | new (HF API) | compare two revisions of the same repo; pin a version |
| applicability (finding A) | `provenance.not_applicable` | distinguish N/A from a missed extraction in a diff |
| normalize `parameters.dtype_native` | extractor | `"BF16"` vs `"bfloat16"` must not read as a difference |
| controlled `tags` vocabulary | docs + extractor | two specs must tag the same property identically |

### 2. Production selection

**Works:** `license.commercial_use/redistribution/attribution_required`,
`parameters.total/active`, `context.*`, `quantization.*`.

**Gaps / proposals:**

| Proposed | Where | Why |
| --- | --- | --- |
| `architecture.is_instruct` (or `variant`-derived) | extractor (chat_template + name) | "instruct vs base" is a primary selection filter |
| `modality` | new field | text-only vs vision/audio multimodal — selection-critical and currently assumed |
| capability flags (tool-use, reasoning) | new (model card) | common filter; `chat_template_present` is only a weak proxy |
| `footprint` (weights bytes, est. min VRAM) | **derived layer (B)** | "does it fit my GPU" is the #1 question; derivable from params + dtype + quant |
| `identity.last_modified` / release date | new (HF API) | maturity / recency |

### 3. Further development (fine-tune / merge / continue training)

**Works:** `identity.lineage` (base_models + relation), `architecture.num_layers`,
`architecture.hidden_size`, `merge.raw_recipe` (reproduce a merge).

**Gaps / proposals:**

| Proposed | Where | Why |
| --- | --- | --- |
| `architecture.intermediate_size` | promote from raw | FFN width — needed to size LoRA / reason about FLOPs |
| `architecture.head_dim` | promote/derive | not always `hidden/num_heads` (MLA); needed for targeting |
| `architecture.rope_theta` | promote from raw | continued-training / context-extension decisions |
| `architecture.norm_type` / `activation_fn` | promote from raw | architectural compatibility for merges |
| `tokenizer.special_tokens` (bos/eos/pad) | extractor | required to set up fine-tuning |
| **implement `adapter`** | new sub-model | `peft_type` / `rank` / `target_modules` / `base_model` for LoRA work (currently a reserved `dict`) |

### 4. Inference-engine config (vLLM and friends)

Mapping of current fields to common vLLM launch args:

| vLLM arg | Source field | Status |
| --- | --- | --- |
| `max_model_len` | `context.declared` (or `effective`) | ✅ present |
| `dtype` | `parameters.dtype_native` | ⚠️ present but not normalized to `{auto, half, bfloat16, float16}` |
| `quantization` | `quantization.format` | ✅ present (awq/gptq/gguf); ⚠️ BnB/FP8 not yet modeled |
| `rope_scaling` | `context.rope_scaling` | ✅ present (raw passed through) |
| tensor-parallel divisibility | `attention.num_heads` / `num_kv_heads` | ✅ checkable |
| sliding window | `attention.sliding_window` | ✅ present |
| MoE routing | `moe.num_experts` / `top_k` | ✅ present |
| KV-cache sizing | `head_dim`, `num_kv_heads`, `num_layers`, dtype | ⚠️ `head_dim` missing |
| `trust_remote_code` | — | ❌ no flag (knowable: family inference fell back to tensor patterns) |
| model class match | raw `architectures[0]` | ⚠️ only in raw; `family` is normalized away from the class name |

**Gaps / proposals:**

| Proposed | Where | Why |
| --- | --- | --- |
| `architecture.head_dim` | promote/derive | KV-cache math, MLA correctness |
| `architecture.transformers_class` (raw `architectures[0]`) | promote from raw | engines match on the class name; also a `trust_remote_code` signal |
| `architecture.trust_remote_code_required` | extractor (inference) | true when the class is custom / family fell back to tensor patterns |
| normalized `dtype` / `quantization` vocab | **derived layer (B)** | emit engine-ready values without mutating extracted facts |
| `vllm_args` convenience dict | **derived layer (B)** | one call → a launch-arg dict; ties to M5 |

### 5. Visualization

**Works:** the 8 sub-models map cleanly to UI cards/sections;
`provenance.per_field` gives source + confidence for per-field badges;
`conflicts` / `warnings` drive alerts; `model_json_schema()` can auto-generate
forms; `tags` render as chips.

**Gaps / proposals:**

| Proposed | Where | Why |
| --- | --- | --- |
| `Field(description=...)` on every field (finding D) | `spec.py` | tooltips + self-documenting JSON Schema |
| structured warnings with `severity` | `Provenance` | color/sort alerts in a UI (currently plain strings) |
| applicability (finding A) | provenance | render N/A as "—" instead of "missing" |

## Out of scope (deliberately not added)

To keep the spec a disciplined set of *extracted facts*, the following do **not**
belong in `ModelSpec`:

- Benchmark / eval scores (require running the model).
- Download counts / popularity / trending (volatile market-intelligence, belongs
  to a separate layer).
- Live pricing / availability.

## Suggested sequencing

1. ~~**Low-effort, high-payoff now:** finding D (field descriptions) and finding A
   (applicability).~~ ✅ done — both schema-local, no new extraction.
2. **Data-driven via M4:** run `coverage`, then promote the finding-C fields
   (`intermediate_size`, `head_dim`, `rope_theta`, special tokens, raw
   `architectures` class) as they rank high in `unknown_fields`.
3. **Derived layer (M5):** footprint / size class / normalized dtype-quant /
   `vllm_args` — computed, never stored in provenance.
4. **New sub-models:** implement `adapter`; add `modality` / `is_instruct` /
   capability flags as model-card extraction matures.
