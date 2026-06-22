# Extractor Design

> **Status**: config / safetensors (M1), gguf / license / tokenizer (M2), and merge (M3) are all implemented in `modelspec/extractors/`. Quantization claims are emitted by config_json (AWQ/GPTQ) + gguf (GGUF); see the modeling in [quantization-and-merge.md](quantization-and-merge.md). The third "LLM" tier of license identification below is currently a hook (no model wired up).

## Protocol

Every extractor implements the same protocol, and the `orchestrator` only talks to the protocol:

```python
class Extractor(Protocol):
    name: str
    def can_handle(self, repo_files: list[str]) -> bool: ...
    def extract(self, source: ExtractionSource) -> ExtractorResult: ...
```

`extractors/__init__.py` maintains the `ALL_EXTRACTORS` registry; disabling one is just removing it from the registry.

## Core strategy: do not chase full coverage

Trying to cover "every config.json field" is a trap — config.json has no real schema, `trust_remote_code` can stuff in arbitrary fields, field semantics drift, and old models use old field names.

**The right approach: structured partial coverage + lossless retention + an auto feedback loop.** Each extractor produces three layers:

```
original file (raw bytes)
   ↓
┌─────────────────────────────────────────────┐
│ 1. canonical layer ← fields the schema cares │  strictly controlled small set, alias-normalized, with confidence
│    architecture.num_layers = 32              │
│    attention.num_kv_heads = 8                │
├─────────────────────────────────────────────┤
│ 2. passthrough layer ← recognized but unmapped│  store the original value, a buffer
│    rope_scaling = {...}                      │
│    quantization_config = {...}               │
├─────────────────────────────────────────────┤
│ 3. raw layer       ← the full original dict   │  hash-archived, lossless insurance
│    {... the entire config.json verbatim ...}  │
└─────────────────────────────────────────────┘
```

All three layers go into `ModelSpec` (under `provenance`). `unknown_fields = raw_keys - canonical_keys - passthrough_keys` is the **auto-alerting mechanism**: after a batch run, tally high-frequency unknown fields as promotion signals.

### Field-promotion criteria

- **raw → passthrough**: a field appearing in >10% of models that you can describe in one sentence.
- **passthrough → canonical**: a clear downstream consumer (filter / recommend / display). Once published, a canonical field carries a compatibility burden, so promote carefully — don't promote just because it "looks important".

## config_json extractor

### Alias normalization

The same canonical field maps to several historical HF names (GPT-2 style vs Llama style):

```python
ALIASES = {
    "architecture.num_layers":  ["num_hidden_layers", "n_layer", "n_layers"],
    "architecture.hidden_size": ["hidden_size", "n_embd", "d_model"],
    "context.declared":         ["max_position_embeddings", "n_positions", "seq_length"],
    "attention.num_heads":      ["num_attention_heads", "n_head", "num_heads"],
    "attention.num_kv_heads":   ["num_key_value_heads", "num_kv_heads"],
}
```

### Feature inference

- `num_local_experts` / `num_experts` present → `architecture.is_moe = True`
- `num_key_value_heads != num_attention_heads` → `attention.type = "gqa"` (source=`inferred`)
- `q_lora_rank` / `kv_lora_rank` → MLA
- `sliding_window` → local attention
- `rope_scaling.type` → yarn / ntk / dynamic / linear

## safetensors extractor

**Do not load weights, read the header only.** The first 8 bytes are a little-endian header length, followed by JSON:

```python
import json, struct
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
# {"model.layers.0.self_attn.q_proj.weight":
#   {"dtype": "BF16", "shape": [4096, 4096], "data_offsets": [...]}, ...}
```

Outputs:

- `parameters.total` — summed over all tensor shapes (authoritative)
- `parameters.dtype_native`
- `architecture.tied_embeddings` — whether `lm_head.weight` exists
- the tensor name list — the last-resort fallback for architecture inference
- the whole `__metadata__` dict into passthrough (may contain SAI ModelSpec fields, training hyperparams)

> **Sharded models must read `model.safetensors.index.json` first** and aggregate every shard header, otherwise the parameter count comes out half.

## gguf extractor

Use the `gguf` Python package (official llama.cpp) to read the KV header without loading weights:

```python
from gguf import GGUFReader
r = GGUFReader(path)
arch = r.fields["general.architecture"].parts[-1].tobytes().decode()
n_ctx = int(r.fields[f"{arch}.context_length"].parts[-1][0])
n_expert = r.fields.get(f"{arch}.expert_count")   # present => MoE
file_type = r.fields["general.file_type"]         # quantization-level enum
```

- Parameter count: `sum(prod(t.shape) for t in r.tensors)`
- Byte size: look up the `ggml_type_size` / `blck_size` table per tensor type — **Q4_K_M is not 4 bits/param**; block quantization is easy to miscalculate
- canonical normalizes known keys, all unknown `general.*` / `{arch}.*` keys go to passthrough, and the full KV dump goes to raw

## license extractor (three-tier identification)

1. **Fingerprint match**: normalize the text of `LICENSE` / `LICENSE.md` / `LICENSE.txt` / `MODEL_LICENSE` / `USE_POLICY.md` / `Notice` (strip whitespace, lowercase, take the first N characters), hash it, and compare against a preset fingerprint table — directly matching Apache-2.0 / MIT / Llama 3 Community / Gemma Terms / Qwen License / OpenRAIL, etc. Collecting 20–30 common licenses first covers 90%.
2. **Keyword tags**: for misses, scan `commercial use` / `redistribute` / `derivative` / `attribution` / `acceptable use` and produce capability flags.
3. **LLM fallback**: feed the still-unclassified to a small model and classify into permissive / weak-copyleft / openrail / custom-commercial / proprietary, marked `confidence: low` for human review.

> Treat the HF model-card front-matter `license:` field as **auxiliary evidence**, not ground truth (it often disagrees with the files). When scanning, don't match only `LICENSE*`.

## tokenizer extractor

Read `tokenizer_config.json` / `tokenizer.json`: `type` / `vocab_size` / `chat_template_present`.

## Common pitfalls

- Sharded safetensors must read `index.json` first.
- Missing `lm_head.weight` on tied embeddings is normal, not a bug.
- Q4_K_M block-quant byte sizes need a table lookup, not bits/8.
- MoE shared experts must not be subtracted from active.
- When `trust_remote_code`'s `architectures[0]` is undecidable, **do not execute `modeling_*.py`** — fall back to tensor pattern matching.
- For adapter models, first check the root for `adapter_config.json`; `peft_type` distinguishes LoRA/DoRA; QLoRA is invisible at the adapter-config layer, so mark `quantization: unknown_from_adapter`.
- License file names seen in the wild: `MODEL_LICENSE` / `USE_POLICY.md` / `Notice`.

## Coverage sanity check

> Implemented as `modelspec coverage <repos.txt>` (M4) — see [analytics.md](analytics.md).

Run the ~36K mergekit models already tracked (or sample 1000) and tally:

- the `unknown_fields` frequency histogram → find the top 20 promotion candidates.
- canonical field fill rates → which fields fill in <50% of models (the alias table has gaps).
- fill-rate comparison across families → DeepSeek's `num_kv_heads` should be N/A rather than missing.

> This tally is itself a ParaPulse internal dashboard — metadata coverage is also a form of market intelligence.
