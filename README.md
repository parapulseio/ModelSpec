# ParaPulse ModelSpec

**Extract and normalize LLM model specifications from heterogeneous sources into one unified, type-safe schema.**

ParaPulse ModelSpec treats the various metadata sources a model ships with —
`config.json`, GGUF KV headers, safetensors headers, `LICENSE` files, tokenizer
configs, mergekit recipes — as **different projections of the same model spec**.
A single pipeline reads them, cross-validates, and normalizes everything into one
Pydantic v2 `ModelSpec` with per-field provenance and confidence.

It is a **read/extract system for consumers**: it does not require model authors
to cooperate — if the files can be downloaded, they can be analyzed. Only
metadata is fetched (a few MB), never the weights.

> Not to be confused with [Stability AI's ModelSpec](https://github.com/Stability-AI/ModelSpec),
> which is a *write* standard for image-generation training tools. This project
> goes the opposite direction: a multi-source *reader* for LLMs. See
> [docs/overview.md](docs/overview.md).

## Why

Hugging Face metadata is scattered, heterogeneous, and untrustworthy:

- `architectures[0]` is a custom class name under `trust_remote_code=True`.
- `config.json` has no real schema; the field set is open and family-specific,
  and VLMs nest the language-model fields under `text_config`.
- Quantized GGUF bits-per-weight is an *average*, not the nominal value
  (Q4_K_M is ~4.83 bpw, not 4.0).
- Merged (mergekit) models change parameter count and layer depth.
- Many fields differ by source (config vs tokenizer vocab size) or are
  legitimately N/A (no kv-heads under MLA, no context window for audio models).

ParaPulse ModelSpec collapses that noise into a structure that is **type-safe,
queryable, evolvable, and lossless**, and tells you *where each value came from*.

## Design principles

1. **A unified schema is the foundation** — every extractor fills the same `ModelSpec`.
2. **Field-level provenance + confidence** — every field records its source and confidence; cross-source conflicts are archived, and "legitimately N/A" is distinguished from "missing".
3. **Pydantic appears only at the boundary** — entry validation and exit serialization. Download/parse/infer/merge are plain Python.
4. **Three-layer extraction** — *canonical* (strict small set) / *passthrough* (buffer) / *raw* (lossless insurance). It does not chase full coverage; an `unknown_fields` feedback loop drives what to extract next.
5. **Orthogonal modeling** — quantization / merge / adapter are independent optional sub-structures, never a mutually-exclusive `model_type` enum.

## Project status

**All roadmap milestones (M1–M5) are complete**, plus a round of corpus-driven
hardening. On a mixed 1000-model corpus, extraction succeeds on **~96%** (the
rest are gated/private or 1-off malformed files); the curated decoder-LLM fields
fill at **90%+**. See [docs/roadmap.md](docs/roadmap.md).

| Capability | Status |
| --- | --- |
| `ModelSpec` schema (sub-models + provenance, self-documenting `Field(description=…)`) | ✅ M1 |
| `config.json` extractor (aliases, feature inference, **VLM `text_config`**, **model-kind/scope**) | ✅ M1+ |
| safetensors extractor (header-only, sharded aggregation, param count) | ✅ M1 |
| Pipeline: orchestrator + merger + reshape | ✅ M1 |
| Metadata-only HF fetch (Range-request headers, **parallel shards**, **concurrency cap**) | ✅ M1+ |
| GGUF extractor (**own binary header parser**, no weights, measured bpw) | ✅ M2 |
| License extractor (fingerprint + keyword tiers) | ✅ M2 |
| Tokenizer extractor (type / vocab / chat template / special tokens) | ✅ M2 |
| Cross-validation (param double-path, context, MoE / merge signals) | ✅ M2 |
| Quantization discriminated union (GGUF / AWQ / GPTQ) | ✅ M3 |
| Merge detection (5 signals) + recipe/components + lineage | ✅ M3 |
| Batch extraction + coverage dashboard, **conflict histogram**, throttling | ✅ M4 |
| Help layer: `explain` / `completion` + consumer query helpers | ✅ M5 |
| BnB / FP8 / MLX quantization, adapter extraction | ⏳ later |

`adapter` is a reserved field and is always `null` for now.

## Install

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # package + pytest + PyYAML + gguf
# optional: pip install -e ".[all]"   # runtime extras: gguf / yaml
```

Set `HF_TOKEN` (exported!) for authenticated, higher-rate Hub access:
`export HF_TOKEN=hf_...`.

## Usage

```bash
# Extract one model (HF repo id or local dir). Metadata only — never the weights.
modelspec extract meta-llama/Llama-3.1-8B-Instruct
modelspec extract /path/to/model/dir --offline      # HF or GGUF layout
modelspec extract ./model --offline --format yaml -o spec.yaml
modelspec extract ./model --offline --show-provenance   # per-field sources + raw config

# Export the JSON Schema, or explain a field
modelspec schema
modelspec explain head_dim          # type, choices, description (from the schema)
modelspec completion bash           # shell tab-completion script

# Batch over a list of models -> a coverage dashboard / report
modelspec batch repos.txt --output-dir specs/
modelspec coverage repos.txt --workers 8 --delay 0.3 --format json > report.json
```

### Options

`extract`:

| Option | Description |
| --- | --- |
| `--format json\|yaml` | Output format (default: json) |
| `-o, --output PATH` | Write to a file instead of stdout |
| `--offline` | Local paths only, no network |
| `--revision REV` | Commit / branch / tag |
| `--show-provenance` | Include full provenance (`per_field`, `raw_config_json`) |
| `--strict` | Non-zero exit if any cross-field warning fires |

`batch` / `coverage` (see [docs/analytics.md](docs/analytics.md)): `--workers N`,
`--limit N`, `--target-timeout S`, `--delay S` (throttle the Hub request rate),
`--format text|json`, `--top N`, `--quiet`, `--output-dir DIR` (batch only).

### Example output (excerpt)

```json
{
  "spec_version": "1.0",
  "identity": { "repo_id": "meta-llama/Llama-3.1-8B-Instruct", "source_format": "hf", "file_layout": "sharded" },
  "architecture": { "family": "llama", "num_layers": 32, "hidden_size": 4096, "head_dim": 128,
                    "tied_embeddings": false, "tags": ["decoder-only", "gqa", "rope-llama3"] },
  "attention": { "type": "gqa", "num_heads": 32, "num_kv_heads": 8 },
  "parameters": { "total": 8030261248, "dtype_native": "BF16" },
  "context": { "declared": 131072, "effective": 1048576 },
  "tokenizer": { "type": "BPE", "vocab_size": 128256, "eos_token_id": [128001, 128009] },
  "license": { "spdx_id": "llama3.1", "commercial_use": true, "confidence_tier": "fingerprint" },
  "quantization": null, "merge": null,
  "provenance": { "conflicts": [], "warnings": [], "not_applicable": [], "unknown_fields": ["pretraining_tp"] }
}
```

## Consumer helpers

The schema is normalized for storage; a thin helper layer wraps the common
access patterns so downstream callers don't re-implement edge cases
(see [docs/helpers.md](docs/helpers.md)):

```python
from modelspec.pipeline import extract
from modelspec.query import filter_specs, all_of, is_quantized, min_params, modality_is

spec = extract("TheBloke/Mistral-7B-v0.1-GGUF")
spec.is_quantized(); spec.quant_format; spec.bits_per_weight     # gguf, 4.83
spec.effective_context                                            # 32768
spec.modality; spec.is_decoder_only(); spec.is_multimodal()      # "decoder-only"
spec.source_of("architecture.family"); spec.is_not_applicable("attention.num_kv_heads")

# corpus-level predicate filtering
decoder_llm_quants = list(filter_specs(specs, all_of(
    is_quantized, min_params(7e9), modality_is("decoder-only"))))
```

## Run the tests

```bash
pytest -q     # 104 tests; no network, no Pydantic mocking
```

Extractors are fed fixture files (`tests/conftest.py` writes header-only
safetensors and tiny real GGUFs) and asserted on their `FieldClaim` output; the
schema is fed dicts and asserted on validation.

## How it works

```
extract(path)
  └─ detect_source_format          # hf / gguf / adapter / raw
  └─ fetch metadata (parallel, headers only, global concurrency cap)
  └─ for each can_handle extractor: extract() -> [FieldClaim(path, value, source, confidence), ...]
  └─ merge_claims                  # highest confidence wins; tags unioned; conflicts archived
  └─ reshape                       # flat dotted paths -> nested dict + provenance (+ not_applicable)
  └─ ModelSpec.model_validate      # Pydantic's only appearance (entry)
  └─ cross_validate                # multi-source warnings (never raises)
  └─ model_dump_json               # exit
```

## Project layout

```
modelspec/
├── schema/spec.py          # the ModelSpec Pydantic schema + consumer accessors
├── extractors/             # one file per source; implement the Extractor protocol
│   ├── base.py             # Extractor protocol + FieldClaim + ExtractorResult
│   ├── config_json.py  safetensors.py  gguf.py  license.py  tokenizer.py  merge.py
├── pipeline/               # orchestrator + merger + cross_validate
├── analytics/              # batch (concurrent extraction) + report (coverage)
├── io/hf_fetcher.py        # metadata-only download (HTTP Range, parallel, capped)
├── explain.py  query.py    # field catalog + predicate/filter library (M5)
└── cli.py                  # extract / schema / batch / coverage / explain / completion
tests/                      # parallel structure: extractors/ pipeline/ schema/ analytics/ io/
docs/                       # design docs
```

## Documentation

- [docs/overview.md](docs/overview.md) — overview, vs. Stability AI ModelSpec
- [docs/architecture.md](docs/architecture.md) — architecture, layout, end-to-end flow
- [docs/schema.md](docs/schema.md) — ModelSpec schema design
- [docs/schema-review.md](docs/schema-review.md) — schema fitness review + proposed fields
- [docs/extractors.md](docs/extractors.md) — extractors, three-layer extraction, VLM & scope
- [docs/pipeline.md](docs/pipeline.md) — orchestration, merging, cross-validation
- [docs/quantization-and-merge.md](docs/quantization-and-merge.md) — quantization & merge modeling
- [docs/cli.md](docs/cli.md) — CLI reference
- [docs/analytics.md](docs/analytics.md) — batch extraction, coverage, field promotion
- [docs/helpers.md](docs/helpers.md) — consumer accessors + query/filter library
- [docs/development.md](docs/development.md) — setup, run, test, code map
- [docs/roadmap.md](docs/roadmap.md) — roadmap & milestones
- [AGENTS.md](AGENTS.md) — conventions for AI coding agents

## License

Apache-2.0
