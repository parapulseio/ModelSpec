# ParaPulse ModelSpec

**Extract and normalize LLM model specifications from heterogeneous sources into one unified, type-safe schema.**

ParaPulse ModelSpec treats the various metadata sources a model ships with —
`config.json`, GGUF KV headers, safetensors headers, `LICENSE` files, tokenizer
configs — as **different projections of the same model spec**. A single pipeline
reads them, cross-validates, and normalizes everything into one Pydantic v2
`ModelSpec` with per-field provenance and confidence.

It is a **read/extract system for consumers**: it does not require model authors
to cooperate — if the files can be downloaded, they can be analyzed.

> Not to be confused with [Stability AI's ModelSpec](https://github.com/Stability-AI/ModelSpec),
> which is a *write* standard for image-generation training tools. This project
> goes the opposite direction: a multi-source *reader* for LLMs. See
> [docs/overview.md](docs/overview.md).

## Why

Hugging Face metadata is scattered, heterogeneous, and untrustworthy:

- `architectures[0]` is a custom class name under `trust_remote_code=True`.
- `config.json` has no real schema; the field set is open and family-specific.
- Quantized GGUF bits-per-weight is an *average*, not the nominal value
  (Q4_K_M is ~4.83 bpw, not 4.0).
- Merged (mergekit) models change parameter count and layer depth.

ParaPulse ModelSpec collapses that noise into a structure that is **type-safe,
queryable, evolvable, and lossless**.

## Design principles

1. **A unified schema is the foundation** — every extractor fills the same `ModelSpec`.
2. **Field-level provenance + confidence** — every field records its source and confidence; conflicts are traceable.
3. **Pydantic appears only at the boundary** — entry validation and exit serialization. Download/parse/infer/merge are plain Python.
4. **Three-layer extraction** — *canonical* (strict small set) / *passthrough* (buffer) / *raw* (lossless insurance). It does not chase full coverage.
5. **Orthogonal modeling** — quantization / merge / adapter are independent optional sub-structures, never a mutually-exclusive `model_type` enum.

## Project status

**M1 (MVP) and M2 (multi-source + validation) are complete.** See [docs/roadmap.md](docs/roadmap.md).

| Capability | Status |
| --- | --- |
| `ModelSpec` schema skeleton (8 sub-models + provenance) | ✅ M1 |
| `config.json` extractor (alias normalization, feature inference) | ✅ M1 |
| safetensors extractor (header-only, sharded aggregation, param count) | ✅ M1 |
| Pipeline: orchestrator + merger + reshape | ✅ M1 |
| Metadata-only HF fetch (Range-request headers, no weights) | ✅ M1 |
| CLI `extract` / `schema` (JSON / YAML output) | ✅ M1 |
| GGUF extractor (KV + tensor infos, no weights) | ✅ M2 |
| License extractor (fingerprint + keyword tiers) | ✅ M2 |
| Tokenizer extractor (type / vocab / chat template) | ✅ M2 |
| Cross-validation (param double-path, context, MoE signals) | ✅ M2 |
| Quantization / merge / adapter extraction | ⏳ M3 |

`quantization` / `merge` / `adapter` are reserved fields and are always `null`
until M3.

## Install

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # package + pytest + PyYAML + gguf
# optional: pip install -e ".[all]"   # runtime extras: gguf / yaml
```

## Usage

```bash
# Extract from a local model directory (offline, no network).
# Works for HF (config + safetensors) and GGUF directories alike.
modelspec extract /path/to/model/dir --offline

# Extract from a HF repo (downloads metadata only — a few MB, never the weights)
modelspec extract meta-llama/Llama-3.1-8B-Instruct

# YAML output, written to a file
modelspec extract ./model --offline --format yaml -o spec.yaml

# Include full provenance (per-field sources + raw config)
modelspec extract ./model --offline --show-provenance

# Export the JSON Schema of ModelSpec
modelspec schema
```

### Options (`extract`)

| Option | Description |
| --- | --- |
| `--format json\|yaml` | Output format (default: json) |
| `-o, --output PATH` | Write to a file instead of stdout |
| `--offline` | Local paths only, no network |
| `--revision REV` | Commit / branch / tag |
| `--show-provenance` | Include full provenance (`per_field`, `raw_config_json`) |
| `--strict` | Non-zero exit if any cross-field warning fires |

### Example output (excerpt)

```json
{
  "spec_version": "1.0",
  "identity": { "repo_id": "meta-llama/Llama-3.1-8B-Instruct", "source_format": "hf", "file_layout": "single" },
  "architecture": { "family": "llama", "num_layers": 32, "tied_embeddings": false,
                    "tags": ["decoder-only", "gqa", "rope-llama3"] },
  "attention": { "type": "gqa", "num_heads": 32, "num_kv_heads": 8 },
  "parameters": { "total": 8030261248, "dtype_native": "BF16" },
  "context": { "declared": 131072, "effective": 1048576 },
  "provenance": { "conflicts": [], "warnings": [], "unknown_fields": [] }
}
```

## Run the tests

```bash
pytest -q
```

Tests never hit the network and never mock Pydantic: extractors are fed fixture
files and asserted on their `FieldClaim` output; the schema is fed dicts and
asserted on validation.

## How it works

```
extract(path)
  └─ detect_source_format          # hf / gguf / adapter / raw
  └─ for each can_handle extractor:
        extract() -> [FieldClaim(field_path, value, source, confidence), ...]
  └─ merge_claims                  # highest confidence wins; tags unioned; conflicts logged
  └─ reshape                       # flat dotted paths -> nested dict + provenance
  └─ ModelSpec.model_validate      # Pydantic's only appearance (entry)
  └─ model_dump_json               # exit
```

## Project layout

```
modelspec/
├── schema/spec.py          # the ModelSpec Pydantic schema
├── extractors/             # one file per source; implement the Extractor protocol
│   ├── base.py             # Extractor protocol + FieldClaim + ExtractorResult
│   ├── config_json.py
│   └── safetensors.py
├── pipeline/               # orchestrator + merger
├── io/hf_fetcher.py        # metadata-only download (HTTP Range)
└── cli.py
tests/                      # parallel structure: extractors/ pipeline/ schema/
docs/                       # design docs
```

## Documentation

- [docs/overview.md](docs/overview.md) — overview, vs. Stability AI ModelSpec
- [docs/architecture.md](docs/architecture.md) — architecture, layout, end-to-end flow
- [docs/schema.md](docs/schema.md) — ModelSpec schema design
- [docs/extractors.md](docs/extractors.md) — extractors and three-layer extraction
- [docs/pipeline.md](docs/pipeline.md) — orchestration, merging, cross-validation
- [docs/quantization-and-merge.md](docs/quantization-and-merge.md) — quantization & merge modeling
- [docs/cli.md](docs/cli.md) — CLI reference
- [docs/development.md](docs/development.md) — setup, run, test, M1 code map
- [docs/roadmap.md](docs/roadmap.md) — roadmap & milestones
- [AGENTS.md](AGENTS.md) — conventions for AI coding agents

## License

Apache-2.0
