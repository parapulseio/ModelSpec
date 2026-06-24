# AGENTS.md

This file gives AI coding agents the context and conventions needed to work in this repo. Human developers can use it too.

## What this project is

**ParaPulse ModelSpec** — a model-spec extraction and normalization system. It treats the various metadata sources a model carries on the HF Hub / disk (`config.json`, GGUF KV header, safetensors header, `LICENSE`, tokenizer config) as different projections of the same model spec, then extracts, cross-validates, and normalizes them into a Pydantic v2 `ModelSpec` with field-level confidence.

It is a consumer-facing **read / extract** system: it does not depend on model authors cooperating. See [docs/overview.md](docs/overview.md).

## Core design principles (understand before touching code)

1. **The unified schema is the foundation** — every extractor fills the same `ModelSpec`. Change a field by first updating the design in [docs/schema.md](docs/schema.md).
2. **Pydantic appears only at the two boundaries** — entry validation (`model_validate`) and exit serialization (`model_dump_json`). The download / parse / inference / merge in between is plain Python, decoupled from Pydantic. **Do not let Pydantic dependencies leak into extractors.**
3. **Extractors return flat `FieldClaim` four-tuples** (`field_path, value, source, confidence`), not nested dicts.
4. **Three-layer extraction**: canonical (a strictly controlled small set) / passthrough (buffer) / raw (lossless insurance). Do not chase covering every field.
5. **Orthogonal modeling**: quantization / merge / adapter are independent optional sub-structures, mutually non-exclusive — **never use a `model_type` enum to make them mutually exclusive**.
6. **Field-level provenance + confidence**; conflicts are archived in `provenance.conflicts`.

## Directory layout

```
modelspec/
├── schema/spec.py          # Pydantic ModelSpec — the system's spec definition
├── extractors/             # one file per source, implementing the Extractor protocol
│   ├── base.py             # Extractor Protocol + FieldClaim + ExtractorResult
│   ├── config_json.py / safetensors.py / gguf.py / license.py / tokenizer.py / merge.py
├── pipeline/               # orchestrator / merger / cross_validate
├── analytics/              # batch (batch extraction) / report (coverage aggregation) — M4
├── io/hf_fetcher.py        # downloads, HTTP Range requests (headers only)
└── cli.py                  # extract / schema / batch / coverage
tests/                      # parallel structure: extractors/ pipeline/ schema/ analytics/
docs/                       # design docs
```

## Key constraints (violating these causes bugs)

- **Never download weights** — safetensors uses an HTTP Range request for the header only (the first 8 bytes give the JSON length); GGUF is parsed by our own binary header reader (`parse_gguf_header`), never `gguf.GGUFReader` (which builds numpy views over tensor *data* and crashes on the truncated metadata-only prefix). The `gguf` package is used only for its type/size data tables.
- **Sharded safetensors must read `model.safetensors.index.json` first** and aggregate all shards, otherwise the parameter count comes out half.
- **Look up quantized byte sizes in the ggml_type table**, do not use bits/8. Q4_K_M is actually ~4.83 bpw (an average, summed from the tensor list).
- **MoE shared experts are always active** — do not subtract them from active.
- **Never execute `modeling_*.py`** — when `trust_remote_code`'s `architectures[0]` is undecidable, fall back to tensor-name pattern matching.
- **Missing `lm_head.weight` on tied embeddings is normal**, not a bug.
- **Do not match only `LICENSE*`** — there are also `MODEL_LICENSE` / `USE_POLICY.md` / `Notice`. The HF front-matter `license:` is auxiliary evidence, not ground truth.
- **frankenmerge / passthrough merge change layer count and parameter count** — recompute from tensors; do not assume they match the components.
- **The base_model chain lives in `identity.lineage`**, not in the individual sub-structures.
- **quantization is a `Field(discriminator="format")` discriminated union** — for an unknown `quant_method`, emit no `quantization.*` claim (leave the field `null`), otherwise the union fails validation on an unknown discriminator. GGUF bpw must be measured (look up `GGML_QUANT_SIZES`), not the nominal value.

## Tech stack

- Python, Pydantic v2 (**there is no v3**; use `model_dump` / `model_validate` / `@field_validator` / `model_config = ConfigDict(...)`).
- Dependencies: `pydantic>=2`, `huggingface_hub`, `requests`; `gguf>=0.10` is an optional dependency (conditional import).
- Packaging: a single package with multiple sub-modules (option B). **Do not split the schema into its own package** (an anti-pattern).

## Steps to add an extractor

1. Create a new file under `extractors/` implementing the `Extractor` protocol (`can_handle` + `extract`).
2. Return the three layers: a list of canonical `FieldClaim`s + a passthrough dict + raw + `unknown_fields`.
3. Register it in `ALL_EXTRACTORS` in `extractors/__init__.py`.
4. Add `tests/extractors/test_<name>.py` feeding a fixture and asserting the claims.
5. **Do not touch the orchestrator** — it only talks to the protocol.

## Current status

- **M1 done**: schema skeleton + config/safetensors extractor + pipeline + io/hf_fetcher + CLI.
- **M2 done**: GGUF / license / tokenizer extractors + `pipeline/cross_validate` (parameter double-path, context three-layer, MoE cross-check).
- **M3 done**: quantization discriminated union (GGUF/AWQ/GPTQ, emitted by the gguf + config_json extractors) + merge extractor (five signals + recipe parsing + lineage).
- **M4 done**: `modelspec/analytics/` (`batch` concurrent batch extraction + `coverage` dashboard), aggregating `provenance` into `unknown_fields` frequency / canonical fill rates / per-family fill rates + promotion candidates, driving the field-promotion workflow. 63 unit tests, all green. Code map in [docs/development.md](docs/development.md) and [docs/analytics.md](docs/analytics.md).
- **M5 done**: consumer-facing help. CLI: richer `--help` / usage examples, `modelspec explain <field>` (fuzzy field docs introspected from the schema's `description=`), `modelspec completion bash|zsh|fish`. Library API: `ModelSpec` convenience accessors (`is_quantized()` / `effective_context` / `source_of()` …) wrapping the orthogonal structures + provenance, plus `modelspec.query` (composable predicates + `filter_specs`) and `modelspec.explain` (`field_catalog` / `explain_field`). See [docs/helpers.md](docs/helpers.md). 87 unit tests, all green.
- All roadmap milestones are complete. Follow-up increments: BnB/FP8/MLX quantization branches, adapter extraction, README YAML codeblock parsing. `gguf` / `PyYAML` are optional deps (included in the `dev` extras); without PyYAML, merge is still detected but the `mergekit_config.yml` recipe is not parsed.

## Environment & testing conventions

- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.
- Run tests: `pytest -q` (no network, no Pydantic mocking).
- Extractor unit tests feed fixture files (`tests/conftest.py` provides `write_config` / `write_safetensors_header`).
- Schema unit tests feed dicts and assert validation.
- CLI smoke test: `modelspec extract <dir> --offline`.

## Pre-commit checklist

- Changed the schema → also update [docs/schema.md](docs/schema.md).
- Added a canonical field → confirm there is a downstream consumer (canonical carries a compatibility burden).
- New fields go to passthrough first; let `unknown_fields` frequency decide whether to promote.

## Doc index

- [docs/overview.md](docs/overview.md) — overview, difference vs Stability AI ModelSpec
- [docs/architecture.md](docs/architecture.md) — architecture, layout, end-to-end flow
- [docs/schema.md](docs/schema.md) — ModelSpec schema design
- [docs/schema-review.md](docs/schema-review.md) — schema fitness review (comparison / selection / dev / inference / viz) + proposed additions
- [docs/extractors.md](docs/extractors.md) — extractors and three-layer extraction
- [docs/pipeline.md](docs/pipeline.md) — orchestration, merging, cross-validation
- [docs/quantization-and-merge.md](docs/quantization-and-merge.md) — quantization & merge modeling
- [docs/cli.md](docs/cli.md) — CLI design (incl. `explain` / `completion`, M5)
- [docs/helpers.md](docs/helpers.md) — consumer helpers: spec accessors, query predicates, field catalog (M5)
- [docs/analytics.md](docs/analytics.md) — batch extraction, coverage, field promotion (M4)
- [docs/development.md](docs/development.md) — install, run, test, code map
- [docs/roadmap.md](docs/roadmap.md) — roadmap & milestones
