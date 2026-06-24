# Implementation Roadmap

## Startup priority

Ordered by "broadest coverage, clearest signal, lightest dependency":

1. **Pin down the Pydantic schema skeleton** (2–3 hours) — the top-level sub-models + `FieldClaim`. This is the target for all later extractors.
2. **safetensors header reader + config.json reader** — already covers 80% of HF models, with zero extra dependencies.
3. **GGUF reader** — adds the `gguf` package dependency (optional dependency).
4. **Parameter double-path check** — tensor sum vs config formula.
5. **License fingerprint table** — start with 20–30 common licenses.
6. **Architecture tag set** — multi-tag output, convenient for downstream search/filtering.

## Phased delivery of quantization & merge

Each round is an independent, valuable increment:

| Round | Content |
| --- | --- |
| 1 | the GGUF + AWQ/GPTQ quantization branches (broadest coverage, clearest signal) |
| 1 | merge `detection_signal` + `components` (model_id only), via HF tag + base_model array |
| 2 | the BnB / FP8 / MLX-native quantization branches |
| 3 | parse `mergekit_config.yml`, fill method + weights + density |
| 4 | parse the README YAML codeblock as a fallback signal |

> After the first two rounds, ParaPulse can already output segmentation analyses like "AWQ-quantized merges vs GGUF-quantized merges vs un-quantized merges".

## Milestones

### M1 — MVP ✅ done
- ✅ schema skeleton (the sub-models + `FieldClaim` + `Provenance`)
- ✅ config.json extractor (alias normalization, feature inference, three-layer output)
- ✅ safetensors extractor (header-only, sharded aggregation, parameter summation, tied-embed inference)
- ✅ pipeline: orchestrator + merger (confidence wins + tag union) + reshape
- ✅ io/hf_fetcher (header-only metadata downloads via Range requests)
- ✅ CLI `extract` / `schema`, outputting JSON / YAML
- ✅ 21 unit tests (parallel extractors / pipeline / schema structure), all green

> Code map in [development.md](development.md). In M1's scope `quantization` / `merge` / `adapter` are reserved fields (always `null`), landing in M3.

### M2 — multi-source + validation ✅ done
- ✅ GGUF extractor (`gguf` package conditional import; reads KV + tensor info without loading weights; arch/layers/heads/MoE/RoPE/tokenizer/params; the raw KV dump folds large arrays into a length marker)
- ✅ license extractor (three tiers: marker-phrase fingerprint → capability keywords → an LLM fallback hook; scans non-`LICENSE*` filenames too; README front-matter as low-confidence auxiliary evidence)
- ✅ tokenizer extractor (`tokenizer_config.json` chat_template + `tokenizer.json` type/vocab_size)
- ✅ `pipeline/cross_validate`: parameter double-path check, context three-layer consistency, MoE signal check → written to `provenance.warnings`
- ✅ io/hf_fetcher extensions: GGUF header prefix Range download, README.md download
- ✅ 35 unit tests, all green (including GGUF/license/tokenizer/cross_validate + a GGUF end-to-end)

> Code map in [development.md](development.md). `quantization` / `merge` / `adapter` are still reserved fields (GGUF `file_type` temporarily in passthrough), landing in M3.

### M3 — quantization + merge ✅ done
- ✅ quantization discriminated union (`Field(discriminator="format")`): GGUF / AWQ / GPTQ
  - the GGUF branch emitted by the gguf extractor: `file_type` name, **measured bits-per-weight** (look up `GGML_QUANT_SIZES`, not the nominal value), `tensor_types` mixed-precision distribution, `has_imatrix` filename heuristic
  - the AWQ / GPTQ branches emitted by the config_json extractor from `quantization_config`; an unknown `quant_method` emits no claim (the field stays `null`, so the discriminated union never meets an unknown discriminator)
- ✅ merge extractor (new source): five detection signals (`config_file` > `card_relation` > `base_model_array` > `hf_tag` > `readme_yaml`); parses the `mergekit_config.yml` recipe (method alias normalization + component weight/density); lightweight README front-matter parsing; the base_model chain written to `identity.lineage`
- ✅ `cross_validate` gains a merge-architecture consistency check
- ✅ quantization / merge / adapter coexist orthogonally (the stacking scenario verified end-to-end)
- ✅ 52 unit tests, all green (including quantization / merge + merge×quant end-to-end)

> Code map in [development.md](development.md). `adapter` remains a reserved field; the BnB / FP8 / MLX-native quantization branches and README YAML codeblock parsing are follow-up increments.

### M4 — evolution & feedback ✅ done
- ✅ `modelspec/analytics/batch.py`: `run_batch` (thread-pool concurrency, a single failure doesn't abort, order-preserving results) + `read_targets` (file / stdin, skipping comments and blanks)
- ✅ `modelspec/analytics/report.py`: `build_coverage_report` aggregates the three signals — the `unknown_fields` frequency histogram, canonical fill rates, per-family fill rates — plus promotion candidates and conflict/warning prevalence; text dashboard + JSON dual output
- ✅ field-promotion workflow: the report yields `promotion_candidates` (≥10% threshold), driving raw → passthrough → canonical
- ✅ CLI `batch` (batch extraction + optional spec write-out + unknown_fields frequency) / `coverage` (the full coverage dashboard); partial failure still `0`, all-failure `1`
- ✅ 11 unit tests, all green (batch fault-tolerance/ordering/sampling, report aggregation, CLI integration)

> Code map and field-promotion workflow in [analytics.md](analytics.md). You can run `coverage` over the ~36K mergekit corpus (or sample with `--limit`) as a sanity check.

### M5 — consumer help & ergonomics ✅ done
- ✅ CLI: richer `--help` (usage examples epilog), `explain <field>` (fuzzy field docs introspected from the schema's `description=`), `completion bash|zsh|fish` (static scripts, no extra dep)
- ✅ Library API: `ModelSpec` convenience accessors — `is_quantized()` / `is_merged()` / `is_moe()` / `is_derived()`, `quant_format` / `bits_per_weight` / `effective_context`, and provenance wrappers `source_of()` / `confidence_of()` / `is_not_applicable()` / `conflicts_for()`
- ✅ `modelspec/query.py`: composable predicates (`is_quantized`, `family_is`, `min_params`, …) + combinators (`all_of` / `any_of` / `negate`) + `filter_specs`
- ✅ `modelspec/explain.py`: `field_catalog()` / `explain_field()` flatten the live schema into dotted-path `FieldDoc`s
- ✅ 24 unit tests added (helpers / query / explain / CLI explain + completion)

> The orthogonal modeling + provenance are deliberately normalized for storage; M5 wraps the recurring access patterns so downstream consumers don't re-implement the edge cases. Code map in [helpers.md](helpers.md).

## Testing strategy

- **Extractor unit tests**: feed a fixture file, assert the `FieldClaim` list — no download, no Pydantic mocking needed.
- **Merger unit tests**: feed several groups of conflicting claims, assert the merge result and that conflicts are archived.
- **Schema unit tests**: feed a dict, assert validation passes / fails (cross-field validators).
- **Integration tests**: run a few real repos end-to-end (cacheable small fixtures).

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| config fields grow without bound | three-layer extraction + the unknown_fields feedback loop, promote only high-frequency fields |
| `trust_remote_code` custom architectures | don't execute `modeling_*.py`, fall back to tensor pattern matching |
| miscalculated quantized byte sizes | look up the ggml_type table, don't use bits/8 |
| missing merge-recipe data source | mark confidence low, components carry model_id only, accept the gap is a data-source limitation |
| schema compatibility burden | promote canonical fields carefully, version with `spec_version` |
