# Development Guide

## Environment setup

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # the package + pytest + PyYAML + gguf (dev includes gguf for testing)
# optional: pip install -e ".[all]"  # extra runtime gguf / yaml
```

## Running the CLI

```bash
# extract a local model directory (offline, no network)
modelspec extract /path/to/model/dir --offline

# extract an HF repo (metadata only, no weights)
modelspec extract meta-llama/Llama-3.1-8B-Instruct

# YAML output + write to a file
modelspec extract ./model --offline --format yaml -o spec.yaml

# include full provenance (per_field + raw_config_json)
modelspec extract ./model --offline --show-provenance

# export the JSON Schema
modelspec schema

# batch extraction + coverage dashboard (M4)
modelspec batch repos.txt --offline --output-dir specs/
modelspec coverage repos.txt --offline
```

Full options in [cli.md](cli.md); batch/coverage and the field-promotion workflow in [analytics.md](analytics.md).

## Running tests

```bash
pytest -q
```

Tests don't touch the network and don't mock Pydantic: extractors are fed fixture files and assert `FieldClaim`s; the schema is fed dicts and asserts validation. Fixture-construction helpers are in `tests/conftest.py` (`write_config` / `write_safetensors_header` write a header-only safetensors; `write_gguf` uses the official `gguf` writer to write a tiny real GGUF). GGUF-related tests use `pytest.importorskip("gguf")` and skip automatically when gguf isn't installed.

## Code map (M1 + M2 + M3 + M4)

| Design doc | Code |
| --- | --- |
| [schema.md](schema.md) | `modelspec/schema/spec.py` — `ModelSpec` + sub-models + the `Quantization` discriminated union + `MergeSpec` + `Provenance` |
| [extractors.md](extractors.md) | `extractors/base.py` (protocol + `FieldClaim` + `ExtractorResult`), `config_json.py`, `safetensors.py`, `gguf.py`, `license.py`, `tokenizer.py`, `merge.py` |
| [quantization-and-merge.md](quantization-and-merge.md) | quantization claims emitted by `config_json.py` (AWQ/GPTQ) + `gguf.py` (GGUF); merge by `merge.py` |
| [pipeline.md](pipeline.md) | `pipeline/orchestrator.py` (detect / reshape / extract), `merger.py` (conflict resolution + tag union), `cross_validate.py` (parameter double-path / context three-layer / MoE / merge-architecture checks) |
| [analytics.md](analytics.md) | `analytics/batch.py` (`run_batch` / `read_targets`), `analytics/report.py` (`build_coverage_report`) |
| [architecture.md](architecture.md) | `io/hf_fetcher.py` (safetensors header / GGUF prefix / small-file Range downloads) |
| [cli.md](cli.md) | `modelspec/cli.py` (`extract` / `schema` / `batch` / `coverage`) |

## End-to-end data flow (as tested)

`extract(path)` →
`detect_source_format` →
for each `can_handle` extractor, call `extract` and collect `FieldClaim`s (the GGUF extractor opts out automatically without the `gguf` package) →
`merge_claims` (confidence wins, `architecture.tags` are unioned, conflicts go to `conflicts`) →
`reshape` (flat dotted paths → nested dict, backfilling `provenance.per_field` / `raw_config_json` / `raw_gguf_kv`) →
`ModelSpec.model_validate` (Pydantic's first appearance) →
`cross_validate` (appends to `provenance.warnings`, never raises) →
`model_dump_json`.

## Known boundaries (M4 status)

- Supports six source types: `config.json` + safetensors + GGUF + license + tokenizer + merge; `batch` / `coverage` provide batch extraction and coverage feedback (see [analytics.md](analytics.md)).
- **Batch extraction**: thread-pool concurrency (IO-bound), a single failure is recorded without aborting; the coverage report is aggregated entirely from `provenance`, no re-download.
- **quantization**: the GGUF / AWQ / GPTQ discriminated-union branches are landed; BnB / FP8 / MLX-native are follow-up increments; an unknown `quant_method` emits no claim (the field stays `null`). GGUF bits-per-weight is a measured value (look up `GGML_QUANT_SIZES`).
- **merge**: the five detection signals are landed; recipe details (method/weights/density) depend on `mergekit_config.yml` existing + PyYAML installed (the `yaml` extra); without PyYAML, "it's a merge" is still detected with high confidence but the recipe isn't parsed. README YAML codeblock parsing is a fallback signal (basic matching supported).
- `adapter` remains a reserved field, always `null`.
- The parameter double-path check is a sanity check: path B's estimate ignores bias/norm and MoE routing, so a small diff is normal; for GGUF quantized models the tensor count is authoritative.
- Remote downloads fetch headers only: safetensors fetches the first `8+n` bytes, GGUF fetches a prefix (default 24MB, covering header + tensor info); an oversized KV (very rare) may be truncated, but local files have no such limit. Per-file/per-shard downloads run **concurrently** (`_FETCH_WORKERS`, default 16), so a 685B model with 160+ shards extracts in ~15s; a single failed download fails the target (matching the old sequential behaviour).
- **Range robustness**: if the HF CDN ignores the `Range` header and returns `200` (the whole file), `io/hf_fetcher._read_prefix` streams the body and disconnects once it has the target bytes, so it never downloads multi-GB weights in full. Supports `HF_TOKEN` for higher rate limits.
- The third LLM tier of license identification is currently a hook (no model wired up); the `--no-license-llm` option is reserved.
- The `io` sub-package name collides with the stdlib `io` but doesn't conflict (absolute import `modelspec.io`).

## Adding an extractor (conventions recap)

1. Create a new file under `modelspec/extractors/` implementing the `Extractor` protocol (`name` / `can_handle` / `extract`).
2. `extract` returns an `ExtractorResult`: a list of canonical `FieldClaim`s + `passthrough` + `raw` + `unknown_fields`.
3. Register it in `ALL_EXTRACTORS` in `modelspec/extractors/__init__.py`.
4. Add `tests/extractors/test_<name>.py`.
5. **Do not touch the orchestrator** — it depends only on the protocol.
