# Batch Extraction & Evolution Feedback (M4)

M1–M3 solve "**extract one model**"; M4 turns back to govern the system itself — aggregating the `provenance` accumulated per model (`unknown_fields` / `per_field` / `conflicts` / `warnings`) into **cross-corpus signals** that, in turn, drive the next round of schema and extractor evolution.

> This is a **feedback loop**, not a new source. The code lives in `modelspec/analytics/`, exposed through the `batch` / `coverage` CLI subcommands.

## Three things

| Capability | Signal | Question it answers |
| --- | --- | --- |
| `unknown_fields` frequency | fields that recur in raw but were claimed by neither canonical nor passthrough | **what to start extracting** (promotion candidates) |
| canonical fill rate | the fraction of models in which each canonical field was successfully filled | **what existing extraction is missing** (alias-table gaps / extractor bugs) |
| per-family fill rate | fill rate grouped by `architecture.family` | distinguish "**semantically N/A**" from "**missed extraction**" (e.g. DeepSeek-MLA simply has no `num_kv_heads`) |

## Module structure

```
modelspec/analytics/
├── batch.py      # run_batch: concurrent, fault-tolerant, order-preserving batch extraction; read_targets
└── report.py     # build_coverage_report: aggregate the three signals + text/JSON rendering
```

- **`run_batch(targets, *, offline, revision, max_workers, limit, on_progress)`** → `BatchResult`
  - IO-bound (metadata downloads), uses a thread pool; **a single target's failure is recorded and never aborts the batch**.
  - Results preserve order; `BatchResult` exposes `specs` / `failures` / `succeeded` / `failed` / `total`.
- **`build_coverage_report(result, *, promotion_threshold=0.10, top_n=20)`** → `CoverageReport`
  - Aggregated entirely from `provenance`, **no re-download**.
  - `promotion_candidates`: unknown fields appearing in ≥10% of models (the raw → passthrough promotion threshold, see [extractors.md](extractors.md)).

## CLI

```bash
# batch extraction: optionally write out each spec; the report focuses on unknown_fields frequency
modelspec batch repos.txt --offline --output-dir specs/

# coverage sanity check: the full dashboard (fill rates + per-family + promotion candidates)
modelspec coverage repos.txt --offline

# for a large corpus, sample first; JSON for machine consumption
modelspec coverage repos.txt --limit 1000 --workers 16 --format json
```

`repos.txt`: one HF repo id or local directory path per line; `#` comments and blank lines are ignored; `-` reads from stdin.

| Option | Description |
| --- | --- |
| `--offline` | local paths only, no network |
| `--workers N` | concurrent extractions (default 8) |
| `--limit N` | process only the first N (sampling) |
| `--format text\|json` | output format (default: the text dashboard) |
| `--top N` | rows in the frequency tables (default 20) |
| `--quiet` | suppress the stderr progress line |
| `--output-dir DIR` | (`batch` only) write each spec as JSON |

**Exit codes**: cannot read the targets file → `2`; **partial failure is normal** (the norm at corpus scale), still `0`; only "everything failed, zero successes" → `1`.

## Field-promotion workflow (raw → passthrough → canonical)

Driven by report data, to avoid adding fields on a hunch:

1. Run `coverage` (or sample first with `--limit`) → look at `promotion_candidates`.
2. **raw → passthrough**: a field appearing in >10% of models that you can describe in one sentence → add it to the relevant extractor's `KNOWN_PASSTHROUGH`.
3. **passthrough → canonical**: only promote to a strictly-typed field when there is a clear downstream consumer (canonical carries a compatibility burden, be careful).
4. Also watch the canonical fill rates: a canonical field <50% is usually an **alias-table gap** — fill in `ALIASES`; if it concentrates as N/A within a family (e.g. DeepSeek `num_kv_heads`), that's semantically normal and needs no fix.

## Example output (excerpt)

```
models: 36000  ok: 35712  failed: 288
  warnings in 4120 models, conflicts in 990 models

unknown_fields frequency (top 20):
  62.3%   22270  decoder_sparse_step
  31.0%   11080  final_logit_softcapping
  ...

promotion candidates (>= 10% of models):
  62.3%   22270  decoder_sparse_step
  ...

canonical fill rates:
  98.7%  architecture.family
  61.2%  attention.num_kv_heads        <-- low
  ...

per-family fill rates (family: n):
  llama (n=18020): family=100%  num_kv_heads=99%  ...
  deepseek_v3 (n=210): family=100%  num_kv_heads=0%  ...   # MLA: N/A, not a missed extraction
```
