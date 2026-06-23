# Batch Extraction & Evolution Feedback (M4)

M1–M3 solve "**extract one model**"; M4 turns back to govern the system itself — aggregating the `provenance` accumulated per model (`unknown_fields` / `per_field` / `conflicts` / `warnings`) into **cross-corpus signals** that, in turn, drive the next round of schema and extractor evolution.

> This is a **feedback loop**, not a new source. The code lives in `modelspec/analytics/`, exposed through the `batch` / `coverage` CLI subcommands.

## Three things

| Capability | Signal | Question it answers |
| --- | --- | --- |
| `unknown_fields` frequency | fields that recur in raw but were claimed by neither canonical nor passthrough | **what to start extracting** (promotion candidates) |
| canonical fill rate | the fraction of **applicable** models in which each canonical field was filled (models that flagged the field `not_applicable` are excluded from the denominator) | **what existing extraction is missing** (alias-table gaps / extractor bugs), without N/A false alarms |
| per-family fill rate | fill rate grouped by `architecture.family` | distinguish "**semantically N/A**" from "**missed extraction**" (e.g. DeepSeek-MLA simply has no `num_kv_heads`) |

## Module structure

```
modelspec/analytics/
├── batch.py      # run_batch: concurrent, fault-tolerant, order-preserving batch extraction; read_targets
└── report.py     # build_coverage_report: aggregate the three signals + text/JSON rendering
```

- **`run_batch(targets, *, offline, revision, max_workers, limit, on_progress, target_timeout)`** → `BatchResult`
  - IO-bound (metadata downloads), runs targets concurrently in daemon threads; **a single target's failure is recorded and never aborts the batch**.
  - **Per-target timeout** (`target_timeout`, default 120s): a genuinely slow / hung repo is abandoned and recorded as a `TimeoutError` failure, so it can't stall the whole run. Per-shard header downloads run concurrently (see `io/hf_fetcher`), so even a 685B model with 160+ shards extracts in ~15s rather than timing out. Set a HF token (`HF_TOKEN`) to avoid rate-limit slowdowns.
  - **Auth advisory**: HF attaches a `Warning` header to anonymous requests ("You are sending unauthenticated requests …") that huggingface_hub re-logs; the CLI raises the hub logger to ERROR so this advisory doesn't flood the dashboard. It is harmless — extraction still succeeds. Export a valid `HF_TOKEN` to actually speed up the underlying downloads.
  - Results preserve order; `BatchResult` exposes `specs` / `failures` / `succeeded` / `failed` / `total`.
- **`build_coverage_report(result, *, promotion_threshold=0.10, top_n=20)`** → `CoverageReport`
  - Aggregated entirely from `provenance`, **no re-download**.
  - `promotion_candidates`: unknown fields appearing in ≥10% of models (the raw → passthrough promotion threshold, see [extractors.md](extractors.md)).
  - **Fill rates are over applicable models**: for each field the denominator subtracts the models that listed it in `provenance.not_applicable`, so a legitimately-absent field (e.g. `num_kv_heads` under MLA) is never flagged `<-- low`. The report also exposes per-field `na_counts`.

## CLI

```bash
# batch extraction: optionally write out each spec; the report focuses on unknown_fields frequency
modelspec batch repos.txt --offline --output-dir specs/

# coverage sanity check: the full dashboard (fill rates + per-family + promotion candidates)
modelspec coverage repos.txt --offline

# for a large corpus, sample first; JSON for machine consumption
modelspec coverage repos.txt --limit 1000 --workers 16 --format json
```

### Targets file format

The targets file is plain text, parsed by `read_targets`:

- **One target per line.** A target is anything `extract` accepts: an **HF repo id** (`org/name`) or a **local directory path** (absolute or relative).
- **`#` starts a comment** — both full-line comments and inline trailing comments are stripped.
- **Blank lines are ignored.**
- **No quoting / escaping** — the line (after stripping the comment and surrounding whitespace) is the target verbatim. Don't wrap paths in quotes.
- **`-` as the filename** (i.e. `modelspec coverage -`) reads the list from **stdin** instead of a file, so you can pipe: `cat repos.txt | modelspec coverage -`.

Example (`repos.txt`):

```text
# HF repo ids
meta-llama/Llama-3.1-8B-Instruct
Qwen/Qwen2.5-7B-Instruct
TheBloke/Mistral-7B-v0.1-GGUF      # a GGUF repo

# local directories (use with --offline)
/data/models/my-merge
./fixtures/tiny-llama
```

> Mixing repo ids and local paths in one file is fine. With `--offline`, repo-id lines that aren't local directories will simply fail and be recorded in the report's `failures` (they don't abort the run).

| Option | Description |
| --- | --- |
| `--offline` | local paths only, no network |
| `--workers N` | concurrent extractions (default 8) |
| `--limit N` | process only the first N (sampling) |
| `--target-timeout S` | per-target seconds budget (default 120); a slow/hung repo is recorded as a `TimeoutError` failure instead of stalling the batch. `0` disables it. |
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
4. Also watch the canonical fill rates: a canonical field flagged `<-- low` is now a real gap (N/A models are already excluded) — usually an **alias-table gap**, so fill in `ALIASES`. A field shown with `(n/a: N)` or `(all N N/A)` is legitimately absent for those models and needs no fix; if a family is wrongly counted as a gap, the relevant extractor should emit `not_applicable` for it (as the MLA branch does for `num_kv_heads`).

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

canonical fill rates (over applicable models):
  98.7%  architecture.family
  99.1%  attention.num_kv_heads  (n/a: 210)   # MLA models excluded from the denominator
  45.0%  tokenizer.vocab_size        <-- low  # a real gap
  ...

per-family fill rates (family: n):
  llama (n=18020): family=100%  num_kv_heads=99%  ...
  deepseek_v3 (n=210): family=100%  num_kv_heads=0%  ...   # MLA: N/A, not a missed extraction
```
