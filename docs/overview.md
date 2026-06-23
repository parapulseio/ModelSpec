# ParaPulse ModelSpec — Project Overview

## What this is

ParaPulse ModelSpec is a **model-spec extraction and normalization system**. It treats a model's various metadata sources on disk / the Hugging Face Hub (`config.json`, GGUF KV header, safetensors header, `LICENSE` file, tokenizer config) as **different projections of the same model spec**, and runs them through a single pipeline to extract, cross-validate, and normalize them into a structured `ModelSpec` object with field-level confidence.

In one line: a **consumer-facing "read / extract" system** — it depends on no cooperation from model authors; if the files can be downloaded, they can be analyzed.

## The problem it solves

Model metadata on the HF Hub is scattered, heterogeneous, and untrustworthy:

- `architectures[0]` is a custom class name under `trust_remote_code=True`, so the family can't be inferred from it.
- The HF tag and the model-card front-matter `license:` field often disagree with the actual files.
- config.json has no real schema; its field set is open and ever-growing (family-specific fields like MoE / MLA / RoPE scaling / sliding window).
- A quantized model's bits-per-weight is an average, not a nominal value (Q4_K_M is actually ~4.83 bpw).
- A merged model's (mergekit) parameter count and layer count change, so they can't be assumed equal to the components.

ParaPulse ModelSpec's goal is to collapse this noise into a spec that is **type-safe, queryable, evolvable, and lossless**.

## Core design principles

1. **The unified schema is the foundation** — every extractor fills the same Pydantic v2 `ModelSpec` target shape.
2. **Field-level provenance + confidence** — every field carries its source (`config` / `tensors` / `inferred` / `heuristic` / `fingerprint`) and confidence; conflicts are traceable.
3. **Multi-source fusion, cross-validation** — parameter double-path check, three-layer context recording, MoE/quantization/merge flag cross-checks.
4. **Orthogonal modeling** — quantization, merge, and adapter are three independent optional sub-structures that can coexist, not a mutually exclusive enum.
5. **Structured partial coverage + lossless retention + an auto feedback loop** — canonical fields are a strictly controlled small set, passthrough is the buffer, raw is the insurance.

## Difference vs Stability AI ModelSpec

The name collides, but the direction is opposite:

| Dimension | Stability AI ModelSpec | ParaPulse ModelSpec |
| --- | --- | --- |
| Direction | A **write** standard (training tools write metadata when producing files) | A **read / extract** system (analyzes already-existing files) |
| Scope | Stable Diffusion / image-generation leaning | LLM-centric (MoE / GQA / MLA / RoPE / active params) |
| Source of info | Only the `modelspec.*` keys in the safetensors `__metadata__` | Multi-source fusion + cross-validation |
| Adoption premise | Requires authors to opt in | Requires no author cooperation |
| Depth of info | Presentation-layer metadata (a business card) | Technical spec (a health report) |

> **Worth borrowing**: (1) a field-naming prefix (e.g. `parapulse.*`) to avoid naming collisions; (2) a MUST / SHOULD / CAN three-tier, RFC-style spec. If a safetensors file happens to carry `modelspec.*` fields, they can serve as `confidence: high` auxiliary input (author-declared) — a few lines of code to scan them is enough.

## Doc navigation

- [architecture.md](architecture.md) — system architecture, directory layout, module breakdown
- [schema.md](schema.md) — `ModelSpec` Pydantic v2 schema design
- [schema-review.md](schema-review.md) — schema fitness review for downstream tasks + proposed additions
- [extractors.md](extractors.md) — extractor design and the three-layer extraction strategy
- [pipeline.md](pipeline.md) — pipeline orchestration, merging, and conflict resolution
- [quantization-and-merge.md](quantization-and-merge.md) — orthogonal modeling of quantized and merged models
- [cli.md](cli.md) — command-line interface design
- [analytics.md](analytics.md) — batch extraction, coverage, field-promotion workflow (M4)
- [development.md](development.md) — install, run, test, code map
- [roadmap.md](roadmap.md) — implementation priorities and roadmap

> **Current status**: M1–M4 are all complete — six source types (config / safetensors / GGUF / license / tokenizer / merge) + cross-validation + quantization (GGUF/AWQ/GPTQ) and merge modeling + batch extraction and coverage feedback (`batch` / `coverage`). See [roadmap.md](roadmap.md), [development.md](development.md), [analytics.md](analytics.md).
