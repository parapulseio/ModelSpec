# ParaPulse ModelSpec

**Extract and normalize LLM model specifications from heterogeneous sources into one unified, type-safe schema.**

ParaPulse ModelSpec treats the various metadata sources a model ships with —
`config.json`, GGUF KV headers, safetensors headers, `LICENSE` files, tokenizer
configs — as **different projections of the same model spec**. A single pipeline
reads them, cross-validates, and normalizes everything into one Pydantic v2
`ModelSpec` with per-field provenance and confidence.

It is a **read/extract system for consumers**: it does not require model authors
to cooperate — if the files can be downloaded, they can be analyzed.

!!! note "Not Stability AI's ModelSpec"
    Not to be confused with [Stability AI's ModelSpec](https://github.com/Stability-AI/ModelSpec),
    which is a *write* standard for image-generation training tools. This project
    goes the opposite direction: a multi-source *reader* for LLMs. See the
    [Overview](overview.md).

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

## Where to start

| If you want to… | Read |
| --- | --- |
| Understand the project and how it differs from prior art | [Overview](overview.md) |
| See the end-to-end flow and layout | [Architecture](architecture.md) |
| Learn the unified data model | [Schema Design](schema.md) · [Schema Review](schema-review.md) |
| Understand the source readers | [Extractors](extractors.md) |
| See how sources are merged & cross-validated | [Pipeline](pipeline.md) |
| Dig into tricky model shapes | [Quantization & Merge](quantization-and-merge.md) |
| Use the tool | [CLI](cli.md) · [Consumer Helpers](helpers.md) |
| Run batch extraction & coverage | [Analytics](analytics.md) |
| Hack on the code | [Development](development.md) · [Roadmap](roadmap.md) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Extract a local model directory
modelspec extract ./path/to/model --offline

# Explain a schema field
modelspec explain context_length
```

See the [Development Guide](development.md) for the full setup and test workflow.
