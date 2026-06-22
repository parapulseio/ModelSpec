# System Architecture

## Layered view

The whole system is a pipeline **from files on disk to in-memory Python objects**. Pydantic appears only at the last two steps (entry validation, exit serialization); all the dirty work in between (download, binary parsing, feature inference, conflict merging) is plain Python, decoupled from Pydantic.

```
       Files on disk / HF Hub                  your extractor code            in-memory Python object
┌────────────────────────┐                                          ┌──────────────────────┐
│ config.json            │  ──read JSON──►  parse + field mapping ──►│                      │
│ *.safetensors (header) │  ──read bytes─►  parse + field mapping ──►│  ModelSpec(Pydantic) │
│ *.gguf (KV header)     │  ──read bytes─►  parse + field mapping ──►│                      │
│ LICENSE                │  ──read text──►  fingerprint / keyword ──►│                      │
└────────────────────────┘                                          └──────────────────────┘
        ↑                                                                    ↑
   external formats, not ours                                 the schema we define with Pydantic
```

Benefits of this layering:

- **Extractors are testable in isolation** — feed one a fixture file and check the returned list of claim tuples; no need to mock Pydantic.
- **The Pydantic model is testable in isolation** — feed it a dict and check whether it validates; no need for real files.
- **Replaceable** — to swap Pydantic for attrs + cattrs or a dataclass later, only the last two steps change.

## End-to-end flow (example: `meta-llama/Llama-3.1-8B-Instruct`)

| Step | Action | Pydantic involved? |
| --- | --- | --- |
| 0 | CLI parsing, obtain `repo_id` | No |
| 1 | Probe the remote repo: `list_repo_files()` lists files, classify the repo type (HF standard / GGUF / adapter) | No |
| 2 | Precise metadata download: config.json, index.json, safetensors header (HTTP Range request for the first few hundred KB only), LICENSE, tokenizer_config.json. **No weights** — an 8B model is 16GB, but we actually download a few MB | No |
| 3 | Several extractors parse in parallel, each emitting a list of `(field_path, value, source, confidence)` tuples | No |
| 4 | Merge + conflict resolution: group by field path, higher confidence wins, the loser goes to `provenance.conflicts` | No |
| 5 | Reshape into the nested dict `ModelSpec` expects | No |
| 6 | **`ModelSpec.model_validate(dict)`** — type checking, required-field checks, nested recursive construction, enum/discriminator validation, cross-field validators | **Yes** |
| 7 | Output: `model_dump_json()` to disk / `model_dump()` for downstream / `model_json_schema()` to export docs | **Yes** |

## Directory layout (option B: a single package, multiple sub-modules)

We start with single-package publishing and decouple internally into sub-modules. The directory boundaries are drawn along the likely future package-split lines, keeping a low-cost path toward a multi-package layout.

```
modelspec/
├── __init__.py
├── schema/                 ← Pydantic models (the system's spec definition)
│   ├── __init__.py
│   └── spec.py
├── extractors/             ← one file per source
│   ├── __init__.py         ← the ALL_EXTRACTORS registry
│   ├── base.py             ← Extractor protocol + FieldClaim definition
│   ├── config_json.py
│   ├── safetensors.py
│   ├── gguf.py
│   ├── license.py
│   └── tokenizer.py
├── pipeline/               ← orchestration, merging, conflict resolution, cross-validation
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── merger.py
│   └── cross_validate.py
├── io/                     ← downloads, HTTP Range requests
│   ├── __init__.py
│   └── hf_fetcher.py
└── cli.py

tests/
├── extractors/
│   ├── test_config_json.py     ← feed a fixture config.json, check claims
│   ├── test_safetensors.py
│   └── test_gguf.py
├── pipeline/
│   └── test_merger.py          ← feed conflicting claims, check the merge
└── schema/
    └── test_spec.py            ← feed a dict, check validation
```

### Why option B instead of multiple packages

- **Dependencies are light**: the gguf package is a few MB of pure Python, and the safetensors / config readers have zero dependencies. There is no PyTorch-grade heavy dependency to justify splitting packages.
- **Rapid-evolution phase**: adding a field means changing the schema and extractors together; a single package avoids coordinating multiple version numbers.
- **Splitting out the schema is an anti-pattern**: schema and extractors always evolve together, so a forced split turns every schema change into a cross-package double release. To share with outsiders, just export and distribute a JSON Schema file.

### When to upgrade to multiple packages (option C)

Consider splitting once any of these holds:

- An extractor introduces a heavy dependency (PyTorch / CUDA) that all users would pay for.
- External users appear and explicitly ask for "just one part".
- Different extractors diverge in release cadence (e.g. the GGUF reader tracks llama.cpp upstream with high-frequency updates).
- The schema needs to be shared across multiple independent projects under a strict version contract.

## Optional-dependencies compromise

Within a single package, use `pyproject.toml` optional dependencies + conditional imports to get 95% of the "install on demand" benefit:

```toml
[project]
dependencies = ["pydantic>=2", "huggingface_hub", "requests"]

[project.optional-dependencies]
gguf = ["gguf>=0.10"]
all = ["modelspec[gguf]"]
```

```python
# top of extractors/gguf.py
try:
    from gguf import GGUFReader
    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False
```

## Extractor registration & protocol

The `orchestrator` only deals with the `Extractor` protocol and knows nothing about concrete implementations. To add a source, write a new file implementing the protocol and register it — the orchestrator is untouched. See [extractors.md](extractors.md) and [pipeline.md](pipeline.md).
