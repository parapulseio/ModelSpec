# Consumer Helpers (M5)

> **Status**: implemented. `ModelSpec` convenience accessors (`modelspec/schema/spec.py`), the predicate/filter library (`modelspec/query.py`), and the field catalog (`modelspec/explain.py`).

The schema is deliberately *normalized*: quantization / merge / MoE / adapter are
[orthogonal optional structures](quantization-and-merge.md), and provenance lives
in a side `Provenance` block keyed by dotted path. That is the right shape for
storage and cross-validation, but it pushes work onto every downstream caller —
"is this quantized?" means `spec.quantization is not None`, the context window
falls back through three fields, and "where did this value come from?" is a dict
lookup. M5 wraps those patterns into cheap, side-effect-free helpers so consumers
don't re-implement them (and don't drift on the edge cases).

## Per-spec accessors (methods / properties on `ModelSpec`)

All are pure projections of already-validated state — they never raise, never mutate.

| Accessor | Returns | Notes |
| --- | --- | --- |
| `spec.is_quantized()` | `bool` | `quantization` present |
| `spec.is_merged()` | `bool` | `merge` present |
| `spec.is_moe()` | `bool` | `moe` present |
| `spec.is_adapter()` | `bool` | `adapter` present (reserved) |
| `spec.is_derived()` | `bool` | has `identity.lineage.base_models` |
| `spec.quant_format` | `str \| None` | the union discriminator: `gguf` / `awq` / `gptq` |
| `spec.bits_per_weight` | `float \| None` | GGUF measured avg, else AWQ/GPTQ nominal bits |
| `spec.effective_context` | `int \| None` | `effective` → `declared` → `trained` fallback |
| `spec.source_of(path)` | `SourceLabel \| None` | winning source for a dotted field path |
| `spec.confidence_of(path)` | `Confidence \| None` | confidence of the winning value |
| `spec.is_not_applicable(path)` | `bool` | legitimately N/A vs. merely missing |
| `spec.conflicts_for(path)` | `list[Conflict]` | archived losing claims for a path |

```python
from modelspec.pipeline import extract

spec = extract("TheBloke/Mistral-7B-v0.1-GGUF")
if spec.is_quantized():
    print(spec.quant_format, spec.bits_per_weight)     # gguf 4.83
print(spec.effective_context)                          # 32768
print(spec.source_of("architecture.family"))           # config
```

## Predicate / filter library (`modelspec.query`)

For *collections* of specs. Predicates are plain `Callable[[ModelSpec], bool]`,
so anything with that shape composes — no base class required.

- Bare predicates: `is_quantized`, `is_merged`, `is_moe`, `is_dense`, `is_adapter`, `is_derived`, `commercial_use_allowed`.
- Factories (call to bind): `family_is(*names)`, `quant_format_in(*formats)`, `min_params(n)`, `max_params(n)`, `min_context(n)`, `license_is(*spdx_ids)`. Numeric predicates exclude unknown values.
- Combinators: `all_of(*p)`, `any_of(*p)`, `negate(p)`.
- Driver: `filter_specs(specs, *predicates)` — yields specs satisfying *all* predicates (implicit AND).

```python
from modelspec.query import filter_specs, all_of, is_quantized, min_params, family_is

big_llama_quants = list(
    filter_specs(specs, all_of(is_quantized, min_params(7e9), family_is("llama")))
)
```

## Field catalog (`modelspec.explain`)

`field_catalog()` flattens the live schema (descriptions, types, Literal choices)
into dotted-path `FieldDoc` entries; `explain_field(query)` does the fuzzy lookup
used by `modelspec explain`. There is no second copy of the docs — it introspects
the `description=` already on every field. See [cli.md](cli.md#explain--field-documentation-m5).
