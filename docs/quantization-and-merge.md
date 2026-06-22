# Modeling Quantized and Merged Models

> **Status (landed in M3)**: the GGUF / AWQ / GPTQ branches of the `Quantization` discriminated union and the five-signal `MergeSpec` detection are implemented. Quantization claims are emitted by `gguf.py` (GGUF) + `config_json.py` (AWQ/GPTQ); merge is emitted by `merge.py`. The BnB / FP8 / MLX-native quantization branches below are follow-up increments; schema code is in `modelspec/schema/spec.py`.

## Design premise: orthogonal dimensions, not special cases

A model can be quantized, merged, and carry a LoRA at once (e.g. a Llama-3-70B + Qwen2-72B SLERP merge → GGUF Q4_K_M quantization → further LoRA training). This "stacking" is common on the HF Hub.

So `quantization` / `merge` / `adapter` are three **independent, optional, mutually non-exclusive, mutually non-referencing** sub-structures. **The schema's orthogonality is a core design principle.**

```
ModelSpec
  ├── identity / architecture / parameters / context / license   (required)
  ├── quantization?    (optional)
  ├── merge?           (optional)
  ├── adapter?         (optional)
  └── provenance       (required)
```

> **The base_model chain lives in `identity.lineage`**, not in each sub-structure — quantization, merge, and adapter all have a base_model, so it should be a unified "source relationship graph".

## Quantization modeling

### Signal sources

| Format | Signal |
| --- | --- |
| GGUF | the `general.file_type` enum (`MOSTLY_Q4_K_M=15` …); each tensor has its own `ggml_type` |
| AWQ / GPTQ | config.json's `quantization_config`: `bits` / `group_size` / `desc_act` / `quant_method` |
| BitsAndBytes | `quant_method: "bitsandbytes"`, `load_in_4bit` / `bnb_4bit_quant_type: "nf4"` |
| FP8 | `quant_method: "fp8"` or `fbgemm_fp8` |
| MLX-native | `mlx.quantization` in the safetensors `__metadata__` |
| exl2 | a dedicated `quantization_config.json`: `bits_per_weight` / `head_bits` |
| tensor-name inference | `*.qweight`/`*.qzeros`/`*.scales`→GPTQ; `*.absmax`/`*.code`→BnB; dtype `F8_E4M3`→FP8 |

### Discriminated-union modeling

```python
class GGUFQuant(BaseModel):
    format: Literal["gguf"]
    file_type: str               # "Q4_K_M" / "IQ3_XS"
    bits_per_weight_avg: float   # measured, not the nominal value
    tensor_types: dict[str, int] # {"Q4_K": 200, "Q6_K": 32, "F32": 5}
    has_imatrix: bool | None

class AWQQuant(BaseModel):
    format: Literal["awq"]
    bits: int
    group_size: int
    zero_point: bool

class GPTQQuant(BaseModel):
    format: Literal["gptq"]
    bits: int
    group_size: int
    desc_act: bool

class BnBQuant(BaseModel):
    format: Literal["bitsandbytes"]
    bits: int                    # 4 or 8
    quant_type: str | None       # "nf4" / "fp4"
    double_quant: bool

class FP8Quant(BaseModel):
    format: Literal["fp8"]
    variant: str                 # "e4m3" / "e5m2"
    scheme: str | None

Quantization = Annotated[
    Union[GGUFQuant, AWQQuant, GPTQQuant, BnBQuant, FP8Quant],
    Field(discriminator="format"),
]
```

Downstream: `if spec.quantization and spec.quantization.format == "gguf": ...`, type-safe.

### Quantization pitfalls

- **bits-per-weight is an average**: Q4_K_M is actually ~4.83 bpw, not 4.0. Sum it from the tensor list, not the nominal value. This matters for ParaPulse showing model sizes.
- **imatrix quantization** (the IQ / I series) is higher quality but hard to tell apart at the file level. Heuristics: `general.quantization_version` / a `-imat-` filename / an imatrix field in the GGUF KV — none are fully reliable, so `has_imatrix` allows `None`.
- **Mixed precision** (some Q4, some Q6) is common, so the `tensor_types` dict is necessary; you can't store only a single global bits.

## Merge modeling

### Signal sources (high → low confidence)

1. A `mergekit_config.yml` file exists — high confidence, contains the full recipe.
2. `cardData.base_model_relation: merge` — an explicit HF annotation.
3. `cardData.base_model` is a multi-element array — a strong signal.
4. The HF tag `other=mergekit` — broadest coverage, but only tells you "it's a merge", not the method.
5. A README YAML codeblock containing `merge_method` — medium confidence, needs parsing.

### Modeling

```python
class MergeComponent(BaseModel):
    model_id: str
    weight: float | None
    density: float | None        # DARE / TIES parameter
    role: Literal["base", "ingredient", "donor"] | None

class MergeSpec(BaseModel):
    detection_signal: Literal[
        "hf_tag", "config_file", "card_relation", "base_model_array", "readme_yaml"
    ]
    confidence: Literal["high", "medium", "low"]
    method: str | None           # "slerp" / "dare_ties" / "linear" / "task_arithmetic"
    components: list[MergeComponent]
    base_architecture: str | None
    tokenizer_source: str | None
    mergekit_version: str | None
    has_config_yml: bool
    raw_recipe: dict | None
```

### Merge pitfalls

- **Architecture consistency constraint**: all components should share the same architecture. A mismatch → detection is wrong, or it's a frankenmerge / passthrough. `base_architecture` is a single field; a mismatch goes to `provenance.warnings`.
- **frankenmerge / passthrough** change layer count and parameter count (two 13Bs interleaved into a 20B). `parameters.total` / `num_layers` are **recomputed from the tensor layer**, not assumed equal to the components.
- **Recursive merges** (a merge of merges): the schema records only the direct components; the whole merge tree is the job of a separate lineage-graph system.
- **Method alias normalization**: `dare_ties` / `dare-ties` / `DARE_TIES` must be normalized; build an alias table.
- **Missing weight info**: a merge detected from an HF tag / card usually has no weights, so `components` carries only `model_id`, the rest `None`, with low confidence.
- **A LoRA merge is a different thing**: when LoRA adapters (not base models) are merged, the model goes into the `adapter` sub-structure — don't conflate them.

## Stacking example

`someone/Llama3-MyMerge-AWQ-4bit` (merged, then AWQ-quantized):

```python
ModelSpec(
    identity=Identity(repo_id="someone/Llama3-MyMerge-AWQ-4bit", ...),
    architecture=Architecture(family="llama", num_layers=32, ...),
    parameters=Parameters(total=8e9, dtype_native="INT4", ...),
    quantization=AWQQuant(format="awq", bits=4, group_size=128),
    merge=MergeSpec(detection_signal="hf_tag", method=None, components=[...], confidence="medium"),
    adapter=None,
)
```

Filtering "all AWQ-quantized merged models" is just a predicate: `spec.quantization.format == "awq" and spec.merge is not None`.

## Coverage expectations

- **Quantization** canonical fill rate: GGUF / AWQ / GPTQ / BnB / FP8 together cover 95%+ of HF quantized models; rare formats fall through to raw passthrough.
- **Merge** detection: the five-tier strategy can go from ~132 up to nearly the full 36K set; but recipe details (method / weights) can only be fully filled for the portion that has `mergekit_config.yml` (~10–20%), while the rest gets only "it's a merge" + a components list. This gap is a data-source limitation, not something the schema can solve.
