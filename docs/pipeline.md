# Pipeline Orchestration

> **Status**: orchestration / merging / reshape (M1) and `cross_validate` (M2, §7) are implemented. The spec's only artifact is a JSON document; there is no database storage / query layer.

## Overall orchestration

```
detect_source(path)
   ├─ HF dir       → [config, safetensors, tokenizer, license] extractors
   ├─ GGUF file    → [gguf, license(same dir)] extractors
   └─ raw weights  → [safetensors only], most fields = unknown

merge(outputs)             # later claims override by confidence
cross_validate(spec)       # parameter double-path, context three-layer, MoE flag checks
emit(spec, format=json|yaml)
```

## 1. Source detection `detect_source`

Call `huggingface_hub.list_repo_files()` to list the repo's files, classify by file signature, and dispatch the matching extractors:

| File signature | Repo type | Extractors dispatched |
| --- | --- | --- |
| `config.json` + `*.safetensors` | HF standard | config, safetensors, tokenizer, license |
| `*.gguf` only | GGUF quantized | gguf, license |
| `adapter_config.json` | adapter | config(adapter), safetensors, license |
| bare weights only | raw | safetensors only |

## 2. Precise download `io/hf_fetcher`

**No weights, metadata only.** For safetensors, use an HTTP Range request for the header only (the first 8 bytes give the JSON header length, usually a few hundred KB + buffer). An 8B model's weights are 16GB but the actual download is ~a few MB.

## 3. Parallel parsing

Each extractor runs independently and returns a flat list of `FieldClaim(field_path, value, source, confidence)`. Flat rather than nested keeps merging simple.

## 4. Merge + conflict resolution `pipeline/merger`

Merge all extractor outputs into one big list and group by `field_path`:

- Multiple values for one field → sort by confidence, **highest wins**.
- The losing value goes to `provenance.conflicts` for human review.

> Example: GGUF and config both report `context.declared` with different values (the GGUF converter changed it); take the higher confidence, record the other in conflicts.

Produces a flat dict: `{field_path: {value, source, confidence}}`.

## 5. Reshape

Reorganize the flat dict into the nested structure `ModelSpec` expects (a plain Python dict, still no Pydantic).

## 6. Pydantic validation

`ModelSpec.model_validate(dict)` does five things: type checking, required-field checks, nested recursive construction, enum/discriminator validation, and custom cross-field validators. On success you get a type-safe object.

## 7. Cross-validation `pipeline/cross_validate`

Runs multi-source checks before/after validation, writing results to `provenance.warnings`:

### Parameter double-path check

- **Path A (authoritative)**: summed safetensors / GGUF tensor element counts.
- **Path B (sanity check)**: estimate from the config formula `embedding + layers × (attn + ffn) + lm_head`.
- A >1% gap triggers a warning, usually a missing bias / norm / shared experts.

### Context three-layer check

`trained` ≤ `declared`; consistency check on `effective = declared × rope_scaling.factor`.

### MoE flag check

The config's `num_experts` should agree with the tensor-name pattern (`experts.{j}.`).

### Architecture consistency (merge)

All merge components should share the same architecture; a mismatch triggers a warning (frankenmerge / passthrough excepted, where layer count / parameter count are recomputed from tensors).

## 8. Output

`emit(spec, format=json|yaml)` serializes the normalized `ModelSpec` (`model_dump_json()` to disk, or handed to a downstream consumer). A strongly-typed, multi-tag spec lets downstream filtering, recommendation, and search rely on it with confidence.

> A storage / query layer (e.g. a database) is out of current scope — the spec's only artifact is a JSON document, and downstream decides how to land it.
