# Plan: Quantization Quality Assessment

- **Date**: 2026-10-07
- **Branch**: `feature/assessing-ai-model-quantization-quality-3vxf0`
- **Status**: Phase A done (A1 #18, A4 #19, G #20, A2 #21, A3 #22, A5 #23); Phase B/C moved to [parapulseio/QuantAssessment](https://github.com/parapulseio/QuantAssessment) (see D2)
- **Input**: `quantization_weight_comparison.md` + `quant_weight_compare.py` (external notes on comparing an original model against GGUF / AWQ / GPTQ quantized variants with real weights, PPL, KL divergence, and imatrix)

## 1. Background

The input notes describe three increasingly expensive levels of quantization quality assessment:

| Level | What | Needs |
|---|---|---|
| **A. Static metadata** | GGUF `quantize.imatrix.*` KV, per-tensor quant types, which AWQ/GPTQ Linear layers are actually quantized | headers only |
| **B. Weight reconstruction error** | dequantize → compare against FP16 element-wise (step plot, SQNR / relative error / cosine / max abs error), optionally imatrix-weighted | partial real weight bytes + numpy |
| **C. Output quality** | PPL and KL divergence (`llama-perplexity` or transformers) | full weights, GPU, actual inference |

The notes' own conclusion (§3.7) matters for how we position this: weight error can mislead (AWQ/GPTQ/imatrix optimize *output* error, not weight error). Final quality verdicts come from KLD; weight plots are for explaining and locating problems.

## 2. Feasibility verdict

| Level | Verdict | Reason |
|---|---|---|
| A | **Do now** | Fits the read/extract, header-only design exactly; no new deps; upgrades existing fields' confidence. |
| B | **Prototype as a separate, opt-in module** | Technically feasible, but violates "never download weights" if put in the extract path, adds numpy, and its output is not an intrinsic model property. |
| C | **Ingest results only, never compute** | Requires GPU + full weights + tens of GB of logits; fundamentally outside an extraction system. |

## 3. Phase A — static quantization metadata (extractor work)

All changes stay inside extractors and follow the three-layer rule (new fields go to passthrough first; promote via `unknown_fields` frequency).

### A1. Deterministic `has_imatrix` for GGUF

- **Today**: `modelspec/extractors/gguf.py:372-375` only checks for `imat` in the filename (`heuristic`, `low`). The KV fields are already parsed into `fields` / `raw` but unused.
- **Change**:
  - If any of `quantize.imatrix.file`, `quantize.imatrix.dataset`, `quantize.imatrix.entries_count`, `quantize.imatrix.chunks_count` is present → `FieldClaim("quantization.has_imatrix", True, "gguf", "high")`.
  - Keep the filename heuristic as a `low` fallback only when the KV is absent.
  - Absence of the KV must **not** produce `False` (older llama.cpp builds did not write these keys) — leave `None`.
  - Add the `quantize.imatrix.*` keys to the GGUF `passthrough` dict (the dataset name is the most useful bit for consumers).
- **Tests** (`tests/extractors/test_quantization.py` or `test_gguf.py`): fixture GGUF header with / without the KV; filename-only case still yields `low`; neither yields no claim.

### A2. Mixed-precision layout for GGUF (passthrough)

- **Change**: from the already-parsed tensor list, build a per-role breakdown, e.g. `{"ffn_down": {"Q6_K": [0, 1, 31], "Q4_K": [2, ..., 30]}, ...}` (role = tensor name with `blk.{i}.` stripped). Emit to passthrough only — not canonical.
- **Why**: explains why two `Q4_K_M` files from different quantizers differ, and why `Q4_K_M` bpw > 4.
- **Tests**: fixture with mixed `ffn_down` types across layers.

### A3. Measured bpw for AWQ / GPTQ (canonical)

- **Decision (D1, 2026-10-07)**: the measured bpw becomes a **canonical** schema field on `AWQQuant` / `GPTQQuant`, mirroring the existing `GGUFQuant.bits_per_weight_avg`. Rationale: not a new concept but closing a unit mismatch across the union branches; the downstream consumer already exists (`spec.bits_per_weight`, ParaPulse model-size display).
- **Today**: `ModelSpec.bits_per_weight` (`modelspec/schema/spec.py:393-402`) returns nominal `bits` (from `config.json` `quantization_config.bits`) for AWQ/GPTQ, while GGUF returns the measured value — a nominal 4-bit AWQ and a GGUF Q4_K_M are not comparable.
- **Computation (header only, no GPU, no weight bytes)**: `Σ stored bits of all tensors ÷ logical weight count`.
  - Numerator: every tensor's `dtype` bits × element count (equivalently its `data_offsets` byte span × 8), **including** `qzeros` / `scales` / `g_idx`.
  - Denominator: unquantized tensors count their elements; a quantized Linear counts its unpacked `in × out` (helper from A4); `qzeros` / `scales` / `g_idx` count zero.
  - **Scope: whole model** (decided, D1a; same as GGUF), so FP16 embeddings / `lm_head` are included. Rough illustration: a "4-bit" AWQ Llama-3-8B is ~5.7 bpw whole-model vs ~4.16 on quantized layers alone, while GGUF Q4_K_M is ~4.83.
  - Sharded checkpoints: aggregate across all shards (same rule as parameter counting).
- **Schema**: add `bits_per_weight_avg: Optional[float]` to `AWQQuant` and `GPTQQuant` with the same description as the GGUF field. `bits_per_weight` accessor: prefer measured `bits_per_weight_avg` on every branch, fall back to nominal `bits`. Update `docs/schema.md`, `docs/quantization-and-merge.md`, and the exported JSON Schema snapshot tests if any.
- **Depends on G**: the claim comes from the `safetensors` extractor while `quantization.format` comes from `config_json`. The merger groups by field path only and the orchestrator nests every claim before `model_validate`, so a `quantization.bits_per_weight_avg` with no `format` claim (unknown `quant_method`, or no `quantization_config`) yields `{"quantization": {...}}` without a discriminator and fails validation. G must land first.
- **GPTQ edge case**: without `g_idx`, `in` cannot be derived from the header alone (pack factor = 32/bits, `bits` lives in `config.json`). Either emit the claim only when the shape is derivable, or compute it at pipeline merge time; decide during implementation.
- **Related small fix**: `_avg_bits_per_weight` in `gguf.py:160-162` skips unknown ggml type ids in the numerator but keeps their elements in the denominator, under-reporting bpw. Return `None` (or drop them from the denominator) instead.

### G. Guard: `quantization.*` claims without a `format` (prerequisite for A3)

- **Where**: pipeline (merger or orchestrator, before `_set_nested` / `model_validate`). This is a pipeline change, not an extractor change.
- **Change**: if the merged fields contain `quantization.*` paths but no `quantization.format`, do not build the `quantization` subtree; move those values to passthrough (and/or emit a provenance warning) so the union never sees a missing discriminator.
- **Tests**: claims with a sub-field but no `format` → `spec.quantization is None`, extraction succeeds, value preserved in passthrough.

### A4. Fix: parameter count for packed AWQ / GPTQ checkpoints (existing bug)

- **Observed by reading code (not yet verified on a real repo)**: `modelspec/extractors/safetensors.py:75-82` sums raw element counts across all tensors. For AWQ/GPTQ:
  - each `I32` element of `qweight` packs `32 / bits` weights, but counts as 1;
  - `qzeros`, `scales`, `g_idx` are counted as parameters.
  
  → `parameters.total` for quantized repos is substantially under-reported, and `dtype_native` may come out as `I32` or `F16` for the wrong reason.
- **Change**: when a prefix has `qweight`, count `in_features × out_features` derived from the unpacked shape (AWQ: `[in, out/pack]`; GPTQ: `[in/pack, out]`), and exclude `qzeros` / `scales` / `g_idx` from the total. `bits` comes from `config.json` — since extractors are independent, derive `pack = 32 / bits` from the shape ratio `scales.out / qweight.out` (AWQ) or `scales.in_groups × group_size / qweight.rows` (GPTQ) rather than depending on another extractor's output.
- **First step**: write a failing test with a synthetic AWQ header (`write_safetensors_header`) before fixing.
- Re-check `pipeline/cross_validate` parameter double-path behavior on quantized repos once fixed.

### A5. Exposed list of quantized vs unquantized modules (passthrough)

- For AWQ/GPTQ: list of prefixes that have `qweight`, and the notable non-quantized ones (`lm_head`, embeddings). Passthrough only.

### Phase A acceptance

- `pytest -q` green, new tests for A1–A5.
- `modelspec extract <gguf-dir> --offline` shows `has_imatrix` with `gguf`/`high` provenance on an imatrix-quantized fixture.
- `docs/schema.md`, `docs/quantization-and-merge.md` (imatrix pitfall paragraph) updated; `docs/extractors.md` mentions the packed-weight counting rule.
- Add a key constraint line to `AGENTS.md`: "AWQ/GPTQ `qweight` is packed — count unpacked weights; `qzeros`/`scales`/`g_idx` are not parameters."

## 4. Phase B — weight reconstruction error (opt-in module, prototype first)

> **Moved (D2, 2026-10-07)**: Phase B and Phase C are developed in a separate repo, [parapulseio/QuantAssessment](https://github.com/parapulseio/QuantAssessment), not as a `modelspec/quality/` sub-package. The authoritative plan is [`docs/plans/2026-10-07-phase-b-c-development-plan.md`](https://github.com/parapulseio/QuantAssessment/blob/main/docs/plans/2026-10-07-phase-b-c-development-plan.md) in that repo. The text below is the original design, kept for history; where it says `modelspec/quality/`, `modelspec assess-quant` or `modelspec ingest-eval`, read QuantAssessment's `quantassess` package and CLI instead.

### Positioning

- New sub-package `modelspec/quality/` + CLI subcommand (working name `modelspec assess-quant`). **Never** invoked by `extract`.
- Output is a separate Pydantic report (`QuantAssessment`), not a `ModelSpec` field: the result depends on the reference model, sampled tensors, and metric choice, so it is an evaluation, not a spec property. The report references the spec (`repo_id`, resolved commit SHA) and records method + parameters.
- New optional extra `quality = ["numpy>=1.24", "gguf>=0.10"]`, conditional import like `gguf` / `PyYAML` today. No matplotlib in the library; plotting stays an example script or a `--plot` flag gated on an extra.

### Building blocks

| Block | Approach | Reuse |
|---|---|---|
| Reference resolution | base model from `identity.lineage`; overridable via `--base` | existing lineage extraction |
| Single-tensor fetch (safetensors) | 3 Range requests: length → header → `data_offsets` slice; shard lookup via `model.safetensors.index.json` | `io/hf_fetcher._read_prefix`, header logic |
| Single-tensor fetch (GGUF) | keep the data offset in `parse_gguf_header` (currently discarded at `gguf.py:129`), align with `general.alignment` (default 32), compute byte size from `GGML_QUANT_SIZES`, Range-fetch the slice | own header parser — still no `GGUFReader` |
| GGUF dequant | `gguf.quants.dequantize(ndarray, qtype)` on the fetched bytes | `gguf` data tables/functions only |
| AWQ dequant (GEMM) | unpack with reverse order `[0,4,1,5,2,6,3,7]`, `w = (q - z) * s`, transpose to `[out, in]` | port from input script |
| GPTQ dequant | sequential unpack, `w = (q - (z + 1)) * s`, transpose; **support `g_idx`** (`desc_act=True`) via `scale[g_idx[row]]` | port + extend |
| Name mapping HF ↔ GGUF | per-arch via gguf-py `TensorNameMap`; fall back to "not comparable" rather than guess | — |
| Llama `attn_q`/`attn_k` row permutation | either inverse-permute (needs `n_head`, `n_kv_head` from spec) or restrict to distribution-level (independently sorted / Q-Q) comparison | spec `attention.*` |
| Metrics | rel. error, SQNR dB, cosine, max abs error; per tensor, aggregated per layer/role | — |
| Sampling | default: `mlp.down_proj` + `self_attn.v_proj` at first / middle / last layer; `--all-layers` opt-in with a printed byte estimate before downloading | — |

### Format-variant guardrails (correctness risk, not covered by the input notes)

- GPTQModel `checkpoint_format: gptq_v2` does **not** store `zero - 1` → no `+1` on dequant. Detect from `quantize_config.json` / `quantization_config`.
- AutoAWQ GEMV / Marlin / ExLlama-repacked checkpoints use different layouts → refuse with a clear "unsupported packing" error instead of producing wrong numbers.
- Every dequant path requires a round-trip unit test: synthetic float matrix → pack in the target format → load through our code → compare to expected dequantized matrix. Shape assertion before any metric (the input script once silently compared transposed data).

### imatrix-weighted error (lowest priority within B)

- Only when the user supplies an imatrix file (`--imatrix path`); `importance = in_sum2 / counts` per input column.
- Report warns when the imatrix is the same one used to produce the quant being assessed (biased in its favor).

### Phase B acceptance (prototype)

- One real end-to-end run on a small model (≤ 1B) with GGUF Q4_K_M + one AWQ + one GPTQ variant, transferring < 200 MB total in default sampling mode.
- Round-trip tests for every supported packing; `pytest -q` still runs without the `quality` extra (tests skip cleanly).
- Decision gate after the prototype: keep / extend / drop, based on whether the numbers prove useful to ParaPulse consumers.

## 5. Phase C — PPL / KLD (ingest only)

> **Moved (D2)**: see the note at the top of section 4.

- Do **not** run models. Provide `modelspec ingest-eval <file>` that parses:
  - `llama-perplexity` stdout (both plain PPL and `--kl-divergence` output: Mean PPL(Q)/(base), Mean ln ratio, Mean / Median / 99% / 99.9% / Max KLD, Mean Δp, RMS Δp, Same top p);
  - a documented JSON format for HF-side evaluations (AWQ/GPTQ via transformers).
- Store as an evaluation record (alongside the Phase B report, not inside canonical `ModelSpec`). Comparability metadata is **required**, otherwise records are rejected: tool + version, dataset (+ hash if available), context length, chunks, scoring scheme (llama.cpp second-half scoring vs all positions), reference model id/revision.
- Comparison helpers only compare records whose comparability metadata match.
- Optional later: low-confidence parsing of PPL/KLD tables in model card READMEs into passthrough.

## 6. Ordering

### Phase A dependencies

```
  A1 ---> A2            (GGUF)

  A4 ---+---> A5        (safetensors)
        |
        +---> A3 <--- G
              ^
              |
              D1  (decided: canonical, whole-model)
```

- Start in parallel: A1, A4, G.
- A2 waits on A1 (same file, same passthrough dict). A5 waits on A4. A3 waits on A4 + G.
- Critical path: A4 → A3.

### Staffing (3 developers)

```
        step 1            step 2
dev 1:  A1  ------------> A2
dev 2:  A4  ------------> A3   (also waits on G)
dev 3:  G   ------------> A5   (waits on A4)
```

Each step-2 task starts as soon as its own prerequisites land; there is no global barrier. Docs and `AGENTS.md` updates ship with each task.

### Later phases

Phase B and Phase C are scheduled in [parapulseio/QuantAssessment](https://github.com/parapulseio/QuantAssessment) (see D2). ModelSpec only owes the upstream prerequisites in section 9.

## 7. Decisions

- **D1 (2026-10-07)**: AWQ/GPTQ measured bpw is a canonical field (`bits_per_weight_avg`). See A3.
- **D1a (2026-10-07)**: bpw scope is **whole model** (all tensors, incl. unquantized FP16 embeddings / `lm_head`), consistent with GGUF. Scope does not affect extraction time: both scopes read the same already-fetched headers.

- **D2 (2026-10-07)**: Phase B/C reports (`QuantAssessment`, `EvalRecord`) and the code that produces them live in [parapulseio/QuantAssessment](https://github.com/parapulseio/QuantAssessment), not in ModelSpec. Report models, storage (stdout / `-o`, sidecar in a `--download-only` directory, ParaPulse for persistence) and methodology are defined in that repo's README. ModelSpec stays metadata-only; the static quantization metadata from Phase A stays here because it is a property of the file.

## 8. Open questions

- ~~Where should Phase B/C reports live?~~ Resolved by D2.
- ~~Per-arch HF↔GGUF name mapping, or Llama-family only?~~ Moved to QuantAssessment; tracked in its Phase B/C plan.

## 9. ModelSpec-side prerequisites for QuantAssessment

QuantAssessment depends on ModelSpec (git-tag dependency; ModelSpec is not on PyPI) for extraction, lineage-based reference discovery, architecture info and GGUF header parsing. Two small changes are needed here:

- **U1. GGUF tensor data offsets.** `parse_gguf_header` (`modelspec/extractors/gguf.py`) reads each tensor's data offset and discards it. Add a public, backward-compatible way to get, per tensor, `(name, dims, ggml_type, offset)` plus the start of the data section (end of tensor infos, aligned to `general.alignment`, default 32). Do not change the existing return shape that the extractor and tests rely on. Still no `GGUFReader`.
- **U2. Release.** Tag a release (e.g. `v0.2.0`) that contains Phase A and U1. The latest tag, `v0.1.1`, predates Phase A, so QuantAssessment cannot pin anything that has `bits_per_weight_avg` on AWQ/GPTQ or the quantized-module lists.
