"""safetensors extractor — reads headers only, never loads weights.

The safetensors header is a length-prefixed JSON object: the first 8 bytes are
a little-endian uint64 giving the JSON byte length, followed by the JSON. We
read just that. For sharded models we aggregate every shard header (via the
index) — otherwise the parameter count comes out half. See docs/extractors.md.
"""

from __future__ import annotations

import json
import struct
from math import prod
from pathlib import Path

from modelspec.extractors.base import ExtractionSource, ExtractorResult, FieldClaim

_INDEX = "model.safetensors.index.json"

# Tensor-name patterns used as the last-resort architecture fallback.
_TENSOR_PATTERNS = [
    ("block_sparse_moe.experts.", "moe", "high"),  # Mixtral style
    ("mlp.experts.", "moe", "high"),  # Qwen MoE style
    ("kv_a_proj_with_mqa", "mla", "high"),  # DeepSeek MLA
    ("lora_A", "lora-adapter", "high"),
    ("lora_B", "lora-adapter", "high"),
]


def read_header(path: Path) -> dict:
    """Read and parse the JSON header of a single safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


_PACKED_AUX = ("qzeros", "scales", "g_idx")

# Stored bits per element for each safetensors dtype.
_DTYPE_BITS = {
    "F64": 64, "I64": 64, "U64": 64,
    "F32": 32, "I32": 32, "U32": 32,
    "F16": 16, "BF16": 16, "I16": 16, "U16": 16,
    "I8": 8, "U8": 8, "BOOL": 8, "F8_E4M3": 8, "F8_E5M2": 8,
}  # fmt: skip


def _packed_linear_shapes(tensors: dict[str, dict]) -> tuple[dict[str, tuple[int, str]], bool]:
    """Map each AWQ/GPTQ ``<prefix>.`` to (unpacked weight count, dequantized dtype).

    ``qweight`` is int32-packed (``pack = 32 / bits`` weights per element), so
    its raw element count under-reports the weights. Derived from shapes alone
    (no dependency on config.json), ``scales`` being ``[groups, out]``:

    - AWQ ``qweight`` is ``[in, out/pack]`` (``qweight.cols < out``): ``in = rows``.
    - GPTQ ``qweight`` is ``[in/pack, out]`` (``qweight.cols == out``): ``in`` is
      ``len(g_idx)`` when present, else ``rows * pack`` with ``pack = out /
      qzeros.cols``.

    ``qzeros`` is therefore optional. Returns the resolved prefixes plus whether
    every ``qweight`` prefix was resolved; unresolved ones are counted raw by
    the caller (an under-count, so the caller lowers confidence).
    """
    result: dict[str, tuple[int, str]] = {}
    all_resolved = True
    for name, info in tensors.items():
        if not name.endswith(".qweight"):
            continue
        prefix = name[: -len("qweight")]
        qw = info.get("shape", [])
        sc = tensors.get(prefix + "scales")
        in_features = None
        if len(qw) == 2 and sc and len(sc.get("shape", [])) == 2:
            out = sc["shape"][1]
            if 0 < qw[1] < out and out % qw[1] == 0:  # AWQ
                in_features = qw[0]
            elif qw[1] == out:  # GPTQ
                g_idx = tensors.get(prefix + "g_idx")
                qz = tensors.get(prefix + "qzeros")
                if g_idx and g_idx.get("shape"):
                    in_features = g_idx["shape"][0]
                elif qz and len(qz.get("shape", [])) == 2 and qz["shape"][1]:
                    in_features = qw[0] * (out // qz["shape"][1])
        if in_features is None:
            all_resolved = False
            continue
        result[prefix] = (in_features * out, sc.get("dtype", "unknown"))
    return result, all_resolved


def _shard_files(source: ExtractionSource) -> tuple[list[str], bool, bool]:
    """Return the safetensors files to read, whether it is sharded, and whether
    every shard the index lists is present locally (False => totals are partial)."""
    complete = True
    if source.has(_INDEX):
        index = json.loads(source.path(_INDEX).read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        wanted = set(weight_map.values())
        # Preserve only shards we actually have locally (header-only is fine).
        shards = sorted(fn for fn in wanted if source.has(fn))
        complete = len(shards) == len(wanted)
        if shards:
            return shards, True, complete
    singles = sorted(f for f in source.repo_files if f.endswith(".safetensors"))
    singles = [f for f in singles if source.has(f)]
    return singles, len(singles) > 1, complete


class SafetensorsExtractor:
    name = "safetensors"

    def can_handle(self, source: ExtractionSource) -> bool:
        return any(f.endswith(".safetensors") for f in source.repo_files)

    def extract(self, source: ExtractionSource) -> ExtractorResult:
        files, sharded, shards_complete = _shard_files(source)

        tensors: dict[str, dict] = {}
        metadata: dict = {}
        for fn in files:
            header = read_header(source.path(fn))
            for name, info in header.items():
                if name == "__metadata__":
                    # May contain SAI ModelSpec fields, training hyperparams, etc.
                    if isinstance(info, dict):
                        metadata.update(info)
                    continue
                tensors[name] = info

        claims: list[FieldClaim] = []

        # --- authoritative parameter count: sum of element counts ---
        total = 0
        stored_bits = 0  # every tensor as stored, incl. qzeros/scales/g_idx
        bits_known = True
        dtype_counts: dict[str, int] = {}
        # AWQ/GPTQ: count unpacked weights; qzeros/scales/g_idx are not parameters.
        packed, packed_ok = _packed_linear_shapes(tensors)
        for name, info in tensors.items():
            prefix, _, leaf = name.rpartition(".")
            prefix += "."
            raw_dt = info.get("dtype", "unknown")
            if raw_dt in _DTYPE_BITS:
                stored_bits += _DTYPE_BITS[raw_dt] * prod(info.get("shape", []) or [0])
            else:
                bits_known = False
            if prefix in packed and leaf in _PACKED_AUX:
                continue
            if prefix in packed and leaf == "qweight":
                n, dt = packed[prefix]
            else:
                shape = info.get("shape", [])
                n = prod(shape) if shape else 0
                dt = info.get("dtype", "unknown")
            total += n
            dtype_counts[dt] = dtype_counts.get(dt, 0) + n
        if tensors:
            claims.append(
                FieldClaim("parameters.total", total, "tensors", "high" if packed_ok else "medium")
            )
            # Native dtype = the dtype covering the most parameters.
            dominant = max(dtype_counts.items(), key=lambda kv: kv[1])[0]
            claims.append(FieldClaim("parameters.dtype_native", dominant, "tensors", "high"))
            # Measured whole-model bpw for AWQ/GPTQ: stored bits (scales/zeros
            # included) over logical weights. Needs every shard present, every
            # qweight resolved and every dtype known, else the ratio would be wrong — emit nothing.
            if packed and packed_ok and bits_known and shards_complete and total:
                claims.append(
                    FieldClaim(
                        "quantization.bits_per_weight_avg",
                        round(stored_bits / total, 3),
                        "tensors",
                        "high",
                    )
                )

        # --- tied embeddings: authoritative from tensor presence ---
        names = set(tensors)
        if "model.embed_tokens.weight" in names or any(
            n.endswith("embed_tokens.weight") for n in names
        ):
            tied = not any(n.endswith("lm_head.weight") for n in names)
            claims.append(FieldClaim("architecture.tied_embeddings", tied, "tensors", "high"))

        # --- architecture tags from tensor-name patterns (fallback) ---
        tags: list[str] = []
        for needle, tag, conf in _TENSOR_PATTERNS:
            if any(needle in n for n in names):
                if tag not in tags:
                    tags.append(tag)
                claims.append(FieldClaim("architecture.tags", [tag], "heuristic", conf))

        file_layout = "sharded" if sharded else "single"
        claims.append(FieldClaim("identity.file_layout", file_layout, "tensors", "high"))

        # passthrough: the __metadata__ dict; raw: the tensor name list only
        # (offsets are dropped — too large to keep).
        return ExtractorResult(
            claims=claims,
            passthrough={"__metadata__": metadata} if metadata else {},
            raw={"tensor_names": sorted(names)} if names else None,
            unknown_fields=[],
        )
