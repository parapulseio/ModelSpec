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


def _shard_files(source: ExtractionSource) -> tuple[list[str], bool]:
    """Return the list of safetensors files to read and whether it is sharded."""
    if source.has(_INDEX):
        index = json.loads(source.path(_INDEX).read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        # Preserve only shards we actually have locally (header-only is fine).
        shards = sorted({fn for fn in weight_map.values() if source.has(fn)})
        if shards:
            return shards, True
    singles = sorted(f for f in source.repo_files if f.endswith(".safetensors"))
    singles = [f for f in singles if source.has(f)]
    return singles, len(singles) > 1


class SafetensorsExtractor:
    name = "safetensors"

    def can_handle(self, source: ExtractionSource) -> bool:
        return any(f.endswith(".safetensors") for f in source.repo_files)

    def extract(self, source: ExtractionSource) -> ExtractorResult:
        files, sharded = _shard_files(source)

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
        dtype_counts: dict[str, int] = {}
        for info in tensors.values():
            shape = info.get("shape", [])
            n = prod(shape) if shape else 0
            total += n
            dt = info.get("dtype", "unknown")
            dtype_counts[dt] = dtype_counts.get(dt, 0) + n
        if tensors:
            claims.append(FieldClaim("parameters.total", total, "tensors", "high"))
            # Native dtype = the dtype covering the most parameters.
            dominant = max(dtype_counts.items(), key=lambda kv: kv[1])[0]
            claims.append(FieldClaim("parameters.dtype_native", dominant, "tensors", "high"))

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
