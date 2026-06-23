"""GGUF extractor — parses the KV header + tensor infos, never touches weights.

We do NOT use ``gguf.GGUFReader``: it eagerly builds numpy views over every
tensor's *data*, which lives past the metadata-only prefix we download (and is
gigabytes for a local file), raising "cannot reshape array ..." on truncated
prefixes. Instead we parse the well-defined GGUF v2/v3 binary header ourselves,
reading only the leading bytes (header + KV + tensor infos) and never the data.

The ``gguf`` package is still used (optional dependency) for its pure data
tables: tensor-type names, block sizes (bits-per-weight), and the file-type enum.
See docs/extractors.md.
"""

from __future__ import annotations

import struct
from math import prod
from typing import Any, BinaryIO

from modelspec.extractors.base import ExtractionSource, ExtractorResult, FieldClaim

try:  # optional dependency — see pyproject [project.optional-dependencies].gguf
    from gguf import GGMLQuantizationType, LlamaFileType
    from gguf.constants import GGML_QUANT_SIZES

    _HAS_GGUF = True
except ImportError:  # pragma: no cover - exercised only when gguf is absent
    _HAS_GGUF = False

# Tensor types that are NOT quantization (full / half precision).
_FULL_PRECISION = {"F32", "F16", "BF16", "F64"}

# GGUF value_type -> (struct format, byte size) for fixed-width scalars.
_GGUF_SCALAR = {
    0: ("<B", 1),  # uint8
    1: ("<b", 1),  # int8
    2: ("<H", 2),  # uint16
    3: ("<h", 2),  # int16
    4: ("<I", 4),  # uint32
    5: ("<i", 4),  # int32
    6: ("<f", 4),  # float32
    7: ("<?", 1),  # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


class _GGUFTruncated(ValueError):
    """The downloaded prefix was too short to cover the GGUF header."""


class _Reader:
    """Sequential little-endian reader over a binary file (reads on demand)."""

    def __init__(self, f: BinaryIO):
        self._f = f

    def take(self, n: int) -> bytes:
        b = self._f.read(n)
        if len(b) < n:
            raise _GGUFTruncated("GGUF prefix too short to parse the header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.take(n).decode("utf-8", errors="replace")


def _read_value(r: _Reader, vtype: int) -> Any:
    """Read one KV value. Arrays are consumed but reduced to a length marker."""
    if vtype in _GGUF_SCALAR:
        fmt, size = _GGUF_SCALAR[vtype]
        return struct.unpack(fmt, r.take(size))[0]
    if vtype == _GGUF_STRING:
        return r.string()
    if vtype == _GGUF_ARRAY:
        elem_type = r.u32()
        count = r.u64()
        if elem_type in _GGUF_SCALAR:
            _, size = _GGUF_SCALAR[elem_type]
            r.take(size * count)  # skip the elements, we only need the count
        elif elem_type == _GGUF_STRING:
            for _ in range(count):
                r.string()  # advance past each string
        else:
            raise ValueError(f"unsupported GGUF array element type {elem_type}")
        return {"_array_len": count}
    raise ValueError(f"unsupported GGUF value type {vtype}")


def parse_gguf_header(f: BinaryIO) -> tuple[dict[str, Any], list[tuple[str, list[int], int]]]:
    """Parse a GGUF v2/v3 header from ``f``: (KV fields, tensor infos).

    Reads only the leading metadata; tensor *data* is never touched. ``fields``
    maps key -> value (arrays as ``{"_array_len": n}``); each tensor info is
    ``(name, dims, ggml_type_int)``.
    """
    r = _Reader(f)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file (bad magic)")
    version = r.u32()
    if version < 2:
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count = r.u64()
    kv_count = r.u64()

    fields: dict[str, Any] = {}
    for _ in range(kv_count):
        key = r.string()
        fields[key] = _read_value(r, r.u32())

    tensors: list[tuple[str, list[int], int]] = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        ggml_type = r.u32()
        r.u64()  # data offset — skipped, we never read data
        tensors.append((name, dims, ggml_type))
    return fields, tensors


def _ggml_type_name(t: int) -> str:
    try:
        return GGMLQuantizationType(t).name
    except ValueError:  # pragma: no cover - unknown/future type id
        return f"TYPE_{t}"


def _filetype_name(value: Any) -> str | None:
    """Map a general.file_type enum value to a readable name (e.g. "Q4_K_M")."""
    try:
        return LlamaFileType(int(value)).name.removeprefix("MOSTLY_")
    except (ValueError, TypeError):
        return None


def _avg_bits_per_weight(type_counts: dict[int, int], total: int) -> float | None:
    """Measured average bits-per-weight across all tensors.

    GGML_QUANT_SIZES maps a tensor type to (block_elements, block_bytes); a
    block-quantized type like Q4_K_M is ~4.83 bpw, not 4.0. Returns total bits
    divided by total elements.
    """
    if not total:
        return None
    total_bits = 0.0
    for ggml_type, count in type_counts.items():
        try:
            block_elems, block_bytes = GGML_QUANT_SIZES[GGMLQuantizationType(ggml_type)]
        except (ValueError, KeyError):  # pragma: no cover - unknown type id
            continue
        total_bits += (count / block_elems) * block_bytes * 8
    return round(total_bits / total, 3)


# GGUF tokenizer model name -> normalized tokenizer type.
_TOKENIZER_MODEL_MAP = {
    "gpt2": "BPE",
    "llama": "SPM",
    "bert": "WordPiece",
    "t5": "Unigram",
}


class GGUFExtractor:
    name = "gguf"

    def can_handle(self, source: ExtractionSource) -> bool:
        if not _HAS_GGUF:
            return False
        return any(f.endswith(".gguf") and source.has(f) for f in source.repo_files)

    def _gguf_path(self, source: ExtractionSource):
        for f in source.repo_files:
            if f.endswith(".gguf") and source.has(f):
                return source.path(f)
        return None

    def extract(self, source: ExtractionSource) -> ExtractorResult:
        path = self._gguf_path(source)
        if path is None:
            return ExtractorResult()

        with open(path, "rb") as f:
            fields, tensors = parse_gguf_header(f)

        claims: list[FieldClaim] = []

        def get(key: str) -> Any:
            val = fields.get(key)
            return None if isinstance(val, dict) else val  # arrays -> use array_len

        # Architecture string drives every {arch}.* key below.
        arch = get("general.architecture")
        if arch:
            claims.append(FieldClaim("architecture.family", str(arch), "gguf", "high"))

        # --- canonical scalar mappings ---
        canonical = {
            f"{arch}.block_count": "architecture.num_layers",
            f"{arch}.embedding_length": "architecture.hidden_size",
            f"{arch}.context_length": "context.declared",
            f"{arch}.attention.head_count": "attention.num_heads",
            f"{arch}.attention.head_count_kv": "attention.num_kv_heads",
        }
        for key, path_ in canonical.items():
            val = get(key)
            if val is not None:
                claims.append(FieldClaim(path_, int(val), "gguf", "high"))

        # Attention type inference from head counts.
        n_heads = get(f"{arch}.attention.head_count")
        n_kv = get(f"{arch}.attention.head_count_kv")
        tags: list[str] = ["decoder-only"]
        if n_heads is not None and n_kv is not None:
            if n_kv == 1:
                attn_type = "mqa"
            elif n_kv != n_heads:
                attn_type = "gqa"
            else:
                attn_type = "mha"
            claims.append(FieldClaim("attention.type", attn_type, "inferred", "high"))
            tags.append(attn_type)

        # MoE.
        n_experts = get(f"{arch}.expert_count")
        if n_experts:
            claims.append(FieldClaim("moe.num_experts", int(n_experts), "gguf", "high"))
            used = get(f"{arch}.expert_used_count")
            if used is not None:
                claims.append(FieldClaim("moe.top_k", int(used), "gguf", "high"))
            tags.append("moe")

        # Sliding window.
        sw = get(f"{arch}.attention.sliding_window")
        if sw:
            claims.append(FieldClaim("attention.sliding_window", int(sw), "gguf", "high"))
            tags.append("sliding-window")

        # RoPE scaling -> tag + effective context.
        rope_type = get(f"{arch}.rope.scaling.type")
        rope_factor = get(f"{arch}.rope.scaling.factor")
        if rope_type:
            tags.append(f"rope-{rope_type}")
        declared = get(f"{arch}.context_length")
        if isinstance(rope_factor, (int, float)) and declared is not None:
            claims.append(
                FieldClaim("context.effective", int(declared * rope_factor), "inferred", "medium")
            )

        # --- tokenizer ---
        tok_model = get("tokenizer.ggml.model")
        if tok_model:
            mapped = _TOKENIZER_MODEL_MAP.get(str(tok_model), str(tok_model))
            claims.append(FieldClaim("tokenizer.type", mapped, "gguf", "medium"))
        tokens = fields.get("tokenizer.ggml.tokens")
        if isinstance(tokens, dict) and "_array_len" in tokens:
            claims.append(
                FieldClaim("tokenizer.vocab_size", tokens["_array_len"], "gguf", "high")
            )

        # --- parameters from tensor infos (authoritative, no data loaded) ---
        total = 0
        type_counts: dict[int, int] = {}  # ggml type id -> element count
        for _name, dims, ggml_type in tensors:
            n = int(prod(int(d) for d in dims)) if dims else 0
            total += n
            type_counts[ggml_type] = type_counts.get(ggml_type, 0) + n
        # Readable {type_name: count} for output / quantization.tensor_types.
        named_counts: dict[str, int] = {}
        for ggml_type, c in type_counts.items():
            named_counts[_ggml_type_name(ggml_type)] = named_counts.get(
                _ggml_type_name(ggml_type), 0
            ) + c
        dominant = None
        if tensors:
            claims.append(FieldClaim("parameters.total", total, "tensors", "high"))
            dominant = max(named_counts.items(), key=lambda kv: kv[1])[0]
            claims.append(FieldClaim("parameters.dtype_native", dominant, "tensors", "high"))

        # --- quantization (GGUF branch of the discriminated union) ---
        if dominant is not None and any(name not in _FULL_PRECISION for name in named_counts):
            claims.append(FieldClaim("quantization.format", "gguf", "gguf", "high"))
            ft_name = _filetype_name(get("general.file_type")) or dominant
            claims.append(FieldClaim("quantization.file_type", ft_name, "gguf", "high"))
            bpw = _avg_bits_per_weight(type_counts, total)
            if bpw is not None:
                claims.append(
                    FieldClaim("quantization.bits_per_weight_avg", bpw, "tensors", "high")
                )
            claims.append(
                FieldClaim("quantization.tensor_types", named_counts, "tensors", "high")
            )
            # imatrix can't be told apart at the byte level; filename is the only
            # cheap cue. Leave None when absent rather than asserting False.
            if "imat" in path.name.lower():
                claims.append(FieldClaim("quantization.has_imatrix", True, "heuristic", "low"))

        claims.append(FieldClaim("identity.file_layout", "single", "gguf", "high"))
        claims.append(FieldClaim("architecture.tags", tags, "inferred", "medium"))

        # --- passthrough + raw ---
        passthrough = {
            key: fields[key]
            for key in ("general.file_type", "general.quantization_version", "general.name")
            if key in fields
        }
        return ExtractorResult(
            claims=claims,
            passthrough=passthrough,
            raw=fields,  # scalars + {"_array_len": n} for arrays — already compact
            unknown_fields=[],
        )
