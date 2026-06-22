"""GGUF extractor — reads the KV header and tensor infos, never loads weights.

Uses the official llama.cpp ``gguf`` Python package (an optional dependency,
conditionally imported). The GGUF header carries an architecture string plus
``{arch}.*`` keys; we normalize the ones the schema cares about and pass the
rest through. See docs/extractors.md.
"""

from __future__ import annotations

from math import prod
from typing import Any

from modelspec.extractors.base import ExtractionSource, ExtractorResult, FieldClaim

try:  # optional dependency — see pyproject [project.optional-dependencies].gguf
    from gguf import GGUFReader, GGUFValueType

    _HAS_GGUF = True
except ImportError:  # pragma: no cover - exercised only when gguf is absent
    _HAS_GGUF = False


def _scalar(field: Any) -> Any:
    """Return the Python value of a scalar / string GGUF field.

    Prefers the reader's own ``contents()`` (newer gguf), falling back to a
    minimal manual decode for older versions.
    """
    if hasattr(field, "contents"):
        return field.contents()
    # Fallback: types[0] is the value type; data indexes the value part(s).
    if not field.types:
        return None
    if field.types[0] == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    part = field.parts[field.data[0]]
    value = part.tolist()
    return value[0] if isinstance(value, list) else value


def _array_len(field: Any) -> int:
    """Element count of an ARRAY field without materializing it."""
    return len(field.data)


def _kv_to_python(field: Any) -> Any:
    """Compact a field for the raw dump.

    Large arrays (token tables, merges, scores) are reduced to a length marker
    so the raw KV dump stays small while remaining informative.
    """
    if field.types and field.types[0] == GGUFValueType.ARRAY:
        return {"_array_len": _array_len(field)}
    return _scalar(field)


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

        reader = GGUFReader(path)
        fields = reader.fields
        claims: list[FieldClaim] = []

        def get(key: str) -> Any:
            field = fields.get(key)
            return _scalar(field) if field is not None else None

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
        tokens_field = fields.get("tokenizer.ggml.tokens")
        if tokens_field is not None:
            claims.append(
                FieldClaim("tokenizer.vocab_size", _array_len(tokens_field), "gguf", "high")
            )

        # --- parameters from tensor infos (authoritative, no data loaded) ---
        total = 0
        type_counts: dict[str, int] = {}
        for t in reader.tensors:
            n = int(prod(int(d) for d in t.shape))
            total += n
            tname = getattr(t.tensor_type, "name", str(t.tensor_type))
            type_counts[tname] = type_counts.get(tname, 0) + n
        if reader.tensors:
            claims.append(FieldClaim("parameters.total", total, "tensors", "high"))
            dominant = max(type_counts.items(), key=lambda kv: kv[1])[0]
            claims.append(FieldClaim("parameters.dtype_native", dominant, "tensors", "high"))

        claims.append(FieldClaim("identity.file_layout", "single", "gguf", "high"))
        claims.append(FieldClaim("architecture.tags", tags, "inferred", "medium"))

        # --- passthrough + raw ---
        # Passthrough: recognized-but-unmapped general.* keys (file_type carries
        # the quantization level, consumed in M3).
        passthrough = {}
        for key in ("general.file_type", "general.quantization_version", "general.name"):
            if key in fields:
                passthrough[key] = _scalar(fields[key])

        raw = {name: _kv_to_python(field) for name, field in fields.items()}

        return ExtractorResult(
            claims=claims,
            passthrough=passthrough,
            raw=raw,
            unknown_fields=[],
        )
