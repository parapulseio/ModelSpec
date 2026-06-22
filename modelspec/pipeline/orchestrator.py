"""End-to-end orchestration: source -> extractors -> merge -> reshape -> validate.

The orchestrator only depends on the Extractor protocol and the schema. Pydantic
appears here at the last step (model_validate) and nowhere upstream.
See docs/pipeline.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modelspec.extractors import ALL_EXTRACTORS
from modelspec.extractors.base import ExtractionSource, Extractor
from modelspec.pipeline.cross_validate import cross_validate
from modelspec.pipeline.merger import MergeResult, merge_claims
from modelspec.schema import ModelSpec


def detect_source_format(repo_files: list[str]) -> str:
    """Classify a repo from its file listing (see docs/pipeline.md)."""
    has_config = "config.json" in repo_files
    has_st = any(f.endswith(".safetensors") for f in repo_files)
    has_gguf = any(f.endswith(".gguf") for f in repo_files)
    has_adapter = "adapter_config.json" in repo_files
    if has_adapter:
        return "adapter"
    if has_config and has_st:
        return "hf"
    if has_gguf:
        return "gguf"
    if has_st:
        return "raw"
    return "unknown"


def _set_nested(tree: dict[str, Any], dotted: str, value: Any) -> None:
    """Assign ``value`` into a nested dict following a dotted path."""
    parts = dotted.split(".")
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def reshape(
    merged: MergeResult,
    *,
    repo_id: str | None,
    source_format: str,
    raw_config: dict | None,
    raw_gguf: dict | None,
    unknown_fields: list[str],
) -> dict[str, Any]:
    """Turn the flat merged fields into the nested ModelSpec-shaped dict."""
    tree: dict[str, Any] = {}
    per_field: dict[str, dict[str, str]] = {}
    for path, mf in merged.fields.items():
        _set_nested(tree, path, mf.value)
        per_field[path] = {"source": mf.source, "confidence": mf.confidence}

    tree.setdefault("identity", {})
    tree["identity"]["repo_id"] = repo_id
    tree["identity"]["source_format"] = source_format

    tree["provenance"] = {
        "per_field": per_field,
        "conflicts": merged.conflicts,
        "warnings": [],
        "raw_config_json": raw_config,
        "raw_gguf_kv": raw_gguf,
        "unknown_fields": sorted(set(unknown_fields)),
    }
    return tree


def extract_from_source(source: ExtractionSource) -> ModelSpec:
    """Run all applicable extractors over an already-fetched local source."""
    extractors: list[Extractor] = [e for e in ALL_EXTRACTORS if e.can_handle(source)]

    all_claims = []
    raw_config: dict | None = None
    raw_gguf: dict | None = None
    unknown_fields: list[str] = []
    for ext in extractors:
        result = ext.extract(source)
        all_claims.extend(result.claims)
        unknown_fields.extend(result.unknown_fields)
        if ext.name == "config_json":
            raw_config = result.raw
        elif ext.name == "gguf":
            raw_gguf = result.raw

    merged = merge_claims(all_claims)
    tree = reshape(
        merged,
        repo_id=source.repo_id,
        source_format=source.source_format or detect_source_format(source.repo_files),
        raw_config=raw_config,
        raw_gguf=raw_gguf,
        unknown_fields=unknown_fields,
    )
    # Pydantic's entry-point validation — the first time the schema is touched.
    spec = ModelSpec.model_validate(tree)
    # Cross-source sanity checks append to provenance.warnings (never raise).
    cross_validate(spec)
    return spec


def extract(repo_id_or_path: str, *, revision: str | None = None, offline: bool = False) -> ModelSpec:
    """Top-level entry point.

    A local directory path is treated as an already-materialized source; a HF
    repo id triggers a metadata-only fetch (see io.hf_fetcher).
    """
    path = Path(repo_id_or_path)
    if path.is_dir():
        repo_files = [
            str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()
        ]
        source = ExtractionSource(
            root=path,
            repo_files=repo_files,
            repo_id=repo_id_or_path,
            source_format=detect_source_format(repo_files),
        )
        return extract_from_source(source)

    if offline:
        raise FileNotFoundError(
            f"offline mode: {repo_id_or_path!r} is not a local directory"
        )

    # Remote: fetch metadata into a temp dir, then run the local pipeline.
    from modelspec.io.hf_fetcher import fetch_metadata

    with fetch_metadata(repo_id_or_path, revision=revision) as source:
        return extract_from_source(source)
