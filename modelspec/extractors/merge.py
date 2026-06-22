"""Merge extractor — detects merged models and recovers the recipe.

Signal sources, high -> low confidence (see docs/quantization-and-merge.md):
  1. mergekit_config.yml present        -> full recipe (method, models, params)
  2. README front-matter base_model_relation: merge
  3. README front-matter base_model is a multi-element array
  4. README front-matter tags / library_name == mergekit  -> "is a merge" only
  5. README YAML code-block containing merge_method

The richest available source fills ``components``; HF-tag-only detections often
know nothing but the participating model ids. The base_model chain is also
written to ``identity.lineage`` (the unified lineage home).
"""

from __future__ import annotations

import re
from typing import Any

from modelspec.extractors.base import ExtractionSource, ExtractorResult, FieldClaim

_README = "README.md"
_MERGEKIT = "mergekit_config.yml"

# Normalize mergekit method spellings: lower + dashes -> underscores.
def _normalize_method(value: Any) -> str | None:
    if not value:
        return None
    return str(value).strip().lower().replace("-", "_")


def _parse_front_matter(text: str) -> dict[str, Any]:
    """Minimal YAML front-matter parser (scalars + simple lists).

    Avoids a hard PyYAML dependency for the common model-card shape:
        key: value
        key:
        - item1
        - item2
    Nested structures are ignored (good enough for the keys we read).
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    block = text[3:end] if end != -1 else text[3:]

    result: dict[str, Any] = {}
    current_key: str | None = None
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        list_item = re.match(r"\s*-\s+(.*)$", line)
        if list_item and current_key:
            result.setdefault(current_key, [])
            if isinstance(result[current_key], list):
                result[current_key].append(list_item.group(1).strip().strip("\"'"))
            continue
        kv = re.match(r"^(\w[\w\-]*):\s*(.*)$", line)
        if kv:
            key, value = kv.group(1), kv.group(2).strip()
            if value:
                result[key] = value.strip("\"'")
                current_key = None
            else:
                result[key] = []  # value may follow as a list
                current_key = key
    return result


def _load_recipe(text: str) -> dict[str, Any] | None:
    """Parse a mergekit_config.yml body (PyYAML if available, else None)."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is an optional extra
        return None
    try:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else None
    except Exception:  # pragma: no cover - malformed YAML
        return None


def _components_from_recipe(recipe: dict) -> list[dict[str, Any]]:
    """Extract participating models from a mergekit recipe (best effort)."""
    components: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(model_id: Any, role: str, params: dict | None = None) -> None:
        if not isinstance(model_id, str) or model_id in seen:
            return
        seen.add(model_id)
        params = params or {}
        weight = params.get("weight")
        density = params.get("density")
        components.append(
            {
                "model_id": model_id,
                "weight": weight if isinstance(weight, (int, float)) else None,
                "density": density if isinstance(density, (int, float)) else None,
                "role": role,
            }
        )

    if isinstance(recipe.get("base_model"), str):
        add(recipe["base_model"], "base")
    for m in recipe.get("models", []) or []:
        if isinstance(m, dict):
            add(m.get("model"), "ingredient", m.get("parameters"))
    # frankenmerge / passthrough store models under slices[].sources[].
    for sl in recipe.get("slices", []) or []:
        if isinstance(sl, dict):
            for src in sl.get("sources", []) or []:
                if isinstance(src, dict):
                    add(src.get("model"), "ingredient")
    return components


def _readme_yaml_method(body: str) -> str | None:
    """Find merge_method inside a ```yaml code block in the README body."""
    for block in re.findall(r"```(?:ya?ml)?\n(.*?)```", body, re.DOTALL):
        m = re.search(r"merge_method:\s*([^\s#]+)", block)
        if m:
            return _normalize_method(m.group(1))
    return None


class MergeExtractor:
    name = "merge"

    def can_handle(self, source: ExtractionSource) -> bool:
        return source.has(_MERGEKIT) or source.has(_README)

    def extract(self, source: ExtractionSource) -> ExtractorResult:
        recipe: dict[str, Any] | None = None
        if source.has(_MERGEKIT):
            recipe = _load_recipe(
                source.path(_MERGEKIT).read_text(encoding="utf-8", errors="replace")
            )

        front: dict[str, Any] = {}
        body = ""
        if source.has(_README):
            text = source.path(_README).read_text(encoding="utf-8", errors="replace")
            front = _parse_front_matter(text)
            body = text

        base_model = front.get("base_model")
        base_list = base_model if isinstance(base_model, list) else (
            [base_model] if isinstance(base_model, str) else []
        )
        tags = front.get("tags") or []
        tags = tags if isinstance(tags, list) else [tags]
        is_mergekit_tag = "mergekit" in tags or "merge" in tags or front.get(
            "library_name"
        ) == "mergekit"
        readme_method = _readme_yaml_method(body) if body else None

        # --- decide detection signal (highest priority available) ---
        signal: str | None = None
        confidence = "low"
        if source.has(_MERGEKIT):
            signal, confidence = "config_file", "high"
        elif front.get("base_model_relation") == "merge":
            signal, confidence = "card_relation", "high"
        elif len(base_list) > 1:
            signal, confidence = "base_model_array", "high"
        elif is_mergekit_tag:
            signal, confidence = "hf_tag", "medium"
        elif readme_method:
            signal, confidence = "readme_yaml", "medium"

        if signal is None:
            return ExtractorResult()  # not a merge

        # --- assemble components (richest source wins) ---
        if recipe:
            components = _components_from_recipe(recipe)
        else:
            components = [{"model_id": m, "role": None} for m in base_list if m]

        method = None
        if recipe:
            method = _normalize_method(recipe.get("merge_method"))
        method = method or readme_method

        claims: list[FieldClaim] = [
            FieldClaim("merge.detection_signal", signal, "config", confidence),
            FieldClaim("merge.confidence", confidence, "config", confidence),
            FieldClaim("merge.has_config_yml", source.has(_MERGEKIT), "config", "high"),
        ]
        if method:
            claims.append(FieldClaim("merge.method", method, "config", confidence))
        if components:
            claims.append(FieldClaim("merge.components", components, "config", confidence))
        if recipe:
            claims.append(FieldClaim("merge.raw_recipe", recipe, "config", "high"))

        # base_model chain -> unified lineage home.
        lineage_models = [c["model_id"] for c in components] or base_list
        if lineage_models:
            claims.append(
                FieldClaim("identity.lineage.base_models", lineage_models, "config", confidence)
            )
            claims.append(
                FieldClaim("identity.lineage.relation", "merge", "config", confidence)
            )

        return ExtractorResult(claims=claims)
