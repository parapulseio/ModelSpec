"""Merge extractor — detection signals, recipe parsing, lineage."""

from __future__ import annotations

from pathlib import Path

from modelspec.extractors.base import ExtractionSource
from modelspec.extractors.merge import MergeExtractor, _normalize_method, _parse_front_matter


def _claims(tmp_path: Path, files: dict[str, str]) -> dict:
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    src = ExtractionSource(root=tmp_path, repo_files=list(files))
    result = MergeExtractor().extract(src)
    return {c.field_path: c.value for c in result.claims}


def test_normalize_method():
    assert _normalize_method("DARE-TIES") == "dare_ties"
    assert _normalize_method("SLERP") == "slerp"
    assert _normalize_method(None) is None


def test_front_matter_parser_list_and_scalar():
    fm = _parse_front_matter(
        "---\nlicense: apache-2.0\nbase_model:\n- a/x\n- b/y\ntags:\n- mergekit\n---\nbody"
    )
    assert fm["license"] == "apache-2.0"
    assert fm["base_model"] == ["a/x", "b/y"]
    assert fm["tags"] == ["mergekit"]


def test_mergekit_config_yml_high_confidence(tmp_path: Path):
    recipe = (
        "merge_method: slerp\n"
        "base_model: org/base\n"
        "models:\n"
        "  - model: org/base\n"
        "  - model: org/donor\n"
        "    parameters:\n"
        "      weight: 0.5\n"
    )
    claims = _claims(tmp_path, {"mergekit_config.yml": recipe})
    assert claims["merge.detection_signal"] == "config_file"
    assert claims["merge.confidence"] == "high"
    assert claims["merge.method"] == "slerp"
    assert claims["merge.has_config_yml"] is True
    components = claims["merge.components"]
    ids = [c["model_id"] for c in components]
    assert "org/base" in ids and "org/donor" in ids
    donor = next(c for c in components if c["model_id"] == "org/donor")
    assert donor["weight"] == 0.5
    # lineage chain populated
    assert "org/donor" in claims["identity.lineage.base_models"]
    assert claims["identity.lineage.relation"] == "merge"


def test_base_model_array_signal(tmp_path: Path):
    readme = "---\nbase_model:\n- a/x\n- b/y\n---\n# Model\n"
    claims = _claims(tmp_path, {"README.md": readme})
    assert claims["merge.detection_signal"] == "base_model_array"
    assert [c["model_id"] for c in claims["merge.components"]] == ["a/x", "b/y"]


def test_card_relation_signal(tmp_path: Path):
    readme = "---\nbase_model: a/x\nbase_model_relation: merge\n---\n"
    claims = _claims(tmp_path, {"README.md": readme})
    assert claims["merge.detection_signal"] == "card_relation"


def test_hf_tag_signal(tmp_path: Path):
    readme = "---\ntags:\n- mergekit\n- text-generation\n---\n"
    claims = _claims(tmp_path, {"README.md": readme})
    assert claims["merge.detection_signal"] == "hf_tag"
    assert claims["merge.confidence"] == "medium"


def test_non_merge_emits_nothing(tmp_path: Path):
    readme = "---\nlicense: mit\ntags:\n- text-generation\n---\n# Plain model\n"
    claims = _claims(tmp_path, {"README.md": readme})
    assert claims == {}
