"""Field catalog introspection + the explain CLI (M5)."""

from __future__ import annotations

import json

from modelspec.cli import main
from modelspec.explain import explain_field, field_catalog


def test_catalog_covers_nested_and_carries_descriptions():
    cat = field_catalog()
    paths = {d.path for d in cat}
    # top-level, nested, list-element and union-member fields all flattened
    assert "architecture.family" in paths
    assert "attention.num_kv_heads" in paths
    assert "merge.components.model_id" in paths  # list element model
    assert "quantization.file_type" in paths  # discriminated-union member
    # every documented field has a non-empty description
    by_path = {d.path: d for d in cat}
    assert by_path["architecture.family"].description
    # Literal fields expose their choices
    src_fmt = by_path["identity.source_format"]
    assert "gguf" in src_fmt.choices


def test_explain_field_matching_precedence():
    # exact dotted path
    assert [d.path for d in explain_field("architecture.family")] == ["architecture.family"]
    # bare leaf name resolves uniquely
    assert [d.path for d in explain_field("tied_embeddings")] == ["architecture.tied_embeddings"]
    # substring fallback returns multiple
    assert len(explain_field("quant")) > 1
    # no match
    assert explain_field("definitely_not_a_field") == []


def test_explain_cli_text(capsys):
    rc = main(["explain", "context.effective"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "context.effective" in out
    assert "RoPE" in out


def test_explain_cli_json_and_list_all(capsys):
    rc = main(["explain", "--format", "json"])  # no field -> list all
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) > 50
    assert all("path" in r and "description" in r for r in rows)


def test_explain_cli_unknown_field_errors(capsys):
    rc = main(["explain", "definitely_not_a_field"])
    assert rc == 2
    assert "no field matching" in capsys.readouterr().err


def test_completion_scripts():
    for shell in ("bash", "zsh", "fish"):
        rc = main(["completion", shell])
        assert rc == 0
