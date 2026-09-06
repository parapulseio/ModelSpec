"""CLI integration for `modelspec verify` (no network)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("jsonschema")

from modelspec.cli import main
from modelspec.pipeline import extract
from tests.conftest import write_config, write_safetensors_header


def _make_spec_json(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_config(
        model_dir / "config.json",
        {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )
    write_safetensors_header(
        model_dir / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 16]}},
    )
    spec = extract(str(model_dir), offline=True)
    out = tmp_path / "spec.json"
    out.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return out


def test_verify_valid_json_passes(tmp_path, capsys):
    spec_file = _make_spec_json(tmp_path)
    rc = main(["verify", str(spec_file)])
    assert rc == 0
    assert "valid" in capsys.readouterr().out


def test_verify_rejects_bad_enum_value(tmp_path, capsys):
    spec_file = _make_spec_json(tmp_path)
    data = json.loads(spec_file.read_text())
    data["identity"]["source_format"] = "not-a-real-format"
    spec_file.write_text(json.dumps(data))

    rc = main(["verify", str(spec_file)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "identity/source_format" in err
    assert "not-a-real-format" in err


def test_verify_yaml_file_autodetected(tmp_path, capsys):
    pytest.importorskip("yaml")
    import yaml

    spec_file = _make_spec_json(tmp_path)
    data = json.loads(spec_file.read_text())
    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text(yaml.safe_dump(data), encoding="utf-8")

    rc = main(["verify", str(yaml_file)])
    assert rc == 0
    assert "valid" in capsys.readouterr().out


def test_verify_missing_file(tmp_path, capsys):
    rc = main(["verify", str(tmp_path / "does-not-exist.json")])
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err


def test_verify_malformed_json(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    rc = main(["verify", str(bad)])
    assert rc == 2
    assert "cannot parse" in capsys.readouterr().err
