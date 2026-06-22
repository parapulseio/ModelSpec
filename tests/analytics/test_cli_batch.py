"""CLI integration for the batch / coverage subcommands (offline, local dirs)."""

from __future__ import annotations

import json
from pathlib import Path

from modelspec.cli import main
from tests.conftest import write_config, write_safetensors_header


def _make_model(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    write_config(
        dir_path / "config.json",
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2,
         "num_attention_heads": 4, "num_key_value_heads": 2, "custom_weird_field": 1},
    )
    write_safetensors_header(
        dir_path / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 16]}},
    )


def _targets_file(tmp_path: Path, names: list[str]) -> Path:
    for n in names:
        _make_model(tmp_path / n)
    f = tmp_path / "targets.txt"
    f.write_text("\n".join(str(tmp_path / n) for n in names) + "\n", encoding="utf-8")
    return f


def test_batch_writes_specs_and_json_report(tmp_path: Path, capsys):
    targets = _targets_file(tmp_path, ["a", "b"])
    out_dir = tmp_path / "specs"
    rc = main(
        ["batch", str(targets), "--offline", "--quiet", "--format", "json",
         "--output-dir", str(out_dir)]
    )
    assert rc == 0
    # specs persisted
    assert len(list(out_dir.glob("*.json"))) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["succeeded"] == 2
    # the planted unknown field shows up in the frequency report
    fields = [row["field"] for row in report["unknown_field_frequency"]]
    assert "custom_weird_field" in fields


def test_coverage_text_report(tmp_path: Path, capsys):
    targets = _targets_file(tmp_path, ["a", "b"])
    rc = main(["coverage", str(targets), "--offline", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "canonical fill rates" in out
    assert "architecture.family" in out


def test_batch_nonzero_exit_on_failure(tmp_path: Path, capsys):
    f = tmp_path / "targets.txt"
    f.write_text(str(tmp_path / "missing") + "\n", encoding="utf-8")
    rc = main(["batch", str(f), "--offline", "--quiet", "--format", "json"])
    assert rc == 1  # a failure -> non-zero
    report = json.loads(capsys.readouterr().out)
    assert report["failed"] == 1
