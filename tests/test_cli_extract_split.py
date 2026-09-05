"""CLI integration for --download-only / --analysis-only (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import modelspec.io.hf_fetcher as hf
from modelspec.cli import main
from tests.conftest import write_config, write_safetensors_header


def _patch_fake_download(monkeypatch):
    """Stand in for the HF Hub: one config.json + one safetensors header."""
    repo_files = ["config.json", "model.safetensors", "README.md"]
    monkeypatch.setattr(hf, "_list_repo_files", lambda repo_id, revision: repo_files)

    def fake_full(repo_id, fn, revision, dest):
        p = dest / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        if fn == "config.json":
            write_config(
                p,
                {
                    "architectures": ["LlamaForCausalLM"],
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                },
            )
        else:
            p.write_text("---\nlicense: apache-2.0\n---\n# model card\n")

    def fake_st(session, repo_id, fn, revision, dest):
        write_safetensors_header(
            dest / fn, {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 16]}}
        )

    monkeypatch.setattr(hf, "_download_full", fake_full)
    monkeypatch.setattr(hf, "_download_safetensors_header", fake_st)
    monkeypatch.setattr(hf, "_resolve_commit_sha", lambda repo_id, revision: "cafef00d" * 5)
    monkeypatch.setattr(
        hf,
        "_list_repo_file_hashes",
        lambda repo_id, revision: {
            "model.safetensors": {"oid": "blob-st", "sha256": "full-file-sha256"},
        },
    )


def test_download_only_writes_files_and_manifest(tmp_path, monkeypatch):
    _patch_fake_download(monkeypatch)
    out_dir = tmp_path / "org" / "model"

    rc = main(
        ["extract", "org/model", "--download-only", "--output-dir", str(out_dir)]
    )
    assert rc == 0
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "model.safetensors").is_file()
    assert (out_dir / "README.md").is_file()

    manifest_path = out_dir / "MODELSPEC_MANIFEST.md"
    assert manifest_path.is_file()
    manifest = manifest_path.read_text(encoding="utf-8")
    assert "repo_id: org/model" in manifest
    assert "commit: " + "cafef00d" * 5 in manifest
    assert "config.json" in manifest
    assert "full download" in manifest
    assert "header only (HTTP Range request)" in manifest
    # the header-only file's full-file content hash is recorded as evidence
    assert "full-file-sha256" in manifest
    # next-step commands point at the same directory
    assert f"--analysis-only" in manifest
    assert str(out_dir) in manifest


def test_analysis_only_reads_manifest_repo_id(tmp_path, monkeypatch, capsys):
    _patch_fake_download(monkeypatch)
    out_dir = tmp_path / "org" / "model"
    assert main(["extract", "org/model", "--download-only", "--output-dir", str(out_dir)]) == 0
    capsys.readouterr()  # discard download-only's stderr status lines

    rc = main(["extract", str(out_dir), "--analysis-only"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    # identity.repo_id comes from the manifest, not the local directory path
    assert data["identity"]["repo_id"] == "org/model"
    assert data["architecture"]["num_layers"] == 2
    assert data["parameters"]["total"] == 32 * 16


def test_analysis_only_rejects_missing_directory(tmp_path, capsys):
    missing = tmp_path / "does-not-exist"
    rc = main(["extract", str(missing), "--analysis-only"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--download-only" in err  # points the user at the fix


def test_download_only_rejects_existing_local_directory(tmp_path, capsys):
    local = tmp_path / "already-here"
    local.mkdir()
    rc = main(["extract", str(local), "--download-only"])
    assert rc == 2
    assert "local directory" in capsys.readouterr().err


def test_download_only_and_analysis_only_are_mutually_exclusive(capsys):
    rc = main(["extract", "org/model", "--download-only", "--analysis-only"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_download_only_rejects_offline(capsys):
    rc = main(["extract", "org/model", "--download-only", "--offline"])
    assert rc == 2
    assert "--offline" in capsys.readouterr().err
