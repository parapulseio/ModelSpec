"""Batch extraction — concurrency, ordering, fault tolerance, target parsing."""

from __future__ import annotations

from pathlib import Path

from modelspec.analytics import read_targets, run_batch
from tests.conftest import write_config, write_safetensors_header


def _make_model(dir_path: Path, layers: int = 2) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    write_config(
        dir_path / "config.json",
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": layers,
         "num_attention_heads": 4, "num_key_value_heads": 2},
    )
    write_safetensors_header(
        dir_path / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 16]}},
    )


def test_run_batch_ok_and_failures(tmp_path: Path):
    _make_model(tmp_path / "a")
    _make_model(tmp_path / "b")
    targets = [str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "does-not-exist")]

    result = run_batch(targets, offline=True, max_workers=4)

    assert result.total == 3
    assert result.succeeded == 2
    assert result.failed == 1
    # order preserved
    assert [it.target for it in result.items] == targets
    assert result.failures[0].error is not None
    assert all(s.architecture.family == "llama" for s in result.specs)


def test_limit_samples_first_n(tmp_path: Path):
    for name in ("a", "b", "c"):
        _make_model(tmp_path / name)
    targets = [str(tmp_path / n) for n in ("a", "b", "c")]
    result = run_batch(targets, offline=True, limit=2)
    assert result.total == 2


def test_progress_callback(tmp_path: Path):
    _make_model(tmp_path / "a")
    seen = []
    run_batch([str(tmp_path / "a")], offline=True, on_progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (1, 1)


def test_read_targets_skips_comments_and_blanks(tmp_path: Path):
    f = tmp_path / "targets.txt"
    f.write_text("# header\n\norg/model-a\norg/model-b  # inline\n\n", encoding="utf-8")
    assert read_targets(f) == ["org/model-a", "org/model-b"]
