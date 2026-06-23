"""hf_fetcher — guard against a server that ignores Range and returns 200.

No network: a fake session yields far more bytes than requested, simulating a
CDN that sends the whole (multi-GB) file. _read_prefix must stop early.
"""

from __future__ import annotations

from modelspec.io.hf_fetcher import _read_prefix


class _FakeResponse:
    def __init__(self, total_bytes: int):
        self._total = total_bytes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size: int):
        # Pretend the server ignored Range and is streaming a huge file.
        served = 0
        while served < self._total:
            n = min(chunk_size, self._total - served)
            served += n
            yield b"\x00" * n


class _FakeSession:
    def __init__(self, total_bytes: int):
        self._total = total_bytes
        self.requested_ranges: list[str] = []

    def get(self, url, headers=None, stream=False, timeout=None):
        if headers:
            self.requested_ranges.append(headers.get("Range", ""))
        return _FakeResponse(self._total)


def test_read_prefix_stops_early_when_range_ignored():
    # Server would stream 1 GB; we only want 8 bytes.
    session = _FakeSession(total_bytes=1 << 30)
    data = _read_prefix(session, "http://x", 8, timeout=10)
    assert len(data) == 8
    assert session.requested_ranges == ["bytes=0-7"]


def test_read_prefix_returns_available_when_file_smaller():
    session = _FakeSession(total_bytes=100)
    data = _read_prefix(session, "http://x", 4096, timeout=10)
    assert len(data) == 100  # file shorter than the requested prefix


# --- parallel fetch_metadata (no network: download fns are monkeypatched) ---

import json  # noqa: E402

import pytest  # noqa: E402

import modelspec.io.hf_fetcher as hf  # noqa: E402
from tests.conftest import write_safetensors_header  # noqa: E402


def _sharded_repo_files(n: int) -> list[str]:
    shards = [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    return ["config.json", "model.safetensors.index.json", *shards]


def test_fetch_metadata_parallel_aggregates_many_shards(monkeypatch):
    n = 40  # plenty to exercise the pool; each "shard" carries 100 params
    repo_files = _sharded_repo_files(n)
    shards = repo_files[2:]
    monkeypatch.setattr(hf, "_list_repo_files", lambda repo_id, revision: repo_files)

    def fake_full(repo_id, fn, revision, dest):
        p = dest / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        if fn == "config.json":
            p.write_text('{"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2}')
        else:  # the index maps a tensor per shard back to its file
            p.write_text(json.dumps({"weight_map": {s: s for s in shards}}))

    def fake_st(session, repo_id, fn, revision, dest):
        write_safetensors_header(dest / fn, {f"{fn}.w": {"dtype": "BF16", "shape": [10, 10]}})

    monkeypatch.setattr(hf, "_download_full", fake_full)
    monkeypatch.setattr(hf, "_download_safetensors_header", fake_st)

    from modelspec.pipeline import extract_from_source

    with hf.fetch_metadata("org/huge", max_workers=8) as src:
        assert all(src.has(s) for s in shards)  # every shard header fetched
        spec = extract_from_source(src)
    assert spec.parameters.total == n * 100  # all shards aggregated, none dropped


def test_fetch_metadata_propagates_a_download_failure(monkeypatch):
    repo_files = _sharded_repo_files(5)
    monkeypatch.setattr(hf, "_list_repo_files", lambda repo_id, revision: repo_files)
    monkeypatch.setattr(hf, "_download_full", lambda *a: None)

    def boom(session, repo_id, fn, revision, dest):
        raise OSError("shard fetch failed")

    monkeypatch.setattr(hf, "_download_safetensors_header", boom)

    with pytest.raises(OSError):
        with hf.fetch_metadata("org/x") as src:
            pass
