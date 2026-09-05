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


def test_fetch_metadata_caps_total_concurrency(monkeypatch):
    # Many shards + a high max_workers, but the process-wide semaphore must keep
    # the real concurrency bounded (else fd / connection exhaustion at scale).
    import threading
    import time

    repo_files = _sharded_repo_files(50)
    monkeypatch.setattr(hf, "_list_repo_files", lambda repo_id, revision: repo_files)
    monkeypatch.setattr(hf, "_DOWNLOAD_SEMAPHORE", threading.BoundedSemaphore(3))
    monkeypatch.setattr(hf, "_download_full", lambda *a: None)

    state = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def track(session, repo_id, fn, revision, dest):
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.01)
        with lock:
            state["cur"] -= 1

    monkeypatch.setattr(hf, "_download_safetensors_header", track)

    with hf.fetch_metadata("org/x", max_workers=16):
        pass
    assert state["max"] <= 3  # never more than the global permit, despite 16 workers


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


# --- download_metadata / render_manifest (--download-only) ---


def _patch_no_hash_lookup(monkeypatch):
    """download_metadata's commit/hash lookups are best-effort extras; keep
    tests hermetic by stubbing them out unless a test cares about the content."""
    monkeypatch.setattr(hf, "_resolve_commit_sha", lambda repo_id, revision: None)
    monkeypatch.setattr(hf, "_list_repo_file_hashes", lambda repo_id, revision: {})


def test_download_metadata_persists_files_and_classifies_kinds(tmp_path, monkeypatch):
    repo_files = ["config.json", "model.safetensors", "model.bin"]
    monkeypatch.setattr(hf, "_list_repo_files", lambda repo_id, revision: repo_files)
    _patch_no_hash_lookup(monkeypatch)

    def fake_full(repo_id, fn, revision, dest):
        (dest / fn).write_text("{}")

    def fake_st(session, repo_id, fn, revision, dest):
        write_safetensors_header(dest / fn, {"w": {"dtype": "F32", "shape": [2, 2]}})

    monkeypatch.setattr(hf, "_download_full", fake_full)
    monkeypatch.setattr(hf, "_download_safetensors_header", fake_st)

    dest = tmp_path / "out"
    result = hf.download_metadata("org/model", dest_dir=dest)

    by_name = {e.name: e for e in result.files}
    assert by_name["config.json"].kind == "full"
    assert by_name["model.safetensors"].kind == "safetensors-header"
    assert by_name["model.bin"].kind == "skipped"
    assert by_name["model.bin"].bytes_on_disk is None
    assert not (dest / "model.bin").exists()  # weights are never written
    assert (dest / "config.json").is_file()
    assert (dest / "model.safetensors").is_file()


def test_download_metadata_records_commit_and_file_hashes(tmp_path, monkeypatch):
    repo_files = ["config.json", "model.safetensors"]
    monkeypatch.setattr(hf, "_list_repo_files", lambda repo_id, revision: repo_files)
    monkeypatch.setattr(hf, "_resolve_commit_sha", lambda repo_id, revision: "deadbeef" * 5)
    monkeypatch.setattr(
        hf,
        "_list_repo_file_hashes",
        lambda repo_id, revision: {
            "config.json": {"oid": "blob-config", "sha256": None},
            "model.safetensors": {"oid": "blob-st", "sha256": "content-sha256"},
        },
    )

    def fake_full(repo_id, fn, revision, dest):
        (dest / fn).write_text("{}")

    def fake_st(session, repo_id, fn, revision, dest):
        write_safetensors_header(dest / fn, {"w": {"dtype": "F32", "shape": [2, 2]}})

    monkeypatch.setattr(hf, "_download_full", fake_full)
    monkeypatch.setattr(hf, "_download_safetensors_header", fake_st)

    result = hf.download_metadata("org/model", dest_dir=tmp_path / "out")

    assert result.commit_sha == "deadbeef" * 5
    by_name = {e.name: e for e in result.files}
    assert by_name["config.json"].oid == "blob-config"
    assert by_name["config.json"].sha256 is None  # not LFS-tracked
    assert by_name["model.safetensors"].sha256 == "content-sha256"


def test_render_manifest_has_next_step_commands(tmp_path):
    entries = [
        hf.DownloadedFile("config.json", "full", 42, oid="blob-config"),
        hf.DownloadedFile(
            "model.safetensors", "safetensors-header", 128, oid="blob-st", sha256="content-sha256"
        ),
        hf.DownloadedFile("model.bin", "skipped", None),
    ]
    dest = tmp_path / "org" / "model"
    text = hf.render_manifest(
        repo_id="org/model",
        revision=None,
        dest_dir=dest,
        commit_sha="deadbeef" * 5,
        entries=entries,
    )
    assert "repo_id: org/model" in text
    assert "commit: " + "deadbeef" * 5 in text
    assert "1 weight file(s) skipped" in text
    assert "content-sha256" in text  # full-file hash for the header-only entry
    assert "blob-config" in text
    assert f"modelspec extract {dest} --analysis-only" in text
    assert "modelspec extract org/model --download-only" in text


def test_render_manifest_handles_missing_commit_and_hashes(tmp_path):
    # Best-effort lookups can fail; the manifest must still render cleanly.
    entries = [hf.DownloadedFile("config.json", "full", 42)]
    text = hf.render_manifest(
        repo_id="org/model",
        revision=None,
        dest_dir=tmp_path,
        commit_sha=None,
        entries=entries,
    )
    assert "unknown (Hub lookup failed)" in text
