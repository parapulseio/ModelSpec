"""Metadata-only fetching from the Hugging Face Hub.

Key constraint: never download weights. Small files (config.json, indexes,
licenses, tokenizer configs) are downloaded whole; safetensors / GGUF files are
fetched header-only via HTTP Range requests. An 8B model is ~16GB on disk but
this pulls only a few MB. See docs/architecture.md / docs/pipeline.md.

Robustness note: a CDN may *ignore* the Range header and answer 200 with the
full multi-GB file. We therefore stream the response and stop reading after the
exact number of bytes we need, so we never buffer a whole weight file even when
Range is not honored.
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from modelspec.extractors.base import ExtractionSource
from modelspec.pipeline.orchestrator import detect_source_format

# Name of the manifest ``download_metadata`` writes into the output directory;
# the orchestrator's local-directory branch reads it back to recover the real
# repo_id (see modelspec/pipeline/orchestrator.py).
MANIFEST_FILENAME = "MODELSPEC_MANIFEST.md"

# Small metadata files worth downloading in full when present.
_SMALL_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "adapter_config.json",
    "generation_config.json",
    # README.md carries the model-card front-matter (license + merge evidence).
    "README.md",
    # mergekit recipe — the highest-confidence merge signal.
    "mergekit_config.yml",
    "mergekit_config.yaml",
}
# License file names — note it is NOT just LICENSE* (see AGENTS.md).
_LICENSE_NAMES = {
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "MODEL_LICENSE",
    "MODEL_LICENSE.md",
    "USE_POLICY.md",
    "Notice",
    "NOTICE",
}

# GGUF header + tensor-info section sits at the file start; this prefix is enough
# for GGUFReader to parse all KV and tensor shapes/types without the weights.
_GGUF_PREFIX = 24 * 1024 * 1024  # 24 MB
# Concurrent per-file downloads within one model (shards are independent). Keeps
# huge sharded models fast without hammering the Hub; HF_TOKEN avoids throttling.
_FETCH_WORKERS = 16
# Process-wide cap on simultaneous downloads, shared across ALL targets. In a
# batch run, per-target (_FETCH_WORKERS) × per-batch (--workers) parallelism is
# multiplicative and would otherwise exhaust file descriptors / connections
# ("Too many open files", "Max retries exceeded"). This semaphore bounds the
# real concurrency regardless of how many targets/shards are in flight.
_MAX_CONCURRENT_DOWNLOADS = 16
_DOWNLOAD_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_DOWNLOADS)


def _guarded(fn, args):
    """Run one download holding the global concurrency permit."""
    with _DOWNLOAD_SEMAPHORE:
        return fn(*args)
# Safety cap on the safetensors JSON header (guards against an absurd length).
_SAFETENSORS_HEADER_CAP = 256 * 1024 * 1024  # 256 MB


def _resolve_token() -> str | None:
    """Resolve the HF token explicitly (env first, then the stored token).

    We pass this token into every huggingface_hub call rather than relying on
    implicit env detection, which varies across hub versions and is the usual
    cause of the "unauthenticated requests" warning despite HF_TOKEN being set.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:  # pragma: no cover - older hub versions
            token = None
    return token or None


def _list_repo_files(repo_id: str, revision: str | None) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi(token=_resolve_token()).list_repo_files(repo_id, revision=revision)


def _resolve_commit_sha(repo_id: str, revision: str | None) -> str | None:
    """The full commit SHA that ``revision`` (a branch/tag/SHA) resolves to.

    ``revision`` alone isn't a stable version marker — "main" moves. Recording
    the resolved commit is what lets someone later prove exactly which commit
    a --download-only snapshot came from. Best-effort: a Hub hiccup here must
    not fail the download itself.
    """
    from huggingface_hub import HfApi

    try:
        return HfApi(token=_resolve_token()).model_info(repo_id, revision=revision).sha
    except Exception:
        return None


def _list_repo_file_hashes(repo_id: str, revision: str | None) -> dict[str, dict[str, Any]]:
    """Per-file content hashes at ``revision``: ``{path: {"oid": ..., "sha256": ...}}``.

    ``oid`` is the git blob id of the file's content (present for every file);
    ``sha256`` is the full-file content hash for LFS-tracked files (safetensors
    / gguf / bin) — computed by the Hub over the *complete* file, so it lets a
    reader verify a header-only download against the real published file
    without downloading the rest of it. Best-effort: returns {} on any error,
    so a Hub hiccup here degrades the manifest but never blocks the download.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    try:
        tree = HfApi(token=_resolve_token()).list_repo_tree(
            repo_id, recursive=True, revision=revision
        )
        return {
            item.path: {
                "oid": item.blob_id,
                "sha256": item.lfs.sha256 if item.lfs else None,
            }
            for item in tree
            if isinstance(item, RepoFile)
        }
    except Exception:
        return {}


def _make_session():
    """A requests session carrying the HF token if one is available."""
    import requests

    session = requests.Session()
    token = _resolve_token()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def _read_prefix(session, url: str, length: int, timeout: int) -> bytes:
    """Read at most ``length`` bytes from the start of ``url``.

    Streams the body and breaks once ``length`` bytes are collected, so a server
    that ignores the Range header (answering 200 with the full file) still costs
    us only ``length`` bytes — not the whole weight file.
    """
    headers = {"Range": f"bytes=0-{length - 1}", "Accept-Encoding": "identity"}
    with session.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        out = bytearray()
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if not chunk:
                break
            out.extend(chunk)
            if len(out) >= length:
                break
    return bytes(out[:length])


def _download_full(repo_id: str, filename: str, revision: str | None, dest_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id, filename, revision=revision, token=_resolve_token())
    target = dest_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, target)


def _download_safetensors_header(
    session, repo_id: str, filename: str, revision: str | None, dest_dir: Path
) -> None:
    """Fetch only the header of a safetensors file via a Range request.

    Writes a truncated, header-only file that the safetensors extractor can read
    (it only ever parses the length-prefixed JSON header).
    """
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id, filename, revision=revision)

    # First 8 bytes give the JSON header length.
    first = _read_prefix(session, url, 8, timeout=30)
    if len(first) < 8:
        raise OSError(f"could not read safetensors length prefix for {filename}")
    n = struct.unpack("<Q", first)[0]
    if n <= 0 or n > _SAFETENSORS_HEADER_CAP:
        raise ValueError(f"implausible safetensors header length ({n}) for {filename}")

    data = _read_prefix(session, url, 8 + n, timeout=60)
    target = dest_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _download_gguf_header(
    session, repo_id: str, filename: str, revision: str | None, dest_dir: Path
) -> None:
    """Fetch only the leading header/tensor-info prefix of a GGUF file.

    GGUFReader can parse KV metadata and tensor shapes/types from this prefix as
    long as we never touch tensor .data (which lives past the prefix). Writes a
    truncated file under the original name.
    """
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id, filename, revision=revision)
    data = _read_prefix(session, url, _GGUF_PREFIX, timeout=120)
    target = dest_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _classify(filename: str) -> str:
    """Classify one repo file the way ``_plan_downloads`` treats it.

    Shared by the download planner and the ``--download-only`` manifest so the
    two never drift apart on what counts as "downloaded" vs. "skipped".
    """
    base = filename.rsplit("/", 1)[-1]
    if base in _SMALL_FILES or base in _LICENSE_NAMES:
        return "full"
    if filename.endswith(".safetensors"):
        return "safetensors-header"
    if filename.endswith(".gguf"):
        return "gguf-header"
    return "skipped"  # other weights (.bin, .pth, etc.) are intentionally skipped.


def _plan_downloads(repo_id, revision, tmp, session, repo_files):
    """Build the list of (callable, args) download tasks for a repo.

    Each task is independent and writes a distinct path under ``tmp``, so they
    can run concurrently.
    """
    tasks = []
    for fn in repo_files:
        kind = _classify(fn)
        if kind == "full":
            tasks.append((_download_full, (repo_id, fn, revision, tmp)))
        elif kind == "safetensors-header":
            tasks.append((_download_safetensors_header, (session, repo_id, fn, revision, tmp)))
        elif kind == "gguf-header":
            tasks.append((_download_gguf_header, (session, repo_id, fn, revision, tmp)))
    return tasks


@contextmanager
def fetch_metadata(
    repo_id: str, revision: str | None = None, max_workers: int = _FETCH_WORKERS
) -> Iterator[ExtractionSource]:
    """Download a model's metadata into a temp dir and yield an ExtractionSource.

    File downloads (small files + per-shard safetensors/GGUF headers) run
    concurrently, so a model with hundreds of shards (e.g. a 685B model) is
    fetched in seconds instead of one sequential round-trip per shard. A single
    failed download propagates (the caller records the target as a failure),
    matching the previous sequential behaviour. The temp dir is cleaned up when
    the context exits.
    """
    repo_files = _list_repo_files(repo_id, revision)
    tmp = Path(tempfile.mkdtemp(prefix="modelspec-"))
    session = _make_session()
    try:
        tasks = _plan_downloads(repo_id, revision, tmp, session, repo_files)
        if tasks:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
                # Each task holds the process-wide download permit, so total
                # concurrency stays bounded even across many targets in a batch.
                futures = [pool.submit(_guarded, fn_, args) for fn_, args in tasks]
                # Re-raise the first download error (others are allowed to finish
                # as the pool shuts down) so the target is recorded as failed.
                for fut in futures:
                    fut.result()

        source = ExtractionSource(
            root=tmp,
            repo_files=repo_files,
            repo_id=repo_id,
            source_format=detect_source_format(repo_files),
        )
        yield source
    finally:
        session.close()
        shutil.rmtree(tmp, ignore_errors=True)


@dataclass
class DownloadedFile:
    """One repo file's outcome from ``download_metadata``, for the manifest."""

    name: str
    kind: str  # "full" | "safetensors-header" | "gguf-header" | "skipped"
    bytes_on_disk: int | None  # None for "skipped" (never written to disk)
    oid: str | None = None  # git blob id of the file's content at this revision
    sha256: str | None = None  # full-file hash (LFS files only) — verifiable even for header-only downloads


@dataclass
class DownloadResult:
    """Everything ``download_metadata`` learned, for ``render_manifest``."""

    commit_sha: str | None  # the exact commit `revision` resolved to at fetch time
    files: list[DownloadedFile]


def download_metadata(
    repo_id: str,
    revision: str | None = None,
    *,
    dest_dir: Path,
    max_workers: int = _FETCH_WORKERS,
) -> DownloadResult:
    """Download a model's metadata into ``dest_dir`` and leave it on disk.

    Same file selection and concurrency as ``fetch_metadata`` (small files in
    full, safetensors/GGUF header-only, other weights skipped), but writes into
    a caller-supplied persistent directory instead of a temp dir that gets
    cleaned up — this is what powers ``modelspec extract --download-only``.
    Also resolves the commit ``revision`` points to and each file's content
    hash, so the manifest can prove exactly which upstream version was fetched
    even for header-only files.
    """
    repo_files = _list_repo_files(repo_id, revision)
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = _make_session()
    try:
        tasks = _plan_downloads(repo_id, revision, dest_dir, session, repo_files)
        if tasks:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
                futures = [pool.submit(_guarded, fn_, args) for fn_, args in tasks]
                for fut in futures:
                    fut.result()
    finally:
        session.close()

    commit_sha = _resolve_commit_sha(repo_id, revision)
    hashes = _list_repo_file_hashes(repo_id, revision)

    entries = []
    for fn in repo_files:
        kind = _classify(fn)
        size = None
        if kind != "skipped":
            local = dest_dir / fn
            size = local.stat().st_size if local.is_file() else None
        h = hashes.get(fn, {})
        entries.append(
            DownloadedFile(
                name=fn, kind=kind, bytes_on_disk=size, oid=h.get("oid"), sha256=h.get("sha256")
            )
        )
    return DownloadResult(commit_sha=commit_sha, files=entries)


_KIND_LABEL = {
    "full": "full download",
    "safetensors-header": "header only (HTTP Range request)",
    "gguf-header": f"header only (first {_GGUF_PREFIX // (1024 * 1024)}MB prefix)",
    "skipped": "skipped — weight file, not downloaded",
}


def render_manifest(
    *,
    repo_id: str,
    revision: str | None,
    dest_dir: Path,
    commit_sha: str | None,
    entries: list[DownloadedFile],
) -> str:
    """Render the ``--download-only`` markdown report for one target.

    Documents what was pulled to disk and how (so a reader can trust that no
    weights were downloaded), the exact upstream commit + per-file content
    hashes (so a header-only download can still be verified against the real
    file later), plus copy-pasteable next-step commands.
    """
    total_bytes = sum(e.bytes_on_disk or 0 for e in entries)
    n_full = sum(1 for e in entries if e.kind == "full")
    n_header = sum(1 for e in entries if e.kind in ("safetensors-header", "gguf-header"))
    n_skipped = sum(1 for e in entries if e.kind == "skipped")

    lines = [
        "# ModelSpec download manifest",
        "",
        f"- repo_id: {repo_id}",
        f"- revision: {revision or '(default branch)'}",
        f"- commit: {commit_sha or 'unknown (Hub lookup failed)'}",
        f"- fetched_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- output_dir: {dest_dir}",
        "",
        f"**{len(entries)} files on the Hub — {n_full} downloaded in full, "
        f"{n_header} header-only, {n_skipped} weight file(s) skipped. "
        f"~{total_bytes / (1024 * 1024):.2f} MB written to disk.**",
        "",
        "`revision` can move (e.g. a branch); `commit` is the exact snapshot fetched. "
        "`sha256` is the Hub's hash of the *complete* upstream file — for header-only "
        "safetensors/GGUF entries this is how you verify the header came from that exact "
        "published file without downloading the rest of it.",
        "",
        "## Files",
        "",
        "| file | kind | size on disk | oid | sha256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for e in sorted(entries, key=lambda e: e.name):
        size_str = f"{e.bytes_on_disk:,} B" if e.bytes_on_disk is not None else "—"
        lines.append(
            f"| {e.name} | {_KIND_LABEL[e.kind]} | {size_str} | "
            f"{e.oid or '—'} | {e.sha256 or '—'} |"
        )

    lines += [
        "",
        "## Next steps",
        "",
        "Analyze what's here now — no network access:",
        "",
        f"    modelspec extract {dest_dir} --analysis-only",
        "",
        "Re-download fresh copies (e.g. the repo changed upstream), then analyze:",
        "",
        f"    modelspec extract {repo_id} --download-only --output-dir {dest_dir}",
        f"    modelspec extract {dest_dir} --analysis-only",
        "",
        "Or skip this two-step workflow and do both in one shot "
        "(fetches over the network every time):",
        "",
        f"    modelspec extract {repo_id}",
        "",
    ]
    return "\n".join(lines)
