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
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from modelspec.extractors.base import ExtractionSource
from modelspec.pipeline.orchestrator import detect_source_format

# Small metadata files worth downloading in full when present.
_SMALL_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "adapter_config.json",
    "generation_config.json",
    # README.md carries the model-card front-matter (auxiliary license evidence).
    "README.md",
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
# Safety cap on the safetensors JSON header (guards against an absurd length).
_SAFETENSORS_HEADER_CAP = 256 * 1024 * 1024  # 256 MB


def _list_repo_files(repo_id: str, revision: str | None) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo_id, revision=revision)


def _make_session():
    """A requests session carrying the HF token if one is available."""
    import requests

    session = requests.Session()
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:  # pragma: no cover - older hub versions
            token = None
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

    cached = hf_hub_download(repo_id, filename, revision=revision)
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


@contextmanager
def fetch_metadata(repo_id: str, revision: str | None = None) -> Iterator[ExtractionSource]:
    """Download a model's metadata into a temp dir and yield an ExtractionSource.

    The temp dir is cleaned up when the context exits.
    """
    repo_files = _list_repo_files(repo_id, revision)
    tmp = Path(tempfile.mkdtemp(prefix="modelspec-"))
    session = _make_session()
    try:
        for fn in repo_files:
            base = fn.rsplit("/", 1)[-1]
            if base in _SMALL_FILES or base in _LICENSE_NAMES:
                _download_full(repo_id, fn, revision, tmp)
            elif fn.endswith(".safetensors"):
                _download_safetensors_header(session, repo_id, fn, revision, tmp)
            elif fn.endswith(".gguf"):
                _download_gguf_header(session, repo_id, fn, revision, tmp)
            # other weights (.bin, .pth, etc.) are intentionally skipped.

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
