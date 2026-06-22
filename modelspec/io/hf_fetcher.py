"""Metadata-only fetching from the Hugging Face Hub.

Key constraint: never download weights. Small files (config.json, indexes,
licenses, tokenizer configs) are downloaded whole; safetensors files are fetched
header-only via HTTP Range requests. An 8B model is ~16GB on disk but this pulls
only a few MB. See docs/architecture.md / docs/pipeline.md.
"""

from __future__ import annotations

import json
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
}
# License file names — note it is NOT just LICENSE* (see AGENTS.md).
_LICENSE_NAMES = {
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "MODEL_LICENSE",
    "USE_POLICY.md",
    "Notice",
    "NOTICE",
}

# Header probe size: 8-byte length prefix + a generous JSON header budget.
_HEADER_PROBE = 8 + 16 * 1024 * 1024  # 16 MB is plenty for any real header.


def _list_repo_files(repo_id: str, revision: str | None) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo_id, revision=revision)


def _download_full(repo_id: str, filename: str, revision: str | None, dest_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id, filename, revision=revision)
    target = dest_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, target)


def _download_safetensors_header(
    repo_id: str, filename: str, revision: str | None, dest_dir: Path
) -> None:
    """Fetch only the header of a safetensors file via a Range request.

    Writes a truncated, header-only file that the safetensors extractor can read
    (it only ever parses the length-prefixed JSON header).
    """
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id, filename, revision=revision)

    # First 8 bytes give the JSON header length.
    head = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=30)
    head.raise_for_status()
    n = struct.unpack("<Q", head.content[:8])[0]

    end = 8 + n - 1
    body = requests.get(url, headers={"Range": f"bytes=0-{end}"}, timeout=60)
    body.raise_for_status()

    target = dest_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body.content)


@contextmanager
def fetch_metadata(repo_id: str, revision: str | None = None) -> Iterator[ExtractionSource]:
    """Download a model's metadata into a temp dir and yield an ExtractionSource.

    The temp dir is cleaned up when the context exits.
    """
    repo_files = _list_repo_files(repo_id, revision)
    tmp = Path(tempfile.mkdtemp(prefix="modelspec-"))
    try:
        for fn in repo_files:
            base = fn.rsplit("/", 1)[-1]
            if base in _SMALL_FILES or base in _LICENSE_NAMES:
                _download_full(repo_id, fn, revision, tmp)
            elif fn.endswith(".safetensors"):
                _download_safetensors_header(repo_id, fn, revision, tmp)
            # weights (.bin, .gguf binary body, etc.) are intentionally skipped.

        source = ExtractionSource(
            root=tmp,
            repo_files=repo_files,
            repo_id=repo_id,
            source_format=detect_source_format(repo_files),
        )
        yield source
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
