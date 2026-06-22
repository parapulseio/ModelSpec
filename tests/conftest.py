"""Shared test helpers for building tiny fixture files (no downloads)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


def write_safetensors_header(path: Path, tensors: dict[str, dict[str, Any]], metadata: dict | None = None) -> None:
    """Write a header-only safetensors file.

    ``tensors`` maps tensor name -> {"dtype": str, "shape": [...]}. We only write
    the JSON header (no tensor bytes), which is exactly what the extractor reads.
    """
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, info in tensors.items():
        header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset],  # zero-length is fine for header-only
        }
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config), encoding="utf-8")
