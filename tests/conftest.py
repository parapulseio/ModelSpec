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


def write_gguf(
    path: Path,
    kv: dict[str, Any],
    tensors: dict[str, tuple[list[int], str]],
) -> None:
    """Write a tiny but valid GGUF file using the official gguf writer.

    ``kv`` maps GGUF keys to values (typed below); ``tensors`` maps tensor name
    -> (shape, ggml_type_name). Zero-filled tensor data is written so the file
    is fully valid. Skips automatically if gguf is not installed (caller should
    ``importorskip``).
    """
    import numpy as np
    from gguf import GGUFWriter, GGMLQuantizationType

    arch = kv.get("general.architecture", "llama")
    writer = GGUFWriter(str(path), arch)
    for key, value in kv.items():
        if key == "general.architecture":
            continue  # already written by the GGUFWriter constructor
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, list):  # array of strings (e.g. tokenizer tokens)
            writer.add_array(key, value)

    for name, (shape, _ggml_type) in tensors.items():
        data = np.zeros(shape, dtype=np.float32)
        writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
