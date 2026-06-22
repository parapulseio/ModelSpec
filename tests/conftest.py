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
    -> (shape, ggml_type_name). Full-precision types (F32/F16) write a real
    array; quantized types (e.g. "Q4_K") write correctly-sized zero blocks so
    the file is valid. Skips automatically if gguf is not installed (caller
    should ``importorskip``).
    """
    import numpy as np
    from gguf import GGUFWriter, GGMLQuantizationType
    from gguf.constants import GGML_QUANT_SIZES
    from math import prod

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

    for name, (shape, ggml_type_name) in tensors.items():
        ggml_type = GGMLQuantizationType[ggml_type_name]
        if ggml_type_name in ("F32", "F16"):
            dt = np.float32 if ggml_type_name == "F32" else np.float16
            writer.add_tensor(name, np.zeros(shape, dtype=dt), raw_dtype=ggml_type)
        else:
            # Quantized: the writer takes the *byte* shape and derives the
            # logical shape. Last logical dim must be a multiple of block_size.
            block_elems, block_bytes = GGML_QUANT_SIZES[ggml_type]
            *lead, last = shape
            if last % block_elems != 0:
                raise ValueError(f"{last} not a multiple of block size {block_elems}")
            byte_shape = [*lead, last // block_elems * block_bytes]
            data = np.zeros(byte_shape, dtype=np.uint8)
            writer.add_tensor(name, data, raw_shape=byte_shape, raw_dtype=ggml_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
