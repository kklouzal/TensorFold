"""Bounded, checkpoint-confined metadata reads shared by both backends.

Resolve during preparation, while checkpoint files are immutable. Optional means
absent, never malformed, unreadable, or a broken link. No accelerator is imported.
"""

from __future__ import annotations

import os
from pathlib import Path

from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, _regular_stream, checkpoint_path, read_metadata_json


def _path(model_dir: str | Path, filename: str, required: bool) -> Path | None:
    # Authorize even absent names before testing absence; traversal is not optional.
    path = checkpoint_path(model_dir, filename)
    try:
        (Path(model_dir) / filename).lstat()
    except FileNotFoundError:
        if required:
            raise
        return None
    return path


def read_json(model_dir: str | Path, filename: str, *, required: bool = False) -> dict | None:
    """Read one strict JSON object, or None for an absent optional file."""
    path = _path(model_dir, filename, required)
    return None if path is None else read_metadata_json(path)


def read_text(model_dir: str | Path, filename: str, *, required: bool = False) -> str | None:
    """Read bounded regular UTF-8 metadata under the same limit as JSON."""
    path = _path(model_dir, filename, required)
    if path is None:
        return None
    with _regular_stream(path) as stream:
        size = os.fstat(stream.fileno()).st_size
        if not 0 < size <= MAX_HEADER_BYTES:
            raise ValueError(f"{path}: checkpoint metadata size must lie in [1, {MAX_HEADER_BYTES}]")
        raw = stream.read(size + 1)
    if len(raw) != size:
        raise ValueError(f"{path}: checkpoint metadata changed during its read")
    return raw.decode("utf-8")


def read_config(model_dir: str | Path) -> dict:
    """The model config reconciled with recognized quantization sidecar metadata."""
    from tensorfold.checkpoint_quantization import effective_config

    raw = read_json(model_dir, "config.json", required=True)
    sidecar = read_json(model_dir, "hf_quant_config.json")
    return effective_config(raw, sidecar, where=str(model_dir))
