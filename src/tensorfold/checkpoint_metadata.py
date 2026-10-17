"""Bounded, checkpoint-confined metadata reads shared by both backends.

Resolve during preparation, while checkpoint files are immutable. Optional means
absent, never malformed, unreadable, or a broken link. No accelerator is imported.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, _regular_stream, checkpoint_path, read_metadata_json

_EXL3_FIELDS = ("quant_method", "version", "bits", "head_bits", "mtp_bits", "vision_bits", "codebook", "out_scales")


def _path(model_dir: str | Path, filename: str, required: bool) -> Path | None:
    # Authorize even absent names before testing absence; traversal is not optional.
    root = Path(model_dir)
    path = checkpoint_path(root, filename)
    if not stat.S_ISDIR(root.stat().st_mode):
        raise NotADirectoryError(f"{model_dir}: checkpoint metadata root must be a directory")
    try:
        (root / filename).lstat()
    except FileNotFoundError:
        # A missing leaf through a dangling parent symlink is a broken input,
        # rather than an absent optional file. Ordinary absent directories are
        # allowed (e.g. a checkpoint with no additional_chat_templates).
        for parent in (root / filename).parents:
            if parent == root.parent:
                break
            try:
                mode = parent.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                parent.stat()
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
    """Canonical quantization metadata, with config declarations authoritative.

    Without a config/text declaration, explicitly named sidecar formats take
    part in reconciliation. EXL3's tensor_storage inventory remains loader-owned;
    only its representation summary belongs in repeated metadata preflight.
    """
    from tensorfold.checkpoint_quantization import effective_config

    raw = read_json(model_dir, "config.json", required=True)
    sources = [raw]
    text = raw.get("text_config")
    if text is not None:
        if not isinstance(text, dict):
            raise ValueError("checkpoint text_config must be an object or null")
        sources.append(text)
    declared = False
    for source in sources:
        for key in ("quantization_config", "quantization"):
            block = source.get(key)
            if block is not None and not isinstance(block, dict):
                raise ValueError(f"checkpoint {key} must be an object or null")
            declared = declared or bool(block)
    sidecar = read_json(model_dir, "hf_quant_config.json")
    if not declared:
        raw = dict(raw)
        for filename, key in (("quantization_config.json", "quantization_config"), ("quant_config.json", "quantization")):
            block = read_json(model_dir, filename)
            if block is None or block.get("quant_method") is None:
                continue
            method = block["quant_method"]
            if not isinstance(method, str) or not method.strip():
                raise ValueError(f"{filename}: quant_method must be a nonempty string")
            if method.lower() == "exl3":
                block = {field: block[field] for field in _EXL3_FIELDS if field in block}
            raw[key] = block
    return effective_config(raw, sidecar, where=str(model_dir))
