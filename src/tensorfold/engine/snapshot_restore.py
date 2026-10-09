"""Persisted tensor restoration through explicit current-model authority only.

This boundary imports no tensor runtime and executes no serialized class name.
The native loader sees a private regular immutable copy only after the complete
safetensors header and registered layer/token/geometry/budget checks succeed.
"""

from __future__ import annotations

from pathlib import Path

from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, read_header
from tensorfold.engine.snapshot_file import private_snapshot
from tensorfold.engine.snapshot_registry import Registry


def restore_snapshot(
    path: Path, model_id: str, registry: Registry, *, load_tensors, describe_tensor, convert_numpy, expected_tokens=None
):
    """Return validated tokens/cache; malformed data raises before native load.

    ``load_tensors(private_path)`` must evaluate all lazy native reads before
    returning (tensor map, metadata). Describers/converters are trusted runtime
    adapters, not callbacks read from disk. Expected prefix tokens bind a disk
    index decision across possible source-file replacement before this copy.
    Cache miss/rebuild policy belongs to the API caller; interruption and failed
    owned cleanup remain operation failures with their original context.
    """
    if type(registry) is not Registry or type(model_id) is not str:
        raise TypeError("explicit current-model Registry and string identity required")
    bound = registry.tensor_byte_limit + MAX_HEADER_BYTES + 8
    with private_snapshot(Path(path), max_bytes=bound, integrity_sizes=registry.sizes, hash_copy=False) as copied:
        _, header = read_header(copied.path, registry.sizes)
        metadata = header.get("__metadata__") or {}
        return registry.load(
            metadata,
            header,
            model_id=model_id,
            tensor_loader=lambda: load_tensors(copied.path),
            describe_tensor=describe_tensor,
            convert_numpy=convert_numpy,
            expected_tokens=expected_tokens,
        )
