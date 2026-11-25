"""Versioned startup placement for the DeepSeek-V4.1 routed expert banks."""

from hashlib import sha256
import json
import os
from pathlib import Path
import stat


def checkpoint_placement_identity(paths, config) -> str:
    """Identify this local checkpoint and configuration without reading its weight payload again."""
    files = []
    for path in paths:
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        files.append((str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return placement_identity(files, config)


def placement_identity(files, config) -> str:
    """Name the checkpoint from the supplied ordered local file identities and parsed configuration."""
    payload = json.dumps([files, config], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def read_placement(path: Path, *, checkpoint: str, layers: int, experts: int, etp: int) -> tuple:
    """Read a complete physical-slot-to-logical-expert permutation for every layer."""
    limit = 4096 + layers * experts * (len(str(experts - 1)) + 3)
    def open_nonblocking(name, flags):
        return os.open(name, flags | os.O_NONBLOCK)

    with open(path, "rb", opener=open_nonblocking) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"{path}: expert placement must be a regular file")
        data = stream.read(limit + 1)
    return decode_placement(data, checkpoint=checkpoint, layers=layers, experts=experts, etp=etp, source=str(path))


def decode_placement(data: bytes, *, checkpoint: str, layers: int, experts: int, etp: int, source: str) -> tuple:
    limit = 4096 + layers * experts * (len(str(experts - 1)) + 3)
    if len(data) > limit:
        raise ValueError(f"{source}: DeepSeek expert placement exceeds {limit} bytes")
    try:
        payload = json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{source}: invalid DeepSeek expert placement JSON") from exc
    keys = {"version", "checkpoint", "etp", "placement"}
    if type(payload) is not dict or payload.keys() != keys or type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError(f"{source}: expected DeepSeek expert placement version 1")
    if payload["checkpoint"] != checkpoint or type(payload["etp"]) is not int or payload["etp"] != etp:
        raise ValueError(f"{source}: placement belongs to a different checkpoint or ETP degree")
    placement = payload["placement"]
    if type(placement) is not list or len(placement) != layers:
        raise ValueError(f"{source}: expected {layers} expert placement rows")
    logical = set(range(experts))
    for layer, row in enumerate(placement):
        if (type(row) is not list or len(row) != experts or any(type(expert) is not int for expert in row)
                or set(row) != logical):
            raise ValueError(f"{source}: layer {layer} must place each of its {experts} experts exactly once")
    return tuple(tuple(row) for row in placement)


def encode_placement(placement: tuple, *, checkpoint: str, etp: int) -> bytes:
    return json.dumps({"version": 1, "checkpoint": checkpoint, "etp": etp, "placement": placement},
                      separators=(",", ":"), allow_nan=False).encode()


def agree_placement(path: Path, *, store, rank: int, checkpoint: str, layers: int, experts: int, etp: int) -> tuple:
    """Read exactly once and let every worker verify the same leader-owned placement."""
    key = "deepseek-v41/expert-placement-v1"
    if rank == 0:
        try:
            rows = read_placement(path, checkpoint=checkpoint, layers=layers, experts=experts, etp=etp)
        except (OSError, ValueError) as exc:
            store.set(key, b"!" + str(exc).encode(errors="replace")[:1024])
            raise
        store.set(key, encode_placement(rows, checkpoint=checkpoint, etp=etp))
        return rows
    data = store.get(key)
    if data.startswith(b"!"):
        raise ValueError(data[1:].decode(errors="replace"))
    return decode_placement(data, checkpoint=checkpoint, layers=layers, experts=experts, etp=etp,
                            source="DeepSeek expert placement from rank 0")
