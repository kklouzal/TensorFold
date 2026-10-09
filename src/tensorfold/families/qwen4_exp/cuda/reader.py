"""Flash Next's checkpoint shards read a tensor at a time, and the row and group slices a rank takes."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch

_DT = {"U32": torch.int32, "I32": torch.int32, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
       "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8, "U16": torch.int16, "I16": torch.int16,
       "F8_E4M3": torch.float8_e4m3fn}


class _Reader:
    """Read checkpoint shards sequentially (O_DIRECT where allowed) and release each shard's cached pages."""

    def __init__(self, model_dir: Path, device: str) -> None:
        from tensorfold.cuda.direct_read import ReadAhead, Reader
        from tensorfold.cuda.tensor_file import read_metadata_json

        self.dir = model_dir
        index = read_metadata_json(self._path("model.safetensors.index.json"))
        where = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(where, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in where.items()):
            raise ValueError("checkpoint index must contain a string-to-string weight_map")
        self.where = dict(where)
        self.device = device
        self.headers: dict[str, tuple[int, dict]] = {}
        self.touched: set[str] = set()
        for shard in sorted(set(self.where.values())):
            self._path(shard)                  # authorize every indexed target before starting owned reads
        self.io = Reader()
        self.reads = ReadAhead(self.io)

    def queue(self, names) -> None:
        """Start reading ``names`` ahead (neighbours in shared reads, uploaded on a side stream) for ``get`` to take."""

        items = []
        for name in names:
            info = self.info(name)
            begin, end = info["data_offsets"]
            items.append((name, info["path"], info["byte_offset"], info["byte_offset"] + end - begin, None))
            self.touched.add(self.where[name])
        self.reads.queue(items, self.device)

    def drop(self, names) -> None:
        self.reads.drop(names)

    def layer_names(self, prefix: str, base: str, chosen: list[int], mtp: bool) -> list[list[str]]:
        """Each chosen layer's tensor names, then the MTP layer's; the n-gram shards stay with their memory map."""

        pattern = re.compile(re.escape(prefix) + "(" + re.escape(base) + r"layers\.\d+\.|mtp\.)")
        groups: dict[str, list[str]] = {}
        for name in self.where:
            m = pattern.match(name)
            if m and ".ngram_embedding." not in name:
                groups.setdefault(m.group(0), []).append(name)
        order = [f"{prefix}{base}layers.{i}." for i in chosen] + [f"{prefix}mtp."] * bool(mtp)
        return [groups.get(key, []) for key in order]

    def close(self) -> None:
        try:
            self.reads.close()
        except BaseException as primary:
            # A reported payload failure can still have completed shutdown.
            # Return Reader staging only after all read/upload owners drained.
            if not self.reads._shutdown_pending:
                try:
                    self.io.close()
                except BaseException as cleanup:
                    BaseException.add_note(primary, "checkpoint staging cleanup also failed")
                    raise primary from cleanup
            raise
        self.io.close()

    def _path(self, shard: str) -> Path:
        """Index-relative files in this model or its own HF snapshot blob store."""
        from tensorfold.cuda.tensor_file import checkpoint_path

        return checkpoint_path(self.dir, shard)

    def _header(self, shard: str) -> tuple[int, dict]:
        got = self.headers.get(shard)
        if got is None:
            from tensorfold.cuda.direct_read import read_header

            got = read_header(self._path(shard))
            self.headers[shard] = got
        return got

    def info(self, name: str) -> dict:
        """Validated tensor metadata; copied tuples cannot mutate the cached header."""
        from tensorfold.cuda.capacity import SIZES
        from tensorfold.cuda.tensor_file import tensor_shape

        shard = self.where[name]
        base, header = self._header(shard)
        entry = header[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str) or entry["dtype"] not in _DT:
            raise ValueError(f"{name}: unsupported or invalid safetensors dtype")
        # read_header proves full-buffer coverage once. These public metadata
        # and index objects can be edited by a caller, so recheck the selected
        # shape/range and authorized target after each such boundary crossing.
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        elements = tensor_shape(shape)
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)):
            raise ValueError(f"{name}: invalid tensor shape or byte range")
        begin, end = offsets
        path = self._path(shard)
        if begin < 0 or end - begin != elements * SIZES[entry["dtype"]] or base + end > path.stat().st_size:
            raise ValueError(f"{name}: tensor byte range differs from its shape or checkpoint size")
        return {"dtype": entry["dtype"], "shape": tuple(shape), "data_offsets": (begin, end),
                "byte_offset": base + begin, "path": path}

    def get_rows(self, name: str, lo: int, hi: int, device: str = "cpu") -> torch.Tensor:
        """Read only first-axis rows [lo, hi), without consuming queued CUDA reads.

        Expert spill callers exclude these fields from CUDA read-ahead. Returned
        storage owns the requested rows; no whole stacked expert tensor is read.
        """

        import math

        from tensorfold.cuda.capacity import SIZES

        info = self.info(name)
        shape, dtype = info["shape"], _DT[info["dtype"]]
        if not shape:
            raise ValueError(f"{name}: scalar tensors have no first-axis rows")
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= shape[0]:
            raise ValueError(f"{name}: rows must satisfy 0 <= lo <= hi <= {shape[0]}")
        if lo == hi:
            return torch.empty((0, *shape[1:]), dtype=dtype, device=device)
        row_bytes = math.prod(shape[1:]) * SIZES[info["dtype"]]
        raw = self.io.read(info["path"], info["byte_offset"] + lo * row_bytes,
                           (hi - lo) * row_bytes, device)
        self.touched.add(self.where[name])
        return raw.view(dtype).reshape(hi - lo, *shape[1:])

    def get(self, name: str) -> torch.Tensor:
        info = self.info(name)
        begin, end = info["data_offsets"]
        raw = self.reads.take(name)
        if raw is None:
            raw = self.io.read(info["path"], info["byte_offset"], end - begin, self.device)
        self.touched.add(self.where[name])
        return raw.view(_DT[info["dtype"]]).reshape(info["shape"])

    def has(self, name: str) -> bool:
        return name in self.where

    def release(self) -> None:
        """Drop read shards' cached pages so unified memory does not retain both host-cache and GPU copies."""

        for shard in list(self.touched):
            try:
                fd = os.open(self._path(shard), os.O_RDONLY)
            except (OSError, AttributeError):
                continue
            try:
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except (OSError, AttributeError):
                    pass                            # cache eviction is an optional filesystem advisory
            finally:
                os.close(fd)                        # owned descriptor cleanup failure is an operation failure
        self.touched.clear()


def norms_around_one(reader: _Reader, base: str, layers: list[int]) -> bool:
    """True when the norm weights are stored as scales (around 1), False when centred (around 0: add 1)."""

    means = []
    for i in layers:
        name = f"{base}layers.{i}.attn_hyper_connection.hc_norm.weight"
        if reader.has(name):
            means.append(float(reader.get(name).float().mean()))
    if not means:
        return True
    means = np.array(means)
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    if not (around_one or around_zero):
        raise ValueError(f"cannot tell how the norm weights are stored (median mean {np.median(means):.3f})")
    return around_one


def _rows(t3, lo: int, hi: int):
    """Output rows [lo, hi) of (words, scales, biases), stacked experts included."""

    return tuple(x[..., lo:hi, :].contiguous() for x in t3)


def _rows_at(t3, idx: torch.Tensor):
    return tuple(x.index_select(x.dim() - 2, idx).contiguous() for x in t3)


def _groups(t3, g0: int, g1: int):
    """Input groups [g0, g1) (32 inputs each) of (words, scales, biases): no repacking."""

    w, sc, b = t3
    return w[..., g0 * 4:g1 * 4].contiguous(), sc[..., g0:g1].contiguous(), b[..., g0:g1].contiguous()
