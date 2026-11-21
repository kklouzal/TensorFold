"""DeepSeek-V4.1 Q8_0 / Q5_K dense projections on 1Cat dmv13 segment planes (SM70).

Provenance: 1Cat-vLLM 4d57473d ``vllm/models/deepseek_v41/dense_tm.py`` (SmallM: same-input projections in one
launch, split-K choice), ``quantization/gguf_dmv13_dense.py`` (plane layout and FP16 coefficient rounding) and
``csrc/sm70_turbomind/ops/gguf_dmv13_core.cuh`` (= ``onecat_dmv13_core.cuh``), plus the friend's overlay patch
112 (9..32 rows in one weight pass, 2 or 4 8-row tiles; each row bit-identical to its 8-row dmv13 piece) and
patch 184 (large M: dequantize + cuBLAS HGEMM). Kernels: ``ds41_dense.cu``.

One resident copy per projection: the planes (Q8_0 9 bits/weight incl. the duplicated FP16 scale word, Q5_K
6 bits/weight plus its 16-byte super-block headers, 0.5 bit/weight). dense_rows multiplies what the 1Cat route
multiplies: Q8_0 ``fp16(q * d)`` (identical to the ggml FP16 decode), Q5_K ``fp16(q * fp16(d * sc) + fp16(-dmin
* m))`` (one FP16 FMA over FP16-rounded coefficients). The restore (prefill GEMMs) decodes the ggml FP16 weights
exactly: Q5_K ``fp16(d * sc * q - dmin * m)`` in FP32 from the kept headers (the FP16-rounded coefficients cost
+0.015 nats/token prefill NLL on the DS41 corpus).

* ``rows <= ROWS_BAND``: ``dense_rows`` (mma.m8n8k4, FP32 accumulation, deterministic split-K in-launch) in
  passes of up to ``MAX_ROWS`` rows; every row's value is independent of the row count.
* more rows: each member is restored to FP16 in bounded row chunks of ``Scratch.restore`` (the same weights,
  decoded by the kernel's own ``decode``) and multiplied with ``torch.matmul`` (no persistent FP16 copy).

Contracts. ``Scratch`` is shared by every ``Group`` of one device and serves calls issued in one stream order
(split-K counters start zero and every completed launch leaves them zero; the restore buffer is reused by the
next call). Calls are graph-capturable for ``rows <= ROWS_BAND`` (no allocation besides the outputs, which the
caller may pass). The caller retains inputs and outputs until current-stream completion.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

Q5K, Q8 = 1, 4                       # dmv13 plane formats
FORMAT = {8: Q8, 13: Q5K}            # GGUF type -> plane format
MAX_ROWS = 32                        # dense_rows: up to 4 8-row tiles per weight pass
ROWS_BAND = 64                       # dense_rows (in 32-row passes) up to here, restore + cuBLAS above (measured)
TARGET_CTAS = 80                     # 1Cat VLLM_V41_DMV_CTAS default
_TILE4 = {(4, 1)}                    # (warps, tp) configs whose 4-tile instance fits shared memory
RESTORE_BYTES = 40 << 20             # FP16 restore chunk for rows > MAX_ROWS: every projection but the head in one
                                     # GEMM (engram_wkv 3200 x 6144 = 37.5 MiB)


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("DeepSeek-V4.1 dmv13 dense projections require a Volta-enabled PyTorch build and SM70")
    here = Path(__file__).parent
    sources = [here / "ds41_dense.cpp", here / "ds41_dense.cu"]
    headers = [here / "onecat_dmv13_core.cuh"]
    flags = ["-O3"]
    identity = hashlib.sha256(repr(flags).encode())
    for path in [*sources, *headers]:
        identity.update(path.name.encode() + b"\0" + path.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_ds41_dense_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cflags=["-O3"], extra_cuda_cflags=flags)


def prepare():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the DeepSeek-V4.1 dense extension before graph capture")
    return extension()


def plane_bytes(fmt: int, n: int, k: int) -> tuple[int, int, int]:
    """(codes, high, scale) bytes of an [n, k] projection (gguf_dmv13_dense.pack layout)."""
    tiles, groups = -(-n // 32), -(-(k // 32) // 4)
    codes = tiles * groups * 4 * 32 * 16 * (2 if fmt == Q8 else 1)
    high = tiles * groups * 32 * 16 if fmt == Q5K else 0
    return codes, high, tiles * groups * 32 * 16


class Planes:
    """One [n, k] Q8_0 / Q5_K projection as dmv13 planes (uint8 codes / high / scale)."""

    def __init__(self, fmt: int, n: int, k: int, device):
        if fmt not in (Q5K, Q8) or n < 1 or k < 32 or k % 32:
            raise ValueError("dmv13 planes need Q5_K / Q8_0, n >= 1 and k a positive multiple of 32")
        self.fmt, self.n, self.k = fmt, n, k
        sizes = plane_bytes(fmt, n, k)
        self.codes, self.high, self.scale = (torch.empty(s, dtype=torch.uint8, device=device) for s in sizes)
        self.tiles = -(-n // 32)
        # Q5_K: the original 16-byte super-block headers (d, dmin, 6-bit scales/mins) of the column window, so the
        # restore decodes with the exact FP32 coefficients (the scale plane holds them rounded to FP16 for dmv13)
        self.first_step = None
        self.head = torch.empty(0, dtype=torch.uint8, device=device)   # allocated by the first pack

    @property
    def shape(self):
        return (self.n, self.k)

    @property
    def nbytes(self) -> int:
        return self.codes.numel() + self.high.numel() + self.scale.numel() + self.head.numel()

    def _blocks(self, first_step: int) -> tuple[int, int]:
        """(first, count) Q5_K super-blocks covering column steps [first_step, +k / 32)."""
        first = first_step // 8
        return first, (first_step + self.k // 32 + 7) // 8 - first

    def pack(self, raw: torch.Tensor, ggml_type: int, first_step: int, tile0: int = 0) -> None:
        """Planes of rows [32 tile0, +raw rows) from original GGUF row bytes ``raw`` (CUDA uint8 [rows, bytes],
        unit column stride), columns [32 first_step, +k)."""
        if FORMAT.get(ggml_type) != self.fmt:
            raise ValueError("GGUF type does not match the planes")
        rows = raw.shape[0]
        if rows % 32 and 32 * tile0 + rows != self.n:
            raise ValueError("only the last pack chunk may end inside a 32-row tile")
        end = tile0 + -(-rows // 32)
        if end > self.tiles:
            raise ValueError("pack rows beyond the projection")

        def part(t):
            if t.numel() == 0:
                return t
            per = t.numel() // self.tiles
            return t[tile0 * per:end * per]
        extension().pack_planes(raw, int(ggml_type), int(first_step), self.k // 32, part(self.codes),
                                part(self.high), part(self.scale))
        if self.fmt == Q5K:
            if self.first_step is None:
                self.first_step = int(first_step)
                self.head = torch.empty(self.n * 16 * self._blocks(first_step)[1], dtype=torch.uint8,
                                        device=self.codes.device)
            elif self.first_step != first_step:
                raise ValueError("every pack chunk of a projection takes the same column window")
            b0, nb = self._blocks(first_step)
            heads = raw[:, 176 * b0:176 * (b0 + nb)].reshape(rows, nb, 176)[:, :, :16]
            self.head.view(self.n, nb, 16)[32 * tile0:32 * tile0 + rows].copy_(heads)

    def restore(self, out: torch.Tensor, tile0: int = 0) -> torch.Tensor:
        """FP16 rows [32 tile0, +out rows) into ``out`` [count, k]: the ggml FP16 decode (Q8_0: also what
        dense_rows multiplies; Q5_K: dense_rows multiplies FP16-rounded sub-block coefficients)."""
        if self.fmt == Q5K and self.first_step is None:
            raise ValueError("restore of unpacked Q5_K planes")
        extension().restore_planes(self.codes, self.high, self.scale, self.head, self.fmt, self.n, self.k,
                                   int(self.first_step or 0), int(tile0), out)
        return out


class Scratch:
    """Per-device split-K workspace, counters and FP16 restore chunk shared by every Group (one stream order)."""

    def __init__(self, device, restore_bytes: int = RESTORE_BYTES):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.ws = torch.zeros(0, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(0, dtype=torch.int32, device=device)
        self.restore = torch.empty(restore_bytes // 2, dtype=torch.float16, device=device)

    def reserve(self, floats: int, tiles: int) -> None:
        if floats > self.ws.numel():
            self.ws = torch.zeros(floats, dtype=torch.float32, device=self.device)
        if tiles > self.cnt.numel():
            self.cnt = torch.zeros(tiles, dtype=torch.int32, device=self.device)

    @property
    def nbytes(self) -> int:
        return 4 * (self.ws.numel() + self.cnt.numel()) + 2 * self.restore.numel()


class Group:
    """1..4 same-input projections; ``group(x)`` -> list of FP16 [rows, n_i] outputs, one dmv13 launch."""

    def __init__(self, members, scratch: Scratch):
        members = list(members)
        if not 1 <= len(members) <= 4 or len({m.k for m in members}) != 1:
            raise ValueError("a dmv13 group has 1..4 projections of one input width")
        if len({m.fmt for m in members}) > 2:
            raise ValueError("a dmv13 launch takes at most two formats")
        self.members, self.scratch = members, scratch
        self.k = members[0].k
        self.widths = [m.n for m in members]
        tiles = sum(m.tiles for m in members)
        groups = -(-(self.k // 32) // 4)
        # dense_tm.SmallM: split K so the launch covers about TARGET_CTAS CTAs, >= 4 K-groups per split
        split = max(1, min(8, round(TARGET_CTAS / tiles), groups // 4))
        split = max(c for c in (1, 2, 3, 4, 6, 8) if c <= split)
        self.warps, self.tp, self.split = (8, 1, split) if split > 1 else (4, 1, 1)
        self.tiles = tiles
        scratch.reserve(tiles * self.split * self.tp * 256 * 4 if self.split > 1 else 0, tiles)
        self._codes = [m.codes for m in members]
        self._high = [m.high for m in members]
        self._scale = [m.scale for m in members]
        self._fmts = [m.fmt for m in members]

    @property
    def nbytes(self) -> int:
        return sum(m.nbytes for m in self.members)

    def _rows(self, x, outs):
        rows = x.shape[0]
        native = extension()
        mt_max = 4 if (self.warps, self.tp) in _TILE4 else 2
        step = 8 * mt_max
        for start in range(0, rows, step):
            part = x[start:start + step]
            count = part.shape[0]
            mt = 1 if count <= 8 else 2 if count <= 16 else 4
            native.dense_rows(part, self._codes, self._high, self._scale, [o[start:start + step] for o in outs],
                              self._fmts, self.widths, self.k, self.split, self.warps, self.scratch.ws,
                              self.scratch.cnt, self.tp, mt)

    def _gemm(self, x, member):
        buffer = self.scratch.restore
        chunk = max(32, buffer.numel() // self.k // 32 * 32)
        if chunk >= member.n:
            weight = member.restore(buffer[:member.n * self.k].view(member.n, self.k))
            return torch.matmul(x, weight.t())
        parts = []
        for r0 in range(0, member.n, chunk):
            count = min(chunk, member.n - r0)
            weight = member.restore(buffer[:count * self.k].view(count, self.k), tile0=r0 // 32)
            parts.append(torch.matmul(x, weight.t()))
        return torch.cat(parts, dim=-1)

    def __call__(self, x, outs=None, *, max_rows: int = ROWS_BAND):
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != self.k or x.device != self.scratch.device:
            raise ValueError(f"dmv13 group expects FP16 [rows, {self.k}] on {self.scratch.device}")
        rows = x.shape[0]
        if x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
            x = x.contiguous() if not x.is_contiguous() else x.clone()
        if 1 <= rows <= max_rows:
            if outs is None:
                outs = [torch.empty(rows, n, dtype=torch.float16, device=x.device) for n in self.widths]
            self._rows(x, outs)
            return outs
        if outs is not None:
            raise ValueError("caller-owned outputs are admitted for the dense_rows band only")
        return [self._gemm(x, m) for m in self.members]


def q5k_rows(table: torch.Tensor, ids: torch.Tensor, first: int, out: torch.Tensor) -> torch.Tensor:
    """out[i] = FP16 decode of Q5_K row ``ids[i] - first`` of ``table`` when 0 <= ids[i] - first < rows, else +0."""
    extension().q5k_rows(table, ids, int(first), out)
    return out
