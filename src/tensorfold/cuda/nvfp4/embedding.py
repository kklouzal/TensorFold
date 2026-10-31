"""Explicit row64 FP8 tables borrowing the existing FP8G GEMM byte layout.

This role format is NOT the checkpoint parser's 128x128 ``fp8block`` scheme or
MXFP8. Lookup materializes only requested rows as RN-BF16(FP32(E4M3)*FP32(scale))
BEFORE any Gemma embedding scale/PLE reshape. FP8G GEMM scales partial dot sums;
it shares the codes/scales, but does not promise the same arithmetic as a GEMM
on the separately BF16-rounded lookup rows. Neither path changes input IDs.

The owner must keep packed tensors immutable and retain this object through all
asynchronous consumers. No move/load/export mutation is supported. GPU boundary
checks synchronize; a qualified fused lookup can replace this reference consumer
later without changing the role format. No CUDA extension is loaded here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .linear import Fp8BlockLinear

_FORMAT = "tensorfold.fp8_e4m3_row64_f32.v1"
_LAYOUT = "fp8g_fragment_64x64_v1"
_ROLES = frozenset(("token_embedding", "per_layer_embedding", "assistant_embedding"))


@dataclass(frozen=True)
class Row64Spec:
    role: str
    rows: int
    width: int

    def __post_init__(self):
        if type(self.role) is not str or self.role not in _ROLES:
            raise ValueError("unsupported row64 embedding role")
        if type(self.rows) is not int or self.rows <= 0:
            raise ValueError("row64 rows must be a positive integer")
        if type(self.width) is not int or self.width <= 0 or self.width % 64:
            raise ValueError("row64 width must be a positive multiple of 64")

    def to_manifest(self) -> dict:
        return {"format": _FORMAT, "role": self.role, "rows": self.rows,
                "width": self.width, "code_dtype": "float8_e4m3fn",
                "scale_dtype": "float32", "scale_block": [1, 64],
                "packed_layout": _LAYOUT, "lookup_dtype": "bfloat16"}

    @classmethod
    def from_manifest(cls, value: Mapping) -> "Row64Spec":
        if not isinstance(value, Mapping):
            raise ValueError("row64 manifest must be a mapping")
        expected = cls("token_embedding", 1, 64).to_manifest()
        if set(value) != set(expected):
            raise ValueError("row64 manifest has missing or unknown fields")
        for key in expected.keys() - {"role", "rows", "width"}:
            if type(value[key]) is not type(expected[key]) or value[key] != expected[key]:
                raise ValueError(f"unsupported row64 manifest {key}")
        if any(type(x) is not int for x in value["scale_block"]):
            raise ValueError("row64 scale_block requires exact integer dimensions")
        return cls(value["role"], value["rows"], value["width"])


# Invert the library's documented per-lane k32 byte permutation, once in Python.
# Each tuple gives (lane modulo 4, byte) for one logical input within a k32 half.
_lane_byte = [None] * 32
for _lane in range(4):
    for _byte in range(8):
        _p = 16 * (_byte // 4) + 4 * _lane + _byte % 4
        _kin = 16 * (_p // 16) + 2 * ((_p % 16) // 4) + _p % 2 + 8 * ((_p % 4) // 2)
        _lane_byte[_kin] = (_lane, _byte)
_LANE_BYTE = tuple(_lane_byte)
del _lane_byte, _lane, _byte, _p, _kin


def packed_offsets(rows: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte/code and FP32/scale offsets for selected canonical rows (not IDs).

    Caller admits row bounds against its spec. Temporary index storage is O(R*K),
    independent of vocabulary size. This CPU-testable address map does no IO.
    """
    columns = torch.arange(width, device=rows.device, dtype=torch.int64)
    mapping = torch.tensor(_LANE_BYTE, device=rows.device, dtype=torch.int64)
    lane, byte = mapping[columns % 32].unbind(-1)
    r = rows.reshape(-1, 1)
    group = columns // 64
    tile = (r // 64) * (width // 64) + group
    physical_lane = (r % 8) * 4 + lane
    code = (((tile * 8 + (r % 64) // 8) * 32 + physical_lane) * 2 + (columns % 64) // 32) * 8 + byte
    scale = tile * 64 + r % 64
    return code, scale


class Row64Embedding:
    """One strong packed-operation owner; no canonical-table mirror or fake weight.

    ``from_rows`` admits finite E4M3 bytes and finite positive FP32 scales. A zero
    block is represented with zero codes and a positive scale (normally one), not
    a zero/NaN scale. No clipping or scale repair occurs. Finite BF16 lookup output
    is validated before publication of each result. Constructor scanning is
    bounded by ``chunk_rows``; application admission still owns total model bytes.
    """
    def __init__(self, spec: Row64Spec, operation: Fp8BlockLinear, *, chunk_rows: int = 64):
        w = s = ids = code = scale = decoded = None
        try:
            if type(spec) is not Row64Spec or type(operation) is not Fp8BlockLinear:
                raise TypeError("row64 requires its typed spec and exact Fp8BlockLinear")
            if type(chunk_rows) is not int or chunk_rows <= 0:
                raise ValueError("row64 chunk_rows must be a positive integer")
            n, k, npad = spec.rows, spec.width, -(-spec.rows // 128) * 128
            if (operation.n, operation.k, operation.npad) != (n, k, npad):
                raise ValueError("row64 operation geometry differs from role spec")
            w, s = operation.w8, operation.bs
            if (w.dtype != torch.uint8 or w.shape != (npad * k,) or not w.is_contiguous()
                    or s.dtype != torch.uint8 or s.shape != (npad // 64, k // 64, 256)
                    or not s.is_contiguous() or w.device != s.device
                    or s.data_ptr() % 4 or operation.layout != "fp8block"):
                raise ValueError("row64 packed operation has an unsupported storage layout")
            try:
                versions = (w._version, s._version)
            except RuntimeError as exc:
                raise ValueError("row64 borrowed operation requires versioned storage; use from_rows for inference-mode loading") from exc
            self.spec, self.operation, self.chunk_rows = spec, operation, chunk_rows
            self._owners = (w, s)
            self._versions = versions
            # Admit only logical rows: library padding is not semantic model data.
            for begin in range(0, n, chunk_rows):
                ids = torch.arange(begin, min(n, begin + chunk_rows), device=w.device)
                code, scale = self._read(ids)
                if bool(((code & 127) == 127).any()):
                    raise ValueError("row64 codes contain E4M3 NaN")
                if not bool((torch.isfinite(scale) & (scale > 0)).all()):
                    raise ValueError("row64 scales must be finite and positive")
                decoded = code.view(torch.float8_e4m3fn).float() * scale
                if not bool(torch.isfinite(decoded.to(torch.bfloat16)).all()):
                    raise ValueError("row64 values overflow finite BF16 lookup output")
        except BaseException:
            # Retire only our aliases; preserve the original exception/frames.
            self.operation = None
            self._owners = ()
            raise
        finally:
            operation = w = s = ids = code = scale = decoded = None

    @classmethod
    def from_rows(cls, spec: Row64Spec, codes: torch.Tensor, scales: torch.Tensor,
                  *, chunk_rows: int = 64) -> "Row64Embedding":
        """Pack an explicit exported row64 table, without quantizing it again.

        Codes are uint8 E4M3 representation bytes or float8_e4m3fn. The caller
        retains/may release the source arrays independently after packing returns.
        Existing library packing owns its bounded chunking; no model IO happens.
        """
        if type(spec) is not Row64Spec:
            raise TypeError("row64 requires its typed spec")
        if (codes.dtype not in (torch.uint8, torch.float8_e4m3fn)
                or codes.shape != (spec.rows, spec.width)
                or scales.dtype != torch.float32 or scales.shape != (spec.rows, spec.width // 64)
                or codes.device != scales.device):
            raise ValueError("row64 source codes/scales do not match the typed role")
        # Packing is a load boundary, even when the outer loader uses inference
        # mode. Versioned storage allows a truthful later mutation guard.
        operation = None
        try:
            with torch.inference_mode(False):
                operation = Fp8BlockLinear.from_rows(codes, scales)
            return cls(spec, operation, chunk_rows=chunk_rows)
        finally:
            operation = None

    @property
    def device(self):
        return self._owners[0].device

    def nbytes(self) -> int:
        """Resident authoritative bytes, excluding optional GEMM activation staging."""
        return sum(x.numel() * x.element_size() for x in self._owners)

    def _check_owners(self):
        if (self.operation.w8 is not self._owners[0] or self.operation.bs is not self._owners[1]
                or tuple(x._version for x in self._owners) != self._versions):
            raise RuntimeError("row64 packed authority was mutated or replaced")

    def _read(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = s = code = scale = None
        try:
            code, scale = packed_offsets(ids, self.spec.width)
            w, s = self._owners
            return w[code], s.view(torch.float32).reshape(-1)[scale]
        finally:
            w = s = code = scale = None

    def lookup_rows(self, ids: torch.Tensor, *, max_rows: int) -> torch.Tensor:
        """IDs shape S -> BF16 shape S+(width,), with explicit request admission.

        Order and repeated IDs are preserved; max_rows bounds output/transient
        allocation before indexing. No native embedding scale is applied here.
        Each chunk's indexing temporaries are bounded by chunk_rows*width; final
        output is the admitted requested rows only. A GPU caller must retain the
        owner/output until its lane drains and use the same stream as the GEMM.
        """
        self._check_owners()
        if type(max_rows) is not int or max_rows < 0:
            raise ValueError("max_rows must be a nonnegative integer")
        if not torch.is_tensor(ids) or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("row64 IDs must be integer tensors")
        if ids.device != self.device or ids.numel() > max_rows:
            raise ValueError("row64 lookup device or requested row budget mismatch")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.spec.rows):
            raise ValueError("row64 ID outside the canonical vocabulary")
        out = torch.empty((*ids.shape, self.spec.width), dtype=torch.bfloat16, device=self.device)
        flat = ids.reshape(-1).to(torch.int64)
        dest = out.reshape(-1, self.spec.width)
        for begin in range(0, flat.numel(), self.chunk_rows):
            code, scale = self._read(flat[begin:begin + self.chunk_rows])
            dest[begin:begin + self.chunk_rows] = (code.view(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)
        return out
