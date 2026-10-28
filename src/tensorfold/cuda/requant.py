"""Load-time requantized copies of BF16 weights: one path for drafter proposals and for gated target layers.

A drafter's projections and its copy of the target head never change output (the target verifies every token), so they
may run requantized; target layers (heads, later other BF16 projections) may only behind the lossy-class quality gates.
Both use this module, which builds copies on the existing kernels of ``tensorfold.cuda.nvfp4.linear``:

  ``fp8``    E4M3 codes and one FP32 scale an output row, ``s_n = max_k |W[n, k]| / 448`` (an all-zero row keeps 1),
             codes ``e4m3(W / s_n)`` round to nearest even, saturating: ModelOpt's FP8_PER_CHANNEL layout, run as
             W8A16 (BF16 rows, each channel's scale in the epilogue) by ``Fp8Linear``.
  ``nvfp4``  E2M1 codes with an E4M3 scale per 16 inputs and one FP32 global scale in ModelOpt's NVFP4 layout, run as
             W4A16 by ``Fp4Linear``: global = max|W| / (6 x 448), block scale e4m3(max_block / 6 / global), codes the
             nearest E2M1 magnitude of W / (block scale x global), ties to the smaller. This is the recipe Flash Next's
             draft experts always used, not ModelOpt's bit for bit: ModelOpt's NVFP4QTensor (producer commit 87c9f8cf)
             divides by 6 x global, stores 1 for an all-zero block, clamps block scales to [2**-9, 448] and rounds the
             ties at 0.75, 1.75 and 3.5 up. ``nvfp4_rows`` is the one NVFP4 weight quantizer
             (``tensorfold.cuda.nvfp4.experts.quantize`` applies it per expert).

Every copy is quantized in row chunks so its FP32 working set stays bounded (a vocabulary head has 0.25M rows); the
caller may free the source afterwards. ``resident_bytes`` / ``staging_bytes`` give admission its bytes.
"""

from __future__ import annotations

import math

import torch

from tensorfold.cuda.nvfp4.linear import FRAGMENT_ROWS, Fp4Linear, Fp8Linear

E4M3_MAX = 448.0
KINDS = ("fp8", "nvfp4")     # W8A8 (rows a token at a time on the E4M3 MMA) lost the decode SELECT: not offered
CHUNK_ROWS = 16384
NVFP4_CHUNK_ROWS = 4096     # rows nvfp4_rows quantizes at a time (<= 18.4 bytes an element of temporaries; staging)


def _checked(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"requantization kind must be one of {KINDS}")
    return kind


def requant_rows(weight: torch.Tensor, chunk: int = CHUNK_ROWS) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scales)`` of a floating [N, K] weight: E4M3 codes [N, K] and FP32 scales [N] (one an output row)."""

    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point() \
            or weight.dtype == torch.float8_e4m3fn:
        raise ValueError("FP8 requantization takes a floating [N, K] weight")
    n, k = weight.shape
    codes = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((n,), dtype=torch.float32, device=weight.device)
    for start in range(0, n, chunk):
        w = weight[start:start + chunk].float()
        # max |w| a row in one pass, as max(hi, -lo); aminmax and maximum propagate NaN (documented) and an infinite
        # weight makes its row's top infinite, so ``top`` alone shows a non-finite chunk
        lo, hi = torch.aminmax(w, dim=1)
        top = torch.maximum(hi, -lo)
        if not bool(torch.isfinite(top).all()):
            raise ValueError("FP8 requantization needs finite weights")
        scale = torch.where(top > 0, top / E4M3_MAX, torch.ones_like(top))
        # w / scale lies in [-448, 448] up to the division's rounding; clamping keeps that rounding from overflowing
        codes[start:start + chunk] = (w / scale[:, None]).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        scales[start:start + chunk] = scale
    return codes, scales


def nvfp4_rows(weight: torch.Tensor, chunk: int = NVFP4_CHUNK_ROWS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(codes, scales, global)`` of a floating [N, K] weight (K a multiple of 16): E2M1 codes two a byte (low nibble
    first) [N, K/2] uint8, E4M3 block scales [N, K/16] as uint8 bytes, and the FP32 global scale (a 0-d tensor).

    Every value is computed in FP32 in the recipe's order (module docstring), so a chunked call equals a whole one.
    """

    from tensorfold.cuda.nvfp4.format import E2M1  # the 16 E2M1 values by code; codes 0..7 are magnitudes

    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point() \
            or weight.dtype == torch.float8_e4m3fn or weight.shape[1] % 16:
        raise ValueError("NVFP4 requantization takes a floating [N, K] weight with K a multiple of 16")
    n, k = weight.shape
    top = torch.zeros((), dtype=torch.float32, device=weight.device)
    for start in range(0, n, chunk):
        w = weight[start:start + chunk].float()
        if not bool(torch.isfinite(w).all()):
            raise ValueError("NVFP4 requantization needs finite weights")
        top = torch.maximum(top, w.abs().amax())
    g = (top / (6.0 * E4M3_MAX)).clamp_min(1e-30)
    mags = torch.from_numpy(E2M1[:8]).to(weight.device)
    mids = (mags[1:] + mags[:-1]) / 2                       # E2M1 decision points: ties go to the smaller magnitude
    codes = torch.empty((n, k // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((n, k // 16), dtype=torch.uint8, device=weight.device)
    for start in range(0, n, chunk):
        blocks = weight[start:start + chunk].float().view(-1, k // 16, 16)
        s = (blocks.abs().amax(-1) / 6.0 / g).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
        step = (s.float() * g)[..., None]
        v = torch.where(step > 0, blocks / step.clamp_min(1e-30), 0.0)
        del blocks
        negative = (v < 0).view(-1, k)
        code = torch.bucketize(v.abs_(), mids, out_int32=True).view(-1, k).to(torch.uint8)
        del v
        code |= negative.to(torch.uint8) << 3                # the sign bit above the three magnitude bits
        codes[start:start + chunk] = code[:, 0::2] | (code[:, 1::2] << 4)
        scales[start:start + chunk] = s.view(torch.uint8).view(-1, k // 16)
    return codes, scales, g


def requantize(weight: torch.Tensor, kind: str):
    """A ``kind`` copy of a CUDA floating [N, K] weight (K a multiple of 64): an ``Fp8Linear`` or (nvfp4) ``Fp4Linear``.

    Its ``__call__(x)`` maps BF16 rows [M, K] to BF16 [M, N] (FP32 sums into an FP32 ``out``), ``prefill(x)`` takes
    prompt rows on the prompt GEMM, ``tiles(t0, t1)`` is a view of output rows [64 t0, 64 t1).
    """

    if _checked(kind) == "nvfp4":
        codes, scales, g = nvfp4_rows(weight)
        return Fp4Linear.from_checkpoint(codes, scales, float(g))
    codes, scales = requant_rows(weight)
    return Fp8Linear.from_checkpoint(codes, scales)


def prefix(linear, rows: int):
    """The first ``rows`` output rows of a copy, sharing its storage; e.g. the [k|v] rows of a fused [k|v|q]
    projection. Decode (``__call__``) and prompt (``prefill``) rows both run on it: ``rows`` is a multiple of 128, the
    prompt GEMM's padded output tile (decode's 64-row tiles alone would allow 64)."""

    if type(rows) is not int or not 0 < rows <= linear.n or rows % 128:
        raise ValueError(f"a requantized row prefix is a positive multiple of 128 up to {linear.n} rows")
    return linear.tiles(0, rows // 64)


class Head:
    """A requantized vocabulary head: FP32 logits [M, N] from BF16 rows, softcapped (``tanh(y / c) * c``) if asked.

    Built once per model and shared: a drafter's private proposal head and a gated target head are the same object.
    ``source`` is a floating head (anything with a floating ``weight``) or an existing copy of the same kind, which is
    reused; a head the checkpoint stores quantized is refused (a caller borrows it as it is instead).
    """

    def __init__(self, source, kind: str, *, softcap: float | None = None):
        _checked(kind)
        if softcap is not None and (isinstance(softcap, bool) or not isinstance(softcap, (int, float))
                                    or not math.isfinite(softcap) or softcap <= 0):
            raise ValueError("a head softcap must be a positive finite number")
        if isinstance(source, Head):
            source = source.linear
        if isinstance(source, (Fp8Linear, Fp4Linear)):
            same = (source.act is None and kind == "nvfp4" if isinstance(source, Fp4Linear) else
                    source.cols is not None and not source.token and source.act is None and kind == "fp8")
            if not same:
                raise ValueError("a requantized head is shared as the same kind; other kinds start from BF16")
            self.linear = source
        else:
            weight = getattr(source, "weight", None)
            if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
                raise ValueError(f"a requantized head copies a floating head, not {type(source).__name__}")
            self.linear = requantize(weight, kind)
        self.softcap, self.n = softcap, self.linear.n

    def nbytes(self) -> int:
        return self.linear.nbytes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], self.n), dtype=torch.float32, device=x.device)
        logits = self.linear(x, out)
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


def resident_bytes(kind: str, n: int, k: int) -> int:
    """Device bytes a ``kind`` copy of an [N, K] weight keeps: codes over 128-padded rows and an FP32 scale a row
    (fp8), or half-byte codes and an E4M3 scale per 16 inputs (nvfp4)."""

    _checked(kind)
    padded = -(-n // 128) * 128
    return padded * k // 2 + padded * k // 16 if kind == "nvfp4" else padded * k + padded * 4


def staging_bytes(kind: str, n: int, k: int) -> int:
    """Transient device bytes building a ``kind`` copy. fp8: FP32 row chunks (two live), the codes, then the chunked
    fragment-order gather (a padded byte copy and two int64 indices an element); nvfp4: below."""

    if _checked(kind) == "nvfp4":
        # nvfp4_rows' codes and scales beside either its chunk temporaries (FP32 blocks, quotients and their where,
        # int32 bucket indices, byte codes: 12.4 bytes an element, plus the previous chunk's codes and signs and the
        # finiteness pass's last FP32 chunk: <= 18.4 measured) or qmm.pack's 4096-row chunk temporaries (int32
        # nibbles, the gathered fragment, two int64 shifts: <= 25 bytes an element) with its tiled output and the
        # block-scale relayout (two scale copies).
        padded, rows = -(-n // 128) * 128, min(n, NVFP4_CHUNK_ROWS)
        return n * k // 2 + n * k // 16 + max(rows * k * 19, min(padded, 4096) * k * 25 + padded * k // 2
                                                               + 2 * padded * k // 16)
    return min(n, CHUNK_ROWS) * k * 9 + n * k + min(n, FRAGMENT_ROWS) * k * 17


__all__ = ["CHUNK_ROWS", "E4M3_MAX", "KINDS", "NVFP4_CHUNK_ROWS", "Head", "nvfp4_rows", "prefix", "requant_rows",
           "requantize", "resident_bytes", "staging_bytes"]
