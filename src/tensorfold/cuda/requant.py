"""Load-time requantized copies of BF16 weights: one path for drafter proposals and for gated target layers.

A drafter's projections and its copy of the target head never change output (the target verifies every token), so they
may run requantized; target layers (heads, later other BF16 projections) may only behind the lossy-class quality gates.
Both use this module, which builds copies on the existing kernels of ``tensorfold.cuda.nvfp4.linear``:

  ``fp8``    E4M3 codes and one FP32 scale an output row, ``s_n = max_k |W[n, k]| / 448`` (an all-zero row keeps 1),
             codes ``e4m3(W / s_n)`` round to nearest even, saturating: ModelOpt's FP8_PER_CHANNEL layout, run as
             W8A16 (BF16 rows, each channel's scale in the epilogue) by ``Fp8Linear``.
  ``fp8a8``  the same codes, rows quantized a token at a time onto the E4M3 tensor-core MMA (``Fp8Linear(token)``).

Every copy is quantized in row chunks so its FP32 working set stays bounded (a vocabulary head has 0.25M rows); the
caller may free the source afterwards. ``resident_bytes`` / ``staging_bytes`` give admission its bytes.
"""

from __future__ import annotations

import torch

from tensorfold.cuda.nvfp4.linear import FRAGMENT_ROWS, Fp8Linear

E4M3_MAX = 448.0
KINDS = ("fp8", "fp8a8")
CHUNK_ROWS = 16384


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


def requantize(weight: torch.Tensor, kind: str):
    """A ``kind`` copy of a CUDA floating [N, K] weight (K a multiple of 64): an ``Fp8Linear``.

    Its ``__call__(x)`` maps BF16 rows [M, K] to BF16 [M, N] (FP32 sums into an FP32 ``out``), ``prefill(x)`` takes
    prompt rows on the prompt GEMM, ``tiles(t0, t1)`` is a view of output rows [64 t0, 64 t1).
    """

    token = _checked(kind) == "fp8a8"
    codes, scales = requant_rows(weight)
    return Fp8Linear.from_checkpoint(codes, scales, token=token)


def prefix(linear, rows: int):
    """The first ``rows`` output rows of a copy (a multiple of 64: the copy's tile), sharing its storage; e.g. the
    [k|v] rows of a fused [k|v|q] projection. Decode (``__call__``) and prompt (``prefill``) rows both run on it."""

    if type(rows) is not int or not 0 < rows <= linear.n or rows % 64:
        raise ValueError(f"a requantized row prefix is a positive multiple of 64 up to {linear.n} rows")
    return linear.tiles(0, rows // 64)


class Head:
    """A requantized vocabulary head: FP32 logits [M, N] from BF16 rows, softcapped (``tanh(y / c) * c``) if asked.

    Built once per model and shared: a drafter's private proposal head and a gated target head are the same object.
    ``source`` is a BF16 head (anything with ``weight``) or an existing copy of the same kind, which is reused.
    """

    def __init__(self, source, kind: str, *, softcap: float | None = None):
        token = _checked(kind) == "fp8a8"
        if softcap is not None and not (isinstance(softcap, (int, float)) and softcap > 0):
            raise ValueError("a head softcap must be a positive number")
        if isinstance(source, Head):
            source = source.linear
        if isinstance(source, Fp8Linear):
            if source.cols is None or source.token != token:
                raise ValueError("a requantized head is shared as the same kind; other kinds start from BF16")
            self.linear = source
        else:
            self.linear = requantize(source.weight, kind)
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
    """Device bytes a ``kind`` copy of an [N, K] weight keeps: codes over 128-padded rows and an FP32 scale a row."""

    _checked(kind)
    padded = -(-n // 128) * 128
    return padded * k + padded * 4


def staging_bytes(kind: str, n: int, k: int) -> int:
    """Transient device bytes building a ``kind`` copy: FP32 row chunks (two live), the codes, then the chunked
    fragment-order gather (a padded byte copy and two int64 indices an element)."""

    _checked(kind)
    return min(n, CHUNK_ROWS) * k * 9 + n * k + min(n, FRAGMENT_ROWS) * k * 17


__all__ = ["CHUNK_ROWS", "E4M3_MAX", "Head", "KINDS", "prefix", "requant_rows", "requantize", "resident_bytes",
           "staging_bytes"]
