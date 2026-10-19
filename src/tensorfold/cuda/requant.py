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

Target layers follow a process policy (``set_targets``): which groups of target weights load requantized. Owner: the
CUDA ``tensorfold serve`` path, which sets it from --head-bits / --gdn-bits before any weights load. Lifetime: the
process. Loaders and their byte estimates read it (``target_kind`` / ``target_copy``); the first read fixes it, after
which ``set_targets`` with other values raises, so every load in the process sees one policy. Reset: only ``using``,
a scoped override for tests that restores the previous policy, its fixed state and its record of applied groups.
Loading is single-threaded startup work: nothing here is synchronized.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

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
    """The first ``rows`` output rows of a copy, sharing its storage; e.g. the [k|v] rows of a fused [k|v|q]
    projection. Decode (``__call__``) and prompt (``prefill``) rows both run on it: ``rows`` is a multiple of 128, the
    prompt GEMM's padded output tile (decode's 64-row tiles alone would allow 64)."""

    if type(rows) is not int or not 0 < rows <= linear.n or rows % 128:
        raise ValueError(f"a requantized row prefix is a positive multiple of 128 up to {linear.n} rows")
    return linear.tiles(0, rows // 64)


class Head:
    """A requantized vocabulary head: FP32 logits [M, N] from BF16 rows, softcapped (``tanh(y / c) * c``) if asked.

    Built once per model and shared: a drafter's private proposal head and a gated target head are the same object.
    ``source`` is a BF16 head (anything with ``weight``) or an existing copy of the same kind, which is reused.
    """

    def __init__(self, source, kind: str, *, softcap: float | None = None):
        token = _checked(kind) == "fp8a8"
        if softcap is not None and (isinstance(softcap, bool) or not isinstance(softcap, (int, float))
                                    or not math.isfinite(softcap) or softcap <= 0):
            raise ValueError("a head softcap must be a positive finite number")
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


GROUPS = ("head", "gdn")    # the vocabulary head; Gated DeltaNet projections a checkpoint stores BF16
BITS = {8: "fp8"}           # --head-bits / --gdn-bits value -> kind (16: as stored)
_targets: dict[str, str] = {}
_fixed = False
_applied: set[str] = set()


def set_targets(**kinds: str) -> None:
    """``group=kind`` for each target group requantized at load (e.g. ``head="fp8"``); other groups load as stored."""

    global _targets
    unknown = sorted(set(kinds) - set(GROUPS))
    if unknown:
        raise ValueError(f"requantization groups are {GROUPS}, not {unknown}")
    for kind in kinds.values():
        _checked(kind)
    if _fixed and kinds != _targets:
        raise RuntimeError("the requantization policy is fixed once a weight load has read it")
    _targets = dict(kinds)


def target_kind(group: str) -> str | None:
    """The kind a loader requantizes ``group`` to (None: as stored); the first read fixes the policy."""

    global _fixed
    if group not in GROUPS:
        raise ValueError(f"requantization groups are {GROUPS}, not {group!r}")
    _fixed = True
    return _targets.get(group)


def target_copy(group: str, weight: torch.Tensor):
    """``requantize(weight, kind)`` when the policy names ``group`` (recorded as applied), else None."""

    kind = target_kind(group)
    if kind is None:
        return None
    copy = requantize(weight, kind)
    _applied.add(group)
    return copy


def unapplied() -> list[str]:
    """Groups the policy names that no load requantized (the checkpoint stores them otherwise, or its loader has no
    hook): the CLI refuses those rather than serve unchanged weights under a lossy flag."""

    return sorted(set(_targets) - _applied)


@contextmanager
def using(**kinds: str):
    """A policy inside the block (tests), unfixed and with nothing applied; the previous state after."""

    global _targets, _fixed, _applied
    was = _targets, _fixed, _applied
    _targets, _fixed, _applied = {}, False, set()
    try:
        set_targets(**kinds)
        yield
    finally:
        _targets, _fixed, _applied = was


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


__all__ = ["BITS", "CHUNK_ROWS", "E4M3_MAX", "GROUPS", "Head", "KINDS", "prefix", "requant_rows", "requantize",
           "resident_bytes", "set_targets", "staging_bytes", "target_copy", "target_kind", "unapplied", "using"]
