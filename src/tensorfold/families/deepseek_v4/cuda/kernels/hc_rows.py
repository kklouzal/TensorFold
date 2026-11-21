# SPDX-License-Identifier: Apache-2.0
"""TensorFold-owned row-blocked variant of the vendored V3 HC transition's first kernel (``mhc_fused._transition_parts3``;
the friend's patch 257 idea: HC parts share weight loads across rows).

``_transition_parts3`` runs one program per (token, 256-column chunk, mix group) and loads its mix group's fn
columns (3 mix rows x 4 streams x 256 FP32) for every token. Here one program handles ROWS_PER consecutive tokens
of a (chunk, group): the body is ``_transition_parts3``'s, verbatim, once per token, so every stored value is the
same expression on the same operands (bitwise the per-token kernel), while the repeated fn loads of a program hit
its L1 instead of L2 (per call: tokens / ROWS_PER fn reads instead of tokens).
"""

from .mhc_fused import _v3_new_residual
from .triton_utils import jit, tl


@jit
def _transition_parts3_rows(residual, y, post, comb, fn, cpre, cbuf, dots, squares, csquares, T,
                            MIX: tl.constexpr, H: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
                            ROWS: tl.constexpr, ROWS_PER: tl.constexpr):
    block, chunk, grp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = chunk * COLS + tl.arange(0, COLS)
    for i in tl.static_range(ROWS_PER):
        t = block * ROWS_PER + i
        for _live in range((t < T).to(tl.int32)):
            n0, n1, n2, n3 = _v3_new_residual(residual, y, post, comb, t, col, H)
            if grp == 0:
                pc = cpre + t * 4
                coll = (
                    n0 * tl.load(pc + 0)
                    + n1 * tl.load(pc + 1)
                    + n2 * tl.load(pc + 2)
                    + n3 * tl.load(pc + 3)
                )
                tl.store(cbuf + t * H + col, coll)
                tl.store(csquares + t * CHUNKS + chunk, tl.sum(coll * coll, 0))
                if MIX:
                    tl.store(
                        squares + t * CHUNKS + chunk,
                        tl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, 0),
                    )
            if MIX:
                for j in tl.static_range(ROWS):
                    r = grp * ROWS + j
                    w = fn + r * 4 * H
                    d = (
                        tl.load(w + 0 * H + col) * n0
                        + tl.load(w + 1 * H + col) * n1
                        + tl.load(w + 2 * H + col) * n2
                        + tl.load(w + 3 * H + col) * n3
                    )
                    tl.store(dots + (t * CHUNKS + chunk) * 24 + r, tl.sum(d, 0))
