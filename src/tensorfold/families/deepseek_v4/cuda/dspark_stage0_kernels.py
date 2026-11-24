"""Exact Stage 0 normalization pairs for the pinned SM70 DSpark stack."""

from .kernels.triton_utils import jit, tl


@jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                    dtype=tl.float32, is_pure=True, pack=1)


@jit
def _div_rn(a, b):
    # Triton div_rn can lower to a divide that flushes subnormal operands.
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                    dtype=tl.float32, is_pure=True, pack=1)


@jit
def _last_axis_four(values):
    even, odd = tl.split(tl.reshape(values, (4, 2, 2)))
    a, c = tl.split(even)
    b, d = tl.split(odd)
    return a, b, c, d


@jit
def _normalize_loop(start, output, LOOP: tl.constexpr, EPSILON: tl.constexpr):
    # The pinned ATen kernel reduces contiguous rows with shuffle offsets 2 and 1,
    # and strided columns with an ascending lane-local sum.
    i = tl.arange(0, 4)
    at = tl.program_id(0) * 16 + i[:, None] * 4 + i[None, :]
    value = tl.load(start + at)
    eps = tl.full((), EPSILON, tl.float32)
    for _ in range(LOOP):
        a, b, c, d = _last_axis_four(value)
        row = _add_rn(_add_rn(a, c), _add_rn(b, d))
        value = _div_rn(value, _add_rn(row, eps)[:, None])
        a, b, c, d = _last_axis_four(tl.trans(value))
        col = _add_rn(_add_rn(_add_rn(a, b), c), d)
        value = _div_rn(value, _add_rn(col, eps)[None, :])
    tl.store(output + at, value)
