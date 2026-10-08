"""Independent canonical64 native oracle with explicit supported Gluon layout.

Only symmetric native INT4/INT8 tests select this oracle.
Its exact partial/runtime qualification is recorded separately. Inputs are captured stored BF16 bits.
No production decode, chunk, dot wrapper, or merge implementation is reused.
MMA/[1,4] reductions and two kWidth4 dot sites match observed native TTGIR;
source layout declarations alone do not prove emitted layout or equality.
"""
from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

SOURCE_TEST_SHA256 = "3ac2e7e7cc05553930b87c7ccf575ec71bdb2ddfe8fb1991eae756034a7cd67a"


@gluon.jit
def operand_layout_bit_probe(X, OUT, N: gl.constexpr, ROLE: gl.constexpr):
    load_layout: gl.constexpr = BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    mma: gl.constexpr = NVMMADistributedLayout([2, 0], [1, 4], [16, 8])
    rows: gl.constexpr = 16 if ROLE == 0 or ROLE == 3 else 64
    columns: gl.constexpr = 64 if ROLE == 3 else 256
    row = gl.arange(0, rows, layout=SliceLayout(1, load_layout))
    column = gl.arange(0, columns, layout=SliceLayout(0, load_layout))
    index = gl.program_id(0) * rows * columns + row[:, None] * columns + column[None, :]
    bits = gl.load(X + index, mask=index < N, other=0)
    value = bits.to(gl.bfloat16, bitcast=True)
    if ROLE == 0:
        value = gl.convert_layout(value, DotOperandLayout(0, mma, 4))
        restored = gl.convert_layout(value, load_layout)
    elif ROLE == 1:
        value = gl.convert_layout(gl.permute(value, (1, 0)), DotOperandLayout(1, mma, 4))
        restored = gl.convert_layout(gl.permute(value, (1, 0)), load_layout)
    elif ROLE == 2:
        value = gl.convert_layout(value, DotOperandLayout(1, mma, 4))
        restored = gl.convert_layout(value, load_layout)
    else:
        value = gl.convert_layout(value, mma)
        value = gl.convert_layout(value, DotOperandLayout(0, mma, 4))
        restored = gl.convert_layout(value, load_layout)
    gl.store(OUT + index, restored.to(gl.uint16, bitcast=True), mask=index < N)


@gluon.jit
def _canonical_legacy_chunk(Q, K, V, POS, PO, PM, PL, IDS, NK, SPARSE,
                            H_: gl.constexpr, HK_: gl.constexpr, D_: gl.constexpr, G: gl.constexpr,
                            CH: gl.constexpr, NCH: gl.constexpr, IDW: gl.constexpr,
                            SCALE: gl.constexpr, QSA: gl.constexpr):
    row, head, chunk = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    n = gl.load(POS) + row + 1
    sparse = False
    if QSA:
        sparse = gl.load(SPARSE + row) != 0
        n = gl.where(sparse, gl.load(NK + row), n)
    start = chunk * CH
    if start < n:
        mma: gl.constexpr = NVMMADistributedLayout([2, 0], [1, 4], [16, 8])
        load_layout: gl.constexpr = BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
        g = gl.arange(0, 16, layout=SliceLayout(1, load_layout))
        d = gl.arange(0, D_, layout=SliceLayout(0, load_layout))
        q = gl.load(Q + (row * H_ + head * G + g[:, None]) * D_ + d[None, :],
                    mask=g[:, None] < G, other=0.)
        q = gl.convert_layout(q, DotOperandLayout(0, mma, 4))
        maximum = gl.full((16,), float("-inf"), gl.float32, layout=SliceLayout(1, mma))
        denominator = gl.zeros((16,), gl.float32, layout=SliceLayout(1, mma))
        numerator = gl.zeros((16, D_), gl.float32, layout=mma)
        for tile in range(0, gl.cdiv(gl.minimum(n - start, CH), 64)):
            position = start + tile * 64 + gl.arange(0, 64, layout=SliceLayout(1, load_layout))
            valid = position < n
            if QSA:
                if sparse:
                    position = gl.load(IDS + row * IDW + position, mask=valid, other=0)
            address = (position[:, None].to(gl.int64) * HK_ + head) * D_ + d[None, :]
            k = gl.load(K + address, mask=valid[:, None], other=0.)
            v = gl.load(V + address, mask=valid[:, None], other=0.)
            k = gl.convert_layout(gl.permute(k, (1, 0)), DotOperandLayout(1, mma, 4))
            v = gl.convert_layout(v, DotOperandLayout(1, mma, 4))
            score = mma_v2(q, k, gl.zeros((16, 64), gl.float32, layout=mma), input_precision="tf32") * SCALE
            valid = gl.convert_layout(valid, SliceLayout(0, mma))
            score = gl.where(valid[None, :], score, float("-inf"))
            tile_max = gl.max(score, 1)
            active = tile_max != float("-inf")
            next_max = gl.where(active, gl.maximum(maximum, tile_max), maximum)
            alpha = gl.where(active, gl.where(maximum == float("-inf"), 0., gl.exp(maximum - next_max)), 1.)
            probability = gl.where(valid[None, :] & active[:, None], gl.exp(score - next_max[:, None]), 0.)
            p = gl.convert_layout(probability.to(gl.bfloat16), DotOperandLayout(0, mma, 4))
            numerator = mma_v2(p, v, numerator * alpha[:, None], input_precision="tf32")
            denominator = denominator * alpha + gl.sum(probability, 1)
            maximum = next_max
        address = (row * NCH + chunk) * H_ + head * G + g
        numerator = gl.convert_layout(numerator, load_layout)
        maximum = gl.convert_layout(maximum, SliceLayout(1, load_layout))
        denominator = gl.convert_layout(denominator, SliceLayout(1, load_layout))
        gl.store(PO + address[:, None] * D_ + d[None, :], numerator, mask=g[:, None] < G)
        gl.store(PM + address, maximum, mask=g < G)
        gl.store(PL + address, denominator, mask=g < G)
