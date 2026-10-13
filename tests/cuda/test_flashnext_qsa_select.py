"""Flash Next's multi-CTA QSA selection (T, ``attention._select_multi``) and batched scoring (``_scores_multi``):
whole id/length/flag buffers equal ``_select_tiles``', ``_select``'s and an independent sort's bit for bit, on every
score bit pattern, row boundary and slice split, for windows and for rows from several streams; scores equal the
one-stream kernel's."""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from qsa_select_oracle import qsa_select_reference  # noqa: E402

from tensorfold.families.qwen4_exp.cuda import attention as att  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attn_multi  # noqa: E402

DEV = "cuda"
TOP, RATIO = 512, 4
BIG = 4 * 262144 + 8                      # 1,048,584 keys: 262,146 blocks (the production window's)


def _scratch(rows: int, capacity: int = BIG) -> att.AttnScratch:
    return att.AttnScratch(rows, 8, 64, capacity, DEV)


def _pos(ends, per_row: bool) -> torch.Tensor:
    ends = [int(e) for e in ends]
    if per_row:
        return torch.tensor([e - 1 for e in ends], dtype=torch.int32, device=DEV)
    assert ends == list(range(ends[0], ends[0] + len(ends)))
    return torch.tensor([ends[0] - 1], dtype=torch.int32, device=DEV)


def _lists(sc: att.AttnScratch, launch) -> tuple:
    """The whole id, length and flag buffers after ``launch`` from sentinels (extra or missing writes show)."""

    sc.ids.fill_(-7)
    sc.nk.fill_(-1)
    sc.sparse.fill_(-1)
    launch()
    return sc.ids.clone(), sc.nk.clone(), sc.sparse.clone()


def _oracle(sc: att.AttnScratch, ends) -> tuple:
    rows = sc.ids.shape[0]
    ids = np.full((rows, sc.idw), -7, np.int32)
    nk, sparse = np.full(rows, -1, np.int32), np.full(rows, -1, np.int32)
    qsa_select_reference(sc.scores[:len(ends)].cpu().numpy(), ends, ids, nk, sparse)
    return tuple(torch.from_numpy(a).to(DEV) for a in (ids, nk, sparse))


def _tiles(sc, pos, rows, per_row):
    att._select_tiles[(rows,)](sc.scores, pos, sc.ids, sc.nk, sc.sparse, sc.nb, RATIO=RATIO, TOP=TOP, IDW=sc.idw,
                               TB=8192, PER_ROW=per_row, num_warps=16)


def _select(sc, pos, rows, blocks, per_row):
    att._select[(rows,)](sc.scores, pos, sc.ids, sc.nk, sc.sparse, sc.nb, RATIO=RATIO, TOP=TOP, IDW=sc.idw,
                         BLOCK=triton.next_power_of_2(blocks), PER_ROW=per_row, num_warps=16)


def _same(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _check(sc, ends, *, per_row: bool, blocks: int | None = None, label=None):
    """T against the one-CTA tiles, ``_select`` (within 32,768 blocks) and the oracle, on whole buffers."""

    rows = len(ends)
    blocks = blocks or -(-max(ends) // RATIO)
    pos = _pos(ends, per_row)
    got = _lists(sc, lambda: att._select_multi(sc, pos, rows, blocks, per_row))
    assert _same(got, _lists(sc, lambda: _tiles(sc, pos, rows, per_row))), label
    if triton.next_power_of_2(blocks) <= att.SELECT_REGS:
        assert _same(got, _lists(sc, lambda: _select(sc, pos, rows, blocks, per_row))), label
    assert _same(got, _oracle(sc, ends)), label
    return got


def _patterns(real: torch.Tensor):
    """G2: real, tied, flat 0 and 1, 400 above and 1,000 at the cut, every bit pattern, negative NaNs (key 0) with
    fewer than TOP positive scores, signed zeros."""

    g = torch.Generator(device=DEV).manual_seed(17)
    yield "real", real
    yield "tied", torch.floor(real * 2) / 2
    yield "flat0", torch.zeros_like(real)
    yield "flat1", torch.ones_like(real)
    cut = torch.full_like(real, -5.0)
    order = torch.randperm(real.shape[1], generator=g, device=DEV)
    cut[:, order[:400]] = torch.linspace(1.0, 2.0, 400, device=DEV)
    cut[:, order[400:1400]] = 0.5
    yield "cut-ties", cut
    yield "bits", torch.randint(-2 ** 31, 2 ** 31, real.shape, generator=g, device=DEV,
                                dtype=torch.int64).to(torch.int32).view(torch.float32)
    nan = torch.full(real.shape, -1, dtype=torch.int32, device=DEV).view(torch.float32)   # 0xFFFFFFFF: key 0
    nan[:, order[:300]] = 3.0
    yield "key-zero", nan
    yield "signed-zeros", torch.where(torch.rand(real.shape, generator=g, device=DEV) < 0.5, 0.0, -0.0)


def _real_scores(sc: att.AttnScratch, rows: int, end: int) -> torch.Tensor:
    """Scores ``_scores`` writes for rows ending at end - rows + 1 .. end over random pooled keys."""

    g = torch.Generator(device=DEV).manual_seed(end)
    pooled = (torch.randn((sc.nb, 128), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    iq = (torch.randn((rows, 4, 128), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    sc.scores.zero_()
    att.qsa_rows(iq, pooled, torch.tensor([end - rows], dtype=torch.int32, device=DEV), sc, rows, context=end)
    return sc.scores.clone()


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 7, 20])
def test_window_boundaries_list_the_one_cta_blocks(rows):
    """G1: dense, straddling and just-sparse windows, two TOP, every end % 4, tile multiples, the register width
    (32,768 against 32,769 blocks) and the full 1,048,584-key window, on real scores."""

    sc = _scratch(rows)
    last = [2051, 2052 + rows // 2, 2055, 4096, 4 * 2048 + 1, 4 * 4096 + 3, 4 * 8192 + 2,
            4 * att.SELECT_REGS + rows // 2 + 1, 4 * att.SELECT_REGS + rows + 3, 600_002, BIG]
    for end in last:
        ends = list(range(end - rows + 1, end + 1))
        sc.scores.copy_(_real_scores(sc, rows, end))
        _check(sc, ends, per_row=False, label=(rows, end))
        _check(sc, ends, per_row=True, label=(rows, end, "per-row"))


@pytest.mark.parametrize("rows", [1, 3, 5])
def test_score_patterns_list_the_one_cta_blocks(rows):
    """G2 at a short and a long window: every pattern, through ``_select`` too where it fits."""

    sc = _scratch(rows)
    for end in (4 * 9000 + 1, 4 * 70_001 + 3):
        ends = list(range(end - rows + 1, end + 1))
        real = _real_scores(sc, rows, end)
        for name, scores in _patterns(real):
            sc.scores.copy_(scores)
            _check(sc, ends, per_row=False, label=(rows, end, name))


def test_slice_boundaries_list_the_one_cta_blocks(monkeypatch):
    """G1 at T's own splits: rows ending one block before, at and after slice and tile boundaries, with slices of
    one tile (their smallest), of two (64 slices at most), of SELECT_SPAN tiles, and longer where the slices' count
    bound wins."""

    rows = 12
    sc = _scratch(rows, 4 * 200_000)
    for sms, blocks, cw, span in ((10_000, 120_000, 64, 1), (10_000, 150_000, 64, 2),
                                  (12, 150_000, 64, att.SELECT_SPAN), (12, 150_000, 8, 10)):
        monkeypatch.setattr(sc, "sms", sms)
        monkeypatch.setattr(sc, "cw", cw)                # the placement counts' row stride: fewer slices fit
        sl, n = att._slices(sc, rows, blocks)
        assert n * sl >= blocks > (n - 1) * sl and 1 < n <= sc.cw and sl == span * att.SELECT_TB
        completes = [TOP + 1, sl - 1, sl, sl + 1, 2 * sl - 1, 2 * sl + 1, att.SELECT_TB - 1, att.SELECT_TB + 1,
                     (n - 1) * sl - 1, (n - 1) * sl + 1, blocks - 1, blocks]
        assert all(TOP < c <= blocks for c in completes)
        ends = [RATIO * c + (i % RATIO) for i, c in enumerate(completes)]
        ends[-1] = RATIO * blocks                        # the launch bound's own last block, no tail
        g = torch.Generator(device=DEV).manual_seed(sms + blocks)
        real = torch.relu(torch.randn(sc.scores.shape, generator=g, device=DEV))
        for name, scores in _patterns(real):
            sc.scores.copy_(scores)
            _check(sc, ends, per_row=True, blocks=blocks, label=(sms, blocks, cw, sl, name))


def test_rows_of_several_streams_list_each_streams_own_blocks():
    """G3: rows of a dense, a medium and a long stream in one launch, each from its own position: the lists their
    own windows' launches make, and the oracle's."""

    streams = [(2049, 2), (70_003, 3), (900_001, 5), (2051, 1), (140_000, 4)]
    ends = [e + r for e, n in streams for r in range(n)]
    rows = len(ends)
    sc = _scratch(rows)
    g = torch.Generator(device=DEV).manual_seed(3)
    for name, scores in _patterns(torch.relu(torch.randn(sc.scores.shape, generator=g, device=DEV))):
        sc.scores.copy_(scores)
        got = _check(sc, ends, per_row=True, label=name)
        a0 = 0
        for e, n in streams:                        # each stream's own window, launched alone at its offset
            alone = _lists(sc, lambda: att._select_tiles[(n,)](
                sc.scores[a0:], torch.tensor([e - 1], dtype=torch.int32, device=DEV), sc.ids[a0:], sc.nk[a0:],
                sc.sparse[a0:], sc.nb, RATIO=RATIO, TOP=TOP, IDW=sc.idw, TB=8192, num_warps=16))
            for mine, theirs in zip(got, alone):
                assert torch.equal(mine[a0:a0 + n], theirs[a0:a0 + n]), (name, e)
            a0 += n


def test_scratch_left_dirty_and_back_to_back_calls_change_nothing():
    """G4: garbage in the histograms and placement counts before a call, calls of different rows and inputs back
    to back, and rows past a launch's own (untouched)."""

    sc = _scratch(20)
    g = torch.Generator(device=DEV).manual_seed(5)
    cases = [(list(range(600_000, 600_005)), "real"), (list(range(150_000, 150_020)), "tied"),
             (list(range(300_001, 300_002)), "bits"), (list(range(2050, 2053)), "real")]
    for ends, name in cases + cases[::-1]:
        rows = len(ends)
        real = torch.relu(torch.randn(sc.scores.shape, generator=g, device=DEV))
        sc.scores.copy_(dict(_patterns(real))[name])
        want = _oracle(sc, ends)
        pos = _pos(ends, False)
        blocks = -(-ends[-1] // RATIO)
        sc.hist.copy_(torch.randint(-1000, 1000, sc.hist.shape, generator=g, device=DEV, dtype=torch.int32))
        sc.suf.copy_(torch.randint(-1000, 1000, sc.suf.shape, generator=g, device=DEV, dtype=torch.int32))
        assert _same(_lists(sc, lambda: att._select_multi(sc, pos, rows, blocks, False)), want), (ends[0], name)


def test_repeats_graph_replays_and_a_busy_stream_give_the_same_lists():
    """G5: fifty repeats; the launch sequence captured once in a CUDA graph and replayed over new positions and
    scores; a launch while another stream runs a long kernel."""

    rows, blocks = 5, 150_000
    sc = _scratch(rows, 4 * blocks)
    pos = torch.zeros((1,), dtype=torch.int32, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(9)

    def run():
        att._select_multi(sc, pos, rows, blocks, False)

    sc.scores.copy_(torch.relu(torch.randn(sc.scores.shape, generator=g, device=DEV)))
    pos.fill_(4 * blocks - rows)
    first = _lists(sc, run)
    for _ in range(50):
        assert _same(_lists(sc, run), first)
    side = torch.cuda.Stream()
    a = torch.randn((4096, 4096), device=DEV)
    side.wait_stream(torch.cuda.current_stream())    # ``a`` is written before, and kept alive while, ``side`` reads it
    with torch.cuda.stream(side):
        busy = a
        for _ in range(4):
            busy = busy @ busy.T / 64
    assert _same(_lists(sc, run), first)
    torch.cuda.synchronize()
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for end, name in ((4 * blocks, "real"), (300_003, "tied"), (2052, "bits"), (450_000, "cut-ties")):
        pos.fill_(end - rows)
        sc.scores.copy_(dict(_patterns(torch.relu(torch.randn(sc.scores.shape, generator=g, device=DEV))))[name])
        eager = _lists(sc, run)
        assert _same(_lists(sc, graph.replay), eager), (end, name)
        assert _same(eager, _oracle(sc, list(range(end - rows + 1, end + 1)))), (end, name)


@triton.jit
def _scores_before(IQ, POOLED, POS0, SC, NB, HI: tl.constexpr, DI: tl.constexpr, RATIO: tl.constexpr,
                   TOP: tl.constexpr, BB: tl.constexpr):
    """``attention._scores`` as it was before its block code moved into ``_score_rows`` (the bits it must keep)."""

    r = tl.program_id(0)
    j = tl.program_id(1)
    complete = (tl.load(POS0) + r + 1) // RATIO
    if complete > TOP and j * BB < complete:
        b = j * BB + tl.arange(0, BB)
        ok = b < complete
        d = tl.arange(0, DI)
        k = tl.load(POOLED + b[:, None].to(tl.int64) * DI + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        total = tl.zeros((BB,), dtype=tl.float32)
        for h in tl.static_range(HI):
            q = tl.load(IQ + (r * HI + h) * DI + d).to(tl.float32)
            total = total + tl.maximum(tl.sum(k * q[None, :], axis=1), 0.0)
        tl.store(SC + r * NB + b, total / tl.sqrt(DI * 1.0), mask=ok)


@pytest.mark.parametrize("blocks", [600, 12_500, 150_000])
def test_batched_scores_are_the_one_stream_kernels_bits(blocks):
    """G6: ``_scores`` keeps its bits after sharing ``_score_rows`` (window and prompt row blocks), and
    ``_scores_multi`` over several streams' pooled keys writes each row the bits its stream's launch writes."""

    g = torch.Generator(device=DEV).manual_seed(blocks)
    for rows in (1, 3, 5, 20, 64, 256):
        if rows * blocks > 2_000_000:
            continue
        sc = _scratch(rows, 4 * blocks + 8)
        pooled = (torch.randn((sc.nb, 128), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
        iq = (torch.randn((rows, 4, 128), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
        pos = torch.tensor([4 * blocks - rows], dtype=torch.int32, device=DEV)
        grid = (rows, triton.cdiv(blocks, 64))
        sc.scores.fill_(-3.0)
        att._scores[grid](iq, pooled, pos, sc.scores, sc.nb, HI=4, DI=128, RATIO=RATIO, TOP=TOP, BB=64, num_warps=4)
        now = sc.scores.clone()
        sc.scores.fill_(-3.0)
        _scores_before[grid](iq, pooled, pos, sc.scores, sc.nb, HI=4, DI=128, RATIO=RATIO, TOP=TOP, BB=64,
                             num_warps=4)
        assert torch.equal(now, sc.scores), (blocks, rows)
    # several streams: each its own pooled keys and positions, rows in stream order
    lens = [(4 * blocks - 3, 3), (2400, 2), (4 * blocks // 2 + 1, 5), (4 * blocks, 1)]
    n, rows = len(lens), sum(c for _, c in lens)
    sc = _scratch(rows, 4 * blocks + 8)
    pooled = [(torch.randn((sc.nb, 128), generator=g, device=DEV) * 0.5).to(torch.bfloat16) for _ in lens]
    iq = (torch.randn((rows, 4, 128), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    posr = torch.tensor([p + i for p, c in lens for i in range(c)], dtype=torch.int32, device=DEV)
    sid = torch.tensor([s for s, (_, c) in enumerate(lens) for _ in range(c)], dtype=torch.int32, device=DEV)
    cp = torch.zeros((attn_multi.PTRS * n,), dtype=torch.int64, device=DEV)
    cp[5 * n:6 * n] = torch.tensor([t.data_ptr() for t in pooled], dtype=torch.int64)
    most = -(-max(p + c for p, c in lens) // RATIO)
    sc.scores.fill_(-3.0)
    attn_multi._scores_multi[(rows, triton.cdiv(most, 64))](iq, cp, posr, sid, sc.scores, sc.nb, n, HI=4, DI=128,
                                                            RATIO=RATIO, TOP=TOP, BB=64, num_warps=4)
    batched = sc.scores.clone()
    sc.scores.fill_(-3.0)
    a0 = 0
    for (p, c), keys in zip(lens, pooled):
        end = p + c
        att._scores[(c, triton.cdiv(-(-end // RATIO), 64))](iq[a0:], keys, posr[a0:a0 + 1], sc.scores[a0:], sc.nb,
                                                             HI=4, DI=128, RATIO=RATIO, TOP=TOP, BB=64, num_warps=4)
        a0 += c
    assert torch.equal(batched, sc.scores), blocks


def test_the_launcher_picks_each_kernel_by_its_thresholds(monkeypatch):
    """G8: ``_select`` up to MULTI_MIN_BLOCKS and for row blocks of MULTI_ROWS or more within its registers,
    ``_select_tiles`` past them, T otherwise; too many blocks or too small a scratch fail."""

    calls = []
    monkeypatch.setattr(att, "_select_multi", lambda sc, pos, rows, blocks, per_row: calls.append(("T", rows)))

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *a, **kw: calls.append((self.name, grid[0]))

    monkeypatch.setattr(att, "_select", Kernel("select"))
    monkeypatch.setattr(att, "_select_tiles", Kernel("tiles"))
    sc = _scratch(64)
    pos = torch.zeros((1,), dtype=torch.int32, device=DEV)
    for rows, blocks, want in ((1, att.MULTI_MIN_BLOCKS, "select"), (5, att.MULTI_MIN_BLOCKS + 1, "T"),
                               (att.MULTI_ROWS - 1, 262_146, "T"), (att.MULTI_ROWS, 30_000, "select"),
                               (att.MULTI_ROWS, 40_000, "tiles"), (64, 262_146, "tiles")):
        calls.clear()
        att._launch_select(sc, pos, rows, blocks)
        assert calls == [(want, rows)], (rows, blocks)
    with pytest.raises(ValueError):
        att._launch_select(sc, pos, 1, sc.nb + 1)
    monkeypatch.undo()
    small = _scratch(3)
    with pytest.raises(ValueError):                  # the T scratch holds three rows
        att._select_multi(small, pos, 4, 262_146, False)
