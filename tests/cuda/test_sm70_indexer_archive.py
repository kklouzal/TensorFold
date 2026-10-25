"""Native ring/staging indexer contracts against unchanged full-GPU preparation/pooling."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import family_native

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 staged indexer")


POOL_CASES = [(d, ratio, rows, offset) for d in (64, 128, 256) for ratio in (1, 4, 8)
              for rows in (1, 3, 8, 2048) for offset in sorted({0, ratio - 1})]
FORMATS = [(0, 0, 0, 0), (8, 8, 0, 0), (0, 8, 0, 0), (8, 0, 0, 0),
           (4, 4, 4, 4), (4, 4, 5, 5), (4, 4, 4, 5), (4, 4, 5, 4),
           (4, 8, 4, 0), (8, 4, 0, 5), (0, 4, 0, 4), (4, 0, 5, 0)]


def pool_case(dim, ratio, rows, base, *, capacity=None, ring_rows=None, offset=0):
    torch.manual_seed(311 + dim + ratio + rows)
    capacity = base + rows + 17 if capacity is None else capacity
    ring_rows = max(ratio + 8, 8) if ring_rows is None else ring_rows
    raw = torch.randn((capacity, dim), device="cuda", dtype=torch.float16) * 3
    stage = torch.full((offset + rows + 4, dim), 7, device="cuda", dtype=torch.float16)
    stage[offset:offset + rows].copy_(raw[base:base + rows])
    ring = torch.full((ring_rows, dim), 7, device="cuda", dtype=torch.float16)
    past = torch.arange(max(0, base - ring_rows), base, device="cuda")
    ring.index_copy_(0, past % ring_rows, raw.index_select(0, past))
    pooled = torch.full(((capacity + ratio - 1) // ratio, dim), 7, device="cuda", dtype=torch.float16)
    weight = torch.randn(dim, device="cuda") * 0.1 + 1
    frequency = torch.tensor([1e7 ** (-2 * i / 64) for i in range(min(32, dim // 2))], device="cuda")
    position = torch.tensor([base], device="cuda", dtype=torch.int64)
    delta = torch.tensor([2], device="cuda", dtype=torch.int32)
    status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)
    live = torch.tensor([rows], device="cuda", dtype=torch.int32)
    origin = torch.tensor([base - offset], device="cuda", dtype=torch.int64) if offset else None
    return SimpleNamespace(raw=raw, stage=stage, ring=ring, pooled=pooled, weight=weight, frequency=frequency,
                           position=position, delta=delta, status=status, live=live, origin=origin,
                           capacity=capacity, ratio=ratio, rows=rows, base=base, dim=dim)


def pool_new(case):
    family_native.extension().pool_staged(case.stage, case.ring, case.pooled, case.position, case.weight,
                                         case.frequency, case.delta, case.ratio, case.rows, case.capacity,
                                         1e-6, 1.0, case.status, case.live, 9, case.origin)


def publish(case):
    family_native.extension().ring_publish(case.stage, case.ring, case.position, case.rows, case.capacity,
                                          case.status, case.live, 9, case.origin)


def pool_old(case, out, *, rows=None):
    family_native.extension().pool(case.raw, out, case.position, case.weight, case.frequency, case.delta,
                                  case.ratio, case.rows if rows is None else rows, 1e-6, 1.0)


@pytest.mark.parametrize("dim,ratio,rows,remainder", POOL_CASES)
def test_staged_pool_and_unique_tail_match_full_history_across_wraps(dim, ratio, rows, remainder):
    case = pool_case(dim, ratio, rows, 37 * ratio + remainder, offset=2)
    expected_pool = case.pooled.clone()
    expected_ring = case.ring.clone()
    pool_old(case, expected_pool)
    pool_new(case)
    # Pooling must consume carry before any modulo tail publication, including
    # the 2048-row pass that is much larger than the bounded ring.
    torch.testing.assert_close(case.pooled, expected_pool, rtol=0, atol=0)
    publish(case)
    retained = min(rows, case.ring.shape[0])
    tail = torch.arange(case.base + rows - retained, case.base + rows, device="cuda")
    expected_ring.index_copy_(0, tail % case.ring.shape[0], case.raw.index_select(0, tail))
    torch.testing.assert_close(case.ring, expected_ring, rtol=0, atol=0)
    assert case.status.tolist() == [0, -2]


@pytest.mark.parametrize("capacity", [8, 32, 64, 65, 262147])
def test_pool_preserves_original_logical_reduction_mode_with_bounded_grid(capacity):
    rows, ratio, dim = 4, 4, 256
    case = pool_case(dim, ratio, rows, min(19, capacity - rows), capacity=capacity, ring_rows=8)
    expected = case.pooled.clone()
    pool_old(case, expected)
    pool_new(case)
    torch.testing.assert_close(case.pooled, expected, rtol=0, atol=0)
    assert case.status.tolist() == [0, -2]


def test_captured_one_row_mtp_steps_retain_all_origin_relative_raw_rows():
    case = pool_case(128, 4, 1, 100, capacity=512, ring_rows=16)
    case.stage = torch.full((8, case.dim), 7, device="cuda", dtype=torch.float16)
    case.stage.copy_(case.raw[100:108])
    case.origin = torch.tensor([100], device="cuda", dtype=torch.int64)
    expected = case.pooled.clone()
    pool_new(case)
    publish(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool_new(case)
        publish(case)
    for start in range(100, 108):
        case.position.fill_(start)
        pool_old(case, expected, rows=1)
        graph.replay()
    torch.testing.assert_close(case.pooled, expected, rtol=0, atol=0)
    tail = torch.arange(100, 108, device="cuda")
    torch.testing.assert_close(case.ring.index_select(0, tail % 16), case.raw[100:108], rtol=0, atol=0)
    assert case.status.tolist() == [0, -2]


@pytest.mark.parametrize("change", ["position_negative", "position_huge", "position_exhausted", "live_negative",
                                    "live_large", "origin_negative", "origin_future", "origin_exhausted"])
def test_bad_dynamic_controls_poison_status_without_cache_publication(change):
    case = pool_case(128, 4, 4, 19, capacity=64, offset=2)
    if change == "position_negative":
        case.position.fill_(-1)
    elif change == "position_huge":
        case.position.fill_(2**63 - 1)
    elif change == "position_exhausted":
        case.position.fill_(63)
    elif change.startswith("live"):
        case.live.fill_(-1 if change.endswith("negative") else 5)
    elif change == "origin_negative":
        case.origin.fill_(-1)
    elif change == "origin_future":
        case.origin.fill_(20)
    else:
        case.origin.fill_(0)
    before = case.ring.clone()
    pool_new(case)
    publish(case)
    assert case.status.tolist() == [16, 9]
    assert (case.pooled == 7).all()
    torch.testing.assert_close(case.ring, before, rtol=0, atol=0)


@pytest.mark.parametrize("alias", ["stage_ring", "pool_stage", "pool_ring"])
def test_overlapping_writable_layouts_are_rejected_before_launch(alias):
    case = pool_case(128, 4, 4, 3, capacity=32, ring_rows=8)
    if alias == "stage_ring":
        case.ring = case.stage
    elif alias == "pool_stage":
        case.pooled = case.stage
    else:
        case.pooled = case.ring
    with pytest.raises(RuntimeError):
        pool_new(case)


def prep_case(bits, codecs, *, rows=3, live=3, offset=2, dim=256, base=19, capacity=64):
    torch.manual_seed(1211 + rows)
    qh, kh, ih, di = 3, 1, 4, 128
    p = torch.randn((rows, 2 * qh * dim + 2 * kh * dim + (ih + 1) * di), device="cuda", dtype=torch.float16)
    weights = [torch.randn(size, device="cuda") * 0.1 + 1 for size in (dim, dim, di)]
    freq = torch.tensor([1e7 ** (-2 * i / 64) for i in range(min(32, dim // 2))], device="cuda")
    pos = torch.tensor([base], device="cuda", dtype=torch.int64)
    delta = torch.tensor([2], device="cuda", dtype=torch.int32)
    active = torch.tensor([live], device="cuda", dtype=torch.int32)

    def outputs(stage):
        q = torch.full((rows, qh, dim), 7, device="cuda", dtype=torch.float16)
        iq = torch.full((rows, ih, di), 7, device="cuda", dtype=torch.float16)
        index = torch.full((offset + rows + 4 if stage else capacity, di), 7, device="cuda", dtype=torch.float16)
        values = []
        for width in bits:
            payload = torch.full((capacity, kh, dim // 2 if width == 4 else dim), 7, device="cuda",
                                 dtype={0: torch.float16, 8: torch.int8, 4: torch.uint8}[width])
            scales = None if not width else torch.full((capacity, kh, dim // (128 if width == 4 else 32)), 7,
                                                       device="cuda", dtype=torch.float32 if width == 4 else torch.float16)
            values.append((payload, scales))
        status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)
        args = [p, pos, *weights, freq, q, values[0][0], values[1][0], iq, index,
                values[0][1], values[1][1], status, delta, active, qh, kh, dim, ih, di, *bits, 9,
                1e-6, 1.0, False]
        if stage or any(codecs):
            args.extend(codecs)
        if stage:
            args.append(torch.tensor([base - offset], device="cuda", dtype=torch.int64))
        return SimpleNamespace(args=args, q=q, iq=iq, index=index, k=values[0][0], v=values[1][0],
                               ks=values[0][1], vs=values[1][1], status=status)

    return SimpleNamespace(p=p, pos=pos, active=active, full=outputs(False), stage=outputs(True),
                           rows=rows, offset=offset, base=base, di=di)


@pytest.mark.parametrize("kb,vb,kc,vc", FORMATS)
@pytest.mark.parametrize("rows", [1, 3, 8, 2048])
def test_staged_preparation_preserves_queries_kv_and_all_raw_indexer_rows(kb, vb, kc, vc, rows):
    case = prep_case((kb, vb), (kc, vc), rows=rows, live=rows, capacity=rows + 32)
    native = family_native.extension()
    (native.prep_q4 if kc or vc else native.prep)(*case.full.args)
    native.prep_staged(*case.stage.args)
    assert case.stage.status.tolist() == case.full.status.tolist() == [0, -2]
    for expected, actual in [(case.full.q, case.stage.q), (case.full.iq, case.stage.iq),
                             (case.full.k, case.stage.k), (case.full.v, case.stage.v)]:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in ("ks", "vs"):
        if getattr(case.full, name) is not None:
            torch.testing.assert_close(getattr(case.stage, name), getattr(case.full, name), rtol=0, atol=0)
    torch.testing.assert_close(case.stage.index[2:2 + rows], case.full.index[19:19 + rows], rtol=0, atol=0)
    assert (case.stage.index[:2] == 7).all() and (case.stage.index[2 + rows:] == 7).all()


@pytest.mark.parametrize("format", [(0, 0, 0, 0), (8, 8, 0, 0), (4, 4, 4, 4), (4, 4, 5, 5)])
def test_staged_preparation_ignores_nan_padding_without_overwriting_stage(format):
    kb, vb, kc, vc = format
    case = prep_case((kb, vb), (kc, vc), rows=4, live=2)
    case.p[2:].fill_(float("nan"))
    native = family_native.extension()
    (native.prep_q4 if kc or vc else native.prep)(*case.full.args)
    native.prep_staged(*case.stage.args)
    assert case.stage.status.tolist() == [0, -2]
    torch.testing.assert_close(case.stage.q[:2], case.full.q[:2], rtol=0, atol=0)
    torch.testing.assert_close(case.stage.index[2:4], case.full.index[19:21], rtol=0, atol=0)
    assert (case.stage.index[4:] == 7).all()


@pytest.mark.parametrize("kb,vb,kc,vc", [(8, 8, 0, 0), (4, 4, 4, 4), (4, 4, 5, 5)])
def test_captured_preparation_retains_all_chained_mtp_rows_at_device_origin(kb, vb, kc, vc):
    case = prep_case((kb, vb), (kc, vc), rows=1, live=1, offset=0, capacity=64)
    case.stage.index = torch.full((8, case.di), 7, device="cuda", dtype=torch.float16)
    case.stage.args[10] = case.stage.index
    native = family_native.extension()
    native.prep_staged(*case.stage.args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.prep_staged(*case.stage.args)
    for index in range(8):
        case.pos.fill_(19 + index)
        case.p.fill_(0.125 * (index + 1))
        (native.prep_q4 if kc or vc else native.prep)(*case.full.args)
        graph.replay()
    torch.testing.assert_close(case.stage.index, case.full.index[19:27], rtol=0, atol=0)
    assert case.stage.status.tolist() == [0, -2]


@pytest.mark.parametrize("format", [(0, 0, 0, 0), (8, 8, 0, 0), (4, 4, 4, 4), (4, 4, 5, 5)])
@pytest.mark.parametrize("change", ["origin_negative", "origin_future", "origin_exhausted", "position_huge", "live_large"])
def test_invalid_preparation_frame_cannot_write_stage_or_kv(format, change):
    kb, vb, kc, vc = format
    case = prep_case((kb, vb), (kc, vc), rows=3)
    if change == "origin_negative":
        case.stage.args[-1].fill_(-1)
    elif change == "origin_future":
        case.stage.args[-1].fill_(20)
    elif change == "origin_exhausted":
        case.stage.args[-1].fill_(0)
    elif change == "position_huge":
        case.pos.fill_(2**63 - 1)
    else:
        case.active.fill_(4)
    case.p.fill_(float("nan"))
    family_native.extension().prep_staged(*case.stage.args)
    assert case.stage.status.tolist() == [16, 9]
    assert all((t == 7).all() for t in (case.stage.q, case.stage.k, case.stage.v, case.stage.iq, case.stage.index))


@pytest.mark.parametrize("change", ["status_missing", "status_shape", "origin_dtype", "origin_shape", "ratio_zero",
                                    "ring_short", "stage_short", "logical_negative", "epsilon_range",
                                    "epsilon_underflow", "rotary_range", "rotary_underflow"])
def test_staged_pool_boundary_plan_is_checked_before_launch(change):
    case = pool_case(128, 4, 4, 3, capacity=32, offset=2)
    args = [case.stage, case.ring, case.pooled, case.position, case.weight, case.frequency, case.delta,
            case.ratio, case.rows, case.capacity, 1e-6, 1.0, case.status, case.live, 9, case.origin]
    if change == "status_missing":
        args[12] = None
    elif change == "status_shape":
        args[12] = case.status[:1]
    elif change == "origin_dtype":
        args[15] = case.origin.int()
    elif change == "origin_shape":
        args[15] = torch.zeros(2, device="cuda", dtype=torch.int64)
    elif change == "ratio_zero":
        args[7] = 0
    elif change == "ring_short":
        args[1] = case.ring[:3]
    elif change == "stage_short":
        args[0] = case.stage[:3]
    elif change == "logical_negative":
        args[9] = -1
    else:
        args[10 if change.startswith("epsilon") else 11] = 1e300 if change.endswith("range") else 1e-300
    with pytest.raises(RuntimeError):
        family_native.extension().pool_staged(*args)


def test_empty_staged_frame_performs_no_pool_or_ring_write():
    case = pool_case(128, 4, 0, 3, capacity=32)
    before = case.ring.clone()
    pool_new(case)
    publish(case)
    assert case.status.tolist() == [0, -2]
    assert (case.pooled == 7).all()
    torch.testing.assert_close(case.ring, before, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("bits", [(0, 0), (8, 8)])
def test_raw_and_int8_small_heads_keep_existing_preparation_contract(dim, bits):
    case = prep_case(bits, (0, 0), rows=3, dim=dim)
    native = family_native.extension()
    native.prep(*case.full.args)
    native.prep_staged(*case.stage.args)
    torch.testing.assert_close(case.stage.q, case.full.q, rtol=0, atol=0)
    torch.testing.assert_close(case.stage.index[2:5], case.full.index[19:22], rtol=0, atol=0)
    assert case.stage.status.tolist() == [0, -2]


def test_large_ring_preserves_carry_when_a_full_staged_pass_commits_only_one_row():
    case = pool_case(128, 4, 2048, 2, capacity=4096, ring_rows=2052)
    expected = case.pooled.clone()
    pool_old(case, expected)
    pool_new(case)
    publish(case)
    # The owner accepts only row2, so the next producer resumes at3. Historical
    # rows0/1 plus accepted row2 must survive provisional publication of2048rows.
    case.raw[3].fill_(0.25)
    case.position.fill_(3)
    case.rows = 1
    case.live.fill_(1)
    case.origin = None
    case.stage = case.raw[3:4].clone()
    pool_old(case, expected, rows=1)
    pool_new(case)
    publish(case)
    torch.testing.assert_close(case.pooled, expected, rtol=0, atol=0)
    assert case.status.tolist() == [0, -2]
