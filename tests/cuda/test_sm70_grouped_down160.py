"""Unselected grouped-column WMMA candidate against an independent frozen ELF."""
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from gguf_fixture import blocks
from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.gguf_file import FORMATS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual SM70 grouped down160 qualification')


@pytest.fixture(scope='module')
def native_pair():
    path = Path(os.environ['TF_GROUPED_DOWN160_INCUMBENT_BINARY']).resolve(strict=True)
    digest = os.environ['TF_GROUPED_DOWN160_INCUMBENT_SHA256']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(path.name.split('.', 1)[0], path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    new = gguf_ops.extension()
    assert Path(new.__file__).resolve() != path
    assert hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest() != digest
    return old, new


def packed(kind, rows, width):
    if kind in (0, 1, 7, 8, 12, 13, 20, 30):
        return torch.tensor(blocks(kind, rows, width, seed=1621 + kind), device='cuda')
    with np.load(Path(__file__).parents[1] / 'fixtures/gguf_mixed_ggml.npz', allow_pickle=False) as wire:
        raw, decoded = wire[f'raw_{kind}'], wire[f'bits_{kind}'].view(np.float32)
        selected = np.flatnonzero(np.max(np.abs(decoded), axis=1) <= 4096)
        assert selected.size
        raw = np.tile(raw[selected], (-(-rows // selected.size), -(-width // decoded.shape[1])))[:rows]
        raw = raw[:, :width // FORMATS[kind][1] * FORMATS[kind][2]].copy()
    return torch.tensor(raw, device='cuda')


def operands(kinds, rows, dtype, index_dtype, offset=16, first=0, *, resident_experts=2, banks=None):
    width, dims, slots = 160, 2560, 11
    if banks is None:
        gate = packed(kinds[0], resident_experts * width, dims)
        up = packed(kinds[1], resident_experts * width, dims)
        block = FORMATS[kinds[2]][1]
        down = packed(kinds[2], resident_experts * dims, -(-(width + offset) // block) * block)
    else:
        gate, up, down = banks
    torch.manual_seed(1629)
    x = torch.randn((rows, dims), device='cuda', dtype=torch.float16) * 1e-4
    shared = torch.randn((rows, dims), device='cuda', dtype=torch.float32) * .01
    foreign = 256 if first == 0 else 0
    picks = torch.tensor([[first, first + 1, first, first + 1, first, first + 1, foreign, 512, -1, 513, first]] * rows,
                         device='cuda', dtype=index_dtype)
    return (x, picks, gate, up, down, shared), torch.empty((rows, slots, dims), device='cuda', dtype=dtype)


def execute(module, inputs, kinds, output, *, grouped=False, offset=16, graph=False, first=0):
    x, picks, gate, up, down, shared = inputs
    local = torch.empty_like(picks, dtype=torch.int64)
    activation = torch.full((picks.numel(), 160), float('nan'), device='cuda', dtype=torch.float16)
    scratch = torch.full((gguf_ops.grouped_workspace_words(picks.numel(), gate.shape[0] // 160),), -777,
                         device='cuda', dtype=torch.int32) if grouped else None
    call = module.experts_owned_grouped if grouped else module.experts_owned
    args = (x, picks, gate, kinds[0], 0, up, kinds[1], 0, down, kinds[2], offset,
            160, first, 512, shared, local, activation, output)
    if grouped:
        args += (scratch,)
    call(*args)
    capture = None
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            call(*args)
    return local, activation, scratch, capture


def bits(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def check_groups(local, scratch, resident_experts=2):
    local = local.flatten().cpu().tolist()
    p, e = len(local), resident_experts
    cap = min(p, (p + 15) // 16 + e - 1)
    words = scratch.cpu().tolist()
    counts, offsets = words[:e], words[e:2 * e + 1]
    order = words[2 * e + 1 + p:2 * e + 1 + 2 * p]
    tiles = words[2 * e + 1 + 2 * p:2 * e + 1 + 2 * p + 3 * cap]
    total = words[-1]
    assert counts == [local.count(expert) for expert in range(e)]
    assert offsets == [sum(counts[:i]) for i in range(e + 1)]
    assert total == sum((n + 15) // 16 for n in counts)
    covered = []
    for tile in range(total):
        expert, start, count = tiles[3 * tile:3 * tile + 3]
        assert 0 <= expert < e and offsets[expert] <= start < offsets[expert + 1]
        assert start + count <= offsets[expert + 1]
        members = order[start:start + count]
        assert 1 <= count <= 16 and all(local[i] == expert for i in members)
        covered.extend(members)
    assert sorted(covered) == [i for i, expert in enumerate(local) if expert >= 0]


# All actual target pairs/MTP plus every remaining supported-format fallback.
KINDS = [(18, 18, 20), (18, 18, 42), (21, 21, 20), (21, 21, 42),
         (22, 22, 20), (22, 22, 42), (23, 23, 20), (8, 8, 8),
         (0, 0, 0), (1, 1, 1), (12, 13, 7), (13, 12, 12), (14, 14, 14), (30, 30, 30), (23, 23, 23), (21, 21, 21),
         (13, 13, 13), (18, 18, 18), (22, 22, 22)]


@pytest.mark.parametrize('kinds', KINDS)
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
def test_grouped_actual_and_fallback_formats_preserve_frozen_slot_bits(native_pair, kinds, dtype, index_dtype):
    inputs, actual = operands(kinds, 17, dtype, index_dtype)
    expected = torch.empty_like(actual)
    old = execute(native_pair[0], inputs, kinds, expected)
    new = execute(native_pair[1], inputs, kinds, actual, grouped=True)
    bits(actual, expected)
    bits(new[0], old[0])
    bits(new[1], old[1])
    check_groups(new[0], new[2])
    assert torch.count_nonzero(actual[inputs[1].eq(256)].view(torch.uint8)) == 0
    assert torch.isnan(actual[inputs[1].lt(0) | inputs[1].gt(512)]).all()


@pytest.mark.parametrize('rows', [1, 15, 16, 17, 31, 32, 33, 64, 256])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_skewed_ragged_groups_reuse_weights_without_changing_each_dot(native_pair, rows, dtype):
    kinds = (21, 21, 42)
    inputs, actual = operands(kinds, rows, dtype, torch.int64, offset=160)
    inputs[1].fill_(1)
    expected = torch.empty_like(actual)
    execute(native_pair[0], inputs, kinds, expected, offset=160)
    result = execute(native_pair[1], inputs, kinds, actual, grouped=True, offset=160)
    bits(actual, expected)
    check_groups(result[0], result[2])


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('first', [0, 256])
def test_graph_route_replay_foreign_shared_bad_and_empty_groups(native_pair, dtype, first):
    kinds = (22, 22, 20)
    inputs, actual = operands(kinds, 3, dtype, torch.int64, first=first)
    expected = torch.empty_like(actual)
    held = execute(native_pair[1], inputs, kinds, actual, grouped=True, graph=True, first=first)
    foreign = 256 if first == 0 else 0
    for ids in ([first] * 11, [foreign] * 11, [512] * 11, [-1] * 11, [513] * 11,
                [first, first + 1, first + 1, first, first + 1, first, foreign, 512, -1, 513, 2**40]):
        inputs[1].copy_(torch.tensor([ids] * 3, device='cuda'))
        actual.fill_(19.)
        held[1].fill_(float('nan'))
        held[2].fill_(-777)
        held[3].replay()
        execute(native_pair[0], inputs, kinds, expected, first=first)
        bits(actual, expected)
        check_groups(held[0], held[2])


def test_grouping_workspace_validation_precedes_submission(native_pair):
    kinds = (18, 18, 42)
    inputs, out = operands(kinds, 1, torch.float32, torch.int64)
    x, picks, gate, up, down, shared = inputs
    local = torch.empty_like(picks)
    act = torch.empty((picks.numel(), 160), device='cuda', dtype=torch.float16)
    words = gguf_ops.grouped_workspace_words(picks.numel(), 2)
    args = (x, picks, gate, kinds[0], 0, up, kinds[1], 0, down, kinds[2], 16,
            160, 0, 512, shared, local, act, out)
    for scratch, error in [(torch.empty(words - 1, device='cuda', dtype=torch.int32), 'layout'),
                           (out.view(torch.int32).flatten()[:words], 'overlap')]:
        with pytest.raises(RuntimeError, match=error):
            native_pair[1].experts_owned_grouped(*args, scratch)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_column_positions_keep_subnormal_signed_zero_and_cancellation_bits(native_pair, dtype):
    kinds = (1, 1, 1)
    original, actual = operands(kinds, 33, dtype, torch.int64)
    x, picks, _, _, _, shared = original
    x.zero_()
    x[:, 0].copy_(torch.tensor(([1., -1., 0., -0., .5, 2.] * 6)[:33], device='cuda'))
    picks.fill_(0)
    gate = torch.zeros((320, 2560), device='cuda', dtype=torch.float16)
    up = torch.zeros_like(gate)
    gate[:, 0] = 1.
    up[:, 0].copy_(torch.tensor(([2**-24, 2**-14, 1., -1.] * 80), device='cuda', dtype=torch.float16))
    down = torch.zeros((5120, 176), device='cuda', dtype=torch.float16)
    pattern = torch.tensor(([1024., 2**-10, -1024., 1.] * 40), device='cuda', dtype=torch.float16)
    down[:, 16:176].copy_(pattern.expand(5120, 160))
    inputs = (x, picks, gate.view(torch.uint8), up.view(torch.uint8), down.view(torch.uint8), shared)
    expected = torch.empty_like(actual)
    old = execute(native_pair[0], inputs, kinds, expected)
    new = execute(native_pair[1], inputs, kinds, actual, grouped=True)
    assert torch.any((old[1].abs() > 0) & (old[1].abs() < 2**-14))
    bits(new[1], old[1])
    bits(actual, expected)
    check_groups(new[0], new[2])


@pytest.mark.parametrize('offset', [0, 320, 480])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_all_etp_column_origins_keep_frozen_exact_dot(native_pair, offset, dtype):
    kinds = (21, 21, 42)
    inputs, actual = operands(kinds, 17, dtype, torch.int64, offset=offset)
    expected = torch.empty_like(actual)
    execute(native_pair[0], inputs, kinds, expected, offset=offset)
    execute(native_pair[1], inputs, kinds, actual, grouped=True, offset=offset)
    bits(actual, expected)


@pytest.fixture(scope='module', params=[(18, 18, 20), (22, 22, 42)])
def resident_256_banks(request):
    """Two actual projection tuples; prior107 cover the remaining value decoders."""
    kinds = request.param
    offset = 16
    block = FORMATS[kinds[2]][1]
    banks = (packed(kinds[0], 256 * 160, 2560), packed(kinds[1], 256 * 160, 2560),
             packed(kinds[2], 256 * 2560, -(-(160 + offset) // block) * block))
    return kinds, offset, banks


@pytest.mark.parametrize('rows', [5, 30, 40, 160, 256])
@pytest.mark.parametrize('first', [0, 256])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_actual_256_expert_prefix_and_slot_bits(native_pair, resident_256_banks, rows, first, dtype):
    kinds, offset, banks = resident_256_banks
    inputs, actual = operands(kinds, rows, dtype, torch.int32, offset=offset, first=first,
                              resident_experts=256, banks=banks)
    expected = torch.empty_like(actual)
    foreign = 256 if first == 0 else 0
    top = [0, 127, 255, 1, 2, 31, 63, 64, 128, 254]
    unique = [[first + (expert + row * 13) % 256 for expert in top] + [512] for row in range(rows)]
    # Deep bin255 has >256 arrivals at largeR; sparse bins exercise all257
    # prefix offsets and multiple tiles per bin. Duplicate slots are an explicit
    # low-level native contract stress, separate from the unique router case.
    skew = [[first + 255] * 7 + [first, foreign, 512, 513] for _ in range(rows)]
    sparse = [[first, first + 127, first + 255, foreign, 512, -1, 513,
               first + 127, foreign, 2**31 - 1, first + 255] for _ in range(rows)]
    patterns = [unique, skew, sparse]
    if rows in (5, 256):
        patterns += [[[foreign] * 11 for _ in range(rows)], [[512] * 11 for _ in range(rows)]]
    for ids in patterns:
        inputs[1].copy_(torch.tensor(ids, device='cuda', dtype=torch.int32))
        actual.fill_(19.)
        old = execute(native_pair[0], inputs, kinds, expected, offset=offset, first=first)
        new = execute(native_pair[1], inputs, kinds, actual, grouped=True, offset=offset, first=first)
        bits(actual, expected)
        bits(new[0], old[0])
        bits(new[1], old[1])
        check_groups(new[0], new[2], resident_experts=256)
        picks = inputs[1]
        assert torch.count_nonzero(actual[picks.eq(foreign)].view(torch.uint8)) == 0
        assert torch.isnan(actual[picks.lt(0) | picks.gt(512)]).all()
        shared = inputs[-1][:, None, :].expand_as(actual).to(dtype)
        bits(actual[picks.eq(512)], shared[picks.eq(512)])
