"""Strict native GDN State batching against a separately pinned incumbent ELF."""
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sm70 import gdn_native
from tensorfold.families.qwen4_exp.cuda.gdn import GDNScratch, widths
from tensorfold.families.qwen4_exp.cuda.gdn_multi import RoundTables, NativeTables

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='actual SM70 and frozen incumbent GDN required')


@pytest.fixture(scope='module')
def pair():
    path = Path(os.environ['TF_GDN_BATCH_INCUMBENT_BINARY']).resolve(strict=True)
    digest = os.environ['TF_GDN_BATCH_INCUMBENT_SHA256']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(path.name.split('.', 1)[0], path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    new = gdn_native.extension()
    assert Path(new.__file__).resolve() != path
    assert hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest() != digest
    return old, new


def exact(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def setup(nv, pattern):
    nk, layers, width, slots = nv//3, 2, 3, 24
    channels, pw = widths(nk, nv)
    gen = torch.Generator(device='cuda').manual_seed(725+nv)
    def half(shape):
        return torch.randn(shape, dtype=torch.float16, device='cuda', generator=gen)*.1
    states = []
    for slot in range(slots):
        state = NS(conv=half((layers, 3, channels)),
                   rec=torch.randn((2, layers, nv, 128, 128), device='cuda', generator=gen)*.01,
                   cur=[slot%2, 1-slot%2], scratch=[])
        for _ in range(layers):
            scratch = GDNScratch(width, 'cuda', nk, nv, dtype=torch.float16)
            for name in ('k', 'v', 'g', 'b'):
                getattr(scratch, name).fill_(-77)
            if pattern == 'masked':
                scratch.keep_rows = torch.tensor([slot%4], device='cuda', dtype=torch.int32)
            state.scratch.append(scratch)
        states.append(state)
    rows = slots*width
    b = NS(h=half((rows, 10240)), proj=half((layers, rows, pw)),
           gout=torch.full((rows, nv*128), -31., device='cuda', dtype=torch.float16),
           gxs=torch.full((rows, nv*4), -17., device='cuda'), rows=rows, prefill=False,
           split=None, owner_graph_family='uncaptured')
    if pattern == 'cancellation':
        values = torch.tensor([0., -0., 2**-24, -2**-24, .5, -.5, 1., -1.],
                              device='cuda', dtype=torch.float16)
        b.proj.copy_(values.repeat((b.proj.numel()+7)//8)[:b.proj.numel()].view_as(b.proj))
        for state in states:
            state.rec[..., ::2] *= -1
    static = RoundTables(NS(device=torch.device('cuda')), NS(lin=layers), slots, width, native_half=True)
    static.bind_native(states, b)
    weight, norm = half((channels, 4)), torch.ones(128, device='cuda', dtype=torch.float16)
    bias, log, neg = torch.randn(nv, device='cuda', generator=gen)*.1, torch.zeros(nv, device='cuda'), -torch.ones(nv, device='cuda')
    return static, states, b, weight, norm, bias, log, neg


def frame(static, states, b, count, iteration=0):
    # Permute actual resident slots; unused bins/State owners remain unchanged.
    chosen = list(reversed(states))[iteration:]+list(reversed(states))[:iteration]
    chosen = chosen[:count]
    end, segs = 0, []
    for index, state in enumerate(chosen):
        rows = 1+(index+iteration)%3
        segs.append((state, end, end+rows))
        end += rows
    if count == 1:
        assert static.native_frame(segs, b) is None  # Actual caller keeps eager C1.
        records = [next(row for row in static.native_states if row[0] is chosen[0])]
        return NativeTables(static, segs, b, records)
    return static.native_frame(segs, b)


def oracle(old, table, b, layer, weight, norm, bias, decay, direct):
    expected_rec = {id(state): state.rec.clone() for state, _, _ in table.segs}
    expected_save = {}
    output, sums = b.gout.clone(), b.gxs.clone()
    for state, begin, end in table.segs:
        current = state.cur[layer]
        scratch = state.scratch[layer]
        saved = NS(**{name: getattr(scratch, name).clone() for name in ('k', 'v', 'g', 'b')})
        expected_save[id(state)] = saved
        keep = getattr(scratch, 'keep_rows', None)
        old.chain(b.proj[layer, begin:end], state.conv[layer], weight, expected_rec[id(state)][current, layer],
                  decay, bias, norm, 1e-6, end-begin, output[begin:end], sums[begin:end],
                  expected_rec[id(state)][1-current, layer], saved.k, saved.v, saved.g, saved.b, keep, direct)
    return expected_rec, expected_save, output, sums


def compare(table, b, layer, expected):
    rec, saved, output, sums = expected
    exact(b.gout, output)
    exact(b.gxs, sums)
    for state, _, _ in table.segs:
        exact(state.rec, rec[id(state)])
        for name in ('k', 'v', 'g', 'b'):
            exact(getattr(state.scratch[layer], name), getattr(saved[id(state)], name))


@pytest.mark.parametrize('count', [1, 2, 4, 8, 16, 24])
@pytest.mark.parametrize('pattern', ['random', 'cancellation', 'masked'])
@pytest.mark.parametrize('direct', [False, True])
def test_actual_NV6_activity_ragged_rows_parity_masks_and_saved_bits(pair, count, pattern, direct):
    old, new = pair
    new.prepare_batch(6)
    static, states, b, weight, norm, bias, log, neg = setup(6, pattern)
    conv = [state.conv.clone() for state in states]
    for iteration in range(2):
        table = frame(static, states, b, count, iteration)
        for layer in range(2):
            decay = neg if direct else log
            expected = oracle(old, table, b, layer, weight, norm, bias, decay, direct)
            gdn_native.batch_chain(b.proj[layer, :table.rows], weight, decay, bias, norm, 1e-6,
                                   table, layer, b.gout[:table.rows], b.gxs[:table.rows], negative_decay=direct)
            compare(table, b, layer, expected)
        for state, _, _ in table.segs:
            state.cur = [1-value for value in state.cur]
            state.generation = iteration+1
        b.proj.mul_(-1)
    for state, original in zip(states, conv):
        exact(state.conv, original)


@pytest.mark.parametrize('nv', [12, 24, 48])
@pytest.mark.parametrize('direct', [False, True])
def test_other_existing_head_variants_keep_original_program(pair, nv, direct):
    old, new = pair
    new.prepare_batch(nv)
    static, states, b, weight, norm, bias, log, neg = setup(nv, 'masked')
    table = frame(static, states, b, 2)
    decay = neg if direct else log
    expected = oracle(old, table, b, 0, weight, norm, bias, decay, direct)
    gdn_native.batch_chain(b.proj[0, :table.rows], weight, decay, bias, norm, 1e-6,
                           table, 0, b.gout[:table.rows], b.gxs[:table.rows], negative_decay=direct)
    compare(table, b, 0, expected)


def test_current_stream_replay_reads_changed_owners_parity_and_mask(pair):
    old, new = pair
    new.prepare_batch(6)
    static, states, b, weight, norm, bias, _, decay = setup(6, 'masked')
    original, stream = torch.cuda.current_stream(), torch.cuda.Stream()
    stream.wait_stream(original)
    primary, errors, transport = None, [None, None], [None]
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            # Use one row each so the graph's original launch dimensions stay fixed.
            segs = [(state, i, i+1) for i, state in enumerate(states[:24])]
            table = static.native_frame(segs, b)
            gdn_native.batch_chain(b.proj[0, :24], weight, decay, bias, norm, 1e-6,
                                   table, 0, b.gout[:24], b.gxs[:24], negative_decay=True)
            with torch.cuda.graph(graph, stream=stream):
                gdn_native.batch_chain(b.proj[0, :24], weight, decay, bias, norm, 1e-6,
                                       table, 0, b.gout[:24], b.gxs[:24], negative_decay=True)
            for iteration in range(3):
                for state in states:
                    state.cur[0] ^= 1
                    state.scratch[0].keep_rows.fill_(iteration%2)
                segs = [(state, i, i+1) for i, state in enumerate(reversed(states))]
                table = static.native_frame(segs, b)
                expected = oracle(old, table, b, 0, weight, norm, bias, decay, True)
                graph.replay()
                compare(table, b, 0, expected)
    except BaseException as error:
        primary = error
    finally:
        try:
            torch.cuda.synchronize()
        except BaseException as error:
            errors[0] = error
        if errors[0] is None:
            try:
                graph.reset()
            except BaseException as error:
                errors[1] = error
        try:
            failures = [error for error in errors if error is not None]
        except BaseException as error:
            transport[0] = error
            raise_failures(primary, transport)
        raise_failures(primary, failures)


@pytest.mark.parametrize('nv', [6, 12, 24, 48])
@pytest.mark.parametrize('direct', [False, True])
def test_retained_eager_wrapper_matches_frozen_incumbent(pair, nv, direct):
    old, new = pair
    static, states, b, weight, norm, bias, log, neg = setup(nv, 'masked')
    table = frame(static, states, b, 2)
    decay = neg if direct else log
    expected = oracle(old, table, b, 0, weight, norm, bias, decay, direct)
    actual = oracle(new, table, b, 0, weight, norm, bias, decay, direct)
    for index in (2, 3):
        exact(actual[index], expected[index])
    for state, _, _ in table.segs:
        exact(actual[0][id(state)], expected[0][id(state)])
        for name in ('k', 'v', 'g', 'b'):
            exact(getattr(actual[1][id(state)], name), getattr(expected[1][id(state)], name))


@pytest.mark.parametrize('target', ['output', 'sums'])
@pytest.mark.parametrize('descriptor_row', [0, 7])
def test_accessed_strided_descriptor_words_cannot_alias_outputs(pair, target, descriptor_row):
    _, new = pair
    static, states, b, weight, norm, bias, _, decay = setup(6, 'random')
    table = frame(static, states, b, 2)
    pointers, starts = table.operands(0, b.proj[0, :table.rows], b.gout[:table.rows], b.gxs[:table.rows])
    storage = torch.zeros(16384, device='cuda', dtype=torch.uint8)
    aliased = storage.view(torch.int64).as_strided((8, 2), (64, 1))
    aliased.copy_(pointers)
    begin = descriptor_row*64*8
    output = b.gout[:table.rows]
    sums = b.gxs[:table.rows]
    if target == 'output':
        output = storage[begin:begin+table.rows*768*2].view(torch.float16).view(table.rows, 768)
    else:
        sums = storage[begin:begin+table.rows*24*4].view(torch.float32).view(table.rows, 24)
    before = storage.clone()
    with pytest.raises(RuntimeError, match=r'alias(?:es)? an active operand'):
        new.batch_chain(b.proj[0, :table.rows], weight, decay, bias, norm, 1e-6,
                        aliased, starts, output, sums, True)
    exact(storage, before)


def test_output_in_unread_descriptor_gap_is_legal_and_matches_frozen_bits(pair):
    old, new = pair
    static, states, b, weight, norm, bias, _, decay = setup(6, 'random')
    table = frame(static, states, b, 2)
    pointers, starts = table.operands(0, b.proj[0, :table.rows], b.gout[:table.rows], b.gxs[:table.rows])
    expected = oracle(old, table, b, 0, weight, norm, bias, decay, True)
    storage = torch.zeros(8*1024*8, device='cuda', dtype=torch.uint8)
    strided = storage.view(torch.int64).as_strided((8, 2), (1024, 1))
    strided.copy_(pointers)
    # After the first two accessed words, before the next accessed row. The
    # helper compares actual rows; the unused row-gap envelope is not read.
    output = storage[16:16+table.rows*768*2].view(torch.float16).view(table.rows, 768)
    output.fill_(-31)
    saved = strided.clone()
    new.batch_chain(b.proj[0, :table.rows], weight, decay, bias, norm, 1e-6,
                    strided, starts, output, b.gxs[:table.rows], True)
    exact(output, expected[2][:table.rows])
    exact(b.gxs, expected[3])
    exact(strided, saved)
    for state, _, _ in table.segs:
        exact(state.rec, expected[0][id(state)])
        for name in ('k', 'v', 'g', 'b'):
            exact(getattr(state.scratch[0], name), getattr(expected[1][id(state)], name))
