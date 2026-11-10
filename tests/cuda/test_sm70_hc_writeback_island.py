"""Strict current-SM70 HC writeback/PSS island against the retained Torch program."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='actual SM70 HC island qualification required')


@pytest.fixture(scope='module')
def prepared():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip('SM70 island')
    return ops.prepare_owned_hc_writeback(torch.device('cuda', torch.cuda.current_device()))


def buffers(rows):
    return NS(rows=rows, h=torch.empty((rows, 10240), dtype=torch.float16, device='cuda'),
        streams=torch.empty((rows, 10240), dtype=torch.float16, device='cuda'),
        pss=torch.empty((rows, 10, 4), device='cuda'), inj_a=torch.empty((rows, 4), dtype=torch.float16, device='cuda'),
        inj_m=torch.empty((rows, 4), dtype=torch.float16, device='cuda'),
        g_branch=torch.empty(8*rows*2560, device='cuda'), g_moe=torch.empty(8*rows*2560, device='cuda'))


def bits_equal(got, expected):
    dtype = torch.int16 if got.dtype == torch.float16 else torch.int32
    assert torch.equal(got.view(dtype), expected.view(dtype))


@pytest.mark.parametrize('mode', [0, 3])
def test_every_row_count_matches_original_half_and_pss_bits(prepared, mode):
    b = buffers(256)
    cap = prepared.bind(b)
    gen = torch.Generator(device='cuda').manual_seed(621)
    initial = torch.randn(b.h.shape, device='cuda', dtype=torch.float16, generator=gen) * .2
    b.g_branch.normal_(generator=gen)
    b.inj_a.copy_(torch.randn(b.inj_a.shape, device='cuda', dtype=torch.float16, generator=gen))
    # The entire1..256 owner domain matters to ATen's output-count-dependent reduction program.
    for rows in range(1, 257):
        h, pss = b.h[:rows], b.pss[:rows]
        h.copy_(initial[:rows])
        pss.fill_(torch.nan)
        branch = b.g_branch[:8*rows*2560].view(8, rows, 2560) if mode else None
        inject = b.inj_a[:rows] if mode else None
        expected, sums = h.clone(), torch.empty_like(pss)
        ops._reference['hc_writeback'](expected, expected, sums, 4, mode, branch=branch, inject=inject)
        hv, pv = h._version, pss._version
        assert cap.run(h, h, pss, 4, mode, branch=branch, inject=inject)
        assert h._version == hv + 1 and pss._version == pv + 1
        bits_equal(h, expected)
        bits_equal(pss, sums)


def test_mode0_preserves_every_half_bit_including_snan_payload(prepared):
    b = buffers(7)
    cap = prepared.bind(b)
    words = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
    raw = words.repeat(2)[:b.h.numel()].reshape_as(b.h)
    b.h.copy_(raw.view(torch.float16))
    expected, pss = b.h.clone(), torch.empty_like(b.pss)
    ops._reference['hc_writeback'](expected, expected, pss, 4, 0)
    assert cap.run(b.h, b.h, b.pss, 4, 0)
    assert torch.equal(b.h.view(torch.int16), raw)
    bits_equal(b.h, expected)
    bits_equal(b.pss, pss)


@pytest.mark.parametrize('pattern', ['cancellation', 'rounding', 'overflow', 'nonfinite'])
def test_ordered_tp8_cast_and_special_value_boundaries(prepared, pattern):
    b = buffers(5)
    cap = prepared.bind(b)
    b.h.zero_()
    b.inj_a.fill_(1)
    branch = b.g_branch.view(8, 5, 2560)
    branch.zero_()
    if pattern == 'cancellation':
        branch[0].fill_(2**24)
        branch[1].fill_(1)
        branch[2].fill_(-2**24)
        branch[3].fill_(.500244140625)
        branch[4].fill_(-.25)
    elif pattern == 'rounding':
        branch[0].copy_(torch.linspace(-2, 2, 2560, device='cuda').expand(5, -1))
        b.inj_a.copy_(torch.tensor([1., .333251953125, -0., -1.], device='cuda', dtype=torch.float16))
    elif pattern == 'overflow':
        branch[0].fill_(torch.finfo(torch.float32).max)
        branch[1].fill_(torch.finfo(torch.float32).max)
        b.inj_a[:, 1] = 0
    else:
        branch[0, :, :3] = torch.tensor([float('inf'), -float('inf'), float('nan')], device='cuda')
        branch[1, :, :2] = torch.tensor([-float('inf'), float('inf')], device='cuda')
    expected, pss = b.h.clone(), torch.empty_like(b.pss)
    ops._reference['hc_writeback'](expected, expected, pss, 4, 3, branch=branch, inject=b.inj_a)
    assert cap.run(b.h, b.h, b.pss, 4, 3, branch=branch, inject=b.inj_a)
    bits_equal(b.h, expected)
    bits_equal(b.pss, pss)


@pytest.mark.parametrize('inference', [False, True])
def test_current_stream_output_consumption_and_mutation_domain(prepared, inference):
    with torch.inference_mode(inference):
        b = buffers(3)
        cap = prepared.bind(b)
        source = torch.ones_like(b.h)
        stream, current = torch.cuda.Stream(), torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            b.h.copy_(source)
            assert cap.run(b.h, b.h, b.pss, 4, 0)
            consumed = b.pss.clone()
        event = torch.cuda.Event()
        event.record(stream)
        current.wait_event(event)
        assert torch.all(consumed == 256)
        bits_equal(b.h, source)


@pytest.mark.parametrize('alias', ['partial', 'pss', 'branch', 'inject'])
def test_raw_native_refuses_unsafe_output_read_aliases_before_write(prepared, alias):
    b = buffers(8)
    b.h.zero_()
    b.inj_a.fill_(1)
    b.g_branch.zero_()
    h, out, pss, mode = b.h[:3], b.h[:3], b.pss[:3], 3
    branch, inject = b.g_branch[:8*3*2560].view(8, 3, 2560), b.inj_a[:3]
    if alias == 'partial':
        h, out, pss = b.h[:2], b.h[1:3], b.pss[:2]
        branch, inject = b.g_branch[:8*2*2560].view(8, 2, 2560), b.inj_a[:2]
    elif alias == 'pss':
        pss = h.view(torch.float32).reshape(-1)[:120].view(3, 10, 4)
    elif alias == 'branch':
        branch = b.h.view(torch.float32).reshape(-1)[:8*2560].view(8, 1, 2560)
        h, out, pss, inject = h[:1], out[:1], pss[:1], inject[:1]
    else:
        inject = h.reshape(-1)[:12].view(3, 4)
    before = out.view(torch.int16).clone()
    with pytest.raises(RuntimeError):
        prepared.module.writeback(h, out, pss, 4, mode, branch, inject, None, None)
    assert torch.equal(out.view(torch.int16), before)
