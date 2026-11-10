"""Active GPU wire bytes versus the independent committed CPU producer."""

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import onecat_gpu_packing as gpu, onecat_moe_packing as cpu
from tests.test_onecat_moe_packing import raw_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='active GU packing requires actual SM70')


def outputs(kind, device):
    facts = gpu.layout(kind)
    slab = torch.full((facts['pair_plane_bytes'],), 0xa5, device=device, dtype=torch.uint8)
    c, s = np.prod(facts['codes_shape']), np.prod(facts['scales_shape'])
    c, s = int(c), int(s)
    return slab, (slab[:c].view(facts['codes_shape']), slab[c:c+s].view(facts['scales_shape']),
                  slab[c+s:2*c+s].view(facts['codes_shape']), slab[2*c+s:].view(facts['scales_shape']))


def raw_input(kind, *, offset=256, strided=True, distinct=False):
    block = cpu.BLOCK_BYTES[kind]
    source = raw_rows(kind, 640, 11 if offset else 10)
    blocks = source.reshape(640, -1, block)
    # The original archive repeats every8rows: vary BOTH metadata and codes
    # so incorrect32-row tile/lane addressing cannot pass by periodicity.
    row_scales = ((np.arange(640) % 61 + 1) * 2.**-10).astype('<f2').view(np.uint8).reshape(640, 2)
    blocks[:, :, :2] = row_scales[:, None, :]
    blocks[:, :, 8 if kind == 23 else 2] ^= (np.arange(640) % 251).astype(np.uint8)[:, None]
    selected = source[320:480]
    if distinct:
        values = np.broadcast_to(selected, (256, *selected.shape)).copy()
        # Unique expert byte in EVERY block, including nonzero-offset K.
        # Keep independent row metadata/code markers above intact.
        fields = values.reshape(256, 160, -1, block)
        fields[:, :, :, 9 if kind == 23 else 3] ^= np.arange(256, dtype=np.uint8)[:, None, None]
        tensor = torch.from_numpy(values).cuda()
        if strided:
            padded = torch.full((40960, values.shape[-1]+17), 191, dtype=torch.uint8, device=tensor.device)
            padded[:, :values.shape[-1]].copy_(tensor.view(40960, -1))
            tensor = padded[:, :values.shape[-1]]
        return tensor, lambda expert: values[expert, :, offset//256*block:offset//256*block+10*block].copy()
    if strided:
        padded = torch.full((320, selected.shape[-1]+17), 191, dtype=torch.uint8, device='cuda')
        tensor = padded[::2, :selected.shape[-1]]
        tensor.copy_(torch.from_numpy(selected))
    else:
        tensor = torch.from_numpy(selected.copy()).cuda()
    raw = tensor.unsqueeze(0).expand(256, -1, -1)
    return raw, lambda expert: selected[:, offset//256*block:offset//256*block+10*block].copy()


def picks_for(rows, first, dtype=torch.int32):
    picks = torch.arange(first, first+11, device='cuda', dtype=dtype).repeat(rows, 1)
    picks[:, 10] = 512
    if rows > 1:
        picks[1, :10] = torch.tensor([first+255, first+127, *range(first+12, first+20)], device='cuda', dtype=dtype)
    return picks


def active_ids(picks, first):
    active = set()
    for row in picks.cpu().tolist():
        if row[-1] != 512 or any(x < 0 or x >= 512 for x in row[:10]) or len(set(row)) != 11:
            continue
        active.update(x-first for x in row[:10] if first <= x < first+256)
    return sorted(active)


def compare(planes, expected, active, kind):
    for expert in active:
        with np.errstate(invalid='ignore', over='ignore'):
            _, codes, scales = cpu._pack_expert(expected(expert), kind)
        for actual, original in ((planes[0][expert], codes), (planes[1][expert], scales)):
            np.testing.assert_array_equal(actual.cpu().numpy(), original)
    inactive = [e for e in range(256) if e not in active]
    if inactive:
        for plane in planes:
            assert bool(torch.all(plane[inactive] == 0xa5))


@pytest.mark.parametrize('kind,first,rows,distinct,dtype', [
    (18, 0, 1, False, torch.int32), (21, 256, 3, True, torch.int64),
    (22, 256, 24, False, torch.int32), (23, 0, 32, True, torch.int64)])
@torch.no_grad()
def test_all_four_formats_nonzero_offset_strided_active_reused_foreign_and_bad_rows(kind, first, rows, distinct, dtype):
    device = torch.device('cuda', torch.cuda.current_device())
    prepared = gpu.prepare(device)
    gate, expected = raw_input(kind, distinct=distinct)
    up, expected_up = raw_input(kind, distinct=distinct)
    slab, planes = outputs(kind, device)
    picks = picks_for(rows, first, dtype)
    if rows > 3:
        picks[2, :10] = torch.arange(256-first, 266-first, device=device, dtype=dtype)  # valid foreign
        picks[3, 1] = picks[3, 0]  # entire row invalid; no bank activation
        picks[4, 10] = 511
        picks[5, 0] = -1
        picks[6, 0] = 1 << 30
    before_gate, before_up = gate.clone(), up.clone()
    versions = [p._version for p in planes]
    result = prepared.pack_active(gate, up, picks, *planes, first=first, ggml_type=kind,
                                  gate_column_offset=256, up_column_offset=256)
    torch.cuda.synchronize()
    active = active_ids(picks, first)
    compare(planes[:2], expected, active, kind)
    compare(planes[2:], expected_up, active, kind)
    assert all(a is b for a, b in zip(result, planes))
    assert [p._version for p in planes] == [v+4 for v in versions]  # all four disjoint views share one version owner
    assert torch.equal(gate, before_gate) and torch.equal(up, before_up)
    assert slab.numel() == gpu.layout(kind)['pair_plane_bytes']


@pytest.mark.parametrize('kind', [18, 21, 22, 23])
@torch.no_grad()
def test_empty_or_invalid_rows_never_touch_any_plane(kind):
    prepared = gpu.prepare(torch.device('cuda', torch.cuda.current_device()))
    raw, _ = raw_input(kind, offset=0)
    slab, planes = outputs(kind, raw.device)
    picks = picks_for(4, 256)
    picks[1, 1] = picks[1, 0]
    picks[2, 10] = 7
    picks[3, 0] = -1
    prepared.pack_active(raw, raw, picks, *planes, first=0, ggml_type=kind)
    torch.cuda.synchronize()
    assert bool(torch.all(slab == 0xa5))


@torch.no_grad()
def test_capture_nondefault_stream_replays_changed_active_experts_and_retains_all_owners():
    device = torch.device('cuda', torch.cuda.current_device())
    prepared = gpu.prepare(device)
    raw, expected = raw_input(18, offset=0)
    slab, planes = outputs(18, device)
    picks = picks_for(2, 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.pack_active(raw, raw, picks, *planes, first=0, ggml_type=18)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prepared.pack_active(raw, raw, picks, *planes, first=0, ggml_type=18)
    stream.synchronize()
    slab.fill_(0xa5)
    picks[:, :10] += 31
    graph.replay()
    torch.cuda.synchronize()
    compare(planes[:2], expected, active_ids(picks, 0), 18)
    del graph  # completed above, and all borrowed Tensor/module owners remain live


@pytest.mark.parametrize('fault', ['alias', 'shape', 'offset', 'negative', 'dtype'])
@torch.no_grad()
def test_native_refusals_precede_every_plane_write(fault):
    device = torch.device('cuda', torch.cuda.current_device())
    prepared = gpu.prepare(device)
    raw, _ = raw_input(18, offset=0)
    slab, planes = outputs(18, device)
    picks = picks_for(1, 0)
    values = list(planes)
    offset = 0
    if fault == 'alias':
        values[2] = values[0]
    elif fault == 'shape':
        values[0] = values[0][:-1]
    elif fault == 'offset':
        offset = 256
    elif fault == 'negative':
        raw = torch._neg_view(raw)
    else:
        picks = picks.float()
    with pytest.raises((ValueError, RuntimeError)):
        prepared.pack_active(raw, raw, picks, *values, first=0, ggml_type=18, gate_column_offset=offset)
    torch.cuda.synchronize()
    assert bool(torch.all(slab == 0xa5))


@torch.no_grad()
def test_IQ4_scale_half_classes_match_pinned_CPU_wire_including_payloads():
    device = torch.device('cuda', torch.cuda.current_device())
    prepared = gpu.prepare(device)
    raw, _ = raw_input(23, offset=0, strided=False)
    bits = np.array([0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff,
                     0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e11, 0xfe11], dtype='<u2')
    original = raw[0].cpu().numpy().copy().reshape(160, 10, 136)
    for row in range(160):
        original[row, :, :2] = bits[row % len(bits):row % len(bits)+1].view(np.uint8)
        six = [0, 31, 32, 63, 1, 33, 62, 16]
        high = sum((value >> 4) << (2*i) for i, value in enumerate(six))
        original[row, :, 2:4] = np.array([high], dtype='<u2').view(np.uint8)
        original[row, :, 4:8] = np.array([(six[2*i]&15) | ((six[2*i+1]&15)<<4) for i in range(4)], dtype=np.uint8)
    selected = original.reshape(160, 1360)
    raw[0].copy_(torch.from_numpy(selected))
    _, planes = outputs(23, device)
    picks = picks_for(1, 0)
    prepared.pack_active(raw, raw, picks, *planes, first=0, ggml_type=23)
    torch.cuda.synchronize()
    compare(planes[:2], lambda expert: selected.copy(), list(range(10)), 23)
