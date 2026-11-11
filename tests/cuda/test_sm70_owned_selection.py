"""Exact prepared owned routing against the explicit original Torch path."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import routing_native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual SM70')


def buffers(rows):
    return NS(rows=rows, act=torch.empty((), device='cuda', dtype=torch.float16),
              logits=torch.empty(rows, 513, device='cuda'),
              pick=torch.full((rows, 11), -17, device='cuda', dtype=torch.int32),
              wts=torch.full((rows, 11), -19., device='cuda'))


@pytest.fixture
def prepared():
    return routing_native.prepare_owned_selection(torch.device('cuda', torch.cuda.current_device()))


def check(candidate, reference, rows):
    moe._torch_select_rows(candidate.logits[:rows], reference, 10, 512)
    before = candidate.pick._version, candidate.wts._version
    moe.select_rows(candidate.logits[:rows], candidate, 10, 512)
    assert candidate.pick._version == before[0] + 2 and candidate.wts._version == before[1] + 2
    torch.testing.assert_close(candidate.pick[:rows], reference.pick[:rows], rtol=0, atol=0)
    torch.testing.assert_close(candidate.wts[:rows].view(torch.int32), reference.wts[:rows].view(torch.int32), rtol=0, atol=0)


def test_prepared_selection_every_actual_row_count_and_guard_regions(prepared):
    candidate, reference = buffers(256), buffers(256)
    candidate.selection = prepared.bind(candidate)
    generator = torch.Generator(device='cuda').manual_seed(19073)
    candidate.logits.copy_(torch.randn(candidate.logits.shape, generator=generator, device='cuda') * 3)
    for rows in range(1, 257):
        candidate.pick.fill_(-17)
        candidate.wts.fill_(-19.)
        check(candidate, reference, rows)
        assert torch.all(candidate.pick[rows:] == -17)
        assert torch.all(candidate.wts[rows:] == -19.)


def test_prepared_selection_exceptional_payloads_and_stable_global_ties(prepared):
    candidate, reference = buffers(16), buffers(16)
    candidate.selection = prepared.bind(candidate)
    candidate.logits.zero_()
    candidate.logits[0].fill_(-torch.inf)
    candidate.logits[1].fill_(torch.inf)
    candidate.logits[2, :512:2] = -0.
    candidate.logits[3, :512] = torch.arange(512, device='cuda') % 3
    candidate.logits[4, :512] = torch.finfo(torch.float32).max
    candidate.logits[5, :512] = -torch.finfo(torch.float32).max
    for row, bits in enumerate((0x7f800001, 0x7fc00001, -8388607, -4194303), 6):
        candidate.logits[row, 400].view(torch.int32).fill_(bits)
        candidate.logits[row, 512].view(torch.int32).fill_(bits)
    candidate.logits[10, 3:8] = torch.inf
    candidate.logits[11, :512].fill_(torch.nan)
    candidate.logits[12, :512].view(torch.int32).fill_(1)  # Smallest positive subnormal.
    candidate.logits[13, :512].view(torch.int32).fill_(-2147483647)
    candidate.logits[14, :4].view(torch.int32).copy_(torch.tensor([-8388607, 0x7fc00001, -4194303, 0x7f800001], device='cuda', dtype=torch.int32))
    candidate.logits[15, :512].view(torch.int32)[::2] = -4194303
    check(candidate, reference, 16)


def test_prepared_selection_current_stream_and_dynamic_graph_replay(prepared):
    candidate, reference = buffers(3), buffers(3)
    candidate.selection = prepared.bind(candidate)
    stream, done = torch.cuda.Stream(), torch.cuda.Event()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate.logits.fill_(.25)
        candidate.logits[:, 17].fill_(2.)
        moe.select_rows(candidate.logits, candidate, 10, 512)
        done.record()
    done.synchronize()
    moe._torch_select_rows(candidate.logits, reference, 10, 512)
    torch.testing.assert_close(candidate.pick, reference.pick, rtol=0, atol=0)
    torch.testing.assert_close(candidate.wts.view(torch.int32), reference.wts.view(torch.int32), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe.select_rows(candidate.logits, candidate, 10, 512)
    for seed in (17, 113, 981):
        candidate.logits.copy_(torch.randn(candidate.logits.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(seed)))
        moe._torch_select_rows(candidate.logits, reference, 10, 512)
        graph.replay()
        torch.testing.assert_close(candidate.pick, reference.pick, rtol=0, atol=0)
        torch.testing.assert_close(candidate.wts.view(torch.int32), reference.wts.view(torch.int32), rtol=0, atol=0)


def test_inference_output_refusal_outside_mode_matches_original_error_and_state(prepared, monkeypatch):
    from tensorfold.cuda.sm70 import family_ops
    with torch.inference_mode():
        candidate, reference = buffers(3), buffers(3)
        candidate.logits.zero_()
        candidate.selection = prepared.bind(candidate)
    monkeypatch.setattr(family_ops, '_NATIVE_FAMILY', False)
    assert candidate.selection.run(candidate.logits, candidate, 10, 512) is False
    with pytest.raises(RuntimeError, match='inference tensor'):
        moe._torch_select_rows(candidate.logits, reference, 10, 512)
    with pytest.raises(RuntimeError, match='inference tensor'):
        moe.select_rows(candidate.logits, candidate, 10, 512)
    # Preserve the original copy_'s actual error-side effects, which need not
    # promise rollback of a CUDA write before its metadata check fails.
    torch.testing.assert_close(candidate.pick, reference.pick, rtol=0, atol=0)
    torch.testing.assert_close(candidate.wts.view(torch.int32), reference.wts.view(torch.int32), rtol=0, atol=0)
