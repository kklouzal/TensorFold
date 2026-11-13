"""Unrun until an exclusive native gate: exact original CUDA packet oracle.

The fake eight-rank gather exercises packet storage only; real TP8/NCCL and
full-model/state/quality qualification remain separate integration gates.
"""
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.forward import candidates
from tests.test_row_sampling_scratch import fixture


pytestmark = [pytest.mark.torch, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


@pytest.mark.parametrize('width', [7, 65, 31040])
@pytest.mark.parametrize('rows,first', [(1, 0), (5, 1), (40, 2), (120, 0)])
def test_cuda_complete_packets_and_changed_input_graph_replays_match_original_bits(width, rows, first):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip('required native target is SM70')
    w, b = fixture(width, rows + first, selected=True, device='cuda')
    reference_weights, reference = fixture(width, rows + first, selected=False, device='cuda')
    owners = tuple(t.data_ptr() for t in b.row_sampling_scratch.tensors)
    inputs = torch.arange(b.logits.numel(), device='cuda').view_as(b.logits)
    graph = torch.cuda.CUDAGraph()
    try:
        for change in range(2):
            values = ((inputs + change) % 17).float().sub_(8).div_(8)
            values[:, 0] = -0.0
            if width > 1:
                values[:, 1] = 0.0
            b.logits.copy_(values)
            reference.logits.copy_(values)
            if change == 0:
                before = b.cand._version, b.cand_all._version
                old_before = reference.cand._version, reference.cand_all._version
                candidates(w, b, b.logits[first:first + rows], rows, offset=320)
                candidates(reference_weights, reference, reference.logits[first:first + rows], rows, offset=320)
                torch.cuda.synchronize()
                assert torch.equal(b.cand[:rows].view(torch.int32), reference.cand[:rows].view(torch.int32))
                assert (b.cand._version - before[0], b.cand_all._version - before[1]) == (
                    reference.cand._version - old_before[0], reference.cand_all._version - old_before[1])
                with torch.cuda.graph(graph):
                    candidates(w, b, b.logits[first:first + rows], rows, offset=320)
            b.row_sampling_scratch.validate_bindings(w, b)
            graph.replay()
            candidates(reference_weights, reference, reference.logits[first:first + rows], rows, offset=320)
            torch.cuda.synchronize()
            assert torch.equal(b.cand[:rows].view(torch.int32), reference.cand[:rows].view(torch.int32))
            assert tuple(t.data_ptr() for t in b.row_sampling_scratch.tensors) == owners
    finally:
        torch.cuda.synchronize()
        graph.reset()
        b.row_sampling_scratch.close()


def test_cuda_mapped_compact_head_keeps_original_id_bits_and_padding():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip('required native target is SM70')
    mapping = torch.tensor([991, 47, 82, 13, 712], dtype=torch.int64, device='cuda')
    w, b = fixture(65, 7, selected=True, role='mtp', mapping=mapping, device='cuda')
    oldw, old = fixture(65, 7, selected=False, role='mtp', mapping=mapping, device='cuda')
    try:
        source = b.logits.view(-1)[:35].view(7, 5)
        original = old.logits.view(-1)[:35].view(7, 5)
        values = torch.arange(15, device='cuda', dtype=torch.float32).view(3, 5).remainder_(4)
        source[2:5].copy_(values)
        original[2:5].copy_(values)
        candidates(w, b, source[2:5], 3, id_map=mapping, offset=320)
        candidates(oldw, old, original[2:5], 3, id_map=mapping, offset=320)
        torch.cuda.synchronize()
        assert torch.equal(b.cand[:3].view(torch.int32), old.cand[:3].view(torch.int32))
    finally:
        torch.cuda.synchronize()
        b.row_sampling_scratch.close()
