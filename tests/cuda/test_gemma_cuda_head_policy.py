"""Existing B16 adaptive policy bits/capture; direct policy is not SM121 selection proof."""
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA only', allow_module_level=True)

from tensorfold.families.gemma4.cuda.forward import Gemma4Target
from tensorfold.families.gemma4.cuda.head import _AdaptiveRowsHead, _RowsHead
from tensorfold.families.qwen3_5.cuda.weights import Plain


@pytest.fixture(scope='module')
def heads():
    generator = torch.Generator(device='cuda').manual_seed(20261023)
    weight = torch.empty((262144, 2816), dtype=torch.bfloat16, device='cuda').uniform_(
        -.02, .02, generator=generator)
    hidden = torch.empty((17, 2816), dtype=torch.bfloat16, device='cuda').uniform_(
        -.5, .5, generator=generator)
    incumbent = Plain(weight)
    return incumbent, _RowsHead(incumbent, 16), _AdaptiveRowsHead(incumbent), hidden


@pytest.mark.parametrize('adversarial', [False, True])
def test_existing_kernel_policy_preserves_complete_raw_softcap_and_partition_bits(heads, adversarial):
    incumbent, fixed, adaptive, normal = heads
    if adversarial:
        values = torch.tensor([0., -0., 2.**-100, -2.**-100, 2.**-10, -2.**-10,
                               .5, -.5, 2.**50, -2.**50], dtype=torch.bfloat16, device='cuda')
        x = values[torch.arange(normal.numel(), device='cuda') % len(values)].view_as(normal)
    else:
        x = normal
    before, version = x.view(torch.int16).clone(), incumbent.weight._version
    singleton = torch.cat([incumbent(row[None]) for row in x])
    for rows in (1, 2, 3, 4, 5, 8, 9, 16, 17):
        actual = adaptive(x[:rows])
        assert torch.equal(actual.view(torch.int16), fixed(x[:rows]).view(torch.int16)), rows
        assert torch.equal(actual.view(torch.int16), singleton[:rows].view(torch.int16)), rows
    partition = torch.cat([adaptive(x[:4]), adaptive(x[4:9]), adaptive(x[9:])])
    assert torch.equal(partition.view(torch.int16), singleton.view(torch.int16))
    order = torch.tensor([16, 0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8], device='cuda')
    assert torch.equal(adaptive(x[order]).view(torch.int16), singleton[order].view(torch.int16))
    target = object.__new__(Gemma4Target)
    target.config, target.weights = SimpleNamespace(softcap=30.), SimpleNamespace(embed=adaptive)
    assert torch.equal(target.project_logits(x).view(torch.int16),
                       (torch.tanh(singleton / 30.) * 30.).view(torch.int16))
    assert torch.equal(x.view(torch.int16), before) and incumbent.weight._version == version


@pytest.mark.parametrize('rows', [4, 5, 8, 9])
def test_captured_row_policy_replays_new_inputs_with_fixed_reference_bits(heads, rows):
    _incumbent, fixed, adaptive, hidden = heads
    staged = hidden[:rows].clone()
    adaptive(staged)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = adaptive(staged)
    for offset in (0, 3):
        staged.copy_(hidden[offset:offset + rows])
        graph.replay()
        expected = fixed(hidden[offset:offset + rows])
        torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_zero_rows_keep_original_native_refusal(heads):
    _incumbent, fixed, adaptive, hidden = heads
    with pytest.raises(RuntimeError) as reference:
        fixed(hidden[:0])
    with pytest.raises(RuntimeError) as candidate:
        adaptive(hidden[:0])
    assert type(candidate.value) is type(reference.value) and str(candidate.value) == str(reference.value)
