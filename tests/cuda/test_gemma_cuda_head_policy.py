"""B16 adaptive/CG bits and capture; direct policy is not SM121 selection proof."""
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA only', allow_module_level=True)

from tensorfold.families.gemma4.cuda.forward import Gemma4Target
from tensorfold.families.gemma4.cuda.head import _AdaptiveRowsHead, _RowsHead
from tensorfold.families.qwen3_5.cuda.weights import Plain


@pytest.fixture(scope='module', params=[False, True], ids=['h-reference', 'cg-weight'])
def heads(request):
    generator = torch.Generator(device='cuda').manual_seed(20261023)
    weight = torch.empty((262144, 2816), dtype=torch.bfloat16, device='cuda').uniform_(
        -.02, .02, generator=generator)
    hidden = torch.empty((17, 2816), dtype=torch.bfloat16, device='cuda').uniform_(
        -.5, .5, generator=generator)
    incumbent = Plain(weight)
    return incumbent, _RowsHead(incumbent, 16), _AdaptiveRowsHead(incumbent, global_cache=request.param), hidden


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


@pytest.mark.parametrize('rows', [1, 4, 5, 8, 9])
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


def test_exceptional_bits_and_fresh_outputs_keep_original_owners(heads):
    incumbent, fixed, adaptive, hidden = heads
    # Preserve payload/sign bits through the existing conversion/FMA chain;
    # integer views compare NaN outputs without treating NaN != NaN as a failure.
    words = torch.tensor([0, -32768, 1, -32767, 0x7f80, -128, 0x7fc1, -63],
                         dtype=torch.int16, device='cuda')
    special = words[torch.arange(hidden.numel(), device='cuda') % len(words)].view(torch.bfloat16).view_as(hidden)
    input_bits, version = special.view(torch.int16).clone(), incumbent.weight._version
    first = adaptive(special[:5])
    frozen = first.view(torch.int16).clone()
    assert torch.equal(first.view(torch.int16), fixed(special[:5]).view(torch.int16))
    second = adaptive(hidden[:5])
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first.view(torch.int16), frozen)
    assert torch.equal(second.view(torch.int16), fixed(hidden[:5]).view(torch.int16))
    assert torch.equal(special.view(torch.int16), input_bits) and incumbent.weight._version == version


def test_cg_raw_boundary_rejects_geometry_and_layout_before_launch(heads):
    from tensorfold.families.gemma4.cuda.head_cg import _ext

    incumbent, _fixed, _adaptive, hidden = heads
    native = _ext().gemma_b16_head_cg
    for rows in (0, 2, 3, 32):
        with pytest.raises(RuntimeError, match='CG rows'):
            native(hidden[:1], incumbent.weight, rows)
    small = torch.zeros((1, 8), dtype=torch.bfloat16, device='cuda')
    with pytest.raises(RuntimeError, match=r'BF16\[262144,2816\]'):
        native(small, small, 1)
    unaligned = torch.zeros((1, 2817), dtype=torch.bfloat16, device='cuda')[:, 1:]
    with pytest.raises(RuntimeError, match='aligned'):
        native(unaligned, incumbent.weight, 1)
    with pytest.raises(RuntimeError, match='contiguous'):
        native(hidden[:, ::2], incumbent.weight, 4)
