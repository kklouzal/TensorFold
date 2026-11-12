"""Prepared exact-pin varlen integration; component checks, no quality/P3 acceptance."""
import json
import threading
import pytest
import torch
from tensorfold.cuda.sm70 import flash_v100_forward as fa
from tests.cuda.test_flash_v100_forward import native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual SM70')


def tensors(batch=4):
    q = torch.zeros(batch * 12, 1, 256, device='cuda', dtype=torch.float16)
    k = torch.zeros(batch, 2051, 1, 256, device='cuda', dtype=torch.float16)
    v = torch.ones_like(k)
    valid = torch.zeros(batch, 1, 2051, device='cuda', dtype=torch.bool)
    return q, k, v, valid, torch.full_like(q, -711.), torch.full((1, batch * 12), -713., device='cuda')


def prepare(module, args):
    return fa.prepare_compacted(module, *args, tensor_budget_bytes=fa.compacted_workspace_bytes(len(args[0]) // 12))


def test_mixed_empty_prefix_suffix_holes_and_changed_values_under_capture(native):
    args = tensors()
    q, k, v, valid, out, lse = args
    valid[1, 0, 2050] = True
    valid[2, 0, 1:129:2] = True
    valid[3, 0, :2048] = True
    # Invalid slots may contain nonfinite payload: the compactor must not read them.
    k.masked_fill_(~valid[:, 0, :, None, None], torch.nan)
    v.masked_fill_(~valid[:, 0, :, None, None], torch.nan)
    with prepare(native, args) as owner:
        allocated = torch.cuda.memory_allocated()
        owner.run(*args)
        assert torch.cuda.memory_allocated() == allocated
        assert torch.equal(out[12:], torch.ones_like(out[12:]))
        assert not out[:12].any() and torch.isneginf(lse[:, :12]).all()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                owner.run(*args)
            valid.zero_(); valid[:, :, -1] = True
            k[:, -1].zero_(); v[:, -1].fill_(3)
            graph.replay()
            assert torch.equal(out, torch.full_like(out, 3)) and not lse.any()
            valid.zero_()
            graph.replay()
            assert not out.any() and torch.isneginf(lse).all()
        finally:
            torch.cuda.synchronize(); graph.reset()


def test_stable_compaction_consumes_exact_survivor_order_and_head_rows(native, record_property):
    args = tensors(2)
    q, k, v, valid, out, lse = args
    generator = torch.Generator(device='cuda').manual_seed(117)
    q.copy_(torch.randn(q.shape, generator=generator, device='cuda').half())
    k.copy_(torch.randn(k.shape, generator=generator, device='cuda').half())
    v.copy_(torch.randn(v.shape, generator=generator, device='cuda').half())
    chosen = [torch.tensor([0, 2, 31, 64, 99, 256, 2049, 2050], device='cuda'),
              torch.arange(2051, device='cuda')]
    for row in range(2): valid[row, 0, chosen[row]] = True
    with prepare(native, args) as owner:
        owner.run(*args)
        errors = []
        for row, ids in enumerate(chosen):
            qr = q[row * 12:(row + 1) * 12].view(1, 12, 1, 256)
            kr = k[row:row + 1, ids].contiguous(); vr = v[row:row + 1, ids].contiguous()
            reference_out = torch.empty_like(qr); reference_lse = torch.empty(1, 1, 12, device='cuda')
            # Independent dense boundary consumes explicitly indexed survivors,
            # rather than reusing the owned compactor or GPU count producer.
            with fa.prepare(native, qr, kr, vr, reference_out, reference_lse) as reference:
                reference.run(qr, kr, vr, reference_out, reference_lse)
            assert torch.equal(out[row * 12:(row + 1) * 12], reference_out.reshape(12, 1, 256))
            assert torch.equal(lse[:, row * 12:(row + 1) * 12], reference_lse.reshape(1, 12))
            scores = qr[0, :, 0].double() @ kr[0, :, 0].double().T / 16
            ideal = scores.softmax(-1) @ vr[0, :, 0].double()
            errors.append(float((out[row * 12:(row + 1) * 12, 0].double() - ideal).abs().max()))
        record_property('varlen_vs_double_diagnostic', json.dumps({'max_abs': max(errors), 'quality_accepted': False}))


def test_row_embedding_permutation_and_other_row_lengths_preserve_exact_bits(native, record_property):
    args = tensors()
    q, k, v, valid, out, lse = args
    generator = torch.Generator(device='cuda').manual_seed(12)
    for value in (q, k, v): value.copy_(torch.randn(value.shape, generator=generator, device='cuda').half())
    for row, count in enumerate([1, 17, 65, 2051]): valid[row, 0, :count] = True
    with prepare(native, args) as owner:
        owner.run(*args)
        expected_out, expected_lse = out.clone(), lse.clone()
        for row in range(4):
            single = (q[row*12:(row+1)*12], k[row:row+1], v[row:row+1], valid[row:row+1],
                      torch.empty(12, 1, 256, device='cuda', dtype=torch.float16), torch.empty(1, 12, device='cuda'))
            with prepare(native, single) as solo: solo.run(*single)
            assert torch.equal(single[-2].view(torch.int16), expected_out[row*12:(row+1)*12].view(torch.int16))
            assert torch.equal(single[-1].view(torch.int32), expected_lse[:, row*12:(row+1)*12].view(torch.int32))
        heads = torch.tensor([11, 0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5], device='cuda')
        reordered_q = q.view(4, 12, 1, 256)[:, heads].reshape_as(q)
        owner.run(reordered_q, k, v, valid, out, lse)
        assert torch.equal(out, expected_out.view(4, 12, 1, 256)[:, heads].reshape_as(out))
        assert torch.equal(lse, expected_lse.view(4, 12)[:, heads].reshape_as(lse))
        valid[1:].zero_(); owner.run(*args)
        assert torch.equal(out[:12], expected_out[:12]) and torch.equal(lse[:, :12], expected_lse[:, :12])
    order = torch.tensor([3, 0, 2, 1], device='cuda')
    # Restore lengths, then permute physical batch slots without changing a row.
    for row, count in enumerate([1, 17, 65, 2051]): valid[row, 0, :count] = True
    permuted = (q.view(4, 12, 1, 256)[order].reshape_as(q), k[order], v[order], valid[order],
                torch.empty_like(out), torch.empty_like(lse))
    with prepare(native, permuted) as owner: owner.run(*permuted)
    assert torch.equal(permuted[-2], expected_out.view(4, 12, 1, 256)[order].reshape_as(out))
    assert torch.equal(permuted[-1], expected_lse.view(4, 12)[order].reshape_as(lse))
    record_property('varlen_row_bit_drift', '0')


@pytest.mark.parametrize('index', [0, 1, 4])
def test_misaligned_operand_refused_before_output_write(native, index):
    args = list(tensors(1)); tensor = args[index]
    args[index] = torch.empty(tensor.numel() + 1, device='cuda', dtype=tensor.dtype)[1:].view_as(tensor)
    args[4].fill_(-711.)
    with pytest.raises(RuntimeError, match='16-byte|contiguous16'):
        prepare(native, args)
    assert torch.equal(args[4], torch.full_like(args[4], -711.))


def test_alias_credit_and_geometry_refusals_leave_output_untouched(native):
    args = list(tensors(1)); args[4] = args[0]
    with pytest.raises(RuntimeError, match='overlap'): prepare(native, args)
    args = tensors(1)
    with pytest.raises(RuntimeError, match='credit'):
        native.Prepared(args[0], args[1], args[2], args[4], args[5], .0625, args[3], 0)
    for batch in (0, 257):
        with pytest.raises(RuntimeError, match='1..256'): native.compacted_workspace_bytes(batch)
    assert torch.equal(args[4], torch.full_like(args[4], -711.))


def test_nonempty_nonfinite_output_is_not_repaired(native):
    args = tensors(1); args[3][0, 0, 7] = True; args[2][0, 7].fill_(torch.nan)
    with prepare(native, args) as owner:
        owner.run(*args)
        assert torch.isnan(args[-2]).any()


def test_nondefault_stream_handoff_and_control_retirement(native):
    args = tensors(1); args[3][0, 0, -1] = True
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): owner = prepare(native, args)
    errors = []
    def worker():
        try:
            owner.handoff_to_current_thread(); owner.run(*args)
        except BaseException as error: errors.append(error)
    thread = threading.Thread(target=worker); thread.start(); thread.join()
    assert not errors
    with pytest.raises(RuntimeError, match='Thread'): owner.run(*args)
    owner.close()
    assert torch.equal(args[-2], torch.ones_like(args[-2]))


def test_aligned_subviews_preserve_prefix_and_suffix_canaries(native):
    args = list(tensors(1)); args[3][0, 0, 2050] = True
    slab = torch.full((12 * 256 + 16,), -333., device='cuda', dtype=torch.float16)
    lse_slab = torch.full((20,), -335., device='cuda')
    args[-2] = slab[8:-8].view(12, 1, 256)
    args[-1] = lse_slab[4:-4].view(1, 12)
    with prepare(native, args) as owner: owner.run(*args)
    assert torch.equal(args[-2], torch.ones_like(args[-2]))
    assert torch.equal(slab[:8], torch.full_like(slab[:8], -333.))
    assert torch.equal(slab[-8:], torch.full_like(slab[-8:], -333.))
    assert torch.equal(lse_slab[:4], torch.full_like(lse_slab[:4], -335.))
    assert torch.equal(lse_slab[-4:], torch.full_like(lse_slab[-4:], -335.))


def test_large_common_score_lse_mass_loss_remains_explicit(native, record_property):
    args = tensors(2)
    q, k, v, valid, out, lse = args
    q.fill_(4096); k.fill_(4096)
    valid[0, 0, 0] = True; valid[1, 0, :65] = True; v[1].fill_(3)
    with prepare(native, args) as owner: owner.run(*args)
    # Both nonempty owners retain finite LSE; the empty canonicalizer must not
    # repair the known representation loss by treating either as an empty rank.
    assert torch.isfinite(lse).all() and torch.equal(lse[:, :12], lse[:, 12:])
    wrong_mass_merge = (out[:12].float() + out[12:].float()) / 2
    ideal = (1 + 65 * 3) / 66
    record_property('varlen_lse_mass_counterexample', json.dumps({
        'actual_lse': [float(lse[0, 0]), float(lse[0, 12])],
        'correct_mass_ratio': [1, 65], 'LSE_implied_ratio': [1, 1],
        'max_abs_merge_error': float((wrong_mass_merge - ideal).abs().max()),
        'quality_accepted': False}))
