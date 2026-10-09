"""ROOT-only CUDA contract gates; never execute on the editing host."""
from __future__ import annotations

import gc
import weakref

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

from tensorfold.cuda.kernels import gdn, qmm  # noqa: E402


def _rows(w=7, hk=2, hv=6, dv=128):
    generator = torch.Generator(device='cuda').manual_seed(317)
    q = torch.randn((w, hk, 128), generator=generator, device='cuda', dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn((w, hv, dv), generator=generator, device='cuda').bfloat16()
    gates = torch.rand((w, hv), generator=generator, device='cuda')
    beta = torch.rand((w, hv), generator=generator, device='cuda')
    state = torch.randn((hv, dv, 128), generator=generator, device='cuda') * .01
    return q, k, v, gates, beta, state


def _weight(n=128, k=512, gs=32):
    generator = torch.Generator(device='cuda').manual_seed(511)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device='cuda', generator=generator)
    scales = torch.randn((n, k // gs), device='cuda', generator=generator).bfloat16() * .01
    biases = torch.randn((n, k // gs), device='cuda', generator=generator).bfloat16() * .01
    return qmm.pack(words, scales, biases, gs)


def _raw_qmm(x, weight, out, *, sk=1, part=None, reduce=True, f32=False, xs=None):
    xs = qmm.group_sums(x, weight.gs) if xs is None else xs
    qmm._ext().qmm(x, xs, weight.weight, weight.scales, weight.biases, out, part, weight.n, sk,
                   weight.gs, qmm.bucket(x.shape[0]), f32, reduce)


def test_gdn_raw_zero_heads_rejected_before_modulo_and_replay_narrowing():
    q, k, v, gates, beta, state = _rows()
    p = gdn.plan([list(range(-1, 6))], 'cuda')
    for hk in (0, 4):
        bad_q = torch.empty((7, hk, 128), dtype=q.dtype, device='cuda')
        with pytest.raises(RuntimeError):
            gdn._ext().tree(bad_q, bad_q, v, gates, beta, state, None, None, p.entries, 0, 7, None, None)
    table = gdn.to_device(gdn.replay_table([k], [v], [gates], [beta], [[state]]), torch.int64, 'cuda')
    rows = torch.zeros((1, 7), dtype=torch.int32, device='cuda')
    counts = torch.zeros((1,), dtype=torch.int32, device='cuda')
    for geometry in ((0, 1, 2, 6, 128), (1, 0, 2, 6, 128), (1, 1, 0, 6, 128),
                     (1, 1, 4, 6, 128), (1, 1, 2, 6, 0), (2**63 - 1, 1, 2, 6, 128)):
        layers, streams, hk, hv, dv = geometry
        with pytest.raises(RuntimeError):
            gdn._ext().replay(table, layers, streams, rows, counts, hk, hv, dv, True, False)
    with pytest.raises(RuntimeError):
        gdn._ext().replay(table, 1, 1, rows.cpu(), counts, 2, 6, 128, True, False)
    with pytest.raises(RuntimeError):
        gdn._ext().tree(q, k, v, gates, beta, state, None, None, p.entries, 0, 2**63 - 1, None, None)
    torch.cuda.synchronize()


def test_gdn_vector_alignment_write_alias_and_empty_chain_contract():
    q, k, v, gates, beta, state = _rows()
    unaligned = torch.empty(q.numel() + 1, dtype=q.dtype, device='cuda')[1:].view_as(q)
    final = torch.empty_like(state)
    with pytest.raises(RuntimeError):
        gdn.chain(unaligned, k, v, gates, beta, state, final)
    with pytest.raises(RuntimeError):
        gdn.chain(q, k, v, gates, beta, state, state)
    before = state.clone()
    empty = gdn.chain(q[:0], k[:0], v[:0], gates[:0], beta[:0], state, state)
    assert empty.shape == (0, 6, 128) and torch.equal(before, state)


def test_gdn_owner_retention_side_stream_lifetime_and_packed_views():
    # A real side-stream consumer exercises record_stream, while explicit waits
    # cover producer completion. Allocation pressure begins after owner deletion.
    q, k, v, gates, beta, state = _rows()
    state_copy = state.clone()
    p = gdn.plan([list(range(-1, 6))], 'cuda')
    tables = gdn.pointer_tables([[state], [state_copy]], 'cuda')
    reference = weakref.ref(state)
    del state
    gc.collect()
    assert reference() is not None
    wanted = gdn.tree(q, k, v, gates, beta, p, state=state_copy)
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        got = gdn.tree(q, k, v, gates, beta, p, table=tables[0])
    del tables
    gc.collect()
    assert reference() is None
    pressure = [torch.empty_like(state_copy) for _ in range(32)]
    consumer.synchronize()
    assert torch.equal(got, wanted)
    del pressure


def test_gdn_owned_shared_readonly_states_and_inplace_exclusivity():
    q, k, v, gates, beta, state = _rows()
    shared = gdn.replay_table([k], [v], [gates], [beta], [[state], [state]])
    assert len(shared) == 6
    with pytest.raises(ValueError):
        gdn.replay_table([k], [v], [gates], [beta], [[state], [state]], in_place=True)
    # Two read-only states may be identical; pending would write that shared state.
    table = gdn.to_device(gdn.pointers([state, state]), torch.int64, 'cuda')
    two = [tensor.repeat((2,) + (1,) * (tensor.dim() - 1)) for tensor in (q, k, v, gates, beta)]
    plan = gdn.plan([list(range(-1, 6))] * 2, 'cuda')
    together = gdn.tree(*two, plan, table=table)
    assert torch.equal(together[:7], together[7:])
    rows = torch.zeros((2, 1), dtype=torch.int32, device='cuda')
    counts = torch.zeros((2,), dtype=torch.int32, device='cuda')
    with pytest.raises(ValueError):
        gdn.tree(*two, plan, table=table, pending=(k, v, gates, beta, rows, counts))


def test_qmm_disjoint_same_allocation_row_gaps_remain_legal():
    weight = _weight(k=128)
    storage = torch.randn(512, device='cuda').bfloat16()
    x = storage.as_strided((2, 128), (384, 1))
    out = storage[128:384].view(2, 128)
    wanted = qmm.matmul(x, weight, sk=1)
    _raw_qmm(x, weight, out)
    assert torch.equal(out, wanted)
    scale_storage = torch.empty((4, 256), dtype=torch.bfloat16, device='cuda')
    bias_storage = torch.empty_like(scale_storage)
    scales, biases = scale_storage[:, :128], bias_storage[:, :128]
    scales.copy_(weight.scales)
    biases.copy_(weight.biases)
    strided = qmm.Q4(weight.weight, scales, biases, 128, 128, 32)
    source = torch.randn((1, 128), device='cuda').bfloat16()
    out = scale_storage[0, 128:].view(1, 128)
    wanted = qmm.matmul(source, weight, sk=1)
    _raw_qmm(source, strided, out)
    assert torch.equal(out, wanted)


def test_qmm_staged_input_alias_unused_output_and_scratch_tail_remain_legal():
    # SK=16 always stages, including CUDA hardware supporting up to 8 clustered slices.
    weight = _weight(n=512, k=512)
    x = torch.randn((2, 512), device='cuda').bfloat16()
    wanted = torch.empty_like(x)
    _raw_qmm(x, weight, wanted, sk=16)
    _raw_qmm(x, weight, x, sk=16)
    assert torch.equal(x, wanted)
    source = torch.randn_like(x)
    scratch = torch.empty((16, 2, 512), dtype=torch.float32, device='cuda')
    _raw_qmm(source, weight, source, sk=16, part=scratch, reduce=False)
    reduced = scratch[0].clone()
    for part in scratch[1:]:
        reduced.add_(part)
    expected = torch.empty_like(source)
    _raw_qmm(source, weight, expected, sk=16)
    assert torch.equal(reduced.bfloat16(), expected)
    backing = torch.empty(scratch.numel() + expected.numel(), dtype=torch.float32, device='cuda')
    out = backing[scratch.numel():].view(2, 512)
    _raw_qmm(source, weight, out, sk=16, part=backing, f32=True)
    assert torch.equal(out, reduced)
    # SK=1 ignores caller scratch even when it aliases the output.
    direct = torch.empty_like(source)
    _raw_qmm(source, weight, direct, part=direct)


def test_qmm_actual_simultaneous_write_aliases_fail_before_launch():
    weight = _weight()
    x = torch.randn((2, 512), device='cuda').bfloat16()
    aliased = x.reshape(-1)[:256].view(2, 128)
    with pytest.raises(RuntimeError):
        _raw_qmm(x, weight, aliased)
    with pytest.raises(RuntimeError):
        qmm._ext().qmm_prefill(x, weight.weight, weight.scales, weight.biases, aliased, 128, 32, False, 9)
    x8, sums, factors = qmm.quantize_rows(x, 32)
    aliased8 = x8.reshape(-1)[:512].view(torch.bfloat16).view(2, 128)
    with pytest.raises(RuntimeError):
        qmm._ext().qmm_prefill8(x8, sums, factors, weight.weight, weight.scales, weight.biases,
                              aliased8, 128, 32, False, 1)
    w8 = torch.randint(0, 127, (128, 512), dtype=torch.uint8, device='cuda')
    with pytest.raises(RuntimeError):
        qmm._ext().qmm_prefill8w(x8, factors, w8, weight.scales, aliased8, 128, 32, False, 1, False)
    scratch = torch.empty((16, 2, 128), dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError):
        _raw_qmm(x, weight, scratch.reshape(-1)[:256].view(2, 128), sk=16, part=scratch, f32=True)
    torch.cuda.synchronize()


def test_grouped_output_cross_group_alias_and_selector_narrowing():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('grouped kernel requires sm_12x')
    weight = _weight(gs=64)
    x = torch.randn((1, 512), device='cuda').bfloat16()
    xs = qmm.group_sums(x, 64)
    output = torch.empty((1, 128), dtype=torch.bfloat16, device='cuda')
    args = [x, xs, [weight.weight] * 2, [weight.scales] * 2, [weight.biases] * 2,
            [output, output], [128, 128], [1, 1], False, 0, -1]
    with pytest.raises(RuntimeError):
        qmm._ext().qmm_group(*args)
    aliased = weight.weight.reshape(-1).view(torch.bfloat16)[:128].view(1, 128)
    args[5] = [output, aliased]
    with pytest.raises(RuntimeError):
        qmm._ext().qmm_group(*args)
    storage = torch.empty((2, 1, 128), dtype=torch.bfloat16, device='cuda')
    args[5] = list(storage.unbind(0))
    for tile in (-1, 13, 2**32 + 1):
        args[9] = tile
        with pytest.raises(RuntimeError):
            qmm._ext().qmm_group(*args)
    args[9] = 0
    for early in (-2**63, -1, 0, 1, 2**63 - 1):
        args[10] = early
        qmm._ext().qmm_group(*args)
        assert torch.equal(storage[0], storage[1])


def test_owned_gdn_pointer_table_graph_capture():
    q, k, v, gates, beta, state = _rows(w=1)
    p = gdn.plan([[-1]], 'cuda')
    table = gdn.to_device(gdn.pointers([state]), torch.int64, 'cuda')
    gdn.tree(q, k, v, gates, beta, p, table=table)  # compile/warm before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gdn.tree(q, k, v, gates, beta, p, table=table)
    graph.replay()
    wanted = gdn.tree(q, k, v, gates, beta, p, state=state)
    torch.cuda.synchronize()
    assert torch.equal(output, wanted)


def test_qmm_raw_staging_alignment_and_scalar_output_exceptions():
    weight = _weight()
    x = torch.randn((1, 512), device='cuda').bfloat16()
    sums = qmm.group_sums(x, 32)
    output = torch.empty((1, 128), dtype=torch.bfloat16, device='cuda')
    for component in ('weight', 'scales', 'biases'):
        original = getattr(weight, component)
        shifted = torch.empty(original.numel() + 1, dtype=original.dtype, device='cuda')[1:].view_as(original)
        inputs = [weight.weight, weight.scales, weight.biases]
        inputs[('weight', 'scales', 'biases').index(component)] = shifted
        with pytest.raises(RuntimeError):
            qmm._ext().qmm(x, sums, *inputs, output, None, 128, 1, 32, 16, False, True)
    rows = torch.empty((16, 129), dtype=torch.bfloat16, device='cuda')
    with pytest.raises(RuntimeError):
        qmm._ext().qmm(x, sums, weight.weight, rows[:, :128], torch.empty_like(rows)[:, :128],
                       output, None, 128, 1, 32, 16, False, True)
    shifted_out = torch.empty(129, dtype=torch.bfloat16, device='cuda')[1:].view(1, 128)
    with pytest.raises(RuntimeError):
        _raw_qmm(x, weight, shifted_out)
    # Separate reduction stores bf16 scalars, so this 2-byte alignment is legal.
    wanted = torch.empty_like(output)
    _raw_qmm(x, weight, wanted, sk=16)
    _raw_qmm(x, weight, shifted_out, sk=16)
    assert torch.equal(shifted_out, wanted)
    x8, xs, factors = qmm.quantize_rows(x, 32)
    shifted_x8 = torch.empty(x8.numel() + 1, dtype=torch.uint8, device='cuda')[1:].view_as(x8)
    with pytest.raises(RuntimeError):
        qmm._ext().qmm_prefill8(shifted_x8, xs, factors, weight.weight, weight.scales, weight.biases,
                              output, 128, 32, False, 1)
    # FP8 prefill reads its biases as scalar bf16 values, not staged cp16 rows.
    shifted_bias = torch.empty(weight.biases.numel() + 1, dtype=torch.bfloat16, device='cuda')[1:].view_as(weight.biases)
    shifted_bias.copy_(weight.biases)
    qmm._ext().qmm_prefill8(x8, xs, factors, weight.weight, weight.scales, weight.biases, wanted, 128, 32, False, 1)
    qmm._ext().qmm_prefill8(x8, xs, factors, weight.weight, weight.scales, shifted_bias, output, 128, 32, False, 1)
    assert torch.equal(output, wanted)
    # L64 stages its weight with cp8 and permits 8-byte-aligned byte views.
    w8 = torch.randint(0, 127, (128, 512), dtype=torch.uint8, device='cuda')
    shifted_w8 = torch.empty(w8.numel() + 8, dtype=torch.uint8, device='cuda')[8:].view_as(w8)
    shifted_w8.copy_(w8)
    qmm._ext().qmm_prefill8w(x8, factors, w8, weight.scales, wanted, 128, 32, False, 1, True)
    qmm._ext().qmm_prefill8w(x8, factors, shifted_w8, weight.scales, output, 128, 32, False, 1, True)
    assert torch.equal(output, wanted)
    with pytest.raises(RuntimeError):
        qmm._ext().qmm_prefill8w(x8, factors, shifted_w8, weight.scales, output, 128, 32, False, 1, False)
    if torch.cuda.get_device_capability()[0] == 12:
        grouped = _weight(gs=64)
        grouped_xs = qmm.group_sums(x, 64)
        for tile in (6, 7):
            arguments = [x, grouped_xs, [grouped.weight], [grouped.scales], [grouped.biases], [output],
                         [128], [1], False, tile, -1]
            qmm._ext().qmm_group(*arguments)
            arguments[5] = [shifted_out]
            qmm._ext().qmm_group(*arguments)
            assert torch.equal(shifted_out, output)
    torch.cuda.synchronize()


def _padding_output(weight, component, n):
    if component == 'weight':
        start = 64 * weight.k // 8
        return weight.weight.reshape(-1)[start:start + n].view(torch.float32).view(1, n)
    # A contiguous fp32 output fills two bf16 cells per column in one unread row gap.
    return getattr(weight, component)[0, 64:64 + 2 * n].view(torch.float32).view(1, n)


@pytest.mark.parametrize('component', ['weight', 'scales', 'biases'])
def test_direct_and_prefill_selected_unread_padding_aliases(component):
    weight = _weight(n=32, k=128)
    x = torch.randn((1, 128), device='cuda').bfloat16()
    wanted = torch.empty((1, 32), dtype=torch.float32, device='cuda')
    _raw_qmm(x, weight, wanted, f32=True)
    aliased = _padding_output(weight, component, 32)
    _raw_qmm(x, weight, aliased, f32=True)
    assert torch.equal(aliased, wanted)
    for tile in range(12):
        qmm._ext().qmm_prefill(x, weight.weight, weight.scales, weight.biases, wanted, 32, 32, True, tile)
        if tile in (2, 3):
            qmm._ext().qmm_prefill(x, weight.weight, weight.scales, weight.biases, aliased, 32, 32, True, tile)
            assert torch.equal(aliased, wanted)
        else:
            with pytest.raises(RuntimeError, match='loaded'):
                qmm._ext().qmm_prefill(x, weight.weight, weight.scales, weight.biases, aliased, 32, 32, True, tile)
    # Even a tiny output crossing the read/unread boundary overlaps active loads.
    if component == 'weight':
        crossing = weight.weight.reshape(-1)[64 * weight.k // 8 - 1:64 * weight.k // 8 + 31].view(torch.float32).view(1, 32)
    else:
        crossing = getattr(weight, component)[0, 62:126].view(torch.float32).view(1, 32)
    with pytest.raises(RuntimeError, match='loaded'):
        _raw_qmm(x, weight, crossing, f32=True)
    torch.cuda.synchronize()


def test_split_k_scratch_can_use_unread_packed_or_parameter_padding():
    x = torch.randn((1, 128), device='cuda').bfloat16()
    output = torch.empty((1, 4), dtype=torch.bfloat16, device='cuda')
    for component in ('weight', 'scales', 'biases'):
        weight = _weight(n=4, k=128)
        wanted = torch.empty((4, 1, 4), dtype=torch.float32, device='cuda')
        _raw_qmm(x, weight, output, sk=4, part=wanted, reduce=False)
        if component == 'weight':
            aliased = weight.weight.reshape(-1)[64 * weight.k // 8:].view(torch.float32)
        else:
            aliased = getattr(weight, component)[0, 64:].view(torch.float32)
        _raw_qmm(x, weight, output, sk=4, part=aliased, reduce=False)
        assert torch.equal(aliased[:wanted.numel()].view_as(wanted), wanted)


def test_grouped_padding_alias_legality_follows_exact_selected_column_tile():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('grouped kernel requires sm_12x')
    x = torch.randn((1, 512), device='cuda').bfloat16()
    sums = qmm.group_sums(x, 64)
    for component in ('weight', 'scales', 'biases'):
        weight = _weight(n=32, k=512, gs=64)
        aliased = _padding_output(weight, component, 32)
        wanted = torch.empty((1, 32), dtype=torch.float32, device='cuda')
        for tile in range(13):
            arguments = [x, sums, [weight.weight], [weight.scales], [weight.biases], [wanted],
                         [32], [1], True, tile, -1]
            qmm._ext().qmm_group(*arguments)
            arguments[5] = [aliased]
            selected = tile
            if selected == 0:
                major, minor = torch.cuda.get_device_capability()
                selected = 2 if (major, minor) == (12, 1) else 7  # original one-row device policy
            if selected in (1, 2, 3, 4, 6):
                qmm._ext().qmm_group(*arguments)
                assert torch.equal(aliased, wanted)
            else:
                with pytest.raises(RuntimeError, match='loaded'):
                    qmm._ext().qmm_group(*arguments)
    torch.cuda.synchronize()


def test_prefill_256_column_padding_is_checked_before_kernel_narrowing():
    weight = _weight()
    x = torch.randn((1, 512), device='cuda').bfloat16()
    output = torch.empty((1, 128), dtype=torch.bfloat16, device='cuda')
    for tile in (7, 8, 11):
        with pytest.raises(RuntimeError, match='padding exceeds signed-int'):
            qmm._ext().qmm_prefill(x, weight.weight, weight.scales, weight.biases, output,
                                 2**31 - 128, 32, False, tile)


def test_shared_qmm_specializations_initialize_on_each_compatible_cuda_device():
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices; a one-device pass cannot prove per-device initialization')
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        pytest.skip('the JIT builds one declared GPU ISA; this gate requires two devices with that same ISA')
    # One extension module, repeated device switching, all available provider paths.
    for device in (0, 1, 0, 1):
        with torch.cuda.device(device):
            weight = _weight()
            x = torch.randn((1, 512), device='cuda').bfloat16()
            out = torch.empty((1, 128), dtype=torch.float32, device='cuda')
            _raw_qmm(x, weight, out, f32=True)
            qmm.prefill_matmul(x, weight, f32=True, tile=7)
            qmm.prefill_matmul8(qmm.quantize_rows(x, 32), weight, f32=True)
            x8, _, factors = qmm.quantize_rows(x, 32)
            w8 = torch.randint(0, 127, (128, 512), dtype=torch.uint8, device='cuda')
            qmm._ext().qmm_prefill8w(x8, factors, w8, weight.scales, out, 128, 32, True, 0, False)
            if torch.cuda.get_device_capability()[0] == 12:
                grouped = _weight(gs=64)
                qmm.matmul_group(x, [grouped], f32=True, tile=10)
            torch.cuda.synchronize()
