"""Native integration oracles for unselected dense candidates, not PPL/P3 gates."""
import json
import threading

import pytest
import torch

from tensorfold.cuda.sm70 import dense_backends as dense

pytestmark = pytest.mark.skipif(not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0), reason='requires actual SM70')


def operands(rows, dtype, *, seed):
    generator = torch.Generator().manual_seed(seed)
    # Integer dot products are exact in Float32 over this bounded domain.
    # int64 CPU matmul is independent of the CUDA vendor/CUTLASS algorithms.
    a = torch.randint(-1, 2, (rows, 2560), generator=generator)
    b = torch.randint(-1, 2, (160, 2560), generator=generator)
    x, w = a.to('cuda', torch.float16), b.to('cuda', torch.float16)
    storage = torch.full((rows, 164), -731., dtype=dtype, device='cuda')
    out = storage[:, 2:162]
    gold = (a @ b.T).to(dtype)
    return x, w, out, storage, gold


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
@pytest.mark.parametrize('rows', [1, 4, 32, 64, 256])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_exact_input_consumption_strided_output_and_pointer_reuse(backend, rows, dtype):
    args = operands(rows, dtype, seed=801 + rows)
    second = operands(rows, dtype, seed=901 + rows)
    with dense.Context('cuda:0', backend=backend, max_plans=1) as context:
        plan = dense.prepare(context, *args[:3])
        for x, w, out, storage, gold in (args, second, args):
            plan.run(x, w, out)
            assert torch.equal(out.cpu(), gold)
            assert torch.all(storage[:, :2] == -731) and torch.all(storage[:, 162:] == -731)
        assert plan.settings()['workspace_bytes'] == 0


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
def test_capture_on_nonpreparing_stream_then_replay_changed_values(backend):
    x, w, out, _, gold = operands(4, torch.float32, seed=177)
    with dense.Context('cuda:0', backend=backend, max_plans=1) as context:
        plan = dense.prepare(context, x, w, out)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            plan.run(x, w, out)  # Warm selected stream before capture.
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            plan.run(x, w, out)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.cpu(), gold)
        x.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, torch.zeros_like(out))
        graph.reset()  # Retire graph before context release.


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
def test_startup_scheduler_thread_handoff_then_control_thread_retirement(backend):
    x, w, out, _, gold = operands(4, torch.float32, seed=143)
    context = dense.Context('cuda:0', backend=backend, max_plans=1)
    temporary = torch.cuda.Stream()
    temporary.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(temporary):
        plan = dense.prepare(context, x, w, out)  # Positive warmup fence.
    del temporary  # Original stream wrapper is not a retained caller loan.
    errors, results = [], []
    def worker():
        try:
            context.handoff_to_current_thread()
            plan.run(x, w, out)
            results.append(out.cpu())
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    try:
        assert errors == []
        assert len(results) == 1 and torch.equal(results[0], gold)
        with pytest.raises(RuntimeError, match='host thread'):
            plan.run(x, w, out)
    finally:
        # The old startup/control thread closes after worker/graphs retire.
        context.close()


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
def test_changed_geometry_and_overlap_rejected_before_write(backend):
    x, w, out, _, _ = operands(4, torch.float16, seed=19)
    with dense.Context('cuda:0', backend=backend, max_plans=1) as context:
        plan = dense.prepare(context, x, w, out)
        snapshot = out.clone()
        with pytest.raises(RuntimeError, match='geometry'):
            plan.run(x[:3], w, out)
        assert torch.equal(snapshot, out)
    # Overlap is also rejected before native plan/resource acquisition.
    with dense.Context('cuda:0', backend=backend, max_plans=1) as context:
        with pytest.raises(RuntimeError, match='overlaps'):
            dense.prepare(context, x, w, x[:, :160])
    if backend == 'cutlass':
        # Eight-byte alignment passes upstream TensorRef_aligned(...,8),
        # but cannot satisfy the actual Half8 AlignedArray access's16-byte ABI.
        backing = torch.zeros(x.numel() + 4, dtype=x.dtype, device=x.device)
        misaligned = backing[4:].view_as(x)
        assert misaligned.data_ptr() % 16 == 8
        with dense.Context('cuda:0', backend=backend, max_plans=1) as context:
            plan = dense.prepare(context, x, w, out)
            snapshot = out.clone()
            with pytest.raises(RuntimeError, match='pointer alignment'):
                plan.run(misaligned, w, out)
            assert torch.equal(snapshot, out)


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
def test_real_half_observations_no_invented_acceptance(backend, record_property):
    generator = torch.Generator(device='cuda').manual_seed(611)
    x = torch.randn(4, 2560, device='cuda', dtype=torch.float16, generator=generator) * .1
    w = torch.randn(160, 2560, device='cuda', dtype=torch.float16, generator=generator) * .1
    out = torch.empty(4, 160, device='cuda', dtype=torch.float16)
    torch_result = x @ w.T
    with dense.Context('cuda:0', backend=backend, max_plans=1) as context:
        plan = dense.prepare(context, x, w, out)
        plan.run(x, w, out)
        assert torch.isfinite(out).all()
        record_property('dense_diagnostic', json.dumps({
            'backend': backend, 'settings': plan.settings(),
            'torch_half_mismatches': int((out != torch_result).sum().item()),
            'torch_max_abs': float((out.float() - torch_result.float()).abs().max().item()),
            'quality_accepted': False, 'P3_qualified': False}, sort_keys=True))


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_float_row_embedding_and_reordering_exact_bit_drift(backend, dtype, record_property):
    """Direct-M arithmetic must be checked across MTP serial/verify grouping.

    Differences are recorded exactly, without a fitted tolerance. Vendor direct-M
    variants remain unqualified when row grouping changes bits. The fixed CUTLASS
    template is expected to preserve a row's reduction regardless of row position.
    This component property is still insufficient for full-model MTP acceptance.
    """
    generator = torch.Generator(device='cuda').manual_seed(7132)
    anchors = torch.randn(2, 2560, device='cuda', dtype=torch.float16, generator=generator) * .1
    weight = torch.randn(160, 2560, device='cuda', dtype=torch.float16, generator=generator) * .1
    bit_type = torch.int16 if dtype == torch.float16 else torch.int32
    diagnostics = []
    with dense.Context('cuda:0', backend=backend, max_plans=4) as context:
        single_x = torch.empty_like(anchors[:1])
        single_out = torch.empty((1, 160), device='cuda', dtype=dtype)
        single_x.copy_(anchors[:1])
        single_plan = dense.prepare(context, single_x, weight, single_out)
        serial = []
        for index in range(2):
            single_x.copy_(anchors[index:index + 1])
            single_plan.run(single_x, weight, single_out)
            serial.append(single_out.cpu()[0])
        for rows in (4, 32, 64):
            x = torch.randn(rows, 2560, device='cuda', dtype=torch.float16, generator=generator) * .1
            out = torch.empty(rows, 160, device='cuda', dtype=dtype)
            x[0].copy_(anchors[0])
            x[-1].copy_(anchors[1])
            plan = dense.prepare(context, x, weight, out)
            for reordered in (False, True):
                if reordered:
                    x.copy_(x.flip(0))
                plan.run(x, weight, out)
                actual = out.cpu()
                assert torch.isfinite(actual).all()
                for index, position in enumerate((rows - 1, 0) if reordered else (0, rows - 1)):
                    mismatch = int((actual[position].view(bit_type) != serial[index].view(bit_type)).sum().item())
                    diagnostics.append({'rows': rows, 'row': position, 'anchor': index,
                        'reordered': reordered, 'serial_R1_bit_mismatches': mismatch,
                        'serial_R1_max_abs': float((actual[position].float() - serial[index].float()).abs().max().item())})
        record_property('dense_row_regroup_diagnostic', json.dumps({
            'backend': backend, 'output_dtype': str(dtype), 'observations': diagnostics,
            'serial_verify_component_bits_match': all(d['serial_R1_bit_mismatches'] == 0 for d in diagnostics),
            'quality_accepted': False, 'MTP_qualified': False, 'P3_qualified': False,
            'risk': 'different M-dependent reductions can change MTP serial/verify results'}, sort_keys=True))
        if backend == 'cutlass':
            assert all(d['serial_R1_bit_mismatches'] == 0 for d in diagnostics)
