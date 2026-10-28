"""Optional fused stacks share one retained allocation with their reference views."""

import gc
import json

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFConcat, GGUFDense  # noqa: E402
import test_sm70_gguf_exact_actual as actual_source  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="SM70 projection stacks")


@pytest.mark.parametrize("counts", [(32, 4), (160, 96, 6, 6), (192, 32, 32, 64, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_projection_stack_reference_views_preserve_bits_and_fused_matches_fp64(counts, dtype):
    torch.manual_seed(1627)
    weights = [torch.randn((n, 256), device="cuda", dtype=torch.float16) * .02 for n in counts]
    originals = [GGUFDense(w.clone()) for w in weights]
    stack = GGUFConcat([GGUFDense(w) for w in weights])
    assert stack.fused is not None
    assert len(stack.device_tensors()) == 1
    retained = stack.device_tensors()[0]
    assert stack.nbytes() == sum(w.numel() * w.element_size() for w in weights)
    assert all(part.weight.untyped_storage().data_ptr() == retained.untyped_storage().data_ptr()
               for part in stack.parts)
    assert all(part.weight.data_ptr() % 512 == 0 for part in stack.parts)
    x = torch.randn((18, 256), device="cuda", dtype=torch.float16)
    expected = torch.cat([part.matmul_reference(x, f32=dtype == torch.float32) for part in originals], dim=1)
    reference = stack.matmul_reference(x, f32=dtype == torch.float32)
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    stack.implementation = "fused"
    backing = torch.full((18, stack.n + 19), -999, device="cuda", dtype=dtype)
    actual = stack.matmul(x, out=backing[:, 7:7 + stack.n])
    oracle = x.cpu().double() @ torch.cat(weights).cpu().double().T
    torch.testing.assert_close(actual.cpu().double(), oracle, rtol=.001, atol=3e-5)
    assert torch.all(backing[:, :7] == -999) and torch.all(backing[:, 7 + stack.n:] == -999)
    serial = torch.cat([stack.matmul(x[row:row + 1], f32=dtype == torch.float32) for row in range(18)])
    torch.testing.assert_close(actual, serial, rtol=0, atol=0)


def test_fused_projection_stack_graph_tracks_live_inputs_and_keeps_its_weight_owner():
    torch.manual_seed(1628)
    stack = GGUFConcat([GGUFDense(torch.randn((n, 256), device="cuda", dtype=torch.float16) * .02)
                        for n in [32, 4]])
    stack.implementation = "fused"
    x = torch.randn((3, 256), device="cuda", dtype=torch.float16)
    stack.matmul(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = stack.matmul(x)
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, stack.matmul(x), rtol=0, atol=0)


def test_unaligned_generic_stack_retains_its_reference_owners_and_refuses_fused_selection():
    stack = GGUFConcat([GGUFDense(torch.zeros((n, 33), device="cuda", dtype=torch.float16)) for n in [1, 2]])
    assert stack.fused is None and len(stack.device_tensors()) == 2
    stack.implementation = "fused"
    with pytest.raises(ValueError, match="aligned"):
        stack.matmul(torch.zeros((1, 33), device="cuda", dtype=torch.float16))


@pytest.mark.parametrize("counts", [(32, 4), (160, 96, 6, 6), (192, 32, 32, 64, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 3, 18, 32, 33])
def test_shared_native32_pad_and_fused_copy_preserve_original_part_bits(counts, dtype, rows):
    torch.manual_seed(1629)
    stack = GGUFConcat([GGUFDense(torch.randn((n, 256), device="cuda", dtype=torch.float16) * .02)
                        for n in counts])
    stack.implementation = "native32"
    x = torch.randn((rows, 256), device="cuda", dtype=torch.float16)
    expected = torch.cat([part.matmul_reference(x, f32=dtype == torch.float32) for part in stack.parts], dim=1)
    actual = stack.matmul(x, f32=dtype == torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if rows:
        serial = torch.cat([stack.matmul(x[row:row + 1], f32=dtype == torch.float32) for row in range(rows)])
        torch.testing.assert_close(actual, serial, rtol=0, atol=0)
    if rows == 33:
        with pytest.raises(RuntimeError, match="at most32"):
            stack.matmul_native32(x)


def test_native32_stack_replays_dynamic_inputs_with_strided_concat_outputs():
    torch.manual_seed(1630)
    stack = GGUFConcat([GGUFDense(torch.randn((n, 256), device="cuda", dtype=torch.float16) * .02)
                        for n in [32, 4]])
    stack.implementation = "native32"
    x = torch.randn((18, 512), device="cuda", dtype=torch.float16)[:, ::2]
    backing = torch.full((18, stack.n + 19), -997, device="cuda", dtype=torch.float32)
    output = backing[:, 5:5 + stack.n]
    stack.matmul(x, out=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stack.matmul(x, out=output)
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.cat([part.matmul_reference(x, f32=True) for part in stack.parts], dim=1)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.all(backing[:, :5] == -997) and torch.all(backing[:, 5 + stack.n:] == -997)


class _DirectOutputs(TorchDispatchMode):
    """Observe the selected ATen output-layout contract at the native boundary."""
    def __init__(self):
        self.outputs = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if function._schema.name == "aten::mm" and "out" in kwargs:
            value = kwargs["out"]
            self.outputs.append((tuple(value.shape), tuple(value.stride()), value.data_ptr()))
        return function(*args, **kwargs)


def _current_stack(plan):
    originals = [GGUFDense(actual_source.current_dense(part)) for part in plan.parts]
    # Keep the originals as the independent decoded-weight oracle. The concat
    # owns its joined allocation and views, so input/owner mutations are visible.
    stack = GGUFConcat([GGUFDense(part.weight.clone()) for part in originals])
    stack.implementation = "native32"
    assert stack.nbytes() == sum(part.nbytes() for part in originals)
    return stack, originals


def _current_concat_compare(plan, dtype, rows, seed):
    stack, originals = _current_stack(plan)
    x = actual_source.current_inputs(rows, stack.k, seed)
    before = x.clone()
    expected = torch.cat([part.matmul_reference(x, f32=dtype == torch.float32) for part in originals], dim=1)
    output = torch.empty((rows, stack.n), device="cuda", dtype=dtype)
    with _DirectOutputs() as observed:
        assert stack.matmul(x, out=output) is output
    actual_source.bits(output, expected)
    actual_source.bits(x, before)
    for part, original in zip(stack.parts, originals):
        actual_source.bits(part.weight, original.weight)
    if rows == 32:
        assert x.data_ptr() % 512 == output.data_ptr() % 512 == 0
        direct = ((dtype == torch.float32 and stack.k == 10240 and tuple(p.n for p in stack.parts) == (320, 4))
                  or (dtype == torch.float16 and stack.k == 2560
                      and tuple(p.n for p in stack.parts) == (1536, 256, 256, 512, 128)))
        if direct:
            offset, contracts = 0, []
            for part in stack.parts:
                contracts.append(((rows, part.n), (stack.n, 1),
                                  output.data_ptr() + offset * output.element_size()))
                offset += part.n
            assert observed.outputs == contracts
        else:
            assert observed.outputs == []
        # The same actual weights with a disjoint, strided output exercise the
        # generic path rather than silently testing only its direct-output case.
        backing = torch.full((rows, stack.n + 13), -719, device="cuda", dtype=dtype)
        with _DirectOutputs() as generic:
            stack.matmul(x, out=backing[:, 5:5 + stack.n])
        assert generic.outputs == []
        actual_source.bits(backing[:, 5:5 + stack.n], expected)
        assert bool((backing[:, :5] == -719).all()) and bool((backing[:, 5 + stack.n:] == -719).all())
    if rows == 33:
        with pytest.raises(RuntimeError, match="at most32"):
            stack.matmul_native32(x)
    return {"parts": [{"tensor": p.source.name, "encoding": p.source.encoding,
                       "shape": (p.n, p.k), "columns": p.columns} for p in plan.parts],
            "rows": rows, "dtype": str(dtype)}


@pytest.mark.parametrize("role", ["target", "mtp"])
@pytest.mark.parametrize("rank", range(8))
def test_current_checkpoint_concat_unprepared_all_tp8_views(role, rank, record_property):
    checked = [{"rank": rank, "layer": layer, "kind": kind,
                **_current_concat_compare(plan, dtype, 3, 1981 + rank)}
               for layer, kind, plan, dtype in actual_source.current_concats(role, rank)]
    record_property("current_source_cases", json.dumps(checked, sort_keys=True))


@pytest.mark.parametrize("role", ["target", "mtp"])
@pytest.mark.parametrize("rows", actual_source.CURRENT_ROWS)
def test_current_checkpoint_concat_unprepared_row_and_direct_stride_boundaries(role, rows, record_property):
    checked = [{"rank": 3, "layer": layer, "kind": kind,
                **_current_concat_compare(plan, dtype, rows, 2092 + rows)}
               for layer, kind, plan, dtype in actual_source.current_concats(role, 3)]
    record_property("current_source_cases", json.dumps(checked, sort_keys=True))


@pytest.mark.parametrize("role", ["target", "mtp"])
def test_current_checkpoint_concat_temporary_and_weight_owners_retire(role, record_property):
    target, _mtp, _receipt = actual_source.current_context(3)
    # Reuse the maintained target/MTP dense component admission equations;
    # this is not a new allocator tolerance or permission to credit capacity.
    width = target.cfg.streams * target.cfg.hidden
    allowance = 32 * (max(width, target.head.n) if role == "target" else width) * 4
    peaks = []
    for _layer, _kind, plan, dtype in actual_source.current_concats(role, 3):
        stack, originals = _current_stack(plan)
        owners = tuple(stack.device_tensors())
        x = actual_source.current_inputs(3, stack.k, 3109)
        expected = torch.cat([p.matmul_reference(x, f32=dtype == torch.float32) for p in originals], dim=1)
        out = torch.empty_like(expected)
        stack.matmul(x, out=out)  # Warm the exact native/GEMM paths before accounting.
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            stack.matmul(x, out=out)
            consumer = out.clone()  # Read before reused scratch can be clobbered.
            scratch = torch.full((32, stack.k), 193, device="cuda", dtype=torch.float16)
            del scratch
            gc.collect()
            actual_source.bits(consumer, expected)
            del consumer
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == baseline
            assert all(value is owner for value, owner in zip(stack.device_tensors(), owners))
        peak = torch.cuda.max_memory_allocated() - baseline
        assert peak <= allowance
        peaks.append({"kind": _kind, "temporary_peak_bytes": peak, "existing_dense_allowance_bytes": allowance})
    record_property("current_temporary_peaks", json.dumps(peaks, sort_keys=True))
