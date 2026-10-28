"""Bounded case/control contracts; actual CUDA arithmetic is Root-only pending."""
import json
import pytest
from tools import v100_dense_shape_diagnostic as d


def test_source_geometry_outputs_and_small_m_shapes_are_explicit():
    assert len(d.GEOMETRIES) == 7
    by_role = {g.role: g for g in d.GEOMETRIES}
    assert (by_role["hc_down"].n, by_role["hc_down"].k, by_role["hc_down"].output) == (320, 10240, "float32")
    assert by_role["hc_inject"].n == 4 and by_role["vocab_head_rank"].n == 31040
    assert by_role["mtp_fc_h_physical"].rows == (4, 8, 12)
    assert d.shapes(1) == (1, 4, 8, 16, 32)
    assert d.shapes(3) == (3, 4, 8, 16, 32)
    assert d.shapes(12) == (12, 16, 32)
    assert all(d.payload_bound(g) < d.GPU_LIMIT for g in d.GEOMETRIES)
    assert d.SAMPLES == 3 and d.UNROLL == 4 and d.REPLAYS == 8


@pytest.mark.parametrize("rows", [0, 33, True, 1.5])
def test_invalid_geometry_refuses_without_gpu_or_side_effect(rows):
    with pytest.raises(ValueError):
        d.shapes(rows)


def test_actual_metric_keeps_bits_errors_and_nonfinite_guard():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[1., 2.]], dtype=torch.float16)
    b = torch.tensor([[1., 2.]], dtype=torch.float16)
    record = d.metrics(torch, a, b)
    assert record["bits_exact"] and record["max_abs"] == 0
    b[0, 1] = 3
    record = d.metrics(torch, a, b)
    assert not record["bits_exact"] and record["different_elements"] == 1 and record["max_abs"] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        d.metrics(torch, a, torch.full_like(a, float("nan")))


def test_report_bound_and_units_are_not_unbounded(tmp_path, monkeypatch):
    path = tmp_path / "bounded.json"
    d.write(path, {"ns": 123, "scope": "microdiagnostic only"})
    assert json.loads(path.read_text())["ns"] == 123
    monkeypatch.setattr(d, "REPORT_LIMIT", 4)
    with pytest.raises(ValueError, match="envelope"):
        d.write(path, {"too_large": True})



def execute_actual_failure_handler(primary, args, report, writer):
    import ast
    import inspect
    import textwrap
    from tensorfold.cleanup import raise_failures

    tree = ast.parse(textwrap.dedent(inspect.getsource(d.run)))
    actual = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                  and any(isinstance(handler.name, str) and handler.name == "error" for handler in node.handlers))
    control = ast.Try(body=[ast.Raise(exc=ast.Name(id="primary", ctx=ast.Load()), cause=None)],
                      handlers=actual.handlers, orelse=[], finalbody=[])
    module = ast.fix_missing_locations(ast.Module(body=[control], type_ignores=[]))
    scope = dict(primary=primary, args=args, report=report, write=writer,
                 diagnostic_failures=[None], raise_failures=raise_failures)
    exec(compile(module, "actual_run_failure_handler", "exec"), scope)


def test_actual_handler_retains_primary_after_successful_failure_report(tmp_path):
    from types import SimpleNamespace
    primary = RuntimeError("original GPU operation failure")
    cause, context = ValueError("original explicit cause"), OSError("original implicit context")
    primary.__cause__, primary.__context__, primary.__suppress_context__ = cause, context, True
    report = {}
    with pytest.raises(RuntimeError) as caught:
        execute_actual_failure_handler(primary, SimpleNamespace(out=tmp_path), report, d.write)
    assert caught.value is primary
    assert primary.__cause__ is cause and primary.__context__ is context and primary.__suppress_context__
    assert not any("NoneType" in note for note in getattr(primary, "__notes__", ()))
    assert json.loads((tmp_path / "failure.json").read_text())["failure"]["message"] == str(primary)


def test_actual_handler_keeps_primary_and_actual_diagnostic_write_failure(tmp_path):
    from types import SimpleNamespace
    primary, diagnostic = RuntimeError("original operation"), OSError("failure report write")
    def failed_write(_path, _value):
        raise diagnostic
    with pytest.raises(RuntimeError) as caught:
        execute_actual_failure_handler(primary, SimpleNamespace(out=tmp_path), {}, failed_write)
    assert caught.value is primary
    from tests.test_v100_backend_probe import retained_failure
    assert retained_failure(primary, diagnostic)


def fake_timing_cuda(*, exit_failure=None, fence_failure=None, reset_failure=None):
    from types import SimpleNamespace
    events = []
    class Graph:
        def __init__(self):
            events.append("created")
        def reset(self):
            events.append("reset")
            if reset_failure is not None:
                raise reset_failure
    class Context:
        def __enter__(self):
            events.append("enter")
        def __exit__(self, *_args):
            events.append("exit")
            if exit_failure is not None:
                raise exit_failure
    def synchronize():
        events.append("fence")
        if fence_failure is not None and "enter" in events:
            raise fence_failure
    cuda = SimpleNamespace(CUDAGraph=Graph, graph=lambda _graph: Context(), synchronize=synchronize)
    return SimpleNamespace(cuda=cuda), events


def test_actual_timed_body_primary_survives_context_exit_failure():
    primary, cleanup = RuntimeError("body"), OSError("capture exit")
    target, events = fake_timing_cuda(exit_failure=cleanup)
    class Operation:
        def __init__(self):
            self.calls = 0
        def call(self):
            self.calls += 1
            if self.calls > 1:
                assert self.timing_owner.graphs  # Published before accepted capture.
                raise primary
    operation = Operation()
    with pytest.raises(RuntimeError) as caught:
        d.timed(target, [operation])
    assert caught.value is primary
    from tests.test_v100_backend_probe import retained_failure
    assert retained_failure(primary, cleanup)
    assert "fence" in events and "reset" in events
    assert operation.timing_owner is not None and not operation.timing_owner.retired


@pytest.mark.parametrize("cut", ["fence", "reset"])
def test_actual_timed_retains_owner_on_uncertain_retirement(cut):
    primary, fault = RuntimeError("body"), OSError(cut)
    target, events = fake_timing_cuda(fence_failure=fault if cut == "fence" else None,
                                      reset_failure=fault if cut == "reset" else None)
    class Operation:
        def __init__(self):
            self.calls = 0
        def call(self):
            self.calls += 1
            if self.calls > 1:
                raise primary
    operation = Operation()
    with pytest.raises(RuntimeError) as caught:
        d.timed(target, [operation])
    assert caught.value is primary
    from tests.test_v100_backend_probe import retained_failure
    assert retained_failure(primary, fault)
    owner = operation.timing_owner
    assert owner is not None and owner.graphs and owner.context and not owner.retired
    assert ("reset" in events) == (cut == "reset")


def test_actual_timing_owner_releases_only_after_positive_fence_and_reset():
    from types import SimpleNamespace
    target, events = fake_timing_cuda()
    variant = SimpleNamespace()
    owner = d.TimingOwner([variant])
    owner.graphs.append(target.cuda.CUDAGraph())
    owner.retire(target)
    assert owner.retired and variant.timing_owner is None and not owner.graphs
    assert events == ["created", "fence", "reset"]
