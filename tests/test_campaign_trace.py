"""Profiling integration owns policy/markers/evidence; no model, CUDA runtime or GPU is used."""
from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location("campaign_trace_boundary", ROOT / "tools/campaign_trace.py")
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
finally:
    sys.path.pop(0)


def test_duplicate_cpu_gpu_annotation_names_keep_wall_time_and_round_domains_separate():
    def annotation(name, category, start, duration):
        return {"ph": "X", "name": name, "cat": category, "ts": start, "dur": duration}

    events = [annotation("tf.sample", "user_annotation", start, 1) for start in (0, 30, 60)]
    events += [annotation("tf.sample", "gpu_user_annotation", start, 5) for start in (5, 15, 42)]
    events += [annotation("tf.q.readback", "user_annotation", 1, 10),
               annotation("tf.q.readback", "user_annotation", 31, 5),
               annotation("tf.q.readback", "gpu_user_annotation", 2, 40),
               annotation("tf.q.readback", "gpu_user_annotation", 32, 60),
               annotation("tf.gpu_only", "gpu_user_annotation", 2, 9),
               annotation("tf.q.readback", "cpu_op", 1, 900),
               annotation("tf.sample", "cpu_op", 10, 900),
               annotation("owned-kernel", "kernel", 2, 10)]
    result = trace.summarize({"traceEvents": events})
    assert result["rounds"] == 2 and result["sampler_markers"] == 3
    assert [row["wall_us"] for row in result["per_round"]] == [30, 30]
    assert [row["gpu_busy_us"] for row in result["per_round"]] == [10, 0]
    assert result["stages"] == {"tf.q.readback": {"calls": 2, "cpu_wall_us_inclusive": 15}}
    assert result["gpu_stages"] == {"tf.q.readback": {"calls": 2, "gpu_range_us_inclusive": 100},
                                    "tf.gpu_only": {"calls": 1, "gpu_range_us_inclusive": 9}}


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    import tensorfold
    import tensorfold.engine

    cuda = ModuleType("tensorfold.cuda")
    cuda.__path__ = []
    sampling, speculative, rejection = (ModuleType(name) for name in
        ("tensorfold.cuda.sampling", "tensorfold.cuda.speculative", "tensorfold.engine.rejection"))
    alias = ModuleType("tensorfold.trace_fixture_alias")
    sampling.sample_rows = lambda *args, **kwargs: [1]
    sampling.effective_rows = lambda *args, **kwargs: ["p"]
    rejection.verify = lambda *args, **kwargs: ([], 1)
    alias.sample_rows, alias.effective_rows, alias.verify = sampling.sample_rows, sampling.effective_rows, rejection.verify

    class Chains:
        def __init__(self): pass
        def _readback(self): return "readback"
        def sample(self): return self._readback()
        def finish(self): return self._readback()

    speculative.Chains = Chains
    cuda.sampling, cuda.speculative = sampling, speculative
    for module in (cuda, sampling, speculative, rejection, alias):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(tensorfold, "cuda", cuda, raising=False)
    monkeypatch.setattr(tensorfold.engine, "rejection", rejection, raising=False)

    events, phases = [], []

    @contextmanager
    def record_function(name):
        start = len(events) * 10
        event = {"ph": "X", "name": name, "ts": start, "dur": 5, "cat": "user_annotation"}
        events.append(event)
        phases.append(name)
        yield

    class Profiler:
        def __enter__(self): return self
        def __exit__(self, *args):
            if state.exit_error is not None:
                raise state.exit_error
        def export_chrome_trace(self, path):
            state.exported = True
            if state.export_error is not None:
                raise state.export_error
            payload = list(events)
            if state.gpu_annotations:
                payload.extend({**row, "cat": "gpu_user_annotation"} for row in events)
            if state.cuda_events:
                payload.append({"ph": "X", "cat": "kernel", "name": "owned-fixture-kernel", "ts": 1, "dur": 2})
            if state.drop_marker:
                payload = [row for row in payload if not (row["name"] == "tf.sample"
                                                         and row["cat"] == "user_annotation")]
            Path(path).write_text(json.dumps({"traceEvents": payload}))

    state = SimpleNamespace(cuda_events=True, gpu_annotations=False, drop_marker=False, export_error=None, exit_error=None,
                            generation_error=None, exported=False, policies=[], bodies=[], phases=phases,
                            sampling=sampling, speculative=speculative, rejection=rejection, alias=alias,
                            events=events)

    class Head:
        def logits(self): return "logits"
        def append(self): return "append"

    class Engine:
        def __init__(self): self.head = Head()
        def generate(self, prompt, tokens, policy, callback, *, draft):
            state.policies.append(policy)
            callback([1])
            sampling.sample_rows(None, None, policy)
            for _ in range(2):
                if draft:
                    self.head.logits()
                    packet = Chains()
                    packet.sample() if policy is not None else packet.finish()
                if policy is not None and draft:
                    sampling.effective_rows(None, policy)
                    rejection.verify(None)
                else:
                    sampling.sample_rows(None, None, policy)
                callback([2])
                if draft:
                    self.head.append()
            if state.generation_error is not None and len(state.policies) == 2:
                raise state.generation_error
            return {"rounds": 2, "decode_s": .01}

    state.engine = Engine()
    state.policy = SimpleNamespace(seed=7, temperature=1.0, top_k=64, top_p=.95, min_p=0.0)
    def prepare(body, chat):
        assert chat is True
        state.bodies.append(body)
        return SimpleNamespace(prompt=[1, 2], sampling=state.policy if body["temperature"] > 0 else None)
    state.app = SimpleNamespace(prepare=prepare, engine=state.engine)
    state.args = SimpleNamespace(out=str(tmp_path / "profile"), prompts=None, prompt="test", warm=8, tokens=4,
        serial=False, serve=["serve", "/model", "--thinking"], temperature=1.0, top_k=64, top_p=.95, seed=7)
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(synchronize=lambda: phases.append("cuda.synchronize"))
    torch.profiler = SimpleNamespace(record_function=record_function,
        ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"), profile=lambda **kwargs: Profiler())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(trace.HotGate, "wait", lambda: phases.append("HotGate.wait"))
    return state


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_prepared_policy_is_identical_for_warm_and_profile_and_rounds_are_covered(boundary, temperature):
    boundary.args.temperature = temperature
    head_before = dict(vars(boundary.engine.head))
    sample_before, effective_before = boundary.sampling.sample_rows, boundary.sampling.effective_rows
    trace.trace(boundary.app, boundary.args)
    expected = boundary.policy if temperature > 0 else None
    assert boundary.policies == [expected, expected]
    assert all(policy is expected for policy in boundary.policies)
    assert boundary.bodies[0]["top_k"] == 64 and boundary.bodies[0]["top_p"] == .95
    assert boundary.bodies[0]["seed"] == 7
    summary = json.loads((Path(boundary.args.out) / "summary.json").read_text())
    assert summary["status"] == "PASS" and summary["rounds"] == 2 and summary["sampler_markers"] == 3
    calls = summary["marker_calls"]
    assert calls.get("tf.target.effective_rows", 0) == (2 if temperature else 0)
    assert calls["tf.target.sample_rows"] == (1 if temperature else 3)
    assert calls["tf.draft.logits"] == calls["tf.draft.append"] == 2
    assert calls["tf.q.packet"] == calls["tf.q.readback"] == 2
    assert calls.get("tf.rejection.verify", 0) == (2 if temperature else 0)
    assert vars(boundary.engine.head) == head_before
    assert boundary.sampling.sample_rows is sample_before and boundary.alias.sample_rows is sample_before
    assert boundary.sampling.effective_rows is effective_before and boundary.alias.effective_rows is effective_before
    assert summary["deferred_tail_s"] >= 0


@pytest.mark.parametrize("fault", ["generation", "export", "exit"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failure_restores_class_instance_and_module_bindings_and_retains_evidence(boundary, fault, error_type):
    cause, primary = OSError("original cause"), error_type("controlled failure")
    primary.__cause__ = cause
    setattr(boundary, {"generation": "generation_error", "export": "export_error", "exit": "exit_error"}[fault], primary)
    before = {name: getattr(boundary.speculative.Chains, name) for name in ("__init__", "_readback", "sample", "finish")}
    effective = boundary.sampling.effective_rows
    with pytest.raises(error_type) as caught:
        trace.trace(boundary.app, boundary.args)
    assert caught.value is primary and caught.value.__cause__ is cause
    assert not vars(boundary.engine.head)
    assert boundary.sampling.effective_rows is effective and boundary.alias.effective_rows is effective
    assert all(getattr(boundary.speculative.Chains, name) is value for name, value in before.items())
    summary = json.loads((Path(boundary.args.out) / "summary.json").read_text())
    assert summary["status"] == "FAIL" and "original cause" in summary["errors"][0]["traceback"]


@pytest.mark.parametrize("fault", ["no-cuda", "missing-marker"])
def test_absent_cuda_or_incomplete_round_trace_is_not_success(boundary, fault):
    boundary.cuda_events = fault != "no-cuda"
    boundary.drop_marker = fault == "missing-marker"
    with pytest.raises(RuntimeError, match="CUDA kernel|decode rounds"):
        trace.trace(boundary.app, boundary.args)
    assert (Path(boundary.args.out) / "trace.json").exists()
    summary = json.loads((Path(boundary.args.out) / "summary.json").read_text())
    assert summary["status"] == "FAIL"


def test_gpu_sampler_annotations_cannot_satisfy_missing_cpu_round_coverage(boundary):
    boundary.gpu_annotations, boundary.drop_marker = True, True
    with pytest.raises(RuntimeError, match="decode rounds"):
        trace.trace(boundary.app, boundary.args)
    summary = json.loads((Path(boundary.args.out) / "summary.json").read_text())
    assert summary["status"] == "FAIL" and summary["sampler_markers"] == summary["rounds"] == 0
    assert summary["gpu_kernel_events"] == 1 and summary["gpu_stages"]


def test_cli_defaults_stay_greedy_and_explicit_card_options_reach_work(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(trace.HotGate, "configure", lambda value: None)
    monkeypatch.setattr(trace, "with_app", lambda serve, port, work: work(object()))
    monkeypatch.setattr(trace, "trace", lambda app, args: seen.append(args))
    for flags in ([], ["--temperature", "1", "--top-k", "64", "--top-p", ".95", "--seed", "7"]):
        monkeypatch.setattr(sys, "argv", ["campaign_trace.py", "--out", str(tmp_path), *flags, "--", "serve", "/model"])
        assert trace.main() == 0
    assert seen[0].temperature == 0 and seen[0].top_k is None and seen[0].top_p is None and seen[0].seed is None
    assert (seen[1].temperature, seen[1].top_k, seen[1].top_p, seen[1].seed) == (1.0, 64, .95, 7)


@pytest.mark.parametrize("flags", [["--temperature", "nan"], ["--top-k", "-1"], ["--top-p", "1.1"],
                                    ["--tokens", "0"]])
def test_bad_diagnostic_policy_is_refused_before_cli_model_construction(monkeypatch, tmp_path, flags):
    monkeypatch.setattr(trace, "with_app", lambda *args: pytest.fail("bad options must precede model load"))
    monkeypatch.setattr(sys, "argv", ["campaign_trace.py", "--out", str(tmp_path), *flags, "--", "serve", "/model"])
    with pytest.raises(SystemExit) as caught:
        trace.main()
    assert caught.value.code == 2


def test_primary_and_all_profiler_publication_failures_survive_together(boundary, monkeypatch):
    primary, cause = RuntimeError("generation failed"), OSError("original native cause")
    primary.__cause__ = cause
    exit_error, export_error, publication = (OSError(name) for name in ("profiler exit", "trace export", "summary write"))
    boundary.generation_error, boundary.exit_error, boundary.export_error = primary, exit_error, export_error
    original = trace.write_json
    def write(path, value):
        if path.name == "summary.json":
            raise publication
        return original(path, value)
    monkeypatch.setattr(trace, "write_json", write)
    with pytest.raises(RuntimeError) as caught:
        trace.trace(boundary.app, boundary.args)
    assert caught.value is primary and boundary.exported
    pending, seen = [caught.value], set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        pending.extend(error.exceptions if isinstance(error, BaseExceptionGroup) else ())
        pending.extend(item for item in (error.__cause__, error.__context__) if item is not None)
    assert all(id(error) in seen for error in (primary, cause, exit_error, export_error, publication))
    assert not vars(boundary.engine.head)
