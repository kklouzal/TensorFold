"""Actual Qwen loading callback batch consumption and failure ownership."""

import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_direct_read_batch import reader
from test_nvfp4_loader_phase_control import fixture
from test_qwen36_nvfp4_load_controls import context_class


def loader(tmp_path):
    load, events, state, modules = fixture()
    load.__globals__["MLPContext"] = context_class()
    path = tmp_path / "scalars"
    path.write_bytes(b"abcdefgh")
    entries = {"expert.a": (path, 0, 4, "F32", [1]), "expert.b": (path, 4, 4, "F32", [1])}
    get_scalars, calls, _ = reader(entries)
    original = object()
    reader_type = modules["tensorfold.families.qwen3_5.cuda.weights"]._Tensors

    class Reader(reader_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.where = dict.fromkeys(entries)
            self.files = SimpleNamespace(get=self.get, get_f32_scalars=get_scalars)
            state["reader"] = self

        def get(self, name, device):
            events.append((name, device))
            return original

    modules["tensorfold.families.qwen3_5.cuda.weights"]._Tensors = Reader
    return load, events, state, modules, calls, original


def invoke(monkeypatch, load, modules, mlp):
    with monkeypatch.context() as scoped:
        for name, module in modules.items():
            scoped.setitem(sys.modules, name, module)
        with contextlib.redirect_stdout(io.StringIO()):
            return load(Path("opaque"), mlp=mlp)


def test_complete_loader_closes_after_successful_cpu_batch(tmp_path, monkeypatch):
    load, events, state, modules, calls, _ = loader(tmp_path)
    bank = object()

    def mlp(_prefix, context, _cfg):
        got = context.read_f32_scalars(["expert.b", "expert.a"], max_bytes=8)
        assert bytes(got.raw) == b"efghabcd"
        assert not state["reader"].where
        return {"moe": bank}

    result = invoke(monkeypatch, load, modules, mlp)
    assert result.layers[0].moe is bank and len(calls) == 1
    assert state["closes"] == 1 and events.index("model.norm.weight") < events.index("close")


@pytest.mark.parametrize("repeat", ["single", "batch"])
def test_successful_batch_consumes_each_name_once_before_later_reads(tmp_path, monkeypatch, repeat):
    load, events, state, modules, calls, _ = loader(tmp_path)

    def mlp(_prefix, context, _cfg):
        got = context.read_f32_scalars(["expert.b", "expert.a"], max_bytes=8)
        assert bytes(got.raw) == b"efghabcd"
        assert state["reader"].where == {}
        if repeat == "single":
            context.read("expert.a", device="cpu")
        else:
            context.read_f32_scalars(["expert.a"], max_bytes=4)

    with pytest.raises(KeyError, match="expert.a"):
        invoke(monkeypatch, load, modules, mlp)
    assert len(calls) == 1 and not any(isinstance(event, tuple) for event in events)
    assert state["closes"] == 1


def test_single_consumption_rejects_complete_later_batch_before_io(tmp_path, monkeypatch):
    load, events, state, modules, calls, original = loader(tmp_path)

    def mlp(_prefix, context, _cfg):
        assert context.read("expert.a", device="cpu") is original
        context.read_f32_scalars(["expert.b", "expert.a"], max_bytes=8)

    with pytest.raises(KeyError, match="expert.a"):
        invoke(monkeypatch, load, modules, mlp)
    assert calls == [] and events.count(("expert.a", "cpu")) == 1
    assert set(state["reader"].where) == {"expert.b"} and state["closes"] == 1


@pytest.mark.parametrize("fault", ["duplicate", "budget", "unknown"])
def test_batch_validation_failure_leaves_names_unconsumed_and_closes(tmp_path, monkeypatch, fault):
    load, _, state, modules, calls, _ = loader(tmp_path)

    def mlp(_prefix, context, _cfg):
        names = ["expert.a", "expert.b"]
        if fault == "duplicate":
            names.append("expert.a")
        elif fault == "unknown":
            names.append("absent")
        context.read_f32_scalars(names, max_bytes=7 if fault == "budget" else 8)

    with pytest.raises((ValueError, KeyError)):
        invoke(monkeypatch, load, modules, mlp)
    assert calls == [] and set(state["reader"].where) == {"expert.a", "expert.b"}
    assert state["closes"] == 1


def test_batch_read_failure_preserves_primary_and_reader_close_error(tmp_path, monkeypatch):
    load, _, state, modules, _, _ = loader(tmp_path)
    primary, cleanup = OSError("batch read failed"), RuntimeError("reader close failed")
    state["close_failure"] = cleanup

    def fail(*_args, **_kwargs):
        raise primary

    def mlp(_prefix, context, _cfg):
        state["reader"].files.get_f32_scalars = fail
        context.read_f32_scalars(["expert.a", "expert.b"], max_bytes=8)

    with pytest.raises(OSError) as caught:
        invoke(monkeypatch, load, modules, mlp)
    assert caught.value is primary and state["closes"] == 1
    assert set(state["reader"].where) == {"expert.a", "expert.b"}
    assert primary.__cause__ is cleanup


def test_owned_run_preparation_is_read_free_and_each_successful_interval_consumes_once(tmp_path, monkeypatch):
    load, _, state, modules, _, original = loader(tmp_path)
    calls = []

    def mlp(_prefix, context, _cfg):
        def plan(names, **geometry):
            assert names == ("expert.b", "expert.a")
            assert geometry == {"dtype": "U8", "field_bytes": 4, "max_read_bytes": 8}
            return SimpleNamespace(read=lambda first, stop: calls.append((first, stop)) or original)

        state["reader"].files.prepare_cpu_run = plan
        run = context.prepare_cpu_run(["expert.b", "expert.a"], dtype="U8", field_bytes=4, max_read_bytes=8)
        assert calls == [] and set(state["reader"].where) == {"expert.a", "expert.b"}
        assert run(0, 1) is original and set(state["reader"].where) == {"expert.a"}
        with pytest.raises(KeyError, match="expert.b"):
            run(0, 2)
        assert calls == [(0, 1)]
        assert run(1, 2) is original and not state["reader"].where
        return {}

    invoke(monkeypatch, load, modules, mlp)
    assert state["closes"] == 1 and calls == [(0, 1), (1, 2)]


def test_owned_run_read_failure_retains_all_names_and_primary_cleanup_chain(tmp_path, monkeypatch):
    load, _, state, modules, _, _ = loader(tmp_path)
    primary, cleanup = OSError("owned run failed"), RuntimeError("reader close failed")
    state["close_failure"] = cleanup

    def fail(*_args):
        raise primary

    def mlp(_prefix, context, _cfg):
        state["reader"].files.prepare_cpu_run = lambda *_a, **_k: SimpleNamespace(read=fail)
        run = context.prepare_cpu_run(["expert.a", "expert.b"], dtype="U8", field_bytes=4, max_read_bytes=8)
        run(0, 2)

    with pytest.raises(OSError) as caught:
        invoke(monkeypatch, load, modules, mlp)
    assert caught.value is primary and primary.__cause__ is cleanup
    assert set(state["reader"].where) == {"expert.a", "expert.b"} and state["closes"] == 1


def test_unsupported_run_does_not_consume_names_and_retains_the_ordinary_callback(tmp_path, monkeypatch):
    load, events, state, modules, _, original = loader(tmp_path)

    def mlp(_prefix, context, _cfg):
        state["reader"].files.prepare_cpu_run = lambda *_a, **_k: None
        assert context.prepare_cpu_run(["expert.a"], dtype="U8", field_bytes=4, max_read_bytes=4) is None
        assert set(state["reader"].where) == {"expert.a", "expert.b"}
        assert context.read("expert.a", device="cpu") is original
        return {}

    invoke(monkeypatch, load, modules, mlp)
    assert events.count(("expert.a", "cpu")) == 1 and state["closes"] == 1
