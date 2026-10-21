"""Bookkeeping contracts for the bounded canonical head comparison tool."""

import argparse
import importlib.util
import json
import sys
import threading
import weakref
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "gemma4_cuda_head_bench.py"
_SPEC = importlib.util.spec_from_file_location("gemma_head_bench", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_orders_balance_positions_and_reverse_each_pair():
    names = ("reference", "reuse2", "reuse4", "reuse8", "reuse16")
    orders = list(bench._orders(names, len(names)))
    for index in range(0, len(orders), 2):
        assert orders[index + 1] == list(reversed(orders[index]))
    for position in range(len(names)):
        assert Counter(order[position] for order in orders) == Counter({name: 2 for name in names})


@pytest.mark.parametrize("text", ["", "1", "3,3", "3,5,11,16", "3,wat", "128"])
def test_verify_width_configuration_is_bounded(text):
    with pytest.raises(argparse.ArgumentTypeError):
        bench._widths(text)
    assert bench._widths("3,5,11") == (3, 5, 11)
    assert bench._widths("16") == (16,)


def test_timed_wrapper_keeps_complete_ids_and_drains_both_ends(monkeypatch):
    class Engine:
        drains = 0
        def drain(self):
            self.drains += 1
        def generate(self, prompt, count, sampling, callback, *, draft, stop_eos):
            assert prompt == [5] and count == 3 and sampling is None
            assert draft is True and stop_eos is True
            callback([7])
            callback([8, 9])
            return {"decode_s": 1.5}
    ticks = iter((10., 12.))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    engine = Engine()
    output, stats = bench._run(engine, [5], 3, None, True)
    assert output == [7, 8, 9] and engine.drains == 2
    assert stats["wall_s"] == 2. and stats["output_tokens"] == 3
    assert stats["decode_tps"] == 2 / 1.5


def test_draft_only_restricts_timing_without_changing_reference_capture_policy():
    assert bench._timed_modes(True, 0) == (True,)
    assert bench._timed_modes(True, 1) == (True,)
    assert bench._timed_modes(False, 0) == (False, True)
    assert bench._timed_modes(False, 1) == (True, False)


@pytest.mark.parametrize("sampled", [False, True])
def test_capture_retains_separate_mode_ids_and_restores_head_borrowers(monkeypatch, sampled):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)))
    def target(hidden):
        pass
    def draft(hidden):
        pass

    class Hidden:
        shape = (1, 2)
        def detach(self):
            return self
        def clone(self):
            return self

    engine = SimpleNamespace(target=SimpleNamespace(project_logits=target),
                             head=SimpleNamespace(project_logits=draft), drain=lambda: None)
    hidden = Hidden()
    def graph_forward(*args, **kwargs):
        return SimpleNamespace(hidden=hidden)
    engine.graphs = SimpleNamespace(forward=graph_forward)

    def run(engine, prompt, tokens, sampling, drafting):
        engine.target.project_logits(hidden)
        engine.graphs.forward([1], object(), output_taps=drafting)
        if drafting:
            engine.head.project_logits(hidden)
        return [1, 2, 4 if drafting else 3], {"cached": len(prompt)}

    monkeypatch.setattr(bench, "_run", run)
    if sampled:
        ids, captured, _ = bench._capture(engine, [5], 3, object())
        assert ids == {"serial": [1, 2, 3], "dflash": [1, 2, 4]}
        assert set(captured) == {"target", "dflash"}
    else:
        with pytest.raises(RuntimeError, match="complete serial token IDs"):
            bench._capture(engine, [5], 3, None)
    assert engine.target.project_logits is target
    assert engine.head.project_logits is draft
    assert engine.graphs.forward is graph_forward


@pytest.mark.parametrize("draft_only", [False, True])
def test_references_use_matching_head_and_mode_without_rebasing_cache_failures(monkeypatch, draft_only):
    engine = SimpleNamespace(w=SimpleNamespace(embed=None), drain=lambda: None)
    outputs = {(False, "reference"): [1, 2], (True, "reference"): [3, 4],
               (False, "reuse8"): [1, 2], (True, "reuse8"): [3, 4]}
    calls = []

    def run(engine, prompt, tokens, sampling, draft):
        key = draft, engine.w.embed
        calls.append(key)
        return list(outputs[key]), {"cached": len(prompt)}

    monkeypatch.setattr(bench, "_run", run)
    monkeypatch.setattr(bench, "_warm_request", lambda engine, head, *args: setattr(engine.w, "embed", head) or {})
    modes = bench._timed_modes(draft_only, 0)
    warm = {"serial": [1, 2], "dflash": [3, 4]}
    heads = {"reference": "reference", "reuse8": "reuse8"}
    references = bench._references(engine, heads, modes, [9], 2, object(), warm)
    assert calls == [(draft, name) for draft in modes for name in heads]
    assert {key: value[0] for key, value in references.items()} == {
        key: value for key, value in outputs.items() if key[0] in modes}
    outputs[True, "reference"][-1] += 1
    with pytest.raises(RuntimeError, match="reference/dflash.*complete token IDs"):
        bench._references(engine, heads, (True,), [9], 2, object(), warm)
    outputs[True, "reference"] = [3, 4]
    monkeypatch.setattr(bench, "_run", lambda *args: ([3, 4], {"cached": 0}))
    with pytest.raises(RuntimeError, match="complete warm prefix"):
        bench._references(engine, heads, (True,), [9], 2, object(), warm)


@pytest.mark.parametrize("sampled", [False, True])
def test_exact_head_candidate_requires_complete_incumbent_identity_in_its_mode(monkeypatch, sampled):
    engine = SimpleNamespace(w=SimpleNamespace(embed=None), drain=lambda: None)
    monkeypatch.setattr(bench, "_run", lambda *args: ([1, 2, 4], {"cached": 1}))
    monkeypatch.setattr(bench, "_warm_request", lambda engine, head, *args: setattr(engine.w, "embed", head) or {})
    with pytest.raises(RuntimeError, match="reuse8/dflash.*complete token IDs"):
        bench._references(engine, {"reuse8": object()}, (True,), [5], 3, object() if sampled else None,
                          {"serial": [1, 2, 3], "dflash": [1, 2, 3]})


def _main_fixture(monkeypatch, tmp_path, *, corrupt=None):
    """Exercise the real tool control flow with no Torch/GPU/model imports."""
    for name in ("config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    engines = []

    class State:
        position = 1
        def reset(self):
            self.position = 0

    class Graphs:
        def __init__(self, target, state, *, max_rows, span=1):
            owner = target.owner
            assert not any(bank() is not None for bank in owner.bank_refs)
            assert state is owner._state
            self.bound_head = target.weights.embed
            self.target, self.state, self.span = target, state, span
            self.graphs = {}
            owner.bank_refs.append(weakref.ref(self))
        def prepare(self, *, taps):
            owner = self.target.owner
            assert owner.in_stream and self.state.position == 0 and owner._draft.position == 0
            assert owner._prefix is None and self.bound_head is owner.w.embed
            assert taps == (False, True)
            owner.pending_initialization = True
            self.graphs["captured"] = self.bound_head
            return len(self.graphs)

    @contextmanager
    def stream(selected):
        selected.owner.in_stream = True
        try:
            yield
        finally:
            selected.owner.in_stream = False

    class Engine:
        def __init__(self, model, **kwargs):
            self.rows = kwargs["depth"] + 1
            self.context_window = kwargs["context"]
            self.max_rows = self.rows
            self.w = SimpleNamespace(embed=SimpleNamespace(incumbent="reference"),
                                     numerics="ct_nvfp4", nbytes=1)
            self.chat_template = SimpleNamespace(render=lambda messages, **kwargs: messages[0]["content"])
            self.calls = Counter()
            self.closed = False
            self._state, self._draft = State(), State()
            self._stream = SimpleNamespace(owner=self)
            self._prefix_lock = threading.Lock()
            self._prefix = "stale logits"
            self.head = object()
            self.bank_refs = []
            self.target = SimpleNamespace(owner=self, weights=self.w)
            self.in_stream = self.pending_initialization = False
            self.graphs = Graphs(self.target, self._state, max_rows=self.rows)
            engines.append(self)
        def drain(self):
            self.pending_initialization = False
        def close(self):
            self.closed = True

    def run(engine, prompt, tokens, sampling, draft):
        name = engine.w.embed
        assert engine.graphs.bound_head is name
        assert not engine.pending_initialization and not engine.in_stream
        key = draft, name, tuple(prompt)
        engine.calls[key] += 1
        output = [prompt[0], int(draft), 0, engine.rows if draft else 0] if sampling else [1, 2, 3, 4]
        if corrupt == "greedy-width" and sampling is None and engine.rows == 5:
            output[-1] += 1
        if corrupt == "sampled-serial-width" and sampling and not draft and engine.rows == 5:
            output[-1] += 1
        if corrupt == "sampled-head" and name == "reuse8":
            output[-1] += 1
        if corrupt == "sampled-replay" and name == "reuse8" and engine.calls[key] >= 4:
            output[-1] += 1
        cached = len(prompt) if engine._prefix == tuple(prompt) else 0
        engine._prefix = tuple(prompt)
        if corrupt == "timed-cache" and name == "reuse8" and engine.calls[key] >= 4:
            cached = 0
        return output, {"cached": cached, "decode_s": 1., "wall_s": 2.}

    def capture(engine, prompt, tokens, sampling):
        serial, serial_stats = run(engine, prompt, tokens, sampling, False)
        drafted, draft_stats = run(engine, prompt, tokens, sampling, True)
        return {"serial": serial, "dflash": drafted}, {}, {"serial": serial_stats, "dflash": draft_stats}

    modules = {
        "torch": SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test"),
                                 inference_mode=nullcontext,
                                 cuda=SimpleNamespace(get_device_name=lambda: "test", empty_cache=lambda: None,
                                                      stream=stream,
                                                      get_device_capability=lambda: (12, 1))),
        "triton": SimpleNamespace(__version__="test"),
        "tensorfold.engine.exact_sampling": SimpleNamespace(Sampling=lambda *args, **kwargs: object()),
        "tensorfold.families.gemma4.cuda.engine": SimpleNamespace(Gemma4Engine=Engine),
        "tensorfold.families.gemma4.cuda.head": SimpleNamespace(
            prepare_head=lambda incumbent, *, reuse_rows: f"reuse{reuse_rows}"),
        "tensorfold.families.gemma4.cuda.graphs": SimpleNamespace(Graphs=Graphs),
        "tensorfold.token_metadata": SimpleNamespace(load_tokenizer=lambda path: SimpleNamespace(
            encode=lambda text, **kwargs: SimpleNamespace(ids=[len(text)]))),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(bench, "_capture", capture)
    monkeypatch.setattr(bench, "_screen", lambda *args: ("reuse8", {}))
    monkeypatch.setattr(bench, "_run", run)
    out = tmp_path / "evidence.jsonl"
    argv = [str(tmp_path), "--drafter", str(tmp_path), "--artifact-id", "cpu-fixture", "--out", str(out),
            "--rows", "3,5", "--all-heads", "--code-tokens", "4", "--chat-tokens", "4"]
    return argv, out, engines


@pytest.mark.parametrize("draft_only", [False, True])
def test_sampled_main_replays_each_geometry_head_and_mode(monkeypatch, tmp_path, draft_only):
    argv, out, engines = _main_fixture(monkeypatch, tmp_path)
    assert bench.main([*argv, "--sampled", *(["--draft-only"] if draft_only else [])]) == 0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    references = {(row["verify_rows"], row["workload"], row["head"], row["draft"]): row["reference_ids"]
                  for row in records if row["kind"] == "request_reference"}
    integrated = [row for row in records if row["kind"] == "integrated"]
    assert len(references) == 2 * 2 * 5 * (1 if draft_only else 2)
    assert len(integrated) == 2 * len(references)
    for row in integrated:
        key = row["verify_rows"], row["workload"], row["head"], row["draft"]
        assert row["same_reference_ids"] and row["output_ids"] == references[key] == row["reference_ids"]
        if draft_only:
            assert row["draft"] is True
    assert all(row["reference_ids"] != row["dflash_reference_ids"]
               for row in records if row["kind"] == "warmup")
    for rows in (3, 5):
        for workload in ("code", "chat"):
            for draft in (True,) if draft_only else (False, True):
                incumbent = references[rows, workload, "reference", draft]
                assert all(ids == incumbent for (width, task, _, mode), ids in references.items()
                           if (width, task, mode) == (rows, workload, draft))
    for workload in ("code", "chat"):
        assert references[3, workload, "reference", True] != references[5, workload, "reference", True]
        if not draft_only:
            assert references[3, workload, "reference", False] == references[5, workload, "reference", False]
    assert all(engine.closed for engine in engines)
    assert all(sum(bank() is not None for bank in engine.bank_refs) == 1 for engine in engines)
    preparations = [row for row in records if row["kind"] == "request_preparation"]
    assert len(preparations) == len(integrated)
    for row in integrated:
        at = records.index(row)
        prepared = records[at - 1]
        assert prepared["kind"] == "request_preparation" and prepared["warm_ids"] == row["reference_ids"]
        assert prepared["head"] == row["head"] and prepared["draft"] == row["draft"]
        assert row["cached"] == 1
    assert any(row["binding"]["changed"] and row["warm_stats"]["cached"] == 0 for row in preparations)


@pytest.mark.parametrize("corrupt, sampled, message", [
    ("sampled-replay", True, "complete request changed reference token IDs"),
    ("timed-cache", True, "timed request did not use.*complete warm prefix"),
    ("greedy-width", False, "serial token IDs changed across prepared verify widths"),
    ("sampled-head", True, "untimed reference changed complete token IDs"),
    ("sampled-serial-width", True, "serial token IDs changed across prepared verify widths"),
])
def test_main_preserves_complete_replay_cache_and_greedy_width_failures(monkeypatch, tmp_path, corrupt,
                                                                     sampled, message):
    argv, out, engines = _main_fixture(monkeypatch, tmp_path, corrupt=corrupt)
    with pytest.raises(RuntimeError, match=message):
        bench.main([*argv, *(["--sampled"] if sampled else [])])
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records[-1]["kind"] == "failure"
    assert not any(row["kind"] == "complete" for row in records)
    assert all(engine.closed for engine in engines)


def test_capture_clones_graph_static_output_outside_capture_and_restores_inherited_hooks(monkeypatch):
    capturing = False
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing)))

    class Hidden:
        def __init__(self, rows, value):
            self.shape, self.value = (rows, 2), value
        def detach(self):
            return self
        def clone(self):
            assert not capturing, "observer must not inject clone kernels into captured production graphs"
            return Hidden(self.shape[0], self.value)

    static = Hidden(3, 17)
    class Target:
        def project_logits(self, hidden):
            return None
    class Graphs:
        def forward(self, *args, **kwargs):
            nonlocal capturing
            capturing = True
            try:
                engine.target.project_logits(static)
            finally:
                capturing = False
            return SimpleNamespace(hidden=static)

    engine = SimpleNamespace(target=Target(), graphs=Graphs(), head=SimpleNamespace(project_logits=lambda h: None),
                             drain=lambda: None)
    original_draft = engine.head.project_logits

    def run(engine, prompt, tokens, sampling, draft):
        engine.target.project_logits(Hidden(1, 2))
        engine.graphs.forward([1], object(), output_taps=draft)
        static.value = 99  # the next replay invalidates the original output buffer
        if draft:
            engine.head.project_logits(Hidden(1, 3))
        return [1, 2], {"cached": 1}

    monkeypatch.setattr(bench, "_run", run)
    _, inputs, stats = bench._capture(engine, [5], 2, None)
    assert inputs["target"].shape[0] == 3 and inputs["target"].value == 17
    assert static.value == 99 and stats["target_graph_calls"] == 2
    assert "project_logits" not in vars(engine.target) and "forward" not in vars(engine.graphs)
    assert engine.head.project_logits is original_draft

    primary, cleanup = RuntimeError("request failed"), OSError("drain failed")
    def fail_run(*args):
        raise primary
    def fail_drain():
        raise cleanup
    monkeypatch.setattr(bench, "_run", fail_run)
    engine.drain = fail_drain
    with pytest.raises(RuntimeError) as caught:
        bench._capture(engine, [5], 2, None)
    assert caught.value is primary and primary.__cause__ is cleanup
    assert "project_logits" not in vars(engine.target) and "forward" not in vars(engine.graphs)
    assert engine.head.project_logits is original_draft
