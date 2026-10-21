"""Launch-tool ownership/comparison controls only; no model or GPU result is simulated as qualification."""
from __future__ import annotations

import copy
import json
import threading
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools import gemma4_cuda_launch_qualify as tool


def fixture_plan(tmp_path, *, configurations=None):
    model, draft = tmp_path / "model", tmp_path / "draft"
    model.mkdir()
    draft.mkdir()
    (model / "config.json").write_text("{}")
    (draft / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    template = tmp_path / "template.jinja"
    template.write_bytes((Path(__file__).resolve().parents[1] /
                          "deploy/gb10/gemma4-chat-template-google-4d7ae498.jinja").read_bytes())
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"items": [{"id": name, "thinking": False,
                                             "messages": [{"role": "user", "content": name}]}
                                            for name in tool.proposal.WORKLOADS]}))
    paths = {"model_config": model / "config.json", "drafter_config": draft / "config.json",
             "tokenizer": model / "tokenizer.json", "template": template}
    plan = {"schema": 1, "kind": "launch-screen", "artifact_id": "sha256:" + "0" * 64,  # CPU fixture only
            "source_sha256": {name: "0" * 64 for name in tool.SOURCE_KEYS},
            "assets": {name: {"path": str(path), "sha256": tool.proposal._sha(path.read_bytes())}
                       for name, path in paths.items()},
            "prompts_file": {"path": str(prompts), "sha256": tool.proposal._sha(prompts.read_bytes())},
            "prompt_sha256": {name: tool.proposal._sha(json.dumps(list(range(count)), separators=(",", ":")).encode())
                              for name, count in tool.proposal.WORKLOADS.items()},
            "policy": "q64-g8-partial", "tokens": 64, "seed": 7, "port": 8899,
            "configurations": configurations or [{"rows": 3, "bits": 16}, {"rows": 5, "bits": 8}],
            "serial_ids": {name: {"greedy-serial": list(range(64)), "sampled-serial": list(range(100, 164))}
                           for name in tool.proposal.WORKLOADS},
            "serve_args": ["serve", str(model), "--drafter", str(draft), "--chat-template", str(template),
                           "--thinking", "--thinking-budget", "0", "--context", "0", "--vision"]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return plan, path


def test_plan_declares_all_positions_and_snapshots_actual_bound_inputs(tmp_path):
    plan, path = fixture_plan(tmp_path)
    actual, items, raw, prompt_raw, consumed = tool.load_plan(path)
    assert actual == plan and json.loads(raw) == plan and len(items) == 2
    rows = tool._matrix(plan)
    assert len(rows) == 72 and all(row["status"] == "NOT_RUN" for row in rows)
    assert [row["case"] for row in rows] == list(range(72))
    assert [row["head"] for row in rows if row["configuration"] == 0 and row["phase"] == "timed"] == [16, 8, 8, 16] * 2
    assert len(consumed) == 5 and consumed["template"]["sha256"] == tool.TEMPLATE_SHA256
    Path(plan["prompts_file"]["path"]).write_text("changed")
    assert len(json.loads(prompt_raw)["items"]) == 2
    with pytest.raises(ValueError, match="hash changed"):
        tool.load_plan(path)


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("schema", 1.0), ("kind", "holdout"), ("tokens", 63), ("tokens", True),
    ("seed", 13), ("port", False), ("artifact_id", "latest"), ("policy", "arbitrary"),
    ("configurations", []), ("configurations", [{"rows": 5, "bits": 4}]),
    ("configurations", [{"rows": True, "bits": 16}]), ("configurations", [{"rows": 7, "bits": 16}]),
    ("configurations", [{"rows": 5, "bits": 16}] * 2),
    ("configurations", [{"rows": r, "bits": b} for r, b in [(3, 8), (3, 16), (5, 8), (5, 16), (11, 8)]]),
])
def test_unsealed_or_out_of_scope_plan_refuses_before_model_work(tmp_path, field, value):
    plan, path = fixture_plan(tmp_path)
    plan[field] = value
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        tool.load_plan(path)


@pytest.mark.parametrize("option", ["--drafter-bits", "--draft-rows", "--no-thinking", "--api-key=private",
                                    "--no-drafts", "--tokenizer", "--vision-offload"])
def test_base_options_cannot_override_controlled_axes_or_global_thinking(tmp_path, option):
    plan, path = fixture_plan(tmp_path)
    plan["serve_args"].append(option)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="controlled or secret"):
        tool.load_plan(path)


@pytest.mark.parametrize("option", ["--context=4096", "--thinking-budget=5", "--drafter=other", "--chat-template=other"])
def test_duplicate_alternate_option_forms_cannot_change_bound_inputs(tmp_path, option):
    plan, path = fixture_plan(tmp_path)
    plan["serve_args"].append(option)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="single explicit"):
        tool.load_plan(path)


def test_regular_input_boundary_rejects_fifo_without_blocking_and_accepts_hf_symlink(tmp_path):
    import os

    path = tmp_path / "regular"
    path.write_bytes(b"bounded")
    link = tmp_path / "link"
    link.symlink_to(path)
    assert tool._read(link, 7) == b"bounded"
    with pytest.raises(ValueError, match="actual-byte"):
        tool._read(link, 6)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        tool._read(fifo, 10)


def runtime(monkeypatch, tmp_path, *, fault=None):
    plan, path = fixture_plan(tmp_path)
    states = {"engines": [], "requests": [], "proof": False, "timed": 0, "closed": 0}
    partial = object()
    class Chains:
        pass
    speculative = NS(_PROPOSAL_GUARD=1, Chains=Chains)
    sampling = NS(_canonical_repair_order=partial)
    class Plain:
        weight = object()
    class Head:
        def __init__(self, incumbent, rows):
            self.incumbent, self.reuse_rows, self.weight = incumbent, rows, incumbent.weight
    class Engine:
        def __init__(self, rows, bits):
            assert speculative._PROPOSAL_GUARD == 32
            assert all(item.closed for item in states["engines"])
            self.max_rows, self.context_window = rows, 262144
            self.w = NS(numerics="ct_nvfp4", embed=Head(Plain(), 16))
            self.head, self.vision, self.graphs = NS(bits=bits), object(), object()
            self.capacity_plan = {"context_window": 262144, "cpu_fixture": True}
            self._prefix, self._prefix_lock, self.closed = None, threading.Lock(), False
            states["engines"].append(self)
        def drain(self):
            if fault == "drain":
                raise RuntimeError("native drain failure")
        def generate(self, prompt, count, policy, receive, *, draft, stop_eos, penalties):
            assert count == 64 and not stop_eos and penalties is None and speculative._PROPOSAL_GUARD == 8
            if fault == "request":
                receive([1, 2])
                raise RuntimeError("native request failure")
            greedy = policy is None
            if draft and not greedy:
                output = [1000 + self.max_rows * 100 + self.head.bits + j for j in range(64)]
                if fault == "head-drift" and self.w.embed.reuse_rows == 8:
                    output[-1] += 1
            else:
                output = list(range(64)) if greedy else list(range(100, 164))
                if fault == "serial-drift" and self.max_rows == 5:
                    output[-1] += 1
            cached = len(prompt) if self._prefix == tuple(prompt) else 0
            self._prefix = tuple(prompt)
            if fault == "short":
                output.pop()
            if fault == "cache" and self.w.embed.reuse_rows == 8:
                cached = 0
            states["requests"].append((self.max_rows, self.head.bits, greedy, draft, states["proof"]))
            if not states["proof"]:
                states["timed"] += 1
            receive(output)
            return {"rounds": 64 if not draft else 64 // self.max_rows + 1, "drafted": 2 * self.max_rows if draft else 0,
                    "accepted": self.max_rows if draft else 0, "min_rows": 1, "cached": cached, "decode_s": .1}
        def close(self):
            assert speculative._PROPOSAL_GUARD == 32
            self.closed = True
            states["closed"] += 1
    class App:
        def __init__(self, engine):
            self.engine = engine
        def prepare(self, body, chat):
            assert chat and body["ignore_eos"] and body["chat_template_kwargs"] == {"enable_thinking": False}
            name = body["messages"][0]["content"]
            prompt = list(range(tool.proposal.WORKLOADS[name]))
            if fault == "prompt":
                prompt[-1] += 1
            policy = NS(seed=7, temperature=1., top_k=64, top_p=.95, min_p=0.) if body["temperature"] else None
            return NS(prompt=prompt, max_tokens=64, thinking=False, ignore_eos=True, grammar=None, vision=None,
                      penalties=None, sampling=policy)
    modules = {name: NS() for name in tool.SOURCES}
    modules[tool.proposal.SOURCES[0]], modules[tool.proposal.SOURCES[1]] = speculative, sampling
    modules[tool.proposal.SOURCES[2]] = NS(Gemma4Engine=Engine)
    modules["tensorfold.families.gemma4.cuda.head"] = NS(_RowsHead=Head,
        prepare_head=lambda incumbent, *, reuse_rows: Head(incumbent, reuse_rows))
    def with_app(argv, port, work):
        assert port == 8899
        engine = Engine(int(tool._option(argv, "--draft-rows")), int(tool._option(argv, "--drafter-bits")))
        try:
            work(App(engine))
        finally:
            engine.close()
    def bind(engine, head):
        changed = engine.w.embed is not head
        engine.w.embed = head
        if changed:
            engine._prefix = None
        return {"changed": changed, "seconds": 0., "prepared_graphs": 2}
    @contextmanager
    def inputs(engine):
        assert not states["proof"]
        states["proof"] = True
        try:
            yield {"target": object(), "dflash": object()}, {"graph_calls": 1}
        finally:
            states["proof"] = False
    @contextmanager
    def diagnostics(*args):
        engine = states["engines"][-1]
        yield {"q_packets": 1, "q_rows": engine.max_rows - 1, "q_sha256": f"{engine.max_rows}-{engine.head.bits}"}
    monkeypatch.setattr(tool, "_sources", lambda modules: plan["source_sha256"] if fault != "source" else {})
    original_import = tool.importlib.import_module
    monkeypatch.setattr(tool.importlib, "import_module", lambda name: modules[name] if name in modules else original_import(name))
    monkeypatch.setattr(tool.proposal, "with_app", with_app)
    monkeypatch.setattr(tool.proposal, "diagnostics", diagnostics)
    monkeypatch.setattr(tool.head_bench, "_bind_head", bind)
    monkeypatch.setattr(tool, "_inputs", inputs)
    monkeypatch.setattr(tool, "_bits", lambda *a, **k: {"target": {"verified": True}})
    monkeypatch.setattr(tool.proposal.HotGate, "wait", lambda: 0.)
    monkeypatch.setattr(tool.proposal.HotGate, "configure", lambda path: None)
    monkeypatch.setattr(tool.proposal, "host_snapshot", lambda: {"cpu_fixture": True})
    monkeypatch.setattr(tool.proposal.Progress, "update", lambda *a, **k: None)
    # Only runtime metadata APIs are controlled; there is no GPU/model operation in this unit fixture.
    import torch
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU fixture")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    return plan, path, states, speculative, sampling


def test_launch_uses_fresh_owners_truthful_modes_and_exact_comparison_scopes(monkeypatch, tmp_path):
    _plan, path, states, speculative, sampling = runtime(monkeypatch, tmp_path)
    original = speculative.Chains, sampling._canonical_repair_order
    assert tool.main(["--plan", str(path), "--out", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "PASS" and len(report["records"]) == 72 and states["closed"] == 2
    assert all(row["status"] == "RETIRED" for row in report["configurations"])
    assert all(row["comparison"] == "PASS" for row in report["records"])
    assert all(states["engines"][i].closed for i in (0, 1))
    assert speculative._PROPOSAL_GUARD == 1 and (speculative.Chains, sampling._canonical_repair_order) == original
    assert any(not greedy and not draft for _r, _b, greedy, draft, _p in states["requests"])
    sampled = [r for r in report["records"] if r["phase"] == "timed"]
    assert sampled[0]["token_ids"] != sampled[-1]["token_ids"]  # legitimate cross-geometry/FP8 difference
    assert len(states["requests"]) == 72 and states["timed"] == 32  # warm/timed have no proof observers


@pytest.mark.parametrize("fault", ["source", "prompt", "head-drift", "serial-drift", "short", "cache", "request", "drain"])
def test_failures_preserve_full_partial_and_not_run_positions_and_retire_owned_engine(monkeypatch, tmp_path, fault):
    _plan, path, states, speculative, _sampling = runtime(monkeypatch, tmp_path, fault=fault)
    with pytest.raises((RuntimeError, ValueError)):
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "ERROR" and len(report["records"]) == 72
    assert any(row["status"] == "NOT_RUN" for row in report["records"])
    assert all(engine.closed for engine in states["engines"])
    assert speculative._PROPOSAL_GUARD == 1
    if fault == "request":
        assert next(row for row in report["records"] if row["status"] == "ERROR")["token_ids"] == [1, 2]
    if fault in ("head-drift", "prompt"):
        assert states["timed"] == 0  # invalid proof never becomes timing evidence


def test_exact_head_bit_observer_excludes_private_fp8_draft_caller(monkeypatch):
    import torch

    seen = []
    def incumbent(hidden):
        seen.append(hidden.shape[0])
        return hidden
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    engine = NS(_stream=object(), drain=lambda: None)
    captured = {"target": torch.zeros(3, 4, dtype=torch.bfloat16),
                "dflash": torch.zeros(2, 4, dtype=torch.bfloat16)}
    heads = {16: lambda hidden: hidden, 8: lambda hidden: hidden}
    actual = tool._bits(engine, heads, incumbent, captured, 8, draft=True)
    assert set(actual) == {"target"} and seen == [3]
    seen.clear()
    assert set(tool._bits(engine, heads, incumbent, captured, 16, draft=True)) == {"target", "dflash"}
    assert seen == [3, 2]
    with pytest.raises(ValueError, match="score bits"):
        tool._bits(engine, {8: lambda hidden: hidden + 1}, incumbent, captured, 8, draft=True)


def test_actual_instance_input_hooks_restore_ownership_on_failure(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    class Target:
        def project_logits(self, hidden):
            return hidden
    class Graph:
        def forward(self):
            return NS(hidden=torch.zeros(3, 4))
    target, draft, graph = Target(), Target(), Graph()
    engine = NS(target=target, head=draft, graphs=graph, drain=lambda: None)
    with pytest.raises(KeyboardInterrupt), tool._inputs(engine) as (captured, state):
        target.project_logits(torch.zeros(1, 4))
        draft.project_logits(torch.zeros(2, 4))
        graph.forward()
        assert captured["target"].shape == (3, 4) and state["graph_calls"] == 1
        raise KeyboardInterrupt()
    assert "project_logits" not in vars(target) and "project_logits" not in vars(draft)
    assert "forward" not in vars(graph)


def test_compare_rejects_q_drift_but_allows_cross_configuration_sampled_changes(monkeypatch, tmp_path):
    plan, path, _states, _speculative, _sampling = runtime(monkeypatch, tmp_path)
    tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    rows = json.loads((tmp_path / "out/report.json").read_text())["records"]
    changed = copy.deepcopy(rows)
    candidate = next(row for row in changed if row["mode"] == "sampled-draft" and row["head"] == 8 and "proof" in row)
    candidate["proof"]["q_sha256"] = "changed"
    with pytest.raises(ValueError, match="q bytes"):
        tool._compare(changed, plan["serial_ids"])


def test_request_primary_retains_drain_and_publication_failures_with_owned_engine_retirement(monkeypatch, tmp_path):
    _plan, path, states, _speculative, _sampling = runtime(monkeypatch, tmp_path)
    engine_type = tool.importlib.import_module(tool.proposal.SOURCES[2]).Gemma4Engine
    primary = RuntimeError("actual request failed after partial output")
    def generate(self, prompt, count, policy, receive, **kwargs):
        receive([1, 2])
        states["native_started"] = True
        raise primary
    def drain(self):
        if states.get("native_started"):
            raise OSError("deferred drain failed")
    original_write = tool.proposal.write_json
    def publish(destination, report):
        if any(row["status"] == "ERROR" for row in report["records"]):
            raise OSError("owned evidence publication failed")
        original_write(destination, report)
    monkeypatch.setattr(engine_type, "generate", generate)
    monkeypatch.setattr(engine_type, "drain", drain)
    monkeypatch.setattr(tool.proposal, "write_json", publish)
    with pytest.raises(RuntimeError) as caught:
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    assert caught.value is primary and states["closed"] == 1
    diagnostic = "".join(traceback.format_exception(caught.value))
    assert "deferred drain failed" in diagnostic and "owned evidence publication failed" in diagnostic


def test_observation_primary_retains_drain_failure_and_restores_actual_instance_methods(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    class Target:
        def project_logits(self, hidden):
            return hidden
    class Graph:
        def forward(self):
            return NS(hidden=torch.zeros(1, 4))
    def drain():
        raise OSError("owned observation drain failed")
    target, draft, graph = Target(), Target(), Graph()
    primary = ValueError("proof failed")
    engine = NS(target=target, head=draft, graphs=graph, drain=drain)
    with pytest.raises(ValueError) as caught, tool._inputs(engine):
        raise primary
    assert caught.value is primary and isinstance(primary.__cause__, OSError)
    assert "project_logits" not in vars(target) and "project_logits" not in vars(draft) and "forward" not in vars(graph)


def test_failed_bit_proof_is_recorded_as_error_and_cannot_enter_timing(monkeypatch, tmp_path):
    _plan, path, states, _speculative, _sampling = runtime(monkeypatch, tmp_path)
    def bad_bits(*args, **kwargs):
        raise ValueError("actual BF16 bit proof failed")
    monkeypatch.setattr(tool, "_bits", bad_bits)
    with pytest.raises(ValueError, match="bit proof"):
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["records"][0]["status"] == "ERROR" and states["timed"] == 0 and states["closed"] == 1
