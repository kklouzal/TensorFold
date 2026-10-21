"""Softcap qualification orchestration adversaries; fixtures are not model evidence."""
from __future__ import annotations

import copy
import json
import sys
import textwrap
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_gemma4_cuda_launch_qualify import fixture_plan as launch_plan

from tools import gemma4_cuda_softcap_qualify as tool


@pytest.fixture(autouse=True)
def baseline_fixture(monkeypatch, tmp_path):
    # A deliberately synthetic reference envelope tests the AST/byte boundary;
    # it is never represented as the named full-model historical source.
    raw = ("class Gemma4Target:\n" + textwrap.indent(tool.BASELINE_METHOD, "    ")).encode()
    (tmp_path / "baseline.py").write_bytes(raw)
    monkeypatch.setattr(tool, "BASELINE_FORWARD_SHA256", tool.proposal._sha(raw))


def fixture_plan(tmp_path):
    old, path = launch_plan(tmp_path)
    plan = {key: old[key] for key in ("schema", "artifact_id", "assets", "prompts_file", "prompt_sha256", "serve_args", "port")}
    plan.update(kind="softcap-holdout", tokens=256, source_sha256={key: "0" * 64 for key in tool.SOURCE_KEYS},
                blocks=[{"seed": seed, "order": list(("torch", "fused", "fused", "torch") if index % 2 == 0
                                                     else ("fused", "torch", "torch", "fused"))}
                        for index, seed in enumerate(tool.SEEDS)])
    plan["assets"]["baseline_source"] = {"path": str(tmp_path / "baseline.py"),
                                        "sha256": tool.BASELINE_FORWARD_SHA256}
    for key, value in tool.FIXED.items():
        if key not in plan["serve_args"]:
            plan["serve_args"].extend((key, value))
    path.write_text(json.dumps(plan))
    return plan, path


def test_plan_seals_complete96_positions_six_seeds_and_transition_cache_states(tmp_path):
    plan, path = fixture_plan(tmp_path)
    actual, items, raw, prompts, consumed, method = tool.load_plan(path)
    assert actual == plan and json.loads(raw) == plan
    assert len(items) == 2 and len(json.loads(prompts)["items"]) == 2 and len(consumed) == 6
    assert method.name == "project_logits"
    rows = tool.matrix(plan)
    assert len(rows) == 96 and sum(row["stage"] == "timed" for row in rows) == 48
    assert all(row["phase"] == "warm" and not row["rebind"] for row in rows if row["stage"] == "timed")
    assert all((row["phase"] == "cold") == row["rebind"] for row in rows if row["stage"] == "proof")
    assert all(row["status"] == "NOT_RUN" for row in rows)
    assert len({row["id"] for row in rows}) == 96
    plan["blocks"] = [{**block, "order": list(reversed(block["order"]))} for block in plan["blocks"]]
    path.write_text(json.dumps(plan))
    assert tool.load_plan(path)[0] == plan


@pytest.mark.parametrize("field,value", [("schema", True), ("tokens", 64), ("port", False),
                                        ("artifact_id", "latest"), ("blocks", []), ("source_sha256", {})])
def test_unsealed_plan_refuses_before_model(tmp_path, field, value):
    plan, path = fixture_plan(tmp_path)
    plan[field] = value
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        tool.load_plan(path)


@pytest.mark.parametrize("fault", ["seed", "order", "duplicate", "joined", "precision", "input", "template"])
def test_region_and_sequence_drift_refuses(tmp_path, fault):
    plan, path = fixture_plan(tmp_path)
    if fault == "seed":
        plan["blocks"][0]["seed"] = 7
    elif fault == "order":
        plan["blocks"][0]["order"] = ["torch", "fused", "torch", "fused"]
    elif fault == "duplicate":
        plan["serve_args"].extend(("--draft-rows", "5"))
    elif fault == "joined":
        plan["serve_args"].append("--context=4096")
    elif fault == "precision":
        plan["serve_args"][plan["serve_args"].index("--drafter-bits") + 1] = "8"
    elif fault == "input":
        Path(plan["assets"]["tokenizer"]["path"]).write_text("changed")
    else:
        plan["assets"]["template"]["sha256"] = "f" * 64
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        tool.load_plan(path)


def test_historical_method_ast_refuses_added_work_even_in_synthetic_authenticated_envelope(monkeypatch, tmp_path):
    plan, path = fixture_plan(tmp_path)
    target = Path(plan["assets"]["baseline_source"]["path"])
    data = target.read_bytes().replace(b"logits = self.weights.embed(hidden)",
                                       b"logits = self.weights.embed(hidden)\n        print('unwanted work')")
    target.write_bytes(data)
    digest = tool.proposal._sha(data)
    monkeypatch.setattr(tool, "BASELINE_FORWARD_SHA256", digest)
    plan["assets"]["baseline_source"]["sha256"] = digest
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="method AST"):
        tool.load_plan(path)


def runtime(monkeypatch, tmp_path, fault=None):
    state = {"calls": 0, "binds": [], "proof": False, "closed": False, "constructed": 0}
    native, retirement = RuntimeError("native partial failure"), OSError("final native drain failure")

    class Target:
        def _softcap_logits(self, value):
            return value

        def project_logits(self, value):
            return self._softcap_logits(value)

    def fused(value):
        return value

    class Head:
        def __init__(self, incumbent):
            self.incumbent, self.reuse_rows = incumbent, 16

    class Engine:
        context_window, max_rows, vision = 262144, 5, object()

        def __init__(self):
            self.capacity_plan = {"fixture": True}
            self.w = NS(numerics="ct_nvfp4", embed=Head(object()))
            self.original_head = self.w.embed
            self.head, self.target = NS(bits=16), Target()
            self.target._softcap_logits = fused
            self.head.project_logits = self.target.project_logits
            self.graphs, self._prefix, self._prefix_lock = object(), None, threading.Lock()
            self.arm, self.cache = "fused", None

        def drain(self):
            if (fault == "retirement" and state["calls"] == 96) or (fault == "partial-retirement" and state["calls"]):
                raise retirement

        def generate(self, prompt, count, sampling, receive, *, draft, stop_eos):
            assert draft and not stop_eos and count == 256
            state["calls"] += 1
            state["proof_by_call"] = state.get("proof_by_call", []) + [state["proof"]]
            cached = len(prompt) if self.cache == tuple(prompt) else 0
            self.cache = tuple(prompt)
            tokens = [sampling.seed + index for index in range(count)]
            if fault == "ids" and self.arm == "fused":
                tokens[-1] += 1
            receive(tokens[:16])
            if fault in ("partial", "partial-retirement"):
                raise native
            receive(tokens[16:-1] if fault == "short" else tokens[16:])
            if state["proof"]:
                state["audit"].update(q_packets=1, q_rows=4,
                                      q_sha256=("e" if fault == "q" and self.arm == "fused" else "d") * 64)
            return {"cached": cached, "rounds": 100, "drafted": 400, "accepted": 155,
                    "min_rows": 5, "prefill_s": .1, "decode_s": .2}

    engine = Engine()

    def prepare(body, chat):
        assert chat and not body["chat_template_kwargs"]["enable_thinking"]
        workload = body["messages"][0]["content"]
        prompt = list(range(tool.proposal.WORKLOADS[workload]))
        if fault == "prompt":
            prompt[-1] += 1
        return NS(prompt=prompt, max_tokens=256, ignore_eos=True, thinking=False, vision=None, grammar=None,
                  calls=None, stop=(), think_budget=0, penalties=None,
                  sampling=NS(seed=body["seed"], temperature=1., top_k=64, top_p=.95, min_p=0.))

    app = NS(engine=engine, prepare=prepare)

    def with_app(argv, port, work):
        state["constructed"] += 1
        try:
            work(app)
        finally:
            state["closed"] = True
        if fault == "close":
            raise retirement

    def bind_head(engine, head):
        assert head is not engine.w.embed
        state["binds"].append(head)
        engine.w.embed, engine.graphs, engine._prefix, engine.cache = head, object(), None, None
        assert engine.head.project_logits is engine.target.project_logits
        assert engine.target._softcap_logits is fused
        engine.arm = "fused" if engine.target.project_logits.__func__ is Target.project_logits else "torch"
        return {"changed": fault != "stale", "seconds": 0., "prepared_graphs": 10}

    @contextmanager
    def diagnostics(*args):
        state["proof"] = True
        state["audit"] = {"q_packets": 0, "q_rows": 0}
        try:
            yield state["audit"]
        finally:
            state["proof"] = False

    torch = NS(__version__="CPU adapter", version=NS(cuda="fixture"), cuda=NS(
        synchronize=lambda: None, get_device_capability=lambda: (12, 1), get_device_name=lambda: "CPU adapter"))
    modules = {name: NS(__file__=str(tmp_path / "model/config.json")) for name in tool.SOURCES}
    modules[tool.proposal.SOURCES[0]]._PROPOSAL_GUARD = 32
    modules[tool.proposal.SOURCES[0]].Chains = object()
    modules[tool.proposal.SOURCES[2]].Gemma4Engine = Engine
    modules["tensorfold.families.gemma4.cuda.head"].__dict__.update(
        _RowsHead=Head, prepare_head=lambda incumbent, reuse_rows: Head(incumbent))
    modules["tensorfold.families.gemma4.cuda.forward"].Gemma4Target = Target
    modules[tool.SOURCES[-1]].softcap30 = fused
    original_import = tool.importlib.import_module
    monkeypatch.setattr(tool.importlib, "import_module", lambda name, *args, **kwargs:
                        modules[name] if name in modules else original_import(name, *args, **kwargs))
    monkeypatch.setattr(tool, "sources", lambda modules: {key: "f" * 64 if fault == "source" else "0" * 64
                                                        for key in tool.SOURCE_KEYS})
    monkeypatch.setattr(tool.phase, "with_app", with_app)
    monkeypatch.setattr(tool.phase, "host_snapshot", lambda: {"fixture": True})
    monkeypatch.setattr(tool.phase.HotGate, "wait", lambda: 0.)
    monkeypatch.setattr(tool.head_bench, "_bind_head", bind_head)
    monkeypatch.setattr(tool.proposal, "diagnostics", diagnostics)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", NS(__version__="fixture"))
    return state, engine, native, retirement


def test_real_phase_attempt_covers_all_positions_preparation_and_q_observer_lifetimes(monkeypatch, tmp_path):
    _, path = fixture_plan(tmp_path)
    state, engine, _, _ = runtime(monkeypatch, tmp_path)
    report = tool.run(path, tmp_path / "out")
    assert report["status"] == "COMPLETE" and state["closed"] and state["constructed"] == 1
    assert state["calls"] == 96 and state["proof_by_call"] == [True, False] * 48
    assert len(state["binds"]) == sum(row["rebind"] for row in report["records"])
    assert len({id(head) for head in state["binds"]}) == len(state["binds"])
    assert engine.graphs is None and engine.w.embed is engine.original_head
    assert engine.target._softcap_logits is tool.importlib.import_module(tool.SOURCES[-1]).softcap30
    assert report["restoration"]["status"] == "COMPLETE"
    assert all(row["status"] == "COMPLETE" and row["comparison"] == "MATCH" and len(row["token_ids"]) == 256
               for row in report["records"])
    for row in report["records"]:
        assert row["prepare_s"] <= row["first_from_prepare_s"] <= row["prepared_request_wall_s"]
        assert row["first_s"] <= row["last_s"] <= row["generated_s"] <= row["settled_wall_s"]
    assert (tmp_path / "out/plan.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("fault,message", [("source", "source differs"), ("prompt", "prepared prompt"),
                                          ("stale", "stale target graphs"), ("short", "fewer"),
                                          ("ids", "complete IDs"), ("q", "actual q bytes")])
def test_corruption_stale_graphs_and_incomplete_results_fail_with_all_outcomes(monkeypatch, tmp_path, fault, message):
    _, path = fixture_plan(tmp_path)
    state, engine, _, _ = runtime(monkeypatch, tmp_path, fault)
    with pytest.raises(ValueError, match=message):
        tool.run(path, tmp_path / "out")
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "FAILED" and len(report["records"]) == 96
    if fault == "source":
        assert state["constructed"] == 0 and state["calls"] == 0
    else:
        assert state["closed"] and engine.graphs is None and engine.w.embed is engine.original_head
    if fault in ("prompt", "stale"):
        assert not state["calls"] and report["records"][1]["status"] == "NOT_RUN"
    if fault == "short":
        assert len(report["records"][0]["token_ids"]) == 255


@pytest.mark.parametrize("fault", ["partial", "retirement", "close"])
def test_primary_partial_ids_and_final_retirement_remain_required(monkeypatch, tmp_path, fault):
    _, path = fixture_plan(tmp_path)
    state, engine, native, retirement = runtime(monkeypatch, tmp_path, fault)
    with pytest.raises((RuntimeError, OSError)) as caught:
        tool.run(path, tmp_path / "out")
    assert caught.value is (native if fault == "partial" else retirement) and state["closed"]
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "FAILED"
    if fault == "partial":
        assert len(report["records"][0]["token_ids"]) == 16 and report["records"][1]["status"] == "NOT_RUN"
        assert report["records"][0]["proof"] == {"q_packets": 0, "q_rows": 0}
    elif fault == "retirement":
        assert report["restoration"]["status"] == "FAILED" and engine.graphs is not None
    else:
        assert all(row["status"] == "COMPLETE" for row in report["records"])


@pytest.mark.parametrize("fault", ["publication", "partial-retirement"])
def test_inference_primary_survives_publication_and_multiple_retirement_failures(monkeypatch, tmp_path, fault):
    _, path = fixture_plan(tmp_path)
    state, engine, native, _ = runtime(monkeypatch, tmp_path, "partial" if fault == "publication" else fault)
    original = tool.phase.write_json

    def publish(path, document):
        if fault == "publication" and state["calls"] and document["status"] == "RUNNING":
            raise OSError("evidence publication failed")
        return original(path, document)

    monkeypatch.setattr(tool.phase, "write_json", publish)
    with pytest.raises(RuntimeError) as caught:
        tool.run(path, tmp_path / "out")
    assert caught.value is native and state["closed"]
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "FAILED" and len(report["records"][0]["token_ids"]) == 16
    if fault == "partial-retirement":
        assert engine.graphs is not None and report["restoration"]["status"] == "FAILED"
        assert report["records"][0]["retirement_errors"]


def test_compare_cannot_rebase_changed_timed_output_or_uncovered_proof():
    rows = [{"status": "COMPLETE", "workload": "code-sampled", "seed": 13, "arm": arm, "stage": stage,
             "token_ids": list(range(256)), "stats": {key: 5 for key in tool.proposal.COUNTS},
             "proof": {"q_packets": 3, "q_rows": 12, "q_sha256": "d" * 64}}
            for arm, stage in (("torch", "proof"), ("torch", "timed"), ("fused", "proof"), ("fused", "timed"))] * 24
    for field in ("token_ids", "proof"):
        changed = copy.deepcopy(rows)
        if field == "token_ids":
            changed[-1][field][-1] += 1
        else:
            changed[-2][field]["q_sha256"] = "e" * 64
        with pytest.raises(ValueError):
            tool.compare(changed)
