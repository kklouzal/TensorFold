"""Qualification-owned policy/data/lifetime controls; no Torch, GPU or model."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location("proposal_qualification", ROOT / "tools/gemma4_cuda_proposal_qualify.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
finally:
    sys.path.pop(0)


def plan_fixture(tmp_path, *, kind="screen"):
    prompts = {"items": [{"id": name, "thinking": False, "messages": [{"role": "user", "content": name}]}
                         for name in tool.WORKLOADS]}
    raw = json.dumps(prompts).encode()
    (tmp_path / "prompts.json").write_bytes(raw)
    policies = list(tool.POLICIES) if kind == "screen" else [tool.BASELINE, "q64-g32-partial"]
    blocks = [{"seed": 7, "order": policies + list(reversed(policies))}] if kind == "screen" else [
        {"seed": seed, "order": ([tool.BASELINE, policies[1], policies[1], tool.BASELINE] if at % 2 == 0
                                 else [policies[1], tool.BASELINE, tool.BASELINE, policies[1]])}
        for at, seed in enumerate([13, 31, 47, 83, 101, 127])]
    return {"schema": 1, "kind": kind, "artifact_id": "sha256:" + "0" * 64,  # CPU fixture, no real image claimed
            "source_sha256": {name: "0" * 64 for name in ("tool", *tool.SOURCES)},
            "prompts_file": {"path": "prompts.json", "sha256": hashlib.sha256(raw).hexdigest()},
            "policies": policies, "reference": tool.BASELINE, "blocks": blocks,
            "tokens": 64 if kind == "screen" else 256, "serve_args": ["serve", "/model"], "port": 8899}


def save_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.mark.parametrize("kind,expected", [("screen", 146), ("holdout", 60)])
def test_finite_plan_builds_complete_explicit_matrix_and_snapshots_consumed_bytes(tmp_path, kind, expected):
    plan = plan_fixture(tmp_path, kind=kind)
    loaded, items, raw, prompt_raw = tool.load_plan(save_plan(tmp_path, plan))
    assert loaded == plan and json.loads(raw) == plan and len(items) == 2
    rows = tool._matrix(loaded)
    assert len(rows) == expected and all(row["status"] == "NOT_RUN" for row in rows)
    assert len([row for row in rows if row["phase"] == "cold"]) == 2
    assert len({row["case"] for row in rows}) == len(rows)
    (tmp_path / "prompts.json").write_text("changed")
    assert json.loads(prompt_raw)["items"] == items
    with pytest.raises(ValueError, match="hash changed"):
        tool.load_plan(tmp_path / "plan.json")


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("schema", 1.0), ("tokens", True), ("tokens", 128), ("port", False),
    ("policies", [tool.BASELINE]), ("policies", [tool.BASELINE, tool.BASELINE]),
    ("reference", "q64-g1-lex"), ("artifact_id", "latest"),
    ("serve_args", ["serve", "/model", "--api-key=private"]),
    ("blocks", [{"seed": True, "order": list(tool.POLICIES) * 2}]),
])
def test_invalid_or_unsealed_plan_refuses_before_model_construction(tmp_path, field, value):
    plan = plan_fixture(tmp_path)
    plan[field] = value
    with pytest.raises(ValueError):
        tool.load_plan(save_plan(tmp_path, plan))


def test_policy_controls_only_exact_selected_library_boundaries_and_restores_on_interrupt():
    class Chains:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
    partial = object()
    speculative = SimpleNamespace(_PROPOSAL_GUARD=32, Chains=Chains)
    sampling = SimpleNamespace(_canonical_repair_order=partial)
    actual = object()
    for name, (bounded, guard, use_partial) in tool.POLICIES.items():
        with pytest.raises(KeyboardInterrupt), tool.policy(speculative, sampling, name, Chains, partial):
            packet = speculative.Chains(1, [None], 4, sampled=True, sampling=actual)
            assert type(packet) is Chains
            assert packet.kwargs.get("sampling") is (actual if bounded else None)
            assert speculative._PROPOSAL_GUARD == guard
            assert sampling._canonical_repair_order is (partial if use_partial else tool.legacy_order)
            raise KeyboardInterrupt()
        assert speculative._PROPOSAL_GUARD == 32 and speculative.Chains is Chains
        assert sampling._canonical_repair_order is partial


def runtime_fixture(monkeypatch, tmp_path, *, fault=None, kind="screen"):
    plan = plan_fixture(tmp_path, kind=kind)
    states = {"engine": None, "constructed": 0}
    speculative = SimpleNamespace(_PROPOSAL_GUARD=1)
    sampling = SimpleNamespace()

    def partial(values, ids, kept):
        return np.argsort(-values, kind="stable")[:kept]

    def canonical(values, ids, count, width, full_row):
        data, tokens = full_row(0)
        order = sampling._canonical_repair_order(data, tokens, count)
        return data[order], tokens[order]

    class Chains:
        def __init__(self, *args, **kwargs):
            self.bounded = "sampling" in kwargs
        def sample(self, *args):
            values, ids = np.array([2., 1.], dtype=np.float32), np.array([4, 5], dtype=np.int64)
            speculative.canonical_candidates(values[None], ids[None], 1, 2, lambda row: (values, ids))
            probs = np.array([.75, .25])
            if fault == "q-bytes" and speculative._PROPOSAL_GUARD == 8:
                probs = np.array([.5, .5])
            return SimpleNamespace(first_position=49, tokens=(4,), q_rows=(SimpleNamespace(ids=ids, probs=probs),))

    speculative.Chains, speculative.canonical_candidates = Chains, canonical
    sampling._canonical_repair_order, sampling.canonical_candidates = partial, canonical
    original_sample = Chains.sample
    report_path = tmp_path / "out/report.json"

    class Engine:
        def __init__(self):
            assert speculative._PROPOSAL_GUARD == 32
            intent = json.loads(report_path.read_text())
            assert len(intent["records"]) == (146 if kind == "screen" else 60)
            assert all(row["status"] == "NOT_RUN" for row in intent["records"])
            states["constructed"] += 1
            self.w, self.max_rows, self.graphs, self.head = SimpleNamespace(numerics="ct_nvfp4"), 5, object(), object()
            self.context_window = 262144
            self._prefix_lock, self._prefix = threading.Lock(), None
        def drain(self):
            pass
        def generate(self, prompt, count, policy, receive, *, draft, stop_eos, penalties):
            assert not stop_eos and penalties is None
            current = next(row for row in json.loads(report_path.read_text())["records"] if row["status"] == "RUNNING")
            if current["phase"] in {"warm", "timed"}:
                assert Chains.sample is original_sample
                assert speculative.canonical_candidates is canonical and sampling.canonical_candidates is canonical
            cached = len(prompt) if self._prefix == tuple(prompt) else 0
            self._prefix = tuple(prompt)
            if fault == "timed-cache" and current["phase"] == "timed" and current["seed"] == 31:
                cached = 0
            if draft:
                packet = speculative.Chains(None, [None], 4, sampled=policy is not None, sampling=policy)
                if policy is not None:
                    packet.sample(0, 4, policy, len(prompt) + 1)
            if policy is not None:
                sampling.canonical_candidates(np.array([[2., 1.]]), np.array([[4, 5]]), 1, 2,
                                             lambda row: (np.array([2., 1.]), np.array([4, 5])))
            ids = [17 if policy is None else 100 + policy.seed] * count
            if fault == "phase-ids" and current["phase"] in {"warm", "timed"}:
                ids[-1] += 1
            if fault == "partial-error" and speculative._PROPOSAL_GUARD == 8:
                receive(ids[:2])
                raise RuntimeError("native fixture failed after partial output")
            if fault == "short" and speculative._PROPOSAL_GUARD == 8:
                ids.pop()
            receive(ids)
            return {"cached": cached, "rounds": 1, "drafted": 4 if draft else 0,
                    "accepted": (2 if draft else 0) + (1 if fault == "phase-counts"
                                                       and current["phase"] in {"warm", "timed"} else 0),
                    "min_rows": 5 if draft else 1,
                    "decode_s": float("nan") if fault == "nonfinite-stats" and speculative._PROPOSAL_GUARD == 8 else 1.}

    class App:
        def __init__(self):
            self.engine = Engine()
        def prepare(self, body, chat):
            assert chat and body["ignore_eos"] and body["chat_template_kwargs"] == {"enable_thinking": False}
            policy = None if body["temperature"] == 0 else SimpleNamespace(
                seed=body["seed"], temperature=1., top_k=64, top_p=.95, min_p=0.)
            count = tool.WORKLOADS[body["messages"][0]["content"]] + (1 if fault == "prepared-count" else 0)
            return SimpleNamespace(prompt=[1] * count,
                                   max_tokens=body["max_tokens"], sampling=policy, thinking=False, ignore_eos=True,
                                   grammar=None, vision=None, penalties=None)

    def with_app(args, port, work):
        app = App()
        states["engine"] = app.engine
        try:
            work(app)
        finally:
            assert speculative._PROPOSAL_GUARD == 32
            states["closed"] = True

    source = str(Path(__file__).resolve())
    modules = {name: SimpleNamespace(__file__=source) for name in tool.SOURCES}
    modules[tool.SOURCES[0]], modules[tool.SOURCES[1]] = speculative, sampling
    modules[tool.SOURCES[2]].Gemma4Engine = Engine
    speculative.__file__ = sampling.__file__ = source
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    plan["source_sha256"] = {name: digest for name in tool.SOURCES}
    plan["source_sha256"]["tool"] = hashlib.sha256(Path(tool.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(tool, "importlib", SimpleNamespace(import_module=lambda name: modules[name]))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="cpu-fixture", version=SimpleNamespace(cuda="not-used"),
        cuda=SimpleNamespace(get_device_name=lambda: "no-GPU-fixture", get_device_capability=lambda: (12, 1))))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="not-used"))
    monkeypatch.setattr(tool, "with_app", with_app)
    monkeypatch.setattr(tool, "HotGate", SimpleNamespace(configure=lambda path: None, wait=lambda: 0))
    # Keep the actual maintained helper signature/behavior; an unconstrained
    # callback fake previously hid a duplicate phase argument at this boundary.
    monkeypatch.setattr(tool, "Progress", type("IsolatedProgress", (tool.Progress,), {"path": None}))
    monkeypatch.setattr(tool, "host_snapshot", lambda: {"fixture": True})
    return save_plan(tmp_path, plan), report_path, states, speculative, sampling, original_sample, Chains, canonical


def test_full_finite_matrix_uses_one_admitted_engine_separate_proof_and_all_outcomes(monkeypatch, tmp_path):
    path, report_path, state, speculative, sampling, sample, chains, canonical = runtime_fixture(monkeypatch, tmp_path)
    assert tool.main(["--plan", str(path), "--out", str(tmp_path / "out")]) == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "PASS" and len(report["records"]) == 146
    assert all(row["status"] == "PASS" and len(row["token_ids"]) == 64 for row in report["records"])
    assert state["constructed"] == 1 and state["closed"]
    assert speculative._PROPOSAL_GUARD == 1 and speculative.Chains is chains and chains.sample is sample
    assert speculative.canonical_candidates is canonical and sampling.canonical_candidates is canonical
    assert len(report["repair_assets"]) == 2 and all(a["status"] == "COMPLETE" for a in report["repair_assets"])
    proof = [row for row in report["records"] if row["phase"] == "proof" and row["mode"] == "sampled-draft"]
    assert len({row["proof"]["q_sha256"] for row in proof}) == 1
    assert all("proof" not in row for row in report["records"] if row["phase"] in {"warm", "timed"})
    assert all(row["proof_anchor"]["status"] == "MATCH" for row in report["records"]
               if row["phase"] in {"warm", "timed"})


@pytest.mark.parametrize("fault,message", [("q-bytes", "logical q bytes"), ("short", "complete declared token count"),
                                          ("partial-error", "native fixture failed")])
def test_bad_or_partial_outcome_is_retained_without_filtering_replacement_or_timing(monkeypatch, tmp_path, fault,
                                                                                message):
    path, report_path, state, speculative, _, _, _, _ = runtime_fixture(monkeypatch, tmp_path, fault=fault)
    with pytest.raises(RuntimeError, match=message):
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    report = json.loads(report_path.read_text())
    assert report["status"] == "ERROR" and len(report["records"]) == 146
    assert any(row["status"] == "NOT_RUN" for row in report["records"])
    assert all(row["status"] == "NOT_RUN" for row in report["records"] if row["phase"] in {"warm", "timed"})
    if fault != "q-bytes":
        row = next(row for row in report["records"] if row["status"] == "ERROR")
        assert len(row["token_ids"]) == (2 if fault == "partial-error" else 63)
    assert state["closed"] and speculative._PROPOSAL_GUARD == 1


def test_source_drift_is_explicit_and_precedes_model_load(monkeypatch, tmp_path):
    path, report_path, state, *_ = runtime_fixture(monkeypatch, tmp_path)
    plan = json.loads(path.read_text())
    plan["source_sha256"]["tool"] = "0" * 64
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="installed source differs"):
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    assert state["constructed"] == 0
    assert all(row["status"] == "NOT_RUN" for row in json.loads(report_path.read_text())["records"])


def test_actual_preparation_count_mismatch_refuses_before_any_generation(monkeypatch, tmp_path):
    path, report_path, state, *_ = runtime_fixture(monkeypatch, tmp_path, fault="prepared-count")
    with pytest.raises(ValueError, match="prepared request differs"):
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    assert state["closed"]
    assert all(row["status"] == "NOT_RUN" for row in json.loads(report_path.read_text())["records"])


def test_nonfinite_native_stats_are_error_outcomes_with_publishable_complete_ids(monkeypatch, tmp_path):
    path, report_path, state, *_ = runtime_fixture(monkeypatch, tmp_path, fault="nonfinite-stats")
    with pytest.raises(ValueError, match="Out of range"):
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    report = json.loads(report_path.read_text())
    row = next(row for row in report["records"] if row["status"] == "ERROR")
    assert len(row["token_ids"]) == 64 and "stats" not in row and state["closed"]


def test_request_partial_primary_cleanup_and_failed_publication_are_retained(monkeypatch, tmp_path):
    primary, cleanup, publication = RuntimeError("partial request"), OSError("settlement"), OSError("publication")
    class Engine:
        drains = 0
        def drain(self):
            self.drains += 1
            if self.drains > 1:
                raise cleanup
        def generate(self, prompt, count, sampling, receive, **kwargs):
            receive([1, 2])
            raise primary
    writes = []
    def publish(path, report):
        writes.append(copy.deepcopy(report))
        if len(writes) > 1:
            raise publication
    monkeypatch.setattr(tool, "write_json", publish)
    monkeypatch.setattr(tool, "host_snapshot", dict)
    monkeypatch.setattr(tool, "HotGate", SimpleNamespace(wait=lambda: 0))
    row = {"case": 0, "phase": "proof", "mode": "greedy-draft"}
    request = SimpleNamespace(prompt=[5], max_tokens=64, sampling=None, penalties=None)
    with pytest.raises(RuntimeError) as caught:
        tool._attempt(Engine(), request, row, {"records": [row]}, tmp_path / "report.json")
    assert caught.value is primary
    assert row["status"] == "ERROR" and row["token_ids"] == [1, 2]
    assert len(row["errors"]) == 2 and writes[1]["records"][0]["status"] == "ERROR"
    causes = primary.__cause__
    assert isinstance(causes, BaseExceptionGroup)
    assert cleanup in causes.exceptions and publication in causes.exceptions


def test_warm_ids_and_mode_correct_stats_are_not_filtered_from_comparison():
    rows = [{"case": at, "phase": phase, "status": "PASS", "workload": "code-sampled", "policy": policy,
             "mode": "sampled-draft",
             "seed": 7, "block": 0, "token_ids": [1, 2], "stats": dict.fromkeys(tool.COUNTS, 1)}
            for at, (phase, policy) in enumerate((phase, policy) for phase in ["warm", "timed"]
                                                 for policy in [tool.BASELINE, "q64-g1-lex"])]
    tool._compare(rows, tool.BASELINE, proof=False)
    changed = copy.deepcopy(rows)
    changed[1]["token_ids"][-1] = 3
    with pytest.raises(RuntimeError, match="complete same-policy"):
        tool._compare(changed, tool.BASELINE, proof=False)
    changed = copy.deepcopy(rows)
    changed[-1]["stats"]["accepted"] = 2
    with pytest.raises(RuntimeError, match="mode-correct"):
        tool._compare(changed, tool.BASELINE, proof=False)


@pytest.mark.parametrize("fault,message", [("phase-ids", "proof-to-timing complete token IDs"),
                                          ("phase-counts", "proof-to-timing proposal/acceptance")])
def test_uniform_timing_drift_cannot_rebase_validated_proof(monkeypatch, tmp_path, fault, message):
    path, report_path, state, *_ = runtime_fixture(monkeypatch, tmp_path, fault=fault)
    with pytest.raises(RuntimeError, match=message):
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    report = json.loads(report_path.read_text())
    assert report["status"] == "ERROR" and state["closed"]
    # Every request and the disjoint cross-arm checks agree: only the missing
    # phase bridge distinguishes this failure from the old false PASS.
    assert all(row["status"] == "PASS" for row in report["records"])
    timing = [row for row in report["records"] if row["phase"] in {"warm", "timed"}]
    assert all(row["comparison"] == "PASS" for row in timing)
    assert any(row.get("proof_anchor", {}).get("status") == "ERROR" for row in timing)


def test_distinct_holdout_seed_does_not_claim_seed7_proof_coverage():
    proof = [{"case": 1, "phase": "proof", "workload": "code-sampled", "policy": tool.BASELINE,
              "mode": "sampled-draft", "seed": 7, "token_ids": [1], "stats": dict.fromkeys(tool.COUNTS, 1)}]
    timing = [dict(proof[0], case=2, phase="timed", seed=13, token_ids=[2])]
    tool._anchor_phases(proof, timing)
    assert timing[0]["proof_anchor"]["status"] == "NOT_COVERED"


def test_holdout_has_full_proofs_one_warm_and_all_48_timed_positions(tmp_path):
    plan = plan_fixture(tmp_path, kind="holdout")
    rows = tool._matrix(plan)
    assert len(rows) == 60 and plan["tokens"] == 256
    proof = [row for row in rows if row["phase"] in {"cold", "proof"}]
    assert len(proof) == 10 and rows[:10] == proof
    for workload in tool.WORKLOADS:
        group = [row for row in rows if row["workload"] == workload]
        assert len([row for row in group if row["phase"] == "cold"]) == 1
        assert {(row["policy"], row["mode"]) for row in group if row["phase"] == "proof"} == {
            (policy, mode) for policy in plan["policies"] for mode in ("greedy-draft", "sampled-draft")}
        warm = [row for row in group if row["phase"] == "warm"]
        assert len(warm) == 1 and (warm[0]["policy"], warm[0]["seed"], warm[0]["position"]) == (
            tool.BASELINE, 13, -1)
        timed = [row for row in group if row["phase"] == "timed"]
        assert warm[0]["case"] < timed[0]["case"] and len(timed) == 24
        assert [(row["seed"], row["position"], row["policy"]) for row in timed] == [
            (block["seed"], position, policy) for block in plan["blocks"]
            for position, policy in enumerate(block["order"])]
        assert [row["seed"] for row in timed[::4]] == [13, 31, 47, 83, 101, 127]


@pytest.mark.parametrize("start_candidate", [False, True])
def test_holdout_accepts_both_predeclared_alternating_starts(tmp_path, start_candidate):
    plan = plan_fixture(tmp_path, kind="holdout")
    if start_candidate:
        for block in plan["blocks"]:
            block["order"] = [plan["policies"][1] if p == tool.BASELINE else tool.BASELINE for p in block["order"]]
    loaded, *_ = tool.load_plan(save_plan(tmp_path, plan))
    assert loaded == plan


@pytest.mark.parametrize("orders", ["all-ABBA", "balanced-not-alternating"])
def test_holdout_rejects_unbalanced_or_non_alternating_blocks(tmp_path, orders):
    plan = plan_fixture(tmp_path, kind="holdout")
    a, b = plan["policies"]
    abba, baab = [a, b, b, a], [b, a, a, b]
    for index, block in enumerate(plan["blocks"]):
        block["order"] = abba if orders == "all-ABBA" or index < 3 else baab
    with pytest.raises(ValueError, match="alternating"):
        tool.load_plan(save_plan(tmp_path, plan))


def test_fast_holdout_keeps_every_complete_cached_timing_and_honest_proof_scope(monkeypatch, tmp_path):
    path, report_path, state, speculative, *_ = runtime_fixture(monkeypatch, tmp_path, kind="holdout")
    assert tool.main(["--plan", str(path), "--out", str(tmp_path / "out")]) == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "PASS" and len(report["records"]) == 60
    assert all(row["status"] == "PASS" and len(row["token_ids"]) == 256 for row in report["records"])
    assert len([row for row in report["records"] if row["phase"] == "warm"]) == 2
    timed = [row for row in report["records"] if row["phase"] == "timed"]
    assert len(timed) == 48 and all(row["stats"]["cached"] == tool.WORKLOADS[row["workload"]] for row in timed)
    assert all(row["proof_anchor"]["status"] == "NOT_COVERED" for row in timed)
    assert "without per-position same-completion" in report["cache_condition"]
    assert state["constructed"] == 1 and state["closed"] and speculative._PROPOSAL_GUARD == 1


def test_fast_holdout_still_refuses_a_timed_prefix_cache_miss(monkeypatch, tmp_path):
    path, report_path, state, *_ = runtime_fixture(monkeypatch, tmp_path, kind="holdout", fault="timed-cache")
    with pytest.raises(RuntimeError, match="declared prefix-cache"):
        tool.main(["--plan", str(path), "--out", str(tmp_path / "out")])
    report = json.loads(report_path.read_text())
    failed = next(row for row in report["records"] if row["status"] == "ERROR")
    assert failed["phase"] == "timed" and failed["seed"] == 31 and len(failed["token_ids"]) == 256
    assert any(row["status"] == "NOT_RUN" for row in report["records"]) and state["closed"]
