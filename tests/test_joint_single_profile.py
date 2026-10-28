"""Tool-only combined-profile boundaries; no GPU or changed application code."""
import json
from types import SimpleNamespace as NS
import pytest
from tools import v100_backend_probe as probe, finite_concurrency as finite
from tests.test_v100_backend_probe import cli


def diagnostic_args(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "source-router-gdn-shared-fusion-trial")
    monkeypatch.setenv("TENSORFOLD_SM70_GDN_STATE_BATCH", "1")
    monkeypatch.setenv("TENSORFOLD_SM70_MOE_REGION_GRAPHS", "1")
    monkeypatch.setenv("TENSORFOLD_SM70_CANDIDATE_REGION_GRAPHS", "1")
    monkeypatch.delenv("TENSORFOLD_STARTUP_CONTENT_OVERLAP", raising=False)
    monkeypatch.delenv("TENSORFOLD_STARTUP_CONTENT_WORKERS", raising=False)
    manifest = tmp_path / "execution.json"
    manifest.write_text(json.dumps({"schema":"tensorfold.execution-profile-manifest","version":1,"artifacts":{}}))
    return cli(tmp_path) + ["--native-dense-trial", "--owner-batch-prelude-trial", "--router-blas-trial",
        "--gdn-state-batch-trial", "--packed-shared-fusion-trial", "--session-ceiling", "24", "--concurrency-grid", "--fast-grid",
        "--fast-grid-protocol", "fast-grid-v2", "--tokens", "2048", "--profile-concurrency", "1",
        "--overlap-content-capture", "--startup-content-workers", "3", "--execution-profile-manifest", str(manifest)]


def test_combined_diagnostic_preserves_explicit_flags_but_excludes_quality_and_all6_credit(tmp_path, monkeypatch):
    raw = diagnostic_args(tmp_path, monkeypatch)
    args = probe.arguments(raw)
    assert args.profile_concurrency == 1 and args.gdn_state_batch_trial and args.router_blas_trial
    assert args.startup_content_workers == 3 and args.overlap_content_capture
    assert not args.quality_spotchecks and args.tokens == 2048
    with pytest.raises(SystemExit):
        probe.arguments(raw + ["--quality-spotchecks"])
    without_profile = raw[:]
    index = without_profile.index("--profile-concurrency")
    del without_profile[index:index+2]
    with pytest.raises(SystemExit):
        probe.arguments(without_profile)


def test_actual_caller_prepays_reader_observers_and_both_counter_families_before_submit(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retention_resources
    args = probe.arguments(diagnostic_args(tmp_path, monkeypatch))
    monkeypatch.setattr(probe, "prepared_prompt", lambda *_args: ([1], {"rendered":"Notebook"}))
    engine = NS(sm70_profile="source-router-gdn-shared-fusion-trial", retention_catalog=object(), scheduler=object())
    stop = RuntimeError("actual admission boundary")
    def finalize(_catalog, *, caller, **_kwargs):
        expected = (finite.control_payload_bytes(2048) + finite.screen_control_payload_bytes(2048)
            + probe.BATCH_EVIDENCE_CONTROL_BYTES + probe.GDN_EVIDENCE_CONTROL_BYTES
            + 8 * ((128 << 10) + (6 << 30)))
        assert caller.control_payload_bytes == expected
        assert caller.max_concurrent_requests == 24
        raise stop
    monkeypatch.setattr(retention_resources, "finalize_backend_resources", finalize)
    with pytest.raises(RuntimeError) as caught:
        probe.run_concurrency_leader(engine, object(), args, {})
    assert caught.value is stop


def test_new_labels_wrap_actual_router_and_gdn_methods_only(monkeypatch):
    import inspect
    from tensorfold.cuda.sm70.router_blas import RolePlans
    from tensorfold.families.qwen4_exp.cuda.gdn_state_batch import GDNStateBatch
    from tools import concurrent_observers
    monkeypatch.setattr(concurrent_observers, "_owner", None)
    plan = probe.project_observers(lambda _label: None)
    for owner, name, label in [(RolePlans, "run", "tensorfold/body/router-batched"),
                               (GDNStateBatch, "submit", "tensorfold/body/gdn-batched")]:
        matches = [row for row in plan if row[0] is owner and row[1] == name]
        assert len(matches) == 1
        assert matches[0][2] is getattr(owner, name)
        assert matches[0][3].__wrapped__ is matches[0][2]
        assert inspect.getclosurevars(matches[0][3]).nonlocals["label"] == label
    assert len(plan) <= 128


def test_phase_operation_boundary_preserves_real_operation_result_and_primary():
    from contextlib import contextmanager
    from tensorfold.families.qwen4_exp.cuda import execution_identity
    events = []
    @contextmanager
    def label(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))
    engine = NS(comm=None)
    def operation():
        return {"actual": 17}
    observed = probe.observed_phase_call(execution_identity._phase, "phase", label)
    assert observed(engine, "local", "prepare", operation) == [{"actual": 17}]
    assert events == [("enter", "phase"), ("enter", "tensorfold/cpu/phase-operation"),
                      ("exit", "tensorfold/cpu/phase-operation"), ("exit", "phase")]
    error = ValueError("actual operation refusal")
    def failed():
        raise error
    with pytest.raises(execution_identity._ManifestRefusal) as caught:
        observed(engine, "local", "prepare", failed)
    assert caught.value.__cause__ is error



def test_joint_labels_bind_existing_head_graph_owner_methods(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.target_greedy import TargetGreedy
    from tensorfold.families.qwen4_exp.cuda.bound_graphs import BoundStateGraphs
    from tensorfold.families.qwen4_exp.cuda.candidate_region_graphs import CandidateRegion
    from tensorfold.families.qwen4_exp.cuda.candidate_region_catalogue import Catalogue
    from tools import concurrent_observers
    monkeypatch.setattr(concurrent_observers, "_owner", None)
    plan = probe.project_observers(lambda _label: None)
    for owner, name in [(TargetGreedy, "gather"), (TargetGreedy, "finish"),
                        (BoundStateGraphs, "forward"), (CandidateRegion, "replay"), (Catalogue, "local_packet")]:
        matches = [row for row in plan if row[0] is owner and row[1] == name]
        assert len(matches) == 1 and matches[0][2] is getattr(owner, name)
        assert matches[0][3].__wrapped__ is matches[0][2]
    assert len(plan) <= 128
