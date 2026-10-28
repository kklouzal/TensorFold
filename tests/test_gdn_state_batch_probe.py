"""Closed FASTv2 GDN intent, real factory envelope and dispatch evidence controls."""
from types import SimpleNamespace as NS
import os

import pytest

torch = pytest.importorskip("torch")
from tools import v100_backend_probe as probe  # noqa: E402
from tests.test_v100_backend_probe import cli  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gdn_state_batch import GDNStateBatch, workspace_bytes  # noqa: E402


PROFILE = "source-dense-owner-batch-prelude-trial"
ENV = "TENSORFOLD_SM70_GDN_STATE_BATCH"


def options(tmp_path):
    return cli(tmp_path) + ["--owner-batch-prelude-trial", "--native-dense-trial", "--session-ceiling", "24",
        "--concurrency-grid", "--fast-grid", "--fast-grid-protocol", "fast-grid-v2", "--tokens", "2048",
        "--quality-spotchecks", "--gdn-state-batch-trial"]


def test_closed_explicit_gdn_intent_preserves_all_requirements(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", PROFILE)
    monkeypatch.setenv(ENV, "1")
    args = probe.arguments(options(tmp_path))
    assert args.gdn_state_batch_trial and args.owner_batch_prelude_trial and args.native_dense_trial
    assert args.session_ceiling == 24 and args.tokens == 2048 and args.quality_spotchecks
    assert (probe.CONTEXT, probe.DEPTH, probe.CONFIDENCE, probe.WORLD) == (262144, 2, .7, 8)


@pytest.mark.parametrize("bad", ["missing-env", "bad-env", "missing-intent", "missing-batch", "missing-quality", "wrong-profile"])
def test_partial_intent_refuses_before_factory(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", PROFILE)
    monkeypatch.setenv(ENV, "0" if bad == "bad-env" else "1")
    args = options(tmp_path)
    if bad == "missing-env":
        monkeypatch.delenv(ENV)
    elif bad == "missing-intent":
        args.remove("--gdn-state-batch-trial")
    elif bad == "missing-batch":
        args.remove("--owner-batch-prelude-trial")
    elif bad == "missing-quality":
        args.remove("--quality-spotchecks")
    elif bad == "wrong-profile":
        monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "reference")
    with pytest.raises(SystemExit):
        probe.arguments(args)


@pytest.mark.parametrize("count,delta,passed", [(1,0,True),(1,1,False),(2,1,True),(8,0,False),(16,36,True),(24,-1,False)])
def test_actual_submission_evidence_requires_n1_original_and_multistate_batch(count, delta, passed):
    before = {"counters": {"gdn.batch_submitted_layers": 100}}
    after = {"counters": {"gdn.batch_submitted_layers": 100 + delta}}
    assert probe.gdn_batch_evidence(count, before, after)["passed"] is passed
    assert not probe.gdn_batch_evidence(count, None, after)["passed"]


def engine():
    owner = object.__new__(GDNStateBatch)
    owner.layers = 36
    owner.host = torch.empty((36,24,10), dtype=torch.int64)
    owner.frame = torch.empty((24,4), dtype=torch.int64)
    return NS(multi=NS(buf=NS(gdn_state_batch=owner)), execution_rows=32,
        capacity_plan={"execution_envelope":{"target_rows":32},"host_workspace_bytes_estimate":8*workspace_bytes(36)[1]},
        w=NS(cfg=NS(nk=2,nv=6),meta={"sm70_gdn_state_batch_trial":True}))


def test_actual_factory_owner_and_funding_are_required():
    value = engine()
    assert probe.check_gdn_batch_engine(value)["extra_device_bytes"] == 0
    value.capacity_plan["host_workspace_bytes_estimate"] = 0
    with pytest.raises(ValueError, match="factory/funding"):
        probe.check_gdn_batch_engine(value)


@pytest.mark.parametrize("field", ["rows", "pointer-shape", "enabled"])
def test_changed_factory_shape_or_flag_refuses(field):
    value = engine()
    if field == "rows":
        value.execution_rows = 64
    elif field == "pointer-shape":
        value.multi.buf.gdn_state_batch.host = torch.empty(1)
    else:
        value.w.meta["sm70_gdn_state_batch_trial"] = False
    with pytest.raises(ValueError, match="factory/funding"):
        probe.check_gdn_batch_engine(value)


def test_actual_worker_preserves_factory_mtp_rows_and_floor(tmp_path, monkeypatch):
    from tensorfold import hub, serve_options
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.families import qwen4_exp

    primary = ValueError("stop at real factory boundary")
    seen = []
    def construct(*_args, **kwargs):
        seen.append(kwargs)
        raise primary
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(hub, "resolve", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(serve_options, "drafter_metadata_source", lambda _source: tmp_path)
    monkeypatch.setattr(qwen4_exp, "cuda_engine", construct)
    monkeypatch.setenv(ENV, "1")
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(FAILURE_FD_ENV, str(write_fd))
    args = NS(rank=1,wheel=tmp_path,model=tmp_path,mtp=tmp_path,master_port=29607,tokens=2048,
        out=tmp_path,requested_sm70_profile=PROFILE,repetitions=1,native_dense_trial=True,
        owner_batch_prelude_trial=True,gdn_state_batch_trial=True,session_ceiling=24)
    try:
        with pytest.raises(ValueError) as failure:
            probe.worker(args)
        assert failure.value is primary
        call = seen[0]
        assert call.get("execution_rows") is None  # Existing row32 default.
        assert (call["mtp_drafts"],call["mtp_confidence"],call["tp"],call["context"],call["parallel"]) == (2,.7,8,262144,24)
        assert call["kv_dtype"] == "int8" and call["graphs"] and call["output_queue_tokens"] == 2048
        assert os.environ[ENV] == "1"
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_actual_successful_follower_factory_passes_floor_checks_before_follow(tmp_path, monkeypatch):
    from tensorfold import hub, serve_options
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.families import qwen4_exp

    value = engine()
    value.tp, value.depth, value.confidence = 8, 2, .7
    value.context_window, value.indexer_history = 262144, "gpu"
    value.kv_pair = NS(key_dtype="int8", value_dtype="int8")
    value.scheduler, value.retention_catalog = None, object()
    value.w.mtp = object()
    value.w.meta.update(execution_provider="sm70", execution_dtype="float16",
                        full=NS(hidden=2560,layers=48,vocab=248320))
    value.capacity_plan.update(session_capacity={"context_tokens":262144,"configured_ceiling":24,
        "resident_capacity":24,"full_context_commitments":24}, kv_full_windows={"materialized":True})
    value.multi.w = value.w
    value.multi.buf.gdn_state_batch.submitted = 0
    calls = []
    primary = ValueError("stopped after successful follower admission")
    def follow():
        calls.append("follow")
        raise primary
    value.follow, value.close = follow, lambda: calls.append("close")
    def construct(*_args, **_kwargs):
        calls.append("construct")
        return value
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(probe, "record_execution_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hub, "resolve", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(serve_options, "drafter_metadata_source", lambda _source: tmp_path)
    monkeypatch.setattr(qwen4_exp, "cuda_engine", construct)
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(FAILURE_FD_ENV, str(write_fd))
    args = NS(rank=1,wheel=tmp_path,model=tmp_path,mtp=tmp_path,master_port=29607,tokens=2048,
        out=tmp_path,requested_sm70_profile=PROFILE,repetitions=1,native_dense_trial=True,
        owner_batch_prelude_trial=True,gdn_state_batch_trial=True,session_ceiling=24)
    try:
        with pytest.raises(ValueError) as failure:
            probe.worker(args)
        assert failure.value is primary
        assert calls == ["construct", "follow", "close"]
        with pytest.raises(ValueError, match="loaded engine"):
            probe.check_engine(value, session_ceiling=24)
        probe.check_engine(value, session_ceiling=24, require_scheduler=False)
    finally:
        os.close(write_fd)
        os.close(read_fd)
