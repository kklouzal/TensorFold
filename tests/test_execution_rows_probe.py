"""Existing row-budget API, factory and physical-geometry integration; no GPU."""
from types import SimpleNamespace as NS
import os
import pytest
from tools import v100_backend_probe as probe
from tests.test_v100_backend_probe import cli


def options(tmp_path, rows):
    return cli(tmp_path)+["--owner-batch-prelude-trial","--native-dense-trial","--session-ceiling","24",
        "--concurrency-grid","--fast-grid","--fast-grid-protocol","fast-grid-v2","--tokens","2048",
        "--quality-spotchecks","--execution-rows",str(rows)]


@pytest.mark.parametrize("profile",["source-dense-owner-batch-prelude-trial","source-dense-owner-expert72-trial"])
def test_explicit_rows_require_bounded_batch_profile_and_default32_is_preserved(tmp_path,monkeypatch,profile):
    from tensorfold.cuda.sm70.execution_policy import PROFILE_ENV
    assert probe.arguments(cli(tmp_path)).execution_rows==32
    monkeypatch.setenv(PROFILE_ENV,profile)
    for rows in (32,64,72):
        assert probe.arguments(options(tmp_path,rows)).execution_rows==rows
    for rows in (0,31,33,63,73,96):
        with pytest.raises(SystemExit):
            probe.arguments(options(tmp_path,rows))
    with pytest.raises(SystemExit):
        probe.arguments(cli(tmp_path)+["--execution-rows","64"])


@pytest.mark.parametrize("rows",[32,64,72])
@pytest.mark.parametrize("profile",["source-dense-owner-batch-prelude-trial","source-dense-owner-expert72-trial"])
def test_actual_worker_forwards_only_row_override_preserving_baseline_factory(tmp_path,monkeypatch,rows,profile):
    from tensorfold import hub,serve_options
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.families import qwen4_exp
    primary=ValueError("stop at real factory boundary")
    seen=[]
    def construct(*_args,**kwargs):
        seen.append(kwargs)
        raise primary
    monkeypatch.setattr(probe,"installed_identity",lambda _wheel:{})
    monkeypatch.setattr(hub,"resolve",lambda *_args,**_kwargs:tmp_path)
    monkeypatch.setattr(serve_options,"drafter_metadata_source",lambda _source:tmp_path)
    monkeypatch.setattr(qwen4_exp,"cuda_engine",construct)
    read_fd,write_fd=os.pipe()
    monkeypatch.setenv(FAILURE_FD_ENV,str(write_fd))
    args=NS(rank=1,wheel=tmp_path,model=tmp_path,mtp=tmp_path,master_port=29607,tokens=2048,
        out=tmp_path,requested_sm70_profile=profile,repetitions=1,
        execution_rows=rows,native_dense_trial=True,owner_batch_prelude_trial=True,session_ceiling=24)
    try:
        with pytest.raises(ValueError)as failure:
            probe.worker(args)
        assert failure.value is primary
        assert seen[0].get("execution_rows")== (None if rows==32 else rows)
        assert seen[0]["mtp_drafts"]==2 and seen[0]["mtp_confidence"]==0.7
        assert seen[0]["tp"]==8 and seen[0]["context"]==262144 and seen[0]["parallel"]==24
        assert seen[0]["kv_dtype"]=="int8" and seen[0]["graphs"]
    finally:
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.parametrize("rows",[32,64,72])
def test_actual_engine_row_and_floor_are_checked_before_caller_admission(rows):
    engine=NS(capacity_plan={"session_capacity":{"context_tokens":262144,"configured_ceiling":24,
        "resident_capacity":24,"full_context_commitments":24},"kv_full_windows":{"materialized":True},
        "execution_envelope":{"target_rows":rows}},execution_rows=rows,tp=8,depth=2,confidence=0.7,
        context_window=262144,indexer_history="gpu",kv_pair=NS(key_dtype="int8",value_dtype="int8"),
        w=NS(mtp=object(),meta={"execution_provider":"sm70","execution_dtype":"float16",
            "full":NS(hidden=2560,layers=48,vocab=248320)}),scheduler=NS(_request_limit=None),retention_catalog=object())
    probe.check_engine(engine,session_ceiling=24,execution_rows=rows)
    engine.execution_rows=rows+1
    with pytest.raises(ValueError,match="row budget"):
        probe.check_engine(engine,session_ceiling=24,execution_rows=rows)
    engine.execution_rows=rows
    engine.capacity_plan["session_capacity"]["resident_capacity"]=23
    with pytest.raises(ValueError):
        probe.check_engine(engine,session_ceiling=24,execution_rows=rows)


def test_existing_geometry_and_owner_buckets_fund_larger_rows_at_same24windows():
    from tensorfold.cuda.geometry import indexed_stream_geometry
    from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement
    from tensorfold.families.qwen4_exp.cuda.owner_execution import row_buckets
    from tests.test_adaptive_session_capacity import TEXT
    sizes=[]
    for rows in (32,64,72):
        geometry=indexed_stream_geometry(TEXT,streams=24,each=3,keep=8,mtp=True,world=8,
            full_windows=24,execution_rows=rows,prefill_rows=256,kv_pair=((8,32,2),)*2,
            history_placement=HistoryPlacement(8,2,12),rank=0,device_metadata_only=True)
        sizes.append(geometry.needed(262144))
        assert row_buckets(rows)[-1]==rows and max(row_buckets(rows))==rows
    assert sizes[0]<sizes[1]<sizes[2]
