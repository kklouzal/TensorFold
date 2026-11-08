"""Private entry/lifecycle boundary spies; no native arithmetic or target qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli
from tensorfold.cuda.cohort import FAILURE_FD_ENV
from tools import launch_cuda_tp as launcher
from tools import v100_owned_qualification as owned
from tests.cli_server_fixture import cuda_cli_app, cuda_cli_engine, install_cli_server


def test_scheduler_trace_selects_only_five_finite_nonwarmup_calls(tmp_path):
    import threading
    engine = NS(scheduler=NS(thread=threading.current_thread()))
    tracer = owned._SchedulerTrace(engine, tmp_path)
    stream = NS(done=False, st=object(), out=[1, 2])
    observer = NS(current=0, plan={"cohorts": [
        {"id": 0, "level": 1, "warmup": True},
        {"id": 1, "level": 1, "warmup": False},
        {"id": 2, "level": 8, "warmup": False}]},
        decoder=NS(streams={0: stream}, filling=[stream]))
    assert tracer.label(observer) is None
    observer.current = 1
    assert tracer.label(observer) == "c1-cold-prefill-containing-round"
    assert tracer.label(observer) is None
    observer.decoder.filling = []
    assert [tracer.label(observer) for _ in range(3)] == [
        "c1-warmed-decode-round-1", "c1-warmed-decode-round-2", None]
    observer.current = 2
    observer.decoder.streams = {n: NS(done=False, st=object(), out=[1, 2]) for n in range(8)}
    assert [tracer.label(observer) for _ in range(3)] == [
        "c8-warmed-decode-round-1", "c8-warmed-decode-round-2", None]


def test_scheduler_trace_preserves_primary_stop_export_and_receipt_failures(tmp_path, monkeypatch):
    import threading
    primary, stop, export, receipt = [RuntimeError(name) for name in ("model", "stop", "export", "receipt")]
    calls = []
    class Profiler:
        def __enter__(self):
            calls.append("start")
        def __exit__(self, kind, value, tb):
            assert value is primary
            calls.append("stop")
            raise stop
        def export_chrome_trace(self, path):
            calls.append("export")
            raise export
    engine = NS(rank=0, scheduler=NS(thread=threading.current_thread()))
    tracer = owned._SchedulerTrace(engine, tmp_path, factory=Profiler)
    monkeypatch.setattr(owned, "write_json", lambda *_: (_ for _ in ()).throw(receipt))
    def run():
        calls.append("work")
        raise primary
    with pytest.raises(RuntimeError) as raised:
        tracer.capture("c1-warmed-decode-round-1", run)
    def leaves(error):
        return sum((leaves(child) for child in error.exceptions), []) if isinstance(error, BaseExceptionGroup) else [error]
    assert raised.value is primary
    assert leaves(raised.value.__cause__) == [stop, export, receipt]
    assert calls == ["start", "work", "stop", "export"] and tracer.active is False


def test_scheduler_trace_refuses_foreign_thread_before_profiler_or_work(tmp_path):
    tracer = owned._SchedulerTrace(NS(scheduler=NS(thread=object())), tmp_path,
                                  factory=lambda: pytest.fail("profiler created on foreign thread"))
    with pytest.raises(RuntimeError, match="actual scheduler thread"):
        tracer.capture("c1-warmed-decode-round-1", lambda: pytest.fail("foreign work"))


def test_scheduler_trace_records_large_export_without_failing_completed_work(tmp_path):
    import threading
    size, result = (256 << 20) + 1, object()
    class Profiler:
        def __enter__(self):
            pass
        def __exit__(self, kind, value, tb):
            assert kind is value is tb is None
        def export_chrome_trace(self, path):
            # Only the export/size boundary is under test, not the vendor's JSON
            # encoder. Sparse storage avoids allocating a large test buffer.
            with open(path, "wb") as stream:
                stream.truncate(size)
    engine = NS(rank=0, scheduler=NS(thread=threading.current_thread()))
    tracer = owned._SchedulerTrace(engine, tmp_path, factory=Profiler)
    label = "c1-warmed-decode-round-1"
    assert tracer.capture(label, lambda: result) is result
    receipt = json.loads((tmp_path / (label + "-receipt.json")).read_text())
    assert receipt["exported_trace_bytes"] == size
    assert receipt["primary"] is None and receipt["post_errors"] == []
    assert tracer.active is False


@pytest.fixture
def channel(monkeypatch):
    read, write = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    monkeypatch.setenv(FAILURE_FD_ENV, str(write))
    try:
        yield read, write
    finally:
        os.close(write)
        os.close(read)


@pytest.mark.parametrize("dense,rows", [(None, None), ("reference", None), ("native32", None), (None, 40)])
def test_parent_delegates_eight_ranks_and_borrows_supervised_pipe(monkeypatch, tmp_path, dense, rows):
    captured = {}
    def supervise(commands, environments, **options):
        captured.update(commands=commands, environments=environments, options=options)
        os.fstat(options["pass_fds"][0])
        return 23
    monkeypatch.setattr(launcher, "supervise", supervise)
    selection = [] if dense is None else ["--dense-implementation", dense]
    if rows is not None:
        selection += ["--execution-rows", str(rows)]
    assert owned.main([*selection, "--gpus", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "new"),
                       "--", "model", "--backend", "cuda"]) == 23
    assert [env["CUDA_VISIBLE_DEVICES"] for env in captured["environments"]] == ["1", "2", "3", "4", "5", "6", "7", "0"]
    for command in captured["commands"]:
        assert command[:3] == [sys.executable, str(Path(owned.__file__).resolve()), "--worker"]
        if dense is None:
            assert "--dense-implementation" not in command
        else:
            assert command[command.index("--dense-implementation") + 1] == dense
        if rows is None:
            assert "--execution-rows" not in command
        else:
            assert command[command.index("--execution-rows") + 1] == str(rows)
            assert command.index("--execution-rows") < command.index("--")
        assert command[command.index("--tp") + 1] == "8"
        assert command[command.index("--") + 1] == "model"
    assert captured["options"]["stop_on_departure"] is True
    with pytest.raises(OSError):
        os.fstat(captured["options"]["pass_fds"][0])


def test_parent_is_torch_and_cuda_free_in_fresh_interpreter(tmp_path):
    code = '''import sys
from tools import v100_owned_qualification as owned, launch_cuda_tp
assert 'torch' not in sys.modules
launch_cuda_tp.supervise=lambda *a,**k: 0
assert owned.main(['--gpus','0,1,2,3,4,5,6,7','--out',sys.argv[1],'--','model'])==0
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "new")], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_private_worker_requires_supervised_capability(monkeypatch, tmp_path):
    monkeypatch.delenv(FAILURE_FD_ENV, raising=False)
    with pytest.raises(ValueError, match="supervised failure pipe"):
        owned.worker(["--out", str(tmp_path)])
    with (tmp_path / "regular").open("w") as stream:
        monkeypatch.setenv(FAILURE_FD_ENV, str(stream.fileno()))
        with pytest.raises(ValueError, match="supervisor writer"):
            owned.worker(["--out", str(tmp_path)])


def worker_args(tmp_path):
    return ["--out", str(tmp_path), "--", "model", "--backend", "cuda", "--tp", "8",
            "--rank", "3", "--expert-tensor-parallel-size", "4"]


def test_worker_uses_shared_parser_and_typed_private_capability(channel, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "cmd_serve", lambda args: seen.append(args) or 17)
    assert owned.worker(worker_args(tmp_path)) == 17
    assert type(seen[0]._owned_expert_qualification) is cli._OwnedExpertQualification
    assert seen[0].rank == 3 and seen[0].tp == 8
    os.fstat(channel[1])  # Borrowed until rank exit, not closed by the worker.
    with pytest.raises(BlockingIOError):
        os.read(channel[0], 8)



@pytest.mark.parametrize("dense", [None, "reference", "native32"])
def test_dense_worker_selects_before_constructor_and_records_every_rank(channel, monkeypatch, tmp_path, dense):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat
    # An omitted selector preserves existing process behavior rather than silently resetting it.
    monkeypatch.setattr(GGUFDense, "implementation", "reference" if dense else "native32")
    monkeypatch.setattr(GGUFConcat, "implementation", "reference")
    monkeypatch.delenv("TENSORFOLD_SM70_PROFILE", raising=False)
    expected = dense or "native32"
    monkeypatch.setattr(owned, "bound_receipt", lambda engine: {"rank": engine.rank})
    calls = []
    def serve(args):
        assert GGUFDense.implementation == expected and GGUFConcat.implementation == "reference"
        engine = NS(rank=args.rank, w=NS(meta={"sm70_profile": "reference"}),
                    comm=NS(ready=lambda label: calls.append((args.rank, label))))
        args._owned_expert_qualification.before_listener(engine)
        return 0
    monkeypatch.setattr(cli, "cmd_serve", serve)
    selection = [] if dense is None else ["--dense-implementation", dense]
    for rank in range(8):
        argv = worker_args(tmp_path)
        argv[argv.index("--rank") + 1] = str(rank)
        assert owned.worker([*selection, *argv]) == 0
        receipt = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert receipt["dense_implementation"] == {"requested": dense, "actual": expected, "concat": "reference"}
    assert calls == [(rank, "owned_expert_qualification") for rank in range(8)]


@pytest.mark.parametrize("bad", ["dense", "concat", "profile"])
def test_dense_trial_mismatch_refuses_receipt_and_ready(channel, monkeypatch, tmp_path, bad):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat
    monkeypatch.setattr(GGUFDense, "implementation", "reference" if bad == "dense" else "native32")
    monkeypatch.setattr(GGUFConcat, "implementation", "native32" if bad == "concat" else "reference")
    engine = NS(rank=2, _abort_comm=False,
                w=NS(meta={"sm70_profile": "optimized" if bad == "profile" else "reference"}),
                comm=NS(ready=lambda _: pytest.fail("mismatch reached all-rank ready")))
    monkeypatch.setattr(owned, "bound_receipt", lambda _: {})
    with pytest.raises(ValueError, match="selected reference-profile binding"):
        owned.publish_receipt(engine, tmp_path, dense_implementation="native32")
    assert engine._abort_comm and os.read(channel[0], 8) == b"F"
    assert not (tmp_path / "rank-2.json").exists()


def test_dense_trial_refuses_whole_optimized_profile_before_constructor(channel, monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "optimized")
    monkeypatch.setattr(cli, "cmd_serve", lambda _: pytest.fail("unqualified profile reached constructor"))
    with pytest.raises(ValueError, match="requires the reference SM70 profile"):
        owned.worker(["--dense-implementation", "native32", *worker_args(tmp_path)])
    assert os.read(channel[0], 8) == b"F"


def test_private_dense_choice_refuses_invalid_parent_and_worker(channel, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "cmd_serve", lambda _: pytest.fail("invalid choice reached constructor"))
    monkeypatch.setattr(launcher, "main", lambda *a, **k: pytest.fail("invalid choice launched workers"))
    with pytest.raises(SystemExit) as parent:
        owned.main(["--dense-implementation", "wmma", "--gpus", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "new"), "--", "model"])
    with pytest.raises(SystemExit) as worker:
        owned.worker(["--dense-implementation", "wmma", *worker_args(tmp_path)])
    assert parent.value.code == worker.value.code == 2
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("requested,configured", [("inherit", "inherit"), ("native-half-v1", "native-half-v1"),
                                                   (None, "inherit"), (None, None)])
@pytest.mark.parametrize("native_family", [False, True])
def test_half_attestation_precedes_constructor_and_keeps_initial_request(
        channel, monkeypatch, tmp_path, requested, configured, native_family):
    from tensorfold.cuda.sm70 import family_ops
    name = "TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL"
    if requested is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, requested)
    if configured is None:
        monkeypatch.delattr(family_ops, "_PRIVATE_HALF_PARTIAL", raising=False)
    else:
        monkeypatch.setattr(family_ops, "_PRIVATE_HALF_PARTIAL", configured)
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", native_family)
    monkeypatch.setattr(owned, "bound_receipt", lambda _: {})
    calls = []
    def serve(args):
        # A later environment edit cannot select another policy or replace the captured request.
        monkeypatch.setenv(name, "changed-after-preparation")
        engine = NS(rank=3, comm=NS(ready=lambda label: calls.append(label)))
        args._owned_expert_qualification.before_listener(engine)
        return 0
    monkeypatch.setattr(cli, "cmd_serve", serve)
    assert owned.worker(worker_args(tmp_path)) == 0
    receipt = json.loads((tmp_path / "rank-3.json").read_text())["half_partial_policy"]
    assert receipt == {"requested": requested, "configured": configured or "legacy-inherited",
                       "private_selector_supported": configured is not None,
                       "inherited_native_family": native_family}
    assert getattr(family_ops, "_PRIVATE_HALF_PARTIAL", None) == configured
    assert calls == ["owned_expert_qualification"]


@pytest.mark.parametrize("requested,configured", [("inherit", None), ("native-half-v1", None),
                                                   ("native-half-v1", "inherit"), ("invalid", "inherit"),
                                                   (None, "native-half-v1")])
def test_half_attestation_refuses_unsupported_or_mismatched_before_constructor(
        channel, monkeypatch, tmp_path, requested, configured):
    from tensorfold.cuda.sm70 import family_ops
    if requested is None:
        monkeypatch.delenv("TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL", raising=False)
    else:
        monkeypatch.setenv("TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL", requested)
    if configured is None:
        monkeypatch.delattr(family_ops, "_PRIVATE_HALF_PARTIAL", raising=False)
    else:
        monkeypatch.setattr(family_ops, "_PRIVATE_HALF_PARTIAL", configured)
    monkeypatch.setattr(cli, "cmd_serve", lambda _: pytest.fail("Half request reached constructor"))
    with pytest.raises(ValueError, match="Half policy request"):
        owned.worker(worker_args(tmp_path))
    assert os.read(channel[0], 8) == b"F"
    assert not (tmp_path / "rank-3.json").exists()


def test_half_attestation_configuration_drift_refuses_publication(channel, monkeypatch, tmp_path):
    from tensorfold.cuda.sm70 import family_ops
    monkeypatch.setenv("TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL", "native-half-v1")
    monkeypatch.setattr(family_ops, "_PRIVATE_HALF_PARTIAL", "native-half-v1")
    monkeypatch.setattr(owned, "bound_receipt", lambda _: {})
    def serve(args):
        monkeypatch.setattr(family_ops, "_PRIVATE_HALF_PARTIAL", "inherit")
        engine = NS(rank=3, _abort_comm=False,
                    comm=NS(ready=lambda _: pytest.fail("Half drift reached ready")))
        with pytest.raises(ValueError, match="Half policy request"):
            args._owned_expert_qualification.before_listener(engine)
        assert engine._abort_comm and os.read(channel[0], 8) == b"F"
        assert not (tmp_path / "rank-3.json").exists()
        return 0
    monkeypatch.setattr(cli, "cmd_serve", serve)
    assert owned.worker(worker_args(tmp_path)) == 0


@pytest.mark.parametrize("field,bad", [("_PRIVATE_HALF_PARTIAL", None), ("_PRIVATE_HALF_PARTIAL", "changed"),
                                      ("_PRIVATE_HALF_PARTIAL", "legacy-inherited"),
                                      ("_NATIVE_FAMILY", None), ("_NATIVE_FAMILY", 1), ("_NATIVE_FAMILY", "false")])
@pytest.mark.parametrize("stage", ["constructor", "publication"])
def test_half_attestation_refuses_corrupted_present_metadata(
        channel, monkeypatch, tmp_path, field, bad, stage):
    from tensorfold.cuda.sm70 import family_ops
    monkeypatch.delenv("TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL", raising=False)
    monkeypatch.setattr(family_ops, "_PRIVATE_HALF_PARTIAL", "inherit")
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
    monkeypatch.setattr(owned, "bound_receipt", lambda _: {})
    engine = NS(rank=3, _abort_comm=False,
                comm=NS(ready=lambda _: pytest.fail("corrupted metadata reached ready")))
    def serve(args):
        assert stage == "publication", "corrupted metadata reached constructor"
        monkeypatch.setattr(family_ops, field, bad)
        args._owned_expert_qualification.before_listener(engine)
    monkeypatch.setattr(cli, "cmd_serve", serve)
    if stage == "constructor":
        monkeypatch.setattr(family_ops, field, bad)
    with pytest.raises(ValueError, match="private Half"):
        owned.worker(worker_args(tmp_path))
    assert os.read(channel[0], 8) == b"F"
    if stage == "publication":
        assert engine._abort_comm
    assert not (tmp_path / "rank-3.json").exists()

def test_failed_receipt_poison_notifies_once_before_shared_cli_retirement(channel, monkeypatch, tmp_path):
    engine = NS(rank=3, _abort_comm=False)
    primary = ValueError("actual native binding mismatch")
    monkeypatch.setattr(owned, "bound_receipt", lambda _engine: (_ for _ in ()).throw(primary))
    def serve(args):
        try:
            args._owned_expert_qualification.before_listener(engine)
        finally:
            assert engine._abort_comm
            assert os.read(channel[0], 8) == b"F"
    monkeypatch.setattr(cli, "cmd_serve", serve)
    with pytest.raises(ValueError) as caught:
        owned.worker(worker_args(tmp_path))
    assert caught.value is primary
    with pytest.raises(BlockingIOError):
        os.read(channel[0], 8)


def test_receipt_publishes_before_existing_all_rank_ready(channel, monkeypatch, tmp_path):
    calls = []
    engine = NS(rank=0, _abort_comm=False)
    engine.comm = NS(ready=lambda label: calls.append((label, json.loads((tmp_path / "rank-0.json").read_text()))))
    monkeypatch.setattr(owned, "bound_receipt", lambda _engine: {"qualification": "host-boundary-fixture"})
    owned.publish_receipt(engine, tmp_path)
    assert calls[0][0] == "owned_expert_qualification"
    assert len(calls[0][1]["tool_sha256"]) == len(calls[0][1]["runtime_cli_sha256"]) == 64
    with pytest.raises(ValueError, match="already exists"):
        owned.publish_receipt(engine, tmp_path)
    assert engine._abort_comm and os.read(channel[0], 8) == b"F"


def test_public_parser_cannot_select_native_and_untyped_marker_fails_closed(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["serve", "model", "--owned-expert-native"])
    assert error.value.code == 2
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-update-check"])
    args._owned_expert_qualification = True
    with pytest.raises(TypeError, match="capability"):
        cli.cmd_serve(args)


@pytest.mark.parametrize("private,rows", [(False, None), (True, None), (True, 40)])
def test_shared_cuda_factory_gets_private_bool_only_with_capability(monkeypatch, tmp_path, request, private, rows):
    import tensorfold.cuda.server as server
    (tmp_path / "config.json").write_text('{"max_position_embeddings":128}')
    calls, options = [], []
    engine = cuda_cli_engine(close=lambda: calls.append("engine-close"))
    family = NS(package=NS(cuda_engine=lambda *a, **k: options.append(k) or engine), title="Flash Next", model_type="qwen4_exp")
    monkeypatch.setattr(server, "App", lambda *a, **k: cuda_cli_app(effective_context_window=128))
    install_cli_server(monkeypatch, request, server, on_serve=lambda _app: calls.append("serve"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    if private:
        args._owned_expert_qualification = cli._OwnedExpertQualification(
            lambda selected: calls.append("receipt") if selected is engine else pytest.fail("different engine"),
            execution_rows=rows)
    assert cli._serve_cuda(args, family, tmp_path, 128) == 0
    assert options[0].get("_owned_expert_native", False) is private
    if not private:
        assert "_owned_expert_native" not in options[0]
    assert options[0].get("execution_rows") == rows
    if rows is None:
        assert "execution_rows" not in options[0]
    assert calls == (["receipt"] if private else []) + ["serve", "engine-close"]


def test_rejected_private_binding_never_builds_listener_and_retires_engine(monkeypatch, tmp_path):
    import tensorfold.cuda.server as server
    (tmp_path / "config.json").write_text('{"max_position_embeddings":128}')
    calls = []
    primary = ValueError("receipt refused")
    engine = cuda_cli_engine(close=lambda: calls.append("engine-close"))
    family = NS(package=NS(cuda_engine=lambda *a, **k: engine), title="Flash Next", model_type="qwen4_exp")
    monkeypatch.setattr(server, "App", lambda *a, **k: pytest.fail("refused qualification reached listener frontend"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    args._owned_expert_qualification = cli._OwnedExpertQualification(lambda _engine: (_ for _ in ()).throw(primary))
    with pytest.raises(ValueError) as caught:
        cli._serve_cuda(args, family, tmp_path, 128)
    assert caught.value is primary and calls == ["engine-close"]


def test_family_wrapper_preserves_private_request_and_normal_default(monkeypatch, tmp_path):
    from gguf_fixture import tiny_model
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine as implementation
    made = []
    monkeypatch.setattr(implementation, "FlashNextEngine", lambda *a, **k: made.append(k) or NS(**k))
    path = tiny_model(tmp_path / "target.gguf")
    options = dict(no_drafts=True, tp=8, rank=0, kv_dtype="fp16", expert_tensor_parallel_size=4,
                   max_waiting_input_bytes=16384)
    qwen4_exp.cuda_engine(path, **options)
    assert "_owned_expert_native" not in made[-1]
    qwen4_exp.cuda_engine(path, **options, _owned_expert_native=True)
    assert made[-1]["_owned_expert_native"] is True and made[-1]["adaptive_sessions"] is True
    with pytest.raises(TypeError, match="literal boolean"):
        qwen4_exp.cuda_engine(path, **options, _owned_expert_native="true")


@pytest.mark.parametrize("bad", [None, "layers", "bank-mode", "arena", "state", "coordinate", "coordinate-marker", "label", "cpu"])
def test_bound_projection_checks_actual_plan_policy_resource_and_numerical_owners(monkeypatch, bad):
    """Typed metadata/ownership boundary spies; mocked capability is NOT SM70 evidence."""
    import torch
    from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts
    from tensorfold.families.qwen4_exp.cuda.owned_experts import Plan, POLICY

    partition = ExpertPartition(8, 0, 512, 640, 4)
    plan = Plan(partition, 2560, 11, 32, 256, False)
    bank = GGUFExperts(NS(n=256 * 160), NS(n=256 * 160), NS(n=256 * 2560),
                      (NS(n=80), NS(n=80), NS(k=80)), count=512, width=160, dims=2560, partition=partition)
    w = NS(cfg=NS(hidden=2560, top_k=10, moe_width=160, shared_width=80, layers=48),
           meta={"expert_partition": partition.identity(), "coordinate_profile_identity": "a" * 64},
           layers=[NS(moe=NS(experts=bank)) for _ in range(48)],
           mtp=None, device=torch.device("cuda"))
    plan.bind(w)
    resource = plan.resources(5)
    resource.update(materialized_local_arena_bytes=resource["local_arena_bytes"], arena_rows=[5, 256])
    decoder = NS(w=w, owned_expert_resources=dict(resource),
                 states=[NS(kv_identity=(("owned-expert-native", plan.identity), ("coordinate-profile-v1", "a" * 64)),
                            coordinate_profile_identity="a" * 64)])
    monkeypatch.setattr(Plan, "reconcile", lambda self, selected: dict(resource) if selected is decoder else pytest.fail("wrong decoder"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    engine = NS(w=w, _owned_expert_plan=plan, multi=decoder, tp=8, rank=0, expert_partition=partition,
                capacity_plan={"owned_expert_native": dict(resource), "expert_parallel": {"implementation": POLICY}})
    if bad == "layers":
        w.layers.pop()
    elif bad == "bank-mode":
        bank.implementation = "reference"
    elif bad == "arena":
        decoder.owned_expert_resources["local_arena_bytes"] += 1
    elif bad == "state":
        decoder.states[0].kv_identity = ()
    elif bad == "coordinate":
        w.meta["coordinate_profile_identity"] = decoder.states[0].coordinate_profile_identity = None
    elif bad == "coordinate-marker":
        decoder.states[0].kv_identity = (("owned-expert-native", plan.identity),)
    elif bad == "label":
        engine.capacity_plan["expert_parallel"]["implementation"] = "reference-owned-rows-f32-v1"
    elif bad == "cpu":
        w.device = torch.device("cpu")
    if bad is None:
        receipt = owned.bound_receipt(engine)
        assert receipt["policy"] == POLICY and receipt["plan_identity"] == list(plan.identity)
        assert receipt["resources"] == resource and receipt["target_banks"] == 48 and receipt["mtp_banks"] == 0
    else:
        with pytest.raises(ValueError):
            owned.bound_receipt(engine)


def test_normal_launcher_default_command_is_byte_equivalent(monkeypatch):
    seen = []
    monkeypatch.setattr(launcher, "supervise", lambda commands, envs, **kwargs: seen.extend(commands) or 0)
    assert launcher.main(["--gpus", "0,1", "--", "model", "--context", "8192"]) == 0
    assert seen == [[sys.executable, "-m", "tensorfold", "serve", "model", "--context", "8192",
                     "--tp", "2", "--rank", str(rank), "--master", "127.0.0.1", "--master-port", "29551"]
                    for rank in (1, 0)]


@pytest.mark.parametrize("backend,name", [("cuda", "other.family"), ("mlx", "tensorfold.families.qwen4_exp")])
def test_private_capability_refuses_wrong_backend_family_before_weight_resolution(monkeypatch, tmp_path, backend, name):
    from tensorfold import families, hub
    family = NS(package=NS(__name__=name, cuda_engine=lambda *_a, **_k: None, load=lambda *_a, **_k: None),
                title="fixture", model_type="fixture")
    monkeypatch.setattr(families, "detect", lambda _: family)
    monkeypatch.setattr(cli, "_config_dir", lambda _: tmp_path)
    monkeypatch.setattr(hub, "resolve", lambda *_a, **_k: pytest.fail("invalid private mode resolved payload"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", backend, "--no-update-check"])
    args._owned_expert_qualification = cli._OwnedExpertQualification(lambda _: None)
    with pytest.raises(ValueError, match="Flash Next CUDA"):
        cli.cmd_serve(args)


@pytest.mark.skipif(sys.platform != "linux", reason="same-host supervisor requires Linux session ownership")
def test_private_prefix_uses_real_supervisor_failure_and_process_retirement(monkeypatch):
    import tensorfold.cuda.expert_launch as sessions
    children = []
    original = sessions.subprocess.Popen
    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(sessions.subprocess, "Popen", spawn)
    supervise = launcher.supervise
    monkeypatch.setattr(launcher, "supervise", lambda commands, envs, **kw: supervise(commands, envs, **{**kw, "grace": .05}))
    script = f"import os,time;os.write(int(os.environ[{FAILURE_FD_ENV!r}]),b'F');time.sleep(60)"
    assert launcher.main(["--gpus", "0,1", "--", "controlled-cpu-worker"],
                         _command_prefix=[sys.executable, "-c", script]) == 1
    assert len(children) == 2 and all(child.returncode is not None for child in children)
    for child in children:
        with pytest.raises(ProcessLookupError):
            os.kill(child.pid, 0)


def history_fixture(tmp_path):
    coordinate, identity = 'a' * 64, ('owned-etp4-native-slots-v1', 8, 4)
    states = [NS(pos=12, history_slot=index, slot_generation=3, history_generation=7,
                 coordinate_profile_identity=coordinate,
                 kv_identity=(("owned-expert-native", identity), ("coordinate-profile-v1", coordinate)))
              for index in range(2)]
    streams = [NS(st=state, done=False, prompt=[index + 1] * 12, count=4) for index, state in enumerate(states)]
    result, calls = object(), []
    def original():
        calls.append('round')
        return result
    decoder = NS(states=states, streams=dict(enumerate(streams)), filling=[], round=original)
    engine = NS(multi=decoder, tp_decoder=NS(wires={id(stream): index for index, stream in enumerate(streams)}, prefix_operation=19),
                w=NS(meta={"coordinate_profile_identity": coordinate}), _owned_expert_plan=NS(identity=identity),
                rank=0, tp=8, _abort_comm=False, context_window=16,
                _history_run_identity="c" * 32, _history_server_instance="instance")
    data = {"C": 2, "N": 12, "R": 4, "L": 16,
            "prompt_token_ids_sha256": [owned._prompt_digest(stream.prompt) for stream in streams]}
    resources = {"receipt_bytes_limit": 6144, "global_host_obligation_bytes": 8000}
    observer = owned._HistoryAttestation(engine, data, 'b' * 64, tmp_path, resources)
    return engine, data, observer, calls, result


def test_streamed_prompt_hash_matches_independent_compact_json():
    import hashlib
    values = [0, 1, 10, 248046]
    assert owned._prompt_digest(values) == hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
    assert owned._prompt_digest(iter(values)) == owned._prompt_digest(values)
    for bad in ([True], [-1], [1 << 31], ['1']):
        with pytest.raises(ValueError):
            owned._prompt_digest(bad)


def test_private_round_captures_before_mutation_publishes_after_and_restores(tmp_path, monkeypatch):
    engine, data, observer, calls, result = history_fixture(tmp_path)
    original = observer.original
    def step():
        engine.multi.states[0].pos = 13
        return original()
    observer.original = step
    publish = owned.write_json
    def write(path, record):
        assert calls == ['round'] and engine.multi.states[0].pos == 13
        assert record['owners'][0]['cached_length'] == 12
        publish(path, record)
    monkeypatch.setattr(owned, 'write_json', write)
    observer.install()
    assert engine.multi.round() is result
    receipt = json.loads((tmp_path / 'history-rank-0.json').read_text())
    assert receipt['prefix_sequence'] == 19 and receipt['C'] == 2
    assert receipt['input_sha256'] == 'b' * 64
    assert engine.multi.round is step
    assert engine.multi.round() is result and calls == ['round', 'round']


@pytest.mark.parametrize('incomplete', ['underfilled', 'done', 'filling'])
def test_private_round_no_witness_keeps_original_work_without_success(tmp_path, incomplete):
    engine, data, observer, calls, result = history_fixture(tmp_path)
    if incomplete == 'underfilled':
        engine.multi.states[0].pos = 11
    elif incomplete == 'done':
        engine.multi.streams[0].done = True
    else:
        engine.multi.filling = [engine.multi.streams[0]]
    observer.install()
    assert engine.multi.round() is result and calls == ['round']
    assert not (tmp_path / 'history-rank-0.json').exists()


@pytest.mark.parametrize('bad', ['alias', 'coordinate', 'native', 'poisoned', 'prompt', 'wire', 'slot', 'generation', 'count'])
def test_private_round_refuses_falsified_owner_before_model(channel, tmp_path, bad):
    engine, data, observer, calls, result = history_fixture(tmp_path)
    state = engine.multi.states[0]
    if bad == 'alias':
        engine.multi.streams[1].st = state
    elif bad == 'coordinate':
        state.coordinate_profile_identity = 'c' * 64
    elif bad == 'native':
        state.kv_identity = ()
    elif bad == 'poisoned':
        state._history_error = RuntimeError('bad history')
    elif bad == 'prompt':
        engine.multi.streams[0].prompt[0] += 1
    elif bad == 'wire':
        engine.tp_decoder.wires[id(engine.multi.streams[1])] = 0
    elif bad == 'slot':
        state.history_slot = 1
    elif bad == 'generation':
        state.history_generation = -1
    else:
        engine.multi.streams[0].count = 3
    observer.install()
    with pytest.raises(ValueError):
        engine.multi.round()
    assert calls == [] and engine._abort_comm and os.read(channel[0], 8) == b'F'
    assert not (tmp_path / 'history-rank-0.json').exists()


def test_private_model_failure_retains_primary_without_publishing(tmp_path):
    engine, data, observer, calls, result = history_fixture(tmp_path)
    primary = RuntimeError('model step')
    def step():
        calls.append('round')
        raise primary
    observer.original = step
    observer.install()
    with pytest.raises(RuntimeError) as caught:
        engine.multi.round()
    assert caught.value is primary and calls == ['round']
    assert not (tmp_path / 'history-rank-0.json').exists()


def test_private_receipt_failure_notifies_and_preserves_primary(channel, tmp_path, monkeypatch):
    engine, data, observer, calls, result = history_fixture(tmp_path)
    primary = OSError('durability')
    monkeypatch.setattr(owned, 'write_json', lambda *a: (_ for _ in ()).throw(primary))
    observer.install()
    with pytest.raises(OSError) as caught:
        engine.multi.round()
    assert caught.value is primary and calls == ['round'] and engine._abort_comm
    assert os.read(channel[0], 8) == b'F'


def test_history_data_and_actual_preparation_bind_paid_geometry(tmp_path):
    engine, data, observer, calls, result = history_fixture(tmp_path)
    source = tmp_path / 'input.json'
    source.write_text(json.dumps(data))
    parsed, digest, limit = owned._history_input(source, 2, 16)
    assert parsed == data and limit == 1536 and len(digest) == 64
    engine._owned_qualification_host = (8000, digest)
    engine.retention_catalog = NS(host_resources=NS(baseline_host_bytes=8000))
    assert owned.prepare_history(engine, (parsed, digest, observer.resources), tmp_path) == observer.resources
    engine._owned_qualification_host = (1, digest)
    with pytest.raises(ValueError, match='paid obligation'):
        owned.prepare_history(engine, (parsed, digest, observer.resources), tmp_path)


@pytest.mark.parametrize('change', ['bool', 'duplicate', 'samehash', 'wrongcontext', 'oversize'])
def test_history_input_rejects_untrusted_data(tmp_path, change):
    engine, data, observer, calls, result = history_fixture(tmp_path)
    raw = json.dumps(data)
    if change == 'bool':
        data['C'] = True
        raw = json.dumps(data)
    elif change == 'duplicate':
        raw = raw[:-1] + ',"C":2}'
    elif change == 'samehash':
        data['prompt_token_ids_sha256'][1] = data['prompt_token_ids_sha256'][0]
        raw = json.dumps(data)
    elif change == 'wrongcontext':
        data['L'] = 17
        raw = json.dumps(data)
    else:
        raw += ' ' * 2000
    source = tmp_path / 'data.json'
    source.write_text(raw)
    with pytest.raises(ValueError):
        owned._history_input(source, 2, 16)


def test_private_host_obligation_defaults_and_validation():
    def callback(engine):
        return None
    assert cli._OwnedExpertQualification(callback).host_obligation == ()
    assert cli._OwnedExpertQualification(callback, (8000, 'a' * 64)).host_obligation[0] == 8000
    for invalid in ([], (True, 'a' * 64), (-1, 'a' * 64), (8000, 'bad'), (8000, 'A' * 64)):
        with pytest.raises(TypeError):
            cli._OwnedExpertQualification(callback, invalid)


def test_history_worker_prepares_obligation_before_shared_constructor(channel, monkeypatch, tmp_path):
    engine, data, observer, calls, result = history_fixture(tmp_path)
    path = tmp_path / 'data.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(owned, '_history_resources', lambda count, limit: observer.resources)
    seen = []
    monkeypatch.setattr(cli, 'cmd_serve', lambda args: seen.append(args._owned_expert_qualification) or 0)
    assert owned.worker(['--history-input', str(path), '--history-run-id', 'c' * 32, *worker_args(tmp_path)]) == 0
    assert seen[0].host_obligation == (8000, __import__('hashlib').sha256(path.read_bytes()).hexdigest())


def test_failed_history_input_notifies_existing_supervisor(channel, tmp_path):
    source = tmp_path / 'bad.json'
    source.write_text('{}')
    with pytest.raises(ValueError):
        owned.worker(['--history-input', str(source), '--history-run-id', 'c' * 32, *worker_args(tmp_path)])
    assert os.read(channel[0], 8) == b'F'


@pytest.fixture
def history_acceptance_packet(tmp_path, monkeypatch):
    import copy
    from test_v100_adaptive_exercise import supplied_long_arguments, correctness_service, actual_client_module
    from tools import v100_adaptive_exercise as exercise
    monkeypatch.setattr(exercise.live, 'Client', actual_client_module().Client)
    args = supplied_long_arguments(tmp_path)
    with correctness_service(long_capacity=2) as (args.url, service):
        assert exercise.run(args) == 2
    client_path = args.out / 'suite.json'
    client = json.loads(client_path.read_text())
    result = client['supplied_long']
    root = tmp_path / 'workers'
    root.mkdir()
    for rank in range(8):
        owners = [{'wire': index, 'slot': index, 'slot_generation': 3, 'ownership_generation': 7,
                   'cached_length': 4032, 'prompt_tokens': 4032, 'prompt_token_ids_sha256': pair['prompt_token_ids_sha256']}
                  for index, pair in enumerate(result['cases'].values())]
        receipt = {'schema': 'tensorfold.private-owned-history-v1', 'rank': rank, 'world': 8,
                   'prefix_sequence': 19, 'input_sha256': 'b' * 64, 'C': 2, 'N': 4032, 'R': 64, 'L': 4096,
                   'coordinate_profile_identity': 'a' * 64, 'plan_identity': ['owned', 8, 4], 'owners': owners,
                   'run_identity': 'c' * 32, 'server_instance_id': client['initial_stats']['server']['instance_id']}
        startup = {key: copy.deepcopy(receipt[key]) for key in ('run_identity', 'server_instance_id', 'coordinate_profile_identity', 'plan_identity')}
        startup.update(schema='tensorfold.private-owned-qualification-v1', rank=rank, world=8)
        startup['history_host_obligation'] = {'input_sha256': 'b' * 64}
        (root / f'history-rank-{rank}.json').write_text(json.dumps(receipt))
        (root / f'rank-{rank}.json').write_text(json.dumps(startup))
    return root, client_path


def test_offline_intersection_binds_completed_client_and_all_rank_facts(history_acceptance_packet):
    root, client = history_acceptance_packet
    accepted = owned.accept_history(root, client)
    assert accepted['passed'] and accepted['facts']['prefix_sequence'] == 19
    assert accepted['run_identity'] == 'c' * 32
    assert 'not GPU wall-clock' in accepted['scope']
    assert json.loads(client.read_text())['passed'] is False  # Source advisory report is never rewritten.


@pytest.mark.parametrize('bad', ['sequence', 'run', 'instance', 'input', 'prompt', 'generation', 'client-answer', 'client-identity', 'missing'])
def test_offline_intersection_refuses_unbound_or_inconsistent_evidence(history_acceptance_packet, bad):
    root, client = history_acceptance_packet
    path = root / 'history-rank-0.json'
    receipt = json.loads(path.read_text())
    if bad == 'sequence':
        receipt['prefix_sequence'] += 1
    elif bad == 'run':
        receipt['run_identity'] = 'd' * 32
    elif bad == 'instance':
        receipt['server_instance_id'] = 'different-service'
    elif bad == 'input':
        receipt['input_sha256'] = 'd' * 64
    elif bad == 'prompt':
        receipt['owners'][0]['prompt_token_ids_sha256'] = 'd' * 64
    elif bad == 'generation':
        receipt['owners'][0]['slot_generation'] += 1
    elif bad.startswith('client-'):
        report = json.loads(client.read_text())
        if bad == 'client-answer':
            report['supplied_long']['cases']['case-0']['mtp']['choices'][0]['message']['content'] = 'wrong'
        else:
            report['final_stats']['server']['instance_id'] = 'another-server'
        client.write_text(json.dumps(report))
    else:
        path.unlink()
    if bad != 'missing':
        path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError)):
        owned.accept_history(root, client)


@pytest.mark.parametrize('bad', ['no-work', 'mtp-dispatch', 'serial-dispatch', 'serial-work', 'counter-bool'])
def test_offline_acceptor_recomputes_actual_mode_work(history_acceptance_packet, bad):
    root, client = history_acceptance_packet
    report = json.loads(client.read_text())
    result = report['supplied_long']
    if bad == 'no-work':
        for pair in result['cases'].values():
            pair['mtp']['tensorfold']['drafted'] = 0
        for row in result['concurrent']['results']:
            row['reply']['tensorfold']['drafted'] = 0
    elif bad == 'mtp-dispatch':
        result['cases']['case-0']['mtp']['tensorfold']['drafts'] = False
    elif bad == 'serial-dispatch':
        result['cases']['case-0']['serial']['tensorfold']['drafts'] = True
    elif bad == 'serial-work':
        result['cases']['case-0']['serial']['tensorfold']['drafted'] = 1
    else:
        result['cases']['case-0']['mtp']['tensorfold']['drafted'] = True
    assert result['drafted_total'] > 0
    client.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        owned.accept_history(root, client)


def test_family_forwards_only_nonempty_private_host_obligation(monkeypatch, tmp_path):
    from gguf_fixture import tiny_model
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine as implementation
    made = []
    monkeypatch.setattr(implementation, 'FlashNextEngine', lambda *a, **k: made.append(k) or NS(**k))
    path = tiny_model(tmp_path / 'target.gguf')
    options = dict(no_drafts=True, tp=8, rank=0, kv_dtype='fp16', expert_tensor_parallel_size=4,
                   max_waiting_input_bytes=16384)
    qwen4_exp.cuda_engine(path, **options)
    assert '_owned_qualification_host' not in made[-1]
    obligation = (8000, 'a' * 64)
    qwen4_exp.cuda_engine(path, **options, _owned_expert_native=True, _owned_qualification_host=obligation)
    assert made[-1]['_owned_qualification_host'] == obligation
    with pytest.raises(ValueError, match='requires private owned'):
        qwen4_exp.cuda_engine(path, **options, _owned_qualification_host=obligation)


def test_private_observer_resource_plan_pays_all_ranks_and_future_work(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retention_resources as resources
    monkeypatch.setattr(resources, '_json_payload_policy', lambda: NS(identity='selected-layout-boundary-spy', frame_work=4096))
    small = owned._history_resources(2, 1536)
    large = owned._history_resources(27, 1024 + 256 * 27)
    assert small['global_host_obligation_bytes'] == 8 * small['per_rank_payload_bytes'] > 0
    assert large['global_host_obligation_bytes'] > small['global_host_obligation_bytes']
    assert large['materialized_credit_bytes'] == 0
    assert large['runtime_policy'] == 'selected-layout-boundary-spy'
    assert large['receipt_bytes_limit'] >= 4096 + 1024 * 27


def test_offline_acceptor_remains_torch_free_in_fresh_interpreter(history_acceptance_packet):
    root, client = history_acceptance_packet
    code = '''import sys
from tools import v100_owned_qualification as owned
assert 'torch' not in sys.modules
assert owned.accept_history(sys.argv[1],sys.argv[2])['passed']
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(root), str(client)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('bad', ['rank', 'service'])
def test_offline_acceptor_requires_own_startup_rank_service(history_acceptance_packet, bad):
    root, client = history_acceptance_packet
    path = root / 'rank-0.json'
    startup = json.loads(path.read_text())
    if bad == 'rank':
        startup['rank'] = 1
    else:
        startup['server_instance_id'] = 'other-process'
    path.write_text(json.dumps(startup))
    with pytest.raises(ValueError):
        owned.accept_history(root, client)


def test_private_benchmark_events_exact_cohort_identity_and_full_active_owner_proof(monkeypatch):
    import torch
    from test_bench_concurrent import matrix_fixture
    plan = matrix_fixture()
    plan['cohorts'] = [plan['cohorts'][1]]
    cohort = plan['cohorts'][0]
    clock, queries = [0], []
    actual_stream = object()
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: actual_stream)
    class Event:
        def __init__(self):
            self.at, self.records = None, 0
        def record(self, stream):
            assert stream is actual_stream
            clock[0] += 1
            self.at, self.records = clock[0], self.records + 1
        def query(self):
            queries.append('query')
            return True
        def elapsed_time(self, other):
            return (other.at - self.at) * 10.
    observations = NS(target=lambda *a, **k: None, record=lambda **k: {'existing': True})
    streams = []
    from tools.bench_concurrent import matrix_ids
    for request in cohort['requests']:
        streams.append(NS(prompt=matrix_ids(plan, request), count=3, out=[], done=False, st=object()))
    decoder = NS(streams={}, filling=[], controller=NS(work_observations=observations))
    def admit(stream):
        decoder.filling.append(stream)
    def take(stream, values):
        stream.out.extend(values)
        stream.done = len(stream.out) >= stream.count
    def joined(pieces, *args):
        for stream, start, count in pieces:
            decoder.filling.remove(stream)
            decoder.streams[id(stream)] = stream
            decoder._take(stream, [20])
    def finish(done):
        for stream in done:
            decoder.streams.pop(id(stream))
    decoder.admit, decoder._take, decoder._joined, decoder.finish = admit, take, joined, finish
    decoder._prepare_pass = lambda pieces: 'prepared'
    decoder.round = lambda: []
    engine = NS(rank=0, multi=decoder, w=NS(device='cuda'), scheduler=NS(thread=object()),
                tp_decoder=NS(wires={id(stream): index for index, stream in enumerate(streams)}), _abort_comm=False)
    events = [Event() for _ in range(4)]
    observer = owned._BenchmarkObserver(engine, plan, events=events)
    observer.install()
    smoke = NS(prompt=[99, 98], count=3, out=[], done=False, st=object())
    engine.tp_decoder.wires[id(smoke)] = 99
    decoder.admit(smoke)
    decoder._prepare_pass([(smoke, 0, 2)])
    decoder._joined([(smoke, 0, 2)], None, None, 0.)
    decoder._take(smoke, [21, 22])
    decoder.finish([smoke])
    assert observer.published['cohorts'] == {} and all(event.records == 0 for event in events)
    for stream in streams:
        decoder.admit(stream)
    pieces = [(stream, 0, len(stream.prompt)) for stream in streams]
    assert decoder._prepare_pass(pieces) == 'prepared'
    observations.target([(streams[0].st, 0, 6)], prefill=True)  # Microbatch1 is not active-owner concurrency2.
    decoder._joined(pieces, None, None, 0.)
    for stream in streams:
        decoder._take(stream, [21, 22])
    assert queries == []  # No timer query/fence while model cohort executes.
    decoder.round()
    decoder.finish(streams)
    snapshot = observations.record()['benchmark']['cohorts'][str(cohort['id'])]
    assert snapshot['events_ready'] and snapshot['completed'] == 2
    assert snapshot['active_owner_highwater'] == 2 and snapshot['prefill_batch_highwater'] == 1
    assert snapshot['prompt_hashes'] == sorted(row['prompt_sha256'] for row in cohort['requests'])
    assert snapshot['prefill_ms'] > 0 and snapshot['decode_ms'] > 0
    assert [event.records for event in events] == [1, 1, 1, 1]
    from tensorfold.server.stats import INSTANCE_ID
    assert observer.published['server_instance_id'] == INSTANCE_ID
    published = observer.published
    credited = []
    def credit(value):
        credited.append(value)
        observer.published = {**published, 'later_generation': True}
    assert observations.record(credit=credit)['benchmark'] is published
    assert credited == [published]
    monkeypatch.setattr(owned, 'notify_failure', lambda: True)
    with pytest.raises(ValueError, match='does not belong'):
        decoder.admit(smoke)  # Foreign traffic is forbidden once the matrix started.
    assert engine._abort_comm


def test_benchmark_plan_does_not_add_public_cli_flag():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['serve', 'model', '--benchmark-plan', 'data.json'])


def test_router_input_is_bounded_strict_and_distinct(tmp_path):
    path = tmp_path / 'input.json'
    data = {'schema': 'tensorfold.router-row-input-v1', 'run_identity': 'a' * 32,
            'prompt_token_ids_sha256': ['b' * 64, 'c' * 64]}
    path.write_text(json.dumps(data))
    assert owned._router_input(path)[0] == data
    data['prompt_token_ids_sha256'][1] = 'b' * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='ordered prompt'):
        owned._router_input(path)
    path.write_bytes(b' ' * 4097)
    with pytest.raises(ValueError, match='exceeds4096'):
        owned._router_input(path)


def test_router_tensor_capture_uses_only_direct_cpu_copy_and_npy_view(tmp_path):
    import hashlib
    import numpy as np
    import torch
    host = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    calls = []
    class CUDAView:
        is_cuda, requires_grad, ndim, dtype, shape = True, False, 2, torch.float32, (3, 4)
        def is_contiguous(self):
            return True
        def is_neg(self):
            return False
        def is_conj(self):
            return False
        def detach(self):
            calls.append('detach')
            return self
        def to(self, **kwargs):
            assert kwargs == {'device': 'cpu', 'copy': True}
            calls.append('direct-cpu')
            return host
        def numel(self):
            return 12
        def element_size(self):
            return 4
    target = tmp_path / 'actual.npy'
    record = owned._save_router_tensor(target, CUDAView())
    np.testing.assert_array_equal(np.load(target, allow_pickle=False), host.numpy())
    assert calls == ['detach', 'direct-cpu']
    assert record['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert record['shape'] == [3, 4] and record['bytes'] <= 48 + 256
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        owned._save_router_tensor(target, CUDAView())


def test_router_resources_fund_host_payload_without_device_allocation(tmp_path):
    import torch
    before = torch.cuda.is_initialized()
    resources = owned._router_resources(tmp_path)
    assert resources['additional_device_bytes'] == 0
    assert resources['maximum_capture_rows'] == 4
    assert resources['rank0_tensor_bytes'] == 513 * 2560 * 4 + 4 * (2560 * 2 + 513 * 4)
    assert resources['global_host_obligation_bytes'] >= resources['rank0_tensor_bytes']
    assert torch.cuda.is_initialized() is before


@pytest.mark.parametrize('fault', [None, 'bank', 'segments', 'marker'])
def test_router_capture_selects_c8_and_alone_once_preserving_original_calls_and_states(tmp_path, monkeypatch, fault):
    import hashlib
    data = {'schema': 'tensorfold.router-row-input-v1', 'run_identity': 'a' * 32,
            'prompt_token_ids_sha256': [owned._prompt_digest([2] * 64), owned._prompt_digest([7] * 64)]}
    events, files, marker = [], [], tmp_path / 'router-row-stage.json'
    placement = NS(indexer_owner=lambda slot, layer: (slot + layer) % 8,
                   kv_owner=lambda slot, layer, head, page: (slot + layer) % 4 ^ page)
    streams = [NS(prompt=[index] * 64, count=128, draft=True, sampling=NS(temperature=0, seed=None, top_k=0, top_p=1, min_p=0),
                  st=NS(history_slot=slot, slot_generation=1, history_generation=2, att_index={3: 0}, history_placement=placement))
               for slot, index in enumerate((2, 7))]
    class Matrix:
        def __init__(self, shape, pointer):
            self.shape, self.pointer = shape, pointer
        def data_ptr(self):
            return self.pointer
        def narrow(self, axis, offset, count):
            assert axis == 0 and count == 1
            return NS(shape=(1, self.shape[1]), source=self, offset=offset)
    x, logits, weight = Matrix((128, 2560), 10), Matrix((256, 513), 20), Matrix((513, 2560), 30)
    original_error = RuntimeError('original bank failed')
    def bank(*args, **kwargs):
        events.append('bank')
        if fault == 'bank':
            raise original_error
        return 'original-result'
    decoder = NS(pbuf=NS(rows=256, mixed=x, moe=NS(logits=logits)))
    observations = NS(target=lambda *args, **kwargs: events.append('target'))
    decoder.controller = NS(work_observations=observations)
    decoder._prepare_pass = lambda pieces: events.append('prepare') or 'prepared'
    bank_owner = NS(forward_partitioned_native_candidate=bank)
    engine = NS(rank=0, multi=decoder, w=NS(layers=[NS(moe=NS(experts=bank_owner, router=weight))]),
                tp_decoder=NS(wires={id(s): i for i, s in enumerate(streams)}, prefix_operation=5))
    chosen = [streams[0], streams[1]]
    def rounded():
        pieces = [(s, 0, 64) for s in chosen]
        assert decoder._prepare_pass(pieces) == 'prepared'
        x.shape = (len(chosen) * 64, 2560)
        assert bank_owner.forward_partitioned_native_candidate(x, object()) == 'original-result'
        segments = [(s.st, i * 64, (i + 1) * 64) for i, s in enumerate(chosen)]
        if fault == 'segments':
            segments[0] = (object(), 0, 64)
        observations.target(segments, prefill=True)
        return 'round-result'
    decoder.round = rounded
    def save(path, tensor):
        files.append((Path(path).name, tensor))
        assert 'bank' in events or tensor is weight
        return {'file': Path(path).name, 'sha256': hashlib.sha256(Path(path).name.encode()).hexdigest(),
                'shape': list(tensor.shape)}
    monkeypatch.setattr(owned, '_save_router_tensor', save)
    failures = []
    monkeypatch.setattr(owned, '_observer_failure', lambda engine, error: failures.append(error))
    observer = owned._RouterRows(engine, data, 'd' * 64, {'receipt_bytes_limit': 200000}, tmp_path)
    observer.install()
    def phase(stage):
        marker.write_text(json.dumps({'schema': 'tensorfold.router-row-stage-v1',
                                     'run_identity': 'wrong' if fault == 'marker' else 'a' * 32, 'stage': stage}))
        return decoder.round()
    if fault:
        with pytest.raises(RuntimeError if fault == 'bank' else ValueError) as caught:
            phase('baseline-c8-mtp')
        assert failures[-1] is caught.value
        if fault == 'bank':
            assert caught.value is original_error and len(files) == 1
        assert not (tmp_path / 'router-rank-0.json').exists()
        return
    assert phase('baseline-c8-mtp') == 'round-result'
    assert len(observer.records) == 2 and len(files) == 5
    phase('baseline-c8-mtp-repeat')
    assert len(observer.records) == 2 and len(files) == 5
    chosen[:] = [streams[0]]
    phase('alone-2-mtp')
    chosen[:] = [streams[1]]
    phase('alone-7-mtp')
    assert len(observer.records) == 4 and len(files) == 9
    phase('alone-7-serial')
    assert len(observer.records) == 4 and len(files) == 9
    receipt = json.loads((tmp_path / 'router-rank-0.json').read_text())
    assert receipt['complete'] and receipt['run_identity'] == 'a' * 32
    assert [r['row_offset'] for r in receipt['records']] == [63, 127, 63, 63]
    assert [r['selected']['slot'] for r in receipt['records']] == [0, 1, 0, 1]
    assert events.count('bank') == events.count('prepare') == events.count('target') == 5
    assert receipt['records'][0]['selected']['attention_model_layer'] == 3


@pytest.mark.parametrize('fault', [None, 'obligation', 'baseline', 'capacity', 'context', 'shape'])
def test_router_capture_uses_original_immutable_host_obligation_before_install(monkeypatch, tmp_path, fault):
    resources = {'global_host_obligation_bytes': 1000}
    data, sha = {}, 'a' * 64
    engine = NS(_owned_qualification_host=(1000, sha), context_window=262144,
                multi=NS(states=[object()] * 32),
                retention_catalog=NS(host_resources=NS(baseline_host_bytes=1000)),
                w=NS(layers=[NS(moe=NS(router=NS(shape=(513, 2560))))]))
    if fault == 'obligation':
        engine._owned_qualification_host = (999, sha)
    elif fault == 'baseline':
        engine.retention_catalog.host_resources.baseline_host_bytes = 999
    elif fault == 'capacity':
        engine.multi.states.pop()
    elif fault == 'context':
        engine.context_window = 128
    elif fault == 'shape':
        engine.w.layers[0].moe.router.shape = (512, 2560)
    calls = []
    monkeypatch.setattr(owned, '_RouterRows', lambda *args: NS(install=lambda: calls.append(args)))
    if fault:
        with pytest.raises(ValueError):
            owned.prepare_router_rows(engine, (data, sha, resources), tmp_path)
        assert calls == []
    else:
        assert owned.prepare_router_rows(engine, (data, sha, resources), tmp_path) is resources
        assert calls == [(engine, data, sha, resources, tmp_path)]


def test_router_frame_cardinality_refuses_before_metadata_or_original_prepare():
    observer = object.__new__(owned._RouterRows)
    with pytest.raises(ValueError, match='prepared frame'):
        observer.preparing([object()] * 33)


@pytest.mark.parametrize('phase', ['write-close', 'directory-close'])
def test_router_npy_failure_preserves_authentic_primary_and_close_failure(tmp_path, monkeypatch, phase):
    import numpy as np
    import torch
    import stat
    host = torch.zeros((1, 2), dtype=torch.float16)
    tensor = NS(is_cuda=True, requires_grad=False, ndim=2, dtype=torch.float16, shape=(1, 2),
                is_contiguous=lambda: True, is_neg=lambda: False, is_conj=lambda: False,
                numel=lambda: 2, element_size=lambda: 2)
    tensor.detach = lambda: NS(to=lambda **kwargs: host)
    primary, secondary = OSError('authentic I/O primary'), OSError('close secondary')
    original_cause = RuntimeError('existing source cause')
    primary.__cause__ = original_cause
    primary.add_note('existing diagnostic note')
    original_open, original_fsync, original_close = Path.open, os.fsync, os.close
    if phase == 'write-close':
        class Output:
            def __init__(self, stream):
                self.stream = stream
            def close(self):
                self.stream.close()
                raise secondary
        def opening(path, mode='r', *args, **kwargs):
            result = original_open(path, mode, *args, **kwargs)
            return Output(result) if mode == 'xb' else result
        monkeypatch.setattr(Path, 'open', opening)
        def fail(*args, **kwargs):
            raise primary
        monkeypatch.setattr(np, 'save', fail)
    else:
        directory_fds = []
        def syncing(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directory_fds.append(fd)
                raise primary
            return original_fsync(fd)
        def closing(fd):
            result = original_close(fd)
            if fd in directory_fds:
                raise secondary
            return result
        monkeypatch.setattr(os, 'fsync', syncing)
        monkeypatch.setattr(os, 'close', closing)
    with pytest.raises(OSError) as caught:
        owned._save_router_tensor(tmp_path / 'failed.npy', tensor)
    assert caught.value is primary
    def roots(error):
        if isinstance(error, BaseExceptionGroup):
            return [root for child in error.exceptions for root in roots(child)]
        return [error]
    assert secondary in roots(primary.__cause__)
    assert original_cause in roots(primary.__cause__)
    assert 'existing diagnostic note' in primary.__notes__


@pytest.mark.parametrize('requested,configured', [(None, 'inherit'), ('inherit', 'inherit'),
                                                ('native-serial-v1', 'native-serial-v1'), (None, None)])
@pytest.mark.parametrize('inherited', [False, True])
def test_router_attestation_preserves_process_request_and_actual_inherited_policy(
        channel, monkeypatch, tmp_path, requested, configured, inherited):
    from tensorfold.cuda.sm70 import routing_native
    name = 'TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS'
    if requested is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, requested)
    if configured is None:
        monkeypatch.delattr(routing_native, '_PRIVATE_SERIAL_LOGITS', raising=False)
    else:
        monkeypatch.setattr(routing_native, '_PRIVATE_SERIAL_LOGITS', configured)
    monkeypatch.setattr(routing_native, '_NATIVE_LOGITS', inherited)
    monkeypatch.setattr(owned, 'bound_receipt', lambda _: {})
    calls = []
    def serve(args):
        monkeypatch.setenv(name, 'later-environment-edit')
        engine = NS(rank=3, comm=NS(ready=lambda label: calls.append(label)))
        args._owned_expert_qualification.before_listener(engine)
        return 0
    monkeypatch.setattr(cli, 'cmd_serve', serve)
    assert owned.worker(worker_args(tmp_path)) == 0
    receipt = json.loads((tmp_path / 'rank-3.json').read_text())['private_router_policy']
    assert receipt == {'requested': requested, 'configured': configured or 'legacy-inherited',
                       'private_selector_supported': configured is not None,
                       'inherited_native_logits': inherited}
    assert getattr(routing_native, '_PRIVATE_SERIAL_LOGITS', None) == configured
    assert routing_native._NATIVE_LOGITS is inherited
    assert calls == ['owned_expert_qualification']


@pytest.mark.parametrize('requested,field,bad,absent', [
    ('inherit', '_PRIVATE_SERIAL_LOGITS', None, True),
    ('native-serial-v1', '_PRIVATE_SERIAL_LOGITS', None, True),
    ('native-serial-v1', '_PRIVATE_SERIAL_LOGITS', 'inherit', False),
    ('invalid', '_PRIVATE_SERIAL_LOGITS', 'inherit', False),
    (None, '_PRIVATE_SERIAL_LOGITS', 'native-serial-v1', False),
    (None, '_PRIVATE_SERIAL_LOGITS', None, False),
    (None, '_PRIVATE_SERIAL_LOGITS', 'legacy-inherited', False),
    (None, '_NATIVE_LOGITS', None, False),
    (None, '_NATIVE_LOGITS', 1, False),
    (None, '_NATIVE_LOGITS', 'false', False),
    (None, '_NATIVE_LOGITS', None, True),
])
@pytest.mark.parametrize('stage', ['constructor', 'publication'])
def test_router_attestation_fails_closed_at_both_boundaries_without_mutation(
        channel, monkeypatch, tmp_path, requested, field, bad, absent, stage):
    from tensorfold.cuda.sm70 import routing_native
    name = 'TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS'
    if requested is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, requested)
    initial = requested if requested in ('inherit', 'native-serial-v1') else 'inherit'
    monkeypatch.setattr(routing_native, '_PRIVATE_SERIAL_LOGITS', initial)
    monkeypatch.setattr(routing_native, '_NATIVE_LOGITS', False)
    monkeypatch.setattr(owned, 'bound_receipt', lambda _: {})
    engine = NS(rank=3, _abort_comm=False, comm=NS(ready=lambda _: pytest.fail('router mismatch reached ready')))
    def corrupt():
        if absent:
            monkeypatch.delattr(routing_native, field, raising=False)
        else:
            monkeypatch.setattr(routing_native, field, bad)
    def serve(args):
        assert stage == 'publication', 'corrupt router metadata reached constructor'
        corrupt()
        args._owned_expert_qualification.before_listener(engine)
    monkeypatch.setattr(cli, 'cmd_serve', serve)
    if stage == 'constructor':
        corrupt()
    # An invalid request cannot reach publication: it must stop at first validation.
    with pytest.raises(ValueError, match='private router'):
        owned.worker(worker_args(tmp_path))
    assert os.read(channel[0], 8) == b'F'
    if stage == 'publication' and requested != 'invalid':
        assert engine._abort_comm
    assert not (tmp_path / 'rank-3.json').exists()
    if stage == 'constructor' or requested != 'invalid':
        assert (not hasattr(routing_native, field)) if absent else getattr(routing_native, field) == bad


@pytest.mark.parametrize('requested,configured', [(None, 'inherit'), ('inherit', 'inherit'),
                                                ('cpu-empty-v1', 'cpu-empty-v1'), (None, None)])
def test_empty_prefix_attestation_has_no_native_flag_and_keeps_original_request(
        channel, monkeypatch, tmp_path, requested, configured):
    from tensorfold.families.qwen4_exp.cuda import history_owner
    name = 'TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX'
    if requested is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, requested)
    if configured is None:
        monkeypatch.delattr(history_owner, '_PRIVATE_EMPTY_PREFIX', raising=False)
    else:
        monkeypatch.setattr(history_owner, '_PRIVATE_EMPTY_PREFIX', configured, raising=False)
    monkeypatch.setattr(owned, 'bound_receipt', lambda _: {})
    ready = []
    def serve(args):
        monkeypatch.setenv(name, 'changed-after-construction')
        engine = NS(rank=3, comm=NS(ready=lambda label: ready.append(label)))
        args._owned_expert_qualification.before_listener(engine)
        return 0
    monkeypatch.setattr(cli, 'cmd_serve', serve)
    assert owned.worker(worker_args(tmp_path)) == 0
    receipt = json.loads((tmp_path / 'rank-3.json').read_text())['empty_prefix_policy']
    assert receipt == {'requested': requested, 'configured': configured or 'legacy-inherited',
                       'private_selector_supported': configured is not None}
    assert getattr(history_owner, '_PRIVATE_EMPTY_PREFIX', None) == configured
    assert ready == ['owned_expert_qualification']


@pytest.mark.parametrize('requested,configured,absent', [
    ('inherit', None, True), ('cpu-empty-v1', None, True),
    ('cpu-empty-v1', 'inherit', False), ('invalid', 'inherit', False),
    (None, 'cpu-empty-v1', False), (None, None, False),
    (None, 'legacy-inherited', False), (None, False, False),
])
@pytest.mark.parametrize('stage', ['constructor', 'publication'])
def test_empty_prefix_attestation_refuses_unsupported_or_changed_present_mode(
        channel, monkeypatch, tmp_path, requested, configured, absent, stage):
    from tensorfold.families.qwen4_exp.cuda import history_owner
    name = 'TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX'
    if requested is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, requested)
    initial = requested if requested in ('inherit', 'cpu-empty-v1') else 'inherit'
    monkeypatch.setattr(history_owner, '_PRIVATE_EMPTY_PREFIX', initial, raising=False)
    monkeypatch.setattr(owned, 'bound_receipt', lambda _: {})
    engine = NS(rank=3, _abort_comm=False,
                comm=NS(ready=lambda _: pytest.fail('invalid empty-prefix mode reached ready')))
    def corrupt():
        if absent:
            monkeypatch.delattr(history_owner, '_PRIVATE_EMPTY_PREFIX', raising=False)
        else:
            monkeypatch.setattr(history_owner, '_PRIVATE_EMPTY_PREFIX', configured, raising=False)
    def serve(args):
        assert stage == 'publication', 'invalid empty-prefix mode reached constructor'
        corrupt()
        args._owned_expert_qualification.before_listener(engine)
    monkeypatch.setattr(cli, 'cmd_serve', serve)
    if stage == 'constructor':
        corrupt()
    with pytest.raises(ValueError, match='private empty-prefix'):
        owned.worker(worker_args(tmp_path))
    assert os.read(channel[0], 8) == b'F'
    if stage == 'publication' and requested != 'invalid':
        assert engine._abort_comm
    assert not (tmp_path / 'rank-3.json').exists()
    if stage == 'constructor' or requested != 'invalid':
        assert not hasattr(history_owner, '_PRIVATE_EMPTY_PREFIX') if absent else history_owner._PRIVATE_EMPTY_PREFIX == configured


@pytest.mark.parametrize('rows', [0, -1, True, 1.5, '40', 65536])
def test_private_execution_rows_capability_rejects_invalid_bounds(rows):
    with pytest.raises(TypeError, match='execution rows'):
        cli._OwnedExpertQualification(lambda _: None, execution_rows=rows)


def test_worker_passes_explicit_row_envelope_before_constructor(channel, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, 'cmd_serve', lambda args: seen.append(args) or 0)
    args = ['--execution-rows', '40', *worker_args(tmp_path), '--parallel', '32',
            '--draft-method', 'mtp', '--mtp-drafts', '4']
    assert owned.worker(args) == 0
    assert seen[0]._owned_expert_qualification.execution_rows == 40


@pytest.mark.parametrize('rows', [4, 161])
def test_worker_rejects_rows_outside_complete_window_before_constructor(channel, monkeypatch, tmp_path, rows):
    monkeypatch.setattr(cli, 'cmd_serve', lambda _: pytest.fail('invalid envelope reached constructor'))
    with pytest.raises(ValueError, match='complete-window envelope'):
        owned.worker(['--execution-rows', str(rows), *worker_args(tmp_path), '--parallel', '32',
                      '--draft-method', 'mtp', '--mtp-drafts', '4'])


def test_worker_requires_explicit_draft_geometry_for_row_trial(channel, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, 'cmd_serve', lambda _: pytest.fail('implicit envelope reached constructor'))
    with pytest.raises(ValueError, match='explicit MTP depth'):
        owned.worker(['--execution-rows', '40', *worker_args(tmp_path)])


@pytest.mark.parametrize('field', ['engine', 'target', 'mtp', 'maximum', 'configured'])
def test_execution_row_receipt_requires_real_preparation(field):
    engine = NS(execution_rows=40, multi=NS(buf=NS(rows=40), mbuf=NS(rows=40),
        controller=NS(maximum={'row_budget': 40}, configured={'row_budget': 40})))
    assert owned._execution_rows_receipt(engine, 40) == {
        'requested': 40, 'engine': 40, 'target': 40, 'mtp': 40, 'maximum': 40, 'configured': 40}
    if field == 'engine':
        engine.execution_rows = 32
    elif field in ('target', 'mtp'):
        getattr(engine.multi, 'buf' if field == 'target' else 'mbuf').rows = 32
    else:
        getattr(engine.multi.controller, field)['row_budget'] = 32
    with pytest.raises(ValueError, match='prepared engine/buffers/controller'):
        owned._execution_rows_receipt(engine, 40)
