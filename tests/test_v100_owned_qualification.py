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
    state = NS(history_slot=0, slot_generation=1, pos=0)
    stream = NS(sid=0, done=False, st=state, out=[1, 2])
    engine = NS(scheduler=NS(thread=threading.current_thread()), tp_decoder=NS(wires={id(stream): 17}))
    tracer = owned._SchedulerTrace(engine, tmp_path)
    cohorts = [{'id': index, 'level': level, 'warmup': warmup, 'input_tokens': 64,
                'requests': [{'prompt_sha256': 'b' * 64}]} for index, level, warmup in
               ((0, 1, True), (1, 1, False), (2, 8, False))]
    observer = NS(current=0, plan={'run_identity': 'a' * 32, 'cohorts': cohorts},
        identities={17: (cohorts[1], cohorts[1]['requests'][0])},
        decoder=NS(streams={0: stream}, filling=[stream], fills={0: [None, None, 0]}))
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
    assert captured["options"]["shutdown_rank"] == 7
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


@pytest.mark.parametrize("private,rows,reference,gate_up", [(False, None, False, None), (True, None, False, None),
    (True, 40, False, None), (True, None, True, None), (True, None, False, 2), (True, None, False, 4)])
def test_shared_cuda_factory_gets_private_bool_only_with_capability(monkeypatch, tmp_path, request, private, rows, reference, gate_up):
    import tensorfold.cuda.server as server
    (tmp_path / "config.json").write_text('{"max_position_embeddings":128}')
    calls, options = [], []
    engine = cuda_cli_engine(close=lambda: calls.append("engine-close"))
    family = NS(package=NS(cuda_engine=lambda *a, **k: options.append(k) or engine), title="Flash Next", model_type="qwen4_exp")
    monkeypatch.setattr(server, "App", lambda *a, **k: cuda_cli_app(effective_context_window=128))
    install_cli_server(monkeypatch, request, server, on_serve=lambda _app: calls.append("serve"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    selection = None
    if gate_up is not None:
        from tensorfold.cuda.sm70.gguf_ops import _GroupedGateUp
        selection = _GroupedGateUp(gate_up)
    if private:
        args._owned_expert_qualification = cli._OwnedExpertQualification(
            lambda selected: calls.append("receipt") if selected is engine else pytest.fail("different engine"),
            execution_rows=rows, reference_tiles=reference, grouped_gate_up=selection)
    assert cli._serve_cuda(args, family, tmp_path, 128) == 0
    assert options[0].get("_owned_expert_gate_up") is selection
    if selection is None:
        assert "_owned_expert_gate_up" not in options[0]
    assert options[0].get("_owned_expert_native", False) is private
    if not private:
        assert "_owned_expert_native" not in options[0]
    assert options[0].get("_reference_owner_tiles", False) is reference
    if not reference:
        assert "_reference_owner_tiles" not in options[0]
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
    qwen4_exp.cuda_engine(path, **options, _owned_expert_native=True, _reference_owner_tiles=True)
    assert made[-1]["_reference_owner_tiles"] is True
    from tensorfold.cuda.sm70.gguf_ops import _GroupedGateUp
    selection = _GroupedGateUp(2)
    qwen4_exp.cuda_engine(path, **options, _owned_expert_native=True, _owned_expert_gate_up=selection)
    assert made[-1]["_owned_expert_gate_up"] is selection
    with pytest.raises(TypeError, match="typed owned"):
        qwen4_exp.cuda_engine(path, **options, _owned_expert_gate_up=selection)
    with pytest.raises(TypeError, match="trusted owned"):
        qwen4_exp.cuda_engine(path, **options, _reference_owner_tiles=True)
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



def test_reference_tile_capability_is_private_typed_and_forwarded_before_constructor(channel, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "cmd_serve", lambda args: seen.append(args) or 0)
    assert owned.worker(["--reference-tiles", *worker_args(tmp_path)]) == 0
    assert seen[0]._owned_expert_qualification.reference_tiles is True
    with pytest.raises(TypeError, match="literal boolean"):
        cli._OwnedExpertQualification(lambda _: None, reference_tiles=1)


def test_parent_forwards_reference_tile_preparation_to_all_workers(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(launcher, "supervise", lambda commands, environments, **kwargs: seen.extend(commands) or 0)
    assert owned.main(["--reference-tiles", "--gpus", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "new"),
                       "--", "model"]) == 0
    assert len(seen) == 8 and all("--reference-tiles" in command for command in seen)


def test_reference_receipt_requires_all_actual_journalled_roles_and_fresh_source(monkeypatch):
    from tensorfold.cuda.sm70 import reference_owner_tiles as module
    from tensorfold.cuda.sm70.reference_owner_tiles import ReferenceTiles
    identity = {"policy": "reference-owner-tile128-preparation-v1", "source_sha256": "b" * 64}
    monkeypatch.setattr(module, "preparation_identity", lambda: identity)
    owners = {}
    def verify(self, workspace, expected):
        assert self.workspace is workspace and expected == identity
        return {"additional_device_bytes": 0, "borrowed_device_bytes": 67072, "preparation": expected}
    monkeypatch.setattr(ReferenceTiles, "verify", verify)
    for role in ("decode", "mtp", "prefill"):
        tiles = object.__new__(ReferenceTiles)
        tiles.workspace = object()
        owners[role] = NS(reference_tiles=tiles, partials=tiles.workspace, rows=32)
    decoder = NS(**{key: NS(history_owner=owners[role], rows=32)
                    for key, role in (("buf", "decode"), ("mbuf", "mtp"), ("pbuf", "prefill"))})
    engine = NS(_reference_owner_tiles_requested=True, _owner_preparation=NS(reference_preparation=identity, workspaces=owners),
                w=NS(meta={"reference_tile_preparation": identity}, mtp=object()), multi=decoder)
    receipt = owned._reference_tile_receipt(engine, True)
    assert set(receipt["roles"]) == {"decode", "mtp", "prefill"}
    decoder.mbuf.history_owner = owners["decode"]
    with pytest.raises(ValueError, match="journalled"):
        owned._reference_tile_receipt(engine, True)
    decoder.mbuf.history_owner = owners["mtp"]
    engine._owner_preparation.reference_preparation = {"source_sha256": "changed"}
    with pytest.raises(ValueError, match="source identity"):
        owned._reference_tile_receipt(engine, True)


def test_reference_tile_request_unexpected_engine_state_refuses_publication(channel, monkeypatch, tmp_path):
    monkeypatch.setattr(owned, "bound_receipt", lambda _: {})
    engine = NS(rank=0, _abort_comm=False, _reference_owner_tiles_requested=False,
                comm=NS(ready=lambda _: pytest.fail("mismatch reached ready")))
    with pytest.raises(ValueError, match="actual Engine"):
        owned.publish_receipt(engine, tmp_path, reference_tiles=True)
    assert engine._abort_comm and os.read(channel[0], 8) == b"F"
    assert not (tmp_path / "rank-0.json").exists()



def test_enabled_reference_worker_carries_request_through_actual_publication_boundary(channel, monkeypatch, tmp_path):
    seen = []
    engine = NS(rank=3, comm=NS(ready=lambda label: seen.append(label)))
    monkeypatch.setattr(owned, "bound_receipt", lambda _: {})
    def reference_receipt(actual, requested):
        assert actual is engine and requested is True
        return {"requested": True, "configured": True, "roles": {"decode": {}, "mtp": {}, "prefill": {}}}
    monkeypatch.setattr(owned, "_reference_tile_receipt", reference_receipt)
    def serve(args):
        assert args._owned_expert_qualification.reference_tiles is True
        args._owned_expert_qualification.before_listener(engine)
        return 0
    monkeypatch.setattr(cli, "cmd_serve", serve)
    assert owned.worker(["--reference-tiles", *worker_args(tmp_path)]) == 0
    receipt = json.loads((tmp_path / "rank-3.json").read_text())
    assert receipt["reference_tile_preparation"]["requested"] is True
    assert seen == ["owned_expert_qualification"]


@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_scheduler_trace_wrap_preserves_arguments_return_and_authentic_error(monkeypatch, tmp_path, active, fails):
    import torch
    calls, labels = [], []
    result, argument, primary = object(), object(), RuntimeError('actual operation')
    primary.add_note('original operation note')
    class Record:
        def __init__(self, label):
            labels.append(('created', label))
        def __enter__(self):
            labels.append(('entered', None))
        def __exit__(self, kind, value, tb):
            labels.append(('exited', value))
    monkeypatch.setattr(torch.profiler, 'record_function', Record)
    def operation(*args, **kwargs):
        calls.append((args, kwargs))
        if fails:
            raise primary
        return result
    owner = NS(operation=operation)
    tracer = owned._SchedulerTrace(NS(), tmp_path)
    tracer.wrap(owner, 'operation', 'diagnostic.boundary')
    tracer.active = active
    if fails:
        with pytest.raises(RuntimeError) as caught:
            owner.operation(argument, selected=argument)
        assert caught.value is primary and primary.__notes__ == ['original operation note']
    else:
        assert owner.operation(argument, selected=argument) is result
    assert calls == [((argument,), {'selected': argument})]
    assert labels == ([('created', 'diagnostic.boundary'), ('entered', None),
                       ('exited', primary if fails else None)] if active else [])


def test_scheduler_trace_labels_actual_attention_work_and_dynamic_half_glue_bindings(monkeypatch, tmp_path):
    import torch
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda import history_owner, glue
    before_cuda = torch.cuda.is_initialized()
    names = [(history_owner, name) for name in ('partial_statistics', 'merge_statistics', 'normalized_partial')]
    names += [(family_ops, name) for name in ('hc_writeback', 'hc_normed', 'hc_act', 'hc_mix')]
    # install() uses ordinary setattr; register every original with pytest so
    # teardown restores all seven module slots, avoiding cumulative wrappers.
    originals = {(owner, name): getattr(owner, name) for owner, name in names}
    for (owner, name), original in originals.items():
        monkeypatch.setattr(owner, name, original)
    stack, labels, multiplications = [], [], []
    class Record:
        def __init__(self, label):
            self.label = label
        def __enter__(self):
            labels.append(self.label)
            stack.append(self.label)
        def __exit__(self, kind, value, tb):
            assert stack.pop() == self.label
    monkeypatch.setattr(torch.profiler, 'record_function', Record)
    matmul = family_ops._matmul32
    def actual_matmul(*args, **kwargs):
        multiplications.append(stack[-1] if stack else None)
        return matmul(*args, **kwargs)
    monkeypatch.setattr(family_ops, '_matmul32', actual_matmul)
    adapter = NS(_invoke=lambda: None, _execute=lambda: None, round=lambda: None)
    decoder = NS(_prepare_pass=lambda: None, _read_ahead=lambda: None)
    engine = NS(tp_decoder=adapter, w=NS(meta={'input_preparation_context': NS(prepare=lambda: None)}))
    tracer = owned._SchedulerTrace(engine, tmp_path)
    tracer.install(NS(decoder=decoder, current=None))
    assert glue._half_ops is family_ops
    for name in ('partial_statistics', 'merge_statistics', 'normalized_partial'):
        assert history_owner._attention_layer.__globals__[name] is getattr(history_owner, name)
        assert getattr(history_owner, name) is not originals[history_owner, name]
    cache = NS(capacity=2, k=torch.arange(8).half().reshape(2, 1, 4), v=torch.ones(2, 1, 4).half(),
               key_format=NS(quantized=False), value_format=NS(quantized=False),
               local_indices=lambda positions: (positions, torch.ones_like(positions, dtype=torch.bool)))
    query, ids, visible = torch.ones(2, 4).half(), torch.arange(2), torch.ones(2, dtype=torch.bool)
    def work():
        # Resolve exactly as _attention_layer does, not a saved original alias.
        scope = history_owner._attention_layer.__globals__
        maximum, denominator = scope['partial_statistics'](query, cache, ids, visible, .5, tile_preparer=None)
        maximum, denominator = scope['merge_statistics'](maximum[None], denominator[None])
        value = scope['normalized_partial'](query, cache, ids, visible, .5, maximum, denominator, tile_preparer=None)
        h, pss = torch.ones(1, 128).half(), torch.empty(1, 1, 4)
        normed, xs = torch.empty_like(h), torch.empty(1, 4)
        glue.hc_writeback(h, h, pss, 4, 0, block=32)
        glue.hc_normed(h, pss, torch.ones(128).half(), normed, xs, 4, 1e-6)
        act, inj = torch.empty(1, 32).half(), torch.empty(1, 4).half()
        glue.hc_act(torch.ones(1, 36).half(), act, torch.empty(1, 1), inj, 4, 32)
        mixed = torch.empty(1, 32).half()
        glue.hc_mix(torch.zeros(1, 128).half(), normed, mixed, torch.empty(1, 1), 4)
        return value, h, normed, act, inj, mixed
    expected = work()
    assert labels == [] and multiplications == [None, None, None]
    multiplications.clear()
    tracer.active = True
    actual = work()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert labels == ['attention.partial_statistics', 'attention.global_statistics', 'attention.normalized_partial',
                      'HC.writeback', 'HC.normed', 'HC.act', 'HC.mix']
    assert multiplications == ['attention.partial_statistics', 'attention.normalized_partial', 'attention.normalized_partial']
    assert not stack and torch.cuda.is_initialized() is before_cuda


@pytest.mark.parametrize('launch_exit', [0, 7])
def test_teacher_parent_only_seals_after_existing_positive_launch(tmp_path, monkeypatch, launch_exit):
    from types import SimpleNamespace as NS
    events, output = [], tmp_path / 'fresh-teacher'
    prepared = {'data': 'pure data marker'}
    def load(path):
        assert not output.exists()
        events.append(('validated', path))
        return prepared
    def seal(directory, selected, result):
        assert selected is prepared and result == 0
        events.append(('sealed', directory))
    monkeypatch.setattr(owned, '_teacher_module', lambda: NS(owned_teacher_plan=load, seal_owned_teacher=seal))
    def launch(argv, **kwargs):
        assert output.is_dir()
        prefix = kwargs['_command_prefix']
        index = prefix.index('--teacher-plan')
        assert prefix[index + 1] == str((tmp_path / 'plan.json').absolute())
        events.append(('launched', argv))
        return launch_exit
    monkeypatch.setattr(launcher, 'main', launch)
    result = owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(output),
                         '--teacher-plan', str(tmp_path / 'plan.json'), '--', 'model'])
    assert result == launch_exit
    assert [event[0] for event in events] == (['validated', 'launched', 'sealed'] if launch_exit == 0 else ['validated', 'launched'])


def test_teacher_obligation_refuses_before_observer_setup(tmp_path, monkeypatch):
    engine = NS(_owned_qualification_host=(99, 'a' * 64))
    prepared = {'resources': {'global_host_obligation_bytes': 100}, 'input_sha256': 'a' * 64}
    def forbidden():
        raise AssertionError('observer import/setup before admitted obligation')
    monkeypatch.setattr(owned, '_teacher_module', forbidden)
    with pytest.raises(ValueError, match='admitted baseline'):
        owned.prepare_owned_teacher(engine, prepared, tmp_path)
    assert not hasattr(engine, '_owned_teacher_frames')


def _one_teacher_client(tmp_path, monkeypatch, mutate=None):
    import copy
    import hashlib
    from tests.test_v100_adaptive_qualification import stats
    from tools import v100_rotorquant_live as live
    expected = {'context_tokens': 262144, 'resident_capacity': 24, 'configured_ceiling': 24,
                'kv_dtype': 'int8', 'kernel_family': 'torch-reference', 'graph_family': 'uncaptured',
                'tp': 8, 'pp': 1, 'etp': 4, 'ep': 2}
    teacher = {'data': {'input_ids': [1, 2], 'max_frames': 2, 'vocab': 16, 'mode': 'capture'}, 'input_sha256': 'a' * 64}
    selected = {'data': {'expected_service': expected, 'base': 'http://127.0.0.1:8000', 'model': 'target-revision',
                         'socket_timeout_s': 1, 'whole_timeout_s': 2, 'readiness_timeout_s': 1,
                         'response_bytes': 1 << 20}, 'sha256': 'b' * 64}
    value = stats(24)
    value['capacity_plan']['session_capacity']['configured_ceiling'] = 24
    value['capacity_plan']['expert_parallel'] = {key: expected[key] for key in ('tp', 'pp', 'etp', 'ep')}
    value['runtime'] = {'execution': {'execution_families': {'effective': {
        'kernel_family': 'torch-reference', 'graph_family': 'uncaptured'}, 'trial': None}}}
    reply = {'choices': [{'index': 0, 'text': 'plain completion', 'finish_reason': 'length'}],
             'usage': {'prompt_tokens': 2, 'completion_tokens': 2, 'total_tokens': 4,
                       'prompt_tokens_details': {'cached_tokens': 0}},
             'tensorfold': {'token_ids': [3, 4], 'token_sha': hashlib.sha256(b'3,4').hexdigest()[:12],
                            'drafts': False, 'drafted': 0, 'accepted': 0, 'cached': 0}}
    for rank in range(8):
        (tmp_path / f'rank-{rank}.json').write_text(json.dumps({
            'rank': rank, 'mtp_banks': 0, 'state_count': 24,
            'teacher_server_instance_id': 'same-process' if rank == 0 else f'follower-{rank}',
            'teacher_windows': {'full_resident': True, 'slots': 24, 'capacity_min': 262144,
                                'capacity_max': 262144, 'limit_min': 262144, 'limit_max': 262144,
                                'logical_min': 262144, 'logical_max': 262144}}))
    calls = []
    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def call(self, label, body=None, **kwargs):
            kwargs['read_control'].check()
            calls.append((label, body, kwargs['path']))
            result = copy.deepcopy(reply if body is not None else value)
            if mutate is not None:
                mutate(label, result)
            return result
    monkeypatch.setattr(live, 'Client', Client)
    return owned._TeacherClient(teacher, selected, tmp_path), calls


def test_teacher_client_one_plain_cold_post_with_actual_all_rank_binding(tmp_path, monkeypatch):
    client, calls = _one_teacher_client(tmp_path, monkeypatch)
    client.run()
    assert [row[2] for row in calls] == ['/stats', '/v1/completions', '/stats']
    assert calls[1][1] == {'model': 'target-revision', 'prompt': [1, 2], 'max_tokens': 2,
                           'temperature': 0, 'draft': False, 'stream': False, 'return_token_ids': True}
    assert client.result['complete'] is True and client.result['post_attempts'] == 1
    assert client.result['token_ids'] == [3, 4] and client.result['server_instance_id'] == 'same-process'
    assert client.result['quality_acceptance'] is False and client.result['P3_valid'] is False
    assert json.loads((tmp_path / 'teacher-client-report.json').read_text()) == client.result


@pytest.mark.parametrize('fault', ['hash', 'cached', 'cached-bool', 'drafts', 'finish', 'text', 'usage', 'instance', 'family', 'physical'])
def test_teacher_client_rejects_protocol_or_actual_profile_without_retry(tmp_path, monkeypatch, fault):
    def mutate(label, value):
        if label == 'teacher-completion':
            if fault == 'hash':
                value['tensorfold']['token_sha'] = '0' * 12
            if fault == 'cached':
                value['tensorfold']['cached'] = 1
            if fault == 'cached-bool':
                value['usage']['prompt_tokens_details']['cached_tokens'] = False
            if fault == 'drafts':
                value['tensorfold']['drafts'] = True
            if fault == 'finish':
                value['choices'][0]['finish_reason'] = None
            if fault == 'text':
                value['choices'][0].pop('text')
            if fault == 'usage':
                value['usage']['total_tokens'] = 3
        if label == 'readiness-0':
            if fault == 'instance':
                value['server']['instance_id'] = 'stale'
            if fault == 'family':
                value['runtime']['execution']['execution_families']['effective']['kernel_family'] = 'native'
            if fault == 'physical':
                value['capacity_plan']['kv_full_windows']['ranks'][7]['startup_capacities'][0] = 128
    client, calls = _one_teacher_client(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError):
        client.run()
    assert sum(body is not None for _, body, _ in calls) <= 1
    assert client.result is None


def test_teacher_client_requires_all_rank_residency_before_http(tmp_path, monkeypatch):
    client, calls = _one_teacher_client(tmp_path, monkeypatch)
    path = tmp_path / 'rank-7.json'
    value = json.loads(path.read_text())
    value['teacher_windows']['full_resident'] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='full-window'):
        client.run()
    assert calls == []


def test_teacher_client_cancelled_before_http_does_not_post(tmp_path, monkeypatch):
    from tools.v100_rotorquant_live import ReadCancelled
    client, calls = _one_teacher_client(tmp_path, monkeypatch)
    client.cancelled.set()
    with pytest.raises(ReadCancelled):
        client.run()
    assert calls == []


@pytest.mark.parametrize('slot', ['_shutdown_probe', '_failure_probe'])
def test_private_launcher_probes_validate_before_pipe(monkeypatch, slot):
    monkeypatch.setattr(launcher.os, 'pipe2', lambda *_: pytest.fail('acquisition before validation'))
    with pytest.raises(ValueError, match='callable'):
        launcher.main(['--gpus', '0,1', '--', 'model'], **{slot: False})


def test_private_launcher_probes_preserve_stop_and_failure_semantics(monkeypatch):
    seen = []
    def supervise(commands, environments, **kwargs):
        assert len(commands) == 2 and kwargs['shutdown_rank'] == 1
        assert kwargs['failure_probe']() is False
        assert kwargs['shutdown_probe']() is True
        return 0
    monkeypatch.setattr(launcher, 'supervise', supervise)
    assert launcher.main(['--gpus', '0,1', '--', 'model'],
                         _shutdown_probe=lambda requested: seen.append(requested) or True,
                         _failure_probe=lambda: False) == 0
    assert seen == [False]


@pytest.mark.parametrize('slot', ['_shutdown_probe', '_failure_probe'])
def test_private_launcher_probe_errors_remain_authentic(monkeypatch, slot):
    primary = RuntimeError('private callback')
    def callback(*args):
        raise primary
    def supervise(commands, environments, **kwargs):
        return kwargs['shutdown_probe' if slot == '_shutdown_probe' else 'failure_probe']()
    monkeypatch.setattr(launcher, 'supervise', supervise)
    with pytest.raises(RuntimeError) as caught:
        launcher.main(['--gpus', '0,1', '--', 'model'], **{slot: callback})
    assert caught.value is primary


def test_teacher_native_observation_is_after_resource_fence_and_no_retry(monkeypatch, tmp_path):
    events = []
    engine = NS(rank=3, _owned_teacher_frames=NS(finished=True, failed=None),
                _close_resources=lambda: events.append('positive-resource-fence') or 'unchanged')
    def record(required, **kwargs):
        events.append('observation')
        assert required == ['tf_gguf_'] and kwargs['rank'] == 3
        assert kwargs['limits'] == owned._model_contract().MODEL_OBSERVER_LIMITS
    selected = {'module': NS(record_native_elfs=record), 'directory': tmp_path}
    owned._attach_teacher_native_observer(engine, selected)
    assert engine._close_resources() == 'unchanged'
    assert events == ['positive-resource-fence', 'observation']
    with pytest.raises(RuntimeError, match='no retry'):
        engine._close_resources()
    assert events == ['positive-resource-fence', 'observation']


def test_teacher_native_observer_failure_notifies_and_preserves_primary(monkeypatch, tmp_path):
    primary, events = RuntimeError('origin'), []
    def fail(*args, **kwargs):
        raise primary
    engine = NS(rank=0, _owned_teacher_frames=NS(finished=True, failed=None), _close_resources=lambda: None)
    monkeypatch.setattr(owned, '_observer_failure', lambda current, error: events.append((current, error)))
    owned._attach_teacher_native_observer(engine, {'module': NS(record_native_elfs=fail), 'directory': tmp_path})
    with pytest.raises(RuntimeError) as caught:
        engine._close_resources()
    assert caught.value is primary and events == [(engine, primary)]


def _teacher_parent_setup(tmp_path, monkeypatch, run):
    import threading
    output = tmp_path / 'parent'
    teacher = {'input_sha256': 'a' * 64, 'data': {'mode': 'capture'}}
    selected = {'bytes': 10, 'sha256': 'b' * 64, 'data': {'expected_service': {
        'kernel_family': 'torch-reference', 'graph_family': 'uncaptured'}}}
    events = []
    module = NS(owned_teacher_plan=lambda path: teacher,
                seal_owned_teacher=lambda *args: events.append('sealed') or {},
                _owned_teacher_read=lambda *args: b'{}',
                _owned_teacher_object=lambda *args: {'teacher_ids': [3], 'frames': [
                    {'execution_family': selected['data']['expected_service']}]})
    monkeypatch.setattr(owned, '_teacher_module', lambda: module)
    monkeypatch.setattr(owned, '_teacher_client_input', lambda *args: selected)
    monkeypatch.setattr(owned, '_teacher_model_observer', lambda *args: {})
    monkeypatch.setattr(owned, '_teacher_operation_credit', lambda *args: None)
    monkeypatch.setattr(owned, '_teacher_model_result', lambda *args: events.append('result'))
    class Client:
        def __init__(self, *args):
            self.cancelled = threading.Event()
            self.entered = threading.Event()
            self.result = None
            self.post_attempts = 0
            events.append(self)
        def run(self):
            self.entered.set()
            run(self)
    monkeypatch.setattr(owned, '_TeacherClient', Client)
    argv = ['--gpus', '0,1,2,3,4,5,6,7', '--out', str(output), '--teacher-plan', 'plan',
            '--teacher-client', 'client', '--', 'model']
    return argv, events


def test_teacher_parent_client_completion_normally_stops_and_seals(tmp_path, monkeypatch):
    def run(client):
        client.result = {'complete': True, 'token_ids': [3]}
    argv, events = _teacher_parent_setup(tmp_path, monkeypatch, run)
    def launch(*args, **kwargs):
        client = events[0]
        assert client.entered.wait(2)
        # The actual Task's completion journal, not thread liveness, is tested.
        assert kwargs['_failure_probe']() is False
        from time import monotonic
        deadline = monotonic() + 2
        while not kwargs['_shutdown_probe'](False):
            assert monotonic() < deadline
        return 0
    monkeypatch.setattr(launcher, 'main', launch)
    assert owned.main(argv) == 0
    assert events[1:] == ['sealed', 'result']


@pytest.mark.parametrize('late_genuine', [False, True])
def test_teacher_parent_nonzero_rank_owns_primary_before_client_cancel(tmp_path, monkeypatch, late_genuine):
    from tools.v100_rotorquant_live import ReadCancelled
    late = RuntimeError('late genuine protocol failure') if late_genuine else ReadCancelled('cancelled readiness')
    retired = []
    def run(client):
        assert client.cancelled.wait(2)
        retired.append(True)
        raise late
    argv, events = _teacher_parent_setup(tmp_path, monkeypatch, run)
    def launch(*args, **kwargs):
        assert events[0].entered.wait(2)
        return 7
    monkeypatch.setattr(launcher, 'main', launch)
    with pytest.raises(RuntimeError, match='launcher status 7') as caught:
        owned.main(argv)
    assert retired == [True] and len(events) == 1
    def contains(error, target):
        if error is target:
            return True
        return (any(contains(item, target) for item in getattr(error, 'exceptions', ()))
                or (error.__cause__ is not None and contains(error.__cause__, target)))
    assert contains(caught.value, late)
    failure = json.loads((tmp_path / 'parent/teacher-client-failure.json').read_text())
    assert failure['complete'] is False


def test_teacher_parent_authentic_client_failure_survives_launch_secondary(tmp_path, monkeypatch):
    import threading
    primary, secondary = RuntimeError('protocol'), RuntimeError('launch cleanup')
    finished = threading.Event()
    def run(client):
        finished.set()
        raise primary
    argv, events = _teacher_parent_setup(tmp_path, monkeypatch, run)
    def launch(*args, **kwargs):
        assert finished.wait(2)
        # Task records its authentic error after callback exit; wait through probe.
        from time import monotonic
        deadline = monotonic() + 2
        while True:
            try:
                kwargs['_failure_probe']()
            except RuntimeError as error:
                assert error is primary
                raise secondary from error
            assert monotonic() < deadline
    monkeypatch.setattr(launcher, 'main', launch)
    with pytest.raises(RuntimeError) as caught:
        owned.main(argv)
    assert caught.value is primary and len(events) == 1


def test_teacher_parent_zero_launch_without_client_completion_cancels_drains_no_seal(tmp_path, monkeypatch):
    from tools.v100_rotorquant_live import ReadCancelled
    retired = []
    def run(client):
        assert client.cancelled.wait(2)
        retired.append(True)
        raise ReadCancelled('ranks exited before request complete')
    argv, events = _teacher_parent_setup(tmp_path, monkeypatch, run)
    monkeypatch.setattr(launcher, 'main', lambda *args, **kwargs: 0)
    with pytest.raises(ReadCancelled):
        owned.main(argv)
    assert retired == [True] and len(events) == 1


@pytest.mark.parametrize('slot', ['_shutdown_probe', '_failure_probe'])
def test_private_launcher_probe_results_are_strict_booleans(monkeypatch, slot):
    def supervise(commands, environments, **kwargs):
        return kwargs['shutdown_probe' if slot == '_shutdown_probe' else 'failure_probe']()
    monkeypatch.setattr(launcher, 'supervise', supervise)
    with pytest.raises(ValueError, match='boolean'):
        launcher.main(['--gpus', '0,1', '--', 'model'], **{slot: lambda *args: 1})


def test_teacher_native_observer_never_certifies_failed_resource_retirement(monkeypatch, tmp_path):
    primary, events = RuntimeError('resource fence'), []
    def fail():
        raise primary
    engine = NS(rank=0, _owned_teacher_frames=NS(finished=True, failed=None), _close_resources=fail)
    selected = {'module': NS(record_native_elfs=lambda *args, **kwargs: events.append('unexpected')), 'directory': tmp_path}
    owned._attach_teacher_native_observer(engine, selected)
    with pytest.raises(RuntimeError) as caught:
        engine._close_resources()
    assert caught.value is primary and events == []


def test_teacher_client_preserves_authentic_transport_error_without_second_post(tmp_path, monkeypatch):
    primary = OSError('socket read failed')
    def mutate(label, result):
        if label == 'teacher-completion':
            raise primary
    client, calls = _one_teacher_client(tmp_path, monkeypatch, mutate)
    with pytest.raises(OSError) as caught:
        client.run()
    assert caught.value is primary
    assert sum(body is not None for _, body, _ in calls) == 1 and client.result is None


def test_teacher_parent_task_start_failure_prevents_cohort_acquisition(tmp_path, monkeypatch):
    from tools.worker_lifetime import Task
    primary = RuntimeError('thread startup')
    argv, events = _teacher_parent_setup(tmp_path, monkeypatch, lambda client: pytest.fail('callback entered'))
    def start(task):
        raise primary
    monkeypatch.setattr(Task, 'start', start)
    monkeypatch.setattr(launcher, 'main', lambda *args, **kwargs: pytest.fail('launch before client task started'))
    with pytest.raises(RuntimeError) as caught:
        owned.main(argv)
    assert caught.value is primary and len(events) == 1


def test_teacher_readiness_exhausts_prepaid_attempts_even_with_early_waits(tmp_path, monkeypatch):
    import urllib.error
    from tools import v100_rotorquant_live as live
    client, _ = _one_teacher_client(tmp_path, monkeypatch)
    calls = []
    class Refused:
        def __init__(self, *args, **kwargs):
            pass
        def call(self, label, body=None, **kwargs):
            calls.append(label)
            assert body is None and kwargs['path'] == '/stats'
            raise urllib.error.URLError(ConnectionRefusedError('not ready'))
    monkeypatch.setattr(live, 'Client', Refused)
    monkeypatch.setattr(owned, '_TEACHER_READINESS_ATTEMPTS', 2)
    monkeypatch.setattr(client.cancelled, 'wait', lambda seconds: False)
    with pytest.raises(TimeoutError, match='prepaid attempt count'):
        client.run()
    assert calls == ['readiness-0', 'readiness-1'] and client.post_attempts == 0


@pytest.mark.parametrize('secondary', [RuntimeError('notify'), MemoryError('notification aggregation')])
def test_teacher_native_observer_preserves_primary_when_failure_transport_fails(monkeypatch, tmp_path, secondary):
    primary = RuntimeError('native origin')
    def fail(*args, **kwargs):
        raise primary
    def notify(*args):
        raise secondary
    engine = NS(rank=0, _owned_teacher_frames=NS(finished=True, failed=None), _close_resources=lambda: None)
    monkeypatch.setattr(owned, '_observer_failure', notify)
    owned._attach_teacher_native_observer(engine, {'module': NS(record_native_elfs=fail), 'directory': tmp_path})
    with pytest.raises(RuntimeError) as caught:
        engine._close_resources()
    assert caught.value is primary and primary.__cause__ is secondary


def _late_trace_fixture(tmp_path, *, start=128, cursor=0, factory=None):
    import threading
    cohorts = [{'id': index, 'level': level, 'warmup': warmup, 'input_tokens': 1024,
                'requests': [{'prompt_sha256': str(index) * 64}]} for index, level, warmup in
               ((0, 1, True), (1, 1, False), (2, 8, False), (3, 1, False))]
    plan = {'run_identity': 'a' * 32, 'cohorts': cohorts}
    state = NS(history_slot=4, slot_generation=7, pos=cursor)
    stream = NS(sid=19, done=False, st=state, out=[])
    adapter = NS(wires={id(stream): 23}, prefix_operation=51)
    engine = NS(rank=0, scheduler=NS(thread=threading.current_thread()), tp_decoder=adapter)
    decoder = NS(streams={}, filling=[stream], fills={19: [None, None, cursor]})
    observer = NS(current=1, plan=plan, decoder=decoder, identities={23: (cohorts[1], cohorts[1]['requests'][0])})
    tracer = owned._SchedulerTrace(engine, tmp_path, factory=factory, prefill_start=start,
                                  selection=owned._trace_prefill_selection(plan, start))
    return tracer, observer, stream


@pytest.mark.parametrize('value', [-1, True, False, None, 1.0, sys.maxsize + 1])
def test_late_trace_selector_rejects_invalid_domain_before_effects(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        owned._trace_prefill_selection({'cohorts': []}, value)


def test_late_trace_selection_is_bound_to_first_nonwarmup_c1_and_prompt_end(tmp_path):
    tracer, observer, stream = _late_trace_fixture(tmp_path, start=128, cursor=0)
    assert tracer.label(observer) is None
    observer.decoder.fills[stream.sid][2] = 160
    stream.st.pos = 160
    observer.current = 0
    assert tracer.label(observer) is None  # Warmup cannot consume the selected window.
    observer.current = 3
    assert tracer.label(observer) is None  # A later C1 repetition cannot substitute.
    observer.current = 1
    assert tracer.label(observer) == 'c1-prefill-start-ge-128-round'
    assert tracer.selection['boundary_start'] == 160
    assert tracer.selection['prompt_sha256'] == '1' * 64
    assert tracer.label(observer) is None
    assert tracer.counts['prefill'] == 1
    with pytest.raises(ValueError, match='prompt end'):
        owned._trace_prefill_selection(observer.plan, 1024)
    no_c1 = {'cohorts': [{'id': 2, 'level': 8, 'warmup': False}]}
    assert owned._trace_prefill_selection(no_c1, 0) is None
    with pytest.raises(ValueError, match='nonwarmup C1'):
        owned._trace_prefill_selection(no_c1, 1)


@pytest.mark.parametrize('fault', ['state', 'wire', 'foreign', 'cursor', 'owner-count', 'finished'])
def test_late_trace_does_not_profile_missing_or_unqualified_owner(tmp_path, fault):
    tracer, observer, stream = _late_trace_fixture(tmp_path, start=128, cursor=160)
    if fault == 'state':
        stream.st = None
    elif fault == 'wire':
        tracer.engine.tp_decoder.wires.clear()
    elif fault == 'foreign':
        observer.identities[23] = (observer.plan['cohorts'][3], {'prompt_sha256': '3' * 64})
    elif fault == 'cursor':
        observer.decoder.fills[stream.sid][2] = 1024
    elif fault == 'owner-count':
        observer.decoder.filling.append(NS(done=False, st=object()))
    else:
        stream.done = True
    assert tracer.label(observer) is None and not tracer.counts
    assert tracer.selection['status'] == 'UNSUPPORTED_NOT_OBSERVED'


@pytest.mark.parametrize('actual', ['positive', 'empty', 'wrong-cursor', 'wrong-owner', 'multiple', 'cancelled'])
def test_late_trace_receipt_distinguishes_requested_boundary_from_actual_prefill(tmp_path, actual):
    class Profiler:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        def export_chrome_trace(self, path):
            Path(path).write_text('{}')
    tracer, observer, stream = _late_trace_fixture(tmp_path, cursor=160, factory=Profiler)
    label = tracer.label(observer)
    result, calls = object(), []
    def round_():
        calls.append(True)
        if actual == 'positive':
            tracer.note_prepare([(stream, 160, 64)])
            tracer.selection['prepared_returned'] = True
        elif actual == 'wrong-cursor':
            tracer.note_prepare([(stream, 128, 64)])
            tracer.selection['prepared_returned'] = True
        elif actual == 'wrong-owner':
            tracer.note_prepare([(NS(), 160, 64)])
            tracer.selection['prepared_returned'] = True
        elif actual == 'multiple':
            tracer.note_prepare([(stream, 160, 64)])
            tracer.note_prepare([(stream, 160, 64)])
            tracer.selection['prepared_returned'] = True
        elif actual == 'cancelled':
            stream.done = True
        return result
    assert tracer.capture(label, round_) is result and calls == [True]
    receipt = json.loads((tmp_path / (label + '-receipt.json')).read_text())
    selection = receipt['prefill_selection']
    assert selection['requested_start'] == 128 and selection['boundary_start'] == 160
    assert selection['slot'] == 4 and selection['slot_generation'] == 7 and selection['wire'] == 23
    assert selection['prefix_operation'] == 51 and receipt['P3_valid'] is False
    if actual == 'positive':
        assert selection['status'] == 'OBSERVED'
        assert selection['actual_start'] == 160 and selection['actual_count'] == 64
    else:
        assert selection['status'] == 'UNSUPPORTED_NOT_OBSERVED'
    assert tracer.profiler is tracer.prefill_owner is None


def test_trace_failure_with_false_exception_and_failed_rendering_preserves_primary(tmp_path, monkeypatch):
    import threading
    class FalseError(RuntimeError):
        def __bool__(self):
            raise AssertionError('truthiness must not be used')
        def __repr__(self):
            raise AssertionError('unsafe error rendering')
    primary, receipt = FalseError('actual round'), MemoryError('receipt allocation')
    events = []
    class Profiler:
        def __enter__(self):
            events.append('enter')
        def __exit__(self, kind, error, tb):
            assert kind is FalseError and error is primary and tb is not None
            events.append('stop')
        def export_chrome_trace(self, path):
            events.append('export')
            Path(path).write_text('{}')
    tracer = owned._SchedulerTrace(NS(rank=0, scheduler=NS(thread=threading.current_thread())), tmp_path, factory=Profiler)
    monkeypatch.setattr(owned, 'write_json', lambda *args: (_ for _ in ()).throw(receipt))
    def round_():
        events.append('round')
        raise primary
    with pytest.raises(FalseError) as caught:
        tracer.capture('one', round_)
    assert caught.value is primary and primary.__cause__ is receipt
    assert events == ['enter', 'round', 'stop', 'export']
    assert tracer.failed is primary and tracer.profiler is not None
    with pytest.raises(FalseError) as again:
        tracer.capture('two', lambda: pytest.fail('retry after uncertain retirement'))
    assert again.value is primary


def test_trace_retirement_and_export_continue_after_cleanup_clock_failure(tmp_path, monkeypatch):
    import threading
    primary, clock = RuntimeError('round'), MemoryError('clock')
    ticks, events = iter([1., 2., clock, 3., 4.]), []
    def monotonic():
        value = next(ticks)
        if isinstance(value, BaseException):
            raise value
        return value
    class Profiler:
        def __enter__(self):
            events.append('enter')
        def __exit__(self, *args):
            events.append('stop')
        def export_chrome_trace(self, path):
            events.append('export')
            Path(path).write_text('{}')
    tracer = owned._SchedulerTrace(NS(rank=0, scheduler=NS(thread=threading.current_thread())), tmp_path, factory=Profiler)
    import time as process_time
    original_clock = process_time.monotonic
    monkeypatch.setattr(owned, 'time', NS(monotonic=monotonic))
    assert process_time.monotonic is original_clock  # Outer diagnostic hooks retain the real process clock.
    with pytest.raises(RuntimeError) as caught:
        tracer.capture('clock', lambda: (_ for _ in ()).throw(primary))
    assert caught.value is primary and primary.__cause__ is clock
    assert events == ['enter', 'stop', 'export']


@pytest.mark.parametrize('tail', [[], ['--scheduler-trace', 'trace'], ['--benchmark-plan', 'plan']])
def test_late_trace_cli_requires_both_owners_before_parent_output(tmp_path, tail):
    output = tmp_path / 'uncreated'
    with pytest.raises(SystemExit):
        owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(output),
                    '--scheduler-trace-prefill-start', '128', *tail, '--', 'model'])
    assert not output.exists()


@pytest.mark.parametrize('failure', [False, True])
def test_late_trace_install_observes_the_actual_prepare_binding_without_gpu_readback(tmp_path, monkeypatch, failure):
    import torch
    from tensorfold.families.qwen4_exp.cuda import history_owner
    from tensorfold.cuda.sm70 import family_ops
    names = [(history_owner, name) for name in ('partial_statistics', 'merge_statistics', 'normalized_partial')]
    names += [(family_ops, name) for name in ('hc_writeback', 'hc_normed', 'hc_act', 'hc_mix')]
    for owner, name in names:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    labels, entered, calls = [], [], []
    class Record:
        def __init__(self, label):
            self.label = label
        def __enter__(self):
            labels.append(self.label)
        def __exit__(self, *args):
            pass
    class Profiler:
        def __enter__(self):
            entered.append(True)
        def __exit__(self, *args):
            pass
        def export_chrome_trace(self, path):
            Path(path).write_text('{}')
    monkeypatch.setattr(torch.profiler, 'record_function', Record)
    before_cuda = torch.cuda.is_initialized()
    tracer, observer, stream = _late_trace_fixture(tmp_path, cursor=0, factory=Profiler)
    result = object()
    primary, notification = RuntimeError('prepare'), MemoryError('notification')
    def prepare(pieces):
        calls.append(('prepare', pieces))
        if failure and pieces[0][1] >= 128:
            raise primary
        return result
    def notify(*args):
        raise notification
    if failure:
        monkeypatch.setattr(owned, '_observer_failure', notify)
    decoder, adapter = observer.decoder, tracer.engine.tp_decoder
    decoder._prepare_pass, decoder._read_ahead = prepare, lambda pieces: None
    adapter._invoke = adapter._execute = lambda *args, **kwargs: None
    def round_():
        adapter.prefix_operation += 1
        return decoder._prepare_pass([(stream, decoder.fills[stream.sid][2], 64)])
    adapter.round = round_
    tracer.engine.w = NS(meta={'input_preparation_context': NS(prepare=lambda: None)})
    tracer.install(observer)
    assert json.loads((tmp_path / 'prefill-selection.json').read_text())['status'] == 'UNSUPPORTED_NOT_OBSERVED'
    assert adapter.round() is result  # Early cursor delegates without profiler/labels.
    assert len(calls) == 1 and entered == labels == []
    decoder.fills[stream.sid][2] = stream.st.pos = 160
    if failure:
        with pytest.raises(RuntimeError) as caught:
            adapter.round()
        assert caught.value is primary and primary.__cause__ is notification
    else:
        assert adapter.round() is result
    assert len(calls) == 2 and entered == [True] and labels == ['PLE.prepare_frame_stage']
    saved = json.loads((tmp_path / 'prefill-selection.json').read_text())
    assert saved['status'] == ('FAILED' if failure else 'OBSERVED') and saved['prefix_operation'] == 53
    assert saved['actual_start'] == saved['actual_state_pos'] == 160 and saved['actual_count'] == 64
    if failure:
        assert tracer.prefill_owner is stream and tracer.profiler is not None
    else:
        assert tracer.prefill_owner is tracer.prefill_state is None
    assert torch.cuda.is_initialized() is before_cuda


def test_trace_primary_survives_neutral_transport_exhaustion(tmp_path, monkeypatch):
    import threading
    primary, cleanup, exhausted = RuntimeError('round'), RuntimeError('stop'), MemoryError('neutral transport')
    class Profiler:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            raise cleanup
        def export_chrome_trace(self, path):
            Path(path).write_text('{}')
    tracer = owned._SchedulerTrace(NS(rank=0, scheduler=NS(thread=threading.current_thread())), tmp_path, factory=Profiler)
    def transport(actual, statuses):
        assert actual is primary and cleanup in statuses
        raise exhausted
    tracer.transport = transport
    with pytest.raises(RuntimeError) as caught:
        tracer.capture('one', lambda: (_ for _ in ()).throw(primary))
    assert caught.value is primary and primary.__cause__ is exhausted
    assert cleanup in tracer.capture_errors and tracer.profiler is not None


def test_late_trace_parent_forwards_explicit_selector_without_new_runner(tmp_path, monkeypatch):
    output, trace = tmp_path / 'out', tmp_path / 'trace'
    def launch(argv, **kwargs):
        prefix = kwargs['_command_prefix']
        assert prefix[prefix.index('--scheduler-trace-prefill-start') + 1] == '128'
        assert prefix[prefix.index('--scheduler-trace') + 1] == str(trace)
        assert prefix[prefix.index('--benchmark-plan') + 1] == str((tmp_path / 'plan.json').absolute())
        return 0
    monkeypatch.setattr(launcher, 'main', launch)
    assert owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(output), '--benchmark-plan', str(tmp_path / 'plan.json'),
                       '--scheduler-trace', str(trace), '--scheduler-trace-prefill-start', '128', '--', 'model']) == 0


def test_late_trace_cancelled_selected_round_does_not_claim_prefill_observation(tmp_path):
    import threading
    class Profiler:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        def export_chrome_trace(self, path):
            Path(path).write_text('{}')
    tracer, observer, stream = _late_trace_fixture(tmp_path, cursor=160, factory=Profiler)
    stream.cancel = threading.Event()
    label = tracer.label(observer)
    def round_():
        tracer.note_prepare([(stream, 160, 64)])
        tracer.selection['prepared_returned'] = True
        stream.cancel.set()
        return 'cancelled callback result'
    assert tracer.capture(label, round_) == 'cancelled callback result'
    selection = json.loads((tmp_path / 'prefill-selection.json').read_text())
    assert selection['status'] == 'UNSUPPORTED_NOT_OBSERVED'
    assert selection['reason'] == 'cancelled_during_selected_round'


@pytest.mark.parametrize('receipt_failure', [False, True])
def test_late_trace_cancellation_metadata_failure_still_retires_profiler(tmp_path, monkeypatch, receipt_failure):
    primary, events = MemoryError('cancel metadata'), []
    class Cancel:
        def is_set(self):
            raise primary
    class Profiler:
        def __enter__(self):
            events.append('enter')
        def __exit__(self, *args):
            events.append('stop')
        def export_chrome_trace(self, path):
            events.append('export')
            Path(path).write_text('{}')
    tracer, observer, stream = _late_trace_fixture(tmp_path, cursor=160, factory=Profiler)
    label = tracer.label(observer)
    receipt_error = OSError('later receipt failure')
    if receipt_failure:
        monkeypatch.setattr(owned, 'write_json', lambda *args: (_ for _ in ()).throw(receipt_error))
    def round_():
        tracer.note_prepare([(stream, 160, 64)])
        tracer.selection['prepared_returned'] = True
        stream.cancel = Cancel()
    with pytest.raises(MemoryError) as caught:
        tracer.capture(label, round_)
    assert caught.value is primary and events == ['enter', 'stop', 'export']
    if receipt_failure:
        assert primary.__cause__ is receipt_error
    assert tracer.failed is primary and tracer.profiler is not None


def test_initial_trace_selection_write_failure_survives_failed_setup_notification(channel, tmp_path, monkeypatch):
    from tools import bench_concurrent
    from tensorfold.families.qwen4_exp.cuda import history_owner
    from tensorfold.cuda.sm70 import family_ops
    names = [(history_owner, name) for name in ('partial_statistics', 'merge_statistics', 'normalized_partial')]
    names += [(family_ops, name) for name in ('hc_writeback', 'hc_normed', 'hc_act', 'hc_mix')]
    for owner, name in names:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    primary, notification = OSError('initial selection write'), MemoryError('notify')
    tracer, observer, stream = _late_trace_fixture(tmp_path, cursor=0)
    decoder, engine = observer.decoder, tracer.engine
    engine.w = NS(meta={'input_preparation_context': NS(prepare=lambda: None)})
    decoder._prepare_pass = decoder._read_ahead = lambda *args, **kwargs: None
    engine.tp_decoder._invoke = engine.tp_decoder._execute = engine.tp_decoder.round = lambda: None
    source = tmp_path / 'plan.json'
    source.write_text('{}')
    monkeypatch.setattr(bench_concurrent, 'load_matrix', lambda path: observer.plan)
    monkeypatch.setattr(owned, '_history_resources', lambda *args: {'global_host_obligation_bytes': 1000})
    def write(path, value):
        assert path.name == 'prefill-selection.json'
        raise primary
    def prepare(current, plan, output, **kwargs):
        assert current is engine and plan is observer.plan
        tracer.install(observer)
    def notify(current, error):
        assert current is engine and error is primary
        raise notification
    monkeypatch.setattr(owned, 'write_json', write)
    monkeypatch.setattr(owned, 'prepare_benchmark', prepare)
    monkeypatch.setattr(owned, '_observer_failure', notify)
    monkeypatch.setattr(cli, 'cmd_serve', lambda args: args._owned_expert_qualification.before_listener(engine))
    argv = worker_args(tmp_path)
    argv[argv.index('--rank') + 1] = '0'
    with pytest.raises(OSError) as caught:
        owned.worker(['--benchmark-plan', str(source), '--scheduler-trace', str(tmp_path),
                      '--scheduler-trace-prefill-start', '128', *argv])
    assert caught.value is primary and primary.__cause__ is notification
    assert tracer.profiler is None  # Setup failed before any profiler acquisition.


def _closed_benchmark_plan():
    cohorts = []
    for index, prompt in enumerate(([10, 11, 12], [20, 21, 22, 23])):
        cohorts.append({'id': index, 'level': 1, 'warmup': index == 0, 'tokens': 2,
                        'input_tokens': len(prompt), 'requests': [{'index': 0, 'prefix': prompt[:2],
                            'input_tokens': len(prompt), 'prompt_sha256': owned._prompt_digest(prompt)}]})
    return {'run_identity': 'a' * 32, 'cohorts': cohorts}, ([10, 11, 12], [20, 21, 22, 23])


def test_closed_benchmark_followers_track_real_completion_without_cuda_events(monkeypatch):
    import torch
    plan, prompts = _closed_benchmark_plan()
    monkeypatch.setattr(torch.cuda, 'Event', lambda *a, **k: pytest.fail('follower created timing event'))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *a: pytest.fail('follower queried CUDA stream'))
    monkeypatch.setattr(owned, 'notify_failure', lambda: True)
    observations = NS(target=lambda *a, **k: None, record=lambda **k: {})
    decoder = NS(streams={}, filling=[], controller=NS(work_observations=observations))
    def admit(stream):
        decoder.filling.append(stream)
    def take(stream, values):
        stream.out.extend(values)
        stream.done = len(stream.out) == 2
    def joined(pieces, *args):
        for stream, _, _ in pieces:
            decoder.filling.remove(stream)
            decoder.streams[id(stream)] = stream
            decoder._take(stream, [1])
    def finish(done):
        for stream in done:
            decoder.streams.pop(id(stream))
    decoder.admit, decoder._take, decoder._joined, decoder.finish = admit, take, joined, finish
    decoder._prepare_pass, decoder.round = lambda pieces: None, lambda: []
    engine = NS(rank=1, multi=decoder, w=NS(device='cuda'), tp_decoder=NS(wires={}), _abort_comm=False)
    observer = owned._BenchmarkObserver(engine, plan, timed=False, strict=True)
    observer.install()
    assert observer.events == [] and not observer.finished
    for index, prompt in enumerate(prompts):
        stream = NS(prompt=list(prompt), count=2, out=[], done=False, st=object(), error=None, cancel=None)
        engine.tp_decoder.wires[id(stream)] = index + 1
        decoder.admit(stream)
        pieces = [(stream, 0, len(prompt))]
        decoder._prepare_pass(pieces)
        decoder._joined(pieces)
        decoder._take(stream, [2])
        assert not observer.finished
        decoder.finish([stream])
        assert observer.finished is (index == 1)
    assert observer.failed is None
    duplicate = NS(prompt=list(prompts[0]), count=2, out=[], done=False, st=object(), error=None, cancel=None)
    engine.tp_decoder.wires[id(duplicate)] = 17
    with pytest.raises(ValueError, match='repeated'):
        decoder.admit(duplicate)
    assert observer.failed is not None and engine._abort_comm


def _closed_benchmark_client(tmp_path, monkeypatch, *, matrix_code=0, matrix_error=None, trace_error=None):
    from tools import bench_concurrent, openai_protocol
    teacher, calls = _one_teacher_client(tmp_path, monkeypatch)
    for rank in range(8):
        path = tmp_path / f'rank-{rank}.json'
        value = json.loads(path.read_text())
        for key in list(value):
            if key.startswith('teacher_'):
                value['benchmark_' + key.removeprefix('teacher_')] = value.pop(key)
        value['tool_sha256'] = owned._source_sha(owned.__file__)
        path.write_text(json.dumps(value))
    plan, _ = _closed_benchmark_plan()
    data = dict(teacher.selected['data'], response_bytes=16 << 20, sse_line_bytes=64 << 10)
    selected = {'data': data, 'sha256': 'b' * 64}
    prepared = {'plan': plan, 'plan_sha256': 'a' * 64, 'client': selected,
                'trace': {'prefill_start': 1}, 'resources': {'matrix_report_bytes': 1 << 20}}
    controls = []
    def run(args, actual_plan, *, read_control):
        assert actual_plan is plan and read_control.cancelled is client.cancelled
        # Use the REAL protocol unit conversion and exact-int Limits boundary.
        limits = openai_protocol.from_arguments(args)
        assert limits.response_bytes == 16 << 20 and limits.line_bytes == 64 << 10
        controls.append(read_control)
        read_control.check()
        if matrix_error is not None:
            raise matrix_error
        args.output.write_text(json.dumps({'plan': plan, 'passed': matrix_code == 0,
            'unrun_cohorts': [], 'instance_id': 'same-process'}))
        return matrix_code
    monkeypatch.setattr(owned, '_benchmark_module', lambda: NS(run_matrix=run, __file__=bench_concurrent.__file__))
    def trace(*args):
        if trace_error is not None:
            raise trace_error
        return {}
    monkeypatch.setattr(owned, '_benchmark_trace_files', trace)
    client = owned._BenchmarkClient(prepared, tmp_path)
    return client, calls, controls


def test_closed_benchmark_client_reuses_matrix_and_original_whole_control(tmp_path, monkeypatch):
    client, calls, controls = _closed_benchmark_client(tmp_path, monkeypatch)
    client.run()
    assert [row[2] for row in calls] == ['/stats', '/stats']  # POSTs belong solely to maintained matrix client.
    assert len(controls) == 1 and controls[0].cancelled is client.cancelled
    assert client.result['complete'] and client.result['matrix_passed'] and client.result['observed_trace']
    assert client.result['request_count'] == 2 and client.result['server_instance_id'] == 'same-process'
    assert client.result['quality_acceptance'] is False and client.result['P3_valid'] is False


@pytest.mark.parametrize('fault', ['matrix-nonzero', 'matrix-primary', 'trace', 'cancel'])
def test_closed_benchmark_client_failure_cannot_request_normal_stop(tmp_path, monkeypatch, fault):
    from tools.v100_rotorquant_live import ReadCancelled
    primary = OSError('matrix owner')
    client, calls, controls = _closed_benchmark_client(tmp_path, monkeypatch,
        matrix_code=2 if fault == 'matrix-nonzero' else 0,
        matrix_error=primary if fault == 'matrix-primary' else None,
        trace_error=ValueError('trace unsupported') if fault == 'trace' else None)
    if fault == 'cancel':
        client.cancelled.set()
    expected = ReadCancelled if fault == 'cancel' else OSError if fault == 'matrix-primary' else ValueError
    with pytest.raises(expected) as caught:
        client.run()
    if fault == 'matrix-primary':
        assert caught.value is primary
    assert client.result is None and not (tmp_path / 'benchmark-client-report.json').exists()


def test_common_native_observer_prepaid_once_and_teacher_key_contract_unchanged():
    base, native = {'global_host_obligation_bytes': 10}, {'global_host_bytes': 20}
    value = owned._owned_observer_credit(base, native)
    assert base == {'global_host_obligation_bytes': 10} and value['global_host_obligation_bytes'] == 30
    assert value['child_native_observer'] is native
    with pytest.raises(ValueError, match='twice'):
        owned._owned_observer_credit(value, native)


def _closed_benchmark_engine(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    pair = get_pair('int8')
    coordinate, position = 'a' * 64, 'b' * 64
    identity = (pair.identity, 'working-policy', ('logical-context-v1', 262144),
                ('position-profile-v1', position, 0), ('coordinate-profile-v1', coordinate))
    states = [NS(capacity=262145, limit=262145, logical_context=262144, kv_identity=identity) for _ in range(24)]
    family = {'owner_kernel_family': 'torch-reference', 'owner_graph_family': 'uncaptured'}
    decoder = NS(states=states, speculation=NS(method='none', maximum_proposals=0), graphs=None, full_resident=True,
                 buf=NS(rows=24, **family), pbuf=NS(rows=256, **family))
    events = []
    engine = NS(rank=0, tp=8, multi=decoder, w=NS(mtp=None, device='cuda'), sm70_profile='reference',
        kv_pair=pair, position_profile=NS(identity=position, epoch=0), context_window=262144,
        _lifecycle=object(), _closed=False, _owned_qualification_host=(1000, 'c' * 64),
        retention_catalog=NS(host_resources=NS(baseline_host_bytes=1000)))
    def resources():
        events.append('resource-fence')
    engine._close_resources = resources
    def close():
        engine._close_resources()
        engine._closed = True
        events.append('closed')
    engine.close = close
    plan, _ = _closed_benchmark_plan()
    native = {'module': NS(record_native_elfs=lambda *args, **kwargs: events.append('native')),
              'directory': tmp_path, 'source': {'path': '/operator/driver.py', 'bytes': 1, 'sha256': 'f' * 64}}
    prepared = {'resources': {'global_host_obligation_bytes': 1000}, 'plan_sha256': 'c' * 64,
                'client': {'sha256': 'd' * 64}, 'trace': {'device_headroom_bytes': 512 << 20},
                'native': native, 'output': tmp_path}
    profile = {'coordinate_profile_identity': coordinate}
    monkeypatch.setattr(owned, 'bound_receipt', lambda current: profile)
    monkeypatch.setattr(owned, '_benchmark_sources', lambda current: {'engine': 'e' * 64})
    headroom = {'free_bytes': 1 << 30, 'total_bytes': 32 << 30, 'required_free_bytes': 512 << 20,
                'scope': 'boundary spy, not physical qualification'}
    monkeypatch.setattr(owned, '_benchmark_headroom', lambda *args: dict(headroom))
    observer = NS(engine=engine, plan=plan, records={c['id']: {'done': {0}, 'hashes': {c['requests'][0]['prompt_sha256']}}
                  for c in plan['cohorts']}, finished=True, failed=None, server_instance_id='local-rank')
    return engine, observer, prepared, events


def test_closed_benchmark_native_proof_follows_real_owner_fence_and_positive_close(tmp_path, monkeypatch):
    engine, observer, prepared, events = _closed_benchmark_engine(tmp_path, monkeypatch)
    owned._prepare_closed_benchmark(engine, observer, prepared)
    engine.close()
    assert events == ['resource-fence', 'native', 'closed']
    record = json.loads((tmp_path / 'benchmark-rank0/summary.json').read_text())
    assert record['closed'] and record['finished'] and not record['failed']
    assert record['actual_owned_profile']['benchmark_semantics']['resident_slots'] == 24
    assert record['headroom'] == observer.headroom and record['P3_valid'] is False
    with pytest.raises(RuntimeError, match='already attempted'):
        engine.close()


@pytest.mark.parametrize('fault', ['obligation', 'baseline', 'State', 'KV', 'method', 'family'])
def test_closed_benchmark_rejects_false_admission_before_hook_or_headroom(tmp_path, monkeypatch, fault):
    engine, observer, prepared, events = _closed_benchmark_engine(tmp_path, monkeypatch)
    if fault == 'obligation':
        engine._owned_qualification_host = (999, 'c' * 64)
    elif fault == 'baseline':
        engine.retention_catalog.host_resources.baseline_host_bytes = 999
    elif fault == 'State':
        engine.multi.states[7].limit = 1024
    elif fault == 'KV':
        engine.multi.states[7].kv_identity = ('wrong',)
    elif fault == 'method':
        engine.multi.speculation.method = 'mtp'
    else:
        engine.multi.pbuf.owner_graph_family = 'captured'
    monkeypatch.setattr(owned, '_benchmark_headroom', lambda *args: pytest.fail('headroom before admission proof'))
    with pytest.raises(ValueError):
        owned._prepare_closed_benchmark(engine, observer, prepared)
    assert events == [] and not (tmp_path / 'benchmark-rank0').exists()


def test_closed_benchmark_failed_close_and_summary_failure_preserve_original(tmp_path, monkeypatch):
    engine, observer, prepared, events = _closed_benchmark_engine(tmp_path, monkeypatch)
    primary, publication = RuntimeError('actual close failure'), OSError('summary')
    def close():
        raise primary
    engine.close = close
    owned._prepare_closed_benchmark(engine, observer, prepared)
    monkeypatch.setattr(owned, 'write_json', lambda *args: (_ for _ in ()).throw(publication))
    with pytest.raises(RuntimeError) as caught:
        engine.close()
    assert caught.value is primary and primary.__cause__ is publication and engine._closed is False
    assert events == []


@pytest.mark.parametrize('free', [1 << 30, (512 << 20) - 1])
def test_benchmark_headroom_is_fresh_free_observation_not_reserve_or_allocation(monkeypatch, free):
    import torch
    calls = []
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: calls.append(device) or (free, 32 << 30))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: pytest.fail('invented physical reservation'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: pytest.fail('extra headroom fence'))
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: pytest.fail('headroom must not allocate'))
    engine = NS(w=NS(device='actual-owned-device'))
    if free < 512 << 20:
        with pytest.raises(MemoryError):
            owned._benchmark_headroom(engine, {'device_headroom_bytes': 512 << 20})
    else:
        row = owned._benchmark_headroom(engine, {'device_headroom_bytes': 512 << 20})
        assert row['free_bytes'] == free and row['required_free_bytes'] == 512 << 20
        assert 'NOT a reservation' in row['scope']
    assert calls == ['actual-owned-device']


def test_benchmark_client_cannot_activate_from_public_or_untrusted_input(tmp_path):
    output = tmp_path / 'not-created'
    with pytest.raises(SystemExit):
        owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(output), '--benchmark-plan', 'plan',
                    '--benchmark-client', 'client', '--scheduler-trace', str(output / 'trace'), '--', 'model'])
    assert not output.exists()


def test_closed_benchmark_parent_uses_shared_task_path_and_only_bounded_trace_dto(tmp_path, monkeypatch):
    import threading
    from tools import v100_owned_model_contract as contract
    output = tmp_path / 'benchmark'
    plan_path, client_path = tmp_path / 'plan.json', tmp_path / 'client.json'
    plan_path.write_text('{}')
    client_path.write_text('{}')
    def identity(path):
        return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': owned._source_sha(path)}
    trace = {'prefill_start': 1, 'host_obligation_bytes': 8 << 30,
             'device_headroom_bytes': 512 << 20, 'disk_bytes': 16 << 30}
    operation = {'benchmark_plan': identity(plan_path), 'benchmark_client': identity(client_path), 'trace': trace}
    plan, _ = _closed_benchmark_plan()
    class Contract:
        def __getattr__(self, name):
            return getattr(contract, name)
        def validate_benchmark_operation(self, value):
            assert value is operation
            return value
        def validate_benchmark_operation_plan(self, actual, selected_plan, client):
            assert actual is operation and selected_plan is plan
    monkeypatch.setattr(owned, '_model_contract', lambda: Contract())
    monkeypatch.setattr(owned, '_benchmark_module', lambda: NS(load_matrix=lambda path: plan))
    monkeypatch.setattr(owned, '_benchmark_client_input', lambda *args: {'data': {}, 'bytes': 2, 'sha256': 'b' * 64})
    monkeypatch.setattr(owned, '_owned_model_observer', lambda *args: {'resources': {}})
    monkeypatch.setattr(owned, '_benchmark_resources', lambda *args: {'global_host_obligation_bytes': 1000})
    completed, events = threading.Event(), []
    class Client:
        def __init__(self, *args):
            self.result, self.cancelled, self.post_attempts = None, threading.Event(), 0
        def run(self):
            self.result = {'complete': True}
            completed.set()
    monkeypatch.setattr(owned, '_BenchmarkClient', Client)
    def launch(command, **kwargs):
        assert completed.wait(2)
        prefix = kwargs['_command_prefix']
        assert '--teacher-client-bytes' not in prefix and '--teacher-plan' not in prefix
        assert prefix[prefix.index('--benchmark-client') + 1] == str(client_path)
        assert json.loads(prefix[prefix.index('--benchmark-trace-data') + 1]) == trace
        from time import monotonic
        deadline = monotonic() + 2
        while not kwargs['_shutdown_probe'](False):
            assert monotonic() < deadline
        assert kwargs['_failure_probe']() is False
        events.append('all8-launch0')
        return 0
    monkeypatch.setattr(launcher, 'main', launch)
    monkeypatch.setattr(owned, '_benchmark_model_result', lambda *args: events.append('typed-result'))
    assert owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(output), '--benchmark-plan', str(plan_path),
        '--benchmark-client', str(client_path), '--scheduler-trace', str(output / 'trace'),
        '--scheduler-trace-prefill-start', '1', '--', 'model'], _benchmark_operation=operation) == 0
    assert events == ['all8-launch0', 'typed-result']


@pytest.mark.parametrize('stage', ['record', 'elapsed'])
def test_closed_benchmark_event_primary_survives_notify_exhaustion(monkeypatch, stage):
    import torch
    primary, notification = RuntimeError('event record'), MemoryError('notify')
    events = []
    class Event:
        def record(self, stream):
            if stage == 'record':
                raise primary
        def query(self):
            return True
        def elapsed_time(self, other):
            raise primary
    plan, prompts = _closed_benchmark_plan()
    observations = NS(target=lambda *a, **k: None, record=lambda **k: {})
    decoder = NS(streams={}, filling=[], controller=NS(work_observations=observations),
        _prepare_pass=lambda pieces: pytest.fail('prepare after event failure'),
        _joined=lambda *a: None, _take=lambda *a: None, round=lambda: [], finish=lambda *a: None)
    stream = NS(prompt=prompts[0], count=2, out=[], done=False, st=object(), error=None, cancel=None)
    decoder.admit = lambda selected: decoder.filling.append(selected)
    engine = NS(rank=0, multi=decoder, w=NS(device='cuda'), tp_decoder=NS(wires={id(stream): 1}))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: 'stream')
    def notify(current, error):
        events.append(error)
        raise notification
    monkeypatch.setattr(owned, '_observer_failure', notify)
    observer = owned._BenchmarkObserver(engine, plan, events=[Event() for _ in range(4)], strict=True)
    observer.install()
    decoder.admit(stream)
    with pytest.raises(RuntimeError) as caught:
        if stage == 'record':
            decoder._prepare_pass([(stream, 0, len(stream.prompt))])
        else:
            decoder.filling = []
            engine.scheduler = NS(thread=object())
            observer.records[0]['done'] = {0}
            observer.records[0]['started'] = [True] * 4
            observations.record()
    assert caught.value is primary and primary.__cause__ is notification
    assert observer.failed is primary and events == [primary]


def _benchmark_aggregate_fixture(tmp_path, *, semantic_rank=None, producer_rank=None):
    import copy
    from tools import v100_owned_model_contract as contract
    output, native = tmp_path / 'benchmark', tmp_path / 'child-native'
    output.mkdir()
    native.mkdir()
    plan, _ = _closed_benchmark_plan()
    trace = {'prefill_start': 1, 'host_obligation_bytes': 8 << 30,
             'device_headroom_bytes': 512 << 20, 'disk_bytes': 16 << 30}
    resources = {'global_host_obligation_bytes': 1000, 'trace_grants': trace}
    source = {'path': '/canonical/tools/v100_native_contract_driver.py', 'bytes': 1, 'sha256': 'f' * 64}
    prepared = {'plan': plan, 'plan_sha256': 'c' * 64, 'client': {'sha256': 'd' * 64}, 'trace': trace,
                'resources': resources, 'native': {'directory': native, 'source': source}}
    sources = {key: 'e' * 64 for key in ('engine', 'multi', 'decode', 'sampling', 'observer_tool')}
    headroom = {'free_bytes': 1 << 30, 'total_bytes': 32 << 30, 'required_free_bytes': 512 << 20, 'scope': 'CPU fixture'}
    cohorts = [{'id': c['id'], 'level': c['level'], 'completed': 1,
                'prompt_hashes': [c['requests'][0]['prompt_sha256']]} for c in plan['cohorts']]
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    for rank in range(8):
        profile = {'rank': rank, 'resources': {'rank_local_observation': rank}, 'benchmark_semantics': {
            'method': 'none', 'state_kv_identity': ['pair', ['position-profile-v1', 'a' * 64, 0]],
            'context': 262144, 'resident_slots': 24}}
        if rank == semantic_rank:
            profile['benchmark_semantics']['state_kv_identity'][1][2] = 1
        startup = {'rank': rank, 'world': 8, 'benchmark_profile': copy.deepcopy(profile),
            'benchmark_worker_PID': 100 + rank, 'benchmark_server_instance_id': f'worker-{rank}',
            'benchmark_resources': resources, 'benchmark_headroom': headroom, 'benchmark_observer': source,
            'tool_sha256': 'e' * 64, 'runtime_cli_sha256': 'a' * 64,
            'private_router_policy': {'configured': 'inherit'}, 'reference_tile_preparation': {'requested': False}}
        if rank == producer_rank:
            startup['private_router_policy'] = {'configured': 'native-serial-v1'}
        summary = {'schema': 'tensorfold.owned-benchmark-rank-result-v1', 'rank': rank, 'world': 8,
            'closed': True, 'finished': True, 'failed': False, 'benchmark_plan_sha256': 'c' * 64,
            'benchmark_client_sha256': 'd' * 64, 'run_identity': plan['run_identity'],
            'server_instance_id': f'worker-{rank}', 'actual_owned_profile': copy.deepcopy(profile),
            'runtime_sources': sources, 'cohorts': cohorts, 'resources': resources, 'headroom': headroom,
            'qualification': False, 'P3_valid': False}
        write(output / f'rank-{rank}.json', startup)
        write(output / f'benchmark-rank{rank}/summary.json', summary)
        write(native / f'benchmark-closed-rank-{rank}-pid-{100+rank}.json', {
            'rank': rank, 'PID': 100 + rank, 'operation_limits': contract.MODEL_OBSERVER_LIMITS,
            'missing_required_prefixes': [], 'observer': {'sha256': 'f' * 64},
            'mapped_elfs': [{'name': 'tf_gguf_fixture', 'sha256': 'a' * 64}]})
    write(output / 'benchmark-client-report.json', {'schema': 'tensorfold.owned-benchmark-client-report-v1',
        'complete': True, 'matrix_passed': True, 'observed_trace': True, 'P3_valid': False})
    write(output / 'matrix-results.json', {'schema': 'tensorfold.phase-matrix-result-v1', 'plan': plan,
        'passed': True, 'unrun_cohorts': []})
    selected = owned._trace_prefill_selection(plan, 1)
    selected.update(status='OBSERVED', boundary_start=1, actual_start=1, actual_count=1,
        actual_state_pos=1, state_pos=1, slot=0, slot_generation=1, wire=17, prefix_operation=4,
        prepared_calls=1, prepared_returned=True, profiler_entry_headroom=headroom)
    write(output / 'trace/prefill-selection.json', selected)
    write(output / 'trace/inventory.json', {'actual_matrix_samples': [{'name': 'CPU fixture'}]})
    label = 'c1-prefill-start-ge-1-round'
    chrome = output / f'trace/{label}.json'
    chrome.write_text('{}')
    write(output / f'trace/{label}-receipt.json', {'primary': None, 'post_errors': [], 'P3_valid': False,
        'prefill_selection': selected, 'exported_trace_bytes': chrome.stat().st_size})
    return output, prepared


def test_benchmark_actual_aggregate_seals_all_eight_and_preserves_rank_local_observations(tmp_path):
    from tools import v100_owned_model_contract as contract
    output, prepared = _benchmark_aggregate_fixture(tmp_path)
    result = owned._benchmark_model_result(output, prepared)
    assert contract.validate_benchmark_model_result(result) is result
    assert result['schema'] == 'tensorfold.owned-benchmark-model-result-v1' and result['P3_valid'] is False
    assert [row['rank'] for row in result['rank_receipts']] == list(range(8))
    assert result['startup_receipts'][7]['file'] == 'benchmark/rank-7.json'
    assert result['rank_receipts'][7]['file'] == 'benchmark/benchmark-rank7/summary.json'
    assert result['child_native_records'][7]['file'] == 'child-native/benchmark-closed-rank-7-pid-107.json'
    assert result['trace']['chrome']['file'] == 'benchmark/trace/c1-prefill-start-ge-1-round.json'
    assert json.loads((output / 'benchmark-model-result.json').read_text()) == result


@pytest.mark.parametrize('drift', ['semantic', 'producer'])
def test_benchmark_actual_aggregate_refuses_rebound_single_rank_semantic_or_selector_drift(tmp_path, drift):
    output, prepared = _benchmark_aggregate_fixture(tmp_path,
        semantic_rank=7 if drift == 'semantic' else None, producer_rank=7 if drift == 'producer' else None)
    if drift == 'semantic':
        startup = json.loads((output / 'rank-7.json').read_text())
        closed = json.loads((output / 'benchmark-rank7/summary.json').read_text())
        assert startup['benchmark_profile'] == closed['actual_owned_profile']  # Rebound within-rank equality is insufficient.
    with pytest.raises(ValueError, match='semantics differ across ranks'):
        owned._benchmark_model_result(output, prepared)
    assert not (output / 'benchmark-model-result.json').exists()
