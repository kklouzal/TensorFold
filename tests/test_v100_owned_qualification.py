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


@pytest.fixture
def channel(monkeypatch):
    read, write = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    monkeypatch.setenv(FAILURE_FD_ENV, str(write))
    try:
        yield read, write
    finally:
        os.close(write)
        os.close(read)


def test_parent_delegates_eight_ranks_and_borrows_supervised_pipe(monkeypatch, tmp_path):
    captured = {}
    def supervise(commands, environments, **options):
        captured.update(commands=commands, environments=environments, options=options)
        os.fstat(options["pass_fds"][0])
        return 23
    monkeypatch.setattr(launcher, "supervise", supervise)
    assert owned.main(["--gpus", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "new"),
                       "--", "model", "--backend", "cuda"]) == 23
    assert [env["CUDA_VISIBLE_DEVICES"] for env in captured["environments"]] == ["1", "2", "3", "4", "5", "6", "7", "0"]
    for command in captured["commands"]:
        assert command[:3] == [sys.executable, str(Path(owned.__file__).resolve()), "--worker"]
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


@pytest.mark.parametrize("private", [False, True])
def test_shared_cuda_factory_gets_private_bool_only_with_capability(monkeypatch, tmp_path, request, private):
    import tensorfold.cuda.server as server
    (tmp_path / "config.json").write_text('{"max_position_embeddings":128}')
    calls, options = [], []
    engine = cuda_cli_engine(close=lambda: calls.append("engine-close"))
    family = NS(package=NS(cuda_engine=lambda *a, **k: options.append(k) or engine), title="Flash Next", model_type="qwen4_exp")
    monkeypatch.setattr(server, "App", lambda *a, **k: cuda_cli_app(effective_context_window=128))
    install_cli_server(monkeypatch, request, server, on_serve=lambda _app: calls.append("serve"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    if private:
        args._owned_expert_qualification = cli._OwnedExpertQualification(lambda selected: calls.append("receipt") if selected is engine else pytest.fail("different engine"))
    assert cli._serve_cuda(args, family, tmp_path, 128) == 0
    assert options[0].get("_owned_expert_native", False) is private
    if not private:
        assert "_owned_expert_native" not in options[0]
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
