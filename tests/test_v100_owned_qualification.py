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
