"""Expert ownership, bounded work wire and rank-process failure contracts; no GPU emulation."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda.expert_launch import rank_commands, supervise
from tensorfold.cuda.expert_parallel import Ownership, ownership
from tensorfold.families.qwen4_exp.cuda.expert_parallel import Follower, Session, decode_frame


def test_asymmetric_shards_cover_every_expert_exactly_once():
    owners = [ownership(2, "341,171", rank, 512) for rank in range(2)]
    assert [list(o.ids) for o in owners] == [list(range(341)), list(range(341, 512))]
    assert ownership(1, None, 0, 512).counts == (512,)


def test_direct_ownership_requires_immutable_validated_counts():
    with pytest.raises(ValueError):
        Ownership([3, 3], 0)
    counts = [3, 3]
    owner = ownership(2, counts, 0, 6)
    counts[0] = 99
    assert owner.counts == (3, 3)


@pytest.mark.parametrize("backend", ["fast", "roce", "unknown"])
def test_expert_session_refuses_transport_override_before_checkpoint_or_device_work(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("TF_COMM_BACKEND", backend)
    with pytest.raises(ValueError, match="requires TF_COMM_BACKEND=nccl"):
        Session(tmp_path, ranks=2, shards="3,3", rank=0, master="127.0.0.1", port=29551,
                budget="auto", mtp=False)


@pytest.mark.parametrize("backend", ["", "  NCCL  "])
def test_expert_session_accepts_normalized_native_transport_policy(monkeypatch, tmp_path, backend):
    from tensorfold.families.qwen4_exp.cuda.weight_types import Config
    monkeypatch.setenv("TF_COMM_BACKEND", backend)
    def metadata(path):
        raise RuntimeError("accepted transport reached checkpoint metadata")
    monkeypatch.setattr(Config, "read", metadata)
    with pytest.raises(RuntimeError, match="accepted transport reached checkpoint metadata"):
        Session(tmp_path, ranks=2, shards="3,3", rank=0, master="127.0.0.1", port=29551,
                budget="auto", mtp=False)


@pytest.mark.parametrize("ranks,shards,rank,experts", [
    (2, "341,170", 0, 512), (2, "512", 0, 512), (2, "0,512", 0, 512),
    (2, "341,171", 2, 512), (2, "341,171", True, 512), (2, "1,x", 0, 512),
    (True, "1", 0, 1), (5, "1,1,1,1,1", 0, 5), (1, (1024,), 0, 1024),
])
def test_invalid_owner_configuration_is_refused(ranks, shards, rank, experts):
    with pytest.raises(ValueError):
        ownership(ranks, shards, rank, experts)


def test_control_frame_has_exact_sequence_bank_geometry_and_arithmetic_mode():
    assert decode_frame(b"[9,2,128,1]", sequence=9, banks=3) == (2, 128, True)
    assert decode_frame(b"[9,-1,0,0]", sequence=9, banks=3) is None


@pytest.mark.parametrize("raw", [
    b"[8,0,1,0]", b"[9,3,1,0]", b"[9,0,129,0]", b"[9,0,0,0]", b"[9,0,1,2]",
    b"[9,-1,0,1]", b"[9,0,true,0]", b"[9,0,1.0,0]", b"{}", b"[9,0,1,0,0]", b" " * 513,
])
def test_invalid_work_does_not_enter_native_collectives(raw):
    with pytest.raises(ValueError):
        decode_frame(raw, sequence=9, banks=3)


def test_one_failed_rank_retires_other_owned_process_groups(tmp_path):
    marker = tmp_path / "peer.pid"
    writer = ("import os,pathlib,time;pathlib.Path(" + repr(str(marker)) +
              ").write_text(str(os.getpid()));time.sleep(60)")
    failing = "import time,sys;time.sleep(.3);sys.exit(17)"
    start = time.monotonic()
    assert supervise([[sys.executable, "-c", writer], [sys.executable, "-c", failing]],
                     [dict(os.environ)] * 2, grace=1) == 17
    assert time.monotonic() - start < 10
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_local_slot_mapping_retains_shared_owner_and_skips_only_other_owners():
    torch = pytest.importorskip("torch")
    picks = torch.tensor([[0, 340, 341, 511, 512]], dtype=torch.int32)
    assert Ownership((341, 171), 0).local(picks).tolist() == [[0, 340, -1, -1, 341]]
    assert Ownership((341, 171), 1).local(picks).tolist() == [[-1, -1, 0, 170, -1]]
    with pytest.raises(ValueError):
        Ownership((341, 171), 1).local(torch.tensor([513]))


def _session_owner(comm, *, store=None):
    session = object.__new__(Session)
    session.lock = threading.RLock()
    session.abort_requested = threading.Event()
    session.running = threading.Event()
    session.following = threading.Event()
    session.rank, session.world = 0, 2
    session.closed = session.retired = False
    session.comm, session.reader = comm, None
    session.cache = SimpleNamespace(close=lambda: None)
    session.source_files = []
    session.store = store or SimpleNamespace(set=lambda *args: None)
    return session


def test_failed_failure_notification_still_aborts_and_retires_payloads():
    retired = []
    def failed_store(*args):
        raise OSError("leader disappeared")
    session = _session_owner(SimpleNamespace(close=lambda **kw: retired.append(kw)),
                             store=SimpleNamespace(set=failed_store))
    with pytest.raises(OSError, match="leader disappeared"):
        session.close(abort=True)
    assert retired == [{"abort": True}]
    assert session.retired


def test_unresolved_native_abort_retains_cache_and_source_for_retry():
    attempts, payloads = [], []
    def retire(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("native abort unresolved")
    session = _session_owner(SimpleNamespace(close=retire))
    session.cache = SimpleNamespace(close=lambda: payloads.append("cache"))
    session.source_files = [SimpleNamespace(close=lambda: payloads.append("source"))]
    with pytest.raises(RuntimeError, match="native abort unresolved"):
        session.close(abort=True)
    assert session.closed and not session.retired and payloads == []
    session.close()
    assert session.retired and payloads == ["cache", "source"]
    assert attempts == [{"abort": True}, {"abort": True}]


def test_follower_never_reads_authority_before_group_admission():
    events = []
    def receive(key):
        events.append(key)
        raise MemoryError("group admission refused")
    session = SimpleNamespace(_receive=receive, _ensure_reader=lambda: events.append("reader"),
                              close=lambda **kw: events.append(kw))
    with pytest.raises(MemoryError, match="group admission refused"):
        Follower(session)
    assert events == ["ep/admitted", {"abort": True}]


def test_follower_refuses_absolute_cap_before_any_expert_payload_allocation(monkeypatch):
    from tensorfold.cuda import capacity
    monkeypatch.setenv(capacity.LIMIT_ENV, "1")
    monkeypatch.setattr(capacity, "available_bytes", lambda torch: 1 << 30)
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: 1 << 30)
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda: 1 << 30)
    events = []
    session = SimpleNamespace(_receive=lambda key: b"ready", _ensure_reader=lambda: events.append("metadata"),
                              layout=SimpleNamespace(host_bytes=0, staging_bytes=0, loading_bytes=0,
                                                     metadata_host_bytes=0, gpu_bytes=60, loading_device_bytes=1),
                              workspace_bytes=40,
                              torch=SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda: (1 << 30) - 50)),
                              load_bank=lambda *a: pytest.fail("expert payload allocated before cap admission"),
                              close=lambda **kw: events.append(kw))
    with pytest.raises(MemoryError, match="admitted device capacity"):
        Follower(session)
    assert events == ["metadata", {"abort": True}]


@pytest.mark.parametrize("outcome", ["bounded", "live-overflow", "physical-shortage", "reusable-reservation"])
def test_follower_auto_cache_honors_absolute_live_cap_and_postallocation_room(monkeypatch, outcome):
    """Independent byte balances; mock allocator accounting is not multi-GPU execution evidence."""
    from tensorfold.cuda import capacity
    from tensorfold.families.qwen4_exp.ram_experts import AUTO_HEADROOM_BYTES
    mib = 1 << 20
    fixed, workspace, cell, slack = 64 * mib, 32 * mib, 16 * mib, 7 * mib
    cap = fixed + AUTO_HEADROOM_BYTES + workspace + 3 * cell + slack
    monkeypatch.setenv(capacity.LIMIT_ENV, str(cap / (1 << 30)))
    state = {"allocated": fixed, "spare": 0, "free": 2 << 30}
    allocations = []
    def allocate(size):
        allocations.append(size)
        state["allocated"] += size + (8 * mib if outcome == "live-overflow" else 0)
        state["free"] -= size
        if outcome in ("physical-shortage", "reusable-reservation"):
            state["free"] = AUTO_HEADROOM_BYTES
            # Non-releasable free fragments within allocated CUDA blocks can
            # satisfy later tensor allocations while empty_cache retains them.
            state["spare"] = workspace - (1 if outcome == "physical-shortage" else 0)
    runtime = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None,
                              mem_get_info=lambda: (state["free"], 4 << 30),
                              memory_allocated=lambda: state["allocated"],
                              memory_reserved=lambda: state["allocated"] + state["spare"]))
    follower = object.__new__(Follower)
    follower.session = SimpleNamespace(torch=runtime, workspace_bytes=workspace,
                                       cache=SimpleNamespace(entry_bytes=cell, score_history_entries=128,
                                                             configure_before_use=allocate))
    if outcome in ("live-overflow", "physical-shortage"):
        with pytest.raises(MemoryError, match="retain its declared workspace"):
            follower._configure_auto_experts()
    else:
        follower._configure_auto_experts()
        assert state["allocated"] + AUTO_HEADROOM_BYTES + workspace <= cap
        assert follower.session.auto_pool_decision["explicit_device_limit_bytes"] == cap
    assert allocations == [3 * cell]  # independent cap balance admits exactly three whole cells


def test_successful_follower_departure_retires_a_still_live_leader():
    commands = [[sys.executable, "-c", "import time; time.sleep(60)"],
                [sys.executable, "-c", "pass"]]
    assert supervise(commands, [dict(os.environ)] * 2, grace=.1) == 1


def test_exited_rank_does_not_leave_a_compiler_in_another_process_group(tmp_path):
    marker = tmp_path / "compiler.pid"
    compiler = ("import os,pathlib,signal,time;os.setpgid(0,0);signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(60)")
    leader = ("import pathlib,subprocess,sys,time;"
              f"subprocess.Popen([sys.executable,'-c',{compiler!r}]);p=pathlib.Path({str(marker)!r});"
              "\nwhile not p.exists(): time.sleep(.01)\nsys.exit(17)")
    assert supervise([[sys.executable, "-c", leader]], [dict(os.environ)], grace=.1) == 17
    stat = Path("/proc") / marker.read_text() / "stat"
    if stat.exists():
        assert stat.read_bytes().rsplit(b") ", 1)[1].split()[0] in (b"Z", b"X")


def test_partial_spawn_preserves_error_and_retires_already_started_rank(tmp_path, monkeypatch):
    import tensorfold.cuda.expert_launch as launcher
    marker = tmp_path / "started.pid"
    real_spawn = subprocess.Popen
    spawned = []
    def spawn(command, **kwargs):
        if spawned:
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            raise FileNotFoundError("second rank executable absent")
        child = real_spawn(command, **kwargs)
        spawned.append(child)
        return child
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    writer = (f"import os,pathlib,time;pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(60)")
    with pytest.raises(FileNotFoundError, match="second rank executable absent"):
        launcher.supervise([[sys.executable, "-c", writer], ["absent"]], [dict(os.environ)] * 2, grace=.1)
    with pytest.raises(ProcessLookupError):
        os.kill(spawned[0].pid, 0)


def test_unresolved_session_retains_explicit_owner_and_still_retires_other_ranks(monkeypatch):
    import tensorfold.cuda.expert_launch as launcher
    real_retire = launcher._retire
    owners = []
    def retire(child, deadline):
        owners.append(child)
        if len(owners) == 1:
            raise OSError("injected session retirement failure")
        real_retire(child, deadline)
    monkeypatch.setattr(launcher, "_retire", retire)
    try:
        with pytest.raises(launcher.UnresolvedRankSessions) as caught:
            launcher.supervise([[sys.executable, "-c", "pass"]] * 2, [dict(os.environ)] * 2, grace=.1)
        assert len(owners) == 2
        assert caught.value.sessions == (owners[0],)
        assert isinstance(caught.value.__cause__, OSError)
        assert owners[0].returncode is None and owners[1].returncode is not None
    finally:
        if owners and owners[0].returncode is None:
            real_retire(owners[0], time.monotonic())


def test_failed_owner_diagnostic_preserves_original_failure_and_retires_independent_rank(monkeypatch):
    import tensorfold.cuda.expert_launch as launcher
    real_retire = launcher._retire
    owners = []
    primary, diagnostic = OSError("injected session retirement failure"), MemoryError("owner diagnostic unavailable")
    def retire(child, deadline):
        owners.append(child)
        if len(owners) == 1:
            raise primary
        real_retire(child, deadline)
    def unavailable(*args):
        raise diagnostic
    monkeypatch.setattr(launcher, "_retire", retire)
    monkeypatch.setattr(launcher, "UnresolvedRankSessions", unavailable)
    try:
        with pytest.raises(OSError) as caught:
            launcher.supervise([[sys.executable, "-c", "pass"]] * 2, [dict(os.environ)] * 2, grace=.1)
        assert caught.value is primary and caught.value.__cause__ is diagnostic
        assert len(owners) == 2 and owners[0].returncode is None and owners[1].returncode is not None
    finally:
        if owners and owners[0].returncode is None:
            real_retire(owners[0], time.monotonic())


def test_lost_direct_child_reservation_never_signals_or_rescans_numeric_session(monkeypatch):
    import tensorfold.cuda.expert_launch as launcher
    child = SimpleNamespace(pid=12345, returncode=None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: child)
    def missing(process):
        raise ChildProcessError("direct child reservation lost")
    monkeypatch.setattr(launcher, "_exited", missing)
    monkeypatch.setattr(launcher.os, "killpg", lambda *a: pytest.fail("unowned numeric group signal"))
    monkeypatch.setattr(launcher, "_retire", lambda *a: pytest.fail("unowned numeric session scan"))
    with pytest.raises(ChildProcessError, match="reservation lost") as caught:
        launcher.supervise([["ignored"]], [dict(os.environ)], grace=0)
    assert isinstance(caught.value.__cause__, launcher.UnresolvedRankSessions)
    assert caught.value.__cause__.sessions == (child,)


def test_unsharded_and_indexed_inventory_identify_the_same_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from tensorfold.families.qwen4_exp.cuda.reader import checkpoint_inventory
    save_file({"owned.weight": torch.zeros((2, 4), dtype=torch.bfloat16)}, tmp_path / "model.safetensors")
    assert checkpoint_inventory(tmp_path) == {"owned.weight": "model.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"owned.weight": "model.safetensors"}}))
    assert checkpoint_inventory(tmp_path) == {"owned.weight": "model.safetensors"}


def test_pin_geometry_is_scaled_from_the_authoritative_layout():
    from tensorfold.families.qwen4_exp.expert_authority import allocations
    layout = SimpleNamespace(format="affine4", authority_allocations=(600, 300, 600, 300))
    assert allocations({"num_experts": 5}, layout, owned=2, shared=True) == (300, 150, 300, 150)
    assert allocations({"num_experts": 5}, layout, owned=3, shared=False) == (300, 150, 300, 150)


@pytest.mark.parametrize("active", [False, True], ids=["idle-control", "accepted-native-borrow"])
def test_follower_close_waits_for_accepted_control_or_native_borrow(active):
    """Injected CPU operations prove lease ordering; real NCCL fault execution still needs two GPUs."""
    entered, released = threading.Event(), threading.Event()
    errors = []
    session = _session_owner(SimpleNamespace(close=lambda **kw: None))
    session.rank, session.sequence, session.keys = 1, 0, ["bank"]
    session.store = SimpleNamespace(set=lambda *a: None, delete_key=lambda *a: None)
    def blocked():
        entered.set()
        assert session.abort_requested.wait(3)
        released.set()
        raise RuntimeError("borrow interrupted")
    def receive(*args, **kwargs):
        return b"[0,0,1,0]" if active else blocked()
    session._receive = receive
    if active:
        session.x = [None]
        session.comm.broadcast = lambda value: blocked()
    session.cache = SimpleNamespace(close=lambda: released.is_set() or pytest.fail("premature cache retirement"))
    def work():
        try:
            session.follow()
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=work)
    thread.start()
    assert entered.wait(3)
    session.close(abort=True)
    thread.join(3)
    assert not thread.is_alive() and session.retired and errors == ["borrow interrupted"]


def test_rank_commands_preserve_owned_harness_entrypoint_and_argument_boundaries(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/approved/cuda/compat")
    commands, envs = rank_commands(["GPU-leader", "GPU-worker"],
                                   ["serve", "/models/with spaces", "--vram-experts", "auto"],
                                   "341,171", 29551, entrypoint="/owned/serving harness.py")
    assert commands[0][:5] == [sys.executable, "/owned/serving harness.py", "serve", "/models/with spaces",
                               "--vram-experts"]
    assert commands[0][-7:] == ["341,171", "--rank", "0", "--master", "127.0.0.1", "--master-port", "29551"]
    assert envs[0]["CUDA_VISIBLE_DEVICES"] == "GPU-leader"
    assert envs[1]["CUDA_VISIBLE_DEVICES"] == "GPU-worker"
    assert all(e["LD_LIBRARY_PATH"] == "/approved/cuda/compat" for e in envs)
    ordinary, _ = rank_commands(["0"], ["serve", "fixture"], "512", 29551)
    assert ordinary[0][:4] == [sys.executable, "-m", "tensorfold", "serve"]
