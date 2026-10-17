"""Expert ownership, bounded work wire and rank-process failure contracts; no GPU emulation."""
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda.expert_launch import supervise
from tensorfold.cuda.expert_parallel import Ownership, ownership
from tensorfold.families.qwen4_exp.cuda.expert_parallel import Follower, Session, decode_frame


def test_asymmetric_shards_cover_every_expert_exactly_once():
    owners = [ownership(2, "341,171", rank, 512) for rank in range(2)]
    assert [list(o.ids) for o in owners] == [list(range(341)), list(range(341, 512))]
    assert ownership(1, None, 0, 512).counts == (512,)


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
