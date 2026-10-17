"""Expert ownership, bounded work wire and rank-process failure contracts; no GPU emulation."""
import os
import sys
import time

import pytest

from tensorfold.cuda.expert_launch import supervise
from tensorfold.cuda.expert_parallel import Ownership, ownership
from tensorfold.families.qwen4_exp.cuda.expert_parallel import decode_frame


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
