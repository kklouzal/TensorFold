import os
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from tensorfold.families.deepseek_v4.cuda.expert_placement import (
    agree_placement,
    checkpoint_placement_identity,
    encode_placement,
    read_placement,
)


class Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def get(self, key):
        return self.values[key]


@pytest.mark.parametrize("etp", [1, 2, 4])
def test_leader_reads_one_checkpoint_bound_placement_for_all_ranks(tmp_path, etp):
    shard = tmp_path / "weights.gguf"
    shard.write_bytes(b"existing bytes")
    checkpoint = checkpoint_placement_identity([shard], {"model_type": "deepseek_v41"})
    plan = tmp_path / "experts.json"
    rows = (tuple(reversed(range(384))), tuple(range(384)))
    plan.write_bytes(encode_placement(rows, checkpoint=checkpoint, etp=etp))
    opts = dict(store=Store(), checkpoint=checkpoint, layers=2, experts=384, etp=etp)
    assert agree_placement(plan, rank=0, **opts) == rows
    plan.unlink()
    for rank in range(1, 8):
        assert agree_placement(plan, rank=rank, **opts) == rows


def test_placement_rejects_duplicate_or_foreign_experts(tmp_path):
    plan = tmp_path / "experts.json"
    rows = (tuple(range(383)) + (382,),)
    plan.write_bytes(encode_placement(rows, checkpoint="checkpoint", etp=4))
    with pytest.raises(ValueError, match="each of its 384 experts exactly once"):
        read_placement(plan, checkpoint="checkpoint", layers=1, experts=384, etp=4)
    plan.write_bytes(encode_placement((tuple(range(384)),), checkpoint="checkpoint", etp=2))
    with pytest.raises(ValueError, match="different checkpoint or ETP degree"):
        read_placement(plan, checkpoint="checkpoint", layers=1, experts=384, etp=4)


def test_leader_reports_invalid_plan_to_followers(tmp_path):
    plan = tmp_path / "experts.json"
    plan.write_bytes(b"{broken")
    opts = dict(store=Store(), checkpoint="checkpoint", layers=1, experts=384, etp=4)
    for rank in (0, 1):
        with pytest.raises(ValueError, match="invalid DeepSeek expert placement JSON"):
            agree_placement(plan, rank=rank, **opts)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO validation requires POSIX")
def test_placement_does_not_wait_on_a_fifo(tmp_path):
    plan = tmp_path / "placement.fifo"
    os.mkfifo(plan)
    with pytest.raises(ValueError, match="regular file"):
        read_placement(plan, checkpoint="checkpoint", layers=1, experts=384, etp=4)


def test_failed_history_does_not_announce_a_forward():
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    write = Future()
    write.set_exception(OSError("history write failed"))

    def sample(phase, *, may_copy):
        assert phase == 0 and may_copy
        return write.result()

    def announce(*args):
        pytest.fail(f"Announced TP8 forward with failed expert history: {args}")

    engine = SimpleNamespace(expert_observer=SimpleNamespace(sample=sample), part=None, announce=announce)
    with pytest.raises(OSError, match="history write failed"):
        DS41Engine.step(engine, [(0, 0, 1)], [42], [0])
