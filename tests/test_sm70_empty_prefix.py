"""Component-only CPU policy and reference-owner empty-prefix controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("policy,valid", [("inherit", True), ("cpu-empty-v1", True), ("", False), ("on", False)])
def test_process_fixed_policy_refuses_unknown_values(policy, valid):
    env = dict(os.environ, TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX=policy)
    result = subprocess.run([sys.executable, "-c", "import tensorfold.families.qwen4_exp.cuda.history_owner"],
                            env=env, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is valid, result.stderr
    if not valid:
        assert "TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX requires inherit or cpu-empty-v1" in result.stderr


@pytest.mark.torch
@pytest.mark.parametrize("format_name", ["int8", "rotorquant-iso4-sm70-v1"])
def test_current_cpu_table_not_arena_capacity_or_retained_identity(format_name):
    from tensorfold.families.qwen4_exp.cuda.history_owner import allocate_target_caches, _first_page_empty
    from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    for slot in (0, 1, 7):
        for degree in (1, 2, 4):
            placement = HistoryPlacement(8, 2, 12, degree=degree)
            for rank in range(8):
                caches = allocate_target_caches(257, 256, "cpu", pair=get_pair(format_name),
                                                placement=placement, slot=slot, rank=rank)
                for cache in caches:
                    expected = cache.offsets[0] < 0
                    for end in (1, 64, 127, 128):
                        assert _first_page_empty(cache, end) is expected
                    for end in (0, -1, 129, 258, True):
                        assert not _first_page_empty(cache, end)
                    # Clone and resize retain their actual page tables; an old
                    # object's identity or arena-wide capacity is no authority.
                    if cache.layer == 0:
                        for other in (cache.clone(), cache.resized(513, 128)):
                            assert _first_page_empty(other, 128) is expected
    assert not _first_page_empty(NS(offsets=(-1,)), 1)


@pytest.mark.torch
def test_capture_and_unqualified_families_keep_device_table_path(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import history_owner as h

    weights = NS(meta={"execution_provider": "sm70"}, device=h.torch.device("cuda:0"))
    buffers = NS(owner_graph_family="uncaptured")
    monkeypatch.setattr(h.torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not h._empty_prefix_reference_allowed(weights, buffers)
    monkeypatch.setattr(h.torch.cuda, "is_current_stream_capturing", lambda: False)
    assert h._empty_prefix_reference_allowed(weights, buffers)
    buffers.owner_graph_family = "owner-attention-v1"
    assert not h._empty_prefix_reference_allowed(weights, buffers)
    buffers.owner_graph_family = "uncaptured"
    weights.meta["execution_provider"] = "other"
    assert not h._empty_prefix_reference_allowed(weights, buffers)


def _exercise_layer():
    """Fresh interpreter: exercise actual eight-owner reference caller/oracle."""
    import torch
    from tensorfold.families.qwen4_exp.cuda import history_owner as h
    from tests.test_flashnext_history_placement import (
        test_real_owner_layer_routes_both_tp_head_groups_with_batched_collectives as exercise,
        test_compact_unowned_operands_are_zeroed_before_metadata_validation as invisible,
    )
    torch.set_num_threads(1)
    visited = []
    zero_rows = []
    initialize = h.OwnerBuffers.__init__
    def inspect_wires(owner, weights, *args, **kwargs):
        initialize(owner, weights, *args, **kwargs)
        gather = weights.comm.all_gather
        calls, empty = 0, None
        def checked(send, recv):
            nonlocal calls, empty
            calls += 1
            if calls == 3:
                empty = torch.isneginf(send[..., 0]).all(-1)
                assert not torch.count_nonzero(send[..., 1][empty].contiguous().view(torch.uint8))
            elif calls == 4:
                assert not torch.count_nonzero(send[empty].contiguous().view(torch.uint8))
                zero_rows.append(int(empty.sum()))
            return gather(send, recv)
        weights.comm.all_gather = checked
    h.OwnerBuffers.__init__ = inspect_wires
    original = h._attention_tiles
    def observe(query, cache, selected, visible, scale, **kwargs):
        # CPU diagnostic only; this instrumentation never enters serving.
        ends = selected[visible].tolist()
        if ends and max(ends) < 128 and cache.offsets[0] < 0:
            visited.append(cache.layer)
        yield from original(query, cache, selected, visible, scale, **kwargs)
    h._attention_tiles = observe
    # Original full-history oracle checks all eight outputs and four collective
    # submissions/rank, both head groups and sparse/future-mask selection.
    for degree in (1, 4):
        for fmt in ("int8", "rotorquant-iso4-sm70-v1"):
            for budget, start in ((8, 0), (11, 19), (2048, 126)):
                exercise(degree, fmt, budget, start)
    if h._PRIVATE_EMPTY_PREFIX == "cpu-empty-v1":
        assert not visited, visited
    else:
        assert visited
    assert sum(zero_rows) > 0  # Both exact S and PV publications are positive-zero bytes.
    # Independent existing corruption oracle remains valid. Low-level helpers
    # retain their ordinary contract and do not consult the private policy.
    invisible()
    print(json.dumps({"policy": h._PRIVATE_EMPTY_PREFIX, "empty_tiles": len(visited),
                      "empty_wire_rows": sum(zero_rows)}))


@pytest.mark.torch
@pytest.mark.parametrize("policy", ["inherit", "cpu-empty-v1"])
def test_actual_reference_caller_keeps_full_oracle_and_collective_order(policy):
    env = dict(os.environ, TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX=policy, OMP_NUM_THREADS="1")
    code = "from tests.test_sm70_empty_prefix import _exercise_layer; _exercise_layer()"
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=Path(__file__).parents[1],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout.splitlines()[-1])
    assert record["policy"] == policy
