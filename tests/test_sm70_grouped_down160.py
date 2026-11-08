"""Preparation/layout contracts; no CPU substitute for WMMA arithmetic."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tensorfold.cuda.sm70.gguf_ops import grouped_workspace_words


@pytest.mark.parametrize("pairs", [0, 1, 15, 16, 17, 31, 32, 33, 352, 2816, 65535])
@pytest.mark.parametrize("experts", [0, 1, 2, 256, 512])
def test_packed_layout_has_space_for_every_ragged_expert_partition(pairs, experts):
    # Extremal independent partitions: concentrated, balanced and singleton.
    layouts = [[pairs] + [0] * (experts - 1)] if experts else [[0]]
    if experts:
        q, r = divmod(pairs, experts)
        layouts.append([q + (i < r) for i in range(experts)])
        singles = min(pairs, experts - 1)
        layouts.append([1] * singles + [pairs - singles] + [0] * (experts - singles - 1))
    cap = min(pairs, (pairs + 15) // 16 + max(0, experts - 1))
    assert grouped_workspace_words(pairs, experts) == 2 * pairs + 2 * experts + 3 * cap + 2
    for counts in layouts:
        if not experts:
            continue
        counts = [n for n in counts if n]
        tiles = sum((n + 15) // 16 for n in counts)
        assert tiles <= cap
        spans = [(e, start, min(16, count - start)) for e, count in enumerate(counts)
                 for start in range(0, count, 16)]
        assert sum(n for _, _, n in spans) == pairs
        assert all(1 <= n <= 16 for _, _, n in spans)


@pytest.mark.parametrize("pairs,experts", [(-1, 2), (65536, 2), (True, 2), (1, -1), (1, 513), (1, True)])
def test_layout_rejects_unbounded_or_wrong_domains(pairs, experts):
    with pytest.raises(ValueError):
        grouped_workspace_words(pairs, experts)


def _prepare():
    import torch
    from types import SimpleNamespace as NS
    from tensorfold.families.qwen4_exp.cuda.owned_experts import Plan, POLICY
    from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition

    plan = Plan(ExpertPartition(8, 0, 512, 640, 4), 2560, 11, 32, 256, True)
    owners = []
    for rows, prefill, dtype in [(32, False, torch.float16), (256, True, torch.float16), (32, False, torch.float32)]:
        b = NS(rows=rows, prefill=prefill,
               moe=NS(act=torch.empty((rows, 11, 160), dtype=torch.float16),
                      y=torch.empty((rows, 11, 2560), dtype=dtype),
                      pick=torch.empty((rows, 11), dtype=torch.int32)))
        b.owned_experts = plan.arena(b)
        owners.append(b)
    receipt = plan.reconcile(NS(buf=owners[0], pbuf=owners[1], mbuf=owners[2]))
    assert receipt['local_arena_bytes'] + receipt['grouping_arena_bytes'] <= receipt['contained_routing_reserve_bytes']
    assert receipt['materialized_grouping_arena_bytes'] == receipt['grouping_arena_bytes']
    if receipt['grouped_down'] == 'grouped-v1':
        assert plan.identity[-1] == 'grouped-v1'
        assert receipt['grouping_arena_bytes'] > 0
    else:
        assert type(plan.identity[-1]) is bool
        assert receipt['grouping_arena_bytes'] == 0
    arena = owners[0].owned_experts
    calls = []
    bank = NS(_owned_expert_plan=plan, implementation=POLICY,
              forward_partitioned_native_candidate=lambda *a, **kw: calls.append(kw))
    picks = owners[0].moe.pick[:3]
    arena.execute(bank, torch.empty((3, 2560), dtype=torch.float16), picks, arena.output[:3])
    if arena.grouping is not None:
        view = calls[0]['grouping']
        assert view.numel() == grouped_workspace_words(33, 256)
        assert view.untyped_storage().data_ptr() == arena.grouping.untyped_storage().data_ptr()
    else:
        assert 'grouping' not in calls[0]
    print(json.dumps(receipt))


@pytest.mark.parametrize("policy", ["inherit", "grouped-v1", "bad"])
def test_process_fixed_policy_prepares_exact_paid_disjoint_role_scratch(policy):
    env = dict(os.environ, TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN=policy, OMP_NUM_THREADS='1')
    r = subprocess.run([sys.executable, '-c', 'from tests.test_sm70_grouped_down160 import _prepare; _prepare()'],
                       cwd=Path(__file__).parents[1], env=env, capture_output=True, text=True, timeout=30)
    if policy == 'bad':
        assert r.returncode != 0 and 'TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN requires' in r.stderr
    else:
        assert r.returncode == 0, r.stdout + r.stderr
        assert json.loads(r.stdout.splitlines()[-1])['grouped_down'] == policy
