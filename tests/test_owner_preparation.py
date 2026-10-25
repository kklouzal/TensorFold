"""Startup ownership/cohort controls with explicit CPU resource counters."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import threading

import pytest

from tensorfold.families.qwen4_exp.cuda.owner_execution import GRAPH_FAMILY, KERNEL_FAMILY, UNCAPTURED, row_buckets
from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation, parse_preparation


def specification(*, trial=False):
    return {"kernel_family": KERNEL_FAMILY, "graph_family": UNCAPTURED,
            "graph_budget_bytes": 0, "trial": trial}


def test_absent_preparation_is_disabled_and_explicit_trial_is_not_a_production_identity_change():
    assert parse_preparation(None) is None
    first = OwnerPreparation(specification(trial=False), role_rows={"decode": 4, "prefill": 8}, manifest_sha256="a" * 64)
    second = OwnerPreparation(specification(trial=True), role_rows={"prefill": 8, "decode": 4}, manifest_sha256="b" * 64)
    assert first.identity() == second.identity()
    assert list(first.role_rows) == list(second.role_rows) == ["decode", "prefill"]


@pytest.mark.parametrize("change", [{"trial": 1}, {"kernel_family": "unknown"},
                                    {"graph_family": GRAPH_FAMILY}, {"graph_budget_bytes": True},
                                    {"row_buckets": {"other": [1]}}, {"row_buckets": {"decode": [1, 1]}}])
def test_malformed_preparation_refuses_before_any_resource_initialization(change):
    with pytest.raises(ValueError):
        parse_preparation({**specification(), **change})


def test_role_bucket_endpoint_must_match_the_prepared_resource_envelope():
    with pytest.raises(ValueError, match="endpoint"):
        OwnerPreparation({**specification(), "row_buckets": {"decode": [1, 2]}},
                         role_rows={"decode": 4, "prefill": 8}, manifest_sha256="a" * 64)


def test_nonreusable_charge_separates_private_spare_from_ordinary_allocator_spare():
    prepared = OwnerPreparation(specification(), role_rows={"decode": 4, "prefill": 8}, manifest_sha256="a" * 64)
    prepared.before = {"physical_free_bytes": 1000, "torch_allocated_bytes": 0, "torch_reserved_bytes": 0}
    prepared.after = {"physical_free_bytes": 760, "torch_allocated_bytes": 160, "torch_reserved_bytes": 200}
    prepared.graphs = {"decode": NS(private_spare_bytes=16)}
    assert prepared.committed_bytes() == 216  # 160 active +16 private inactive +40 outside Torch.
    prepared.graphs["decode"].private_spare_bytes = None
    assert prepared.committed_bytes() is None


@pytest.mark.parametrize("private,limit,outside,expected", [(16, None, 48, 148),
                                                         (16, 200, 48, 120),
                                                         (None, None, None, 100),
                                                         (None, 200, None, 0)])
def test_live_grant_never_credits_unknown_private_pool_or_omits_outside_cap_commitment(
        monkeypatch, private, limit, outside, expected):
    import torch
    from tensorfold.cuda import capacity, kv_arena

    prepared = OwnerPreparation(specification(), role_rows={"decode": 4, "prefill": 8}, manifest_sha256="a" * 64)
    prepared.private_spare_bytes, prepared.outside_active_bytes = private, outside
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 32)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 96)
    monkeypatch.setattr(capacity, "available_bytes", lambda torch: 100)
    monkeypatch.setattr(capacity, "cuda_limit_bytes", lambda: limit)
    monkeypatch.setattr(kv_arena, "mapped_bytes", lambda: 0)
    assert prepared.live_bytes() == expected


@pytest.mark.parametrize("trial", [False, True])
def test_eight_ranks_prepare_once_adopt_the_same_owners_and_retire_metadata(monkeypatch, trial):
    import torch
    from tensorfold.cuda import capacity
    from tensorfold.families.qwen4_exp.cuda import history_owner
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    monkeypatch.delenv(capacity.LIMIT_ENV, raising=False)
    lock, values = threading.Condition(), {}

    class Store:
        def set(self, key, value):
            with lock:
                values[key] = value
                lock.notify_all()

        def get(self, key):
            with lock:
                return values[key]

        def wait(self, keys, timeout):
            with lock:
                if not lock.wait_for(lambda: all(key in values for key in keys), timeout=5):
                    raise TimeoutError("preparation cohort did not publish a dependent phase")

        def delete_key(self, key):
            with lock:
                return values.pop(key, None) is not None

    store = Store()
    memory = [{"free": 1 << 30, "active": 0, "reserved": 0} for _ in range(8)]
    created = []

    class Owner:
        def __init__(self, rank, rows, prefill):
            self.rank, self.rows, self.prefill = rank, rows, prefill
            self.bytes = 0
            self.closed = False
            self.warmed = []

        def allocate(self, count):
            self.bytes += count
            memory[self.rank]["free"] -= count
            memory[self.rank]["active"] += count
            memory[self.rank]["reserved"] += count

        def prepare_native(self, *, pair, page_tokens, buckets):
            self.native_buckets = row_buckets(self.rows, buckets)
            self.allocate(self.rows * 228 + 4)

        def allocation_record(self):
            return {"rank": self.rank, "rows": self.rows, "prefill": self.prefill,
                    "device_tensor_bytes": self.bytes, "host_staging_bytes": self.rows * 128}

        def clear_native_inputs(self, rows):
            assert rows in self.native_buckets

        def wait_native_inputs(self):
            pass

        def native_run(self, rows):
            self.warmed.append(rows)

        def bind_native_states(self, states):
            self.states = states

        def close(self):
            memory[self.rank]["free"] += self.bytes
            memory[self.rank]["active"] -= self.bytes
            memory[self.rank]["reserved"] -= self.bytes
            self.bytes, self.closed = 0, True

    def factory(text, *, rows, prefill, rank, register, **kwargs):
        owner = Owner(rank, rows, prefill)
        register(owner)
        created.append(owner)
        owner.allocate(rows * 1000)  # Independent fake allocator occupancy.
        return owner

    monkeypatch.setattr(history_owner.OwnerBuffers, "from_text", factory)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (memory[device]["free"], 1 << 30))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: memory[device]["active"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: memory[device]["reserved"])
    text = {"num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
            "indexer_budget": 8, "indexer_compress_ratio": 4}

    def prepare(rank):
        engine = NS(rank=rank, tp=8, comm=NS(world=8, rank=rank, store=store), multi=None, _calls={})
        # Insertion order differs deliberately; the operation protocol must not.
        roles = {"decode": 4, "mtp": 4, "prefill": 8} if rank % 2 else {"prefill": 8, "mtp": 4, "decode": 4}
        prepared = OwnerPreparation(specification(trial=trial), role_rows=roles, manifest_sha256="a" * 64)
        prepared.prepare(engine, text, get_pair("int8"), page_tokens=128, device=rank)
        return prepared

    with ThreadPoolExecutor(max_workers=8) as workers:
        prepared = list(workers.map(prepare, range(8)))
    assert values == {} and len(created) == 24
    assert all(item.identity() == prepared[0].identity() for item in prepared)
    for rank, item in enumerate(prepared):
        assert item.committed_bytes() == sum(owner.bytes for owner in item.workspaces.values())
        assert all(owner.warmed == list(owner.native_buckets) for owner in item.workspaces.values())
        buffers = {role: NS(history_owner=owner, rows=owner.rows) for role, owner in item.workspaces.items()}
        decoder = NS(states=(object(),), buf=buffers["decode"], mbuf=buffers["mtp"], pbuf=buffers["prefill"],
                     controller=AdaptiveController(slots=2, row_budget=4, proposal_limit=1, prefill_tokens=8))
        decoder.apply_execution_family = lambda d=decoder: MultiDecoder.apply_execution_family(d)
        item.adopt(NS(meta={"owner_workspaces": item.workspaces}), decoder)
        assert decoder.buf.owner_kernel_family == (KERNEL_FAMILY if trial else "torch-reference")
        assert len(created) == 24  # Adoption did not create a second serving allocation.
        item.close()
        assert memory[rank] == {"free": 1 << 30, "active": 0, "reserved": 0}
