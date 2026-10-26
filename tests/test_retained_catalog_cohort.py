# ruff: noqa: E402 - optional torch capability must be checked before CUDA-family imports.
"""Real CPU Store clone/chunk/ACK protocol over actual eight-rank State views.

All ranks run in this test process; this proves host protocol and logical owner
relocation, not CUDA/NCCL/process isolation or V100 service qualification.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")
from torch.distributed import TCPStore

from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.retained_catalog import LogicalReference, make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.cuda.state_retention import state_inventory
from test_flashnext_history_placement import _seed_zero_indexer_prefix
from test_sm70_rotorquant_state import model


def _cohort_states():
    server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=10), wait_for_workers=False)
    baseline = server.num_keys()
    worlds, sources, destinations = [], [], []
    for rank in range(8):
        client = TCPStore("127.0.0.1", server.port, None, False, timedelta(seconds=10))
        comm = SimpleNamespace(rank=rank, world=8, store=client)
        weights = model(8, mtp=False)
        weights.meta.update(full=SimpleNamespace(kv_heads=2), rank=rank, execution_provider="sm70",
                            history_layout="unique", attention_degree=2)
        source, destination = [State(weights, 257, 4, "fp16", limit=257, slot=slot) for slot in (0, 5)]
        _seed_zero_indexer_prefix(source, 129)
        for ordinal, cache in enumerate(source.kc):
            for _consumed, offset, count in cache._spans(0, source.pos):
                cache.k[offset:offset + count].fill_(ordinal * 8 + rank + 1)
                cache.v[offset:offset + count].fill_(-ordinal * 8 - rank - 1)
        weights.comm = comm
        source.history_comm = destination.history_comm = comm
        worlds.append(weights)
        sources.append(source)
        destinations.append(destination)
    return server, baseline, worlds, sources, destinations


def test_real_eight_rank_cpu_transfer_retires_keys_and_relocates_packed_histories():
    server, baseline, worlds, sources, destinations = _cohort_states()
    inventories = tuple(state_inventory(state, recovery_versions=0) for state in sources)
    policy = RetentionPolicy("ram", 1, 30, 2 << 30, 128 << 10, 2 * 8 * (128 << 10), 0)
    owners = [None] * 8

    def construct(rank):
        return make_retention_catalog(policy, worlds[rank], [sources[rank], destinations[rank]], prefix_limit=2, recovery_versions=0,
                                      register=lambda owner: owners.__setitem__(rank, owner))

    with ThreadPoolExecutor(max_workers=8) as pool:
        catalogs = list(pool.map(construct, range(8)))
    handles, requests = [], []
    try:
        for rank, catalog in enumerate(catalogs):
            handle = catalog.reserve(LogicalReference("p", "response:cohort", 1), future_bytes=catalog.future_claim())
            catalog.capture_idle(Stream([1] * 129, 8, draft=False, st=sources[rank]), handle=handle)
            handles.append(handle)
            catalog.suspend(handle)
        assert all(handle.transfer.retired.wait(15) for handle in handles)
        # A legal publication lag on one rank must leave every actor's prefix
        # and protected-slot decisions unchanged until common settlement.
        handles[3].transfer.retired.clear()
        for catalog, state in zip(catalogs, sources):
            catalog.prefix_entries = [([1], state, state.snapshot(), None)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda catalog: catalog.settle_ready(), catalogs))
        assert all(handle.phase == "saving" for handle in handles)
        assert all(catalog.protected_states() == (state,) and len(catalog.prefix_entries) == 1
                   for catalog, state in zip(catalogs, sources))
        handles[3].transfer.retired.set()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda catalog: catalog.settle_ready(), catalogs))
        assert all(catalog.saved(handle) for catalog, handle in zip(catalogs, handles))
        assert all(not catalog.prefix_entries and not catalog.protected_states() for catalog in catalogs)
        for catalog, handle, destination in zip(catalogs, handles, destinations):
            request = catalog.borrow(handle)
            requests.append(request)
            request.claim(destination, stream=Stream([1] * 129, 8, draft=False))
        assert all(handle.transfer.retired.wait(15) for handle in handles)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda catalog: catalog.settle_ready(), catalogs))
        assert all(catalog.restored(request) is not None for catalog, request in zip(catalogs, requests))
        # Compare actual logical packed history, including newly owned rows.
        expected = {part.name: part.tensor for item in inventories for part in item.parts
                    if part.name.startswith("history.")}
        actual = {part.name: part.tensor for state in destinations
                  for part in state_inventory(state, recovery_versions=0).parts if part.name.startswith("history.")}
        assert expected.keys() == actual.keys()
        assert all(torch.equal(value, actual[name]) for name, value in expected.items())
        assert server.num_keys() == baseline
    finally:
        for request in requests:
            request.close()
        for catalog, handle in zip(catalogs, handles):
            if handle.transfer is not None and handle.transfer.failure is not None:
                # Diagnostic failure retains authoritative payload; retire only
                # the already-failed CPU callback, never force-close its data.
                catalog._worker.drain(catalog._thread, timeout=10, cancel_unentered=False)
            else:
                catalog.release_owner(handle)
                catalog.close(timeout=10)


@pytest.mark.parametrize("failure", ["preflight", "clone"])
def test_one_rank_constructor_refusal_is_unanimous_and_all_partial_owners_retire(failure):
    server, baseline, worlds, sources, _destinations = _cohort_states()
    if failure == "preflight":
        sources[3].target_tentative_length += 1
    else:
        class FailedClone:
            def __init__(self, store):
                self.store = store

            def clone(self):
                raise OSError("injected one-rank connection acquisition failure")

            def __getattr__(self, name):
                return getattr(self.store, name)

        worlds[3].comm.store = FailedClone(worlds[3].comm.store)
    policy = RetentionPolicy("ram", 1, 10, 2 << 30, 128 << 10, 2 * 8 * (128 << 10), 0)
    owners = [None] * 8

    def attempt(rank):
        try:
            make_retention_catalog(policy, worlds[rank], [sources[rank]], prefix_limit=0, recovery_versions=0,
                                   register=lambda owner: owners.__setitem__(rank, owner))
        except BaseException as error:
            return error
        pytest.fail("one-rank constructor refusal was lost")

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))
    assert all(isinstance(error, ValueError) and "rank 3" in str(error) for error in outcomes)
    assert all(owner is not None and owner._worker is None and owner._store is None and owner._ram is None
               for owner in owners)
    assert server.num_keys() == baseline
    for owner in owners:
        owner.close(timeout=0)
