# ruff: noqa: E402 - maintained CUDA orchestration requires optional Torch.
"""Managed profile replacement preserves protected idle claims without shutdown.

These exercise real ServingProfiles/TP adapter orchestration with resource
probes; actual rank transfers, model loading and V100 memory remain unqualified.
"""

import threading
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.serving_runtime import ServingProfiles
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder


class Catalog:
    def __init__(self, events, *, protected):
        self.events, self.protected = events, protected
        self.optional = True

    def drop_optional_prefixes(self):
        self.events.append("drop optional")
        self.optional = False

    @property
    def has_retained(self):
        return self.optional or self.protected

    def record(self):
        return {"protected_idle": self.protected, "resident_gpu_states": 0}

    def begin_shutdown(self):
        pytest.fail("profile replacement may not terminally shut down retained state")


def runtime(*, protected, parallel):
    events = []
    catalog = Catalog(events, protected=protected)
    owner = object.__new__(ServingProfiles)
    owner.lock = threading.RLock()
    owner.pending = NS(record=lambda: {"profile": "replacement"})
    owner.status = {"stage": "draining", "pending": "replacement"}
    owner.gate = NS(borrowers=0, available=True)

    def replace():
        assert not catalog.has_retained
        events.append("replace")
        return {"ready": True}

    adapter = None
    if parallel:
        adapter = object.__new__(TensorParallelDecoder)
        # The actual public adapter method delegates to the same coordinated
        # retention actor, rather than changing rank0 state directly.
        def boundary(*, idle, drop_optional):
            assert idle and drop_optional
            events.append("coordinated boundary")
            catalog.drop_optional_prefixes()
            return {"drained": not catalog.has_retained, "changed": True}
        adapter.retention_boundary = boundary
        adapter.replace_serving_profile = replace
    engine = NS(retention_catalog=catalog, tp_decoder=adapter,
                multi=NS(live=lambda: 0, _release_retained_states=lambda: events.append("release states")))
    owner._engine = lambda: engine  # Keep this explicitly owned resource probe alive through each boundary.
    owner.replace = replace
    scheduler = NS(boxes={}, held=None, waiting=NS(qsize=lambda: 0))
    return owner, scheduler, catalog, events


@pytest.mark.parametrize("parallel", [False, True])
def test_zero_gpu_protected_claim_blocks_replacement_until_its_own_release(parallel):
    owner, scheduler, catalog, events = runtime(protected=True, parallel=parallel)
    owner._boundary(scheduler)
    assert owner.status["stage"] == "draining" and owner.pending is not None
    assert owner.status["blocking_retention"] == {"protected_idle": True, "resident_gpu_states": 0}
    assert not catalog.optional and "replace" not in events
    if parallel:
        assert events == ["coordinated boundary", "drop optional"]
    catalog.protected = False  # Explicit owner release, independent of GPU residency.
    owner._boundary(scheduler)
    assert events[-1] == "replace" and owner.pending is None
    if parallel:
        assert events[-3:] == ["coordinated boundary", "drop optional", "replace"]
    assert owner.status["stage"] == "ready" and "blocking_retention" not in owner.status


@pytest.mark.parametrize("parallel", [False, True])
def test_optional_cache_drop_waits_for_prepared_profile_borrower(parallel):
    owner, scheduler, catalog, events = runtime(protected=False, parallel=parallel)
    owner.gate.borrowers = 1
    owner._boundary(scheduler)
    assert catalog.optional and events == []
    owner.gate.borrowers = 0
    owner._boundary(scheduler)
    assert not catalog.optional and events[-1] == "replace" and owner.pending is None
    if parallel:
        assert events == ["coordinated boundary", "drop optional", "replace"]
