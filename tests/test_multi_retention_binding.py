# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Actual catalog/State prefix authority at the existing Multi caller boundary."""

from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")

from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import prefixes
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.prompt_plan import POINTS
from tensorfold.families.qwen4_exp.cuda.retained_catalog import LogicalReference, make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.state_retention import state_inventory

pytest_plugins = ("test_state_retention",)


def caller(catalog):
    value = object.__new__(MultiDecoder)
    value.w, value.states = catalog.w, catalog.states
    value.streams, value.filling, value._kept = {}, [], []
    value.retention_catalog, value.keep = None, 2
    value.controller = None
    return value


def test_catalog_becomes_only_prefix_authority_and_protected_state_is_not_idle_victim(materialized):
    weights = NS(meta={"rank": 0}, comm=None)
    states = [materialized, materialized.clone()]
    future = state_inventory(materialized, recovery_versions=POINTS).future_bytes
    policy = RetentionPolicy("ram", 2, 10, future * 6 + (32 << 20), 37, 74, 0)
    catalog = make_retention_catalog(policy, weights, states, prefix_limit=2, recovery_versions=POINTS)
    decoder = caller(catalog)
    handle = None
    try:
        decoder.bind_retention_catalog(catalog)
        assert decoder.retention_catalog is catalog and decoder._kept is None
        entry = ([3, 7, 11], materialized, materialized.snapshot(), None)
        decoder.kept = [entry]
        assert catalog.prefix_entries == [entry] and decoder.kept[0] is entry
        handle = catalog.reserve(LogicalReference("trusted", "response:one", 1), future_bytes=catalog.future_claim())
        catalog.capture_idle(Stream([3, 7, 11, 9], 8, draft=False, st=materialized), handle=handle)
        decoder.streams[1] = Stream([1], 1, st=states[1])
        busy = decoder._busy()
        assert busy == {id(state) for state in states}
        record = decoder.history_record()
        assert record["active_request_slots"] == 1 and record["intentional_retained_sessions"] == 1
        assert record["retained_gpu_slots"] == 1
        assert not record["slots"][0]["request_active"] and record["slots"][0]["retained_gpu"]
        assert prefixes._idle(decoder, busy) is None
        decoder._drop_kept(materialized)
        assert catalog.prefix_entries == []
        assert materialized in catalog.protected_states()  # Cache removal never deletes its required owner.
    finally:
        if handle is not None:
            catalog.release_owner(handle)
        catalog.close(timeout=3)


def test_binding_rejects_unmatched_geometry_before_changing_authority(materialized):
    weights = NS(meta={"rank": 0}, comm=None)
    catalog = make_retention_catalog(None, weights, [materialized], prefix_limit=2, recovery_versions=1)
    decoder = caller(catalog)
    try:
        with pytest.raises(ValueError, match="prefix/recovery geometry"):
            decoder.bind_retention_catalog(catalog)
        assert decoder.retention_catalog is None and decoder.kept == [] and catalog.prefix_entries == []
    finally:
        catalog.close(timeout=3)
