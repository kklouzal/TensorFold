# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Actual standard logical identities and full rendered-ID cache-domain binding.

Controlled CPU model step only; this does not qualify model/template quality.
"""
import threading

import pytest
pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.retained_catalog import LOCAL_PRINCIPAL, LogicalReference
from tensorfold.server.api_records import ApiRecordStore
from test_api_records import begin, limits, response
from test_live_scheduler_cohort import running_cohort


def test_noauth_stored_response_full_history_chat_reuses_ram_and_auth_domain_does_not(tmp_path, monkeypatch):
    with running_cohort(pool_size=1) as current:
        scheduler = current.scheduler
        catalog = current.catalogs[0]
        committed = threading.Event()
        actual = catalog.apply_logical_events
        def published(events):
            result = actual(events)
            if any(event["operation"] == "committed" for event in events):
                committed.set()  # Visibility of real actor publication, not a sleep or fake ready flag.
            return result
        monkeypatch.setattr(catalog, "apply_logical_events", published)
        store = ApiRecordStore(tmp_path / "logical.sqlite", limits=limits(),
            reference_unavailable=catalog.logical_reference_unavailable,
            reference_committed=catalog.logical_reference_committed)
        try:
            initial, first_output = [1] * 129, []
            with begin(store, "resp_local", principal=LOCAL_PRINCIPAL) as operation, operation.execution_scope():
                first = scheduler.submit(initial, 2, None, True, lambda values: first_output.extend(values))
                operation.commit(response("resp_local"))
            assert first["cached"] == 0 and first_output == [9, 10]
            assert committed.wait(10)
            full_history = [*initial, *first_output, 3]
            cached = scheduler.submit(full_history, 1, None, True, lambda _values: None)
            assert cached["cached"] == 130
            parent = catalog.lookup(LogicalReference(LOCAL_PRINCIPAL, "response:resp_local", 1))
            assert parent is not None and parent.phase == "ram" and parent.role == "logical"
            assert scheduler._request_limit.current_principal() is None  # No fabricated authenticated quota identity.
            with scheduler._request_limit.principal("opaque-authenticated-other"):
                isolated = scheduler.submit(full_history, 1, None, True, lambda _values: None)
            assert isolated["cached"] == 0
        finally:
            store.close()


def test_ordinary_request_scoped_prefix_has_no_client_alias_and_uses_same_ram_pipeline():
    with running_cohort(pool_size=1) as current:
        scheduler = current.scheduler
        initial, first_output = [1] * 129, []
        first = scheduler.submit(initial, 2, None, True, lambda values: first_output.extend(values))
        assert first["cached"] == 0
        assert all(not catalog._references for catalog in current.catalogs)
        cached = scheduler.submit([*initial, *first_output, 3], 1, None, True, lambda _values: None)
        assert cached["cached"] == 130
        assert all(not catalog._references for catalog in current.catalogs)  # Derived cache made no proprietary logical handle.
