# ruff: noqa: E402 - the real catalog factory is an optional Torch component.
"""Stable standard-record callbacks across Engine catalog publication/retirement.

These controls use the real durable Store listener API and Engine lifecycle.
Catalog event probes do not claim whole-profile, CUDA or model qualification.
"""

import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.server.api_records import ApiRecordStore
from test_api_records import begin, limits, response


class Catalog:
    def __init__(self):
        self.events = []
        self.has_retained = False
        self.closed = False
        self.failure = None
        self.entered = self.release = None

    def check_health(self):
        if self.failure is not None:
            raise self.failure
        assert not self.closed

    def logical_reference_committed(self, principal, sequence, *, generation):
        self.events.append(("committed", principal, sequence, generation))
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(5)

    def logical_reference_unavailable(self, principal, sequence, *, generation, reason):
        self.events.append(("unavailable", principal, sequence, generation, reason))

    def close(self, *, timeout):
        assert not self.has_retained
        self.closed = True

    def record(self):
        return {"scope": "notification-component-control"}

    def future_claim(self):
        return 0

    def configure_device_resources(self, **limits):
        self.device_limits = limits

    def configure_host_resources(self, **limits):
        self.host_limits = limits


@pytest.fixture
def engine(monkeypatch, tmp_path):
    def metadata_only(owner, *_args, **_kwargs):
        owner.retention_policy = RetentionPolicy("gpu", 1, 30)

    monkeypatch.setattr(FlashNextEngine, "_initialize", metadata_only)
    value = FlashNextEngine(tmp_path / "metadata-only")
    yield value
    value.close()


@pytest.fixture
def store(tmp_path):
    value = ApiRecordStore(tmp_path / "records.sqlite", limits=limits())
    yield value
    value.close()


def publish(engine, catalog):
    engine.retention_catalog = catalog  # Existing partial-resource journal.
    engine._publish_retention_catalog(catalog)


def commit(store, identifier):
    with begin(store, identifier) as operation:
        operation.commit(response(identifier))


def construct_from_canonical_branch(engine, bind, *, native=None, native_host_bytes=0):
    """Execute the real post-warm catalog branch, without a fake model load."""
    from tensorfold.families.qwen4_exp.cuda import engine as source
    from tensorfold.families.qwen4_exp.cuda.retention_resources import retention_header_resources

    parsed = ast.parse(Path(source.__file__).read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "FlashNextEngine")
    initialize = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_initialize")
    branch = next(node for node in ast.walk(initialize) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Compare)
                  and isinstance(node.test.left, ast.Subscript)
                  and isinstance(node.test.left.value, ast.Attribute)
                  and node.test.left.value.attr == "_admission_options")
    engine._admission_options = {"admission_policy": "fifo"}
    engine._retention_device_resources = {"limits": {
        "recovery_pool_bytes": 0, "pending_hc_bytes": 0, "snapshot_bytes": 0}}
    engine.multi = SimpleNamespace(keep=0, states=(), bind_retention_catalog=bind,
                                   recover_failed_slots=lambda: None)
    engine._retention_header_resources = retention_header_resources(engine.retention_policy, 1)
    engine._native_protocol_resources = native
    header = engine._retention_header_resources.global_host_bytes
    engine.capacity_plan = {"host_steady_workspace_bytes_estimate": header,
                            "host_steady_commitment_bytes": header + 123}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(source.__file__), "exec"),
         {"self": engine, "w": None, "native_host_bytes": native_host_bytes,
          "__name__": source.__name__, "__package__": source.__package__})


def test_one_real_store_binding_routes_a_b_unavailable_a_without_rebinding(engine, store):
    first, second, again = Catalog(), Catalog(), Catalog()
    publish(engine, first)
    engine.bind_api_record_store(store)
    original = store._reference_committed, store._reference_unavailable
    commit(store, "resp_a")
    assert first.events == [("committed", "alice", "response:resp_a", 1)]
    engine._close_resources(preserve_transport=True)
    assert first.closed and engine._published_retention_catalog is None
    commit(store, "resp_unavailable")  # Logical storage needs no live computed profile.
    publish(engine, second)
    commit(store, "resp_b")
    store.delete("resp_a", principal="alice")
    assert second.events[0] == ("committed", "alice", "response:resp_b", 1)
    assert second.events[1][:4] == ("unavailable", "alice", "response:resp_a", 1)
    assert len(first.events) == 1
    engine._close_resources(preserve_transport=True)
    publish(engine, again)
    commit(store, "resp_again")
    assert again.events == [("committed", "alice", "response:resp_again", 1)]
    assert (store._reference_committed, store._reference_unavailable) == original
    assert store.get("resp_unavailable", principal="alice")["status"] == "completed"


def test_real_partial_factory_never_becomes_notification_target(engine, store, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.retained_catalog import RetainedCatalog

    first = Catalog()
    publish(engine, first)
    engine.bind_api_record_store(store)
    engine._close_resources(preserve_transport=True)
    primary = MemoryError("partial catalog construction failed")
    partial = []

    def interrupted(owner, *_args, **_kwargs):
        partial.append(owner)
        assert engine.retention_catalog is owner and engine._published_retention_catalog is None
        commit(store, "resp_during_partial")
        raise primary

    with monkeypatch.context() as fault:
        fault.setattr(RetainedCatalog, "__init__", interrupted)
        with pytest.raises(MemoryError) as caught:
            construct_from_canonical_branch(engine, lambda _catalog: pytest.fail("partial constructor bound decoder"))
    assert caught.value is primary and len(partial) == 1
    assert engine._published_retention_catalog is None and not first.events
    engine._close_resources(preserve_transport=True)  # Actual preinitialized partial factory owner closes.
    assert partial[0]._closed and engine.retention_catalog is None
    store.check_health()
    again = Catalog()
    publish(engine, again)
    commit(store, "resp_after_refusal")
    assert again.events == [("committed", "alice", "response:resp_after_refusal", 1)]


def test_failed_decoder_binding_leaves_only_the_cleanup_journal(engine, store, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retained_catalog

    first = Catalog()
    publish(engine, first)
    engine.bind_api_record_store(store)
    engine._close_resources(preserve_transport=True)
    candidate = Catalog()
    primary = ValueError("decoder catalog binding refused")

    def factory(*_args, register, **_kwargs):
        register(candidate)
        return candidate

    def bind(owner):
        assert engine.retention_catalog is owner is candidate
        assert engine._published_retention_catalog is None
        raise primary

    monkeypatch.setattr(retained_catalog, "make_retention_catalog", factory)
    with pytest.raises(ValueError) as caught:
        construct_from_canonical_branch(engine, bind)
    assert caught.value is primary and engine._published_retention_catalog is None
    commit(store, "resp_after_bind_refusal")
    assert candidate.events == []
    engine._close_resources(preserve_transport=True)
    assert candidate.closed


def test_canonical_factory_publishes_only_after_decoder_binding(engine, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retained_catalog

    candidate, events = Catalog(), []

    def factory(*_args, register, **_kwargs):
        register(candidate)
        assert engine._published_retention_catalog is None
        events.append("constructed")
        return candidate

    def bind(owner):
        assert engine._published_retention_catalog is None and owner is candidate
        assert owner.host_limits == {"baseline_host_bytes": 123, "materialized_host_bytes": 0}
        assert owner.device_limits == engine._retention_device_resources["limits"]
        events.append("bound")

    monkeypatch.setattr(retained_catalog, "make_retention_catalog", factory)
    construct_from_canonical_branch(engine, bind)
    assert events == ["constructed", "bound"] and engine._published_retention_catalog is candidate


def test_canonical_factory_preserves_paid_native_protocol_bounds(engine, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retained_catalog

    candidate = Catalog()

    def factory(*_args, register, **_kwargs):
        register(candidate)
        return candidate

    native = {"fixed": {"global_native_fixed_payload_bytes": 17,
                        "maximum_state_slots": 40, "nofile_hard_limit": 1024},
              "prefix": {"global_native_prefix_payload_bytes": 7}}
    monkeypatch.setattr(retained_catalog, "make_retention_catalog", factory)

    def bind(owner):
        assert engine._published_retention_catalog is None
        assert owner.host_limits == {"baseline_host_bytes": 123, "materialized_host_bytes": 0,
            "prepaid_native_fixed_payload_bytes": 17, "prepaid_native_prefix_payload_bytes": 7,
            "native_state_slots": 40, "native_fd_limit": 1024, "native_host_bytes": 9}

    construct_from_canonical_branch(engine, bind, native=native, native_host_bytes=9)
    assert engine._published_retention_catalog is candidate


def test_protected_claims_prevent_withdrawal_and_keep_current_callbacks(engine, store):
    catalog = Catalog()
    publish(engine, catalog)
    engine.bind_api_record_store(store)
    catalog.has_retained = True
    with pytest.raises(RuntimeError, match="protected retained claims"):
        engine._close_resources(preserve_transport=True)
    assert engine._published_retention_catalog is catalog and not catalog.closed
    commit(store, "resp_while_blocked")
    assert catalog.events[0][2] == "response:resp_while_blocked"
    catalog.has_retained = False


def test_callback_borrower_retires_before_catalog_close_without_holding_lifecycle(engine, store):
    catalog = Catalog()
    catalog.entered, catalog.release = threading.Event(), threading.Event()
    publish(engine, catalog)
    engine.bind_api_record_store(store)
    errors, retired = [], threading.Event()

    def notify():
        try:
            commit(store, "resp_concurrent")
        except BaseException as error:
            errors.append(error)

    def close():
        try:
            engine._close_resources(preserve_transport=True)
        except BaseException as error:
            errors.append(error)
        finally:
            retired.set()

    caller = threading.Thread(target=notify)
    closer = threading.Thread(target=close)
    caller.start()
    try:
        assert catalog.entered.wait(5)
        assert engine._lifecycle.acquire(blocking=False), "notification held model lifecycle across callback IO"
        engine._lifecycle.release()
        closer.start()
        assert not retired.wait(.05) and not catalog.closed
    finally:
        catalog.release.set()
        caller.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not caller.is_alive() and not closer.is_alive() and not errors
    assert retired.is_set() and catalog.closed and not engine._catalog_callback_leases


def test_unhealthy_current_catalog_preserves_primary_and_prevents_false_callback_success(engine, store):
    catalog = Catalog()
    publish(engine, catalog)
    engine.bind_api_record_store(store)
    primary = RuntimeError("current computed owner is poisoned")
    catalog.failure = primary
    with pytest.raises(RuntimeError) as caught:
        commit(store, "resp_failed")
    assert caught.value is primary and catalog.events == []
    assert not engine._catalog_callback_leases
    catalog.failure = None


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_callback_registration_after_effect_has_primary_and_native_lease_retirement(engine, cleanup_fails):
    catalog = Catalog()
    publish(engine, catalog)
    primary, secondary = MemoryError("callback registration failed"), MemoryError("callback journal cleanup failed")

    class Journal(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise primary

        def pop(self, *args):
            if cleanup_fails:
                raise secondary
            return super().pop(*args)

    engine._catalog_callback_leases = Journal()
    with pytest.raises(MemoryError) as caught:
        engine._retention_reference_committed("alice", "response:unpublished", generation=1)
    assert caught.value is primary and catalog.events == []
    assert all(not lease.locked() for lease in engine._catalog_callback_leases)
    if cleanup_fails:
        assert primary.__cause__ is secondary
    engine._close_resources(preserve_transport=True)
    assert catalog.closed and not engine._catalog_callback_leases


def test_notification_cannot_reenter_its_own_model_retirement(engine):
    catalog = Catalog()
    publish(engine, catalog)

    def reenter(*_args, **_kwargs):
        engine._close_resources(preserve_transport=True)

    catalog.logical_reference_committed = reenter
    with pytest.raises(RuntimeError, match="cannot retire its own model owner"):
        engine._retention_reference_committed("alice", "response:reentrant", generation=1)
    assert engine._published_retention_catalog is catalog and not catalog.closed
    assert not engine._catalog_callback_leases


def test_callback_drain_depends_on_native_lease_not_retirement_notification(engine, monkeypatch):
    catalog = Catalog()
    catalog.entered, catalog.release = threading.Event(), threading.Event()
    publish(engine, catalog)
    errors, waiting, retired = [], threading.Event(), threading.Event()
    primary = MemoryError("callback retirement notification failed")
    original_notify = engine._lifecycle.notify_all

    def notify():
        try:
            engine._retention_reference_committed("alice", "response:completed", generation=1)
        except BaseException as error:
            errors.append(error)

    def close():
        try:
            engine._close_resources(preserve_transport=True)
        except BaseException as error:
            errors.append(error)
        finally:
            retired.set()

    caller, closer = threading.Thread(target=notify), threading.Thread(target=close)

    class Journal(dict):
        def items(self):
            if threading.current_thread() is closer:
                waiting.set()
            return super().items()

    def fail_notify():
        if threading.current_thread() is caller:
            raise primary
        original_notify()

    engine._catalog_callback_leases = Journal()
    monkeypatch.setattr(engine._lifecycle, "notify_all", fail_notify)
    caller.start()
    try:
        assert catalog.entered.wait(5)
        closer.start()
        assert waiting.wait(5)
        with engine._lifecycle:
            assert engine._published_retention_catalog is None
    finally:
        catalog.release.set()
        caller.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not caller.is_alive() and not closer.is_alive() and retired.is_set()
    assert errors == [] and catalog.closed and not engine._catalog_callback_leases


def test_failed_catalog_close_retains_cleanup_owner_without_restoring_notifications(engine):
    catalog = Catalog()
    publish(engine, catalog)
    primary = RuntimeError("catalog retirement has unresolved CPU owners")
    actual = catalog.close

    def fail_close(*, timeout):
        raise primary

    catalog.close = fail_close
    with pytest.raises(RuntimeError) as caught:
        engine._close_resources(preserve_transport=True)
    assert caught.value is primary and engine.retention_catalog is catalog
    assert engine._published_retention_catalog is None and not catalog.closed
    engine._retention_reference_committed("alice", "response:no-computed-owner", generation=1)
    assert not catalog.events
    catalog.close = actual
    engine._close_resources(preserve_transport=True)
    assert catalog.closed and engine.retention_catalog is None
