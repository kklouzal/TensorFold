"""Payload credits remain owned across handlers, workers and failed handoffs."""
from types import SimpleNamespace
import gc
import json
import threading
import time
import weakref
import socket

import pytest

from tensorfold.server import host_payload as hp
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import CapacityError


def pool(capacity=100, deadline=None):
    return hp.HostPayloadPool(capacity, body_bound=lambda n: 10 + n * 3,
        generation_bound=lambda n, out, extent: (n + out, out, 0, 0), encoding_bound=lambda n: n + 1,
        cache_bound=lambda n, *, copying: n * (2 if copying else 1), measurement_bytes=0,
        deadline_seconds=deadline)


def test_worker_borrow_keeps_credit_after_socket_owner_retires():
    p = pool()
    owner = p.lease()
    owner.set("request", 70)
    worker = owner.borrow()
    owner.close()
    assert p.record()["reserved_bytes"] == 70
    with pytest.raises(CapacityError):
        p.lease().set("other", 31)
    worker.close()
    assert p.record()["reserved_bytes"] == 0
    with pytest.raises(RuntimeError, match="twice"):
        worker.close()


def test_closed_body_and_lease_retire_without_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    p = pool(1000)
    try:
        with hp.handler_scope(SimpleNamespace(host_payload_pool=p)) as owner:
            body = owner.begin_body()
            body.receive(7)
            body_ref, owner_ref = weakref.ref(body), weakref.ref(owner)
            del body
            assert body_ref() is None  # Lease does not retain its former body.
        assert p.used == 0 and p.active_handlers == 0
        del owner
        assert owner_ref() is None  # No gc.collect() hides an ownership cycle.
    finally:
        if enabled:
            gc.enable()


def test_concurrent_duplicate_borrow_close_cannot_retire_the_http_owner(monkeypatch):
    p = pool()
    owner = p.lease()
    owner.set("body", 70)
    borrowed = owner.borrow()
    original = owner._release
    def delayed_close():
        time.sleep(0.02)  # Releases the GIL inside the adverse check/use window.
        original()
    monkeypatch.setattr(owner, "_release", delayed_close)
    start = threading.Barrier(3)
    errors = []
    def close():
        start.wait()
        try:
            borrowed.close()
        except RuntimeError as error:
            errors.append(error)
    threads = [threading.Thread(target=close) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(1)
    assert all(not thread.is_alive() for thread in threads)
    assert len(errors) == 1 and "twice" in str(errors[0])
    assert owner.borrowers == 1 and not owner.closed and p.used == 70
    owner.close()
    assert p.used == 0


def test_http_owner_cannot_close_twice_and_consume_a_live_worker():
    p = pool()
    owner = p.lease()
    owner.set("body", 70)
    worker = owner.borrow()
    owner.close()
    with pytest.raises(RuntimeError, match="owner retired twice"):
        owner.close()
    assert owner.borrowers == 1 and not owner.closed and p.used == 70
    worker.close()
    assert p.used == 0


def test_phase_growth_refuses_without_waiting_or_partial_reservation():
    p = pool(10, deadline=10)
    first, second = p.lease(), p.lease()
    first.set("body", 4)
    second.set("body", 4)
    start = time.monotonic()
    with pytest.raises(CapacityError):
        first.set("body", 8)
    assert time.monotonic() - start < 1
    assert p.used == 8 and first.claims == {"body": 4}
    first.close()
    second.close()
    assert p.used == 0


def test_borrow_constructor_failure_does_not_publish_a_consumer(monkeypatch):
    p = pool()
    owner = p.lease()
    owner.set("body", 50)
    def fail(_lease):
        raise MemoryError("borrow constructor")
    monkeypatch.setattr(hp, "HostPayloadBorrow", fail)
    with pytest.raises(MemoryError):
        owner.borrow()
    assert owner.borrowers == 1 and p.used == 50
    owner.close()
    assert p.used == 0


def test_claim_preparation_exhaustion_preserves_old_ownership(monkeypatch):
    p = pool()
    owner = p.lease()
    owner.set("body", 50)
    def fail(_value):
        raise MemoryError("claim replacement")
    with monkeypatch.context() as patch:
        patch.setattr(hp, "dict", fail, raising=False)
        with pytest.raises(MemoryError):
            owner.set("body", 70)
    assert owner.claims == {"body": 50} and p.used == 50
    owner.close()


def test_context_reset_failure_still_closes_handler_and_preserves_primary(monkeypatch):
    p = pool()
    class Context:
        def set(self, lease):
            self.lease = lease
            return object()
        def reset(self, _token):
            raise RuntimeError("context reset")
    context = Context()
    monkeypatch.setattr(hp, "_CURRENT", context)
    with pytest.raises(ValueError, match="handler primary") as caught:
        with hp.handler_scope(SimpleNamespace(host_payload_pool=p)) as lease:
            lease.set("body", 50)
            raise ValueError("handler primary")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "context reset"
    assert context.lease.owner_released and not context.lease.closed
    assert context.lease.failure is caught.value and p.used == 50 and p.failure is not None


def test_initial_wait_is_fifo_and_has_no_allocated_payload():
    p = pool(10, deadline=2)
    occupied = p.lease()
    occupied.set("body", 10)
    order, errors = [], []
    def work(label):
        owner = p.lease()
        try:
            owner.set("body", 10)
            order.append(label)
        except BaseException as error:
            errors.append(error)
        finally:
            owner.close()
    first = threading.Thread(target=work, args=(1,))
    second = threading.Thread(target=work, args=(2,))
    first.start()
    with p._condition:
        assert p._condition.wait_for(lambda: len(p._waiters) == 1, timeout=1)
    second.start()
    with p._condition:
        assert p._condition.wait_for(lambda: len(p._waiters) == 2, timeout=1)
    occupied.close()
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert order == [1, 2] and not errors and p.used == 0


def test_cancelled_waiter_retires_ticket_without_borrowing_bytes():
    p = pool(10, deadline=2)
    occupied, waiting = p.lease(), p.lease()
    occupied.set("body", 10)
    waiting.cancelled = lambda: True
    with pytest.raises(RequestCancelled):
        waiting.set("body", 10)
    assert not p._waiters and p.used == 10 and not waiting.claims
    waiting.close()
    occupied.close()


def test_cancel_before_available_acquisition_does_not_publish_bytes():
    p = pool()
    owner = p.lease()
    owner.cancelled = lambda: True
    with pytest.raises(RequestCancelled):
        owner.set("body", 10)
    assert p.used == 0 and not owner.claims
    owner.close()


def test_registration_failure_keeps_an_explicit_poisoned_owner():
    p = pool()
    class FailingSet(set):
        def add(self, owner):
            super().add(owner)
            raise MemoryError("registration after publication")
    p._leases = FailingSet()
    with pytest.raises(MemoryError):
        p.lease()
    assert p.record()["poisoned"] and p.record()["owners"] == 1 and p.used == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        p.lease()


def test_receive_credit_funds_parse_before_reader_returns(monkeypatch):
    p = pool()
    def reader(_handler):
        body = hp.current().begin_body()
        body.receive(7)
        return b'{"x":1}'
    from tensorfold.server import request_body
    import json
    original = json.loads
    def parse(value):
        assert p.used == 31  # complete raw+parse phase, before parser allocation
        return original(value)
    monkeypatch.setattr(request_body, "read_body", reader)
    monkeypatch.setattr(json, "loads", parse)
    with hp.handler_scope(SimpleNamespace(host_payload_pool=p)):
        # Production handlers use this same reader-then-default-parser path.
        assert json.loads(request_body.read_body(object())) == {"x": 1}
        assert p.used == 31  # Keep the parse/error envelope through retirement.
    assert p.used == 0


def test_handled_traceback_keeps_credit_with_gc_disabled_until_roots_retire():
    p = pool()
    class Parent:
        pass
    parents = []
    def operation():
        parent = Parent()
        parents.append(weakref.ref(parent))
        try:
            raise ValueError("expected request refusal")
        except ValueError as error:
            saved = error  # Reproduces the real _execute error/frame cycle.
        return saved
    enabled = gc.isenabled()
    gc.disable()
    try:
        with hp.handler_scope(SimpleNamespace(host_payload_pool=p)):
            hp.current().set("body", 70)
            retained = operation()
            hp.keep_error(retained)
        assert p.used == 70 and parents[0]() is not None
        del retained
        assert p.used == 70 and parents[0]() is not None
        gc.collect()
        assert parents[0]() is None and p.used == 0 and not p.failure
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("value", [
    {"ascii": "a" * 100, "escape": "\0\n\\\"", "wide": "é漢😀\ud800"},
    {0: [True, None, 10 ** 4299, -(10 ** 4299), 1.7976931348623157e308]},
    {False: float("inf"), None: float("nan"), -1.0: ()},
])
def test_wire_bound_covers_default_library_encoder_without_joining_source(value):
    bound = hp.json_wire_bound(value, maximum=1 << 20)
    assert len(json.dumps(value)) <= bound


def test_wire_measurement_preserves_aliases_and_rejects_only_actual_cycles():
    shared = [1, "value"]
    value = [shared, shared]
    assert hp.json_wire_bound(value, maximum=10000) >= len(json.dumps(value))
    shared.append(value)
    with pytest.raises(ValueError, match="Circular reference"):
        hp.json_wire_bound(value, maximum=10000)


def test_wire_bound_uses_wire_units_for_many_empty_strings_and_wide_kind():
    for value in ([""] * 1000, {"key": "ASCII" * 1000 + "😀"}):
        encoded = json.dumps(value)
        bound = hp.json_wire_bound(value, maximum=1 << 20)
        assert len(encoded) <= bound <= 12 * len(encoded)


def test_growing_encoder_frames_charge_total_highwater_not_incremental_highwater():
    p = pool(10000)
    p.prepaid_encoder_bytes = 10
    owner = p.lease()
    highwater = 0
    for length in (30, 50, 80, 20):
        wire = owner.encode("x" * length)
        highwater = max(highwater, wire)
        assert p.used + p.prepaid_encoder_bytes >= p.encoding_bound(wire)
    assert p.used == p.encoding_bound(highwater) - 10
    owner.close()


def test_funded_answer_encodes_when_unrelated_consumers_fill_remaining_pool():
    p = pool(1000)
    reply = {"answer": "x" * 64}
    future_wire = hp.json_wire_bound(reply, maximum=p.capacity)
    p.generation_bound = lambda amount, output, extent: (p.encoding_bound(future_wire), future_wire, 0, 0)
    owner, other = p.lease(), p.lease()
    owner.generation(body_bytes=0, output_tokens=64, extent_tokens=64)
    other.set("other", p.capacity - p.used)
    owner.encode(reply)
    assert owner.claims["encoder"] == 0 and p.used == p.capacity
    other.close()
    owner.close()


def test_control_response_encoder_is_funded_before_metadata_work():
    p = pool(10000)
    reply = {"status": "ready"}
    wire = hp.json_wire_bound(reply, maximum=p.capacity)
    p.control_bound = lambda amount, kind: (amount + p.encoding_bound(wire), wire)
    owner, other = p.lease(), p.lease()
    owner.control(100)
    other.set("other", p.capacity - p.used)
    owner.encode(reply)
    assert owner.claims["encoder"] == 0 and p.used == p.capacity
    other.close()
    owner.close()


def test_cache_transfer_drops_encoder_authority_of_released_control_phase():
    p = pool(10000)
    p.metadata_wire_bytes = 0
    p.control_bound = lambda amount, kind: (1000, 900)
    p.cache_bound = lambda amount, copying: amount + p.encoding_bound(12 * amount)
    owner = p.lease()
    owner.control(1)
    owner.cache(1, copying=False)
    assert owner.funded_output_bound == 12 and owner.claims["control"] == 0
    wire = owner.encode("abcd")
    assert p.used >= p.encoding_bound(wire)
    owner.close()


def test_owned_tokenizer_callbacks_never_wait_even_with_empty_owner():
    p = pool(10, deadline=60)
    p.ids_bound = lambda *args, **kwargs: (10, 0)
    occupied, waiting = p.lease(), p.lease()
    occupied.set("other", 10)
    entered = time.monotonic()
    with pytest.raises(CapacityError):
        waiting.ids(2)
    assert time.monotonic() - entered < 1 and not p._waiters
    waiting.close()
    occupied.close()


def test_tokenize_observes_native_count_before_ids_and_lexical_materialization():
    from tensorfold.cuda.server import App

    p = pool(10000)
    p.ids_bound = lambda count, *args, **kwargs: (10 * count + 1001, 1000)
    app = App.__new__(App)
    app.host_payload_pool = p
    app._context_limit = lambda: 128
    class Encoding:
        def __len__(self):
            return 7  # Native normalization may produce more IDs than input chars.
        @property
        def ids(self):
            assert p.used >= 1071
            return [0] * 7
    class Tokenizer:
        def encode(self, *args, **kwargs):
            return Encoding()
        def id_to_token(self, value):
            assert hp.current().lexical_output and hp.current().funded_output_bound == 1000
            return "lexical"
    app.tok = Tokenizer()
    with hp.handler_scope(app):
        reply = app.tokenize({"prompt": "x", "return_token_strs": True})
        assert reply["count"] == 7 and reply["token_strs"] == ["lexical"] * 7
    assert p.used == 0


@pytest.mark.parametrize("thinking,prompt,expected,constraint", [
    (False, 4096, (64, 4160), {}), (False, 262100, (64, 262144), {}),
    (True, 4096, (262144, 262144), {}), (True, 1, (262144, 262144), {}),
    # Any gated/forced continuation keeps the full-context output envelope.
    (False, 4096, (262144, 262144), {"tools": [{"type": "function"}]}),
    (False, 4096, (262144, 262144), {"grammar": ("spec", "compiled")}),
    (False, 4096, (262144, 262144), {"calls": object()})])
def test_generation_pays_its_prepared_token_extent(thinking, prompt, expected, constraint):
    from tensorfold.cuda.server import App

    p = pool(1 << 20)
    p.context_tokens = 262144
    calls = []
    p.generation_bound = lambda amount, output, extent: (calls.append((amount, output, extent)) or 10, output, 0, 0)
    app = App.__new__(App)
    app.host_payload_pool = p
    prepared = SimpleNamespace(**{"thinking": thinking, "tools": [], "grammar": None, "calls": None,
                                  "max_tokens": 64, "prompt": [7] * prompt, **constraint})
    with hp.handler_scope(app):
        owner = app._http_generation_owner(None)
        app._reserve_http_generation(owner, prepared)
        assert owner.claims["generation"] == 10
    assert calls == [(0, *expected)]


def test_generation_prepayments_offset_later_ids_and_rpc_only():
    p = pool(1000)
    p.generation_bound = lambda amount, output, extent: (300, 0, 100, 150)
    p.ids_bound = lambda count, *args, **kwargs: (count, 0)
    p.rpc_bound = lambda amount: amount
    with hp.handler_scope(SimpleNamespace(host_payload_pool=p)) as owner:
        owner.generation(body_bytes=0, output_tokens=1, extent_tokens=1)
        owner.ids(100)
        assert owner.claims["tokenizer-ids"] == 0
        owner.rpc(150)
        assert owner.claims["rpc"] == 0
        owner.ids(130)
        owner.rpc(170)
        assert owner.claims["tokenizer-ids"] == 30 and owner.claims["rpc"] == 20
    assert p.used == 0


def test_generation_rebases_tokenization_ids_paid_before_it():
    p = pool(1000)
    p.generation_bound = lambda amount, output, extent: (300, 0, 100, 150)
    p.ids_bound = lambda count, *args, **kwargs: (count, 0)
    with hp.handler_scope(SimpleNamespace(host_payload_pool=p)) as owner:
        owner.ids(130)  # Tokenization precedes the generation grant.
        assert owner.claims["tokenizer-ids"] == 130
        owner.generation(body_bytes=0, output_tokens=1, extent_tokens=1)
        assert owner.claims["tokenizer-ids"] == 30 and p.used == 330
    assert p.used == 0


def test_keepalive_without_a_header_slot_closes_before_parsing_without_poison():
    from tensorfold.server.json_output import bind_handler

    p = pool()
    p.max_handlers = 1
    owner = p.lease()
    owner.set("body", 100)
    worker = owner.borrow()
    owner.retired_handler_bytes = 1  # No dynamic capacity for the transfer.
    owner.close()
    assert p.fixed_slots_full()
    class Handler:
        def handle_one_request(self):
            pytest.fail("unpaid headers parsed")
        def parse_request(self):
            pytest.fail("unpaid headers parsed")
        def do_GET(self):
            pass
        def do_POST(self):
            pass
    app = SimpleNamespace(host_payload_pool=p)
    handler = bind_handler(Handler, app)()
    handler.server = SimpleNamespace(fail_json_encoding=lambda *args: pytest.fail("pressure became fatal"))
    handler.close_connection = False
    handler.handle_one_request()
    assert handler.close_connection and p.failure is None and p.used == 100
    worker.close()
    assert not p.fixed_slots_full() and p.used == 0


def test_stream_rpc_and_mailbox_share_one_borrow_until_terminal_retirement():
    p = pool(1000)
    p.generation_bound = lambda amount, output, extent: (100, 0, 0, 50)
    p.rpc_bound = lambda amount: 2 * amount
    app = SimpleNamespace(host_payload_pool=p)
    stream, mailbox = SimpleNamespace(), SimpleNamespace()
    with hp.handler_scope(app) as owner:
        owner.generation(body_bytes=0, output_tokens=1, extent_tokens=1)
        hp.borrow_request(p, stream, mailbox)
        hp.reserve_rpc(stream, 80)
        assert owner.claims["rpc"] == 110
    assert p.used == 210 and stream._host_payload_borrow is mailbox._host_payload_borrow
    hp.retire_mailbox(mailbox)
    assert p.used == 0 and mailbox._host_payload_borrow is None


def test_actual_worker_error_terminal_transfers_loan_without_http_context():
    from tensorfold.cuda.scheduler import Scheduler

    enabled = gc.isenabled()
    gc.disable()
    p = pool(1000)
    app = SimpleNamespace(host_payload_pool=p)
    stream, mailbox = SimpleNamespace(), SimpleNamespace()
    values = []
    mailbox.put = values.append
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.host_payload_pool = p
    scheduler.serving_profiles = scheduler._request_limit = None
    scheduler._error_retirements = {}
    try:
        with hp.handler_scope(app) as owner:
            owner.set("generation", 100)
            hp.borrow_request(p, stream, mailbox)
        def fail():
            payload = object()
            try:
                raise ValueError("expected backend failure")
            except ValueError as error:
                held = error  # A normal backend local may form a traceback cycle.
                assert payload is not None
                return held
        retained = fail()
        assert hp.current() is None
        scheduler._terminal(mailbox, "error", retained)
        assert mailbox._host_payload_borrow is stream._host_payload_borrow and p.used == 100
        values.clear()
        del retained
        assert p.used == 100  # Uncollected traceback cycle is still paid.
        del stream, mailbox
        gc.collect()
        assert p.used == 0 and p.failure is None
    finally:
        if enabled:
            gc.enable()


def test_normal_container_loan_outlives_either_stream_or_buffered_mailbox():
    p = pool()
    stream, mailbox = SimpleNamespace(), SimpleNamespace()
    with hp.handler_scope(SimpleNamespace(host_payload_pool=p)) as owner:
        owner.set("generation", 70)
        hp.borrow_request(p, stream, mailbox)
    borrowed = weakref.ref(stream._host_payload_borrow)
    del stream
    assert borrowed() is not None and p.used == 70
    del mailbox
    assert borrowed() is None and p.used == 0


def test_container_finalizer_install_failure_poison_retains_uncertain_loan(monkeypatch):
    p = pool()
    owner = p.lease()
    owner.set("generation", 70)
    token = hp._CURRENT.set(owner)
    primary = MemoryError("container finalizer allocation")
    def fail(*args):
        raise primary
    monkeypatch.setattr(hp.weakref, "finalize", fail)
    try:
        with pytest.raises(MemoryError) as raised:
            hp.borrow_request(p, SimpleNamespace(), SimpleNamespace())
        assert raised.value is primary and p.failure is primary
        assert owner.borrowers == 2 and p.used == 70
        owner.close()
        assert p.used == 70
    finally:
        hp._CURRENT.reset(token)


def test_actual_fifo_idle_wait_drops_only_obsolete_snapshot_aliases():
    from tensorfold.cuda.scheduler import Scheduler

    p = pool()
    stream, mailbox = SimpleNamespace(), SimpleNamespace()
    entry = (0, 0, (stream, mailbox))
    class IdleReached(Exception):
        pass
    class Waiting:
        def __init__(self, first):
            self.first = first
        def requests(self, **kwargs):
            first, self.first = self.first, []
            return 0, first
        def get_control(self, **kwargs):
            assert p.used == 0  # No new grant required to retire old local snapshots.
            raise IdleReached
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting = Waiting([entry])
    scheduler._retiring = []
    scheduler._state_lock = threading.RLock()
    scheduler._stopping = scheduler._stop_published = False
    scheduler._serving_profile_available = True
    scheduler._execution_boundary = lambda: None
    scheduler._admit_fifo = lambda entries: [item[2][0] for item in entries]
    scheduler._round_requests = lambda done, **kwargs: False  # Fake backend already retired.
    scheduler.decoder = SimpleNamespace(live=lambda: 0)
    with hp.handler_scope(SimpleNamespace(host_payload_pool=p)) as owner:
        owner.set("generation", 70)
        hp.borrow_request(p, stream, mailbox)
    del stream, mailbox, entry
    with pytest.raises(IdleReached):
        scheduler._run_fifo()


def test_listener_does_not_clear_a_pool_retirement_notification_after_predicate():
    from tensorfold.server.http import Server

    changed = threading.Event()
    class Pool:
        def fixed_slots_full(self):
            changed.set()  # Inject release between predicate and old event.clear.
            return True
    server = Server.__new__(Server)
    server._accept_gate = threading.Lock()
    server._connection_lock = threading.Lock()
    server.stopping = threading.Event()
    server._serve_stop = threading.Event()
    server._connections = {}
    server.max_connections = 1
    server.RequestHandlerClass = SimpleNamespace(_host_payload_pool=Pool())
    server._capacity_changed = changed
    server._capacity_poll = 0
    server.service_actions = lambda: None
    server._handle_request_with_capacity()
    assert changed.is_set()


@pytest.mark.parametrize("stop", [False, True], ids=["peer-close", "server-stop"])
def test_actual_operation_early_credit_wait_observes_socket_or_stop_without_poison(stop):
    from tensorfold.server.json_output import bind_handler

    p = pool(100, deadline=5)
    p.metadata_wire_bytes = 1
    p.control_bound = lambda amount, kind: (50, 10)
    occupied = p.lease()
    occupied.set("occupied", 70)
    left, right = socket.socketpair()
    class Handler:
        path = "/ordinary"
        def handle_one_request(self):
            self.do_GET()
        def do_GET(self):
            pytest.fail("cancelled initial control executed metadata work")
        def do_POST(self):
            pass
        def parse_request(self):
            pass
    handler = bind_handler(Handler, SimpleNamespace(host_payload_pool=p))()
    handler.connection = left
    handler.close_connection = False
    handler.server = SimpleNamespace(stopping=threading.Event(),
        fail_json_encoding=lambda *args: pytest.fail("ordinary cancellation became cohort fatal"))
    errors = []
    def run():
        try:
            handler.handle_one_request()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    try:
        thread.start()
        with p._condition:
            assert p._condition.wait_for(lambda: len(p._waiters) == 1, timeout=1)
        if stop:
            handler.server.stopping.set()
        else:
            right.close()
        thread.join(1)
        assert not thread.is_alive() and not errors
        assert handler.close_connection and p.failure is None and not p._waiters
    finally:
        right.close()
        left.close()
        thread.join(1)
        occupied.close()


def test_metadata_snapshots_accumulate_copy_obligations_before_allocation():
    p = pool(10000)
    p.metadata_wire_bytes = p.monitor_wire_bytes = 10
    p.snapshot_bound = lambda wire: 2 * wire
    owner = p.lease()
    value = {"metadata": [1, 2, 3]}
    owner.snapshot(value)
    first = p.used
    owner.snapshot(value)
    assert p.used > first and owner._snapshot_graph > first
    owner.close()


def test_failed_metadata_claim_does_not_pin_replaced_source_in_retained_error():
    import sys
    p = pool(100000)
    p.metadata_wire_bytes = p.monitor_wire_bytes = 0
    p.snapshot_bound = lambda wire: 100000
    owner = p.lease()
    metadata = {"payload": "x" * 1000}
    before = sys.getrefcount(metadata)
    try:
        owner.snapshot(metadata)
    except CapacityError as error:
        retained = error
    assert sys.getrefcount(metadata) == before
    assert p.used == 0 and p.failure is None
    assert retained._tensorfold_host_refusal["phase"] == "metadata-snapshots"
    owner.close()


def test_controller_snapshot_refusal_precedes_clones_and_releases_identity_lock(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import adaptive_controller as module

    controller = module.AdaptiveController(slots=2, row_budget=6, proposal_limit=2, prefill_tokens=16)
    copies = []
    original = dict
    monkeypatch.setattr(module, "dict", lambda value: copies.append(value) or original(value), raising=False)
    primary = CapacityError("snapshot admission refused")
    def credit(value):
        assert controller._lock.locked() and value[0] is controller.configured
        raise primary
    with pytest.raises(CapacityError) as raised:
        controller.record(credit=credit)
    assert raised.value is primary and copies == [] and not controller._lock.locked()


def test_execution_snapshot_copies_exact_generation_that_was_credited():
    from tensorfold.families.qwen4_exp.cuda.execution_observations import ExecutionObservations

    observations = ExecutionObservations(4)
    before = observations._published
    newer = (tuple((key, value + 5) for key, value in before[0]), *before[1:])
    def credit(value):
        assert value is before
        observations._published = newer  # Producer publishes an immutable replacement.
    record = observations.record(credit=credit)
    assert record["totals"] == dict(before[0]) and record["counts_complete"]
    assert observations.record()["totals"] == dict(newer[0])


def presentation_pool(capacity, deadline=None, table=(1, 3, 128)):
    p = pool(capacity, deadline=deadline)
    calls = []

    def bound(amount, output, extent, *, presentation_bytes=None):
        calls.append(presentation_bytes)
        return extent + presentation_bytes, presentation_bytes, 0, 0
    p.generation_bound = bound
    p.piece_bytes, p.piece_bytes_max = list(table), max(table)
    return p, calls


def test_generation_presentation_starts_small_and_grows_by_actual_decoded_bytes(monkeypatch):
    monkeypatch.setattr(hp, "PRESENTATION_INITIAL_BYTES", 8)
    p, calls = presentation_pool(10_000)
    owner = p.lease()
    owner.generation(body_bytes=0, output_tokens=1000, extent_tokens=1100)
    assert calls == [8] and owner.claims["generation"] == 1108
    owner.present([0, 1, 0])           # 5 bytes: inside the first allowance
    assert calls == [8]
    owner.present([1, 1])              # 11 > 8: geometric growth before appending
    assert calls == [8, 12] and owner.claims["generation"] == 1112 and owner._presented_bytes == 11
    owner.present([2])                 # one worst-case piece
    assert calls[-1] == 139 and owner.claims["generation"] == 1239
    owner.close()
    assert p.used == 0


def test_generation_presentation_never_exceeds_the_worst_case_and_tables_absent_keep_it():
    p, calls = presentation_pool(10_000, table=(128,))
    owner = p.lease()
    owner.generation(body_bytes=0, output_tokens=2, extent_tokens=10)
    assert calls == [256]               # min(initial allowance, output * worst piece)
    owner.present([0, 0])
    with pytest.raises(RuntimeError, match="worst-case"):
        owner.present([0])
    owner.close()
    plain = pool(1000)
    plain.generation_bound = lambda n, out, extent: (extent, 0, 0, 0)
    other = plain.lease()
    other.generation(body_bytes=0, output_tokens=5, extent_tokens=7)
    other.present([1, 2, 3])            # no table: the admission already paid the worst case
    assert other.claims["generation"] == 7
    other.close()


def test_presentation_growth_waits_ahead_of_new_requests_then_succeeds(monkeypatch):
    monkeypatch.setattr(hp, "PRESENTATION_INITIAL_BYTES", 8)
    p, _calls = presentation_pool(1200, deadline=5)
    owner = p.lease()
    owner.generation(body_bytes=0, output_tokens=1000, extent_tokens=1000)   # 1008
    blocker = p.lease()
    blocker.set("body", 180)                                                 # 12 left
    fresh, order, errors, ticks = p.lease(), [], [], []

    def grow():
        try:
            owner.present([2], tick=lambda: ticks.append(1) or True)        # needs +120
            order.append("grow")
        except Exception as error:  # noqa: BLE001 - the thread reports any failure to the test
            errors.append(error)

    def admit():
        try:
            fresh.set("body", 50)
            order.append("fresh")
        except Exception as error:  # noqa: BLE001 - the thread reports any failure to the test
            errors.append(error)
    grower = threading.Thread(target=grow)
    grower.start()
    with p._condition:
        assert p._condition.wait_for(lambda: len(p._growers) == 1, timeout=2)
    newcomer = threading.Thread(target=admit)
    newcomer.start()
    with p._condition:
        assert p._condition.wait_for(lambda: len(p._waiters) == 1, timeout=2)
    blocker.close()
    grower.join(3)
    newcomer.join(3)
    assert not errors and order == ["grow", "fresh"] and owner.claims["generation"] == 1128
    fresh.close()
    owner.close()
    assert p.used == 0 and not p._growers and not p._waiters


def test_presentation_growth_wait_is_bounded_and_fails_only_that_request(monkeypatch):
    monkeypatch.setattr(hp, "PRESENTATION_INITIAL_BYTES", 8)
    p, _calls = presentation_pool(1200, deadline=0.3)
    owner = p.lease()
    owner.generation(body_bytes=0, output_tokens=1000, extent_tokens=1000)
    other = p.lease()
    other.set("body", 190)
    start = time.monotonic()
    with pytest.raises(CapacityError, match="stayed exhausted"):
        owner.present([2])
    assert 0.25 <= time.monotonic() - start < 2
    assert owner.claims == {"generation": 1008} and owner._presented_bytes == 0 and not p._growers
    assert p.failure is None and other.claims == {"body": 190}
    owner.close()
    other.close()
    assert p.used == 0


def test_presentation_growth_wait_stops_when_the_client_leaves(monkeypatch):
    monkeypatch.setattr(hp, "PRESENTATION_INITIAL_BYTES", 8)
    p, _calls = presentation_pool(1200, deadline=30)
    owner = p.lease()
    owner.generation(body_bytes=0, output_tokens=1000, extent_tokens=1000)
    other = p.lease()
    other.set("body", 190)
    with pytest.raises(RequestCancelled):
        owner.present([2], tick=lambda: False)
    assert owner.claims == {"generation": 1008} and not p._growers
    owner.close()
    other.close()


def test_generation_grant_waits_with_its_frame_held_instead_of_refusing():
    p = pool(100, deadline=5)
    p.generation_bound = lambda n, out, extent: (extent, 0, 0, 0)
    running = p.lease()
    running.set("generation", 80)
    queued = p.lease()
    queued.set("body/1", 10)
    done = []
    worker = threading.Thread(target=lambda: (queued.generation(body_bytes=0, output_tokens=1, extent_tokens=50),
                                              done.append(True)))
    worker.start()
    with p._condition:
        assert p._condition.wait_for(lambda: len(p._waiters) == 1, timeout=2)
    # A live owner's ordinary phase is not starved by queued grants when it fits.
    running.set("rpc", 5)
    running.close()
    worker.join(3)
    assert done and queued.claims == {"body/1": 10, "generation": 50}
    queued.close()
    assert p.used == 0


def test_incremental_generation_pays_prompt_plus_initial_reply_and_funds_ahead(monkeypatch):
    monkeypatch.setattr(hp, "OUTPUT_INITIAL_TOKENS", 100)
    p = pool(1 << 20)
    calls = []
    p.generation_bound = lambda amount, output, extent: (calls.append((output, extent)) or extent * 10, 0, 0, 0)
    owner = p.lease()
    owner.generation(body_bytes=0, output_tokens=262144, extent_tokens=262144, prompt_tokens=2000)
    assert calls == [(100, 2100)] and owner.claims["generation"] == 21000
    funding = owner.output_funding
    assert funding.tokens == 100
    owner.fund_output(40)                       # 40 + 50 <= 100: still ahead
    assert calls == [(100, 2100)]
    owner.fund_output(60)                       # less than half the initial allowance left: grow before parking
    assert calls[-1] == (160, 2160) and funding.tokens == 160 and owner.claims["generation"] == 21600
    owner.close()
    assert p.used == 0


def test_reply_tokens_past_the_paid_ceiling_are_funded_before_presentation(monkeypatch):
    # A gate's fix tokens follow scheduler output that already reached the paid
    # ceiling; App.run funds len(out)+len(new) and only then presents them.
    monkeypatch.setattr(hp, "OUTPUT_INITIAL_TOKENS", 8)
    monkeypatch.setattr(hp, "PRESENTATION_INITIAL_BYTES", 8)
    p, calls = presentation_pool(1 << 20, table=(128,))
    p.generation_bound = (lambda amount, output, extent, *, presentation_bytes=None:
                          (extent + presentation_bytes, presentation_bytes, 0, 0))
    owner = p.lease()
    owner.generation(body_bytes=0, output_tokens=1000, extent_tokens=1100, prompt_tokens=100)
    chunk = [0] * 10                             # 8 produced at the ceiling + 2 fix tokens
    with pytest.raises(RuntimeError):
        owner.present(chunk)                     # unfunded past the ceiling: never silently over budget
    owner.fund_output(len(chunk))
    owner.present(chunk)
    assert owner._funded_output >= 10 and owner._presented_bytes == 1280
    owner.close()
    assert p.used == 0


def test_output_funding_ceiling_parks_production_until_paid_and_follows_continuations():
    from tensorfold.cuda.output_mailbox import OutputFunding, OutputMailbox

    funding = OutputFunding(8)
    woke = []
    first = OutputMailbox(1000)
    first.set_wake(lambda: woke.append(1))
    funding.start(0)
    funding.bind(first)
    lease = first.reserve(8)
    assert lease is not None
    first.put(("tokens", list(range(8))))
    lease.close()
    assert first.reserve(1) is None              # the paid ceiling, not the reply allowance, stops production
    funding.extend(12)
    assert woke and first.reserve(4) is not None
    second = OutputMailbox(1000)                 # a budget/call continuation after 10 kept reply tokens
    funding.start(10)
    funding.bind(second)
    assert second.available(2) and not second.available(3)
    with pytest.raises(ValueError):
        funding.extend(11)


def test_output_funding_does_not_keep_its_mailbox_or_payload_borrow_alive():
    from tensorfold.cuda.output_mailbox import OutputFunding, OutputMailbox

    funding = OutputFunding(8)
    box = OutputMailbox(100)
    funding.bind(box)
    ref = weakref.ref(box)
    del box
    gc.collect()
    assert ref() is None
    funding.extend(16)                           # a retired call's mailbox is simply gone
