"""Mixed resident windows: K full + (N-K) short slots, solved, materialized, placed and recorded per slot."""

import ast
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.capacity import Geometry, Weights, fit_short_window
from tensorfold.cuda.geometry import indexed_stream_geometry
from tensorfold.cuda.memory_gate import MemoryGate, NoRoom
from tensorfold.families.qwen4_exp.cuda import prefixes
from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement
from tensorfold.families.qwen4_exp.cuda.service_memory import (
    MIXED_SESSION_POLICY, context_reservation, full_window_headroom_required, materialized_graph_room,
    mixed_slot_capacities)

GIB = 1 << 30


# ---------------------------------------------------------------- capacity solver

def test_short_window_solver_returns_the_largest_fitting_granule_below_the_full_window():
    def geometry(short):
        return Geometry(lambda rows: 1000 + 2 * rows + 4 * short, 5)

    weights, window = Weights(100, 10), 1000
    fixed = weights.resident + 1000 + 2 * (window + 5)
    for tokens in (64, 65, 127, 128, 500):
        budget = fixed + 4 * (tokens + 5)
        assert fit_short_window(window, budget, weights, geometry, 5, granule=64) == tokens // 64 * 64
    # Unlimited memory still leaves the short window strictly below the full one.
    assert fit_short_window(window, 1 << 60, weights, geometry, 5, granule=64) == 960
    assert fit_short_window(window, fixed + 4 * (63 + 5), weights, geometry, 5, granule=64) == 0
    assert fit_short_window(window, 1 << 60, weights, geometry, 5, granule=64, additional_fits=lambda: False) == 0


@pytest.fixture
def admission(monkeypatch):
    from tensorfold.cuda import build

    monkeypatch.setattr(build, "refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "floor", lambda p: (7, 0))
    monkeypatch.setattr(capacity, "config", lambda p: {"max_position_embeddings": 100})
    monkeypatch.setattr(capacity, "unified", lambda t: False)
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda: None)
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: None)
    monkeypatch.setattr(capacity, "page_room", lambda t: None)
    monkeypatch.setattr(capacity, "available_bytes", lambda t: 4500)
    monkeypatch.setattr(capacity, "SHORT_WINDOW_GRANULE", 10)
    calls = []

    def session_geometry(text, n, short=None):
        calls.append((n, short))
        if short is None:
            return Geometry(lambda r: 200 + n * r, 0)
        # Two full windows at the complete context, the other n-2 at ``short`` rows (60 bytes a row).
        return Geometry(lambda r: 200 + 2 * r + (n - 2) * short * 15, 0)

    return dict(model_dir=".", requested=None, explicit=False, torch=NS(), geometry=Geometry(lambda r: r, 0),
                transform=lambda n, i: (0, 0), weight_estimator=lambda p: (Weights(100, 50), 50),
                session_geometry=session_geometry, session_ceiling=6), calls


def test_mixed_admission_keeps_full_windows_and_solves_one_agreed_short_window(admission):
    kwargs, calls = admission
    result = capacity.admit(**kwargs, session_short=(2, None))
    plan = result["session_capacity"]
    # 100 weights + 200 + 2 x 100 full + 4 x 15 x S <= 4500  ->  S <= 66  ->  60 tokens (granule 10, reserve 0)
    assert result["context_window"] == plan["context_tokens"] == 100
    assert plan["policy"] == MIXED_SESSION_POLICY
    assert plan["resident_capacity"] == 6 and plan["full_context_commitments"] == 2
    assert plan["short_windows"] == {"slots": 4, "context_tokens": 60, "cache_slots": 60, "selection": "largest-fitting",
                                     "largest_fitting_tokens": 60, "granule_tokens": 10}
    assert plan["slot_capacities"] == [100, 100, 60, 60, 60, 60]
    assert plan["limiting_resource"] == "rank-device-memory"
    assert plan["ranks"][0]["session_execution_bytes"] == 200 + 200 + 4 * 15 * 60
    assert plan["ranks"][0]["next_session_execution_bytes"] == 200 + 200 + 4 * 15 * 70
    assert all(n == 6 for n, short in calls if short is not None)  # the slot count never shrinks


def test_explicit_short_window_is_verified_not_enlarged(admission):
    kwargs, _ = admission
    plan = capacity.admit(**kwargs, session_short=(2, 35))["session_capacity"]
    assert plan["short_windows"]["context_tokens"] == 35 and plan["short_windows"]["selection"] == "explicit"
    assert plan["short_windows"]["largest_fitting_tokens"] == 60
    assert plan["slot_capacities"] == [100, 100, 35, 35, 35, 35]
    assert plan["limiting_resource"] == "explicit-short-window"
    with pytest.raises(ValueError, match="largest agreed short window 60 tokens"):
        capacity.admit(**kwargs, session_short=(2, 61))


def test_all_ranks_agree_on_the_worst_rank_short_window(admission):
    kwargs, _ = admission
    packets = []

    def gather(packet):
        packets.append(packet)
        if len(packet) == 11:
            peer = list(packet)
            peer[8] = peer[9] = 40
            return [packet, peer]
        return [packet, [1, *packet[1:]]]

    plan = capacity.admit(**kwargs, session_short=(2, None), world=2, gather=gather)["session_capacity"]
    assert plan["short_windows"]["context_tokens"] == 40 and plan["slot_capacities"][2:] == [40] * 4
    assert plan["binding_ranks"] == [1]


def test_mixed_admission_refuses_instead_of_shrinking_slots(admission, monkeypatch):
    kwargs, _ = admission
    monkeypatch.setattr(capacity, "available_bytes", lambda t: 100 + 50 + 200 + 200 + 4 * 15 * 9)
    with pytest.raises(ValueError, match="cannot retain 2 complete 100-token windows plus 4 slots"):
        capacity.admit(**kwargs, session_short=(2, None))
    for bad in ((6, None), (0, None), (2, 0), (2, 1.5)):
        with pytest.raises(ValueError, match="mixed resident windows"):
            capacity.admit(**kwargs, session_short=bad)


TEXT = dict(hidden_size=2560, num_attention_heads=24, num_key_value_heads=8, head_dim=256,
            num_hidden_layers=48, layer_types=["linear_attention"] * 36 + ["full_attention"] * 12,
            linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4, vocab_size=248320, hc_count=4,
            num_experts_per_tok=10, moe_intermediate_size=640, indexer_head_dim=128,
            indexer_compress_ratio=4)


@pytest.mark.parametrize("short", [127, 128, 1029])
def test_geometry_charges_each_short_slot_its_own_materialized_window(short):
    placement = HistoryPlacement(8, 2, 12)
    full = 4093

    def slot_bytes(rows):
        caches = 13
        pages = (rows + 127) // 128
        return (caches * pages * 128 * 2 * 544 + caches * ((rows + 3) // 4) * 128 * 2
                + (caches * 8 + 8) * (128 * 2 + 3 * 8) + caches * 8 * pages * 8)

    def ledgers(**kw):
        return [indexed_stream_geometry(TEXT, 4, 4, 8, mtp=True, world=8, prefill_rows=256,
                kv_pair=((8, 32, 2),) * 2, execution_rows=32, history_placement=placement, rank=rank,
                **kw).breakdown_at(full) for rank in range(8)]

    mixed = ledgers(full_windows=1, short_slots=short)
    uniform = ledgers(full_windows=4)
    assert sum(ledger["history"] for ledger in mixed) == slot_bytes(full) + 3 * slot_bytes(short)
    assert sum(ledger["history"] for ledger in uniform) == 4 * slot_bytes(full)
    # Only the cache term changes; fixed execution, recovery and workspace are unchanged.
    for a, b in zip(mixed, uniform):
        assert {k: v for k, v in a.items() if k != "history"} == {k: v for k, v in b.items() if k != "history"}


@pytest.mark.parametrize("kw", [dict(full_windows=4, short_slots=100), dict(full_windows=None, short_slots=100),
                                dict(full_windows=1, short_slots=0)])
def test_geometry_refuses_short_slots_outside_the_mixed_unique_history_layout(kw):
    with pytest.raises(ValueError, match="short resident windows"):
        indexed_stream_geometry(TEXT, 4, 4, 8, mtp=True, world=8, prefill_rows=256, kv_pair=((8, 32, 2),) * 2,
                                execution_rows=32, history_placement=HistoryPlacement(8, 2, 12), **kw)
    with pytest.raises(ValueError, match="short resident windows"):
        indexed_stream_geometry(TEXT, 4, 4, 8, mtp=True, world=8, prefill_rows=256, kv_pair=((8, 32, 2),) * 2,
                                execution_rows=32, full_windows=1, short_slots=100)


# ---------------------------------------------------------------- materialization (actual constructor block)

def test_constructor_materializes_every_state_at_its_own_limit_with_shared_logical_identity():
    from tests.test_resident_window_materialization import Array, allocation_block

    calls = []

    class State:
        def __init__(self, w, rows, max_rows, kv_dtype, *, limit, **kwargs):
            self.capacity, self.limit = rows, limit
            self.dspark_runtime = None
            self.array = Array(self.cache_bytes())
            calls.append((rows, limit, kwargs["logical_context"], kwargs.get("short_window", False), kwargs["slot"]))

        def cache_bytes(self, rows=None):
            return (self.capacity if rows is None else rows) * 100

        def layer_bytes(self, rows):
            return rows * 10

    file, module = allocation_block()
    ns = {"State": State, "_tensors": lambda state: [state.array], "FIRST": 256, "GIB": GIB,
          "context_reservation": context_reservation, "MemoryGate": MemoryGate,
          "torch": NS(cuda=NS(is_available=lambda: False)), "memory_live": None, "native_gdn_batch": False}
    exec(compile(ast.fix_missing_locations(module), str(file), "exec"), ns)
    dec = NS(controller=None, media_runtime=None)
    limits = (262149, 262149, 65541, 65541)
    ns["prepare"](dec, NS(), 4, 262149, 4, "int8", None, None, None, 0, 4, 262144, limits)
    assert calls == [(262149, 262149, 262144, False, 0), (262149, 262149, 262144, False, 1),
                     (65541, 65541, 262144, True, 2), (65541, 65541, 262144, True, 3)]
    assert dec.full_resident and dec.slot_limits == limits
    assert [st.capacity for st in dec.free] == list(limits)
    p = context_reservation(dec.free, 4, preserve_overlap=True)
    assert p["growth_bytes"] == 0 and p["full_windows_reserved"] == 4


# ---------------------------------------------------------------- placement and queueing

class Slot:
    def __init__(self, name, limit):
        self.name, self.limit, self.copied, self.generations = name, limit, None, 0
        self.compact_indexer = None

    def begin_sequence_generation(self):
        self.generations += 1

    def copy_prefix(self, source, pos, mtp_len):
        self.copied = (source, pos, mtp_len)

    def __repr__(self):
        return self.name


def owner(free=(), kept=(), busy=()):
    o = NS(free=list(free), kept=list(kept), used={}, clock=0, depth=4, keep=8, grown=[])
    o._busy = lambda: {id(st) for st in busy}
    o._drop_kept = lambda st: setattr(o, "kept", [k for k in o.kept if k[1] is not st])

    def grow(st, rows, **kwargs):
        o.grown.append((st, rows))
        return True
    o._grow = grow
    return o


def test_best_fit_places_short_requests_in_short_slots_and_keeps_full_windows_free():
    a, b, c, d = Slot("full-a", 1000), Slot("short-b", 300), Slot("full-c", 1000), Slot("short-d", 300)
    o = owner([a, b, c, d])
    assert prefixes.slot_for(o, [1] * 10, False, extent=300)[0] is d      # shortest fitting, last among equals
    assert prefixes.slot_for(o, [1] * 10, False, extent=301)[0] is c      # longer extents take a full window
    assert prefixes.slot_for(o, [1] * 10, False, extent=20)[0] is b
    assert o.free == [a]
    # Uniform placement (no extent) is unchanged: the last free slot.
    o = owner([a, b, c, d])
    assert prefixes.slot_for(o, [1] * 10, False)[0] is d


def test_request_waits_when_only_shorter_windows_are_free():
    full, short = Slot("full", 1000), Slot("short", 300)
    o = owner([short], kept=[([7, 7], full, {}, None)], busy=[full])
    assert not prefixes.has_room(o, 301) and prefixes.has_room(o, 300)
    with pytest.raises(NoRoom, match="301-row request waits"):
        prefixes.slot_for(o, [1] * 10, False, extent=301)
    assert o.free == [short] and short.generations == 0


def test_prefix_reuse_forks_into_a_fitting_slot_and_never_resumes_a_short_one_past_its_window():
    short, spare = Slot("short", 300), Slot("full", 1000)
    prefix = list(range(200))
    snap = {"mtp_len": 199}
    o = owner([spare], kept=[(prefix, short, snap, "tail")])
    st, resume, cached = prefixes.slot_for(o, prefix + [5] * 50, True, extent=600)
    assert st is spare and cached == 200 and spare.copied == (short, 200, 199)
    assert resume == {"state": snap, "tail": "tail"}
    assert o.kept == [(prefix, short, snap, "tail")]                    # the short slot keeps its chain
    # A fitting kept slot resumes in place, exactly as before.
    full = Slot("full", 1000)
    o = owner([], kept=[(prefix, full, snap, "tail")])
    st, resume, cached = prefixes.slot_for(o, prefix + [5] * 50, True, extent=600)
    assert st is full and cached == 200 and full.copied is None


def test_eviction_only_takes_an_idle_slot_whose_window_holds_the_request():
    short, full = Slot("short", 300), Slot("full", 1000)
    o = owner([], kept=[([3], full, {}, None), ([4], short, {}, None)])
    o.used = {id(short): 0, id(full): 5}                                  # the short chain is worth least
    assert prefixes.slot_for(o, [9] * 10, False, extent=400)[0] is full
    assert [k[1] for k in o.kept] == [short]


# ---------------------------------------------------------------- receipts and guards

def receipt(slots=32, full=24, short=65541):
    return {"native_window": 262144, "context_window": 262144, "cache_slots": 262149,
            "session_capacity": {"policy": MIXED_SESSION_POLICY, "context_tokens": 262144,
                                 "resident_capacity": slots, "full_context_commitments": full,
                                 "short_windows": {"cache_slots": short},
                                 "slot_capacities": [262149] * full + [short] * (slots - full)}}


def test_mixed_receipt_records_and_guards_per_slot_capacities():
    plan = receipt()
    assert mixed_slot_capacities(plan, 32) == [262149] * 24 + [65541] * 8
    assert mixed_slot_capacities({"session_capacity": {"policy": "fixed-context-worst-rank-v1"}}, 32) is None
    assert full_window_headroom_required(plan, key_bits=8, value_bits=8, slots=32, full_windows=32,
                                         physical_capacity=262149)
    for broken in (dict(slots=31), dict(full=32), dict(short=262149)):
        with pytest.raises(ValueError, match="mixed-window session receipt"):
            mixed_slot_capacities(receipt(**broken), 32)
    tampered = receipt()
    tampered["session_capacity"]["slot_capacities"][30] = 262149
    with pytest.raises(ValueError, match="mixed-window session receipt"):
        mixed_slot_capacities(tampered, 32)


def test_graph_room_proof_checks_each_state_at_its_own_admitted_limit():
    states = [NS(capacity=262149, limit=262149), NS(capacity=65541, limit=65541)]
    promise = {"growth_bytes": 0, "full_windows_reserved": 2}
    future = {"growth_bytes": 0, "copy_overlap_bytes": 1, "snapshot_bytes": 1, "workspace_bytes": 2 * GIB,
              "future_bytes": 2 * GIB + 2}
    assert materialized_graph_room(states, [262149, 65541], 3 * GIB, promise, future, gpu_history=True) == GIB - 2
    with pytest.raises(ValueError, match="materialized"):
        materialized_graph_room(states, 262149, 3 * GIB, promise, future, gpu_history=True)
    with pytest.raises(ValueError, match="materialized"):
        materialized_graph_room(states, [262149, 65541, 1], 3 * GIB, promise, future, gpu_history=True)


def test_short_window_destination_copies_only_rows_within_its_own_storage():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tests.test_flashnext_prefix_copy import slot

    src, dst = slot("int8", 64, 50), slot("int8", 32, 0)
    dst.logical_context = 64                                  # short window: shared logical identity, less storage
    dst.copy_prefix(src, 30, 29)
    assert torch.equal(dst.kc[0].k[:30], src.kc[0].k[:30])
    with pytest.raises(ValueError, match="fit the destination"):
        dst.copy_prefix(src, 33, 32)


def test_mixed_decoder_queues_a_request_no_free_window_holds():
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.cuda.streams import Stream
    from tests.test_flashnext_prefix_fork import decoder

    dec, source, spare, prefix, _, _ = decoder(busy=True)
    spare.limit = 300                                          # the only free slot is a short window
    dec.slot_limits = (1024, 300)
    kept = list(dec.kept)
    with pytest.raises(NoRoom, match="waits for a slot whose resident window holds it"):
        dec.admit(Stream(prefix + [6, 7, 8], 8))
    assert dec.free == [spare] and dec.kept == kept and spare.copied is None


# ---------------------------------------------------------------- bounded window bypass (actual FIFO scheduler)

def _window_scheduler(limits, bypass_limit):
    import threading

    from tensorfold.cuda.scheduler import Scheduler
    from tests.test_cuda_fifo_admission import Decoder

    class WindowDecoder(Decoder):
        """Controlled resident windows: a request's extent is prompt + reply; it needs a free slot that holds it."""

        def __init__(self):
            super().__init__(pause=99)
            self.limits, self.owner = list(limits), [None] * len(limits)
            self.window_bypass_limit = bypass_limit
            self.queries = []

        def validate_request(self, stream):
            if not stream.prompt or len(stream.prompt) + stream.count > max(self.limits):
                raise ValueError("controlled infeasible context")

        def _fits(self, stream):
            need = len(stream.prompt) + stream.count
            return [i for i, n in enumerate(self.limits) if self.owner[i] is None and n >= need]

        def window_blocked(self, stream):
            self.queries.append((stream.prompt[0], threading.current_thread().name))
            return not self._fits(stream) and any(o is None for o in self.owner)

        def admit(self, stream):
            fitting = self._fits(stream)
            if not fitting:
                raise NoRoom("controlled window wait")
            slot = min(fitting, key=lambda i: (self.limits[i], -i))
            self.owner[slot] = stream
            super().admit(stream)

        def finish(self, streams):
            for stream in streams:
                self.owner = [None if o is stream else o for o in self.owner]
            super().finish(streams)

    decoder = WindowDecoder()
    scheduler = Scheduler(decoder, max_streams=len(limits), max_pending_requests=8, admission_policy="fifo",
                          max_waiting_input_bytes=4096).start()
    return decoder, scheduler


def _run_window_case(limits, bypass_limit, shorts=2):
    from tests.test_cuda_fifo_admission import client, joined

    decoder, scheduler = _window_scheduler(limits, bypass_limit)
    try:
        active = client(scheduler, 99, prompt=[99] * 2, count=6)    # occupies the only full window
        assert decoder.in_round.wait(3)
        head = client(scheduler, 5, prompt=[5] * 4, count=4)        # extent 8: needs the full window
        later = []
        for index in range(shorts):
            with scheduler.waiting.not_empty:
                assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision >= 2 + index, 3)
            later.append(client(scheduler, 7 + index, prompt=[7 + index], count=1))   # extent 2: fits a short slot
        with scheduler.waiting.not_empty:
            assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision >= 2 + shorts, 3)
        decoder.release_round.set()
        joined(active, head, *later)
    finally:
        decoder.release_round.set()
        scheduler.close()
    assert active.errors == head.errors == [] and all(item.errors == [] for item in later)
    return [token for token, _ticket, _active in decoder.admissions], decoder


def test_short_requests_pass_a_head_that_waits_only_for_a_full_window():
    order, decoder = _run_window_case([10, 3, 3], 2)
    assert order == [99, 7, 8, 5]                      # both shorts start while the long head waits
    assert decoder.queries and all(token == 5 for token, _thread in decoder.queries)   # only the head is assessed


def test_bypass_count_is_bounded_per_waiting_head():
    order, _ = _run_window_case([10, 3, 3], 1)
    assert order == [99, 7, 5, 8]                      # one pass allowed; then strict FIFO behind the head


def test_head_takes_the_next_full_window_before_any_later_request():
    # Shorts keep arriving and fitting; the head still owns the first full window that frees.
    order, decoder = _run_window_case([10, 3, 3, 3], 8, shorts=4)
    assert order == [99, 7, 8, 9, 10, 5]
    head = next(item for item in decoder.admissions if item[0] == 5)
    assert 99 not in head[2]                           # admitted at the pass the full window freed, not later


def test_bypass_order_is_deterministic():
    assert _run_window_case([10, 3, 3], 2)[0] == _run_window_case([10, 3, 3], 2)[0]


def test_uniform_slots_keep_strict_fifo():
    order, decoder = _run_window_case([10, 3, 3], 0)
    assert order == [99, 5, 7, 8] and decoder.queries == []


def test_decoder_reports_window_waits_and_one_short_pool_turnover_as_the_bypass_bound():
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    full, short = Slot("full", 1000), Slot("short", 300)
    dec = object.__new__(MultiDecoder)
    dec.slot_limits, dec.depth, dec.capacity = (1000, 1000, 300, 300, 300), 4, 1000
    dec.free, dec.kept, dec.used, dec.clock = [short], [], {}, 0
    dec._busy = lambda: {id(full)}
    assert dec.window_bypass_limit == 3
    assert dec.window_blocked(NS(prompt=[1] * 400, count=10))            # a free slot, none long enough
    assert not dec.window_blocked(NS(prompt=[1] * 10, count=10))         # fits the free short slot
    dec.kept = [([1] * 50, full, {}, None)]
    dec._busy = lambda: set()
    assert not dec.window_blocked(NS(prompt=[1] * 400, count=10))        # an idle kept full slot: the head evicts it
    dec.kept, dec._busy = [], lambda: {id(full)}
    dec.free = []
    assert not dec.window_blocked(NS(prompt=[1] * 400, count=10))        # nothing free: ordinary capacity wait
    dec.slot_limits = (1000,) * 5
    assert dec.window_bypass_limit == 0 and not dec.window_blocked(NS(prompt=[1] * 400, count=10))
