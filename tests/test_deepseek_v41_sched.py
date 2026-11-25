"""DeepSeek-V4.1 SM70 prompt layer groups on CPU: the half-layer pause points of ``Target.layers`` (a paused
forward runs the same operations on the same values as an unpaused one), the cost estimates the Decoder cuts
groups from (``schedule``) and its per-round cut."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_w576_experimental_math_flag_is_independent_of_packet_format():
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine, F_AHEAD, F_COMPACT, F_SCORE, F_W576_FP32
    from tensorfold.families.deepseek_v4.cuda.graphs import Shape

    engine = object.__new__(DS41Engine)
    engine.target = SimpleNamespace(device=torch.device("cpu"), rotor_cache=object(), expert_usage=object(),
                                    pool=SimpleNamespace(owned={0: [1]}))
    engine.expert_observer = None
    engine._experimental_w576_fp32 = True
    engine.geometry = SimpleNamespace(bos=7)
    engine.slots, engine.rank, engine.slot_pages = 2, 1, SimpleNamespace(width=2)
    shape = Shape(engine, 1, 131072)
    entries, ids = [(0, 3, 1)], [42]

    def replay(entries, ids, compact, *, full, ahead, expert_sample, retain_routed):
        shape.stage(entries, ids, 2, retain_routed=retain_routed)
        return (compact, ahead, shape.step.round_w576_routed.item(), shape.step.expert_sample.item(),
                shape.step.ids.item())

    engine.graphs = SimpleNamespace(sizes=[1], run=replay)
    assert engine._step(entries, ids, [0], 0) == (False, False, 1, 2, 42)
    assert engine._step(entries, ids, [0], F_COMPACT) == (True, False, 1, 2, 42)
    assert engine._step(entries, ids, [0], F_W576_FP32) == (False, False, 0, 2, 42)
    assert engine._step(entries, ids, [0], F_COMPACT | F_AHEAD | F_W576_FP32) == (True, True, 0, 2, 42)
    assert engine._step(entries, ids, [0], F_COMPACT) == (True, False, 1, 2, 42)
    with pytest.raises(ValueError, match="continuation frames"):
        engine._step(entries, ids, [0], F_SCORE | F_W576_FP32)

    engine._experimental_w576_fp32 = False
    stock = Shape(engine, 1, 131072)
    stock.stage(entries, ids, 2)
    assert stock.step.round_w576_routed is None and stock.host.shape == (4, 1)
    with pytest.raises(ValueError, match="continuation frames"):
        engine._step(entries, ids, [0], F_W576_FP32)


# ----------------------------------------------------------------------------------------------- schedule

def test_unit_costs_interpolate_inside_and_bound_outside_the_anchors():
    from tensorfold.families.deepseek_v4.cuda.schedule import UnitCosts
    c = UnitCosts(256, [10.0, 20.0])
    assert c.predict(512) == [20.0, 40.0]               # one anchor: proportional above it (an overestimate)
    assert c.predict(64) == [10.0, 20.0]                # and its own time below it (a unit never gets dearer)
    c.add(4096, [100.0, 40.0])
    got = c.predict(2048)                                # linear between (256, a) and (4096, b)
    assert got[0] == pytest.approx(10 + 90 * (2048 - 256) / 3840)
    assert got[1] == pytest.approx(20 + 20 * (2048 - 256) / 3840)
    assert c.predict(8192) == pytest.approx([200.0, 80.0])   # above the large anchor: its time per row
    c.add(1000, [1.0, 1.0])                              # neither anchor's size: no change
    assert c.small[0] == 256 and c.large[0] == 4096
    c.add(300, [1.0, 1.0])                               # at most twice the small anchor's rows: the small one
    assert c.small == (300, [1.0, 1.0])
    c.add(8192, [200.0, 80.0])
    assert c.large == (8192, [200.0, 80.0])
    c.add(5000, [1.0, 1.0])                              # at least half the large anchor's rows: the large one
    assert c.large[0] == 5000
    with pytest.raises(ValueError):
        c.add(512, [1.0])
    # a measured large chunk cheaper per unit than the warm-up never predicts below the warm-up
    d = UnitCosts(256, [50.0])
    d.add(1024, [40.0])
    assert d.predict(512) == [50.0] and d.predict(8192) == [320.0]


def test_small_chunks_never_set_the_cost_of_a_large_one():
    """Short chat turns (14..57 rows) cost the fixed launch time per unit, flat within noise; their two anchors' slope
    put a 4096-row chunk near their own cost, so it ran in one group (a 1.4 s stall) while streams decoded. The
    estimate beyond the largest measured chunk is now that chunk's time per row, and the warm-up stays an anchor."""
    from tensorfold.families.deepseek_v4.cuda.schedule import UnitCosts
    warm = [2.2] * 80                                    # 256 rows
    c = UnitCosts(256, warm)
    for rows, ms in ((18, 1.26), (14, 1.25), (35, 1.27), (31, 1.24), (44, 1.24), (57, 1.21)):
        c.add(rows, [ms + 0.02 * (u % 3) for u in range(80)])     # noise across units
    assert sum(c.predict(4096)) >= sum(warm) * 4096 / 256
    big = [17.0] * 80
    c.add(4096, big)
    c.add(40, [1.25] * 80)                               # a later short turn keeps the large anchor
    assert c.large == (4096, big) and c.predict(4096) == big


def test_round_costs_bounds_for_unmeasured_buckets():
    from tensorfold.families.deepseek_v4.cuda.schedule import DEFAULT_ROUND_MS, RoundCosts
    r = RoundCosts()
    assert r.estimate(8) == DEFAULT_ROUND_MS
    r.add(16, 66.0)
    assert r.estimate(16) == 66.0
    assert r.estimate(24) == pytest.approx(66.0 * 24 / 16)       # narrower bucket scaled: an upper bound
    assert r.estimate(8) == 66.0                                 # a wider bucket's time bounds a narrower one
    r.add(24, 81.0)
    assert r.estimate(24) == 81.0
    r.add(24, 91.0)
    assert r.estimate(24) == pytest.approx(0.8 * 81 + 0.2 * 91)
    assert r.estimate(20) == pytest.approx(min(r.estimate(24), 66.0 * 20 / 16))


# ----------------------------------------------------------------------------------------------- unit pauses

def _fake_target(monkeypatch, layers=5, engram=(2,)):
    from tensorfold.families.deepseek_v4.cuda import target as T
    from tensorfold.families.deepseek_v4.cuda.kernels import mhc, mhc_fused
    log = []

    def hc_mixes(hidden, fn, scale, base):
        log.append(("mixes", fn))
        s = torch.tanh(hidden.mean(-1))
        return (s * scale, s + base, s * 0.5)

    def collapse_hc(hidden, pre, norm):
        log.append(("collapse", norm))
        return torch.tanh((hidden * pre[..., None]).sum(1) * norm)

    def hc_transition(hidden, y, post, comb, pre, norm, fn=None, scale=None, base=None):
        log.append(("transition", norm, fn))
        hidden.copy_(torch.tanh(hidden + y[:, None] * post[..., None] + comb[..., None]))   # in place, as the kernel
        inputs = torch.tanh((hidden * pre[..., None]).sum(1) * norm)
        if fn is None:
            return inputs, None
        s = torch.tanh(hidden.mean(-1))
        return inputs, (s * scale, s + base, s * 0.25)

    def expand_hc(y, hidden, post, comb, inplace=True):
        log.append(("expand",))
        hidden.copy_(torch.tanh(hidden + y[:, None] * post[..., None] - comb[..., None]))
        return hidden

    monkeypatch.setattr(mhc, "hc_mixes", hc_mixes)
    monkeypatch.setattr(mhc, "collapse_hc", collapse_hc)
    monkeypatch.setattr(mhc, "expand_hc", expand_hc)
    monkeypatch.setattr(mhc_fused, "hc_transition", hc_transition)
    t = object.__new__(T.Target)
    hc = lambda i: {k: float(i + j) / 7 for j, k in enumerate(
        ("attn_fn", "attn_scale", "attn_base", "ffn_fn", "ffn_scale", "ffn_base"))}
    t.w = SimpleNamespace(norm=0.75, layers=[
        SimpleNamespace(index=i, engram_wkv=object() if i in engram else None, ratio=i % 2, hc=hc(i),
                        attn_norm=1 + i / 10, ffn_norm=1 - i / 20) for i in range(layers)])
    t.compressed_rope, t.window_rope = "compressed", "window"
    t.attention = lambda ly, x, step, table, shared, shard: (log.append(("attention", ly.index, table)), x * (1 + ly.index / 9) + 1)[1]
    t.moe = lambda ly, x, image, shard, rows: (log.append(("moe", ly.index)), x * 0.5 - ly.index / 3)[1]
    t.engram_layer = lambda ly, hidden, hashes, valid, shard: (log.append(("engram", ly.index)), hidden * 1.01 + 0.5)[1]
    return t, log


def _run(t, stops):
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(6, 4, 8, generator=g, dtype=torch.float64)
    pre = torch.zeros(6, 4, dtype=torch.float64)
    pre[:, 0] = 1
    st = {"step": None, "hashes": None, "hidden": hidden, "pre": pre, "shared": {"topk": None}, "mix": None,
          "half": None, "next": 0, "result": None,
          "valid": None, "image": None, "shard": None}
    for stop in stops:
        t.layers(st, stop)
    assert st["next"] == t.units
    return st["result"], st["hidden"]


@pytest.mark.parametrize("stops", [[1], [2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 7], [5, 6, 9]])
def test_paused_forward_runs_the_same_operations_on_the_same_values(monkeypatch, stops):
    t, log = _fake_target(monkeypatch)
    assert t.units == 10
    want, want_hidden = _run(t, [t.units])
    assert torch.isfinite(want).all()
    want_log = list(log)
    log.clear()
    got, got_hidden = _run(t, [s for s in stops if s < t.units] + [t.units])
    assert log == want_log
    assert torch.equal(got, want) and torch.equal(got_hidden, want_hidden)


def test_marks_time_each_unit(monkeypatch):
    t, _ = _fake_target(monkeypatch, layers=2, engram=())

    class Event:
        def __init__(self, **_):
            pass

        def record(self):
            pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    st = {"step": None, "hashes": None, "hidden": torch.ones(2, 4, 8), "pre": torch.ones(2, 4), "shared": {},
          "mix": None, "half": None, "next": 0, "result": None,
          "valid": None, "image": None, "shard": None}
    marks = []
    t.layers(st, 1, marks)
    t.layers(st, 4, marks)
    assert [(first, len(events)) for first, events in marks] == [(0, 2), (1, 4)]


# ----------------------------------------------------------------------------------------------- the cut

class _Event:
    def __init__(self, at, done=True):
        self.at, self.done = at, done

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.at - self.at


def _decoder(units=8, streams=1):
    from tensorfold.families.deepseek_v4.cuda.engine import Decoder
    from tensorfold.families.deepseek_v4.cuda.schedule import RoundCosts, UnitCosts
    d = object.__new__(Decoder)
    d.e = SimpleNamespace(units=units, graphs=SimpleNamespace(sizes=(1, 2, 4, 8, 16, 24, 32)))
    d.decode_math = 0                           # Stock Decoder.__init__ mode.
    d.GAP_MS = 175.0
    d.streams = {i: object() for i in range(streams)}
    d.costs = UnitCosts(100, [25.0] * units)           # 100 rows: 25 ms per unit
    d.rounds = RoundCosts()
    d.rounds.add(1, 75.0)                              # target 100 ms: four units per group
    d.timing = (100, [])
    return d


def test_cut_is_every_unit_without_decoding_streams():
    d = _decoder(streams=0)
    assert d._stop(100, 0) == 8


def test_cut_follows_the_gap_and_this_chunks_measured_progress():
    d = _decoder()
    assert d._stop(100, 0) == 4                        # 8 x 25 ms in groups of at most 100 ms
    # the first group took twice its prediction (a later chunk of a long prompt): the rest is cut finer
    d.timing[1].append((0, [_Event(0.0)] + [_Event(50.0 * k) for k in range(1, 5)]))
    assert d._stop(100, 4) == 6
    # an unfinished group is not read (never waits)
    d.timing[1][0] = (0, [_Event(0.0)] + [_Event(50.0 * k, done=k < 4) for k in range(1, 5)])
    assert d._stop(100, 4) == 8
    # a wider decode batch (bucket 24, never measured: bounded by bucket 1 x 24) leaves the floor, a quarter gap
    d.timing[1].clear()
    d.streams = {i: object() for i in range(20)}
    assert d._stop(100, 0) == 1                        # 25 ms units under a 43.75 ms target


def test_bucket():
    d = _decoder()
    assert [d._bucket(n) for n in (1, 3, 9, 17, 32)] == [1, 4, 16, 24, 32]
    d.e.graphs = None
    assert d._bucket(9) == 9


@pytest.mark.parametrize("tokens", [57, 64, 1024, 1031, 4100, 8192])
def test_shards_cover_every_row_once_and_none_is_empty(tokens):
    from tensorfold.families.deepseek_v4.cuda.target import Shard
    shards = [Shard.of(tokens, r, 8) for r in range(8)]
    assert all(s.rows >= 1 and s.size == shards[0].size for s in shards)
    covered = [i for s in shards for i in range(s.first, s.first + s.rows)]
    assert covered == list(range(tokens))
    x = torch.arange(tokens * 2, dtype=torch.float16).view(tokens, 2)
    padded = shards[0].padded(x)
    assert padded.shape == (8 * shards[0].size, 2) and padded.dtype == torch.float32
    assert torch.equal(padded[:tokens], x.float()) and not padded[tokens:].any()


def test_engram_layer_with_a_shard_looks_up_every_row_and_gates_its_own(monkeypatch):
    """K15: the column-sharded Engram projection needs every row on every rank; the gate takes the shard's rows."""
    from tensorfold.families.deepseek_v4.cuda import target as T
    from tensorfold.families.deepseek_v4.cuda.kernels import engram as kengram
    tokens, h = 64, 4
    shard = T.Shard.of(tokens, 3, 8)
    seen = {}

    class Engram:
        def lookup(self, hashes, channel, out):
            assert out.shape[0] == hashes.shape[0] == tokens and out.is_contiguous()
            out.copy_(hashes[:, 0, :1].to(out.dtype).expand_as(out))      # row r carries r
            return out

    def gate(hidden, projected, q, k, valid, inplace=False):
        seen.update(hidden=hidden.shape, projected=projected.clone(), valid=valid.clone())
        return hidden

    monkeypatch.setattr(kengram, "apply_engram_gate", gate)
    t = object.__new__(T.Target)
    t.g = SimpleNamespace(engram_layers=(1,))
    t.engram = Engram()
    t.engram_rows = torch.empty(tokens, 6, dtype=torch.float16)
    t.c = SimpleNamespace(gather=lambda x: x)
    t._linear = lambda x, w: x
    hashes = torch.arange(tokens)[:, None, None].expand(tokens, 2, 24).contiguous()
    step = SimpleNamespace(valid=torch.arange(tokens) % 5 != 0)
    hidden = torch.zeros(shard.rows, 4, h)
    ly = SimpleNamespace(index=1, engram_wkv=None, engram_q=None, engram_k=None)
    t.engram_layer(ly, hidden, hashes, step.valid, shard)
    rows = torch.arange(shard.first, shard.first + shard.rows)
    assert seen["hidden"] == (shard.rows, 4, h)
    assert torch.equal(seen["projected"][:, 0], rows.half())
    assert torch.equal(seen["valid"], step.valid[rows])


# ----------------------------------------------------------------------------------------------- the round timeline

class _Timeline:
    """One GPU stream and the rank-0 host on a modelled clock (ms): work queues behind the stream's last work;
    an event completes when the work queued before it did; a host wait advances the host to it."""

    def __init__(self):
        self.now = self.gpu = 0.0

    def run(self, ms, host=0.2):
        self.now += host
        self.gpu = max(self.gpu, self.now) + ms
        return self.gpu


def _timeline_decoder(monkeypatch, unit_ms, round_ms=20.0, units=8, warm_rows=256):
    """The real Decoder over a fake engine: decode rounds of ``round_ms``, prompt units of ``unit_ms(rows)``."""
    from tensorfold.families.deepseek_v4.cuda import engine as E
    from tensorfold.families.deepseek_v4.cuda import round_sampling as RS
    from tensorfold.families.deepseek_v4.cuda.schedule import RoundCosts, UnitCosts
    clock = _Timeline()

    class Event:
        def __init__(self, **_):
            self.t = None

        def record(self):
            self.t = max(clock.gpu, clock.now)

        def query(self):
            return clock.now >= self.t

        def synchronize(self):
            clock.now = max(clock.now, self.t)

        def elapsed_time(self, other):
            return other.t - self.t

    class Logits:                                   # a chunk's last-row logits: reading them waits for the chunk
        def __init__(self):
            self.t = clock.gpu

        def __getitem__(self, _):
            return self

    class Engine:
        def __init__(self):
            self.units, self.graphs, self.layer_marks, self.image_features = units, SimpleNamespace(sizes=(1, 2, 4)), None, None
            self.rows = self.at = None

        def _units(self, stop):
            events = None
            if self.layer_marks is not None:
                events = [Event()]
                events[0].record()
                self.layer_marks.append((self.at, events))
            for _ in range(self.at, stop):
                clock.run(unit_ms(self.rows))
                if events is not None:
                    events.append(Event())
                    events[-1].record()
            self.at = stop
            return Logits() if stop == self.units else None

        def step(self, entries, ids, logits_rows, flags=0):
            if flags & E.F_COMPACT:
                clock.run(round_ms)
                return ("packet", None)
            stop = flags & 0xFFFF
            self.rows, self.at = sum(c for _, _, c in entries), 0
            return self._units(stop if 0 < stop < self.units else self.units)

        def announce(self, op, stop, *_):
            assert op == E.OP_LAYERS
            return self._units(stop)

    monkeypatch.setattr(E, "time", SimpleNamespace(perf_counter=lambda: clock.now / 1e3))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(RS, "greedy_tokens", lambda packet, n: [(7, None)] * n)
    d = object.__new__(E.Decoder)
    d.e = Engine()
    d.GAP_MS, d.streams, d.filling, d._broken, d.eos = 175.0, {}, [], None, ()
    d.inflight = d.ahead = d.split = d.finishing = d.timing = None
    d.timed = []
    d.costs = UnitCosts(warm_rows, [unit_ms(warm_rows)] * units)
    d.rounds = RoundCosts()
    d.events = [(Event(), Event()) for _ in range(2)]
    d.flip = 0
    d.keep_before = lambda s, start: None

    def choose(s, row, position, emitted=0):          # sampling the first token reads the chunk's logits
        clock.now = max(clock.now, row.t)
        return 7
    d._choose = choose
    return d, clock


class _Stream:
    def __init__(self, sid, rows, count, clock):
        self.sid, self.slot, self.prompt, self.count, self.bounds, self.clock = sid, sid, [1] * rows, count, [], clock
        self.pos, self.head, self.vision, self.background, self.probabilities = 0, None, None, False, None
        self.prefill_s, self.keep_end, self.on_fill, self.forced = 0.0, False, None, None
        self.constraint = self.history = self.sampling = None
        self.done, self.stop_eos, self.out, self.times = False, False, [], []

    def take(self, new, eos=()):
        self.out += new
        self.times += [self.clock.now] * len(new)
        self.done = len(self.out) >= self.count

    def counted(self, rows):
        pass


def _serve(d, streams):
    d.filling += streams
    while d.live():
        for s in d.round():
            d.streams.pop(s.sid, None)


def test_a_cancelled_one_row_prompt_does_not_supply_run_ahead_tokens(monkeypatch):
    from tensorfold.families.deepseek_v4.cuda import engine as E

    d, clock = _timeline_decoder(monkeypatch, lambda rows: 1.0)
    step = d.e.step
    last_graph_slot = None

    def replay(entries, ids, logits_rows, flags=0):
        nonlocal last_graph_slot
        if len(entries) == 1 and entries[0][2] == 1 and logits_rows == [0]:
            slot = entries[0][0]
            if flags & E.F_AHEAD:
                assert last_graph_slot == slot, "run-ahead took a token from another stream's replay"
            last_graph_slot = slot
        return step(entries, ids, logits_rows, flags)

    d.e.step = replay
    decoding = _Stream(0, 64, 10, clock)
    d.filling.append(decoding)
    while not decoding.out:
        d.round()

    cancelled = _Stream(1, 1, 2, clock)
    d.filling.append(cancelled)
    d.round()
    assert d.inflight is not None and last_graph_slot == cancelled.slot
    assert d.cancel([cancelled]) == [cancelled]

    _serve(d, [])
    assert decoding.out == [7] * decoding.count and cancelled.out == []


def test_a_chunks_last_group_does_not_hold_back_a_decoded_token(monkeypatch):
    """A decoding stream waits about one group plus a round between tokens, also where a prompt chunk ends: the
    chunk's result is read after the round's decided tokens are delivered, not before (which held the token decided
    behind the previous group until the last group finished as well: two groups and a round)."""
    unit = 40.0
    d, clock = _timeline_decoder(monkeypatch, lambda rows: unit * rows / 4096)
    decoding = _Stream(0, 64, 40, clock)
    _serve(d, [decoding])                       # learns the unit cost of a 64-row chunk, decodes to the end
    decoding = _Stream(1, 64, 60, clock)
    d.filling.append(decoding)
    while not decoding.out:
        d.round()
    filling = [_Stream(2 + i, 4096, 2, clock) for i in range(3)]
    _serve(d, filling)
    assert all(s.done for s in filling)
    gaps = [b - a for a, b in zip(decoding.times, decoding.times[1:])]
    # groups of at most 175 - 20 ms: 3 units (120 ms); one group, one round and host work per gap
    assert max(gaps) <= 3 * unit + 20.0 + 10.0, gaps


def test_a_chunk_measurement_unread_when_the_next_chunk_starts_is_kept(monkeypatch):
    """Under load a chunk run whole (or any chunk the host did not wait for) is still running when the next one
    starts; its per-unit costs are read once its events complete instead of being replaced by the next chunk's."""
    d, clock = _timeline_decoder(monkeypatch, lambda rows: 10.0)
    read = []
    add = d.costs.add
    d.costs.add = lambda rows, ms: (read.append((rows, ms)), add(rows, ms))
    for rows in (4096, 8192):
        d.timing = (rows, [])
        d.e.layer_marks = d.timing[1]
        d.e.rows, d.e.at = rows, 0
        d.e._units(d.e.units)
        d._timed()
        d._absorb()                              # still running: nothing read yet, nothing dropped
        assert read == []
    clock.now = clock.gpu
    d._absorb()
    assert d.timed == [] and [rows for rows, _ in read] == [4096, 8192]     # both read, oldest first
    assert all(ms == pytest.approx([10.0] * 8, abs=0.5) for _, ms in read)


def test_the_warm_up_cost_sample_is_not_the_first_prompt_forward(monkeypatch):
    """The first prompt forward of the process carries first-use host stalls on its first units (kernel loads and
    compiles, library handles, allocator growth). Timed as the Decoder's first cost sample (the small anchor that
    4096-row chunks were planned from in a session without short prompts), it cut unit 0 and unit 1 alone and then,
    the chunk's progress scaled by their inflated predictions, ran every other unit in one group: 1.0-1.4 s decode
    gaps at C24 4K (fin-h2, i5-h2). The sample is timed on a second identical chunk."""
    from tensorfold.cuda.sm70 import ds41_lattice_moe, ds41_sparse_attention, onecat_ds41
    from tensorfold.families.deepseek_v4.cuda import engine as E
    for module in (ds41_lattice_moe, ds41_sparse_attention, onecat_ds41):
        monkeypatch.setattr(module, "prepare", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    e = object.__new__(E.DS41Engine)
    e.rank, e.tower, e.geometry, e.layer_marks = 0, None, SimpleNamespace(bos=0), None
    from tensorfold.families.deepseek_v4.cuda.startup import Timeline
    e.timeline = Timeline(0)                # the startup phase clock _warm reports into (ds41-start)
    prompts = []

    class Event:
        def __init__(self, at):
            self.at = at

        def elapsed_time(self, other):
            return other.at - self.at

    def step(entries, ids, rows, stop=0):
        if entries[0][2] > 1:
            first = not prompts
            prompts.append(e.layer_marks is not None)
            if e.layer_marks is not None:            # first-use stalls inflate the first forward's first units
                e.layer_marks.append((0, [Event(t) for t in ([0, 900, 1800] if first else [0, 2, 4])]))
    e._step, e._attach, e._detach = step, lambda slot, tokens: None, lambda slot: None
    e._warm()
    assert prompts[0] is False and prompts.count(True) == 1
    assert e.warm_unit_ms[1] == [2, 2]
