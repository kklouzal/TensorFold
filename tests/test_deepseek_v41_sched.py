"""DeepSeek-V4.1 SM70 prompt layer groups on CPU: the half-layer pause points of ``Target.layers`` (a paused
forward runs the same operations on the same values as an unpaused one), the cost estimates the Decoder cuts
groups from (``schedule``) and its per-round cut."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


# ----------------------------------------------------------------------------------------------- schedule

def test_unit_costs_proportional_then_linear():
    from tensorfold.families.deepseek_v4.cuda.schedule import UnitCosts
    c = UnitCosts(256, [10.0, 20.0])
    assert c.predict(512) == [20.0, 40.0]               # one anchor: proportional (an overestimate)
    c.add(4096, [100.0, 40.0])
    got = c.predict(2048)                                # linear between (256, a) and (4096, b)
    assert got[0] == pytest.approx(10 + 90 * (2048 - 256) / 3840)
    assert got[1] == pytest.approx(20 + 20 * (2048 - 256) / 3840)
    assert c.predict(8192)[0] == pytest.approx(10 + 90 * (8192 - 256) / 3840)
    c.add(300, [1.0, 1.0])                               # not at most half the large anchor? 600 <= 4096: small
    assert c.small == (300, [1.0, 1.0])
    c.add(8192, [200.0, 80.0])
    assert c.large == (8192, [200.0, 80.0])
    c.add(5000, [1.0, 1.0])                              # >= 2 x small: the newest large chunk
    assert c.large[0] == 5000
    with pytest.raises(ValueError):
        c.add(512, [1.0])
    # a measured large chunk cheaper per unit than the warm-up never predicts below zero or a negative slope
    d = UnitCosts(256, [50.0])
    d.add(1024, [40.0])
    assert d.predict(8192) == [50.0]


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
    t.engram_layer = lambda ly, hidden, hashes, step, valid, shard: (log.append(("engram", ly.index)), hidden * 1.01 + 0.5)[1]
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
    t.engram_layer(ly, hidden, hashes, step, step.valid, shard)
    rows = torch.arange(shard.first, shard.first + shard.rows)
    assert seen["hidden"] == (shard.rows, 4, h)
    assert torch.equal(seen["projected"][:, 0], rows.half())
    assert torch.equal(seen["valid"], step.valid[rows])
