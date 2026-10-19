"""Qwen CT MoE chain rounds replayed as CUDA graphs equal eager decoding token for token.

Resident native A4 banks (device routing plans) replace the tiny model's experts; the same DFlash drafter, copy
proposals, samplers and prefix states drive both runs. The runner's fixed state (reserved at its capacity) serves
several requests across context buckets; through the engine it is also the stream's attention cache: prefills
extend it in place and kept prompt states share it, moving to private buffers when another conversation needs it.
"""
import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from nvfp4_a4_inputs import PREFIX, Context, linears, swiglu  # noqa: E402
from test_qwen36_moe import WIDTH, D, E, _model  # noqa: E402
from test_qwen_moe_prefill_batch import make_draft  # noqa: E402

from tensorfold.cuda import moe  # noqa: E402
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda import decode  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.chain_graphs import ChainGraphs  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.nvfp4_load import NativeMoE, SharedExpert  # noqa: E402


def _context():
    """E routed experts; expert E doubles as the layer's shared_expert (same geometry, its own globals)."""
    return Context(count=E + 1, width=WIDTH, dims=D).alias_shared_expert(E)


def _native_model(shared_cell):
    """The tiny model on resident native A4 banks; the shared expert is the bank's last cell or separate projections."""
    from types import SimpleNamespace

    w, _ = _model()
    w.bind_draft_taps((0, 1))
    w.precision = "checkpoint"
    w.__dict__.pop("fast_prefill", None)
    full = SimpleNamespace(experts=E, moe_width=WIDTH, hidden=D, top_k=w.layers[0].moe.top_k)
    for layer in w.layers:
        bank = load_host_experts_a4(_context(), PREFIX, full, "cuda", swiglu=swiglu, shared=shared_cell)
        assert bank.capturable and bank.count == (E + 1 if shared_cell else E)
        # Exact power-of-two weight globals keep the random two-layer model's residual stream finite.
        for field in bank.fields[2:4]:
            field.mul_(1 / 64)
        shared = None
        if not shared_cell:
            source = _context()
            for projection in ("gate_proj", "up_proj", "down_proj"):           # the same 1/64 weight scale
                source.payload[f"{PREFIX}.experts.{E}.{projection}.weight_global_scale"].mul_(64)
            shared = SharedExpert(*linears(source, E))
        layer.moe = moe.Routed(layer.moe.router, bank, layer.moe.top_k, run=NativeMoE(bank, shared))
    return w


@pytest.fixture(scope="module")
def model():
    return _native_model(True)


LONG = [3 + (i * 7) % 50 for i in range(180)]           # repeats itself: copy proposals of several widths
PROMPTS = [[5, 6, 7, 8, 9, 10], LONG, list(range(20, 90)), [3, 4, 5]]
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95), None, Sampling(99, 0.8, 0, 1.0)]


def _decode(w, draft, prompt, sampling, count, runner, rows):
    empty = ([None] * draft.layers, [None] * draft.layers, 0, 0)
    draft.restore(empty)
    st, pending = decode.prefill(w, prompt, sampling, draft)[:2]
    result = decode.draft_decode(w, st, prompt, pending, count, sampling, draft, max_rows=rows, allow_copy=True,
                                 stop_eos=False, inplace=True, runner=runner)
    return result.tokens, result.widths


def test_chain_graph_rounds_equal_eager_rounds_across_requests_and_buckets(model, tmp_path):
    w = model
    draft = make_draft(tmp_path, w)
    runner = ChainGraphs(w, 512, bucket=64)
    buffers = runner._addresses()
    for prompt, sampling, rows in zip(PROMPTS, SAMPLINGS, (4, 3, 2, 4)):
        eager, eager_widths = _decode(w, draft, prompt, sampling, 40, None, rows)
        replayed, widths = _decode(w, draft, prompt, sampling, 40, runner, rows)
        assert replayed == eager
        assert widths == eager_widths
    assert len(set(eager_widths)) > 1 or any(r > 1 for r in eager_widths)
    # Private prefill states were copied in; the fixed buffers never moved; spans and widths each took graphs.
    assert runner._addresses() == buffers
    assert len({key[0] for key in runner.graphs}) > 1 and len({key[1] for key in runner.graphs}) > 1
    # Serial rounds (no drafter, no copies) replay the one-row graph.
    st, pending = decode.prefill(w, PROMPTS[2], None)[:2]
    eager = decode.draft_decode(w, decode.clone_state(st), PROMPTS[2], pending, 12, None, None, allow_copy=False,
                                stop_eos=False).tokens
    replayed = decode.draft_decode(w, st, PROMPTS[2], pending, 12, None, None, allow_copy=False, stop_eos=False,
                                   inplace=True, runner=runner).tokens
    assert replayed == eager and any(key[0] == 1 and not key[2] for key in runner.graphs)


def test_non_chain_rounds_run_eagerly_on_the_fixed_state(model):
    w = model
    runner = ChainGraphs(w, 256, bucket=64)
    st, _ = decode.prefill(w, PROMPTS[0], None)[:2]
    fixed = runner.load(st, st.pos + 8)
    assert fixed is runner.st and fixed.pos == st.pos
    tree = runner.verify([5, 6, 7], [-1, 0, 0], capture_taps=False)
    eager_tree = decode.tree_forward(w, torch.tensor([5, 6, 7], dtype=torch.int32, device="cuda"), [-1, 0, 0], fixed)
    assert torch.equal(tree[0], eager_tree[0]) and not runner.graphs
    chain = runner.verify([5, 6], [-1, 0], capture_taps=False)
    eager = decode.tree_forward(w, torch.tensor([5, 6], dtype=torch.int32, device="cuda"), [-1, 0], fixed)
    assert torch.equal(chain[0], eager[0]) and len(runner.graphs) == 1


def test_prepare_captures_every_width_and_later_rounds_replay_them(model, tmp_path):
    w = model
    draft = make_draft(tmp_path, w)
    runner = ChainGraphs(w, 512, bucket=64)
    assert runner.prepare(range(1, 5), taps=True) == 5          # widths 1..4 with taps, plus the one-row serial form
    held = dict(runner.graphs)
    eager, _ = _decode(w, draft, PROMPTS[0], None, 24, None, 4)
    replayed, _ = _decode(w, draft, PROMPTS[0], None, 24, runner, 4)
    assert replayed == eager
    assert all(runner.graphs[key] is entry for key, entry in held.items())      # no recapture of prepared keys


def test_separate_shared_projections_replay_like_eager(tmp_path):
    # A shared expert outside the routed A4 form (another width or representation) keeps its own projections beside
    # the resident bank; chain rounds over that operator replay exactly as eager rounds.
    w = _native_model(False)
    draft = make_draft(tmp_path, w)
    runner = ChainGraphs(w, 512, bucket=64)
    eager, eager_widths = _decode(w, draft, LONG, SAMPLINGS[1], 24, None, 3)
    replayed, widths = _decode(w, draft, LONG, SAMPLINGS[1], 24, runner, 3)
    assert replayed == eager and widths == eager_widths
    assert runner.graphs and max(eager_widths) > 1                  # drafted chains wider than one row


def _engine(w, draft, runner, budget):
    from tensorfold.cuda.streams import KVRoom, PrefixCache
    from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE, Qwen27Engine

    engine = object.__new__(Qwen27Engine)
    engine.torch = torch
    engine.tp, engine.rank, engine.max_rows, engine.allow_copy = 1, 0, 4, True
    engine.w, engine.draft, engine.eos, engine.cache = w, draft, tuple(w.config.eos), PrefixCache(KEEP_ONE)
    engine.points, engine.vision = None, None
    engine.context_window, engine.concurrent, engine.multi, engine.scheduler = 400, False, None, None
    engine.room, engine.runner = KVRoom(engine.cache, budget), runner
    return engine


def _generate(engine, prompt, sampling, count=24):
    out = []
    stats = engine.generate(list(prompt), count, sampling, lambda tokens: out.extend(tokens) and False)
    return out, stats


@pytest.mark.parametrize("budget", [1 << 34, 0])
def test_engine_prefills_on_the_fixed_buffers_and_conversations_switch_like_eager(model, tmp_path, budget):
    """Through the engine: every prefill writes the runner's fixed buffers (no second attention cache); a new
    conversation moves the kept prompt state to private buffers sized to its rows (or drops it when the stream's
    room admits no copy), and resuming it copies it back. Tokens equal the eager engine's every time."""

    w = model
    runner = ChainGraphs(w, 512, bucket=64)
    fixed = runner.st.kv
    for name in ("eager", "graphed"):
        (tmp_path / name).mkdir()
    eager = _engine(w, make_draft(tmp_path / "eager", w), None, 1 << 34)
    graphed = _engine(w, make_draft(tmp_path / "graphed", w), runner, budget)
    first = LONG[:120]
    reply, _ = _generate(eager, first, None)
    assert _generate(graphed, first, None)[0] == reply
    [(ids, kept, _)] = graphed.cache.entries
    assert ids == first[:-1] and kept.kv is fixed                   # the prompt state shares the fixed buffers
    other = list(range(20, 90))
    assert _generate(graphed, other, SAMPLINGS[1])[0] == _generate(eager, other, SAMPLINGS[1])[0]
    moved = [entry for entry in graphed.cache.entries if entry[0] == first[:-1]]
    if budget:
        [(_, kept, _)] = moved                                       # moved out, sized to its rows
        assert kept.kv is not fixed and all(kv is None or kv[0].shape[0] == len(first) - 1 for kv in kept.kv)
    else:
        assert not moved                                             # no room for a copy: dropped
    turn = first + reply + [7, 8, 9]
    again, stats = _generate(graphed, turn, None)
    assert again == _generate(eager, turn, None)[0]
    assert stats["cached"] == (len(first) - 1 if budget else 0)
    assert all(kept.kv is fixed for ids, kept, _ in graphed.cache.entries if turn[:len(ids)] == ids)
    # The same conversation's next turn resumes on the fixed buffers in place (no copy either way).
    last = turn + again + [4, 5, 6]
    fourth, stats = _generate(graphed, last, SAMPLINGS[1])
    assert fourth == _generate(eager, last, SAMPLINGS[1])[0] and stats["cached"] == len(turn) - 1
    assert all(kept.kv is fixed for ids, kept, _ in graphed.cache.entries if last[:len(ids)] == ids)
    assert runner._addresses() == tuple(t.data_ptr() for kv in fixed if kv is not None for t in kv)


def test_begin_refuses_kept_states_on_the_fixed_buffers_that_are_not_one_prompts_prefixes(model):
    """V4: a kept state on the fixed buffers that is not a prefix of the prompt (or of the chain being moved out)
    would read another prompt's rows; begin raises instead of resuming or moving it."""

    from tensorfold.cuda.streams import PrefixCache
    from tensorfold.families.qwen3_5.cuda.forward import State

    w = model
    runner = ChainGraphs(w, 256, bucket=64)

    def kept(ids):
        st = State(w)
        st.kv, st.pos = runner.st.kv, len(ids)
        return st

    cache = PrefixCache(4)
    base = kept([1, 2, 3, 4])
    cache.add([1, 2, 3, 4], base, None)
    cache.add([1, 2], kept([1, 2]), None)
    assert runner.begin(cache, None, base, [1, 2, 3, 4, 5]) is base          # one prompt's prefixes: resumed
    cache.add([1, 9, 9], kept([1, 9, 9]), None)                              # another prompt's rows on the same buffers
    with pytest.raises(RuntimeError, match="not one prompt's prefixes"):
        runner.begin(cache, None, base, [1, 2, 3, 4, 5])
    with pytest.raises(RuntimeError, match="not one prompt's prefixes"):
        runner.begin(cache, None, None, [7, 7, 7])                           # moving them out checks the chain too
    # Resuming a moved-out chain checks the states sharing its buffers before they join the fixed ones.
    private = [None if kv is None else (kv[0][:3].clone(), kv[1][:3].clone()) for kv in runner.st.kv]
    resumed, stray = State(w), State(w)
    resumed.kv, resumed.pos, stray.kv, stray.pos = private, 3, private, 2
    moved = PrefixCache(4)
    moved.add([5, 6, 7], resumed, None)
    moved.add([5, 9], stray, None)                                           # not a prefix of the resumed prompt
    with pytest.raises(RuntimeError, match="not one prompt's prefixes"):
        runner.begin(moved, None, resumed, [5, 6, 7, 8])
    assert resumed.kv is private and stray.kv is private                    # raised before any buffer changed


def _same_rows(kvs, sources, rows):
    return all(kv is None or torch.equal(kv[0][:rows], src[0][:rows]) and torch.equal(kv[1][:rows], src[1][:rows])
               for kv, src in zip(kvs, sources))


def test_moving_a_conversation_out_keeps_the_resumed_one_and_evicts_older_ones_first(model):
    """M6: resuming a moved-out conversation while another one's kept states hold the fixed buffers. Their copy must
    fit the stream's room beside the resumed state's private buffers (live until copied in): older conversations'
    kept states go first, the resumed one's never; when the copy still does not fit, the fixed buffers' kept
    states are dropped instead. Copies keep every row."""

    from tensorfold.cuda.streams import KVRoom, PrefixCache
    from tensorfold.families.qwen3_5.cuda.forward import State

    w = model
    older, resumed, current = list(range(100, 140)), list(range(200, 260)), list(range(300, 350))
    for spare, moves in ((len(current), True), (len(current) - 1, False)):
        runner = ChainGraphs(w, 256, bucket=64)
        fixed = runner.st.kv
        row = sum(t.stride(0) * t.element_size() for kv in fixed if kv is not None for t in kv)

        def kept(ids, kv):
            st = State(w)
            st.kv, st.pos = kv, len(ids)
            return st

        def private(rows, buffers):
            return [None if kv is None else (torch.randn_like(kv[0][:rows]), torch.randn_like(kv[1][:rows]))
                    for kv in buffers]

        for kv in fixed:
            if kv is not None:
                kv[0].normal_()
                kv[1].normal_()
        before = [None if kv is None else (kv[0].clone(), kv[1].clone()) for kv in fixed]
        cache = PrefixCache(4)
        cache.add(older, kept(older, private(len(older), fixed)), None)
        own = private(len(resumed), fixed)
        cache.add(resumed, kept(resumed, own), None)
        cache.add(current[:20], kept(current[:20], fixed), None)
        cache.add(current, kept(current, fixed), None)
        hit = cache.longest(resumed + [1])                                   # the engine's resume: newest, and hit
        room = KVRoom(cache, 0)
        room.budget = room.held(runner.st) - len(older) * row + spare * row  # the copy fits only without ``older``
        st = runner.begin(cache, room, hit[1], resumed + [1])
        assert st is hit[1] and st.kv is fixed and _same_rows(fixed, own, len(resumed))
        assert cache.entries[-1][0] == resumed and older not in [ids for ids, _, _ in cache.entries]
        out = [kept.kv for ids, kept, _ in cache.entries if ids in (current, current[:20])]
        if moves:
            assert len(out) == 2 and out[0] is out[1] is not fixed and _same_rows(out[0], before, len(current))
            assert room.held(runner.st) + len(resumed) * row <= room.budget  # made beside the resumed buffers
        else:
            assert not out                                                   # dropped: no room beside the resumed one
