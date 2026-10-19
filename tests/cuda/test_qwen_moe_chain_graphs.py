"""Qwen CT MoE chain rounds replayed as CUDA graphs equal eager decoding token for token.

Resident native A4 banks (device routing plans) replace the tiny model's experts; the same DFlash drafter, copy
proposals, samplers and prefix states drive both runs. The runner's state grows across context buckets and serves
several requests.
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
    grown = []
    for prompt, sampling, rows in zip(PROMPTS, SAMPLINGS, (4, 3, 2, 4)):
        eager, eager_widths = _decode(w, draft, prompt, sampling, 40, None, rows)
        replayed, widths = _decode(w, draft, prompt, sampling, 40, runner, rows)
        assert replayed == eager
        assert widths == eager_widths
        grown.append(runner.rows)
    assert len(set(eager_widths)) > 1 or any(r > 1 for r in eager_widths)
    # A longer request grew the fixed state (graphs dropped and recaptured), a later short one reused it.
    assert grown[1] > grown[0] and grown[-1] == max(grown)
    assert len({key[0] for key in runner.graphs}) > 1
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
