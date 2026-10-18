"""Qwen CT MoE chain rounds replayed as CUDA graphs equal eager decoding token for token.

Resident native A4 banks (device routing plans) replace the tiny model's experts; the same DFlash drafter, copy
proposals, samplers and prefix states drive both runs. The runner's state grows across context buckets and serves
several requests.
"""
import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from nvfp4_a4_inputs import PREFIX, Context, swiglu  # noqa: E402
from test_qwen36_moe import WIDTH, D, E, V, _model  # noqa: E402
from test_qwen_moe_prefill_batch import make_draft  # noqa: E402

from tensorfold.cuda import moe  # noqa: E402
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda import decode  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.chain_graphs import ChainGraphs  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.nvfp4_load import NativeMoE  # noqa: E402


def _context():
    """E routed experts; expert E doubles as the layer's shared_expert (same geometry, its own globals)."""
    ctx = Context(count=E + 1, width=WIDTH, dims=D)
    for name in list(ctx.payload):
        if f".experts.{E}." in name:
            alias = name.replace(f".experts.{E}.", ".shared_expert.")
            ctx.payload[alias], ctx.info[alias] = ctx.payload[name], ctx.info[name]
    return ctx


@pytest.fixture(scope="module")
def model():
    from types import SimpleNamespace

    w, _ = _model()
    w.bind_draft_taps((0, 1))
    w.precision = "checkpoint"
    w.__dict__.pop("fast_prefill", None)
    full = SimpleNamespace(experts=E, moe_width=WIDTH, hidden=D, top_k=w.layers[0].moe.top_k)
    for layer in w.layers:
        bank = load_host_experts_a4(_context(), PREFIX, full, "cuda", swiglu=swiglu, shared=True)
        assert bank.capturable and bank.count == E + 1
        # Exact power-of-two weight globals keep the random two-layer model's residual stream finite.
        for field in bank.fields[2:4]:
            field.mul_(1 / 64)
        layer.moe = moe.Routed(layer.moe.router, bank, layer.moe.top_k, run=NativeMoE(bank, None))
    return w


LONG = [3 + (i * 7) % 50 for i in range(180)]           # repeats itself: copy proposals of several widths
PROMPTS = [[5, 6, 7, 8, 9, 10], LONG, list(range(20, 90)), [3, 4, 5]]
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95), None, Sampling(99, 0.8, 0, 1.0)]


def _decode(w, draft, prompt, sampling, count, runner, rows):
    empty = ([None] * draft.layers, [None] * draft.layers, 0, 0)
    draft.restore(empty)
    st, pending = decode.prefill(w, prompt, sampling, draft)[:2]
    result = decode.draft_decode(w, st, prompt, pending, count, sampling, draft, max_rows=rows, allow_copy=True,
                                 stop_eos=False, inplace=True, runner=runner)
    widths = result.widths
    return result.tokens, widths


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
