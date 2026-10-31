"""Routed prompt batches equal independent stream forwards, including kept draft taps.

Affine BF16/FP8 and actual native A4 resident/evicting-cache banks use their
shipping router, expert, GDN, attention and DFlash kernels. No tolerance.
"""
import json

import pytest
import torch
from nvfp4_a4_inputs import PREFIX, Context, linears, swiglu
from qwen_engine_checks import same_state
from safetensors.torch import save_file
from test_qwen36_moe import WIDTH, D, E, V, _model

from tensorfold.cleanup import rollback
from tensorfold.cuda import moe, prompt_precision
from tensorfold.cuda.nvfp4.cache_a4 import A4ExpertCache
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4
from tensorfold.cuda.nvfp4.host_layout import entry_layout_a4
from tensorfold.families.qwen3_5.cuda.forward import State
from tensorfold.families.qwen3_5.cuda.prefill import Piece, prefill_batch, prefill_chunk, prefill_rows, prefill_state
from tensorfold.families.qwen3_5_moe.cuda.dflash import DFlash
from tensorfold.families.qwen3_5_moe.cuda.nvfp4_load import NativeMoE, SharedExpert
from tensorfold.families.qwen3_5_moe.dflash_metadata import DraftConfig

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

# New row counts1/5/17/129 span the router's16-row tile and A4's128-row
# grouping boundary. Includes fresh/resumed, end/start/interior/no kept prefix.
CASES = [(1, 0, 1), (7, 2, 2), (20, 3, None), (132, 3, 68)]


@pytest.fixture(params=[("affine", False), ("affine", True), ("a4-resident", False), ("a4-cache", False)],
                ids=["affine-bf16", "affine-fp8", "a4-resident", "a4-cache1"])
def target(request):
    kind, fp8 = request.param
    if kind != "affine" and torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native activation FP4 requires SM12")
    w, _ = _model()
    cache = None
    try:
        w.bind_draft_taps((0, 1))
        if kind != "affine":
            # Keep the same real affine attention/GDN backbone; replace routed
            # and shared experts with checkpoint-native A4 implementations.
            w.precision = "checkpoint"
            w.__dict__.pop("fast_prefill", None)
            if kind == "a4-cache":
                entry = entry_layout_a4(WIDTH, D).entry_bytes
                cache = A4ExpertCache(entry, entry, "cuda", width=WIDTH, dims=D)
            for layer_id, layer in enumerate(w.layers):
                ctx = Context(count=E, width=WIDTH, dims=D)
                bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda", cache=cache,
                                           layer_id=layer_id, swiglu=swiglu)
                shared = SharedExpert(*linears(Context(count=1, width=WIDTH, dims=D), 0))
                original = layer.moe
                layer.moe = moe.Routed(original.router, bank, original.top_k, run=NativeMoE(bank, shared))
        with prompt_precision.using(fp8):
            yield w, cache
    except BaseException as primary:
        if cache is not None:
            rollback(cache, primary, cache.close)
        raise
    else:
        if cache is not None:
            cache.close()


def prompts():
    gen = torch.Generator().manual_seed(431)
    return [torch.randint(1, V, (n,), generator=gen).tolist() for n, _, _ in CASES]


def state_at(w, prompt, start):
    state = State(w)
    if start:
        prefill_state(w, prompt[:start], state)
    return state


def same_snapshot(actual, expected):
    assert actual[2:] == expected[2:]
    for left, right in zip(actual[:2], expected[:2], strict=True):
        for a, b in zip(left, right, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.equal(a, b)


@pytest.mark.parametrize("reverse", [False, True])
def test_routed_batch_matches_sequential_states_cuts_and_all_residual_taps(target, monkeypatch, reverse):
    w, cache = target
    pieces = list(zip(prompts(), CASES, strict=True))
    if reverse:
        pieces.reverse()
    references, states, items = [], [], []
    for prompt, (_, start, keep) in pieces:
        cut = keep - start if keep is not None and start < keep < len(prompt) else 0
        reference = state_at(w, prompt, start)
        ids = torch.tensor(prompt[start:], dtype=torch.int32, device="cuda")
        normed, taps, *part = prefill_chunk(w, ids, reference, capture_taps=True, cut=cut)
        references.append((normed, taps, part[0] if part else None, reference))
        state = state_at(w, prompt, start)
        states.append(state)
        items.append((prompt[start:], state, cut))
    calls, run = [], moe.run

    def measured(x, routed, *, prefill=False):
        calls.append((x.shape[0], x.dtype, prefill))
        return run(x, routed, prefill=prefill)

    # One routed call per layer over all rows proves this remains a batch;
    # its BF16 boundary must also hold when backbone prompt glue takesFP8.
    with monkeypatch.context() as patch:
        patch.setattr(moe, "run", measured)
        actual = prefill_rows(w, items, capture_taps=True)
    rows = sum(len(ids) for ids, _, _ in items)
    assert calls == [(rows, torch.bfloat16, True)] * len(w.layers)
    for (normed, taps, part), state, reference, (prompt, (_, _, keep)) in zip(
            actual, states, references, pieces, strict=True):
        ref_normed, ref_taps, ref_part, ref_state = reference
        same_state(state, ref_state)
        assert torch.equal(normed, ref_normed) and torch.equal(taps, ref_taps)
        assert torch.count_nonzero(taps).item() > 0
        assert (part is None) == (ref_part is None)
        if part is not None:
            same_state(part, ref_part)
            # A separately prefilled prefix never executes either cut branch.
            same_state(part, state_at(w, prompt, keep))
    if cache is not None:
        assert cache.evictions > 0 and cache.copied_bytes == cache.misses * cache.entry_bytes


def make_draft(directory, w):
    raw = {"architectures": ["DFlashDraftModel"], "model_type": "qwen3", "hidden_size": D,
           "intermediate_size": D, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
           "num_hidden_layers": 1, "num_target_layers": 2, "vocab_size": V, "block_size": 4,
           "rms_norm_eps": 1e-6, "max_position_embeddings": 256, "rope_theta": 10000,
           "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": V - 1}}
    cfg = DraftConfig.from_dict(raw)
    gen = torch.Generator().manual_seed(137)
    arrays = {name: ((torch.randn(shape, generator=gen) * .05).bfloat16() if len(shape) == 2
                     else torch.ones(shape, dtype=torch.bfloat16)) for name, shape in cfg.shapes().items()}
    (directory / "config.json").write_text(json.dumps(raw))
    save_file(arrays, directory / "model.safetensors")
    return DFlash(directory, w, context_limit=256)


def test_retained_dflash_context_and_prefix_cuts_equal_each_sequential_prefill(target, tmp_path):
    w, cache = target
    draft = make_draft(tmp_path, w)
    empty, references, pieces = draft.snapshot(), [], []
    for prompt, (_, start, keep) in zip(prompts(), CASES, strict=True):
        draft.restore(empty)
        ref_state = State(w)
        if start:
            prefill_state(w, prompt[:start], ref_state, draft=draft)
        normed = prefill_state(w, prompt, ref_state, draft=draft, keep_at=keep)
        normed, kept = (normed, None) if keep is None else normed
        references.append((normed, kept, draft.snapshot(), ref_state))
        draft.restore(empty)
        state = State(w)
        if start:
            prefill_state(w, prompt[:start], state, draft=draft)
        pieces.append(Piece(prompt, state, keep, draft.snapshot()))
    actual = prefill_batch(w, pieces, draft=draft)
    for piece, (normed, kept, snap), (ref_normed, ref_kept, ref_snap, ref_state) in zip(
            pieces, actual, references, strict=True):
        same_state(piece.st, ref_state)
        assert torch.equal(normed, ref_normed)
        same_snapshot(snap, ref_snap)
        assert (kept is None) == (ref_kept is None)
        if kept is not None:
            same_state(kept[0], ref_kept[0])
            same_snapshot(kept[1], ref_kept[1])
    if cache is not None:
        assert cache.evictions > 0
