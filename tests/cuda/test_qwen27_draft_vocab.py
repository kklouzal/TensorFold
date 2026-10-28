"""DFlash2's whole-vocabulary rounds: a pending token outside the fast head rows widens the round to every token.

Proposals only (the target verifies each token), so these tests check what the drafter offers: fast-row rounds keep
their candidates in the fast rows, wide rounds score the whole head (equal to a drafter whose fast rows are the whole
vocabulary), a batch widens when any live stream needs it, and an NVFP4 checkpoint's bf16 head is scored on one FP8
copy whose fast rows are views of it.
"""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from tensorfold.cuda.requant import Head  # noqa: E402
from tensorfold.families.qwen3_5.cuda import dflash2  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Config, Plain, QLinear, Weights  # noqa: E402

H, V, RANK, INTER = 2048, 512, 512, 512
FAST = ((0, 256), (500, 512))           # a vocabulary's fast rows for these tests: ids 256..499 are outside


def _qlinear(gen, n, k):
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda",
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
    return QLinear(words, scales, (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16())


def _files(tmp_path):
    gen = torch.Generator().manual_seed(5)

    def r(*shape, s=0.02):
        return (torch.randn(*shape, generator=gen) * s).bfloat16()

    t = {"candidate_selector.hidden_projection.weight": r(RANK, H),
         "candidate_selector.predecessor_codebook": r(V, RANK, s=0.1),
         "candidate_selector.successor_codebook": r(V, RANK, s=0.1),
         "fc.weight": r(H, 5 * H), "hidden_norm.weight": torch.ones(H).bfloat16(), "norm.weight": torch.ones(H).bfloat16()}
    for i in range(2):
        p = f"layers.{i}."
        t |= {p + "self_attn.q_proj.weight": r(1024, H), p + "self_attn.k_proj.weight": r(256, H),
              p + "self_attn.v_proj.weight": r(256, H), p + "self_attn.o_proj.weight": r(H, 1024),
              p + "self_attn.q_norm.weight": torch.ones(128).bfloat16(),
              p + "self_attn.k_norm.weight": torch.ones(128).bfloat16(),
              p + "input_layernorm.weight": torch.ones(H).bfloat16(),
              p + "post_attention_layernorm.weight": torch.ones(H).bfloat16(),
              p + "mlp.gate_proj.weight": r(INTER, H), p + "mlp.up_proj.weight": r(INTER, H),
              p + "mlp.down_proj.weight": r(H, INTER)}
        for conv in ("attention_conv", "mlp_conv"):
            t[p + conv + ".base_kernel"] = r(2, 2, H, s=0.5)
            t[p + conv + ".kernel_projection.weight"] = r(4 * H // 16, H)
    save_file(t, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": H, "head_dim": 128, "num_attention_heads": 8, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000}, "num_hidden_layers": 2, "sliding_window": 64, "is_causal": False,
        "dflash_config": {"mask_token_id": V - 1, "conv_group_size": 16}}))
    return tmp_path


def _target(head, quant="mlx"):
    g = torch.Generator(device="cuda").manual_seed(3)
    config = Config(hidden=H, intermediate=INTER, layers=0, heads=8, kv_heads=2, head_dim=128, vocab=V, k_heads=1,
                    v_heads=1, dk=128, dv=128, conv_kernel=4, interval=4, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    return Weights(config, _qlinear(g, V, H), [], torch.ones(H, device="cuda", dtype=torch.bfloat16), head,
                   torch.ones(16, device="cuda"), quant=quant)


def _drafter(monkeypatch, files, target, fast=FAST):
    monkeypatch.setattr(dflash2, "HOT_SPANS", fast)
    return dflash2.DFlash2(files, target)


def _context(d, n, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    d.restore(([None] * d.layers, [None] * d.layers, 0, 0))
    d.add_taps((torch.randn(n, 5 * H, generator=gen, device="cuda") * 0.5).bfloat16())
    return d.snapshot()


def _candidates(launched):
    """A launched block's (ids, logits) per depth, sorted by id (``topk`` leaves them unsorted)."""

    shared, start, rows, _ = launched
    ids, floats = shared[0][start:start + rows], shared[1][start:start + rows, :16]
    order = ids.argsort(dim=1)
    return ids.gather(1, order), floats.gather(1, order)


def _fast(ids):
    return torch.stack([(ids >= a) & (ids < b) for a, b in FAST]).any(0)


def test_fast_rounds_stay_in_the_fast_rows_and_wide_rounds_score_every_token(tmp_path, monkeypatch):
    files = _files(tmp_path)
    g = torch.Generator(device="cuda").manual_seed(11)
    head = _qlinear(g, V, H)
    d = _drafter(monkeypatch, files, _target(head))
    whole = _drafter(monkeypatch, files, _target(head), fast=((0, V),))
    snap = _context(d, 40, 9)
    for pending in (7, 505, 300, 499):
        d.restore(snap)
        ids, logits = _candidates(d.launch_block(pending, 11))
        whole.restore(snap)
        want_ids, want_logits = _candidates(whole.launch_block(pending, 11))
        if dflash2.hot(pending, FAST):
            assert bool(_fast(ids).all())                       # the fast rows only
        else:                                                   # every row of the head, as a whole-vocabulary drafter
            assert torch.equal(ids, want_ids) and torch.equal(logits, want_logits)
            assert not bool(_fast(want_ids).all())              # the head's best rows include ids outside the fast rows
    assert d.in_vocab(300) and d.in_vocab(7)


def test_a_batch_widens_when_one_live_stream_leaves_the_fast_rows(tmp_path, monkeypatch):
    files = _files(tmp_path)
    g = torch.Generator(device="cuda").manual_seed(12)
    head = _qlinear(g, V, H)
    d = _drafter(monkeypatch, files, _target(head))
    whole = _drafter(monkeypatch, files, _target(head), fast=((0, V),))
    snaps = [_context(d, n, s) for n, s in ((3, 1), (70, 2))]
    empty = ([None] * d.layers, [None] * d.layers, 0, 0)
    together = d.launch_blocks([snaps[0], empty, snaps[1]], [7, 300, 400], 11)
    assert together[1] is None
    for snap, pending, got in zip(snaps, (7, 400), (together[0], together[2])):
        whole.restore(snap)                                     # both live streams scored over the whole head
        want = _candidates(whole.launch_block(pending, 11))
        assert all(torch.equal(a, b) for a, b in zip(_candidates(got), want))
    fast = d.launch_blocks([snaps[0], empty], [7, 300], 11)    # an empty stream's pending neither drafts nor widens
    assert bool(_fast(_candidates(fast[0])[0]).all())


def test_an_nvfp4_bf16_head_is_scored_on_one_fp8_copy(tmp_path, monkeypatch):
    files = _files(tmp_path)
    gen = torch.Generator(device="cuda").manual_seed(13)
    weight = torch.randn(V, H, generator=gen, device="cuda") * 0.02
    weight[256:500] *= 8                                        # rows outside the fast rows dominate the whole head
    d = _drafter(monkeypatch, files, _target(Plain(weight.bfloat16()), quant="nvfp4"))
    assert isinstance(d.full, Head) and d.full.kind == "fp8"
    codes = d.full.linear
    for part, _, _ in d.sub_parts:                              # the fast rows: views of the copy, no second copy
        start = part.w8.data_ptr() - codes.w8.data_ptr()
        assert 0 <= start < codes.w8.numel()
    snap = _context(d, 30, 4)
    d.restore(snap)
    wide, _ = _candidates(d.launch_block(300, 11))
    assert bool(((wide >= 256) & (wide < 500)).all())           # the dominant rows, reachable only on wide rounds
    d.restore(snap)
    fast, _ = _candidates(d.launch_block(7, 11))
    assert bool(_fast(fast).all())
