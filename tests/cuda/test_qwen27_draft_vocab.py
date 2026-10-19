"""Native DFlash2 proposal policies: the default remains truncated, full mode covers every target token, and
private FP8 copies own only selected proposal rows. The target head is always unchanged. Full-head candidates are
compared with an independently configured original span path covering every row; target serial identity is covered
by test_qwen27_nvfp4.py's real-constructor matrix for greedy requests; sampled chains repeat under their fixed
configuration across warm/cold cache and supported batching. These tests require actual CUDA execution.
"""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from safetensors.torch import save_file

from tensorfold.cuda.requant import Head, resident_bytes
from tensorfold.families.qwen3_5.cuda import dflash2, draft_tree
from tensorfold.families.qwen3_5.cuda.weights import Config, Plain, QLinear, Weights

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


def _drafter(monkeypatch, files, target, fast=FAST, **options):
    monkeypatch.setattr(draft_tree, "HOT_SPANS", fast)
    return dflash2.DFlash2(files, target, **options)


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


@pytest.mark.parametrize("bits", [16, 8])
def test_full_policy_matches_the_original_span_path_over_every_row_and_default_never_widens(tmp_path, monkeypatch, bits):
    files = _files(tmp_path)
    g = torch.Generator(device="cuda").manual_seed(11)
    head = Plain((torch.randn(V, H, generator=g, device="cuda") * 0.02).bfloat16())
    target = _target(head, quant="nvfp4")
    original = _drafter(monkeypatch, files, target, head_bits=bits)
    full = _drafter(monkeypatch, files, target, draft_vocab="full", head_bits=bits)
    whole_span = _drafter(monkeypatch, files, target, fast=((0, V),), head_bits=bits)
    snap = _context(original, 40, 9)
    for pending in (7, 505, 300, 499):
        original.restore(snap)
        ids, _ = _candidates(original.launch_block(pending, 11))
        assert bool(_fast(ids).all())                    # the default does not change policy with pending language
        full.restore(snap)
        got = _candidates(full.launch_block(pending, 11))
        whole_span.restore(snap)
        want = _candidates(whole_span.launch_block(pending, 11))
        assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert not original.in_vocab(300) and full.in_vocab(300)
    assert target.head is head


def test_full_policy_partitions_candidates_by_live_stream_without_pending_based_dispatch(tmp_path, monkeypatch):
    files = _files(tmp_path)
    g = torch.Generator(device="cuda").manual_seed(12)
    head = Plain((torch.randn(V, H, generator=g, device="cuda") * 0.02).bfloat16())
    d = _drafter(monkeypatch, files, _target(head, quant="nvfp4"), draft_vocab="full")
    snaps = [_context(d, n, seed) for n, seed in ((3, 1), (70, 2))]
    empty = ([None] * d.layers, [None] * d.layers, 0, 0)
    together = d.launch_blocks([snaps[0], empty, snaps[1]], [7, 300, 400], 11)
    assert together[1] is None
    for snap, pending, got in zip(snaps, (7, 400), (together[0], together[2])):
        d.restore(snap)
        want = _candidates(d.launch_block(pending, 11))
        assert all(torch.equal(a, b) for a, b in zip(_candidates(got), want))


@pytest.mark.parametrize("vocab", ["truncated", "full"])
def test_private_fp8_storage_covers_only_requested_ranges_and_preserves_target(tmp_path, monkeypatch, vocab):
    files = _files(tmp_path)
    weight = torch.zeros(V, H, device="cuda", dtype=torch.bfloat16)
    weight[:16, 0] = torch.arange(1, 17, device="cuda").bfloat16() / 16
    weight[300:316, 0] = torch.arange(16, 32, device="cuda").bfloat16()
    saved = weight.clone()
    head = Plain(weight)
    d = _drafter(monkeypatch, files, _target(head, quant="nvfp4"), draft_vocab=vocab, head_bits=8)
    parts = [d.full_head] if vocab == "full" else [part for part, _, _ in d.sub_parts]
    spans = ((0, V),) if vocab == "full" else FAST
    assert all(isinstance(part, Head) for part in parts)
    assert sum(part.nbytes() for part in parts) == sum(resident_bytes("fp8", b - a, H) for a, b in spans)
    assert all(part.linear.w8.data_ptr() != head.weight.data_ptr() for part in parts)
    hidden = torch.zeros(3, H, device="cuda", dtype=torch.bfloat16)
    hidden[:, 0] = 1
    _, ids = d._candidates(hidden)
    expected = set(range(300, 316)) if vocab == "full" else set(range(16))
    assert all(set(row.tolist()) == expected for row in ids)
    assert torch.equal(head.weight, saved) and head.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("bits", [16, 8])
def test_full_sampled_q_matches_original_whole_span_and_repeats_across_batches(tmp_path, monkeypatch, bits):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.penalties import History, Penalties

    files = _files(tmp_path)
    gen = torch.Generator(device="cuda").manual_seed(31)
    head = Plain((torch.randn(V, H, generator=gen, device="cuda") * .02).bfloat16())
    saved = head.weight.clone()
    target = _target(head, quant="nvfp4")
    full = _drafter(monkeypatch, files, target, draft_vocab="full", head_bits=bits)
    reference = _drafter(monkeypatch, files, target, fast=((0, V),), head_bits=bits)
    snaps = [_context(full, count, seed) for count, seed in ((7, 4), (70, 5))]
    empty = ([None] * full.layers, [None] * full.layers, 0, 0)
    histories = [History(Penalties(presence=.2, repetition=1.1), [7]), History(Penalties(frequency=.1), [300])]
    policies = [Sampling(6, .6, 0, .95), Sampling(1234, 1.0, 20, .95)]
    grouped = full.launch_chains([snaps[0], empty, snaps[1]], [7, 300, 400], 11,
                                 histories=[histories[0], None, histories[1]])
    assert grouped[1] is None

    def identity(draft, launched, i):
        packet = draft.finish_chain(launched, snaps[i][3], 11, policies[i], histories[i])
        return packet.tokens, tuple((q.ids.tobytes(), q.probs.tobytes()) for q in packet.q_rows)

    grouped_ids = [identity(full, grouped[0], 0), identity(full, grouped[2], 1)]
    assert grouped[0][0][3]["logits"] is None and grouped[0][0][3]["ids"] is None
    permuted = full.launch_chains(list(reversed(snaps)), [400, 7], 11, histories=list(reversed(histories)))
    for i, pending in enumerate((7, 400)):
        solo = full.launch_chains([snaps[i]], [pending], 11, histories=[histories[i]])[0]
        want = reference.launch_chains([snaps[i]], [pending], 11, histories=[histories[i]])[0]
        assert grouped_ids[i] == identity(full, solo, i) == identity(full, permuted[1 - i], i)
        assert grouped_ids[i] == identity(reference, want, i)
    assert target.head is head and torch.equal(head.weight, saved)
