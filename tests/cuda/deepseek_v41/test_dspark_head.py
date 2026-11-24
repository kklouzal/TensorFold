"""DeepSeek-V4.1 DSpark head on one V100: MXFP4 expert op and the full head forward vs the pure-torch reference.

Needs an SM70 GPU and the release head (/home/kklouzal/models/ds41-dspark-1017; override with DS41_DSPARK_DIR).
The head gate uses the official BF16 input embedding (shard 2) as embedding and as a stand-in vocabulary head
(the target's Q5_K head is the lead's callback); it gates the drafter's own arithmetic, not acceptance.
Metrics are printed (pytest -s) for the report.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(os.environ.get("DS41_DSPARK_DIR", "/home/kklouzal/models/ds41-dspark-1017"))
cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                          reason="SM70 CUDA required")
model = pytest.mark.skipif(not (ROOT / "official" / "config.json").is_file(), reason="DSpark release not present")


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def synthetic_bank(e, n, k, gen, device):
    codes = torch.randint(0, 256, (e, n, k // 2), generator=gen, dtype=torch.uint8).to(device)
    # the release's scale range, s - 127 in [-16, 0], weighted toward -5
    s = (127 - 5 + torch.randint(-11, 6, (e, n, k // 32), generator=gen)).clamp(111, 127).to(torch.uint8)
    return codes, s.to(device)


@cuda
def test_mxfp4_experts_synthetic():
    from tensorfold.cuda.sm70 import mxfp4_ds41 as mx
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(7)
    for (m, top, e, n, k, h) in ((5, 3, 8, 576, 5120, 5120), (1, 3, 6, 2304, 5120, 5120), (10, 3, 4, 576, 5120, 5120)):
        gate, gs = synthetic_bank(e, n, k, gen, dev)
        up, us = synthetic_bank(e, n, k, gen, dev)
        down, ds = synthetic_bank(e, h, n, gen, dev)
        x = (torch.randn(m, k, generator=gen) * 0.6).half().to(dev)
        ids = torch.randint(-1, e, (m, top), generator=gen, dtype=torch.int32).to(dev)
        probs = torch.rand(m, top, generator=gen).to(dev)
        exact = mx.reference(x, ids, probs, gate, gs, up, us, down, ds, dtype=torch.float64)
        for q8 in (False, True):
            got = mx.experts(x, ids, probs, gate, gs, up, us, down, ds, q8=q8)
            ref = mx.reference(x, ids, probs, gate, gs, up, us, down, ds, q8=q8)
            r_ref, r_exact = rel(got, ref), rel(got, exact)
            print(f"mxfp4 M{m} top{top} N{n} q8={q8}: vs same-contract ref {r_ref:.2e}, vs FP64 {r_exact:.2e}, "
                  f"bit-equal {float((got == ref).float().mean()):.4f}")
            assert torch.isfinite(got).all()
            assert r_ref < (2e-3 if not q8 else 4e-3)
            if not q8:
                assert r_exact < 2e-3
        none = torch.full_like(ids, -1)
        assert torch.equal(mx.experts(x, none, probs, gate, gs, up, us, down, ds), torch.zeros(m, h, dtype=torch.half,
                                                                                               device=dev))


@cuda
def test_stage0_route_partials_retain_fp32_until_independent_branch_cuts():
    from tensorfold.cuda.sm70 import mxfp4_ds41 as mx

    device = "cuda"
    m, top, hidden, k = 2, 3, 8, 32
    expert_count = 4
    width = 4 * k
    ids = torch.tensor([[0, 3, -1], [2, 1, 2]], dtype=torch.int64, device=device)
    weights = torch.tensor([[0.5, 0.375, float("nan")], [0.625, 0.125, 0.75]], device=device)
    entries = m * top * width
    inputs = ((torch.arange(entries, device=device).view(m, top, width) % 29 - 14) / 256).half()
    codes = torch.arange(expert_count * hidden * width // 2, device=device).to(torch.uint8)
    codes = codes.view(expert_count, hidden, width // 2)
    scales = torch.full((expert_count, hidden, width // 32), 125, dtype=torch.uint8, device=device)
    full_weights = mx.decode(codes, scales, torch.float64)
    contributions = []
    for rank in range(8):
        first = rank // 4 * 2
        shard = rank % 4
        local = ids - first
        local = torch.where((local >= 0) & (local < 2), local, -1).int()
        inter = inputs[..., shard * k:(shard + 1) * k].contiguous()
        inter[local < 0] = float("nan")
        bank = codes[first:first + 2, :, shard * (k // 2):(shard + 1) * (k // 2)].contiguous()
        bank_scales = scales[first:first + 2, :, shard:shard + 1].contiguous()
        parts = torch.empty((top + 1, m, hidden), dtype=torch.float32, device=device)
        parts.fill_(float("nan"))
        mx.extension().down_partials(parts[:top], inter, local, bank, bank_scales)
        expected = torch.zeros_like(parts[:top])
        for row, routes in enumerate(local.tolist()):
            for route, expert in enumerate(routes):
                if expert >= 0:
                    w = full_weights[first + expert, :, shard * k:(shard + 1) * k]
                    expected[route, row] = (w @ inter[row, route].double()).float()
        assert torch.equal(parts[:top].view(torch.int32), expected.view(torch.int32))
        assert torch.isnan(parts[-1]).all()  # The routed writer cannot publish the shared plane.
        parts[-1].fill_(0.125 + rank / 4096)
        contributions.append(parts)

    reduced = torch.stack(contributions).sum(0)
    reduced[2, 0].fill_(float("nan"))  # An invalid route cannot consume either operand.
    result = mx.finish_partials(reduced, ids, weights, experts=expert_count)
    routed = torch.zeros((m, hidden), device=device)
    for row, routes in enumerate(ids.tolist()):
        for route, expert in enumerate(routes):
            if expert >= 0:
                full = (full_weights[expert] @ inputs[row, route].double()).half().float()
                routed[row] += full * weights[row, route]
    shared = torch.tensor(sum(0.125 + rank / 4096 for rank in range(8)), device=device).half().float()
    expected = routed.half().float() + shared
    assert torch.equal(result.view(torch.int32), expected.view(torch.int32))

    # The halfway expert value must be cut before its weighted cancellation.
    # The shared value must independently round to 1 before the FP32 add.
    corner = torch.tensor([2049.0, -2048.0, float("nan"), 1.0004], device=device).view(4, 1, 1)
    result = mx.finish_partials(corner, ids[:1], torch.tensor([[0.5, 0.5, float("nan")]], device=device),
                               experts=expert_count)
    assert result.item() == 1


@cuda
def test_stage0_route_partials_reject_invalid_expert_on_graph_replay():
    program = """
import torch
from tensorfold.cuda.sm70 import mxfp4_ds41 as mx

ext = mx.prepare()
inter = torch.ones((1, 1, 32), dtype=torch.float16, device='cuda')
ids = torch.zeros((1, 1), dtype=torch.int32, device='cuda')
bank = torch.full((1, 8, 16), 0x22, dtype=torch.uint8, device='cuda')
scales = torch.full((1, 8, 1), 125, dtype=torch.uint8, device='cuda')
out = torch.empty((1, 1, 8), dtype=torch.float32, device='cuda')
ext.down_partials(out, inter, ids, bank, scales)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    ext.down_partials(out, inter, ids, bank, scales)
graph.replay()
assert torch.all(out == 8).item()
ids.fill_(1)
try:
    graph.replay()
    torch.cuda.synchronize()
except RuntimeError as error:
    assert any(mark in str(error).lower() for mark in ('illegal instruction', 'launch failure', 'device-side assert'))
    print('invalid Stage0 expert trapped', flush=True)
else:
    raise AssertionError('invalid Stage0 expert reached a bank read')
"""
    child = subprocess.run((sys.executable, "-B", "-c", program), capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "invalid Stage0 expert trapped" in child.stdout


@cuda
@pytest.mark.parametrize("invalid", [-2, 4])
def test_stage0_route_reduction_rejects_invalid_global_expert_on_graph_replay(invalid):
    program = """
import sys
import torch
from tensorfold.cuda.sm70 import mxfp4_ds41 as mx

mx.prepare()
ids = torch.tensor([[0, -1]], dtype=torch.int64, device='cuda')
probs = torch.tensor([[0.5, float('nan')]], device='cuda')
planes = torch.tensor([1.125, float('nan'), 0.25], device='cuda').view(3, 1, 1)
warm = mx.finish_partials(planes, ids, probs, experts=4)
assert warm.item() == 0.8125
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    actual = mx.finish_partials(planes, ids, probs, experts=4)
graph.replay()
assert actual.item() == 0.8125
ids[0, 0] = int(sys.argv[1])
try:
    graph.replay()
    torch.cuda.synchronize()
except RuntimeError as error:
    assert any(mark in str(error).lower() for mark in ('illegal instruction', 'launch failure', 'device-side assert'))
    print('invalid global Stage0 expert trapped', flush=True)
else:
    raise AssertionError('invalid global Stage0 expert reached successful reduction')
"""
    child = subprocess.run((sys.executable, "-B", "-c", program, str(invalid)), capture_output=True,
                           text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "invalid global Stage0 expert trapped" in child.stdout


@cuda
@model
def test_mxfp4_experts_release_bank():
    """Stage-0 experts of the release (full 2304 rows, world 1) on random routed rows."""
    from tensorfold.cuda.sm70 import mxfp4_ds41 as mx
    from tensorfold.families.deepseek_v4.cuda import dspark_weights as dw
    ck = dw.Checkpoint(ROOT / "official")
    dev = torch.device("cuda")
    picks = [3, 17, 64, 101, 127]
    bank = {}
    for w in ("w1", "w3", "w2"):
        bank[w] = (torch.stack([torch.from_numpy(ck.array(f"mtp.0.ffn.experts.{e}.{w}.weight").copy()) for e in picks]).to(dev),
                   torch.stack([torch.from_numpy(ck.array(f"mtp.0.ffn.experts.{e}.{w}.scale").copy()) for e in picks]).to(dev))
    gen = torch.Generator().manual_seed(3)
    x = (torch.randn(5, 5120, generator=gen) * 1.0).half().to(dev)
    ids = torch.tensor([[0, 1, 2], [3, 4, 0], [1, 3, -1], [2, 2, 4], [4, 0, 1]], dtype=torch.int32, device=dev)
    probs = torch.rand(5, 3, generator=gen).to(dev)
    args = (x, ids, probs, *bank["w1"], *bank["w3"], *bank["w2"])
    exact = mx.reference(*args, dtype=torch.float64)
    for q8 in (False, True):
        got = mx.experts(*args, q8=q8)
        ref = mx.reference(*args, q8=q8)
        print(f"release bank q8={q8}: vs same-contract ref {rel(got, ref):.2e}, vs FP64 {rel(got, exact):.2e}")
        assert rel(got, ref) < 4e-3


def _embedding(dev):
    from safetensors import safe_open
    with safe_open(str(ROOT / "embed" / "model-00002-of-00048.safetensors"), framework="pt") as f:
        return f.get_tensor("embed.weight").to(dev).half()


@cuda
@pytest.mark.parametrize("rank", [0, 7])
def test_stage_zero_tp8_padding_matches_full_attention_in_cuda_graph(rank, monkeypatch):
    from tensorfold.families.deepseek_v4.cuda import dspark_stage0 as native, dspark_reference as ref

    g = torch.Generator(device="cuda").manual_seed(171)
    q = (torch.randn(5, 64, 512, generator=g, device="cuda") * 0.125).half()
    kv = (torch.randn(5, 512, generator=g, device="cuda") * 0.125).half()
    window = (torch.randn(128, 512, generator=g, device="cuda") * 0.125).half()
    live = torch.arange(128, device="cuda") >= 71
    sink = torch.linspace(-0.875, 0.875, 64, device="cuda")
    group_weights = (torch.rand(8, 1024, 4096, generator=g, dtype=torch.float16, device="cuda") - 0.5).half()
    pos = torch.arange(57, 62, device="cuda")
    cos_sin = pos[:, None].float() * native.frequencies(q.device)
    cos, sin = cos_sin.cos(), cos_sin.sin()
    heads = slice(rank * 8, rank * 8 + 8)

    def reference():
        keys = torch.cat((torch.where(live[:, None], window, 0), kv)).float()
        valid = torch.cat((live, torch.ones(5, dtype=torch.bool, device="cuda")))
        scores = (q.float() @ keys.t() * 512 ** -0.5).masked_fill(~valid, float("-inf"))
        bias = sink[None, :, None]
        peak = torch.maximum(scores.amax(-1, keepdim=True), bias)
        numerator = (scores - peak).exp()
        probs = (numerator / (numerator.sum(-1, keepdim=True) + (bias - peak).exp())).half()
        attended = (probs.float() @ keys).half()
        rotated = ref.rope(attended, pos, inverse=True)
        values = ref.qdq_fp8(rotated.reshape(5, 8, 4096))
        return torch.einsum("ngk,gok->ngo", values.float(), group_weights.float()).half()[:, rank]

    def candidate():
        attended = native._attention_rows(q[:, heads].contiguous(), kv, window, live, sink[heads], rank=rank)
        values = native._qdq(native._rope(attended, cos, -sin).reshape(5, 1, 4096))
        return native._project_attention([values], group_weights[rank:rank + 1], rank)[0]

    pads = []
    original = native._pad_axis

    def observe(x, dim, size, offset):
        pads.append((tuple(x.shape), dim, size, offset))
        return original(x, dim, size, offset)

    with monkeypatch.context() as scoped:
        scoped.setattr(native, "_pad_axis", observe)
        actual = candidate()
    assert pads == [((5, 8, 512), 1, 64, rank * 8), ((5, 8, 133), 1, 16, rank * 8 % 16),
                    ((1, 1024, 4096), 0, 2, rank % 2), ((5, 1, 4096), 1, 2, rank % 2)]
    assert torch.equal(actual.view(torch.int16), reference().view(torch.int16))

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = candidate()
    for scale in (0.6875, -1.25):
        q.mul_(scale)
        window.mul_(scale)
        graph.replay()
        assert torch.equal(replay.view(torch.int16), reference().view(torch.int16))


@cuda
@model
@pytest.mark.parametrize("variant", ["official", "ft"])
def test_head_forward_vs_reference(variant):
    from tensorfold.families.deepseek_v4.cuda import dspark_head as dh
    from tensorfold.families.deepseek_v4.cuda import dspark_reference as ref
    from tensorfold.families.deepseek_v4.cuda import dspark_weights as dw
    dev = torch.device("cuda")
    w = dw.load(ROOT / variant, device=dev, config_dir=ROOT / "official")
    print(f"{variant}: head bytes {w.nbytes() / 2**30:.2f} GiB; audit problems "
          f"{[(a.name, a.flushed, a.inexact) for a in w.audit.problems()]}")
    table = _embedding(dev)
    embed = lambda ids: table[ids]                                       # noqa: E731
    head = lambda x: x.float() @ table.float().t()                       # noqa: E731  stand-in vocabulary head
    gen = torch.Generator().manual_seed(11)
    head_state = dh.DSparkHead(w, slots=4, device=dev)
    lengths = {0: 300, 2: 57}                                            # slot 2: window partly before position 0
    for slot, length in lengths.items():
        taps = (torch.randn(length, 3 * dw.HIDDEN, generator=gen) * 0.05).half().to(dev)
        pos = torch.arange(length, device=dev)
        # insert contract: <= RING distinct positions per slot and call; two calls exercise the ring's wrap
        # (positions 256..299 overwrite 0..43)
        cut = max(0, length - dh.WINDOW)
        for a, b in ((0, cut), (cut, length)):
            head_state.insert_context(torch.full((b - a,), slot, device=dev), taps[a:b], pos[a:b])
        if slot == 0:
            rows = ref.context_rows(w, taps[-dh.WINDOW:], pos[-dh.WINDOW:])
            for i in range(dw.STAGES):
                got = head_state.ring[slot, i, pos[-dh.WINDOW:] % dh.RING]
                print(f"context stage {i}: rel {rel(got, rows[i]):.2e}")
                assert rel(got, rows[i]) < 5e-3
    slots = torch.tensor([0, 2], device=dev)
    anchor_pos = torch.tensor([lengths[0], lengths[2]], device=dev)
    anchors = torch.tensor([4021, 1000], device=dev)
    for q8 in (False, True):
        head_state.q8 = q8
        prop = head_state.propose(anchors, anchor_pos, slots, embed, head)
        for r, slot in enumerate((0, 2)):
            at = int(anchor_pos[r]) - dh.WINDOW + torch.arange(dh.WINDOW, device=dev)
            windows = [head_state.ring[slot, i, at % dh.RING] for i in range(dw.STAGES)]
            ids = torch.full((dw.BLOCK,), dw.NOISE, device=dev)
            ids[0] = anchors[r]
            h_ref = ref.forward_one(w, embed(ids), int(anchor_pos[r]), windows, at >= 0, q8=q8)
            logits_ref = head(ref.rms(h_ref, w.norm).half())
            tokens_ref = ref.markov_greedy(w, logits_ref, int(anchors[r]))
            logits = head(head_state.head_input(prop.hidden[r]))
            conf_ref = torch.cat((h_ref, w.markov_embed[torch.tensor([int(anchors[r])] + tokens_ref[:-1], device=dev)]
                                  .float()), -1) @ w.confidence.t()
            got_tokens = prop.tokens[r].tolist()
            print(f"{variant} q8={q8} slot {slot}: hidden rel {rel(prop.hidden[r], h_ref):.2e}, logits rel "
                  f"{rel(logits, logits_ref):.2e}, drafts {got_tokens} ref {tokens_ref}, confidence "
                  f"{prop.confidence[r].tolist()} ref {conf_ref.squeeze(-1).tolist()}")
            assert torch.isfinite(prop.hidden[r]).all()
            assert rel(prop.hidden[r], h_ref) < 2e-2
            if got_tokens != tokens_ref:
                # allowed only at a near-tie of the reference's corrected logits at the first difference
                j = next(i for i, (a, b) in enumerate(zip(got_tokens, tokens_ref)) if a != b)
                prev = int(anchors[r]) if j == 0 else tokens_ref[j - 1]
                value = logits_ref[j] + w.markov_head.float() @ w.markov_embed[prev].float()
                top2 = value.topk(2).values
                assert float(top2[0] - top2[1]) < 0.05 * float(top2[0].abs()), "draft differs away from a tie"
        # the serving path: batched sharded Markov greedy (world 1) equals the per-request MarkovGreedy drafts
        drafts = head_state.draft(anchors, anchor_pos, slots, embed, head)
        print(f"{variant} q8={q8}: draft() {drafts.tolist()} propose() {prop.tokens.tolist()}")
        assert torch.equal(drafts, prop.tokens)
        # the TP8 merge: eight contiguous vocabulary shards' packets merged = the full-vocabulary argmax
        from tensorfold.families.deepseek_v4.cuda.dspark_markov import ShardedMarkov
        logits = head(head_state.head_input(prop.hidden.reshape(-1, dw.HIDDEN))).view(2, dw.BLOCK, -1)[:, 0]
        part = dw.VOCAB // 8
        packets = [ShardedMarkov(w.markov_head[k * part:(k + 1) * part], w.markov_embed, k * part, None, 1)
                   .local_packet(logits[:, k * part:(k + 1) * part], anchors) for k in range(8)]
        merged = ShardedMarkov.merge(torch.cat(packets, -1), 8)
        full = (logits + w.markov_embed[anchors].float() @ w.markov_head.float().t()).argmax(-1)
        assert torch.equal(merged, full), (merged.tolist(), full.tolist())
    accepted = dh.greedy_accept(prop.tokens, torch.cat((prop.tokens, prop.tokens[:, :1]), 1))
    assert accepted.tolist() == [5, 5]
