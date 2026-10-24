"""Independent tiny CPU math/state oracles for the unregistered DSpark adapter."""

from types import SimpleNamespace
import math

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import dspark_reference as ref
from tensorfold.families.qwen4_exp.cuda.dspark_provider import DSparkProvider, DraftSample
from tensorfold.families.qwen4_exp.cuda.dspark_weights import load_reference
from test_flashnext_dspark_metadata import write_tiny, target


@pytest.fixture
def provider(tmp_path):
    config, _ = write_tiny(tmp_path)
    weights = load_reference(tmp_path, target=target(config), chunk_bytes=128)
    torch.manual_seed(23)
    embedding = (torch.randn(config.vocab, config.hidden) * 0.25).half()
    head = (torch.randn(config.vocab, config.hidden) * 0.2).half()
    result = DSparkProvider(
        weights,
        target=target(config),
        embed=lambda ids: embedding[ids],
        head=lambda x: ref.linear(x, head),
        target_identity="tiny-target-v1",
        prefill_rows=4,
    )
    result.embedding = embedding
    result.target_head = head
    return result


def owner(position=0, capacity=24):
    return SimpleNamespace(pos=position, capacity=capacity)


def absorb(provider, state, rows=3, accepted=None):
    torch.manual_seed(29)
    features = torch.randn(rows, len(provider.config.taps) * provider.config.hidden).half() * 0.1
    update = provider.stage_context(state, features)
    count = rows if accepted is None else accepted
    state.target.pos = state.length + count
    provider.commit(state, update, count)
    return features


def test_tap_translation_fp32_mean_and_owned_storage(provider):
    c = provider.config
    collector = provider.taps(2)
    x = torch.randn(2, 4, c.hidden).half()
    expected = x.float().mean(1).half()
    collector.before_layer(0, x)  # Not a declared output tap.
    collector.before_layer(1, x)
    x.zero_()  # Later layer scratch mutation must not overwrite prior capture.
    collector.before_layer(3, x.flatten(1))
    features = collector.finish().view(2, 2, c.hidden)
    assert torch.equal(features[:, 0].view(torch.uint8), expected.view(torch.uint8))
    assert torch.count_nonzero(features[:, 1]) == 0
    with pytest.raises(ValueError, match="finished"):
        collector.before_layer(1, x)
    with pytest.raises(ValueError, match="incomplete|finished"):
        provider.taps(2).finish()
    with pytest.raises(ValueError, match="admitted"):
        provider.taps(5)
    collector = provider.taps(2)
    collector.before_layer(1, x)
    with pytest.raises(ValueError, match="duplicate"):
        collector.before_layer(1, x)


def test_gqa_attention_matches_scalar_fp64_and_is_bidirectional():
    torch.manual_seed(31)
    q = torch.randn(3, 4, 4).half() * 0.2
    k = torch.randn(5, 2, 4).half() * 0.2
    v = torch.randn(5, 2, 4).half()
    out = ref.attend(q, k, v).double().reshape(3, 4, 4)
    expected = torch.empty_like(out)
    for row in range(3):
        for head in range(4):
            kv = head // 2
            logits = [sum(float(q[row, head, d]) * float(k[col, kv, d]) for d in range(4)) / 2 for col in range(5)]
            denom = sum(math.exp(x - max(logits)) for x in logits)
            for dim in range(4):
                expected[row, head, dim] = sum(
                    math.exp(logits[col] - max(logits)) / denom * float(v[col, kv, dim]) for col in range(5)
                )
    torch.testing.assert_close(out, expected, rtol=0.001, atol=5e-5)
    changed = v.clone()
    changed[-1] += 2
    assert not torch.equal(ref.attend(q, k, v)[0], ref.attend(q, k, changed)[0])


def test_rope_matches_scalar_neox_and_ordinary_gamma():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[0.5, 1.0, 1.5, 2.0]]], dtype=torch.float16)
    positions = torch.tensor([3, 11])
    out = ref.rope(x, positions, 10000)
    expected = x.clone()
    for row, p in enumerate(positions.tolist()):
        for d in range(2):
            angle = p * 10000 ** (-2 * d / 4)
            a, b = float(x[row, 0, d]), float(x[row, 0, d + 2])
            expected[row, 0, d] = a * math.cos(angle) - b * math.sin(angle)
            expected[row, 0, d + 2] = a * math.sin(angle) + b * math.cos(angle)
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    # Ordinary gamma=1 must not become the stock MTP centered parameter+1.
    torch.testing.assert_close(
        ref.rms(x, torch.ones(4), 1e-6).double(),
        x.double() / torch.sqrt(x.double().square().mean(-1, keepdim=True) + 1e-6),
        rtol=0.001,
        atol=0.0001,
    )


def test_context_commit_uses_projected_features_not_autoregressive_draft_rows(provider):
    state = provider.make_state(owner(), capacity=24)
    features = absorb(provider, state)
    c = provider.config
    t = provider.weights.tensors
    # Independent FP64 projection, then explicit FP16 boundaries.
    projected = (features.double() @ t["fc.weight"].double().T).half()
    projected = (
        projected.double()
        / torch.sqrt(projected.double().square().mean(-1, keepdim=True) + c.eps)
        * t["hidden_norm.weight"].double()
    ).half()
    first_v = (projected.double() @ t["layers.0.self_attn.v_proj.weight"].double().T).half().view(3, 2, 4)
    assert torch.equal(state.values[0, :3].view(torch.uint8), first_v.view(torch.uint8))
    before = (state.keys[:, :3].clone(), state.values[:, :3].clone(), state.length, state.version, state.target.pos)
    p = provider.propose(state, 4)
    assert p.verify_ids.tolist() == [4, *p.ids.tolist()] and p.ids.shape == (3,)
    assert torch.equal(state.keys[:, :3], before[0]) and torch.equal(state.values[:, :3], before[1])
    assert (state.length, state.version, state.target.pos) == before[2:]


def test_markov_chain_and_confidence_follow_previous_sample(provider):
    state = provider.make_state(owner(), capacity=24)
    absorb(provider, state)
    t = provider.weights.tensors
    c = provider.config
    forced = [7, 8, 9]
    seen = []

    def sampler(logits, step):
        seen.append(logits.clone())
        q = torch.zeros(1, c.vocab, dtype=torch.float32)
        q[0, forced[step]] = 1
        return DraftSample(torch.tensor([forced[step]]), q)

    proposal = provider.propose(state, 4, sampler=sampler)
    inputs = torch.tensor([4, c.mask, c.mask])
    hidden = ref.block_hidden(provider.weights, provider.embedding[inputs], state.keys, state.values, state.length)
    base = ref.linear(hidden, provider.target_head)
    previous = [4, 7, 8]
    expected = []
    confidence = []
    for step, prev in enumerate(previous):
        bias = (t["markov_head.markov_w1.weight"][prev].double() @ t["markov_head.markov_w2.weight"].double().T).half()
        expected.append((base[step].double() + bias.double()).half())
        feat = torch.cat((hidden[step], t["markov_head.markov_w1.weight"][prev])).double()
        scalar = float(feat @ t["confidence_head.proj.weight"][0].double() + t["confidence_head.proj.bias"][0].double())
        confidence.append(1 / (1 + math.exp(-scalar)))
    assert proposal.ids.tolist() == forced
    torch.testing.assert_close(proposal.corrected_logits, torch.stack(expected), rtol=0, atol=0)
    assert all(torch.equal(x[0], proposal.corrected_logits[i]) for i, x in enumerate(seen))
    torch.testing.assert_close(
        proposal.confidence.double(), torch.tensor(confidence, dtype=torch.float64), rtol=1e-6, atol=1e-7
    )
    assert torch.equal(proposal.probabilities.argmax(-1), proposal.ids)


def test_partial_commit_rollback_copy_reset_and_stale_updates(provider):
    state = provider.make_state(owner(), capacity=24)
    features = absorb(provider, state, rows=4, accepted=2)
    assert state.length == 2
    snapshot = state.keys[:, :2].clone()
    child_owner = owner(2)
    child = provider.copy_prefix(state, child_owner, capacity=24)
    assert child.keys.data_ptr() != state.keys.data_ptr() and torch.equal(child.keys[:, :2], snapshot)
    update = provider.stage_context(state, features[:1])
    with pytest.raises(ValueError, match="one pending"):
        provider.stage_context(state, features[:1])
    with pytest.raises(ValueError, match="verified target"):
        provider.commit(state, update, 1)
    with pytest.raises(ValueError, match="stale|foreign"):
        provider.commit(child, update, 1)
    provider.discard(state)
    with pytest.raises(ValueError, match="stale|foreign"):
        provider.commit(state, update, 0)
    state.target.pos = 1
    provider.rewind(state, 1)
    assert state.length == 1 and torch.equal(child.keys[:, :2], snapshot)
    state.target.pos = 0
    provider.reset(state)
    assert state.length == 0 and child.length == 2
    assert provider.propose(state, 4).ids.shape == (3,)


@pytest.mark.parametrize(
    "sample",
    [
        lambda v: DraftSample(torch.tensor([v]), torch.ones(1, v) / v),
        lambda v: DraftSample(torch.tensor([0]), torch.zeros(1, v)),
        lambda v: DraftSample(torch.tensor([0]), torch.full((1, v), float("nan"))),
        lambda v: torch.tensor([0]),
    ],
)
def test_sampler_must_supply_exact_valid_global_distribution(provider, sample):
    state = provider.make_state(owner(), capacity=24)
    with pytest.raises(ValueError, match="sampler"):
        provider.propose(state, 1, sampler=lambda *_: sample(provider.config.vocab))
    assert state.length == 0


def test_provider_identity_context_and_head_guards(provider):
    state = provider.make_state(owner(), capacity=4)
    with pytest.raises(ValueError, match="admitted"):
        provider.propose(state, 1, block=4)
    state.target.pos = 1
    with pytest.raises(ValueError, match="cursor"):
        provider.propose(state, 1)
    state.target.pos = 0
    state.identity = "wrong"
    with pytest.raises(ValueError, match="identity"):
        provider.propose(state, 1)
    state.identity = provider.identity
    provider.head = lambda x: torch.zeros(x.shape[0], 3)
    with pytest.raises(ValueError, match="full CPU vocabulary"):
        provider.propose(state, 1)
    with pytest.raises(ValueError, match="empty target"):
        provider.make_state(owner(1), capacity=4)


def test_fused_add_norm_keeps_unrounded_sum_for_variance():
    x = torch.tensor([[1.0, 0.0003]], dtype=torch.float16)
    res = torch.tensor([[0.0004, 1.0]], dtype=torch.float16)
    out, saved = ref.add_norm(x, res, torch.ones(2), 1e-6)
    value = x.double() + res.double()
    expected = (value / torch.sqrt(value.square().mean(-1, keepdim=True) + 1e-6)).half()
    assert torch.equal(saved, (x.float() + res.float()).half())
    assert torch.equal(out, expected)


def test_full_backbone_against_independent_fp64_block_oracle(provider):
    state = provider.make_state(owner(), capacity=24)
    absorb(provider, state)
    c, t = provider.config, provider.weights.tensors
    x = provider.embedding[torch.tensor([4, c.mask, c.mask])]
    residual = None
    pos = list(range(state.length, state.length + 3))

    def mat(a, w):
        return (a.double() @ w.double().T).half()

    def norm(a, w):
        a = a.double()
        return (a / torch.sqrt(a.square().mean(-1, keepdim=True) + c.eps) * w.double()).half()

    def addnorm(a, r, w):
        value = a.double() if r is None else a.double() + r.double()
        return norm(value, w), a if r is None else value.half()

    def rotate(a):
        out = a.clone()
        for row, p in enumerate(pos):
            for head in range(a.shape[1]):
                for d in range(c.head_dim // 2):
                    angle = p * c.theta ** (-2 * d / c.head_dim)
                    u = float(a[row, head, d])
                    v = float(a[row, head, d + c.head_dim // 2])
                    out[row, head, d] = u * math.cos(angle) - v * math.sin(angle)
                    out[row, head, d + c.head_dim // 2] = u * math.sin(angle) + v * math.cos(angle)
        return out

    for layer in range(c.layers):
        base = f"layers.{layer}."
        ab = base + "self_attn."
        z, residual = addnorm(x, residual, t[base + "input_layernorm.weight"])
        q = rotate(norm(mat(z, t[ab + "q_proj.weight"]).view(3, c.heads, c.head_dim), t[ab + "q_norm.weight"]))
        k = rotate(norm(mat(z, t[ab + "k_proj.weight"]).view(3, c.kv_heads, c.head_dim), t[ab + "k_norm.weight"]))
        v = mat(z, t[ab + "v_proj.weight"]).view(3, c.kv_heads, c.head_dim)
        k = torch.cat((state.keys[layer, : state.length], k))
        v = torch.cat((state.values[layer, : state.length], v))
        att = torch.empty(3, c.heads, c.head_dim, dtype=torch.float16)
        for row in range(3):
            for head in range(c.heads):
                kh = head // (c.heads // c.kv_heads)
                scores = [
                    sum(float(q[row, head, d]) * float(k[col, kh, d]) for d in range(c.head_dim))
                    / math.sqrt(c.head_dim)
                    for col in range(k.shape[0])
                ]
                probs = [math.exp(s - max(scores)) for s in scores]
                denom = sum(probs)
                for d in range(c.head_dim):
                    att[row, head, d] = sum(probs[col] * float(v[col, kh, d]) / denom for col in range(k.shape[0]))
        x = mat(att.flatten(1), t[ab + "o_proj.weight"])
        z, residual = addnorm(x, residual, t[base + "post_attention_layernorm.weight"])
        gate = mat(z, t[base + "mlp.gate_proj.weight"]).double()
        up = mat(z, t[base + "mlp.up_proj.weight"]).double()
        active = (gate / (1 + (-gate).exp()) * up).half()
        x = mat(active, t[base + "mlp.down_proj.weight"])
    expected = addnorm(x, residual, t["norm.weight"])[0]
    actual = ref.block_hidden(
        provider.weights, provider.embedding[torch.tensor([4, c.mask, c.mask])], state.keys, state.values, state.length
    )
    torch.testing.assert_close(actual, expected, rtol=0.001, atol=5e-5)


def test_stale_proposal_rejected_after_committed_target_prefix(provider):
    state = provider.make_state(owner(), capacity=24)
    proposal = provider.propose(state, 1)
    provider.check_proposal(state, proposal)
    absorb(provider, state)
    with pytest.raises(ValueError, match="stale|foreign"):
        provider.check_proposal(state, proposal)


def test_sampling_callback_keeps_actual_filtered_distribution(provider):
    state = provider.make_state(owner(), capacity=24)
    expected = []

    def sampler(logits, step):
        q = torch.zeros_like(logits, dtype=torch.float32)
        q[0, 2] = 0.25
        q[0, 3] = 0.75
        expected.append(q[0].clone())
        return DraftSample(torch.tensor([2 if step == 1 else 3]), q)

    proposal = provider.propose(state, 1, sampler=sampler)
    assert proposal.ids.tolist() == [3, 2, 3]
    assert torch.equal(proposal.probabilities, torch.stack(expected))


def test_reference_components_never_initialize_cuda(provider):
    assert not torch.cuda.is_initialized()
    state = provider.make_state(owner(), capacity=24)
    absorb(provider, state)
    provider.propose(state, 1)
    assert not torch.cuda.is_initialized()


def test_context_piece_and_row_regrouping_keep_identical_stored_bytes(provider):
    torch.manual_seed(41)
    features = torch.randn(4, len(provider.config.taps) * provider.config.hidden).half()
    batched = ref.context_kv(provider.weights, features, 0)
    singles = [ref.context_kv(provider.weights, features[i : i + 1], i) for i in range(4)]
    for side in range(2):
        serial = torch.cat([value[side] for value in singles], dim=1)
        assert torch.equal(batched[side].view(torch.uint8), serial.view(torch.uint8))
