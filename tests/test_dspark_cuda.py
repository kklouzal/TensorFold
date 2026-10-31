"""CPU-only independent equations/transactions for the unregistered GPU adapter.

These controls do not establish CUDA execution or full-model qualification.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import dspark_reference as ref
from tensorfold.families.qwen4_exp.cuda.dspark_config import DSparkConfig
from tensorfold.families.qwen4_exp.cuda.dspark_cuda import (
    DSparkCUDA, DeviceWeights, VocabShard, byte_plan, gqa_attention, load_shard,
)
from tensorfold.families.qwen4_exp.cuda.dspark_provider import DraftSample
from tensorfold.families.qwen4_exp.cuda.dspark_weights import DSparkWeights, plan_weights, tensor_shapes
from test_flashnext_dspark_metadata import config_dict, header, target, write_tiny


def config():
    raw = config_dict()
    raw.update(num_attention_heads=8, intermediate_size=32)
    return DSparkConfig.from_dict(raw)


def arrays(c):
    g = torch.Generator().manual_seed(199)
    return {name: (torch.randn(shape, generator=g) * .07 + (1 if "norm" in name else 0)).to(
        torch.float32 if "norm" in name or name.startswith("confidence_head.") else torch.float16
    ) for name, shape in tensor_shapes(c).items()}


def partition(c, full, rank=0, world=1):
    selected = {}
    for p in plan_weights(c, header(c), tp=(rank, world)):
        t = full[p.name]
        if p.rows:
            t = t[p.rows[0]:p.rows[1]]
        if p.columns:
            t = t[:, p.columns[0]:p.columns[1]]
        selected[p.name] = t.contiguous()
    return DeviceWeights(c, selected, "trusted-tiny-source", rank, world, f"piece-{rank}")


def owner(pos=0, capacity=32):
    return SimpleNamespace(pos=pos, capacity=capacity, version=0, kv_identity=("fp16", "fp16"), cur=[0, 0],
                           _kv_error=None, _resize_error=None)


def provider(c=None, full=None, *, rank=0, world=1, comm=None):
    c = config() if c is None else c
    full = arrays(c) if full is None else full
    g = torch.Generator().manual_seed(211)
    embedding = (torch.randn(c.vocab, c.hidden, generator=g) * .2).half()
    head = (torch.randn(c.vocab, c.hidden, generator=g) * .15).half()
    p = DSparkCUDA(partition(c, full, rank, world), target=target(c), embed=lambda ids: embedding[ids],
                   head=lambda x, s: ref.linear(x, head[s.start:s.stop]), target_identity="target-source-id",
                   comm=comm, prefill_rows=8)
    p.embedding, p.head_weight = embedding, head
    return p


def features(c, rows=4):
    return torch.randn(rows, len(c.taps) * c.hidden, generator=torch.Generator().manual_seed(223)).half() * .125


def absorb(p, state, f=None, keep=None):
    f = features(p.config) if f is None else f
    update = p.stage_context(state, f)
    keep = f.shape[0] if keep is None else keep
    state.target.pos = state.length + keep
    p.commit(state, update, keep)
    return f


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def test_attention_scalar_gqa_and_future_block_is_visible():
    gen = torch.Generator().manual_seed(227)
    q = torch.randn(3, 4, 4, generator=gen).half() * .25
    k = torch.randn(5, 2, 4, generator=gen).half() * .25
    v = torch.randn(5, 2, 4, generator=gen).half()
    expected = torch.empty(3, 4, 4, dtype=torch.float64)
    for r in range(3):
        for h in range(4):
            kv = h // 2
            scores = [sum(float(q[r, h, d]) * float(k[t, kv, d]) for d in range(4)) / 2 for t in range(5)]
            weights = [math.exp(x - max(scores)) for x in scores]
            for d in range(4):
                expected[r, h, d] = sum(w * float(v[t, kv, d]) for t, w in enumerate(weights)) / sum(weights)
    result = gqa_attention(q, k, v)
    torch.testing.assert_close(result.view(3, 4, 4).double(), expected, rtol=.001, atol=.00025)
    changed = v.clone()
    changed[-1] += 3
    assert not torch.equal(result[0], gqa_attention(q, k, changed)[0])


def test_world1_dense_backbone_context_markov_and_confidence_match_reference():
    p = provider()
    state = p.make_state(owner(), capacity=32)
    f = absorb(p, state)
    w = DSparkWeights(p.config, p.weights.tensors, "reference")
    expected_k, expected_v = ref.context_kv(w, f, 0)
    bits(state.keys[:, :4], expected_k)
    bits(state.values[:, :4], expected_v)
    before = state.keys[:, :4].clone(), state.values[:, :4].clone(), state.length, state.version
    calls = []
    chosen = (7, 8, 9)

    def sample(logits, *, step, position, shard):
        calls.append((step, position, shard, logits.clone()))
        q = torch.zeros_like(logits, dtype=torch.float32)
        q[0, chosen[step]] = 1
        return DraftSample(torch.tensor([chosen[step]]), q)

    out = p.propose(state, 4, sampler=sample)
    hidden = ref.block_hidden(w, p.embedding[torch.tensor([4, p.config.mask, p.config.mask])],
                              state.keys, state.values, 4)
    base = ref.linear(hidden, p.head_weight)
    corrections, confidences = [], []
    for row, previous in enumerate((4, 7, 8)):
        prev = p.weights.tensors["markov_head.markov_w1.weight"][previous:previous + 1]
        corrections.append((base[row:row + 1].float() + ref.linear(
            prev, p.weights.tensors["markov_head.markov_w2.weight"]).float()).half())
        joined = torch.cat((hidden[row:row + 1], prev), -1).double()
        scalar = (joined @ p.weights.tensors["confidence_head.proj.weight"].double().T
                  + p.weights.tensors["confidence_head.proj.bias"].double())
        confidences.append(scalar.flatten().sigmoid())
    bits(out.corrected_logits, torch.cat(corrections))
    torch.testing.assert_close(out.confidence.double(), torch.cat(confidences), rtol=1e-6, atol=1e-7)
    assert out.verify_ids.tolist() == [4, 7, 8, 9]
    assert [(s, pos) for s, pos, *_ in calls] == [(0, 5), (1, 6), (2, 7)]
    assert all(shard == p.shard for _, _, shard, _ in calls)
    bits(state.keys[:, :4], before[0])
    bits(state.values[:, :4], before[1])
    assert (state.length, state.version) == before[2:]
    p.check_proposal(state, out)
    p.discard(state)
    with pytest.raises(ValueError, match="stale|foreign"):
        p.check_proposal(state, out)


def test_partial_context_commit_only_publishes_target_prefix_and_retires_update():
    p = provider()
    state = p.make_state(owner(), capacity=32)
    state.keys.fill_(123)
    state.values.fill_(124)
    update = p.stage_context(state, features(p.config))
    k, v = update.keys.clone(), update.values.clone()
    with pytest.raises(ValueError, match="verified target"):
        p.commit(state, update, 2)
    state.target.pos = 2
    p.commit(state, update, 2)
    bits(state.keys[:, :2], k[:, :2])
    bits(state.values[:, :2], v[:, :2])
    assert torch.all(state.keys[:, 2:] == 123) and torch.all(state.values[:, 2:] == 124)
    assert update.keys is update.values is update.owner is None
    with pytest.raises(ValueError, match="stale|foreign"):
        p.commit(state, update, 0)
    assert state.length == 2


def test_regrouped_context_rows_exact_no_autoregressive_draft_cache():
    p = provider()
    f = features(p.config, 7)
    a, b = (p.make_state(owner(), capacity=32) for _ in range(2))
    absorb(p, a, f)
    absorb(p, b, f[:2])
    absorb(p, b, f[2:3])
    absorb(p, b, f[3:])
    bits(a.keys[:, :7], b.keys[:, :7])
    bits(a.values[:, :7], b.values[:, :7])
    before = b.keys.clone(), b.values.clone()
    p.propose(b, 2)
    bits(before[0], b.keys)
    bits(before[1], b.values)


def test_copy_snapshot_restore_resize_reset_and_poisoned_close():
    p = provider()
    state = p.make_state(owner(), capacity=16)
    absorb(p, state)
    snapshot = p.snapshot(state)
    child = p.copy_prefix(state, owner(4), capacity=24)
    assert child.keys.data_ptr() != state.keys.data_ptr()
    bits(child.keys[:, :4], state.keys[:, :4])
    state.target.version += 1  # target storage grew; explicit draft resize
    p.resize(state, 24)
    assert state.capacity == 24
    bits(state.keys[:, :4], snapshot.keys)
    absorb(p, state, features(p.config, 2))
    state.target.pos = 4
    state.target.cur = list(state.target.cur)  # target restore replaces history
    with pytest.raises(ValueError, match="identity changed"):
        p.propose(state, 1)
    p.restore(state, snapshot)
    assert state.length == 4
    bits(state.keys[:, :4], snapshot.keys)
    state.target.pos = 0
    state.target.cur = [0, 0]
    p.reset(state)
    assert state.length == 0 and child.length == 4
    state.target._kv_error = ValueError("target numeric error")
    p.close_state(state)
    assert state.nbytes == 0 and state.keys is state.values is state.target is None
    with pytest.raises(ValueError, match="closed"):
        p.propose(state, 1)


@pytest.mark.parametrize("fault", ["id-high", "id-low", "nan", "negative", "sum", "zero-selected", "shape", "throw"])
def test_invalid_sample_does_not_index_markov_or_mutate_context(fault):
    p = provider()
    state = p.make_state(owner(), capacity=32)
    absorb(p, state)
    before = state.keys.clone(), state.values.clone(), state.length, state.version
    def sample(logits, **_):
        if fault == "throw":
            raise RuntimeError("callback failed")
        if fault == "shape":
            return object()
        token = p.config.vocab if fault == "id-high" else -1 if fault == "id-low" else 1
        q = torch.zeros_like(logits, dtype=torch.float32)
        q[0, 1] = 1
        if fault == "nan":
            q[0, 2] = float("nan")
        elif fault == "negative":
            q[0, 2] = -1
        elif fault == "sum":
            q[0, 1] = .5
        elif fault == "zero-selected":
            q.zero_()
            q[0, 2] = 1
        return DraftSample(torch.tensor([token]), q)
    with pytest.raises(ValueError, match="sampler"):
        p.propose(state, 1, sampler=sample)
    bits(before[0], state.keys)
    bits(before[1], state.values)
    assert (state.length, state.version) == before[2:]


def test_loader_tp_slices_common_identity_piece_hash_and_no_float_weight_copy(tmp_path):
    c, raw = write_tiny(tmp_path)
    loaded = [load_shard(tmp_path, target=target(c), checkpoint_identity="trusted-source", tp=(r, 4),
                         device="cpu", chunk_bytes=128) for r in range(4)]
    assert len({w.identity for w in loaded}) == 1
    assert len({w.piece_hash for w in loaded}) == 4
    for rank, w in enumerate(loaded):
        assert w.nbytes == sum(p.nbytes for p in plan_weights(c, header(c), tp=(rank, 4)))
        for plan in plan_weights(c, header(c), tp=(rank, 4)):
            value = raw[plan.name].to(w.tensors[plan.name].dtype)
            if plan.rows:
                value = value[plan.rows[0]:plan.rows[1]]
            if plan.columns:
                value = value[:, plan.columns[0]:plan.columns[1]]
            bits(value, w.tensors[plan.name])
    with pytest.raises(ValueError, match="identity"):
        load_shard(tmp_path, target=target(c), checkpoint_identity="", device="cpu", chunk_bytes=128)
    with pytest.raises(ValueError, match="load chunk"):
        load_shard(tmp_path, target=target(c), checkpoint_identity="source", device="cpu", chunk_bytes=1)


def test_memory_plan_exact_weights_state_staging_and_pinned_target_config():
    c = DSparkConfig.from_dict(__import__("json").loads(Path("tests/fixtures/dspark_qwen38/dspark_config.json").read_text()))
    for world, resident in ((4, 476913668), (8, 334864388)):
        plan = byte_plan(c, tp=(0, world), capacity=8192, slots=6, prefill_rows=2048)
        assert plan["resident_bytes"] == resident
        assert plan["state_bytes"] == 6 * c.layers * 8192 * (c.kv_heads // world) * c.head_dim * 4
        assert plan["context_staging_bytes"] == 6 * c.layers * 2048 * (c.kv_heads // world) * c.head_dim * 4
        assert plan["full_f32_weight_cast_bytes"] == plan["graph_bytes"] == plan["target_shared_weight_bytes"] == 0
        assert plan["eager_transient_bound"] > 0
    for args in ({"capacity": 0}, {"capacity": 8192, "prefill_rows": 2049}, {"capacity": 8192, "block": 8}):
        with pytest.raises(ValueError):
            byte_plan(c, **args)


class ThreadGroup:
    """Real blocking rank-ordered collectives on CPU, not canned partials."""
    def __init__(self, world):
        self.world, self.packets = world, [None] * world
        self.barrier = threading.Barrier(world, timeout=20)

    def comm(self, rank):
        group = self
        class Comm:
            world = group.world
            def __init__(self):
                self.rank = rank
            def all_gather(self, send, recv):
                group.packets[rank] = send.clone()
                group.barrier.wait()
                recv.copy_(torch.stack(group.packets))
                group.barrier.wait()
        return Comm()


def run_ranks(providers, operation):
    with ThreadPoolExecutor(len(providers)) as pool:
        jobs = [pool.submit(operation, p) for p in providers]
        return [j.result(timeout=30) for j in jobs]


@pytest.mark.parametrize("world", [2, 4, 8])
def test_tp_block_kv_replication_global_ids_confidence_and_actual_q(world):
    c, full = config(), arrays(config())
    group = ThreadGroup(world)
    ps = [provider(c, full, rank=r, world=world, comm=group.comm(r)) for r in range(world)]
    f = features(c)
    def operation(p):
        state = p.make_state(owner(), capacity=32)
        absorb(p, state, f)
        out = p.propose(state, 4)
        return state, out
    results = run_ranks(ps, operation)
    unsharded = provider(c, full)
    baseline_state, baseline = operation(unsharded)
    actual = torch.cat([r[1].corrected_logits for r in results], -1)
    torch.testing.assert_close(actual, baseline.corrected_logits, rtol=.004, atol=.002)
    for p, (state, proposal) in zip(ps, results, strict=True):
        assert proposal.ids.tolist() == baseline.ids.tolist()
        bits(proposal.verify_ids, results[0][1].verify_ids)
        bits(proposal.confidence, results[0][1].confidence)
        first = c.partition(p.rank, world)["first_kv"]
        width = c.partition(p.rank, world)["kv_heads"]
        bits(state.keys[:, :4], baseline_state.keys[:, :4, first:first + width])
        bits(state.values[:, :4], baseline_state.values[:, :4, first:first + width])
    torch.testing.assert_close(torch.cat([r[1].probabilities for r in results], -1).sum(-1), torch.ones(c.block), rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["id-mismatch", "remote-invalid-id", "remote-throw", "remote-shape", "remote-nan"])
def test_tp_sampler_remote_error_reaches_every_rank_before_markov_lookup(fault):
    c = config()
    world = 4
    group = ThreadGroup(world)
    ps = [provider(c, rank=r, world=world, comm=group.comm(r)) for r in range(world)]
    def operation(p):
        state = p.make_state(owner(), capacity=32)
        def sample(logits, **_):
            if p.rank == 2 and fault == "remote-throw":
                raise RuntimeError("remote callback error")
            if p.rank == 2 and fault == "remote-shape":
                return object()
            token = 2 if p.rank == 2 and fault == "id-mismatch" else c.vocab if p.rank == 2 and fault == "remote-invalid-id" else 1
            q = torch.zeros_like(logits, dtype=torch.float32)
            if p.shard.start <= 1 < p.shard.stop:
                q[0, 1 - p.shard.start] = 1
            if p.rank == 2 and fault == "remote-nan":
                q[0, 0] = float("nan")
            return DraftSample(torch.tensor([token]), q)
        with pytest.raises(ValueError, match="sampler"):
            p.propose(state, 4, sampler=sample)
        assert state.length == state.version == 0 and state.pending is None
        return True
    assert all(run_ranks(ps, operation))


def test_tp_greedy_tie_uses_minimum_global_id_and_shared_callbacks_are_not_weights():
    c = config()
    group = ThreadGroup(4)
    ps = [provider(c, rank=r, world=4, comm=group.comm(r)) for r in range(4)]
    for p in ps:
        p.weights.tensors["markov_head.markov_w2.weight"].zero_()
        p.head = lambda x, s: torch.zeros(x.shape[0], s.width)
    outputs = run_ranks(ps, lambda p: p.propose(p.make_state(owner(), capacity=32), 4))
    assert all(p.ids.tolist() == [0, 0, 0] for p in outputs)
    assert all("target" not in name for name in ps[0].weights.tensors)


def test_decline_stale_foreign_pending_and_wrong_shared_head():
    p = provider()
    state = p.make_state(owner(), capacity=32)
    out = p.propose(state, 1)
    other = p.make_state(owner(), capacity=32)
    with pytest.raises(ValueError, match="foreign"):
        p.check_proposal(other, out)
    p.stage_context(state, features(p.config))
    with pytest.raises(ValueError, match="pending"):
        p.propose(state, 1)
    p.discard(state)
    p.head = lambda x, _: torch.zeros(x.shape[0], 1)
    with pytest.raises(ValueError, match="local vocabulary"):
        p.propose(state, 1)
    with pytest.raises(ValueError, match="empty target"):
        p.make_state(owner(1), capacity=32)
    state.target.cur = list(state.target.cur)
    with pytest.raises(ValueError, match="identity changed"):
        p.snapshot(state)
    with pytest.raises(ValueError, match="comm"):
        provider(rank=0, world=4)


def test_weight_shape_dtype_and_checkpoint_identity_are_strict():
    c = config()
    data = arrays(c)
    with pytest.raises(ValueError, match="identity"):
        DeviceWeights(c, data, "")
    broken = dict(data)
    broken["fc.weight"] = broken["fc.weight"].float()
    with pytest.raises(ValueError, match="dtype"):
        DeviceWeights(c, broken, "id")
    with pytest.raises(ValueError, match="namespace"):
        DeviceWeights(c, {}, "id")
    with pytest.raises(ValueError, match="vocabulary"):
        VocabShard.make(replace(c, vocab=1), 0, 4)


@pytest.mark.parametrize('committed', [False, True])
def test_typed_tap_batch_stages_independent_owned_kv_and_releases_lease(committed):
    from tensorfold.families.qwen4_exp.cuda.dspark_taps import TapWorkspace
    c = replace(config(), hidden=256)
    p = provider(c)
    owners = [owner(), owner()]
    states = [p.make_state(t, capacity=32) for t in owners]
    workspace = TapWorkspace(c, 6, 'cpu')
    capture = workspace.begin(6, [(owners[0], 0, 3), (owners[1], 3, 6)])
    streams = torch.randn(6, 4 * c.hidden, generator=torch.Generator().manual_seed(229)).half()
    for before in c.capture_before:
        capture.before_layer(before, streams)
    batch = capture.finish()
    features_before = batch.features.clone()
    counts = [2, 1] if committed else None
    if committed:
        owners[0].pos, owners[1].pos = counts
    updates = p.stage_taps(states, batch, accepted=counts)
    assert batch.released and batch.owner is None
    workspace.features.zero_()
    for index, (state, update) in enumerate(zip(states, updates, strict=True)):
        count = [2, 1][index] if committed else 3
        expected = ref.context_kv(DSparkWeights(c, p.weights.tensors, 'ref'),
                                  features_before[index * 3:index * 3 + count], 0)
        bits(update.keys, expected[0])
        bits(update.values, expected[1])
        state.target.pos = count
        p.commit(state, update, count)
        assert state.length == count
    workspace.close()


def test_tap_stage_failure_atomic_across_lanes_and_retires_poisoned_lease():
    from tensorfold.families.qwen4_exp.cuda.dspark_taps import TapWorkspace
    c = replace(config(), hidden=256)
    p = provider(c)
    owners = [owner(), owner()]
    states = [p.make_state(t, capacity=32) for t in owners]
    workspace = TapWorkspace(c, 2, 'cpu')
    capture = workspace.begin(2, [(owners[0], 0, 1), (owners[1], 1, 2)])
    stream = torch.ones(2, 4 * c.hidden, dtype=torch.float16)
    for before in c.capture_before:
        capture.before_layer(before, stream)
    batch = capture.finish()
    workspace.features[1, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        p.stage_taps(states, batch)
    assert batch.released and all(s.pending is None and s.length == 0 for s in states)
    workspace.close()


def test_context_alias_guard_and_resize_cannot_acknowledge_target_reset():
    p = provider()
    state = p.make_state(owner(), capacity=32)
    update = p.stage_context(state, features(p.config))
    old_value = update.values
    update.values = update.keys
    state.target.pos = 1
    with pytest.raises(ValueError, match='independent'):
        p.commit(state, update, 1)
    update.values = old_value
    p.commit(state, update, 1)
    state.target.cur = list(state.target.cur)
    with pytest.raises(ValueError, match='resize'):
        p.resize(state, 32)
    original_values = state.values
    state.values = state.keys
    with pytest.raises(ValueError, match='storage'):
        p.snapshot(state)
    state.values = original_values
    p.close_state(state)


def test_sampler_reused_ids_q_and_inplace_filters_cannot_rewrite_exported_rows():
    p = provider()
    state = p.make_state(owner(), capacity=32)
    baseline = p.propose(state, 4)
    ids = torch.empty(1, dtype=torch.int64)
    q = torch.empty(1, p.config.vocab, dtype=torch.float32)
    raw = []
    def sample(logits, *, step, **_):
        raw.append(logits.clone())
        ids.fill_(7 + step)
        q.zero_()
        q[0, 7 + step] = 1
        logits.fill_(-100)  # permitted callback-local penalty/filter scratch
        return DraftSample(ids, q)
    proposal = p.propose(state, 4, sampler=sample)
    assert proposal.ids.tolist() == [7, 8, 9]
    bits(proposal.corrected_logits, torch.cat(raw))
    assert proposal.probabilities.argmax(-1).tolist() == [7, 8, 9]
    bits(proposal.corrected_logits[:1], baseline.corrected_logits[:1])
    ids.zero_()
    q.zero_()
    assert proposal.ids.tolist() == [7, 8, 9]
    assert proposal.probabilities.sum(-1).tolist() == [1, 1, 1]


def test_context_normalization_fixed_rows_retains_complete_independent_token_geometry():
    from tensorfold.families.qwen4_exp.cuda.dspark_cuda import context_norm
    g = torch.Generator().manual_seed(233)
    for shape in ((35, 256), (35, 2, 128)):
        x = torch.randn(shape, generator=g).half()
        scale = torch.randn(shape[-1], generator=g).float()
        together = context_norm(x, scale, 1e-6)
        pieces = torch.cat([context_norm(x[:1], scale, 1e-6), context_norm(x[1:18], scale, 1e-6),
                            context_norm(x[18:], scale, 1e-6)])
        bits(together, pieces)
        torch.testing.assert_close(together.double(), (x.double() / torch.sqrt(
            x.double().square().mean(-1, keepdim=True) + 1e-6) * scale.double()).half().double(), rtol=.001, atol=.004)


def test_context_publication_requires_external_target_numeric_validation():
    p = provider()
    state = p.make_state(owner(), capacity=32)
    state.target._kv_pending = True
    update = p.stage_context(state, features(p.config))
    state.target.pos = 2
    with pytest.raises(ValueError, match='verified target'):
        p.commit(state, update, 2)
    assert state.length == state.version == 0 and state.pending is update
    state.target._kv_pending = False
    p.commit(state, update, 2)
    state.target._kv_pending = True
    with pytest.raises(ValueError, match='validated'):
        p.propose(state, 4)
    state.target._kv_pending = False
    p.stage_context(state, features(p.config, 1))
    state.target._kv_error = ValueError('target codec failure')
    p.discard(state)  # Cleanup must remain possible without publishing a row.
    assert state.pending is None and state.length == 2
    p.close_state(state)


def test_explicit_position_keyed_choice_does_not_fabricate_stochastic_q():
    from tensorfold.families.qwen4_exp.cuda.dspark_cuda import PositionKeyedChoice
    p = provider()
    state = p.make_state(owner(), capacity=32)
    seen = []
    def sample(logits, *, step, position, shard):
        seen.append(position)
        return PositionKeyedChoice(torch.tensor([7 + step]))
    proposal = p.propose(state, 4, sampler=sample).wait()
    assert proposal.ids.tolist() == [7, 8, 9]
    assert proposal.probabilities is None
    assert seen == [1, 2, 3]
    assert proposal.nbytes > 0


def test_sampling_q_contract_cannot_change_mid_proposal():
    from tensorfold.families.qwen4_exp.cuda.dspark_cuda import PositionKeyedChoice
    p = provider()
    state = p.make_state(owner(), capacity=32)
    def sample(logits, *, step, **_):
        if step:
            return PositionKeyedChoice(torch.tensor([1]))
        q = torch.zeros_like(logits, dtype=torch.float32)
        q[0, 1] = 1
        return DraftSample(torch.tensor([1]), q)
    with pytest.raises(ValueError, match='contract changed'):
        p.propose(state, 4, sampler=sample)


def test_remote_q_contract_disagreement_rejects_all_ranks_before_indexing():
    from tensorfold.families.qwen4_exp.cuda.dspark_cuda import PositionKeyedChoice
    c = config()
    group = ThreadGroup(4)
    ps = [provider(c, rank=r, world=4, comm=group.comm(r)) for r in range(4)]
    def operation(p):
        state = p.make_state(owner(), capacity=32)
        def sample(logits, **_):
            if p.rank == 2:
                return PositionKeyedChoice(torch.tensor([1]))
            q = torch.zeros_like(logits, dtype=torch.float32)
            if p.shard.start <= 1 < p.shard.stop:
                q[0, 1 - p.shard.start] = 1
            return DraftSample(torch.tensor([1]), q)
        with pytest.raises(ValueError, match='sampler'):
            p.propose(state, 4, sampler=sample)
        return True
    assert all(run_ranks(ps, operation))
