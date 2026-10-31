"""Bounded SM70 DSpark math/lifetime gates; no checkpoint or engine registration."""

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import dspark_reference as ref
from tensorfold.families.qwen4_exp.cuda.dspark_config import DSparkConfig
from tensorfold.families.qwen4_exp.cuda.dspark_cuda import DSparkCUDA, DeviceWeights, gqa_attention
from tensorfold.families.qwen4_exp.cuda.dspark_provider import DraftSample
from tensorfold.families.qwen4_exp.cuda.dspark_weights import DSparkWeights, plan_weights, tensor_shapes
from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import Dense

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native SM70 DSpark gate")


def tiny_config(hidden=32):
    return DSparkConfig(hidden, 41, 2, 8, 2, 8, 64, 4, (0, 2), 40, 4, 7, 64, 1e-6, 10000)


def tiny_arrays(c):
    gen = torch.Generator().manual_seed(241)
    return {name: (torch.randn(shape, generator=gen) * .05 + (1 if 'norm' in name else 0)).to(
        torch.float32 if 'norm' in name or name.startswith('confidence_head.') else torch.float16
    ) for name, shape in tensor_shapes(c).items()}


def local_weights(c, arrays, rank=0, world=1, device='cuda'):
    header, offset = {}, 0
    for name, shape in tensor_shapes(c).items():
        size = math.prod(shape) * 2
        header[name] = {'shape': list(shape), 'dtype': 'F16', 'data_offsets': [offset, offset + size]}
        offset += size
    result = {}
    for plan in plan_weights(c, header, tp=(rank, world)):
        value = arrays[plan.name]
        if plan.rows:
            value = value[plan.rows[0]:plan.rows[1]]
        if plan.columns:
            value = value[:, plan.columns[0]:plan.columns[1]]
        result[plan.name] = value.contiguous().to(device)
    return DeviceWeights(c, result, 'trusted-tiny-source', rank, world, f'tiny-{rank}')


def target_state(position=0):
    return SimpleNamespace(pos=position, capacity=64, version=0, kv_identity=('fp16', 'int8'), cur=[0, 0],
                           _kv_error=None, _resize_error=None)


def tiny_provider(c=None, *, rank=0, world=1, comm=None, device='cuda'):
    c = tiny_config() if c is None else c
    gen = torch.Generator().manual_seed(251)
    embeds = (torch.randn(c.vocab, c.hidden, generator=gen) * .2).half().to(device)
    head = (torch.randn(c.vocab, c.hidden, generator=gen) * .1).half().to(device)
    target = SimpleNamespace(hidden=c.hidden, vocab=c.vocab, layers=c.target_layers, streams=4)
    weights = local_weights(c, tiny_arrays(c), rank, world, device)
    def callback(x, shard):
        return Dense(head[shard.start:shard.stop]).matmul(x)
    provider = DSparkCUDA(weights, target=target, embed=lambda ids: embeds.index_select(0, ids), head=callback,
                         target_identity='target-tiny-source', comm=comm, prefill_rows=64)
    provider.embedding, provider.head_weight = embeds, head
    return provider


def tiny_features(c, rows):
    return (torch.randn(rows, len(c.taps) * c.hidden, generator=torch.Generator().manual_seed(257)) * .125).half()


def assert_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.fixture(autouse=True)
def sm70_only():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip('SM70 DSpark scope')


def test_native_gqa_scores_softmax_and_values_match_independent_scalar_double():
    gen = torch.Generator().manual_seed(263)
    q = torch.randn(3, 4, 8, generator=gen).half() * .2
    k = torch.randn(6, 2, 8, generator=gen).half() * .2
    v = torch.randn(6, 2, 8, generator=gen).half()
    expected = torch.empty(3, 4, 8, dtype=torch.float64)
    for row in range(3):
        for head in range(4):
            kv = head // 2
            scores = [sum(float(q[row, head, d]) * float(k[t, kv, d]) for d in range(8)) / math.sqrt(8) for t in range(6)]
            weights = [math.exp(s - max(scores)) for s in scores]
            for dim in range(8):
                expected[row, head, dim] = sum(w * float(v[t, kv, dim]) for t, w in enumerate(weights)) / sum(weights)
    actual = gqa_attention(q.cuda(), k.cuda(), v.cuda())
    torch.testing.assert_close(actual.cpu().view(3, 4, 8).double(), expected, rtol=.002, atol=.0005)


@pytest.mark.parametrize('rows', [1, 3, 18, 35])
def test_native_target_context_regrouping_keeps_exact_stored_bytes(rows):
    p = tiny_provider()
    f = tiny_features(p.config, rows).cuda()
    a, b = (p.make_state(target_state(), capacity=64) for _ in range(2))
    update = p.stage_context(a, f)
    a.target.pos = rows
    p.commit(a, update, rows)
    for start in range(0, rows, 2):
        part = f[start:start + 2]
        update = p.stage_context(b, part)
        b.target.pos += part.shape[0]
        p.commit(b, update, part.shape[0])
    assert_bits(a.keys[:, :rows], b.keys[:, :rows])
    assert_bits(a.values[:, :rows], b.values[:, :rows])
    p.close_state(a)
    p.close_state(b)


@pytest.mark.parametrize('block', [1, 3, 7])
def test_native_block_markov_confidence_against_cpu_reference(block):
    p = tiny_provider()
    c = p.config
    state = p.make_state(target_state(), capacity=64)
    features = tiny_features(c, 5)
    update = p.stage_context(state, features.cuda())
    state.target.pos = 5
    p.commit(state, update, 5)
    reference = DSparkWeights(c, {k: v.cpu() for k, v in p.weights.tensors.items()}, 'ref')
    ck, cv = ref.context_kv(reference, features, 0)
    torch.testing.assert_close(state.keys[:, :5].cpu(), ck, rtol=.003, atol=.0005)
    torch.testing.assert_close(state.values[:, :5].cpu(), cv, rtol=.003, atol=.0005)
    forced = [(7 + row * 11) % c.vocab for row in range(block)]
    def sample(logits, *, step, position, shard):
        assert position == 6 + step
        q = torch.zeros_like(logits, dtype=torch.float32)
        q[0, forced[step]] = 1
        return DraftSample(torch.tensor([forced[step]], device='cuda'), q)
    out = p.propose(state, 4, block=block, sampler=sample).wait()
    inputs = torch.full((block,), c.mask, dtype=torch.int64)
    inputs[0] = 4
    h = ref.block_hidden(reference, p.embedding.cpu()[inputs], ck, cv, 5)
    base = ref.linear(h, p.head_weight.cpu())
    corrected, confidence = [], []
    for row, prev_id in enumerate([4, *forced[:-1]]):
        prev = reference.tensors['markov_head.markov_w1.weight'][prev_id:prev_id + 1]
        corrected.append((base[row:row + 1].float() + ref.linear(prev, reference.tensors['markov_head.markov_w2.weight']).float()).half())
        joined = torch.cat((h[row:row + 1], prev), -1).float()
        confidence.append((joined @ reference.tensors['confidence_head.proj.weight'].T
                           + reference.tensors['confidence_head.proj.bias']).flatten().sigmoid())
    torch.testing.assert_close(out.corrected_logits.cpu(), torch.cat(corrected), rtol=.01, atol=.003)
    torch.testing.assert_close(out.confidence.cpu(), torch.cat(confidence), rtol=.001, atol=.0002)
    assert out.ids.cpu().tolist() == forced
    assert out.verify_ids.cpu().tolist() == [4, *forced]
    assert state.length == 5
    p.close_state(state)


def test_native_callback_scratch_ownership_and_cross_stream_cache_lifecycle():
    p = tiny_provider()
    state = p.make_state(target_state(), capacity=32)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        update = p.stage_context(state, tiny_features(p.config, 4).cuda())
        expected = update.keys.clone()
    with torch.cuda.stream(consumer):
        state.target.pos = 3
        p.commit(state, update, 3)
        snapshot = p.snapshot(state)
    with torch.cuda.stream(producer):
        p.resize(state, 64)
        ids = torch.empty(1, dtype=torch.int64, device='cuda')
        q = torch.empty(1, p.shard.width, dtype=torch.float32, device='cuda')
        def sampler(logits, *, step, **_):
            ids.fill_(7 + step)
            q.zero_()
            q[0, 7 + step] = 1
            logits.fill_(-100)
            return DraftSample(ids, q)
        out = p.propose(state, 4, block=3, sampler=sampler)
    with torch.cuda.stream(consumer):
        out.wait()
        saved = out.ids.clone(), out.probabilities.clone()
        state.target.cur = list(state.target.cur)
        p.restore(state, snapshot)
        p.close_state(state)
    consumer.synchronize()
    assert saved[0].cpu().tolist() == [7, 8, 9]
    assert saved[1].argmax(-1).cpu().tolist() == [7, 8, 9]
    torch.testing.assert_close(snapshot.keys, expected[:, :3], rtol=0, atol=0)
    assert state.keys is state.values is None


def test_native_failed_sample_releases_cache_reader_before_other_stream_writer():
    p = tiny_provider()
    state = p.make_state(target_state(), capacity=64)
    update = p.stage_context(state, tiny_features(p.config, 4).cuda())
    state.target.pos = 4
    p.commit(state, update, 4)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        def invalid(logits, **_):
            return DraftSample(torch.tensor([p.config.vocab], device='cuda'), torch.ones_like(logits, dtype=torch.float32))
        with pytest.raises(ValueError, match='sampler'):
            p.propose(state, 4, sampler=invalid)
    p.close_state(state)
    side.synchronize()
    assert state.closed


def test_native_context_does_not_cast_full_half_weight_matrices(monkeypatch):
    p = tiny_provider()
    pointers = {t.data_ptr() for t in p.weights.tensors.values() if t.ndim == 2 and t.dtype == torch.float16}
    old = torch.Tensor.float
    def checked(t, *args, **kw):
        assert t.data_ptr() not in pointers, 'full FP32 draft weight cast'
        return old(t, *args, **kw)
    monkeypatch.setattr(torch.Tensor, 'float', checked)
    state = p.make_state(target_state(), capacity=64)
    update = p.stage_context(state, tiny_features(p.config, 3).cuda())
    state.target.pos = 3
    p.commit(state, update, 3)
    out = p.propose(state, 4)
    assert out.ids.shape == (7,)
    p.close_state(state)


def test_native_capture_declines_before_any_status_readback():
    p = tiny_provider()
    state = p.make_state(target_state(), capacity=64)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match='eager-only'):
        with torch.cuda.graph(graph):
            p.propose(state, 4)
    p.close_state(state)


def test_native_typed_target_taps_stage_independent_context_before_reuse():
    from tensorfold.families.qwen4_exp.cuda.dspark_taps import TapWorkspace
    c = replace(tiny_config(), hidden=256)
    p = tiny_provider(c)
    states = [p.make_state(target_state(), capacity=64) for _ in range(6)]
    w = TapWorkspace(c, 18, 'cuda')
    capture = w.begin(18, [(s.target, 3 * i, 3 * i + 3) for i, s in enumerate(states)])
    streams = torch.randn(18, 4 * c.hidden, device='cuda', dtype=torch.float16)
    for index in c.capture_before:
        capture.before_layer(index, streams)
    batch = capture.finish()
    counts = [1, 3, 2, 1, 2, 3]
    for state, keep in zip(states, counts, strict=True):
        state.target.pos = keep
    updates = p.stage_taps(states, batch, accepted=counts)
    assert batch.released
    w.features.zero_()
    for state, update, count in zip(states, updates, counts, strict=True):
        assert update.rows == count
        p.commit(state, update, count)
        assert torch.isfinite(state.keys[:, :count]).all()
        p.close_state(state)
    w.close()
