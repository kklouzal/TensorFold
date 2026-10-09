"""Real CUDA keyed/mapped/candidate contracts; no full model or NCCL claim."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_h.cuda import sampler as S

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA primitive qualification')


def oracle(row, ids, position, sampling, *, greedy=False):
    """Independent complete lexicographic FP64/hash/probability rule."""
    values = np.asarray(row, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    order = np.lexsort((ids, -values))
    k = min(20, len(ids)) if greedy else max(1, min(sampling.top_k or len(ids), len(ids)))
    ordered_ids = ids[order[:k]]
    scaled = values[order[:k]] / (1.0 if greedy else max(float(sampling.temperature), 1e-6))
    mass = np.exp(scaled - scaled[0])
    total = mass.sum(dtype=np.float64)
    if greedy:
        return int(ids[order[0]]), np.float32(1.0 / total)
    limit = k
    if 0 < sampling.top_p < 1:
        limit = int((np.cumsum(mass / total) < sampling.top_p).sum()) + 1
    if sampling.min_p > 0:
        limit = min(limit, int((scaled >= scaled[0] + math.log(sampling.min_p)).sum()))
    mask = (1 << 64) - 1
    def mix(value):
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & mask
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)
    scores = []
    for token, value in zip(ordered_ids[:limit], scaled[:limit]):
        key = mix(((sampling.seed & ((1 << 63) - 1)) + 0x9E3779B97F4A7C15) & mask)
        key = mix(key ^ ((position * 0xD1B54A32D192ED03) & mask))
        key = mix(key ^ int(token))
        uniform = (key >> 11) * 2.0 ** -53 + 2.0 ** -54
        scores.append(float(value) - math.log(-math.log(uniform)))
    selected = max(range(limit), key=lambda index: scores[index])
    return int(ordered_ids[selected]), np.float32(mass[selected] / total)


def setup(sampling, rows=2, *, strided=False):
    params = S.Params('cuda')
    params.set(sampling)
    meta = torch.tensor([11, 0, 0, 0], dtype=torch.int32, device='cuda')
    out = torch.empty(rows * (2 if strided else 1), dtype=torch.int32, device='cuda')[::2] if strided else torch.empty(rows, dtype=torch.int32, device='cuda')
    prob = torch.empty(rows * (3 if strided else 1), dtype=torch.float32, device='cuda')[::3] if strided else torch.empty(rows, dtype=torch.float32, device='cuda')
    return params, meta, out, prob


@pytest.mark.parametrize('sampling', [None, Sampling(53, 0.8, 1, 1), Sampling(53, 0.8, 20, 0.9),
                                     Sampling(53, 0.8, 40, 0.9, 0.2), Sampling(53, 0.8, 0, 0.05),
                                     Sampling(53, 0.8, 0, 0), Sampling(53, 0.8, 0, 1, 1)])
def test_full_ties_nonmonotone_map_and_probability(sampling):
    logits = torch.ones((2, 64), dtype=torch.bfloat16, device='cuda')
    mapping = torch.arange(1000, 936, -1, dtype=torch.int64, device='cuda')
    params, meta, out, prob = setup(sampling, strided=True)
    S.sample(logits, meta, params, out, prob=prob, id_map=mapping)
    for row, (token, confidence) in enumerate(zip(out.cpu().tolist(), prob.cpu().tolist())):
        wanted, mass = oracle([1] * 64, mapping.cpu().tolist(), 12 + row, sampling, greedy=sampling is None)
        assert token == wanted
        assert abs(confidence - float(mass)) <= 2e-7


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_candidate_boundary_and_original_value_slots(dtype):
    logits = torch.tensor([[3] * 10 + [1] * 54], dtype=dtype, device='cuda')
    mapping = torch.arange(1000, 936, -1, dtype=torch.int64, device='cuda')
    for count in (1, 9, 28, 48, 64):
        original, _ = torch.topk(logits.float(), count, sorted=False)
        vals, ids = S.candidates(logits, count, mapping, sorted=False)
        assert torch.equal(vals.view(torch.int32), original.view(torch.int32))
        want = sorted(zip(logits[0].double().cpu().tolist(), mapping.cpu().tolist()), key=lambda pair: (-pair[0], pair[1]))[:count]
        assert sorted(zip(vals[0].cpu().tolist(), ids[0].cpu().tolist()), key=lambda pair: (-pair[0], pair[1])) == want


def test_original_dtype_greedy_and_largest_signed32_id():
    logits = torch.tensor([[1.0, np.nextafter(1.0, 2.0)], [0.0, -0.0]], dtype=torch.float64, device='cuda')
    mapping = torch.tensor([(1 << 31) - 1, 9], dtype=torch.int64, device='cuda')
    params, meta, out, prob = setup(None)
    S.sample(logits, meta, params, out, prob=prob, id_map=mapping)
    assert out.cpu().tolist() == [9, 9]
    assert prob.cpu().tolist() == [0.5, 0.5]
    huge = torch.tensor([[1e300, np.nextafter(1e300, math.inf)], [-1e300, -np.nextafter(1e300, math.inf)]],
                        dtype=torch.float64, device='cuda')
    S.sample(huge, meta, params, out, id_map=mapping)
    assert out.cpu().tolist() == [9, (1 << 31) - 1]


def test_fp64_nucleus_ranks_original_values_before_temperature():
    low, high = 1.5, np.nextafter(1.5, math.inf)
    assert low / 1.49 == high / 1.49
    logits = torch.tensor([[low, high]], dtype=torch.float64, device='cuda')
    mapping = torch.tensor([2, 9], dtype=torch.int64, device='cuda')
    sampling = Sampling(53, 1.49, 0, 0.49)
    params, meta, out, prob = setup(sampling, rows=1)
    S.sample(logits, meta, params, out, prob=prob, id_map=mapping)
    wanted, mass = oracle([low, high], [2, 9], 12, sampling)
    assert wanted == 9 and out.item() == wanted
    assert prob.item() == float(mass)


def test_one_finite_score_and_legitimate_negative_infinity_masks():
    logits = torch.full((2, 64), -math.inf, dtype=torch.float32, device='cuda')
    logits[:, 31] = -1
    mapping = torch.arange(1000, 936, -1, dtype=torch.int64, device='cuda')
    for sampling in (None, Sampling(53, 0.8, 20, 0.9, 0.1), Sampling(53, 0.8, 0, 0.9, 0.1)):
        params, meta, out, prob = setup(sampling)
        S.sample(logits, meta, params, out, prob=prob, id_map=mapping)
        assert out.cpu().tolist() == [969, 969]
        assert prob.cpu().tolist() == [1.0, 1.0]


def test_raw_fp64_candidate_union_is_distinct_from_fp32_producer_transport():
    values = torch.tensor([[1e300, np.nextafter(1e300, math.inf)], [1.0, np.nextafter(1.0, math.inf)]],
                          dtype=torch.float64, device='cuda')
    ids = torch.tensor([[2, 9], [2, 9]], dtype=torch.int64, device='cuda')
    params, meta, out, prob = setup(None)
    S.sample_candidates(values, ids, meta, params, out, prob=prob)
    assert out.cpu().tolist() == [9, 9]
    assert prob.cpu().tolist() == [1.0, 0.5]


@pytest.mark.parametrize('sampling', [None, Sampling(53, 0.8, 20, 0.9), Sampling(53, 0.8, 40, 0.9),
                                     Sampling(53, 0.8, 0, 0.05), Sampling(53, 0.8, 0, 1)])
def test_virtual_two_shard_candidate_words_and_draw(sampling):
    # One actual GPU, two complete synthetic shards. No NCCL or 2GPU credit.
    logits = torch.ones((2, 128), dtype=torch.bfloat16, device='cuda')
    mapping = torch.arange(1128, 1000, -1, dtype=torch.int64, device='cuda')
    count = S.candidate_count(64, sampling, minimum=28)
    parts = [S.candidates(logits[:, rank * 64:(rank + 1) * 64], count, mapping[rank * 64:(rank + 1) * 64]) for rank in (0, 1)]
    words = [torch.cat([v.contiguous().view(torch.int32), i.contiguous().view(torch.int32)], dim=1) for v, i in parts]
    vals, ids = S.gather_candidates(lambda payload: torch.cat([payload, words[1]], dim=0), *parts[0], world=2)
    params, meta, out, prob = setup(sampling)
    S.sample_candidates(vals, ids, meta, params, out, prob=prob)
    for row, (token, confidence) in enumerate(zip(out.cpu().tolist(), prob.cpu().tolist())):
        wanted, mass = oracle([1] * 128, mapping.cpu().tolist(), 12 + row, sampling, greedy=sampling is None)
        assert token == wanted
        assert abs(confidence - float(mass)) <= 2e-7


@pytest.mark.parametrize('sampling', [None, Sampling(53, 0.8, 3, 0.9), Sampling(53, 0.8, 0, 0.05)])
def test_capture_replay_keeps_mapped_identity_and_confidence(sampling):
    logits = torch.ones((2, 64), dtype=torch.float32, device='cuda')
    mapping = torch.arange(1000, 936, -1, dtype=torch.int64, device='cuda')
    params, meta, out, prob = setup(sampling)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        S.sample(logits, meta, params, out, prob=prob, id_map=mapping)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    expected = out.clone(), prob.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        S.sample(logits, meta, params, out, prob=prob, id_map=mapping)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected[0]) and torch.equal(prob.view(torch.int32), expected[1].view(torch.int32))


def test_strict_scores_original_probability_rule():
    logits = (torch.arange(64, dtype=torch.float32, device='cuda') / 32 - 1)[None].repeat(2, 1)
    mapping = torch.arange(1000, 936, -1, dtype=torch.int64, device='cuda')
    for sampling in (None, Sampling(53, 0.8, 20, 0.9), Sampling(53, 0.8, 0, 0.9)):
        params, meta, out, prob = setup(sampling)
        S.sample(logits, meta, params, out, prob=prob, id_map=mapping)
        for row, (token, confidence) in enumerate(zip(out.cpu().tolist(), prob.cpu().tolist())):
            wanted, mass = oracle(logits[row].double().cpu().tolist(), mapping.cpu().tolist(), 12 + row,
                                  sampling, greedy=sampling is None)
            assert token == wanted
            assert abs(confidence - float(mass)) <= 2e-7


def test_corrected_malformed_metadata_stops_before_native():
    logits = torch.ones((2, 64), device='cuda')
    params, meta, out, prob = setup(None)
    for changed in ({'id_map': torch.zeros(63, dtype=torch.int64, device='cuda')},
                    {'id_map': torch.zeros(64, dtype=torch.float32, device='cuda')},
                    {'out': torch.empty(1, dtype=torch.int32, device='cuda')},
                    {'out': torch.empty(2, dtype=torch.float32, device='cuda')},
                    {'meta': torch.empty(0, dtype=torch.int32, device='cuda')},
                    {'meta': meta.float()}, {'prob': torch.empty(3, device='cuda')}):
        kwargs = dict(logits=logits, meta=meta, params=params, out=out, prob=prob)
        kwargs.update(changed)
        with pytest.raises(ValueError):
            S.keyed(**kwargs)
    assert torch.cuda.is_available()
