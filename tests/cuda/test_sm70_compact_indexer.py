"""Actual SM70 compact state writes; host/source checks do not qualify these."""

import pytest
import torch


@pytest.fixture(autouse=True)
def required_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an actual SM70 device and the selected source-built SDK")


def test_guarded_publication_preserves_unselected_rows_and_rejects_stale_generation():
    from tensorfold.cuda.sm70.compact_indexer import publish

    source = torch.arange(4 * 128, device="cuda").reshape(4, 128).half()
    destination = torch.full((16, 128), 17, dtype=torch.float16, device="cuda")
    generations = torch.tensor([2, 5], dtype=torch.int64, device="cuda")
    status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
    publish(source, destination, 8, generations, (2, 5), status, 3)
    assert torch.equal(destination[8:12], source)
    assert torch.all(destination[:8] == 17) and torch.all(destination[12:] == 17)
    before = destination.clone()
    generations[0].fill_(3)
    publish(source.neg(), destination, 8, generations, (2, 5), status, 3)
    assert torch.equal(destination, before) and status.tolist() == [32, 3]


def test_native_publication_rejects_generation_and_status_alias_destinations():
    from tensorfold.cuda.sm70.compact_indexer import publish

    generations = torch.tensor([2, 5], dtype=torch.int64, device="cuda")
    status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
    for destination in (generations, status):
        before = destination.clone()
        with pytest.raises(RuntimeError):
            publish(torch.zeros_like(destination), destination, 0, generations, (2, 5), status, 3)
        assert torch.equal(destination, before)


def test_chunked_compact_pools_and_tail_match_same_gpu_full_history_reference():
    from tensorfold.cuda.sm70 import family_ops as ops
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import Cache, Span, Staging
    from types import SimpleNamespace as NS

    torch.manual_seed(37)
    device = torch.device("cuda")
    generations = torch.zeros(2, dtype=torch.int64, device=device)
    status = torch.tensor([0, -2], dtype=torch.int32, device=device)
    cache = Cache(32, 128, device, generations, status, layer=3)
    stage = Staging(7, 12, device, mtp=False)
    raw = torch.randn((32, 128), device=device).half()
    coordinates = torch.stack((torch.arange(32, device=device) * 2,
                               torch.arange(32, device=device) * 3 + 7,
                               torch.arange(32, device=device) * 5 + 9), 1).long()
    gamma, freq = torch.ones(128, device=device), torch.linspace(0.01, 0.1, 32, device=device)
    pooled, reference = torch.zeros(8, 128, dtype=torch.float16, device=device), torch.zeros(8, 128, dtype=torch.float16, device=device)
    history = torch.zeros_like(raw)
    for count in (1, 3, 7, 5):
        start = cache.n
        stage.begin(mtp=False)
        values, axes, epoch = stage.span(0, 0, count, mtp=False)
        span = Span(start, values, axes, stage, epoch, False, (0, 0))
        cache.begin(span, raw[start:start + count], coordinates[start:start + count])
        cache.pool(pooled, gamma, freq, 1e-6, (11, 11, 10), 1.0)
        history[start:start + count].copy_(raw[start:start + count])
        ops.qsa_pool(history, reference, torch.tensor([start], dtype=torch.int32, device=device),
                     gamma, freq, 1e-6, NS(ratio=4), count, rope=coordinates, length=32)
        cache.publish_commit(cache.prepare_commit(start + count, (0, 0)), start + count, (0, 0))
        assert torch.equal(pooled[:cache.n // 4], reference[:cache.n // 4])
        assert status.tolist() == [0, -2]
