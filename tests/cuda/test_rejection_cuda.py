"""Native rejection probability/packet boundaries; no model-quality or performance claim."""

import gc
import json
import weakref
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tensorfold.cuda import sampling, speculative
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import draw
from tensorfold.families.qwen3_5.cuda.weights import Plain
from tensorfold.families.qwen3_5_moe.cuda.dflash import DFlash
from tensorfold.families.qwen3_5_moe.dflash_metadata import DraftConfig
from tests.test_qwen36_dflash import config

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


@pytest.mark.parametrize("top_k,top_p,min_p", [(20, .95, 0), (0, .7, 0), (0, 1., .1)])
def test_effective_native_support_agrees_with_actual_target_draws(top_k, top_p, min_p):
    gen = torch.Generator(device="cuda").manual_seed(702)
    logits = torch.randn((3, 2048), device="cuda", generator=gen)
    policy = Sampling(6, temperature=.7, top_k=top_k, top_p=top_p, min_p=min_p)
    probabilities = sampling.effective_rows(logits, policy)
    for seed in (1, 6, 13, 72):
        policy = Sampling(seed, .7, top_k, top_p, min_p)
        assert sampling.sample_rows(logits, [17, 31, 2049], policy) == [
            draw(p, seed, pos, domain=0) for p, pos in zip(probabilities, [17, 31, 2049])]


def test_native_target_tie_repair_retains_exact_large_global_ids():
    logits = torch.zeros((1, 2048), device="cuda")
    ids = np.arange(2048, dtype=np.int64)[::-1].copy() + (1 << 54)
    row = sampling.effective_rows(logits, Sampling(4, top_k=2, top_p=1), id_map=ids)[0]
    assert row.ids.tolist() == [1 << 54, (1 << 54) + 1]
    assert row.probs.tolist() == [.5, .5]


@pytest.mark.parametrize("top_k,tied", [(20, False), (20, True), (0, False)])
def test_native_effective_target_peak_fits_declared_workspace(top_k, tied):
    logits = torch.zeros((4, 4096), device="cuda") if tied else torch.linspace(-5, 5, 4 * 4096, device="cuda").view(4, 4096)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    rows = sampling.effective_rows(logits, Sampling(7, top_k=top_k, top_p=.95))
    torch.cuda.synchronize()
    charge = sampling.effective_workspace(4, 4096, top_k=top_k)
    assert torch.cuda.max_memory_allocated() - before <= charge["device_bytes"]
    assert sum(r.ids.nbytes + r.probs.nbytes for r in rows) <= charge["host_bytes"]


def test_native_sparse_q_packet_peak_fits_declared_workspace():
    logits = torch.linspace(-2, 2, 6 * 2048, device="cuda").view(6, 2048)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    packet = speculative.Chains(logits, [None, None], 3, sampled=True)
    charge = speculative.chain_workspace(3, 2048, streams=2)
    q = packet.sample(0, 3, Sampling(7), 19)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before <= charge["device_bytes"]
    assert sum(r.ids.nbytes + r.probs.nbytes for r in q.q_rows) <= charge["host_bytes"]


@pytest.mark.parametrize("nondefault_consumer", [False, True])
def test_native_sampled_packet_keeps_foreign_owners_until_cross_stream_readback(nondefault_consumer):
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        source = torch.ones((3, 64), device="cuda")
        weight = torch.arange(64 * 512, dtype=torch.float32, device="cuda").view(64, 512) / 10000
        packet = speculative.Chains(source @ weight, [None], 3, owners=(source, weight), sampled=True)
        done = torch.cuda.Event()
        done.record(producer)
    source_ref, weight_ref = weakref.ref(source), weakref.ref(weight)
    del source, weight
    gc.collect()
    assert source_ref() is not None and weight_ref() is not None
    consumer = torch.cuda.Stream() if nondefault_consumer else torch.cuda.default_stream()
    with torch.cuda.stream(consumer):
        consumer.wait_event(done)
        q = packet.sample(0, 3, Sampling(4), 17)
    gc.collect()
    assert packet.owners is packet.words is None and source_ref() is weight_ref() is None
    assert len(q.tokens) == 3 and packet.sample(0, 3, Sampling(4), 17) == q


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_native_sampled_domain_failure_does_not_release_owners_or_publish(bad):
    logits = torch.ones((2, 1024), device="cuda")
    if bad == -float("inf"):
        logits[1] = bad
    else:
        logits[1, -1] = bad
    owner = object()
    packet = speculative.Chains(logits, [None], 2, owners=owner, sampled=True)
    with pytest.raises(ValueError):
        packet.sample(0, 2, Sampling(2), 3)
    assert packet.host is None and packet.owners is owner and packet.words is not None


@pytest.mark.parametrize("bits", [16, 8, 4])
def test_native_qwen36_probability_packet_is_identical_solo_and_batched(tmp_path, bits):
    raw, gen = config(), torch.Generator().manual_seed(903)
    cfg = DraftConfig.from_dict(raw)
    arrays = {name: ((torch.randn(shape, generator=gen) * .05).bfloat16() if len(shape) == 2
                     else torch.ones(shape, dtype=torch.bfloat16)) for name, shape in cfg.shapes().items()}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    save_file(arrays, tmp_path / "model.safetensors")

    class Target:
        config = NS(hidden=cfg.hidden, layers=cfg.target_layers, vocab=cfg.vocab)
        norm = torch.ones(cfg.hidden, dtype=torch.bfloat16, device="cuda")
        embed = Plain(torch.randn((cfg.vocab, cfg.hidden), generator=gen).bfloat16().cuda())
        head = Plain(torch.randn((cfg.vocab, cfg.hidden), generator=gen).bfloat16().cuda())

        def bind_draft_taps(self, layers):
            assert tuple(layers) == cfg.tap_layers

    draft = DFlash(tmp_path, Target(), context_limit=64, bits=bits)
    taps = torch.randn((7, len(cfg.tap_layers) * cfg.hidden), generator=gen).bfloat16().cuda()
    draft.add_taps(taps[:3])
    short = draft.snapshot()
    draft.add_taps(taps[3:])
    long = draft.snapshot()
    policy = Sampling(72, .7, 20, .95)
    solo = draft.finish_chain(draft.launch_chains([short], [2], 3)[0], 4, 3, policy)
    batched = draft.launch_chains([short, long], [2, 3], 3)
    joint = draft.finish_chain(batched[0], 4, 3, policy)
    assert solo == joint  # includes exact probability bytes, not only the argmax/proposed IDs
    assert draft.context_end == long[3]


@pytest.mark.parametrize("top_k", [0, 20])
@pytest.mark.parametrize("tied", ["all", "boundary"])
def test_native_sparse_q_ties_have_identical_support_bytes_in_solo_and_batch(top_k, tied):
    rows, width = 3, 2048
    logits = torch.zeros((rows, width), device="cuda", dtype=torch.bfloat16)
    if tied == "boundary":
        logits[:, :100] = 2
        logits[:, 100:1500] = 1
        logits[:, 1500:] = -1
    policy = Sampling(11, .7, top_k, 1.)
    batch = speculative.Chains(logits, [None] * rows, 1, sampled=True)
    packets = [batch.sample(r, 1, policy, 17 + r) for r in range(rows)]
    expected_values = logits[0].float().cpu().numpy()
    expected_ids = np.lexsort((np.arange(width), -expected_values))[:256]
    from tensorfold.engine.rejection import proposal_row
    expected = proposal_row(expected_values[expected_ids], expected_ids, policy)
    for r in (2, 0, 1):
        solo = speculative.Chains(logits[r:r + 1], [None], 1, sampled=True).sample(0, 1, policy, 17 + r)
        assert packets[r] == solo and packets[r].q_rows[0] == expected
    assert batch._source is batch.owners is batch.words is None


@pytest.mark.parametrize("offset", [1e30, -1e30])
@pytest.mark.parametrize("top_k", [0, 3])
def test_native_extreme_offset_target_support_noise_and_probabilities(offset, top_k):
    logits = torch.full((3, 8), offset, device="cuda")
    logits[1, :3] = -float("inf")
    centered = torch.where(torch.isfinite(logits), 0., -float("inf"))
    policy = Sampling(1, .7, top_k, .9, .1)
    actual = sampling.effective_rows(logits, policy)
    assert actual == sampling.effective_rows(centered, policy)
    for seed in (0, 1, 6, 13):
        policy = Sampling(seed, .7, top_k, .9, .1)
        expected = sampling.sample_rows(centered, [17, 20, 31], policy)
        assert sampling.sample_rows(logits, [17, 20, 31], policy) == expected
        assert [sampling.sample_rows(logits[r:r + 1], [p], policy)[0] for r, p in enumerate([17, 20, 31])] == expected
        assert expected == [draw(p, seed, pos, domain=0) for p, pos in zip(actual, [17, 20, 31])]
