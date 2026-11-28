"""DSpark keyed Markov proposals on the GPU: greedy rows unchanged, keyed rows the reference Gumbel maximum."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tensorfold.families.deepseek_v4.cuda import device_sampling as ds
from tensorfold.families.deepseek_v4.cuda.dspark_markov import ShardedMarkov


def test_keyed_markov_rows_follow_the_reference_and_keep_greedy_rows():
    g = torch.Generator(device="cuda").manual_seed(1)
    vocab, rank, first, b = 16160, 256, 3 * 16160, 6
    weight = (torch.randn((vocab, rank), generator=g, device="cuda") * 0.02).bfloat16()
    embed = (torch.randn((129280, rank), generator=g, device="cuda") * 0.02).bfloat16()
    markov = ShardedMarkov(weight, embed, first, None, 1)
    logits = torch.randn((b, vocab), generator=g, device="cuda") * 3
    prev = torch.tensor([5, 9, 11, 13, 17, 19], device="cuda")
    seeds = torch.tensor([ds._signed(s) for s in (1, 2, 3, 2 ** 63 + 5, 7, 8)], device="cuda")
    inverse = torch.tensor([0.0, 1.0, 1 / 0.7, 1.0, 0.0, 2.0], dtype=torch.float64, device="cuda")
    positions = torch.tensor([10, 20, 30, 40, 50, 60], device="cuda")
    greedy = markov.local_packet(logits, prev)
    keyed = markov.local_packet(logits, prev, (seeds, inverse, positions))
    value = logits + markov.bias(prev)
    ids = torch.arange(first, first + vocab, device="cuda")
    x = ds._mix(seeds + ds._signed(ds._C1))
    x = ds._mix(x ^ (positions * ds._signed(ds._C2)))
    x = ds._mix(x[:, None] ^ ids[None, :])
    u = ds._shr(x, 12).double() * 2.0 ** -52 + 2.0 ** -53
    reference = (value.double() * inverse[:, None] - torch.log(-torch.log(u))).float()
    for r in range(b):
        if inverse[r] == 0:
            assert torch.equal(keyed[r], greedy[r])
        else:
            assert int(keyed[r, 1]) == first + int(reference[r].argmax())


def test_device_sampler_matches_the_host_nucleus_sampler_on_cuda():
    from tensorfold.cuda.sampling import nucleus_rows
    from tensorfold.engine.exact_sampling import Sampling

    class One:
        def all_gather(self, send, recv):
            recv[0].copy_(send)

    g = torch.Generator(device="cuda").manual_seed(2)
    rows, vocab = 32, 16160
    sampler = ds.DeviceSampler(One(), rank=0, world=1, shard=vocab, device="cuda")
    for policy in (Sampling(3, temperature=1.0, top_k=0, top_p=0.95), Sampling(4, temperature=0.7, top_k=0),
                   Sampling(5, temperature=1.0, top_k=0, top_p=0.9, min_p=0.02)):
        for scale in (0.5, 3.0):
            logits = (torch.randn((rows, vocab), generator=g, device="cuda") * scale).half().float()
            positions = list(range(900, 900 + rows))
            words = [w for _ in range(rows) for w in ds.row_words(policy)]
            got = sampler.sample(logits, logits.max(-1).values, torch.tensor(positions, device="cuda"),
                                 sampler.words(words, rows)).tolist()
            want = [nucleus_rows(logits[r:r + 1], [positions[r]], policy)[0] for r in range(rows)]
            assert got == want
