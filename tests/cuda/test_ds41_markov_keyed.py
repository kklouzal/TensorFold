"""DSpark keyed Markov proposals on the GPU: greedy rows unchanged, keyed rows the reference Gumbel maximum."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tensorfold.families.deepseek_v4.cuda import device_sampling as ds
from tensorfold.families.deepseek_v4.cuda.dspark_markov import ShardedMarkov

C1, C2, M1, M2 = (ds._signed(c) for c in (0x9E3779B97F4A7C15, 0xD1B54A32D192ED03, 0xBF58476D1CE4E5B9,
                                          0x94D049BB133111EB))


def shr(x, k):
    return (x >> k) & ((1 << (64 - k)) - 1)


def mix(x):
    x = x ^ shr(x, 30)
    x = x * M1
    x = x ^ shr(x, 27)
    x = x * M2
    return x ^ shr(x, 31)


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
    x = mix(seeds + C1)
    x = mix(x ^ (positions * C2))
    x = mix(x[:, None] ^ ids[None, :])
    u = shr(x, 12).double() * 2.0 ** -52 + 2.0 ** -53
    reference = (value.double() * inverse[:, None] - torch.log(-torch.log(u))).float()
    for r in range(b):
        if inverse[r] == 0:
            assert torch.equal(keyed[r], greedy[r])
        else:
            assert int(keyed[r, 1]) == first + int(reference[r].argmax())
