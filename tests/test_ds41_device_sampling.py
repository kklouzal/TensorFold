"""DeepSeek-V4.1 device decode sampling against the host nucleus sampler (CPU; ranks simulated by threads)."""

import threading

import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v4.cuda import device_sampling as ds


class ThreadComm:
    """all_gather across ``world`` threads (each thread one rank), in call order."""

    def __init__(self, world):
        self.world = world
        self.slots = [None] * world
        self.barrier = threading.Barrier(world)

    def view(self, rank):
        comm = self

        class View:
            def all_gather(self, send, recv):
                comm.slots[rank] = send.clone()
                comm.barrier.wait()
                for r in range(comm.world):
                    recv[r].copy_(comm.slots[r])
                comm.barrier.wait()
        return View()


def device_tokens(logits, positions, samplings, world=8):
    rows, vocab = logits.shape
    shard = vocab // world
    top = logits.max(-1).values
    words = [w for s in samplings for w in ds.row_words(s)]
    comm = ThreadComm(world)
    out, errors = [None] * world, []

    def run(rank):
        try:
            sampler = ds.DeviceSampler(comm.view(rank), rank=rank, world=world, shard=shard, device="cpu")
            local = logits[:, rank * shard:(rank + 1) * shard].contiguous()
            out[rank] = sampler.sample(local, top, torch.tensor(positions), sampler.words(words, rows))
        except BaseException as error:   # noqa: BLE001
            errors.append(error)
            comm.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    for r in range(1, world):
        assert torch.equal(out[r], out[0])                  # every rank takes the same token
    return out[0].tolist()


@pytest.fixture
def host_tokens(monkeypatch):
    import tensorfold.cuda.sampling as sampling

    monkeypatch.setattr(sampling, "validate_rows", lambda logits, positions, policy: sampling.validate_policy(
        positions, policy))

    def draw(logits, positions, s):
        return [sampling.nucleus_rows(logits[r:r + 1], [positions[r]], s)[0] for r in range(logits.shape[0])]
    return draw


POLICIES = [
    Sampling(11, temperature=1.0, top_k=0, top_p=0.95),
    Sampling(12, temperature=1.0, top_k=0, top_p=1.0),
    Sampling(13, temperature=0.6, top_k=0, top_p=0.5),
    Sampling(14, temperature=1.3, top_k=0, top_p=0.9999),
    Sampling(15, temperature=1.0, top_k=0, top_p=0.95, min_p=0.05),
    Sampling(2 ** 63 + 17, temperature=0.8, top_k=0, top_p=0.7),
]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("flat", [False, True])
def test_device_draws_equal_host_draws(host_tokens, policy, flat):
    g = torch.Generator().manual_seed(int(policy.seed) % 1000 + flat)
    rows, vocab = 24, 8 * 512
    scale = 0.5 if flat else 4.0
    logits = (torch.randn((rows, vocab), generator=g) * scale).half().float()   # FP16 rounding: many ties
    logits[3, ::3] = float("-inf")
    logits[5, :] = logits[5, 0]                                                  # one all-tied row
    logits[7, 100:] = -30.0
    positions = [1000 + 7 * r for r in range(rows)]
    expected = host_tokens(logits, positions, policy)
    assert device_tokens(logits, positions, [policy] * rows) == expected


def test_greedy_rows_are_ignored_and_mixed_policies_follow_their_rows(host_tokens):
    g = torch.Generator().manual_seed(5)
    logits = (torch.randn((6, 8 * 256), generator=g) * 3).float()
    positions = list(range(50, 56))
    policies = [POLICIES[0], None, POLICIES[2], POLICIES[4], None, POLICIES[1]]
    got = device_tokens(logits, positions, policies)
    for r, policy in enumerate(policies):
        if policy is not None:
            assert got[r] == host_tokens(logits[r:r + 1], [positions[r]], policy)[0]


def test_eligibility_and_words():
    assert not ds.eligible(None)
    assert not ds.eligible(Sampling(1, temperature=0.0, top_k=0))
    assert not ds.eligible(Sampling(1, temperature=1.0, top_k=20))
    assert not ds.eligible(Sampling(1, temperature=float("nan"), top_k=0))
    assert not ds.eligible(Sampling(1, temperature=1.0, top_k=0, top_p=3 * 2.0 ** -110))
    assert ds.eligible(Sampling(1, temperature=1.0, top_k=0, top_p=0.95))
    assert ds.row_words(None) == [0] * ds.WORDS
    with pytest.raises(ValueError):
        ds.row_words(Sampling(1, temperature=1.0, top_k=5))


def test_exact_need_limbs():
    import random
    rng = random.Random(3)
    totals, nums, shifts, want = [], [], [], []
    for _ in range(2000):
        p = rng.choice([0.95, 0.5, 0.9999, 1e-9, 0.123456789, 1 - 2 ** -53, 2 ** -51])
        num, den = p.as_integer_ratio()
        total = rng.randrange(1, 1 << 57)
        totals.append(total)
        nums.append(num)
        shifts.append(den.bit_length() - 1)
        want.append(-(-num * total // den))
    got = ds._need(torch.tensor(totals), torch.tensor(nums), torch.tensor(shifts)).tolist()
    assert got == want
