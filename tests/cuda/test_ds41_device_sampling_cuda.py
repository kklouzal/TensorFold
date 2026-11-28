"""DeepSeek-V4.1 device decode sampling on the GPU against the host nucleus sampler (ranks simulated by threads)."""

import threading

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v4.cuda import device_sampling as ds
from tensorfold.families.deepseek_v4.cuda.kernels.triton_utils import jit, tl


class ThreadComm:
    """all_gather across ``world`` threads on one device (each thread one rank), in call order."""

    def __init__(self, world):
        self.world, self.slots, self.barrier = world, [None] * world, threading.Barrier(world)

    def view(self, rank):
        comm = self

        class View:
            def all_gather(self, send, recv):
                torch.cuda.synchronize()
                comm.slots[rank] = send.clone()
                torch.cuda.synchronize()
                comm.barrier.wait()
                for r in range(comm.world):
                    recv[r].copy_(comm.slots[r])
                torch.cuda.synchronize()
                comm.barrier.wait()
        return View()


def device_tokens(logits, positions, samplings, world):
    rows, vocab = logits.shape
    shard = vocab // world
    top = logits.max(-1).values
    words = [w for s in samplings for w in ds.row_words(s)]
    comm = ThreadComm(world)
    out, errors = [None] * world, []

    def run(rank):
        try:
            sampler = ds.DeviceSampler(comm.view(rank), rank=rank, world=world, shard=shard, device="cuda")
            local = logits[:, rank * shard:(rank + 1) * shard].contiguous()
            out[rank] = sampler.sample(local, top, torch.tensor(positions, device="cuda"),
                                       sampler.words(words, rows)).cpu()
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


def host_tokens(logits, positions, policy):
    from tensorfold.cuda.sampling import nucleus_rows

    return [nucleus_rows(logits[r:r + 1], [positions[r]], policy)[0] for r in range(logits.shape[0])]


POLICIES = [
    Sampling(11, temperature=1.0, top_k=0, top_p=0.95),
    Sampling(12, temperature=1.0, top_k=0, top_p=1.0),
    Sampling(13, temperature=0.6, top_k=0, top_p=0.5),
    Sampling(14, temperature=1.3, top_k=0, top_p=0.9999),
    Sampling(15, temperature=1.0, top_k=0, top_p=0.95, min_p=0.05),
    Sampling(2 ** 63 + 17, temperature=0.8, top_k=0, top_p=0.7),
]


@pytest.mark.parametrize("world", [1, 8])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("flat", [False, True])
def test_device_draws_equal_host_draws(policy, flat, world):
    g = torch.Generator(device="cuda").manual_seed(int(policy.seed) % 1000 + flat)
    rows, vocab = 24, 8 * 2020
    logits = (torch.randn((rows, vocab), generator=g, device="cuda") * (0.5 if flat else 4.0)).half().float()
    logits[3, ::3] = float("-inf")
    logits[5, :] = logits[5, 0]                                     # one all-tied row
    logits[7, 100:] = -30.0
    logits[9, 4000:4100] = logits[9].max()                          # a wide tie at the top across shards
    positions = [1000 + 7 * r for r in range(rows)]
    assert device_tokens(logits, positions, [policy] * rows, world) == host_tokens(logits, positions, policy)


def test_greedy_rows_are_ignored_and_mixed_policies_follow_their_rows():
    g = torch.Generator(device="cuda").manual_seed(5)
    logits = (torch.randn((6, 8 * 2020), generator=g, device="cuda") * 3).float()
    positions = list(range(50, 56))
    policies = [POLICIES[0], None, POLICIES[2], POLICIES[4], None, POLICIES[1]]
    got = device_tokens(logits, positions, policies, 8)
    for r, policy in enumerate(policies):
        if policy is not None:
            assert got[r] == host_tokens(logits[r:r + 1], [positions[r]], policy)[0]


@jit
def _probe_need(total, num, shift, out):
    i = tl.program_id(0)
    tl.store(out + i, ds._need(tl.load(total + i), tl.load(num + i), tl.load(shift + i)))


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
    t = lambda v: torch.tensor(v, dtype=torch.int64, device="cuda")
    out = torch.empty(len(want), dtype=torch.int64, device="cuda")
    _probe_need[(len(want),)](t(totals), t(nums), t(shifts), out)
    assert out.tolist() == want
