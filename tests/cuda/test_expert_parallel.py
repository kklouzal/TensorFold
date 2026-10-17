"""Local expert-slot oracle and real one-rank NCCL; this does not emulate two GPUs."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.expert_cache import CachedExperts, HostExpertCache
from tensorfold.cuda.expert_parallel import AffineBank, Ownership, plan_owned


@pytest.mark.parametrize("prefill", [False, True])
def test_owned_banks_equal_native_full_bank_slots_under_eviction(prefill):
    gen = torch.Generator().manual_seed(603)
    E, D, I = 6, 256, 128
    def matrix(n, k):
        return (torch.randint(-(2**31), 2**31 - 1, (E + 1, n, k // 8), generator=gen,
                              dtype=torch.int32).cuda(),
                (torch.rand((E + 1, n, k // 32), generator=gen) * .03).bfloat16().cuda(),
                (torch.randn((E + 1, n, k // 32), generator=gen) * .02).bfloat16().cuda())
    full = grouped.make([matrix(I, D), matrix(I, D)], matrix(D, I), 32)
    picks = torch.tensor([[0, 4, 6], [1, 5, 6], [0, 2, 6], [3, 4, 6]], dtype=torch.int32, device="cuda")
    x = torch.randn((4, D), generator=gen).bfloat16().cuda()
    plan = grouped.Plan(4, 3, E + 1, "cuda", prefill=prefill)
    grouped.route(picks, plan)
    act = torch.empty((12, I), dtype=torch.bfloat16, device="cuda")
    expected = torch.empty((4, 3, D), dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
    grouped.gate_up(x, full, plan, act, 4)
    grouped.down(act, full, plan, expected.view(12, D), 4)
    actual = torch.zeros_like(expected)
    entry = full.bytes_per_expert()
    cache = HostExpertCache(entry, entry, "cuda")
    try:
        for rank in range(2):
            owner = Ownership((3, 3), rank)
            ids = list(owner.ids) + ([E] if rank == 0 else [])
            local = grouped.Experts(full.up[ids].cpu(), full.down[ids].cpu(), 32, I, D)
            bank = AffineBank(CachedExperts(local, cache, rank))
            part = torch.empty_like(actual)
            local_picks = owner.local(picks)
            bank.run_picks(x, local_picks, part, prefill=prefill)
            actual.copy_(torch.where((local_picks >= 0).unsqueeze(-1), part, actual))
        torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert cache.evictions > 0
    finally:
        cache.close()


def test_empty_owned_selection_never_reads_native_expert_storage():
    picks = torch.full((2, 65), -1, dtype=torch.int32, device="cuda")
    plan = plan_owned(picks, 4, prefill=False)
    assert plan.counts.cpu().tolist() == [0, 0]


def test_native_one_rank_nccl_broadcast_and_deferred_error_check():
    from tensorfold.cuda.comm import NCCL
    comm = NCCL()
    try:
        comm.open(0, 1, "127.0.0.1", 0)
        for dtype in (torch.bfloat16, torch.float32, torch.int32):
            value = torch.tensor([3, 5, 7], dtype=dtype, device="cuda")
            comm.broadcast(value)
            torch.cuda.synchronize()
            comm.check()
            assert value.cpu().tolist() == [3, 5, 7]
        with pytest.raises(ValueError):
            comm.send(value, 0)
    finally:
        comm.close()
