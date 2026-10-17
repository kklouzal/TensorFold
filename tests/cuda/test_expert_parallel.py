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
    E, D, width = 6, 256, 128
    def matrix(n, k):
        return (torch.randint(-(2**31), 2**31 - 1, (E + 1, n, k // 8), generator=gen,
                              dtype=torch.int32).cuda(),
                (torch.rand((E + 1, n, k // 32), generator=gen) * .03).bfloat16().cuda(),
                (torch.randn((E + 1, n, k // 32), generator=gen) * .02).bfloat16().cuda())
    full = grouped.make([matrix(width, D), matrix(width, D)], matrix(D, width), 32)
    picks = torch.tensor([[0, 4, 6], [1, 5, 6], [0, 2, 6], [3, 4, 6]], dtype=torch.int32, device="cuda")
    x = torch.randn((4, D), generator=gen).bfloat16().cuda()
    plan = grouped.Plan(4, 3, E + 1, "cuda", prefill=prefill)
    grouped.route(picks, plan)
    act = torch.empty((12, width), dtype=torch.bfloat16, device="cuda")
    expected = torch.empty((4, 3, D), dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
    grouped.gate_up(x, full, plan, act, 4)
    grouped.down(act, full, plan, expected.view(12, D), 4)
    actual = torch.zeros_like(expected)
    entry = full.bytes_per_expert()
    cache = HostExpertCache(entry, entry, "cuda")
    borrowed = []
    try:
        for rank in range(2):
            owner = Ownership((3, 3), rank)
            ids = list(owner.ids) + ([E] if rank == 0 else [])
            local = grouped.Experts(full.up[ids].cpu(), full.down[ids].cpu(), 32, width, D)
            bank = AffineBank(CachedExperts(local, cache, rank))
            part = torch.empty_like(actual)
            local_picks = owner.local(picks)
            borrowed.append(bank.run_picks(x, local_picks, part, prefill=prefill))
            actual.copy_(torch.where((local_picks >= 0).unsqueeze(-1), part, actual))
        torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert cache.evictions > 0
        borrowed.clear()
    finally:
        cache.close()


def test_empty_owned_selection_never_reads_native_expert_storage():
    picks = torch.full((2, 65), -1, dtype=torch.int32, device="cuda")
    plan = plan_owned(picks, 4, prefill=False)
    assert plan.counts.cpu().tolist() == [0, 0]


def test_native_one_rank_nccl_broadcast_and_deferred_error_check():
    from tensorfold.cuda.comm import NCCL, agree_eos
    comm = NCCL()
    try:
        comm.open(0, 1, "127.0.0.1", 0)
        for dtype in (torch.bfloat16, torch.float32, torch.int32):
            value = torch.tensor([3, 5, 7], dtype=dtype, device="cuda")
            comm.broadcast(value)
            torch.cuda.synchronize()
            comm.check()
            assert value.cpu().tolist() == [3, 5, 7]
        agree_eos(comm.all_gather, (0, 2**40 + 3, 0), world=1, device="cuda")
        with pytest.raises(ValueError):
            comm.send(value, 0)
    finally:
        comm.close()


def test_native_world1_session_loads_unsharded_banks_runs_and_retires(tmp_path):
    """Exercise actual Session loader, NCCL calls, all prompt packets and owned close on one real GPU."""
    from test_flashnext_tp import _checkpoint

    from tensorfold.families.qwen4_exp.cuda.expert_parallel import Session
    from tensorfold.families.qwen4_exp.cuda.reader import _Reader

    _checkpoint(tmp_path)
    (tmp_path / "model.safetensors.index.json").unlink()
    session = Session(tmp_path, ranks=1, shards=None, rank=0, master="127.0.0.1", port=0,
                      budget=.001, mtp=False)
    reader = _Reader(tmp_path, "cuda")
    try:
        banks = [session.load_bank(None, f"model.layers.{i}.mlp", "", session.cfg, "mlx") for i in range(2)]
        session.finish_loading()
        base = "model.layers.0.mlp"
        def matrix(projection):
            return tuple(torch.cat([reader.get(f"{base}.switch_mlp.{projection}.{field}"),
                                    reader.get(f"{base}.shared_expert.{projection}.{field}")[None]])
                         for field in ("weight", "scales", "biases"))
        reference = grouped.make([matrix("gate_proj"), matrix("up_proj")], matrix("down_proj"), 32)
        rows, slots, dims = 257, session.slots, session.dims
        gen = torch.Generator().manual_seed(721)
        x = torch.randn((rows, dims), generator=gen).bfloat16().cuda()
        picks = torch.tensor([list(range(slots - 1)) + [session.cfg.experts]] * rows,
                             dtype=torch.int32, device="cuda")
        for prefill in (False, True):
            expected = torch.empty((rows, slots, dims), dtype=torch.bfloat16 if prefill else torch.float32,
                                   device="cuda")
            act = torch.empty((rows * slots, session.cfg.moe_width), dtype=torch.bfloat16, device="cuda")
            plan = grouped.Plan(rows, slots, session.cfg.experts + 1, "cuda", prefill=prefill)
            grouped.route(picks, plan)
            grouped.gate_up(x, reference, plan, act, rows)
            grouped.down(act, reference, plan, expected.view(rows * slots, dims), rows)
            actual = torch.empty_like(expected)
            session.run(banks[0], x, picks, actual, prefill=prefill)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert session.sequence == 6  # 128 +128 +1 packets per arithmetic mode
    finally:
        reader.close()
        session.close()
    assert session.retired and session.store is None and not session.source_files


def test_native_world1_model_preserves_serial_mtp_and_concurrent_replies(tmp_path):
    """Actual model/Session integration on one GPU; multi-GPU hardware qualification remains separate."""
    from concurrent.futures import ThreadPoolExecutor

    from test_flashnext_tp import _checkpoint

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.expert_parallel import Session

    _checkpoint(tmp_path)
    native = FlashNextEngine(tmp_path, depth=2, confidence=0., max_len=128, prefetch=False, graphs=False,
                             draft_vocab=None)
    session = Session(tmp_path, ranks=1, shards=None, rank=0, master="127.0.0.1", port=0,
                      budget=.001, mtp=True)
    engine = None
    sampling = Sampling(seed=977, top_k=20, top_p=.95)
    def generate(model, prompt, draft):
        out = []
        model.generate(prompt, 8, sampling, lambda ids: out.extend(ids), draft=draft, stop_eos=False)
        return out
    prompts = [[5, 17, 99, 250, 7, 64, 30, 11], [12, 45, 250, 27, 19, 300, 34]]
    try:
        expected = [generate(native, p, False) for p in prompts]
        engine = FlashNextEngine(tmp_path, depth=2, confidence=0., max_len=128, prefetch=False, graphs=False,
                                  streams=2, draft_vocab=None, vram_experts=.001, expert_session=session)
        for prompt, reply in zip(prompts, expected):
            assert generate(engine, prompt, False) == reply
            assert generate(engine, prompt, True) == reply
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(generate, engine, prompt, True) for prompt in prompts]
            assert [future.result() for future in futures] == expected
    finally:
        if engine is not None:
            engine.close()
        session.close()
        native.close()


def test_native_world1_nvfp4_close_releases_banks_workspace_and_wire_storage(tmp_path):
    """Keep the closed Session alive while its actual native model storage retires."""
    import gc
    import weakref

    from nvfp4_tiny import write

    from tensorfold.families.qwen4_exp.cuda.expert_parallel import Session

    model = write(tmp_path, hidden=512, shared_width=128)
    session = Session(model, ranks=1, shards=None, rank=0, master="127.0.0.1", port=0,
                      budget=.001, mtp=True)
    def payload_ownership():
        groups = {"wire": [getattr(session, name) for name in ("x", "pick", "output", "peer", "low")],
                  "pool": [session.cache._pool], "workspace": [], "shared": []}
        for plan, act in session.nv_workspace.values():
            groups["workspace"].extend((plan.members, plan.items, plan.counts, plan.rank, plan.hist, act))
        for bank in session.banks.values():
            for projection in (bank.shared.gu, bank.shared.down):
                groups["shared"].extend(value for value in vars(projection).values()
                                        if isinstance(value, torch.Tensor))
        storage = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                   for values in groups.values() for tensor in values}
        return {name: [weakref.ref(tensor) for tensor in values] for name, values in groups.items()}, sum(storage.values())
    try:
        banks = [session.load_bank(None, f"model.layers.{i}.mlp", "", session.cfg, "modelopt")
                 for i in range(session.cfg.layers)]
        banks.append(session.load_bank(None, "mtp.layers.0.mlp", "", session.cfg, "modelopt"))
        session.finish_loading()
        x = torch.ones((1, session.dims), dtype=torch.bfloat16, device="cuda")
        picks = torch.tensor([list(range(session.slots - 1)) + [session.cfg.experts]],
                             dtype=torch.int32, device="cuda")
        for prefill, distributed in ((False, banks[0]), (True, banks[-1])):
            out = torch.empty((1, session.slots, session.dims),
                              dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
            session.run(distributed, x, picks, out, prefill=prefill)
        references, owned_bytes = payload_ownership()
        assert all(references.values()) and owned_bytes > 0 and len(session.nv_workspace) == 1
        # Only the session owns these payloads now; weak tensor references plus
        # the allocator balance also detect storage retained through another view.
        del banks, distributed, x, picks, out
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        session.close()
        gc.collect()
        torch.cuda.synchronize()
        assert session.retired and session.store is None and not session.banks and not session.nv_workspace
        assert all(not hasattr(session, name) for name in ("x", "pick", "output", "peer", "low"))
        assert all(reference() is None for values in references.values() for reference in values)
        assert before - torch.cuda.memory_allocated() >= owned_bytes
    finally:
        session.close(abort=True)
