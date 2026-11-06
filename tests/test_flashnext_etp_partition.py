"""PP1 expert coverage and actual packed subset plans, not hardware qualification."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition


@pytest.mark.parametrize("etp", [8, 4, 2])
def test_all_sc117_expert_channels_have_one_stage_rank_owner(etp):
    coverage = np.zeros((512, 640), dtype=np.uint8)
    for rank in range(8):
        part = ExpertPartition(8, rank, 512, 640, etp)
        assert (part.groups, part.count, part.width) == (8 // etp, 512 // (8 // etp), 640 // etp)
        coverage[part.first:part.stop, part.column:part.column + part.width] += 1
        assert part.identity()["pp"] == 1
        with pytest.raises(FrozenInstanceError):
            part.etp = 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))


@pytest.mark.parametrize("world,rank,experts,width,etp", [
    (8, 0, 512, 640, 1), (8, 0, 512, 640, True), (4, 0, 512, 640, 4),
    (8, 8, 512, 640, 4), (8, 0, 3, 640, 2), (8, 0, 512, 641, 4),
])
def test_invalid_or_unsupported_preparation_refuses(world, rank, experts, width, etp):
    with pytest.raises(ValueError):
        ExpertPartition(world, rank, experts, width, etp)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from gguf_fixture import tiny_model

    return tiny_model(tmp_path_factory.mktemp("pp1-etp") / "target.gguf", experts=8)


@pytest.mark.torch
@pytest.mark.parametrize("etp", [8, 4, 2])
@pytest.mark.parametrize("rank", range(8))
def test_actual_target_packed_plan_keeps_nonexpert_and_shared_tp8(checkpoint, etp, rank):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan

    weights, planner = model_plan(checkpoint, (rank, 8), expert_tensor_parallel_size=etp)
    part = ExpertPartition(8, rank, 8, 640, etp)
    bank = weights.layers[0].moe.experts
    assert weights.cfg.moe_width == bank.width == part.width
    assert weights.cfg.shared_width == 80
    assert weights.cfg.experts == bank.count == 8  # Global router IDs, not local-bank count.
    assert bank.gate.n == bank.up.n == part.count * part.width
    assert bank.down.n == part.count * weights.cfg.hidden
    assert bank.down.k == part.width
    assert bank.down.column_offset == part.column % 32
    assert bank.gate.row_ranges[0][0] == part.first * 640 + part.column
    assert bank.down.row_ranges[0][0] == part.first * weights.cfg.hidden
    assert bank.shared[0].n == bank.shared[1].n == bank.shared[2].k == 80
    assert (weights.cfg.heads, weights.cfg.kv_heads, weights.cfg.nk, weights.cfg.nv) == (1, 1, 1, 2)
    assert len(planner.used) == len(planner.checkpoint.tensors)
    if etp == 8:
        assert bank.partition is None  # Existing all-bank provider path remains selected.
    else:
        assert bank.partition == part
        assert weights.meta["expert_partition"] == part.identity()


@pytest.mark.torch
@pytest.mark.parametrize("etp", [4, 2])
def test_owned_reference_routes_global_ids_and_clears_empty_groups(etp, monkeypatch):
    """Actual routing body, exact power-of-two CPU arithmetic; not CUDA BMM evidence."""
    import torch
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts

    original_bmm = torch.bmm

    def cpu_bmm(a, b, *, out_dtype=None):
        assert a.device.type == b.device.type == "cpu" and out_dtype == torch.float32
        return original_bmm(a.float(), b.float())

    monkeypatch.setattr(torch, "bmm", cpu_bmm)

    class Packed:
        def __init__(self, values):
            self.values = values.half()
            self.n, self.k = values.shape
            self.reads = []

        def decode(self, ids):
            assert ids.dtype == torch.int64
            assert not ids.numel() or (int(ids.min()) >= 0 and int(ids.max()) < self.n)
            self.reads.append(ids.clone())
            return self.values.index_select(0, ids)

    class Linear:
        def __init__(self, values):
            self.values = values.half()

        def matmul(self, x, *, f32=False):
            result = x.float() @ self.values.float().T
            return result if f32 else result.half()

    picks = torch.tensor([[0, 255, 256, 511, 512], [511, 256, 255, 0, 512]], dtype=torch.int64)
    x = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float16)
    for rank in range(8):
        part = ExpertPartition(8, rank, 512, 640, etp)
        base = (torch.arange(part.first + 1, part.stop + 1).float() / 1024)
        gate = Packed(base.repeat_interleave(part.width)[:, None].expand(-1, 2))
        up = Packed(torch.full((part.count * part.width, 2), .125))
        down = Packed(torch.full((part.count * 2, part.width), .015625))
        shared = (Linear(torch.full((80, 2), .25)), Linear(torch.full((80, 2), .125)),
                  Linear(torch.full((2, 80), .015625)))
        bank = GGUFExperts(gate, up, down, shared, count=512, width=part.width, dims=2, partition=part)
        assert bank.implementation == "reference" and not bank.capturable
        out = torch.full((2, 5, 2), float("nan"))
        assert bank.forward(x, picks, out) is out
        expected = torch.zeros_like(out)
        for row in range(2):
            for slot, global_id in enumerate(picks[row].tolist()):
                if global_id == 512:
                    activation = (torch.nn.functional.silu(torch.tensor(.25)) * .125).half().float()
                    expected[row, slot].fill_(float(activation * (80 / 64)))
                elif part.first <= global_id < part.stop:
                    activation = (torch.nn.functional.silu(torch.tensor((global_id + 1) / 1024)) * .125).half().float()
                    expected[row, slot].fill_(float(activation * (part.width / 64)))
        assert torch.equal(out, expected)
        # No local expert is read for a foreign-only request, even expert zero.
        for value in (gate, up, down):
            value.reads.clear()
        foreign = part.stop if part.stop < 512 else 0
        bank.forward(x, torch.full_like(picks, foreign), out)
        assert torch.count_nonzero(out) == 0
        assert not gate.reads and not up.reads and not down.reads
        invalid = picks.clone(); invalid[0, 0] = 513
        with pytest.raises(ValueError, match="global"):
            bank.forward(x, invalid, out)


@pytest.mark.torch
@pytest.mark.parametrize("etp", [4, 2])
@pytest.mark.parametrize("rank", range(8))
def test_actual_mtp_import_uses_identical_partition_and_packed_source_bytes(tmp_path, etp, rank):
    import torch
    from test_flashnext_gguf_mtp import fixture, _owned
    from tensorfold.families.qwen4_exp.cuda import gguf_mtp_weights as mtp
    from tensorfold.families.qwen4_exp.gguf import decode_rows
    from tensorfold.gguf_file import PackedTensor

    target, path, source = fixture(tmp_path, rank=rank, experts=8, etp=etp)
    part = ExpertPartition(8, rank, 8, 64, etp)
    plan = mtp.byte_plan(target, path, tp=(rank, 8), routing_rows=32)
    embedding, head = target.embed, target.head
    mtp.attach(target, path, tp=(rank, 8))
    assert target.embed is embedding and target.head is head
    assert _owned(target.mtp) == plan["resident_bytes"]
    bank = target.mtp.layer.moe.experts
    assert bank.partition == part and bank.width == 64 // etp
    for name, packed in (("gate", bank.gate), ("up", bank.up), ("down", bank.down)):
        descriptor = PackedTensor("control", (packed.n, packed.data.shape[1] // 34 * 32), 8, path, 0, packed.data.numel())
        got = decode_rows(descriptor, packed.data.numpy())[:, packed.column_offset:packed.column_offset + packed.k]
        expected = source[f"ffn_{name}_exps.weight"][part.first:part.stop]
        expected = (expected[:, :, part.column:part.column + part.width] if name == "down" else
                    expected[:, part.column:part.column + part.width, :])
        np.testing.assert_array_equal(got, expected.reshape(packed.n, packed.k))
    assert bank.shared[0].n == bank.shared[1].n == bank.shared[2].k == 8
    assert not torch.cuda.is_initialized()


@pytest.mark.torch
def test_state_recovery_identity_refuses_a_different_expert_reduction_degree():
    from types import SimpleNamespace
    from test_sm70_rotorquant_state import model
    from tensorfold.families.qwen4_exp.cuda.state import State

    states = {}
    for etp in (8, 4, 2):
        w = model(8)
        w.cfg.experts, w.cfg.moe_width, w.cfg.shared_width = 512, 640 // etp, 80
        w.meta.update(rank=0, full=SimpleNamespace(kv_heads=2, moe_width=640))
        if etp != 8:
            w.meta["expert_partition"] = ExpertPartition(8, 0, 512, 640, etp).identity()
        states[etp] = State(w, 16, 4, "int8", limit=16)
    assert len({s.kv_identity for s in states.values()}) == 3
    for dst, state in states.items():
        state.restore(state.snapshot())
        for src, other in states.items():
            if dst == src:
                continue
            with pytest.raises(ValueError):
                state.restore(other.snapshot())
            with pytest.raises(ValueError):
                state.copy_prefix(other, 0, 0)


@pytest.mark.torch
def test_equal_routed_width_does_not_make_prefill_providers_interchangeable(checkpoint, monkeypatch):
    from types import SimpleNamespace
    from tensorfold.families.qwen4_exp.cuda import gguf_prefill, gguf_weights

    legacy = gguf_prefill.choose(width=160, dims=256, rows=32, slots=2, budget_bytes=1 << 30)
    partitioned = gguf_prefill.choose(width=160, dims=256, rows=32, slots=2,
                                     budget_bytes=1 << 30, shared_width=80)
    understated = gguf_prefill.choose(width=160, dims=256, rows=32, slots=2,
                                      budget_bytes=1 << 30, shared_width=1)
    # Both providers have width160, but the old TP4 path owns shared width160
    # and a full output-copy temporary; TP8/ETP4 owns width80 and writes in place.
    gguf_weights.weight_byte_plan(checkpoint, (0, 4), prefill_plan=legacy)
    gguf_weights.weight_byte_plan(checkpoint, (0, 8), prefill_plan=partitioned,
                                   expert_tensor_parallel_size=4)
    for tp, etp, bad in [((0, 4), None, partitioned), ((0, 8), 4, legacy), ((0, 8), 4, understated)]:
        with pytest.raises(ValueError, match="shared width/provider"):
            gguf_weights.weight_byte_plan(checkpoint, tp, prefill_plan=bad,
                                           expert_tensor_parallel_size=etp)

    for part, good, bad, shared_width in [
        (None, legacy, partitioned, 160),
        (ExpertPartition(8, 0, 8, 640, 4), partitioned, legacy, 80),
    ]:
        local = 8 if part is None else part.count
        bank = gguf_weights.GGUFExperts(SimpleNamespace(n=local * 160), SimpleNamespace(n=local * 160),
                                        SimpleNamespace(n=local * 256), (SimpleNamespace(n=shared_width),) * 3,
                                        count=8, width=160, dims=256, partition=part)
        with pytest.raises(ValueError, match="shared width/provider"):
            bank.prepare_prefill(bad)
        assert bank._prefill_plan is None
        bank.prepare_prefill(good)
        assert bank._prefill_plan is good

    monkeypatch.setattr(gguf_weights, "Reader", lambda: pytest.fail("wrong provider acquired a payload reader"))
    with pytest.raises(ValueError, match="shared width/provider"):
        gguf_weights.load_gguf(checkpoint, device="cpu", tp=(0, 4), _prefill_plan=partitioned)
