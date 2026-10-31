"""Pruned GGUF draft heads preserve target bytes and map every local row globally."""

import numpy as np
import pytest
import torch

from gguf_fixture import scalar_reference, tiny_model
from tensorfold.families.qwen4_exp.cuda.gguf_weights import load_gguf, weight_byte_plan
from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 GGUF draft head")


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_pruned_head_loads_global_rows_after_mtp_attach_and_keeps_target_storage(tmp_path, monkeypatch, rank):
    from tensorfold.families.qwen4_exp.cuda import sm70_mtp_weights

    path = tiny_model(tmp_path / "tiny.gguf")
    ids = np.array([0, 2, 4, 6, 9, 17, 18, 22, 29, 31], dtype=np.int64)
    ids.setflags(write=False)
    baseline = load_gguf(path, tp=(rank, 8))

    def attach(weights, directory, *, tp):
        # Stock attachment clears these fields before the GGUF loader publishes
        # the selected head. The stock tensor mapping has its separate tests.
        weights.draft_head, weights.draft_ids = None, None
        return weights

    monkeypatch.setattr(sm70_mtp_weights, "attach", attach)
    weights = None
    try:
        weights = load_gguf(path, tp=(rank, 8), mtp=True, mtp_model_dir=tmp_path,
                            draft_vocab="default", _draft_ids=ids)
        local = np.array_split(ids, 8)[rank]
        assert weights.draft_ids.dtype == torch.int64
        np.testing.assert_array_equal(weights.draft_ids.cpu().numpy(), local)
        assert weights.draft_head.n == local.size
        torch.testing.assert_close(weights.head.weight.view(torch.uint8), baseline.head.weight.view(torch.uint8),
                                   rtol=0, atol=0)
        assert weights.draft_head.weight.untyped_storage().data_ptr() != weights.head.weight.untyped_storage().data_ptr()
        budget = weight_byte_plan(path, (rank, 8), mtp=True, draft_ids=ids)
        assert weights.nbytes() == budget["resident_bytes"]
        source = GGUFCheckpoint.open(path).tensors["output.weight"]
        packed = np.memmap(path, dtype=np.uint8, mode="r", offset=source.offset,
                           shape=(source.rows, source.row_bytes))
        expected = scalar_reference(packed, source.ggml_type, source.shape[-1])[local].astype(np.float16)
        np.testing.assert_array_equal(weights.draft_head.weight.cpu().numpy().view(np.uint16), expected.view(np.uint16))
        torch.manual_seed(1921)
        x = torch.randn((3, 256), device="cuda", dtype=torch.float16)
        weights.draft_head.matmul(x, f32=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = weights.draft_head.matmul(x, f32=True)
        x.mul_(.5)
        graph.replay()
        torch.cuda.synchronize()
        oracle = x.cpu().double() @ torch.from_numpy(expected).double().T
        torch.testing.assert_close(result.cpu().double(), oracle, rtol=1e-4, atol=1e-6)
        serial = torch.cat([weights.draft_head.matmul(row[None], f32=True) for row in x])
        torch.testing.assert_close(result.view(torch.uint8), serial.view(torch.uint8), rtol=0, atol=0)
    finally:
        for item in (baseline, weights):
            if item is not None:
                item.meta["ple_tables"].close(lambda: torch.cuda.synchronize())
