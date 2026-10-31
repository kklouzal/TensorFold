"""Actual SM70 mixed-GGUF decode, matrix, expert and load ownership checks."""

import numpy as np
import pytest
import torch

from gguf_fixture import blocks, scalar_reference, tiny_model
from tensorfold.cuda.sm70.gguf_ops import decode
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, load_gguf, weight_byte_plan
from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint, decode_rows

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70-specific GGUF operators")


@pytest.mark.parametrize("kind,width", [(0, 33), (1, 33), (30, 33), (8, 96), (7, 96),
                                      (12, 512), (13, 512), (20, 96)])
def test_native_mixed_decode_matches_independent_scalar_oracle(kind, width):
    packed = blocks(kind, 3, width)
    expected = scalar_reference(packed, kind, width)
    actual = decode(torch.tensor(packed, device="cuda"), kind, width, dtype=torch.float32)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
    np.testing.assert_array_equal(decode(torch.tensor(packed, device="cuda"), kind, width).cpu().numpy(),
                                  expected.astype(np.float16))


def test_partial_q5_block_columns_and_repeated_row_gather():
    packed = blocks(7, 4, 96)
    expected = scalar_reference(packed, 7, 96)[[3, 0, 3], 16:96]
    actual = decode(torch.tensor(packed, device="cuda"), 7, 80, column_offset=16,
                    ids=torch.tensor([3, 0, 3], device="cuda"), dtype=torch.float32)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


def test_dense_rows_retain_serial_bits_at_all_row_groupings():
    torch.manual_seed(117)
    linear = GGUFDense(torch.randn((81, 256), device="cuda", dtype=torch.float16) * .02)
    x = torch.randn((39, 256), device="cuda", dtype=torch.float16)
    serial = torch.cat([linear(row[None]) for row in x])
    torch.testing.assert_close(linear(x), serial, rtol=0, atol=0)
    full = linear.matmul(x, f32=True)
    singles = torch.cat([linear.matmul(row[None], f32=True) for row in x])
    torch.testing.assert_close(full, singles, rtol=0, atol=0)
    oracle = x.cpu().double() @ linear.weight.cpu().double().T
    torch.testing.assert_close(full.cpu().double(), oracle, rtol=1e-4, atol=1e-6)


def test_complete_tiny_tp8_loader_matches_its_exact_allocation_plan(tmp_path):
    path = tiny_model(tmp_path / "tiny.gguf")
    weights = load_gguf(path, tp=(3, 8))
    assert weights.nbytes() == weight_byte_plan(path, (3, 8))["resident_bytes"]
    assert weights.layers[0].attn_hc.down is weights.layers[0].attn_hc.prefill_down
    assert weights.cfg.ple_eos == 13 and weights.cfg.eos == (14,)
    assert all(t.dtype == torch.float16 for t in weights.layers[0].gdn.proj.device_tensors())
    ids = torch.tensor([0, 14, 31], device="cuda")
    embedded = weights.embed[0].lookup(ids, copies=4)
    assert embedded.shape == (3, 1024) and embedded.dtype == torch.float16
    torch.testing.assert_close(embedded[:, :256], embedded[:, 256:512], rtol=0, atol=0)
    experts = weights.layers[0].moe.experts
    x = torch.randn((3, 256), device="cuda", dtype=torch.float16) * .1
    picks = torch.tensor([[0, 2], [1, 2], [0, 2]], dtype=torch.int32, device="cuda")
    result = experts.forward(x, picks)
    assert result.shape == (3, 2, 256) and result.dtype == torch.float32
    assert torch.isfinite(result).all()
    for row in range(3):
        torch.testing.assert_close(result[row:row + 1], experts.forward(x[row:row + 1], picks[row:row + 1]),
                                   rtol=0, atol=0)
    checkpoint = GGUFCheckpoint.open(path)

    def cpu_weight(name):
        descriptor = checkpoint.tensors[name]
        raw = np.memmap(path, mode="r", offset=descriptor.offset, dtype=np.uint8,
                        shape=(descriptor.rows, descriptor.row_bytes))
        return torch.from_numpy(decode_rows(descriptor, raw).reshape(descriptor.shape).astype(np.float16)).double()

    gate = cpu_weight("blk.0.ffn_gate_exps.weight")[:, 240:320]
    up = cpu_weight("blk.0.ffn_up_exps.weight")[:, 240:320]
    down = cpu_weight("blk.0.ffn_down_exps.weight")[:, :, 240:320]
    shared_g = cpu_weight("blk.0.ffn_gate_shexp.weight")[240:320]
    shared_u = cpu_weight("blk.0.ffn_up_shexp.weight")[240:320]
    shared_d = cpu_weight("blk.0.ffn_down_shexp.weight")[:, 240:320]
    expected = torch.empty_like(result, device="cpu")
    for row in range(3):
        xx = x[row].cpu().double()
        for slot in range(2):
            expert = int(picks[row, slot])
            g, u, d = (shared_g, shared_u, shared_d) if expert == 2 else (gate[expert], up[expert], down[expert])
            hidden = (torch.nn.functional.silu(g @ xx) * (u @ xx)).half().double()
            expected[row, slot] = d @ hidden
    torch.testing.assert_close(result.cpu(), expected, rtol=.003, atol=.003)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = experts.forward(x, picks)
    picks.copy_(torch.tensor([[1, 2], [0, 2], [1, 2]], dtype=torch.int32, device="cuda"))
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, experts.forward(x, picks), rtol=0, atol=0)
    weights.meta["ple_tables"].close(lambda: torch.cuda.synchronize())


def test_native_gguf_graph_capture_replay_and_prelaunch_bounds():
    packed = torch.tensor(blocks(8, 8, 32), device="cuda")
    ids = torch.tensor([0, 1], device="cuda")
    decode(packed, 8, 32, ids=ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = decode(packed, 8, 32, ids=ids)
    ids.copy_(torch.tensor([7, 3], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    expected = scalar_reference(packed.cpu().numpy(), 8, 32)[[7, 3]].astype(np.float16)
    np.testing.assert_array_equal(result.cpu().numpy(), expected)
    with pytest.raises(RuntimeError, match="complete source blocks"):
        decode(packed[:, :-1].contiguous(), 8, 32)
