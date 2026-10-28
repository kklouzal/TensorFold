"""Exact independent projections against original BMM and pinned GGML wire bits."""
from pathlib import Path

import numpy as np
import pytest
import torch

from gguf_fixture import blocks
from test_sm70_gguf_exact import bits
from tensorfold.families.qwen4_exp.gguf import FORMATS
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFPacked, GGUFExperts, GGUFDense

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="actual SM70 independent exact projections")


def wire_rows(kind, rows, columns):
    # The pinned GGML fixture supplies independently produced scalar FP32 bits.
    # Repeat complete original blocks; never regenerate a coefficient oracle
    # with the production decoder or the candidate's packed coefficient reader.
    with np.load(Path(__file__).parents[1] / "fixtures/gguf_mixed_ggml.npz", allow_pickle=False) as fixture:
        raw = fixture[f"raw_{kind}"]
        values = fixture[f"bits_{kind}"].view(np.float32)
    block, stored = FORMATS[kind][1:]
    bank = raw.reshape(8, -1, stored)
    copies = -(-columns // values.shape[1])
    raw = np.tile(bank, (1, copies, 1))[:, :columns // block].reshape(8, -1)
    values = np.tile(values, (1, copies))[:, :columns]
    repeats = -(-rows // 8)
    return np.tile(raw, (repeats, 1))[:rows].copy(), np.tile(values, (repeats, 1))[:rows].copy()


@pytest.fixture(scope="module", params=[(18, 18, 20), (21, 21, 18), (22, 22, 23),
                                         (23, 23, 14), (14, 14, 42), (42, 42, 20), (18, 21, 20)])
def mixed_bank(request):
    gate_kind, up_kind, down_kind = request.param
    width, dims, count, offset = 80, 2560, 2, 16
    packed, coefficients = [], []
    for kind, rows, columns, crop in [(gate_kind, count * width, dims, 0),
                                      (up_kind, count * width, dims, 0),
                                      (down_kind, count * dims, width, offset)]:
        block = FORMATS[kind][1]
        physical = -(-(columns + crop) // block) * block
        if kind == 20:
            raw = blocks(kind, rows, physical)
            values = None  # The retained independent scalar down control owns this existing type.
        else:
            raw, values = wire_rows(kind, rows, physical)
        packed.append(GGUFPacked(torch.tensor(raw, device="cuda"), kind, rows, columns, crop))
        coefficients.append(None if values is None else values[:, crop:crop + columns].astype(np.float16))
    torch.manual_seed(7811)
    shared = [GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * .02)
              for shape in [(width, dims), (width, dims), (dims, width)]]
    return GGUFExperts(*packed, shared, count=count, width=width, dims=dims), coefficients


@pytest.mark.parametrize("rows", [1, 2, 3, 8, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_mixed_wire_formats_match_original_bank_and_original_kernel(mixed_bank, rows, dtype, index_dtype):
    bank, _ = mixed_bank
    torch.manual_seed(8191 + rows)
    x = (torch.randn(rows, 2560, device="cuda") * .1).half()
    picks = (torch.arange(rows * 3, device="cuda", dtype=index_dtype).reshape(rows, 3) % 3).contiguous()
    expected = bank.forward_reference(x, picks).to(dtype)
    original, candidate = torch.empty_like(expected), torch.empty_like(expected)
    bank.forward_exact(x, picks, original)
    bank.forward_exact(x, picks, candidate, independent_gate_up=True)
    bits(candidate, original)
    bits(candidate, expected)


def test_gate_up_original_bmm_has_independent_fp16_coefficient_and_lane_chain_oracle(mixed_bank):
    bank, coefficients = mixed_bank
    torch.manual_seed(7191)
    x = (torch.randn(1, 2560, device="cuda") * .1).half()
    padded = torch.zeros((16, 1, 2560), dtype=torch.float16, device="cuda")
    padded[0].copy_(x)
    for packed, trusted in zip((bank.gate, bank.up), coefficients[:2]):
        decoded = torch.zeros((16, bank.width, 2560), dtype=torch.float16, device="cuda")
        decoded[0].copy_(packed.decode(torch.arange(bank.width, device="cuda")))
        bits(decoded[0], torch.tensor(trusted[:bank.width], device="cuda"))
        actual = torch.bmm(padded, decoded.transpose(1, 2), out_dtype=torch.float32)[0, 0].cpu().numpy()
        weight = trusted[:bank.width].astype(np.float32).reshape(bank.width, 80, 32)
        values = x.cpu().numpy().astype(np.float32).reshape(80, 32)
        partials = np.zeros((bank.width, 32), np.float32)
        # Half*half products are exact FP32, so each add matches a single FMA.
        for step in range(80):
            partials = partials + weight[:, step] * values[step]
        for offset in (16, 8, 4, 2, 1):
            previous = partials.copy()
            partials[:, :32 - offset] = previous[:, :32 - offset] + previous[:, offset:]
        np.testing.assert_array_equal(actual.view(np.uint32), partials[:, 0].view(np.uint32))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mixed_wire_capture_preserves_invalid_nan_shared_zero_and_live_routes(mixed_bank, index_dtype, dtype):
    from tensorfold.cuda.sm70 import gguf_ops

    bank, _ = mixed_bank
    x = (torch.randn(3, 2560, device="cuda") * .1).half()
    picks = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]], device="cuda", dtype=index_dtype)
    candidate = torch.empty((3, 3, 2560), device="cuda", dtype=dtype)
    original = torch.empty_like(candidate)
    def run(output, independent):
        gguf_ops.experts_exact(x, picks, bank.gate, bank.up, bank.down, bank.shared, output,
                               independent_gate_up=independent)
    run(candidate, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(candidate, True)
    picks.copy_(picks.roll(1, 1))
    picks[0] = torch.tensor([-1, 2, 2**30 if index_dtype == torch.int32 else 2**40],
                            device="cuda", dtype=index_dtype)
    graph.replay()
    run(original, False)
    bits(candidate, original)
    assert bool(torch.isnan(candidate[0, 0]).all()) and bool(torch.isnan(candidate[0, 2]).all())
    shared = picks == 2
    bits(candidate[shared], torch.zeros_like(candidate[shared]))
    assert bool(torch.isfinite(candidate[1:]).all())
