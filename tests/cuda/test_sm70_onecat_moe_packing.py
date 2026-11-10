"""Actual GGUF→plane→GU integration; no default or model-quality selection."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops, onecat_moe_gu as gu, onecat_moe_packing as packing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM70 packed GU component")


@pytest.mark.parametrize("kind,name", [(18, "IQ3_XXS"), (21, "IQ3_S"),
                                        (22, "IQ2_S"), (23, "IQ4_XS")])
def test_real_four_format_wire_to_native_onehot_matches_actual_original_gu(kind, name, record_property):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 packed GU component")
    prepared = gu.prepare(torch.device("cuda", torch.cuda.current_device()))
    with np.load(Path(__file__).parents[1] / "fixtures/gguf_mixed_ggml.npz", allow_pickle=False) as archive:
        raw = np.tile(archive[f"raw_{kind}"], (80, 2))[:, :10 * packing.BLOCK_BYTES[kind]].copy()
    raw.reshape(640, 10, -1)[:, :, :2] = np.array([2. ** -7], dtype="<f2").view(np.uint8)
    bank = np.broadcast_to(raw, (512, *raw.shape))
    spec = packing.requirements(kind)
    codes, scales = np.empty(spec["codes_shape"], np.uint8), np.empty(spec["scales_shape"], np.uint8)
    receipt = packing.pack_owned_into(bank, codes, scales, first=256, lane=2, ggml_type=kind)
    record_property("packing_source_sha256", receipt["source_payload_sha256"])
    gc, gs = torch.from_numpy(codes).cuda(), torch.from_numpy(scales).cuda()
    table = torch.from_numpy(packing.table(kind)).cuda()
    rows = 32
    columns = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255,
               256, 511, 512, 767, 1023, 1279, 1535, 1791, 2047, 2303, 2304, 2431,
               2559, 5, 17, 129]
    x = torch.zeros((rows, 2560), dtype=torch.float16, device="cuda")
    x[torch.arange(rows, device="cuda"), torch.tensor(columns, device="cuda")] = 1
    picks = torch.arange(256, 267, dtype=torch.int32, device="cuda").repeat(rows, 1)
    picks[:, 10] = 512
    native_output = torch.empty((rows * 11, 160), dtype=torch.float16, device="cuda")
    # Independent actual incumbent consumes original GGUF bytes, not planes.
    raw_gpu = torch.from_numpy(np.tile(raw[320:480], (10, 1))).cuda()
    original = SimpleNamespace(data=raw_gpu, ggml_type=kind, column_offset=0)
    down = SimpleNamespace(data=torch.zeros((10 * 2560, 160), dtype=torch.float16, device="cuda").view(torch.uint8),
                           ggml_type=1, column_offset=0)
    old_activation = torch.empty_like(native_output)
    old_shared = torch.zeros_like(x, dtype=torch.float32)
    old_local = torch.empty_like(picks, dtype=torch.int64)
    old_output = torch.empty((rows, 11, 2560), dtype=torch.float32, device="cuda")
    gguf_ops.experts_owned(x, picks, original, original, down, first=256, experts=512, width=160,
                          shared=old_shared, local=old_local, activation=old_activation, out=old_output)
    old = old_activation.reshape(rows, 11, 160)[:, :10].clone()
    # One-hot dots and power-of-two scales isolate representation/epilogue
    # from unknown multi-term MMA accumulation error. This is no global bound.
    prepared.gate_up(x, picks, gc, gs, gc, gs, table, native_output,
                     first=256, format_name=name, method=gu.METHODS[0])
    torch.testing.assert_close(native_output.reshape(rows, 11, 160)[:, :10], old, rtol=0, atol=0)
    prepared.gate_up(x, picks, gc, gs, gc, gs, table, native_output,
                     first=256, format_name=name, method=gu.METHODS[1])
    new = native_output.reshape(rows, 11, 160)[:, :10]
    assert torch.isfinite(new).all()
    # Differences are diagnostic data, never an invented acceptance tolerance.
    record_property("upstream_vs_incumbent_half_bit_differences",
                    int(torch.count_nonzero(new.view(torch.int16) != old.view(torch.int16)).item()))
