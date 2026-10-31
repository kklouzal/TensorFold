"""FP32 router residency keeps the original FP16 coefficient boundary."""

from pathlib import Path

import numpy as np
import pytest

from gguf_fixture import scalar_reference, tiny_model

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.cuda.direct_read import Reader  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gguf_weights import (  # noqa: E402
    ArrayPlan, STAGING_BYTES, _array, model_plan, weight_byte_plan,
)
from tensorfold.families.qwen4_exp.gguf import PackedTensor  # noqa: E402


def test_router_plan_charges_fp32_storage_with_original_half_coefficients(tmp_path):
    path = tiny_model(tmp_path / "tiny.gguf")
    weights, _ = model_plan(path, (3, 8))
    reader = Reader()
    reader.direct = False
    try:
        for layer in weights.layers:
            for part in layer.moe.router.parts:
                assert part.dtype == "float32" and part.transform == "half_to_float"
                source = part.source
                raw = np.memmap(path, mode="r", dtype=np.uint8, offset=source.offset,
                                shape=(source.rows, source.row_bytes))
                expected = scalar_reference(raw, source.ggml_type, source.shape[-1]).reshape(source.shape)
                expected = expected.astype(np.float16).astype(np.float32)
                actual = _array(part, reader, "cpu")
                assert actual.dtype == torch.float32
                np.testing.assert_array_equal(actual.numpy().view(np.uint32), expected.view(np.uint32))
        plan = weight_byte_plan(path, (3, 8))
        assert plan["router_resident_bytes"] == len(weights.layers) * 3 * 256 * 4
        assert plan["router_host_staging_bytes"] == 2 * 256 * 14
        assert plan["host_staging_bytes"] >= max(3 * STAGING_BYTES, plan["router_host_staging_bytes"])
    finally:
        reader.close()


def test_router_transform_keeps_half_ties_subnormals_and_signed_zero_bits(tmp_path):
    values = np.array([[0.0, -0.0, 2**-25, -2**-25, 1 + 2**-11, 1 + 3 * 2**-11, 65504, -65504]], np.float32)
    path = tmp_path / "router.bin"
    path.write_bytes(values.tobytes())
    source = PackedTensor("router", values.shape, 0, Path(path), 0, values.nbytes)
    plan = ArrayPlan(source, transform="half_to_float")
    reader = Reader()
    reader.direct = False
    try:
        actual = _array(plan, reader, "cpu")
        expected = values.astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(actual.numpy().view(np.uint32), expected.view(np.uint32))
    finally:
        reader.close()


def test_router_transform_retains_the_original_half_overflow_refusal(tmp_path):
    path = tmp_path / "router.bin"
    values = np.array([[70000.0]], np.float32)
    path.write_bytes(values.tobytes())
    plan = ArrayPlan(PackedTensor("router", values.shape, 0, path, 0, values.nbytes), transform="half_to_float")
    reader = Reader()
    reader.direct = False
    try:
        with np.errstate(over="ignore"), pytest.raises(ValueError, match="overflows"):
            _array(plan, reader, "cpu")
    finally:
        reader.close()
