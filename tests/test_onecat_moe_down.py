"""Down ABI/representation and finite quantizer controls; no target claim."""

from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import onecat_moe_down as down, onecat_moe_down_packing as packing
from tests.onecat_down_oracle import q8_block, quantization_bound
from tests.sm70_native_rn_oracle import bits_to_fraction, round_fraction


@pytest.mark.parametrize("rows", range(1, 33))
@pytest.mark.parametrize("method", down.METHODS)
def test_bounded_resources_and_complete_panels(rows, method):
    facts = down.layout(rows, 256, ggml_type=20, column_offset=480, row_bytes=360, method=method)
    assert sum(count for _, count in facts["panels"]) == rows and max(c for _, c in facts["panels"]) <= 8
    assert facts["q8_bytes"] == rows * 11 * 5 * 36
    assert facts["route_bytes"] == rows * 11 * 2560 * 2
    assert facts["route_written_bytes"] == (0 if method == down.METHODS[0] else facts["route_bytes"])
    assert facts["output_dtype"] == ("float16" if method == down.METHODS[0] else "float32")


@pytest.mark.parametrize("fault", ["rows", "bool", "first", "Q2_0", "offset", "extent", "method"])
def test_unsupported_inputs_refused(fault):
    args = dict(rows=1, first=0, ggml_type=20, column_offset=0, row_bytes=90, method=down.METHODS[0])
    if fault == "rows":
        args["rows"] = 33
    elif fault == "bool":
        args["rows"] = True
    elif fault == "first":
        args["first"] = 1
    elif fault == "Q2_0":
        args["ggml_type"] = 42
    elif fault == "offset":
        args["column_offset"] = 1
    elif fault == "extent":
        args["column_offset"] = 160
    else:
        args["method"] = "auto"
    with pytest.raises(ValueError):
        down.layout(**args)


def test_lossless_original_block_crop_and_source_retention():
    source = np.arange(360, dtype=np.uint16).astype(np.uint8)[None, :].repeat(2560, 0)
    raw = np.broadcast_to(source, (512, 2560, 360))
    output = np.empty(packing.requirements(ggml_type=20)["output_shape"], np.uint8)
    receipt = packing.pack_owned_into(raw, output, first=256, lane=3, ggml_type=20)
    assert receipt["completed_experts"] == 256 and receipt["coefficient_transform"].startswith("NONE")
    assert np.array_equal(output[:2560], source[:, 270:360])
    assert np.array_equal(output[-2560:], source[:, 270:360])
    assert not raw.flags.writeable


def test_distinct_global_experts_lanes_and_single_stage_lifetime():
    raw = np.empty((512, 2560, 360), np.uint8)
    for expert in range(512):
        pattern = np.arange(360, dtype=np.uint16).astype(np.uint8)
        pattern.reshape(20, 18)[:, 0] = expert & 255
        pattern.reshape(20, 18)[:, 1] = expert >> 8
        raw[expert] = pattern
    output = np.empty(packing.requirements(ggml_type=20)["output_shape"], np.uint8)
    allocate = np.array
    previous = None
    count = 0

    def one_stage(*args, **kwargs):
        nonlocal previous, count
        assert previous is None or previous() is None
        stage = allocate(*args, **kwargs)
        previous = weakref.ref(stage)
        count += 1
        return stage

    for first in (0, 256):
        for lane in (0, 3):
            with patch.object(packing.np, "array", side_effect=one_stage):
                packing.pack_owned_into(raw, output, first=first, lane=lane, ggml_type=20)
            assert previous() is None
            for local in (0, 1, 127, 254, 255):
                expected = raw[first + local, 0, lane * 90 : (lane + 1) * 90]
                assert np.array_equal(output[local * 2560], expected)
                assert np.array_equal(output[(local + 1) * 2560 - 1], expected)
    assert count == 4 * 256


@pytest.mark.parametrize(
    "values", [[0] * 32, [1, 254] + [0] * 30, [Fraction(1, 2**24)] + [0] * 31, [65504] * 32, [-1, 254] + [0] * 30]
)
def test_exact_quantizer_wire_and_analytic_activation_error(values):
    bits = [round_fraction(v, 16) for v in values]
    wire = q8_block(bits)
    dh = bits_to_fraction(int.from_bytes(wire[:2], "little"), 16)
    bound = quantization_bound(bits, wire)
    for value, q in zip(values, wire[4:]):
        q = q if q < 128 else q - 256
        assert abs(Fraction(value) - dh * q) <= bound
    if values[:2] == [1, 254]:
        assert wire[4] == 1  # roundf(.5) away; RN-even would0.
    if values[0] == Fraction(1, 2**24):
        assert dh == 0 and wire[4] == 127
    if values[0] == 65504:
        assert int.from_bytes(wire[2:4], "little") == 0x7C00  # unusedsumHalf overflow is defined.


@pytest.mark.parametrize("method", down.METHODS)
def test_actual_wrapper_native_arguments_and_each_written_version(method):
    calls = []
    module = SimpleNamespace(down=lambda *args: calls.append(args))
    prepared = down.Prepared(module, torch.device("cpu"))
    args = [
        torch.zeros((11, 160)).half(),
        torch.zeros((1, 11), dtype=torch.int64),
        torch.ones((1, 11)),
        torch.zeros((256 * 2560, 90), dtype=torch.uint8),
        torch.empty((11, 5, 36), dtype=torch.uint8),
        torch.empty((11, 2560)).half(),
        torch.empty(1, dtype=torch.int32),
        torch.empty((1, 2560), dtype=torch.float16 if method == down.METHODS[0] else torch.float32),
    ]
    versions = [t._version for t in args[4:]]
    prepared.down(*args, first=0, ggml_type=20, column_offset=0, method=method)
    assert len(calls) == 1 and calls[0][-4:] == (20, 0, 0, method == down.METHODS[0])
    expected = [v + (0 if method == down.METHODS[0] and i == 1 else 1) for i, v in enumerate(versions)]
    assert [t._version for t in args[4:]] == expected
    primary = RuntimeError("refused")
    with patch.object(module, "down", side_effect=primary):
        with pytest.raises(RuntimeError) as found:
            prepared.down(*args, first=0, ggml_type=20, column_offset=0, method=method)
    assert found.value is primary and [t._version for t in args[4:]] == expected
