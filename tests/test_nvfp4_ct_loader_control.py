"""Actual loader preserves CT's stored input multiplier only for native NVFP4."""

import contextlib
import io
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_nvfp4_loader_phase_control import Value, fixture

from tensorfold.cuda import precision
from tensorfold.cuda.nvfp4 import format


@pytest.mark.parametrize("mode,capable", [(precision.CHECKPOINT, True), (precision.CHECKPOINT, False),
                                         (precision.FULL, True)])
def test_ct_original_input_global_scale_is_the_selected_native_quantizer_multiplier(mode, capable, monkeypatch):
    original = struct.unpack("f", struct.pack("I", 0x3FB7F13D))[0]
    raw_weight = 1.5
    load, events, state, modules = fixture()
    headers = modules["tensorfold.cuda.capacity"].headers(Path("opaque"))
    for name in list(headers):
        if not ("_proj.weight" in name or name == "lm_head.weight"):
            continue
        base = name[:-7]
        headers[name] = {"dtype": "U8", "shape": [8, 32]}
        for field, dtype, shape in (("weight_scale", "F8_E4M3", [8, 4]),
                                    ("weight_global_scale", "F32", [1]),
                                    ("input_global_scale", "F32", [1])):
            headers[base + "." + field] = {"dtype": dtype, "shape": shape}
    modules["tensorfold.cuda.capacity"].headers = lambda path: headers

    class Scalar(Value):
        def __init__(self, value):
            self.value = value

        def __float__(self):
            return self.value

    weights = modules["tensorfold.families.qwen3_5.cuda.weights"]
    original_reader = weights._Tensors

    class Reader(original_reader):
        def pop(self, name):
            value = super().pop(name)
            if name.endswith(".input_global_scale"):
                return Scalar(original)
            if name.endswith(".weight_global_scale"):
                return Scalar(raw_weight)
            return value

    weights._Tensors = Reader
    checked, constructed = [], []

    def admit(act, scale, **options):
        checked.append((act, scale, options.get("act_inverse")))

    def packed(*args, **options):
        constructed.append((args[2], options["act"], options["act_inverse"]))
        return SimpleNamespace()

    modules["tensorfold.cuda.nvfp4"].format = SimpleNamespace(
        LinearMetadata=format.LinearMetadata, calibration=format.calibration, native_calibration=admit)
    modules["tensorfold.cuda.nvfp4.linear"].Fp4Linear = SimpleNamespace(from_checkpoint=packed)
    config = {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {
        "nv4": {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "input_activations": {"num_bits": 4, "type": "float", "group_size": 16,
                                      "strategy": "tensor_group", "dynamic": "local"}}}}}
    load.__globals__.update(read_config=lambda path: config,
                            SUFFIXES=("weight", "weight_scale", "weight_global_scale", "input_global_scale"),
                            maths=lambda: ({"nvfp4": capable, "fp8": False}, "capability fixture"))
    with monkeypatch.context() as scoped:
        for name, module in modules.items():
            scoped.setitem(sys.modules, name, module)
        with precision.using(mode), contextlib.redirect_stdout(io.StringIO()):
            result = load(Path("opaque"))
    native = mode == precision.CHECKPOINT and capable
    expected = (1.0 / raw_weight, 1.0 / original, original) if native else (1.0 / raw_weight, None, None)
    assert constructed and constructed == [expected] * len(constructed)
    expected_checks = [(1.0 / original, 1.0 / raw_weight, original)] * len(constructed) if native else []
    assert checked == expected_checks
    assert result.precision == ("checkpoint" if native else "full")
    scalars = [name for name in events if name.endswith((".weight_global_scale", ".input_global_scale"))]
    assert state["device_reads"] == [(name, "cpu") for name in scalars]
