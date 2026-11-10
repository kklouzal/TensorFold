"""CPU contracts for the explicitly unselected, panelled E256 GU component."""

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[1] / "src/tensorfold/cuda/sm70/onecat_moe_gu.py"


@pytest.fixture
def component():
    import sys

    spec = importlib.util.spec_from_file_location("onecat_gu_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize("rows", [1, 8, 9, 24, 32])
@pytest.mark.parametrize("first", [0, 256])
def test_complete_panel_and_slot_accounting(component, rows, first):
    facts = component.layout(rows, first, "IQ3_S")
    represented = [row for start, count in facts["panels"] for row in range(start, start + count)]
    assert represented == list(range(rows))
    assert all(count <= 8 and count * 11 <= 256 for _, count in facts["panels"])
    assert facts["activation_shape"] == (rows * 11, 160)
    assert facts["activation_bytes"] == rows * 11 * 160 * 2
    assert facts["device_scratch_bytes"] == 0


@pytest.mark.parametrize("rows,first,fmt", [
    (True, 0, "IQ3_S"), (0, 0, "IQ3_S"), (33, 0, "IQ3_S"),
    (1, True, "IQ3_S"), (1, 128, "IQ3_S"), (1, 0, "Q2_0"),
    (1, 0, "NVFP4"), (1, 0, []),
])
def test_boundary_refusals_before_native(component, rows, first, fmt):
    with pytest.raises(ValueError):
        component.layout(rows, first, fmt)


@pytest.mark.parametrize("fmt,codes,scales,table", [
    ("IQ4_XS", 204800, 25600, 0), ("IQ3_S", 153600, 25600, 3072),
    ("IQ3_XXS", 153600, 12800, 3072), ("IQ2_S", 102400, 51200, 8192),
])
def test_lane_plane_storage_obligation(component, fmt, codes, scales, table):
    facts = component.layout(32, 256, fmt)
    assert facts["codes_shape"] == (256, codes)
    assert facts["scales_shape"] == (256, scales)
    assert facts["table_bytes"] == table
    assert facts["gate_up_plane_bytes"] == 2 * 256 * (codes + scales) + table


def test_import_has_no_torch_build_or_device_effects(component):
    assert "torch" not in component.__dict__
    assert component.METHODS == ("tensorfold-fp32-epilogue", "onecat-half-epilogue")
