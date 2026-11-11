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


@pytest.mark.parametrize('rows,first', [(1, 0), (9, 256), (24, 0), (32, 256)])
def test_raw_iq2_resources_have_no_plane_or_scratch(component, rows, first):
    facts = component.raw_iq2_layout(rows, first)
    assert facts['original_kind'] == 22 and facts['local_raw_shape'] == (256, 160, 820)
    assert facts['table_bytes'] == 8192 and facts['plane_bytes'] == facts['device_scratch_bytes'] == 0
    assert facts['activation_shape'] == (rows*11, 160)
    assert sum(count for _, count in facts['panels']) == rows


@pytest.mark.parametrize('rows,first', [(True, 0), (0, 0), (33, 0), (1, True), (1, 1)])
def test_raw_iq2_region_refused_before_native(component, rows, first):
    with pytest.raises(ValueError):
        component.raw_iq2_layout(rows, first)


def test_raw_iq2_wrapper_owns_no_buffers_and_preserves_output_version(component):
    from types import SimpleNamespace as NS
    import torch
    calls = []
    module = NS(gate_up_raw_iq2=lambda *args: calls.append(args))
    prepared = component.RawIQ2Prepared(module, torch.device('cpu'))
    x, picks = torch.zeros((1, 2560), dtype=torch.float16), torch.zeros((1, 11), dtype=torch.int32)
    raw, table = torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8)
    out = torch.zeros((11, 160), dtype=torch.float16)
    version = out._version
    assert prepared.gate_up_raw_iq2(x, picks, raw, raw, table, out, first=256,
        method=component.METHODS[0], gate_column_offset=256) is out
    assert out._version == version+1
    assert calls[0][-4:] == (256, False, 256, 0)
    assert set(vars(prepared)) == {'module', 'device'}


def test_raw_iq2_inference_guard_has_no_native_effect(component):
    from types import SimpleNamespace as NS
    import torch
    prepared = component.RawIQ2Prepared(NS(gate_up_raw_iq2=lambda *args: pytest.fail('native mutation')), torch.device('cpu'))
    x, picks = torch.zeros((1, 2560), dtype=torch.float16), torch.zeros((1, 11), dtype=torch.int32)
    raw = torch.zeros(1, dtype=torch.uint8)
    with torch.inference_mode():
        out = torch.zeros((11, 160), dtype=torch.float16)
    with pytest.raises(RuntimeError, match='inference output'):
        prepared.gate_up_raw_iq2(x, picks, raw, raw, raw, out, first=0, method=component.METHODS[0])


def test_raw_iq2_cold_residency_runs_in_requested_device(component, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace as NS
    import torch
    calls = []
    module = NS(prepare_raw_iq2=lambda: calls.append('raw-attributes'))
    def old(device):
        calls.append('packed-preparation')
        return component.Prepared(module, torch.device(device))
    @contextmanager
    def device(value):
        calls.append(('enter', str(value)))
        yield
        calls.append(('exit', str(value)))
    monkeypatch.setattr(component, 'prepare', old)
    monkeypatch.setattr(torch.cuda, 'device', device)
    actual = component.prepare_raw_iq2('cuda:2')
    assert actual.module is module and actual.device == torch.device('cuda:2')
    assert calls == ['packed-preparation', ('enter', 'cuda:2'), 'raw-attributes', ('exit', 'cuda:2')]
