"""CPU checks for the maintained CUDA fixture's actual speculation domain."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry


@pytest.fixture
def geometry():
    # Load only the pure fixture planner: the CUDA module's collection guard
    # intentionally requires a device. Runtime MethodGeometry stays installed.
    path = Path(__file__).parent / "cuda/test_flashnext_sm70_bound_graphs.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_six_lane_geometry")
    namespace = {"MethodGeometry": MethodGeometry}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_six_lane_geometry"]


def test_no_drafter_has_only_six_real_anchor_rows(geometry):
    method, counts = geometry(NS(mtp=None))
    assert method == MethodGeometry("none", 0)
    assert counts == (1,) * 6
    assert method.stock_mtp_cache_layers == 0 and method.draft_tail_rows == 0
    for count in counts:
        assert method.target_rows(count - 1) == count == method.target_tail_rows
    with pytest.raises(ValueError, match="cannot allocate proposals"):
        MethodGeometry("none", 2)  # Exact original fixture mismatch, not a supported R3 region.


def test_real_mtp_geometry_retains_every_original_varied_and_partial_prefix_window(geometry):
    method, counts = geometry(NS(mtp=object()))
    assert method == MethodGeometry("mtp", 2)
    assert method.stock_mtp_cache_layers == 1 and method.draft_tail_rows == 2
    assert counts == (1, 3, 2, 1, 2, 3)
    assert set(counts) == {1, 2, 3}
    kept = [min(count, 1 + index % 2) for index, count in enumerate(counts)]
    assert any(accepted < count for accepted, count in zip(kept, counts))
    for count in counts:
        assert method.target_rows(count - 1) == count <= method.target_tail_rows


@pytest.mark.parametrize("mtp", [False, True])
def test_prepared_integration_uses_reference_attention_and_actual_method_domain(geometry, mtp, monkeypatch):
    # Execute only the maintained integration adapter. The two numerical gate
    # functions are observed at their boundary, never replaced on GPU runs.
    path = Path(__file__).parent / "cuda/test_sm70_gguf_prepared.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "test_prepared_bound_six_lane_capture_preserves_valid_rows_state_and_mtp")
    function.decorator_list = []
    family, padded, calls = NS(_NATIVE_FAMILY=False, _NATIVE_ATTENTION=True), NS(_ENABLED=False), []
    target, stock = NS(mtp=None), NS(mtp=object())

    def six_lanes(weights, cache):
        assert family._NATIVE_FAMILY and not family._NATIVE_ATTENTION and padded._ENABLED
        method, counts = geometry(weights)
        expected = MethodGeometry("mtp", 2) if mtp else MethodGeometry("none", 0)
        assert method == expected
        assert counts == ((1, 3, 2, 1, 2, 3) if mtp else (1,) * 6)
        calls.append(("six", weights, cache))

    def mtp_gate(weights, cache):
        assert mtp and weights is stock and not family._NATIVE_ATTENTION
        calls.append(("mtp", weights, cache))

    bound = ModuleType("test_flashnext_sm70_bound_graphs")
    bound._qualify_six_lanes, bound._qualify_mtp = six_lanes, mtp_gate
    companion = ModuleType("test_flashnext_sm70_mtp")
    companion._weights_with_stock_mtp_protocol = lambda: stock
    monkeypatch.setitem(sys.modules, bound.__name__, bound)
    monkeypatch.setitem(sys.modules, companion.__name__, companion)
    namespace = {"family_ops": family, "padded_linears": padded, "dense_fixture": lambda value: value,
                 "_small_weights": lambda **_kwargs: target}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    namespace[function.name]("int8", mtp, monkeypatch)
    expected = stock if mtp else target
    expected_calls = [("six", expected, "int8")]
    if mtp:
        expected_calls.append(("mtp", expected, "int8"))
    assert calls == expected_calls
