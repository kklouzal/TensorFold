"""CPU checks for the maintained CUDA fixture's actual speculation domain."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

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
