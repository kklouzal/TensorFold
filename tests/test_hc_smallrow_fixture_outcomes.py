"""Actual GPU fixture cleanup structure; no mirrored CUDA implementation."""
import ast
from pathlib import Path


def test_actual_finalizer_reserves_secondary_storage_before_capture():
    path = Path(__file__).parent / 'cuda/test_sm70_hc_smallrow_vendor.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == 'test_direct_vendor_bits_original_weight_alias_and_capture_replay')
    slot = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'cleanup_failures' for target in node.targets))
    assert isinstance(slot.value, ast.List) and len(slot.value.elts) == 1
    assert isinstance(slot.value.elts[0], ast.Constant) and slot.value.elts[0].value is None
    captures = [node for node in ast.walk(function) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr in ('CUDAGraph', 'capture_begin')]
    assert captures and all(slot.lineno < call.lineno for call in captures)
    finalizer = next(node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler)
        and node.name == 'cleanup')
    assert not any(isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)) for node in ast.walk(finalizer))
    call = next(node for node in ast.walk(finalizer) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'raise_failures')
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == 'primary'
    assert isinstance(call.args[1], ast.Name) and call.args[1].id == 'cleanup_failures'
