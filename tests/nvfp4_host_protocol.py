"""Actual pure NVFP4 schema operations for opaque host-only loader controls.

The format module also owns NumPy dequantization. Extract its complete schema
class and scheme function without importing that numerical runtime; the actual
shared quantization-policy class remains authoritative.
"""
from __future__ import annotations

import __future__
import ast
from pathlib import Path

from tensorfold.checkpoint_quantization import LinearQuantization


def linear_metadata_class():
    source = Path(__file__).resolve().parents[1] / "src/tensorfold/cuda/nvfp4/format.py"
    names = {"scheme", "LinearMetadata"}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise AssertionError("the actual NVFP4 schema class and storage predicate are required")
    scope = {"LinearQuantization": LinearQuantization}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return scope["LinearMetadata"]
