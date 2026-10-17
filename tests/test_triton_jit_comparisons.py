"""Check Triton syntax without importing a CUDA runtime or a JIT module."""

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
ROTOR = SOURCE / "tensorfold/families/qwen4_exp/cuda/rotorquant_kernel.py"


def jit_functions(path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    modules, decorators = set(), set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "triton")
        elif isinstance(node, ast.ImportFrom) and node.module == "triton":
            decorators.update(alias.asname or alias.name for alias in node.names if alias.name == "jit")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if ((isinstance(target, ast.Name) and target.id in decorators) or
                    (isinstance(target, ast.Attribute) and target.attr == "jit" and
                     isinstance(target.value, ast.Name) and target.value.id in modules)):
                yield node
                break


def test_jit_comparisons_are_supported_by_triton_3_8():
    # Triton3.8 CodeGenerator.visit_Compare rejects chained comparisons,
    # including constexpr expressions passed to static_assert.
    checked, unsupported = 0, []
    for path in SOURCE.rglob("*.py"):
        for function in jit_functions(path):
            checked += 1
            unsupported.extend(f"{path.relative_to(SOURCE)}:{node.lineno}: {ast.unparse(node)}"
                               for node in ast.walk(function)
                               if isinstance(node, ast.Compare) and len(node.ops) > 1)
    assert checked > 0
    assert not unsupported, "\n".join(unsupported)


@pytest.mark.parametrize("codec", [-2, -1, 0, *range(1, 9), 9, 10, 2**63])
def test_rotorquant_codec_static_assert_preserves_inclusive_bounds(codec):
    rotate = next(function for function in jit_functions(ROTOR) if function.name == "rotate")
    conditions = [node.args[0] for node in ast.walk(rotate)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "static_assert"
                  and any(isinstance(name, ast.Name) and name.id == "CODEC" for name in ast.walk(node))]
    assert len(conditions) == 1
    expression = ast.Expression(conditions[0])
    assertion = compile(expression, str(ROTOR), "eval")
    admitted = eval(assertion, {"__builtins__": {}}, {"CODEC": codec})  # noqa: S307 -- Owned Boolean assertion.
    assert admitted is (codec in {1, 2, 3, 4, 5, 6, 7, 8})
