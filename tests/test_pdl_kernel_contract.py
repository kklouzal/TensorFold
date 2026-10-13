"""Source contract of programmatic dependent launch (tensorfold/cuda/pdl.py): every kernel that can be launched with
PDL waits for its predecessor as its first operation, on every path, and every launch passes the same policy to the
kernel and to the launcher. Checked mechanically over the converted sources (no GPU)."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "tensorfold"
TRITON = ["cuda/moe.py", "families/qwen4_exp/cuda/glue.py", "families/qwen4_exp/cuda/qmm.py",
          "families/qwen4_exp/cuda/attn_multi.py", "families/qwen4_exp/cuda/attention.py",
          "families/qwen4_exp/cuda/forward.py"]
CUDA = {"cuda/kernels/gdn.cu": ["tree_kernel", "replay_kernel"],
        "families/qwen4_exp/cuda/gdn_io.cu": ["front_kernel", "back_kernel"],
        "cuda/experts.cu": ["expert_kernel", "plan_kernel", "plan_rank", "plan_offsets", "plan_scatter"],
        "cuda/experts_prefill.cu": ["prefill_kernel"]}


def _jit_kernels(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list):
            yield node


@pytest.mark.parametrize("path", TRITON)
def test_triton_kernels_wait_first_and_launches_pass_one_policy(path):
    tree = ast.parse((ROOT / path).read_text())
    kernels = {}
    for fn in _jit_kernels(tree):
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        if "PDL" not in params:
            continue
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        first = body[0]
        assert (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
                and getattr(first.value.func, "id", None) == "pdl_wait"
                and [getattr(a, "id", None) for a in first.value.args] == ["PDL"]), (path, fn.name)
        kernels[fn.name] = 0
    assert kernels, path
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) \
                and getattr(node.func.value, "id", None) in kernels:
            keywords = {k.arg: ast.unparse(k.value) for k in node.keywords}
            assert keywords.get("PDL") == "pdl.ON" and keywords.get("launch_pdl") == "pdl.ON", \
                (path, node.func.value.id, keywords)
            kernels[node.func.value.id] += 1
    assert all(kernels.values()), (path, kernels)                    # every converted kernel is launched so


@pytest.mark.parametrize("path,names", sorted(CUDA.items()))
def test_cuda_kernels_wait_before_any_other_statement(path, names):
    text = (ROOT / path).read_text()
    for name in names:
        match = re.search(r"\b" + name + r"\s*\([^{]*\)\s*\{\s*(?://[^\n]*\n\s*)*([^\n;]*;)", text)
        assert match, (path, name)
        assert match.group(1).strip() == "tensorfold::pdl_wait();", (path, name, match.group(1))
    assert "<<<" not in text, f"{path}: a plain <<<>>> launch bypasses the PDL launcher"


def test_lane_matmul_waits_before_reading_inputs_when_launched_with_pdl():
    text = (ROOT / "cuda/kernels/qmm.cu").read_text()
    prologue = text[text.index("if constexpr (PDL) {"):text.index("} else {", text.index("if constexpr (PDL) {"))]
    assert prologue.index("load_w(") < prologue.index("grid_wait();") < prologue.index("load_x(") \
        < prologue.index("grid_launch();")
    reduce = text[text.index("__global__ void reduce_kernel"):]
    assert reduce.index("grid_wait();") < reduce.index("part[i]")
