"""Generate the private strided candidate from the frozen correct reference."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE = "a933c061dc2e9b3d790a8e2cbd2458f2252b2b7a"


def generate(repo, receipt):
    source_hashes, products = {}, {}

    def source(path):
        value = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=repo, text=True)
        source_hashes[path] = hashlib.sha256(value.encode()).hexdigest()
        return value

    def replace(value, old, new, count=1):
        if value.count(old) != count:
            raise ValueError(f"frozen source pattern count changed: {old!r}")
        return value.replace(old, new)

    original = source("src/tensorfold/cuda/nvfp4/host_experts.py")
    function = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)
                    and n.name == "load_host_experts_a4")
    loader = ast.get_source_segment(original, function)
    loader = replace(loader, "def load_host_experts_a4(", "def load_blob_host_experts_a4(")
    loader = replace(loader, "    from .experts_a4 import CachedExpertsA4, ExpertsA4, pack_projection_a4",
                     "    from .blob_cache_a4 import BlobCachedExpertsA4, BlobExpertCache\n"
                     "    from .experts_a4 import pack_projection_a4")
    loader = replace(loader, "    device = torch.device(device)",
                     "    if cache is None:\n"
                     "        return load_host_experts_a4(ctx, name, full, device, owned_ids=owned_ids, swiglu=swiglu)\n"
                     "    if not isinstance(cache, BlobExpertCache):\n"
                     "        raise TypeError('private one-blob loader requires BlobExpertCache')\n"
                     "    device = torch.device(device)")
    begin = loader.index("    fields = tuple(")
    end = loader.index("    for local, expert in enumerate(ids):", begin)
    loader = loader[:begin] + (
        "    blob = cache.allocate_authority((len(ids), cache.entry_bytes), dtype=torch.uint8, tag=name)\n"
        "    fields = cache.source_views(blob)\n"
        "    cache.zero_padding(blob)\n") + loader[end:]
    begin = loader.index("    if cache is not None:\n        key =")
    end = loader.index('    ctx.used["nvfp4"] = True', begin)
    loader = loader[:begin] + (
        "    key = len(cache._layers) if layer_id is None else layer_id\n"
        "    bank = BlobCachedExpertsA4(fields, width, dims, blob=blob, cache=cache, layer_id=key, swiglu=swiglu)\n"
        ) + loader[end:]
    # Geometry validation remains before allocation, despite the single blob.
    loader = replace(loader, "    geometry = authority_fields_a4(len(ids), width, dims)",
                     "    authority_fields_a4(len(ids), width, dims)\n"
                     "    from .blob_layout_a4 import blob_geometry_a4\n\n"
                     "    if cache.blob_geometry != blob_geometry_a4(width, dims):\n"
                     "        raise ValueError('private blob cache differs from this expert authority')")
    products["src/tensorfold/cuda/nvfp4/blob_host_a4.py"] = (
        '"""PRIVATE generated single-blob loading variant; regenerate, do not hand-edit."""\n\n'
        "import torch\n\nfrom .host_experts import _ids, load_host_experts_a4\n\n\n" + loader + "\n")

    cpp = source("src/tensorfold/cuda/nvfp4/experts_a4.cpp")
    before = "// Plan rows are (source-row, destination-slot, cache-cell). They are constructed\n"
    helper = '''// Native blob fields have a contiguous per-expert payload and an explicit
// outer expert stride. Admit each affine span before multiplication, including
// singleton/empty edge cases, rather than assuming a fully contiguous bank.
static void expert_field(const at::Tensor& value, const at::Tensor& on, at::ScalarType type, int rank,
                         const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == on.device() && value.scalar_type() == type &&
                value.dim() == rank && value.stride(rank - 1) == 1, name, ": native expert field");
    const auto limit = std::numeric_limits<int64_t>::max() / value.element_size();
    int64_t span = value.size(rank - 1);
    TORCH_CHECK(span > 0, name, ": positive inner field dimension required");
    for (int axis = rank - 2; axis >= 0; --axis) {
        TORCH_CHECK(value.size(axis) > 0 && value.stride(axis) >= span &&
                    (axis == 0 || value.stride(axis) == span), name, ": complete per-expert field");
        span = extend_span(span, value.size(axis), value.stride(axis), limit, name);
    }
}

'''
    cpp = replace(cpp, before, helper + before)
    for call in ('tensor(input, x, at::kFloat, 2, "input calibration");',
                 'tensor(blob, blob, at::kByte, 3, "projection blob");',
                 'tensor(weight, blob, at::kFloat, 2, "weight calibration");',
                 'tensor(input, blob, at::kFloat, 2, "input calibration");'):
        cpp = replace(cpp, call, call.replace("tensor(", "expert_field(", 1))
    cpp = cpp.replace("expert_a4_quant_cuda", "blob_a4_quant_cuda").replace("expert_a4_gemm_cuda", "blob_a4_gemm_cuda")
    cpp = replace(cpp, "void quantize(", "void blob_quantize(")
    cpp = replace(cpp, "void gemm(", "void blob_gemm(")
    cpp = replace(cpp, "&quantize", "&blob_quantize")
    cpp = replace(cpp, "&gemm", "&blob_gemm")
    products["src/tensorfold/cuda/nvfp4/blob_experts_a4.cpp"] = "// PRIVATE generated stride variant; regenerate, do not hand-edit.\n" + cpp

    cu = source("src/tensorfold/cuda/nvfp4/experts_a4.cu")
    cu = replace(cu, "const int* plan, int pairs, const float* input, int cells, int projections,",
                 "const int* plan, int pairs, const float* input, int cells, int projections, int64_t input_stride,")
    cu = replace(cu, "input[static_cast<int64_t>(cell) * projections + projection]",
                 "input[static_cast<int64_t>(cell) * input_stride + projection]")
    cu = replace(cu, "const uint8_t* blob, int64_t blob_bytes, const float* weight, const float* input, int cells, int projections,",
                 "const uint8_t* blob, int64_t blob_bytes, const float* weight, const float* input,\n"
                 "    int64_t blob_stride, int64_t weight_stride, int64_t input_stride, int cells, int projections,")
    cu = replace(cu, "blob + (static_cast<int64_t>(cell) * projections + projection) * blob_bytes",
                 "blob + static_cast<int64_t>(cell) * blob_stride + projection * blob_bytes")
    cu = replace(cu, "    const int64_t scale_at = static_cast<int64_t>(cell) * projections + projection;\n"
                 "    const float alpha = __fmul_rn(__fdiv_rn(1.f, input[scale_at]), weight[scale_at]);",
                 "    const int64_t input_at = static_cast<int64_t>(cell) * input_stride + projection;\n"
                 "    const int64_t weight_at = static_cast<int64_t>(cell) * weight_stride + projection;\n"
                 "    const float alpha = __fmul_rn(__fdiv_rn(1.f, input[input_at]), weight[weight_at]);")
    cu = replace(cu, "input.size(1), codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), paired);",
                 "input.size(1), input.stride(0), codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), paired);")
    cu = replace(cu, "weight.data_ptr<float>(), input.data_ptr<float>(), blob.size(0), blob.size(1), items.data_ptr<int>(),",
                 "weight.data_ptr<float>(), input.data_ptr<float>(), blob.stride(0), weight.stride(0), input.stride(0),\n"
                 "        blob.size(0), blob.size(1), items.data_ptr<int>(),")
    cu = cu.replace("expert_a4_quant_cuda", "blob_a4_quant_cuda").replace("expert_a4_gemm_cuda", "blob_a4_gemm_cuda")
    products["src/tensorfold/cuda/nvfp4/blob_experts_a4.cu"] = "// PRIVATE generated stride variant; regenerate, do not hand-edit.\n" + cu

    py = source("src/tensorfold/cuda/nvfp4/grouped_a4.py")
    py = replace(py, 'name="tensorfold_nvfp4_experts_a4_v1"', 'name="tensorfold_nvfp4_blob_a4_v1"')
    py = replace(py, 'here / "experts_a4.cpp"', 'here / "blob_experts_a4.cpp"')
    py = replace(py, 'here / "experts_a4.cu"', 'here / "blob_experts_a4.cu"')
    # No per-window unsqueeze: this view is prepared with the cache controls.
    py = replace(py, "fields[1].unsqueeze(1)", "bank.cache._native_down")
    # Share the unchanged host plan/window ownership implementation.
    tree = ast.parse(py)
    lo = next(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_lease") - 2
    hi = next(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run") - 1
    lines = py.splitlines(keepends=True)
    py = "".join(lines[:lo]) + "\n" + "".join(lines[hi:])
    py = replace(py, "from contextlib import contextmanager\n", "")
    py = replace(py, "from .experts_a4 import PAIRS\n", "from .experts_a4 import PAIRS\nfrom .grouped_a4 import _lease, _windows\n")
    products["src/tensorfold/cuda/nvfp4/blob_grouped_a4.py"] = (
        '# PRIVATE generated orchestration variant; regenerate, do not hand-edit.\n' + py)

    engine = source("src/tensorfold/families/qwen3_5_moe/cuda/nvfp4_engine.py")
    engine = replace(engine, "    def __init__(self, model_dir, draft_dir=None,", "    _experimental_cache_layout = 'single-blob-grouped'\n\n"
                     "    def __init__(self, model_dir, draft_dir=None,")
    engine = replace(engine, "from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4",
                     "from tensorfold.cuda.nvfp4.blob_host_a4 import load_blob_host_experts_a4 as load_host_experts_a4")
    engine = replace(engine, "from ..ram_experts import layout, plan_pool, runtime_geometry",
                     "from ..ram_experts import plan_pool, runtime_geometry\n"
                     "        from .blob_variant_a4 import layout")
    engine = replace(engine, "from tensorfold.cuda.expert_cache import HostExpertCache",
                     "from tensorfold.cuda.nvfp4.blob_cache_a4 import BlobExpertCache")
    engine = replace(engine, "            from tensorfold.cuda.nvfp4.blob_cache_a4 import BlobExpertCache\n"
                     "            from tensorfold.cuda.expert_pin import ExpertPinPolicy",
                     "            from tensorfold.cuda.expert_pin import ExpertPinPolicy\n"
                     "            from tensorfold.cuda.nvfp4.blob_cache_a4 import BlobExpertCache")
    engine = replace(engine, 'HostExpertCache(plan.entry_bytes, plan.entry_bytes, "cuda", pin_policy=pins)',
                     'BlobExpertCache(plan.entry_bytes, plan.entry_bytes, "cuda", pin_policy=pins,\n'
                     '                                                   width=text["moe_intermediate_size"], dims=text["hidden_size"])')
    products["src/tensorfold/families/qwen3_5_moe/cuda/nvfp4_engine.py"] = engine
    for path, data in products.items():
        if path.endswith(".py"):
            compile(data, path, "exec")
    for path, data in products.items():
        target = repo / path
        target.write_text(data)
    receipt.write_text(json.dumps({"base": BASE, "sources": source_hashes,
        "generated": {path: hashlib.sha256(data.encode()).hexdigest() for path, data in products.items()},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    generate(a.repo, a.receipt)
