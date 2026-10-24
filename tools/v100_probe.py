#!/usr/bin/env python3
"""Inventory GGUF inputs and record bounded SM70 native component checks.

Run from the installed verification image. The JSON qualifies these components;
full-model serving qualification is recorded separately with its workload.
"""

import argparse
from collections import Counter
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess

from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint


def inventory(path):
    checkpoint = GGUFCheckpoint.open(path)
    formats = Counter(t.encoding for t in checkpoint.tensors.values())
    return {
        "architecture": checkpoint.metadata["general.architecture"],
        "shards": [{"path": str(p), "size": p.stat().st_size} for p in checkpoint.paths],
        "tensors": len(checkpoint.tensors), "encodings": dict(sorted(formats.items())),
        "packed_bytes": sum(t.nbytes for t in checkpoint.tensors.values()),
        "ple": [{"name": t.name, "shape": t.shape, "encoding": t.encoding, "row_bytes": t.row_bytes}
                for t in checkpoint.tensors.values() if t.name == "per_layer_token_embd.weight"],
    }


def components():
    modules = {
        "mixed_gguf_weights": "tensorfold.families.qwen4_exp.cuda.gguf_weights",
        "fp16_family": "tensorfold.cuda.sm70.family_ops",
        "native_provider": "tensorfold.cuda.sm70.provider",
        "native_gguf": "tensorfold.cuda.sm70.gguf_ops",
        "native_gdn": "tensorfold.cuda.sm70.gdn_native",
    }
    return {name: {"module": module, "available": find_spec(module) is not None}
            for name, module in modules.items()}


def native_receipt(extension):
    binary = Path(extension.__file__)
    cubins = subprocess.check_output(["cuobjdump", "--list-elf", str(binary)], text=True).strip()
    if "sm_70" not in cubins:
        raise RuntimeError(f"{binary.name} does not contain an SM70 cubin")
    return {
        "module": extension.__name__, "path": str(binary),
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "cubins": cubins,
        "operators": sorted(name for name in dir(extension) if not name.startswith("_")),
    }


def _native_smoke(provider):
    import torch
    from tensorfold.cuda.sm70 import gdn_native, gguf_ops
    from tensorfold.families.qwen4_exp.cuda.gdn import GDNScratch, widths

    x = torch.ones((2, 10240), device="cuda", dtype=torch.float16)
    result = provider.rms_norm(x, torch.ones(10240, device="cuda"))
    torch.testing.assert_close(result, x, atol=0.001, rtol=0)

    # Original Q8_0 bytes: FP16 scale 1, followed by 32 signed quant values.
    packed = torch.cat((torch.tensor([0, 60], device="cuda", dtype=torch.uint8),
                        torch.arange(32, device="cuda", dtype=torch.uint8)))[None]
    decoded = gguf_ops.decode(packed, 8, 32)
    torch.testing.assert_close(decoded[0], torch.arange(32, device="cuda", dtype=torch.float16),
                               atol=0, rtol=0)

    # Exercise the maintained GDN scratch/state ABI with real TP8 head geometry.
    rows, nk, nv = 1, 2, 6
    channels, width = widths(nk, nv)
    half = {"device": "cuda", "dtype": torch.float16}
    projection = torch.zeros((rows, width), **half)
    conv = torch.zeros((3, channels), **half)
    weight = torch.zeros((channels, 4), **half)
    initial = torch.zeros((nv, 128, 128), device="cuda")
    log = torch.zeros(nv, device="cuda")
    bias = torch.zeros_like(log)
    norm = torch.ones(128, **half)
    scratch = GDNScratch(rows, "cuda", nk, nv, dtype=torch.float16)
    final, replayed = torch.empty_like(initial), torch.empty_like(initial)
    output = torch.empty((rows, nv * 128), **half)
    sums = torch.empty((rows, nv * 4), device="cuda")
    gdn_native.chain(projection, conv, weight, initial, log, bias, norm, 1e-6, rows,
                     scratch, final, output, sums)
    gdn_native.replay(initial, scratch, rows, replayed)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    torch.testing.assert_close(final, initial, atol=0, rtol=0)
    torch.testing.assert_close(replayed, final, atol=0, rtol=0)
    torch.cuda.synchronize()
    return {"rms_norm_10240": "passed", "q8_0_decode": "passed", "gdn_tp8_chain_replay": "passed"}


def probe():
    import torch
    from torch.utils import cpp_extension
    import triton
    from tensorfold.cuda.sm70 import SM70Provider, gdn_native, gguf_ops

    provider = SM70Provider()
    extensions = {"provider": provider.prepare(), "gguf": gguf_ops.extension(), "gdn": gdn_native.extension()}
    smoke = _native_smoke(provider)
    cuda_compiler = str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc")
    host_compiler = os.environ.get("CXX", "c++")
    return {
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "triton": triton.__version__,
        "compiler": subprocess.check_output([cuda_compiler, "--version"], text=True).strip(),
        "compilers": {
            "cuda": {"path": cuda_compiler,
                     "version": subprocess.check_output([cuda_compiler, "--version"], text=True).strip()},
            "cxx": {"path": shutil.which(host_compiler) or host_compiler,
                    "version": subprocess.check_output([host_compiler, "--version"], text=True).strip()},
        },
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "torch_architectures": torch.cuda.get_arch_list(),
        "native_modules": {name: native_receipt(extension) for name, extension in extensions.items()},
        "smoke": smoke,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Any shard of the existing mixed qwen4exp GGUF")
    parser.add_argument("--probe-gpu", action="store_true", help="Build and execute small SM70 component checks")
    args = parser.parse_args()
    if args.checkpoint is None and not args.probe_gpu:
        parser.error("select --checkpoint and/or --probe-gpu")
    receipt = {"schema_version": 2, "qualification": "component-receipt", "components": components()}
    if args.checkpoint:
        receipt["checkpoint"] = inventory(args.checkpoint)
    if args.probe_gpu:
        receipt["provider"] = probe()
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
