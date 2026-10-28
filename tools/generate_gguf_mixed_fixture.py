"""Generate wire fixtures with unmodified pinned GGML dequantizers.

Supply the original ggml-quants.c and ggml-common.h from the revision below.
They are read-only reference inputs, never installed or edited. The small C shim
only supplies the documented half conversion and exports the selected routines.
No TensorFold decoder or codebook participates in the oracle.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

REVISION = "b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9"
SOURCES = {
    "ggml-quants.c": "5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab",
    "ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
}
# Kept independent of the production layout map to detect changed wire schemas.
FORMATS = {14: ("q6_K", 256, 210, 208), 18: ("iq3_xxs", 256, 98, 0),
           21: ("iq3_s", 256, 110, 0), 22: ("iq2_s", 256, 82, 0),
           23: ("iq4_xs", 256, 136, 0), 42: ("q2_0", 64, 18, 0)}


def function(source, name):
    start = source.index("void dequantize_row_" + name + "(")
    body = source.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def generate(quants, common, output, *, finite_projections=False):
    for path, name in ((quants, "ggml-quants.c"), (common, "ggml-common.h")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCES[name]:
            raise ValueError("reference source identity mismatch: " + name)
    shim = '''#include <assert.h>
#include <stdint.h>
#include <string.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#define GGML_RESTRICT restrict
static float fp16_to_fp32(uint16_t bits) {
    _Float16 h;
    memcpy(&h, &bits, sizeof h);
    return (float) h;
}
#define GGML_FP16_TO_FP32(x) fp16_to_fp32(x)
'''
    reference = quants.read_text()
    shim += "\n\n".join(function(reference, f[0]) for f in FORMATS.values()) + "\n"
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("a C compiler is required to regenerate the independent fixture")
    arrays = {}
    with tempfile.TemporaryDirectory(prefix="tf-ggml-wire-") as folder:
        directory = Path(folder)
        shutil.copyfile(common, directory / "ggml-common.h")
        (directory / "oracle.c").write_text(shim)
        flags = ["-std=gnu11", "-O2", "-ffp-contract=off", "-fno-strict-aliasing", "-shared", "-fPIC"]
        subprocess.run([compiler, *flags, str(directory / "oracle.c"), "-o", str(directory / "oracle.so")],
                       check=True)
        oracle = ctypes.CDLL(str(directory / "oracle.so"))
        for kind, (name, block, size, scale_at) in FORMATS.items():
            rng = np.random.default_rng(617000 + kind)
            raw = rng.integers(0, 256, size=(64, size), dtype=np.uint8)
            scales = np.array([0., -0., 2.**-24, -2.**-24, 2.**-14, -2.**-14,
                               .001, -.25, 1., -2., 65504., -65504., 64., -64.,
                               3.140625, .33325] + [1.] * 48, dtype="<f2")
            if finite_projections:
                # Independent bounds: Q6_K4096, IQ3_XXS1976.25, IQ3_S7905,
                # IQ2_S988.125, IQ4_XS4064 and Q2_0=2 times the stored scale.
                # Common8192 * |d|<=2^-20 gives |coefficient|<=2^-7.
                # For |input|<=1, K2560 dots stay below21 including rounding,
                # FP16 SwiGLU below441 and K80 down below276.
                # The original full-range wire controls remain the default.
                scales = np.array([0., -0., 2.**-24, -2.**-24, 2.**-23, -2.**-23,
                                   2.**-22, -2.**-22, 2.**-21, -2.**-21,
                                   2.**-20, -2.**-20, 2.**-22, -2.**-22,
                                   2.**-23, 2.**-24] + [2.**-20] * 48, dtype="<f2")
            raw[:, scale_at:scale_at + 2] = scales.view(np.uint8).reshape(64, 2)
            # Exercise every IQ codebook entry, including both high index bits.
            if kind == 22:
                indexes = (np.arange(2048) % 1024).reshape(64, 32)
                raw[:, 2:34] = indexes & 255
                raw[:, 66:74] = ((indexes.reshape(64, 8, 4) >> 8)
                                 << (2 * np.arange(4))).sum(axis=2).astype(np.uint8)
            elif kind in (18, 21):
                indexes = np.arange(4096).reshape(64, 64) % (256 if kind == 18 else 512)
                raw[:, 2:66] = indexes & 255
                if kind == 21:
                    raw[:, 66:74] = ((indexes.reshape(64, 8, 8) >> 8)
                                     << np.arange(8)).sum(axis=2).astype(np.uint8)
            values = np.empty((64, block), dtype="<f4")
            run = getattr(oracle, "dequantize_row_" + name)
            run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            run.restype = None
            run(raw.ctypes.data, values.ctypes.data, values.size)
            arrays[f"raw_{kind}"] = raw.reshape(8, 8 * size)
            arrays[f"bits_{kind}"] = values.view("<u4").reshape(8, 8 * block)
        arrays["provenance"] = np.array(json.dumps({
            "revision": REVISION, "source_sha256": SOURCES,
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "flags": flags,
            "oracle": "unmodified selected dequantize_row functions and common header; _Float16 conversion shim",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scale_domain": "finite-fp16-projections-v1" if finite_projections else "full-fp32-wire-range-v1",
            "projection_bounds": ({"maximum_abs_scale": 2.**-20, "maximum_quant_multiplier": 8192,
                                   "maximum_abs_coefficient": 2.**-7, "maximum_abs_input": 1,
                                   "K": 2560, "maximum_abs_dot_including_rounding": 21,
                                   "maximum_abs_fp16_activation": 441,
                                   "K80_maximum_abs_down_including_rounding": 276}
                                  if finite_projections else None),
        }, sort_keys=True))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quants", required=True, type=Path)
    parser.add_argument("--common", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        help="override the mode-specific fixture path; finite mode never defaults to the full-range corpus")
    parser.add_argument("--finite-projections", action="store_true",
                        help="separate finite FP16 projection fixture; default retains full-range FP32 wire controls")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the selected output and mode without reading sources, compiling or writing files")
    args = parser.parse_args()
    output = args.output
    if output is None:
        name = "gguf_mixed_projection_ggml.npz" if args.finite_projections else "gguf_mixed_ggml.npz"
        output = Path(__file__).resolve().parents[1] / "tests/fixtures" / name
    if args.dry_run:
        print(json.dumps({"output": str(output), "finite_projections": args.finite_projections,
                          "scope": "plan only; source hashes/compiler are checked during generation"}))
        return
    generate(args.quants, args.common, output, finite_projections=args.finite_projections)


if __name__ == "__main__":
    main()
