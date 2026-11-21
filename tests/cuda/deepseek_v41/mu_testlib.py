"""Shared setup for the multi-request (MU) kernel check scripts (ported from the friend overlay snapshot
lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
tests/ds41mu_testlib.py).

The overlay harness rebuilt the upstream 4d57473d kernels from a git checkout with text patches (plain triton,
libdevice exp -> tl.exp, the trap -> a no-op, PTX RNE -> FP32 RNE). TensorFold vendors those modules with
interpreter forms of the same three helpers, so the references here are simply the vendored upstream modules.

DS41MU_DEVICE=cpu (default) requires TRITON_INTERPRET=1; DS41MU_DEVICE=cuda runs the same sections compiled.
"""

import os

DEVICE = os.environ.get("DS41MU_DEVICE", "cpu")
if DEVICE == "cpu":
    assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"


from v41_interpreter import install

install()

import torch

torch.set_default_device(DEVICE)


def qdq(module, x, *, fp4, group, e4m3_scale=False):
    """attention_qdq without the CUDA check (the upstream _qdq kernel)."""
    out = x.new_empty(x.shape)
    module._qdq[(x.numel() // group,)](
        x.contiguous(), out, GROUP=group, FP4=fp4, E4M3_SCALE=e4m3_scale,
        num_warps=4, enable_fp_fusion=False,
    )
    return out
