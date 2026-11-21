"""DeepSeek-V4.1 routed-expert decode MoE for SM70: expert-major integer-dot kernels for 1..64 rows a launch.

Port of the friend's 1Cat overlay patch 240 (EM3, extending patch 110 EM2); see ds41_decode_moe.cu for the
provenance, the kernels and the TensorFold changes. Operands are ``onecat_ds41.v41_experts``'s (one EP/ETP
rank's original GGUF expert bytes: gate/up uint8 [E, 576, 20*block], down [E, 5120, 3*block] with ``left`` in
0/64/128/192; FP16 x [M, 5120]; int32 local ids and FP32 probabilities [M, 6]) and the output is the same
FP16 [M, 5120] partial, bit for bit: every kept route runs the 6-row path's arithmetic (Q8_1 activations,
FP32 integer-dot sums in a fixed order, FP16 gate/up, clamped SwiGLU x probability, FP16 + Q8_1 down input,
FP16 per-route down rows, raw_down_unroute's FP32 route sum). Rows are independent (batch invariant: a row's
output does not depend on M, on the other rows or on the schedule variant). Routes with probability 0 are
dropped (exactly +-0 in the reference); a kept route's id outside [0, E) traps the device.

Graph contract: ``Workspace`` owns one fixed device buffer and the codebook tables (allocated and built at
construction, outside capture). ``experts`` enqueues 4 kernels (5 for non-fused small-M variants) per 64 rows
on the current stream with no allocation (``out`` given), no host synchronization and grids fixed by M, so a
call is capturable and replays with any routing. Calls on one workspace are stream-ordered: the workspace is
reused by the next call, so one stream (or explicit ordering) per workspace. The caller retains operands
until current-stream completion.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

HIDDEN, ROWS, TOP = 5120, 576, 6
EXPERT_TYPES = (16, 17, 29)
MAX_ROWS = 64  # rows per launch sequence; more rows run in pieces of 64 (rows are independent)

# Variant schedule per weight type: [(first M, variant), ...] steps (gate/up "g<type>", down "d<type>").
# PROVISIONAL: the friend's production EM3 schedule (bench/results/ds41/dmoe-schedule.json, his V100 bench);
# replaced by our sweep (tests/cuda/bench_ds41_decode_moe.py).
SCHEDULE = {
    "g16": [(1, 16), (3, 4), (4, 20), (6, 4), (8, 20), (12, 4)],
    "g17": [(1, 10), (2, 4), (3, 20), (8, 4), (12, 20)],
    "g29": [(1, 4), (2, 16), (4, 14), (16, 4), (24, 14)],
    "d16": [(1, 19), (2, 22), (32, 21)],
    "d17": [(1, 20), (6, 16), (8, 13), (16, 12), (24, 21)],
    "d29": [(1, 35)],
}


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("DeepSeek-V4.1 decode experts require a Volta-enabled PyTorch build and SM70 device")
    here = Path(__file__).parent
    sources = [here / "ds41_decode_moe.cpp", here / "ds41_decode_moe.cu"]
    headers = [here / "onecat_ds41_dp4a.cuh", here / "onecat_ds41_codebooks.cuh"]
    flags = ["-O3"]  # onecat_ds41's (upstream's) options: the bit-exact reference is built with them
    identity = hashlib.sha256(repr(flags).encode())
    for path in [*sources, *headers]:
        identity.update(path.name.encode() + b"\0" + path.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_ds41_decode_moe_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cflags=["-O3"], extra_cuda_cflags=flags)


def pick(kind, weight_type, rows, schedule=None):
    """The scheduled variant of ``kind`` ("g" gate/up, "d" down) for ``weight_type`` at ``rows`` (1..64)."""
    steps = (schedule or SCHEDULE)[f"{kind}{weight_type}"]
    variant = steps[0][1]
    for first, value in steps:
        if rows >= first:
            variant = value
    return variant


class Workspace:
    """One device's fixed decode-MoE workspace and codebook tables (construct before graph capture)."""

    def __init__(self, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("create the DeepSeek-V4.1 decode MoE workspace before graph capture")
        ext = extension()
        self.buffer = torch.empty(ext.workspace_bytes(), dtype=torch.uint8, device=device)
        ext.init(self.buffer)

    def experts(self, x, ids, probabilities, gate, up, down, gate_type, down_type, left, *, out=None,
                schedule=None):
        """Routed experts for FP16 x [M >= 1, 5120], int32 top-6 local ids / FP32 probabilities -> FP16 [M, 5120]."""
        if gate_type not in EXPERT_TYPES or down_type not in EXPERT_TYPES:
            raise ValueError("V4.1 decode experts require IQ2_XXS/IQ2_XS/IQ1_M carriers")
        if out is None:
            out = torch.empty_like(x)
        ext = extension()
        rows = x.shape[0]
        for at in range(0, rows, MAX_ROWS):
            end = min(at + MAX_ROWS, rows)
            m = end - at
            ext.experts(self.buffer, out[at:end], x[at:end], ids[at:end], probabilities[at:end], gate, up, down,
                        int(gate_type), int(down_type), int(left), pick("g", gate_type, m, schedule),
                        pick("d", down_type, m, schedule))
        return out
