"""DeepSeek-V4.1 routed-expert prefill on the friend's TurboMind lattice tensor-core GEMMs (SM70).

Vendored from the friend's 1Cat-vLLM production path (base 4d57473d, overlay patches 140 ds41_pf_moe + 340
ds41_kern340 with its measured rule; provenance and licenses in ds41_lattice/PROVENANCE and THIRD_PARTY_NOTICES):
per call the routes are sorted on the device; per tile of experts the routed experts' original IQ2_XXS / IQ2_XS /
IQ1_M rows are transcoded into a canonical lattice bank (``gguf_prefill_transcode_sm70``) and ONE TurboMind sm70
HMMA grouped GEMM (lattice decode in the main loop, FP32 accumulation, FP16 out; tile variant + swizzle per patch
340's rule) runs over every route of the tile; then the clamped SwiGLU (FP16 hidden) and the down projection over
the rank's K window [left, left + 576) the same way; the token output is the FP32 sum of its routes in top-k order,
rounded to FP16.

Operands are ``onecat_ds41.v41_experts``'s (one EP/ETP rank's original GGUF expert bytes, FP16 x [M, 5120], int32
local ids in [0, E) and FP32 probabilities [M, 6]; probability-0 routes (other EP half) contribute nothing); the
output is the same FP16 [M, 5120] partial. Ids are the caller's invariant (Target.moe), not validated here.
TensorFold adaptation for EP2 x ETP4: probability-0 routes sort behind a sentinel expert so the GEMMs never see them
(the friend's TP4 ranks hold all 384 experts). No host synchronization; operands and the workspace must outlive
current-stream completion.

Numerics (TensorFold's exact decode; the friend's patch 140 FP16 arithmetic was measured at about 2.6x the rel L2
error vs the FP32 oracle for <= 7% faster MoE and removed): every weight is ggml's FP32 dequantized value rounded
once to FP16 (the canonical stats keep the source FP16 block scale d and the integer sub-scale; the decode forms
(2s+1)/8 * signed grid exactly in FP16 and rounds once in the HMUL by d), gate/up leave the GEMM in FP32 into the
clamped SwiGLU (FP32), times the probability, rounded to FP16; down rounds per route to FP16; the token output is
the FP32 sum of its routes (the rounding points of the fused-WMMA path this replaced, ds41_prefill_moe, whose
outputs it matched bitwise in lat2-x1). K order is the kernel's (no split-K). The transcode flags in
``Workspace.error`` a group whose (largest) weight overflows FP16 (``check`` raises).
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

HIDDEN, ROWS, TOP = 5120, 576, 6
LIMIT = 10.0
# GGUF type -> (canonical group, stats bytes per group and column) (patch 140 LATTICE).
LATTICE = {16: (32, 4), 17: (16, 4), 29: (16, 2)}
TILE = 128          # experts per canonical tile (the friend's production VLLM_LOCAL_DS41_PF_MOE_TILE)
CTAS = 640          # transcode CTAs (patch 140 default)
# Fewest chunk rows for the lattice (below: Target.moe's EM3 decode experts): the friend's production
# VLLM_LOCAL_DS41_PF_MOE_MIN. K38 (ds41-short, tests/cuda/bench_ds41_moe_crossover.py on real 192-expert rank banks,
# 1 GPU, 180 W, 40-layer GPU ms): EM3 is faster than the lattice through 320 rows (128: 56.7 vs 139.3, 256: 118.0 vs
# 170.3; the lattice transcodes every touched expert per call, ~2.5 ms per layer) and the 8-rank prompt forward
# took 110.8 vs 143.4 ms at 128 rows, but EM3's Q8_1 activations put its output 1.2e-2 (rel L2) from the lattice's
# and the corpus prefill NLL moved d +0.016 (CI [-0.003, +0.041], short-s1): less accurate, so the lattice keeps
# every chunk from 128 rows. Below 128 the lattice is 2.4-3.5x slower than EM3 (no gain to buy with accuracy).
MIN_TOKENS = 128
# Patch 340 rule measured by the friend on his V100s (misc/ds41-knobs/kern340-rule.json, "enabled": 1 in his
# production runtime): (decoder, n, k) -> [(max routes per call, variant, swizzle)]. Routes per call are the
# chunk's token rows x 6 (his TP4 ranks route every token to their 384 experts; an EP2 rank sees half the routes on
# half the experts, the same routes per expert).
RULE = {
    (16, 1152, 5120): [(3456, 6, 0), (6528, 7, 0), (12672, 7, 0), (1 << 30, 7, 0)],
    (16, 5120, 576): [(3456, 6, 3), (6528, 7, 0), (12672, 3, 2), (1 << 30, 3, 2)],
    (17, 1152, 5120): [(3456, 6, 0), (6528, 7, 0), (12672, 7, 0), (1 << 30, 7, 0)],
    (17, 5120, 576): [(3456, 6, 3), (6528, 3, 1), (12672, 3, 2), (1 << 30, 3, 2)],
    (29, 1152, 5120): [(3456, 6, 0), (6528, 7, 0), (12672, 7, 2), (1 << 30, 7, 0)],
    (29, 5120, 576): [(3456, 6, 3), (6528, 3, 0), (12672, 3, 2), (1 << 30, 3, 3)],
}

_HERE = Path(__file__).parent
_VENDOR = _HERE / "ds41_lattice"


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("DeepSeek-V4.1 lattice prefill experts require a Volta-enabled PyTorch build and SM70")
    sources = [_HERE / "ds41_lattice_moe.cpp", _HERE / "ds41_lattice_moe.cu",
               *(_VENDOR / n for n in ("ds41kern_op.cu", "ds41kern_t16.cu", "ds41kern_t17.cu", "ds41kern_t29.cu",
                                       "gguf_prefill_transcode_sm70.cu")),
               _VENDOR / "turbomind" / "kernels" / "gemm" / "kernel.cu"]
    headers = sorted(p for p in _VENDOR.rglob("*") if p.suffix in (".h", ".cuh"))
    # The friend's kern340/src/build.sh flags (default FMA contraction, like his build); TurboMind uses the CUDA
    # half/bfloat16 operators that Torch's extension defaults disable.
    flags = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DENABLE_SM70_TURBOMIND=1",
             "-DNDEBUG", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    identity = hashlib.sha256(repr(flags).encode())
    for path in [*sources, *headers]:
        identity.update(str(path.relative_to(_HERE)).encode() + b"\0" + path.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_ds41_lattice_moe_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cflags=["-O3"], extra_cuda_cflags=flags)


def prepare():
    """Build/load the extension before serving (otherwise the first call does)."""
    return extension()


def canonical(kind, n, k):
    """(group, weight bytes, stats bytes) of one expert's canonical bank [n, k] (patch 140 ``canonical``)."""
    group, stat = LATTICE[kind]
    if k % group or n % 32:
        raise ValueError(f"lattice bank {kind}: n {n} / k {k} not aligned")
    return group, n * k // 4, k // group * n * stat


# The exact decode's instantiations (ds41kern_variants.cuh) of the rule's tilings (the friend's patch-340 variant
# numbers 3/6/7): FP32-output gate|up, FP16 down.
EXACT = {1152: {6: 0, 7: 1}, 5120: {3: 2, 6: 3, 7: 4}}


def pick(decoder, n, k, routes):
    for max_routes, variant, swizzle in RULE[(decoder, n, k)]:
        if routes <= max_routes:
            return EXACT[n][variant], swizzle
    raise AssertionError("the last rule bucket is unbounded")


class Workspace:
    """One rank's canonical gate|up and down banks for ``tile`` experts (the largest formats; allocated at
    construction so the KV pool can be sized after it), their TurboMind (pointer, ld) tables and the transcode
    error word. Owned by its Target; one forward at a time (current stream)."""

    def __init__(self, device, tile=TILE):
        self.tile = tile
        gu = max(canonical(t, 2 * ROWS, HIDDEN)[1] + canonical(t, 2 * ROWS, HIDDEN)[2] for t in LATTICE)
        dn = max(canonical(t, HIDDEN, ROWS)[1] + canonical(t, HIDDEN, ROWS)[2] for t in LATTICE)
        self.banks = {"gu": torch.empty(tile * gu, dtype=torch.uint8, device=device),
                      "dn": torch.empty(tile * dn, dtype=torch.uint8, device=device)}
        self.error = torch.zeros(1, dtype=torch.int32, device=device)
        self.tables = {}
        self.slots = torch.arange(tile, dtype=torch.int64, device=device)

    @property
    def nbytes(self):
        return sum(b.numel() for b in self.banks.values())

    def table(self, key, kind, n, k):
        found = self.tables.get((key, kind))
        if found is None:
            _, wbytes, sbytes = canonical(kind, n, k)
            buffer = self.banks[key]
            weights, stats = buffer[:self.tile * wbytes], buffer[self.tile * wbytes:self.tile * (wbytes + sbytes)]
            pair = []
            for base, stride, ld in ((weights, wbytes, 32 * k), (stats, sbytes, n)):
                t = torch.empty(self.tile, 2, dtype=torch.int64, device=buffer.device)
                t[:, 0] = base.data_ptr() + self.slots * stride
                t[:, 1] = ld
                pair.append(t.view(torch.uint8).flatten())
            found = self.tables[(key, kind)] = (weights, stats, wbytes, sbytes, pair[0], pair[1])
        return found

    def check(self):
        """Raise if any transcode since construction met an FP16 scale overflow (synchronizes)."""
        if int(self.error.item()):
            raise RuntimeError("DeepSeek-V4.1 lattice prefill transcode: an FP16 block scale overflowed")


def _addresses(bank):
    return bank.data_ptr() + torch.arange(bank.shape[0], dtype=torch.int64, device=bank.device) * bank.stride(0)


def experts(x, ids, probabilities, gate, up, down, gate_type, down_type, left, workspace, *, out=None):
    """Routed experts for FP16 x [M, 5120], top-6 local ids/probabilities -> FP16 [M, 5120] (numerics: see the
    module docstring)."""
    if gate_type not in LATTICE or down_type not in LATTICE:
        raise ValueError("V4.1 lattice prefill experts require IQ2_XXS/IQ2_XS/IQ1_M carriers")
    ext = extension()
    experts_n, inter = gate.shape[0], gate.shape[1]
    tokens = x.shape[0]
    routes = tokens * TOP
    if inter != ROWS or x.shape[1] != HIDDEN or ids.shape != (tokens, TOP) or down.shape[:2] != (experts_n, HIDDEN):
        raise ValueError("V4.1 lattice prefill: [E, 576, bytes] gate/up, [E, 5120, bytes] down, top-6 routes")
    device = x.device
    flat = torch.where(probabilities.reshape(-1) != 0, ids.reshape(-1), experts_n).to(torch.int32)
    ordered, order = torch.sort(flat, stable=True)
    offsets = torch.searchsorted(ordered, torch.arange(experts_n + 1, dtype=torch.int32, device=device),
                                 out_int32=True)
    valid = offsets[experts_n:]
    rows = torch.empty(routes, dtype=torch.int32, device=device)
    inverse = torch.empty(routes, dtype=torch.int32, device=device)
    ext.rows(order, rows, inverse, TOP)
    tile = workspace.tile
    plan = [(first, min(tile, experts_n - first)) for first in range(0, experts_n, tile)]
    lists = torch.empty(len(plan) * tile, dtype=torch.int32, device=device)
    counts = torch.empty(len(plan), dtype=torch.int32, device=device)
    ext.tile_lists(offsets, lists, counts, experts_n, tile)
    lists = lists.view(len(plan), tile)
    weights = probabilities.reshape(-1).contiguous()

    def projection(key, kind, n, k, k_offset, src0, src1, stride, source, indices, result):
        group = LATTICE[kind][0]
        bank_w, bank_s, wbytes, sbytes, wt, st = workspace.table(key, kind, n, k)
        variant, swizzle = pick(kind, n, k, routes)
        for index, (first, count) in enumerate(plan):
            ext.transcode(bank_w, bank_s, wbytes, sbytes, src0[first:first + count],
                          None if src1 is None else src1[first:first + count], lists[index],
                          counts[index:index + 1], CTAS, kind, n, k, k_offset, stride, workspace.error)
            ext.lattice_gemm(result, source, offsets[first:first + count + 1], wt[:count * 16], st[:count * 16],
                             kind, n, count, group, indices, variant, swizzle)

    proj = torch.empty(routes, 2 * inter, dtype=torch.float32, device=device)
    projection("gu", gate_type, 2 * inter, HIDDEN, 0, _addresses(gate), _addresses(up), gate.stride(1),
               x.contiguous(), rows, proj)
    hidden = torch.empty(routes, inter, dtype=torch.float16, device=device)
    ext.swiglu(proj, weights, order, valid, hidden, LIMIT)
    del proj
    down_rows = torch.empty(routes, HIDDEN, dtype=torch.float16, device=device)
    projection("dn", down_type, HIDDEN, inter, int(left), _addresses(down), None, down.stride(1), hidden, None,
               down_rows)
    if out is None:
        out = torch.empty_like(x)
    ext.combine(down_rows, inverse, valid, out, TOP)
    return out
