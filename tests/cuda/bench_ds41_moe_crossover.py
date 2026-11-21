"""Routed-experts crossover on one SM70 GPU: EM3 (ds41_decode_moe, Target.moe below lattice MIN_TOKENS) against the
exact lattice (ds41_lattice_moe) at 32..1024 rows on real 192-expert rank banks.

    python bench_ds41_moe_crossover.py HIST_DIR OUT.json

Routing per token: 6 distinct global experts by systematic sampling with inclusion probability 6 * share_e of the
friend's all-calls routing histogram (HIST_DIR/stage{0,1}.json, row 0: prompt-dominated), then one EP2 half's view
(other-half routes -> id 0, probability 0: Target.moe's form). One representative layer per (gate, down) weight-type
pair, both EP2 halves (ETP slice 0 / 1); the model total weights each pair by its layer count.

Timing (prompt forwards run eagerly): GPU time = CUDA events around REPS back-to-back calls / REPS (the host enqueues
ahead); wall = one call + synchronize (host launch + GPU, the short-chunk regime). Medians of 5. Also printed: the
relative L2 distance between the two outputs (EM3 uses Q8_1 activations, the lattice FP16 activations with FP32
gate/up; their accuracy against an FP32 oracle is asserted in test_sm70_ds41_lattice_moe.py).
"""

import json
import sys
import time

import numpy as np
import torch

HERE = __file__.rsplit("/", 1)[0]
sys.path[:0] = [HERE, HERE.rsplit("/", 1)[0]]
from test_sm70_onecat_ds41 import bank, checkpoint  # noqa: E402

from tensorfold.cuda.sm70 import ds41_decode_moe as em, ds41_lattice_moe as lat  # noqa: E402

E, LOCAL, TOP, REPS = 384, 192, 6, 10
MS = (32, 48, 64, 65, 80, 96, 112, 127, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768, 1024)
HIST, OUT = sys.argv[1], sys.argv[2]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def shares():
    out = {}
    for stage in (0, 1):
        j = json.load(open(f"{HIST}/stage{stage}.json"))
        for layer in j["layers"]:
            h = np.array(j["hist"][0][layer], dtype=np.float64)
            out[layer] = h / h.sum()
    return out


SHARES = shares()
TENSORS = checkpoint().tensors
TYPES = {layer: tuple(TENSORS[f"blk.{layer}.ffn_{p}_exps.weight"].ggml_type for p in ("gate", "down"))
         for layer in range(40)}
PAIRS = {}
for pair in TYPES.values():
    PAIRS[pair] = PAIRS.get(pair, 0) + 1


def banks(layer, half):
    size = {16: 66, 17: 74, 29: 56}
    tensors = [TENSORS[f"blk.{layer}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down")]
    s = half
    first = s * 576 // 256
    g, u, d = (bank(t)[half * LOCAL:(half + 1) * LOCAL] for t in tensors)
    db = size[tensors[2].ggml_type]
    out = [torch.from_numpy(np.ascontiguousarray(a)).cuda()
           for a in (g[:, s * 576:(s + 1) * 576], u[:, s * 576:(s + 1) * 576], d[:, :, first * db:(first + 3) * db])]
    return out, tensors[0].ggml_type, tensors[2].ggml_type, s * 576 - 256 * first


def routing(layer, m, half, seed):
    rng = np.random.default_rng(seed)
    pi = np.minimum(1.0, TOP * SHARES[layer])
    pi = pi * (TOP / pi.sum())
    ids = np.empty((m, TOP), dtype=np.int64)
    for t in range(m):
        perm = rng.permutation(E)
        c = np.cumsum(pi[perm])
        ids[t] = perm[np.searchsorted(c, rng.random() + np.arange(TOP), side="right").clip(0, E - 1)]
    probs = rng.random((m, TOP)).astype(np.float32) + 0.05
    probs /= probs.sum(-1, keepdims=True)
    mine = (ids >= half * LOCAL) & (ids < (half + 1) * LOCAL)
    local = np.where(mine, ids - half * LOCAL, 0).astype(np.int32)
    probs = np.where(mine, probs, 0).astype(np.float32)
    x = (rng.standard_normal((m, 5120)) * 0.5).astype(np.float16)
    return (torch.from_numpy(x).cuda(), torch.from_numpy(local).cuda(), torch.from_numpy(probs).cuda(),
            len(set(ids[mine].tolist())))


def gpu_ms(fn):
    fn()
    samples = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(REPS):
            fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) / REPS)
    return sorted(samples)[2]


def wall_ms(fn):
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return sorted(samples)[2]


em.prepare() if hasattr(em, "prepare") else em.extension()
lat.prepare()
WS_EM = em.Workspace("cuda")
WS_LAT = lat.Workspace(torch.device("cuda"))
results = {"pairs": {f"{g},{d}": n for (g, d), n in PAIRS.items()}, "rows": []}
reps = {pair: next(l for l in range(40) if TYPES[l] == pair) for pair in PAIRS}
log("pairs", PAIRS, "layers", reps)
for pair, layer in reps.items():
    for half in (0, 1):
        (gate, up, down), gt, dt, left = banks(layer, half)
        for m in MS:
            x, ids, p, distinct = routing(layer, m, half, 91000 + 1000 * layer + 10 * m + half)
            o_em, o_lat = torch.empty_like(x), torch.empty_like(x)
            f_em = lambda: WS_EM.experts(x, ids, p, gate, up, down, gt, dt, left, out=o_em)
            f_lat = lambda: lat.experts(x, ids, p, gate, up, down, gt, dt, left, WS_LAT, out=o_lat)
            g_em, g_lat = gpu_ms(f_em), gpu_ms(f_lat)
            w_em, w_lat = wall_ms(f_em), wall_ms(f_lat)
            g_em2 = gpu_ms(f_em)                       # A B A drift check
            torch.cuda.synchronize()
            rel = ((o_em.float() - o_lat.float()).norm() / o_lat.float().norm()).item()
            row = dict(layer=layer, half=half, types=[gt, dt], m=m, distinct=distinct, em3_gpu_ms=g_em,
                       em3_gpu2_ms=g_em2, lattice_gpu_ms=g_lat, em3_wall_ms=w_em, lattice_wall_ms=w_lat, rel_l2=rel)
            results["rows"].append(row)
            log(f"layer {layer} {gt}/{dt} half {half} M{m} distinct {distinct}: GPU EM3 {g_em:.3f}/{g_em2:.3f} "
                f"lattice {g_lat:.3f} ms | wall EM3 {w_em:.3f} lattice {w_lat:.3f} | rel_l2 {rel:.3g}")
        del gate, up, down
        torch.cuda.empty_cache()
WS_LAT.check()
print("model totals (40 layers, layer-count weighted, mean of halves): M | EM3 GPU | lattice GPU | EM3 wall | "
      "lattice wall (ms)")
for m in MS:
    tot = np.zeros(4)
    for r in results["rows"]:
        if r["m"] == m:
            n = PAIRS[tuple(r["types"])] / 2
            tot += n * np.array([r["em3_gpu_ms"], r["lattice_gpu_ms"], r["em3_wall_ms"], r["lattice_wall_ms"]])
    results.setdefault("totals", {})[m] = tot.tolist()
    print(f"M{m}: {tot[0]:.2f} | {tot[1]:.2f} | {tot[2]:.2f} | {tot[3]:.2f}  "
          f"{'EM3' if tot[0] < tot[1] else 'lattice'} faster (GPU)", flush=True)
json.dump(results, open(OUT, "w"), indent=1)
print("BENCH DONE", flush=True)
