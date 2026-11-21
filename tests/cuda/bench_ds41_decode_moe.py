"""Decode MoE schedule sweep and per-layer timing on one SM70 GPU: ds41_decode_moe (EM3) vs the 6-row pieces.

    python bench_ds41_decode_moe.py HIST_DIR OUT.json

HIST_DIR holds stage0.json / stage1.json, the friend's measured per-layer decode routing histograms (450
"estats", row 1 = <= 32-row calls). Routing per token: 6 distinct global experts by randomized systematic
sampling with inclusion probability 6 * share_e (exact marginals, tokens independent), then one EP2 half's
view (other-half routes -> id 0, probability 0: Target.moe's form). Weights: one real layer's 192-expert
rank bank (ETP slice 0 for half 0, slice 1 for half 1). All timings are CUDA-graph replays (decode runs in
startup graphs): REPS launches per graph, median of 5 replays, per launch.

Phase sweep: per weight type (a representative layer each), M, routing seed and plan chunk c in 4/8/16, every
gate/up variant (incl. the SwiGLU launch of non-fused ones) and every down variant alone; the schedule is
chosen by coordinate descent on the layer-count-weighted sum over the model's (gate, down) type pairs.
Phase e2e: per test layer, half and M: 6-row pieces vs EM3 with the friend's schedule vs EM3 with the swept
schedule (held-out seeds), with a bit-equality check of every output. Prints a SCHEDULE line and writes JSON.
"""

import json
import sys
import time

import numpy as np
import torch

HERE = __file__.rsplit("/", 1)[0]
sys.path[:0] = [HERE, HERE.rsplit("/", 1)[0]]  # tests/cuda (the oracle module), tests (its fixtures)
from test_sm70_onecat_ds41 import bank, checkpoint  # noqa: E402

from tensorfold.cuda.sm70 import ds41_decode_moe as em, onecat_ds41 as dec  # noqa: E402

E, LOCAL, TOP, REPS = 384, 192, 6, 10
MS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
E2E_MS = (1, 2, 4, 8, 16, 24, 32, 64)
CHUNKS = (4, 8, 16)
FRIEND = dict(em.SCHEDULE)
HIST, OUT = sys.argv[1], sys.argv[2]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def shares():
    out = {}
    for stage in (0, 1):
        j = json.load(open(f"{HIST}/stage{stage}.json"))
        for layer in j["layers"]:
            h = np.array(j["hist"][1][layer], dtype=np.float64)
            out[layer] = h / h.sum()
    return out


SHARES = shares()
TENSORS = checkpoint().tensors
TYPES = {}
for layer in range(40):
    g, d = (TENSORS[f"blk.{layer}.ffn_{p}_exps.weight"].ggml_type for p in ("gate", "down"))
    TYPES[layer] = (g, d)
PAIRS = {}
for pair in TYPES.values():
    PAIRS[pair] = PAIRS.get(pair, 0) + 1
log("type pairs (gate, down): layers", PAIRS)


def banks(layer, half):
    size = {16: 66, 17: 74, 29: 56}
    tensors = [TENSORS[f"blk.{layer}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down")]
    s = half  # ETP slice
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
            int(mine.sum()), len(set(ids[mine].tolist())))


def graph_ms(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(REPS):
            fn()
    graph.replay()
    samples = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) / REPS)
    del graph
    return sorted(samples)[2]


EXT = em.extension()
WS = em.Workspace("cuda")
TABLE = EXT.variants()
GU = {v: row[1] for v, row in enumerate(r for r in TABLE if r[0] == 0)}
DN = {v: row[1] for v, row in enumerate(r for r in TABLE if r[0] == 1)}
results = {"pairs": {f"{g},{d}": n for (g, d), n in PAIRS.items()}, "sweep": {}, "e2e": []}

# ---- sweep
rep_gate = {t: next(l for l in range(40) if TYPES[l][0] == t) for t in {g for g, _ in TYPES.values()}}
rep_down = {t: next(l for l in range(40) if TYPES[l][1] == t) for t in {d for _, d in TYPES.values()}}
log("representative layers gate", rep_gate, "down", rep_down)
gu_t = {}  # (type, m) -> {(v, c): ms}
dn_t = {}
for kind, reps in (("g", rep_gate), ("d", rep_down)):
    for t, layer in sorted(reps.items()):
        (gate, up, down), gt, dt, left = banks(layer, 1)
        for m in MS:
            acc = {}
            for seed in (1, 2):
                x, ids, p, _, _ = routing(layer, m, 1, 1000 * layer + 10 * m + seed)
                for c in CHUNKS:
                    EXT.prep(WS.buffer, x, ids, p, LOCAL, c)
                    if kind == "g":
                        for v, ch in GU.items():
                            if ch >= c:
                                ms = graph_ms(lambda: EXT.gate_up(WS.buffer, p, gate, up, gt, v))
                                acc.setdefault((v, c), []).append(ms)
                    else:
                        # a valid down input: one fused gate/up whose chunk (16) covers every plan chunk
                        EXT.gate_up(WS.buffer, p, gate, up, gt, 9)
                        for v, ch in DN.items():
                            if ch >= c:
                                ms = graph_ms(lambda: EXT.down(WS.buffer, p, down, dt, left, v))
                                acc.setdefault((v, c), []).append(ms)
            table = {k: float(np.mean(v)) for k, v in acc.items()}
            (gu_t if kind == "g" else dn_t)[(t, m)] = table
            best = min(table.items(), key=lambda kv: kv[1])
            results["sweep"][f"{kind}{t}@{m}"] = {f"{v}@{c}": ms for (v, c), ms in table.items()}
            log(f"{kind}{t} layer {layer} M{m}: best {best[0]} {best[1] * 1e3:.1f} us")
        del gate, up, down
        torch.cuda.empty_cache()


def cost(m, choice):
    total = 0.0
    for (g, d), n in PAIRS.items():
        gv, dv = choice[f"g{g}"], choice[f"d{d}"]
        c = min(GU[gv], DN[dv])
        total += n * (gu_t[(g, m)][(gv, c)] + dn_t[(d, m)][(dv, c)])
    return total


schedule = {}
for m in MS:
    choice = {f"g{t}": min((v for v in GU), key=lambda v: gu_t[(t, m)].get((v, min(GU[v], 8)), 1e9))
              for t in rep_gate}
    choice.update({f"d{t}": min((v for v in DN), key=lambda v: dn_t[(t, m)].get((v, min(DN[v], 8)), 1e9))
                   for t in rep_down})
    for _ in range(4):
        for key in list(choice):
            pool = GU if key[0] == "g" else DN
            choice[key] = min(pool, key=lambda v: cost(m, {**choice, key: v}))
    log(f"M{m}: choice {choice} weighted {cost(m, choice) * 1e3:.1f} us")
    for key, v in choice.items():
        steps = schedule.setdefault(key, [])
        if not steps or steps[-1][1] != v:
            steps.append((m, v))
print("SCHEDULE", json.dumps(schedule), flush=True)
results["schedule"] = schedule

# ---- e2e
test_layers = []
for pair in PAIRS:
    test_layers.append(next(l for l in range(40) if TYPES[l] == pair))
log("e2e layers", test_layers)
for layer in test_layers:
    for half in (0, 1):
        (gate, up, down), gt, dt, left = banks(layer, half)
        for m in E2E_MS:
            x, ids, p, routes, distinct = routing(layer, m, half, 777000 + 1000 * layer + 10 * m + half)
            out_p = torch.empty_like(x)

            def run_pieces():
                for at in range(0, m, 6):
                    end = min(at + 6, m)
                    out_p[at:end] = dec.v41_experts(x[at:end], ids[at:end], p[at:end], gate, up, down, gt, dt, left)

            out_f, out_s = torch.empty_like(x), torch.empty_like(x)
            t_p = graph_ms(run_pieces)
            t_f = graph_ms(lambda: WS.experts(x, ids, p, gate, up, down, gt, dt, left, out=out_f, schedule=FRIEND))
            t_s = graph_ms(lambda: WS.experts(x, ids, p, gate, up, down, gt, dt, left, out=out_s, schedule=schedule))
            t_p2 = graph_ms(run_pieces)  # A B C A drift check
            torch.cuda.synchronize()
            same = (torch.equal(out_p.view(torch.int16), out_f.view(torch.int16))
                    and torch.equal(out_p.view(torch.int16), out_s.view(torch.int16)))
            row = dict(layer=layer, half=half, types=[gt, dt], m=m, routes=routes, distinct=distinct,
                       pieces_us=t_p * 1e3, pieces2_us=t_p2 * 1e3, em3_friend_us=t_f * 1e3,
                       em3_swept_us=t_s * 1e3, bit_equal=same)
            results["e2e"].append(row)
            log(f"layer {layer} {gt}/{dt} half {half} M{m} routes {routes} distinct {distinct}: pieces "
                f"{t_p * 1e3:.1f}/{t_p2 * 1e3:.1f} us | EM3 friend {t_f * 1e3:.1f} | EM3 swept {t_s * 1e3:.1f} | "
                f"bit-equal {same}")
        del gate, up, down
        torch.cuda.empty_cache()
# (the crossover with the grouped prefill op lives in tests/cuda/test_sm70_ds41_lattice_moe.py: the lattice path
#  replaced ds41_prefill_moe)
json.dump(results, open(OUT, "w"), indent=1)
bad = [r for r in results["e2e"] if not r["bit_equal"]]
print("BENCH DONE" if not bad else f"BENCH FAIL {len(bad)} unequal rows", flush=True)
