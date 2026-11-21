"""Tables from tools/ds41_profile.py traces (CPU only). usage: ds41_profile_report.py RUN_DIR [--json OUT.json]

Kernels of eager / annotated traces take the category of their launch's enclosing ``ds41:`` annotations (launch
correlation -> CPU thread interval). Graph replays (one ``cudaGraphLaunch`` correlation per replay) take the
categories of the same case's eager twin (``Shape.body`` of the same bucket, the captured code) by kernel position;
a replay whose kernel count or names differ from the twin falls back to the twin's majority category per kernel
name (reported). NCCL kernels: ``wait`` = duration minus the minimum over ranks of the same collective (the
skew/arrival wait; the minimum approximates the transfer itself)."""

from __future__ import annotations

import collections
import difflib
import glob
import json
import os
import re
import sys

ORDER = ["routed", "shared", "router", "moe.misc", "dense", "attn.core", "kv_insert", "idx.proj", "idx.prep",
         "idx.scores", "idx.topk", "idx.prefill", "idx.other", "compressor", "hc/norm", "engram.gather",
         "engram.proj", "engram.gate", "engram.other", "hash", "embed", "logits", "misc",
         "comm:moe", "comm:attn", "comm:indexer", "comm:engram", "comm:embed", "comm:logits"]


def classify(stack, name):
    s = set(stack)
    # NCCL kernels, and the small all-gather / LL row all-reduce paths (``Collectives._reduce_half``, the logged
    # all-gathers): their kernels run under the "comm" / "AG" annotations (rank_sum, the reduction, stays compute)
    if "nccl" in name.lower() or (("comm" in s or "AG" in s) and "rank_sum" not in name):
        for ctx in ("moe", "indexer", "engram", "embed", "attn"):
            if ctx in s:
                return "comm:" + ctx
        return "comm:logits"
    if "hash" in s:
        return "hash"
    if "embed" in s:
        return "embed"
    if "engram" in s:
        for sub in ("engram.gather", "engram.gate"):
            if sub in s:
                return sub
        return "engram.proj" if "linear" in s else "engram.other"
    if "indexer" in s:
        for sub in ("idx.scores", "idx.topk", "idx.prefill"):
            if sub in s:
                return sub
        if "linear" in s:
            return "idx.proj"
        return "idx.prep" if ("rope" in s or "qdq" in s) else "idx.other"
    if "compress" in s or "c2" in s:
        return "compressor"
    if "kv_insert" in s or "ring_insert" in s:
        return "kv_insert"
    if "moe" in s:
        if "router.proj" in s or "router.topk" in s:
            return "router"
        if "routed" in s:
            return "routed"
        return "shared" if ("linear" in s or "shared.act" in s) else "moe.misc"
    if "attn" in s:
        if "linear" in s:
            return "dense"
        if "norm" in s:
            return "hc/norm"
        return "attn.core"
    if "head" in s:
        return "logits"
    if "hc" in s or "norm" in s:
        return "hc/norm"
    return "misc"


def load(path):
    data = json.load(open(path))
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels, launches, ann = [], {}, collections.defaultdict(list)
    copies = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat == "kernel":
            kernels.append(ev)
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = ev.get("args", {}).get("correlation")
            if corr is not None:
                launches[corr] = ev
        elif cat == "user_annotation" and ev["name"].startswith("ds41:"):
            ann[(ev["pid"], ev["tid"])].append((ev["ts"], ev["ts"] + ev["dur"], ev["name"][5:]))
        elif cat in ("gpu_memcpy", "gpu_memset"):
            copies.append(ev)
    kernels.sort(key=lambda e: e["ts"])
    for v in ann.values():
        v.sort()
    return kernels, launches, ann, copies


def stack_of(kernel, launches, ann):
    launch = launches.get(kernel.get("args", {}).get("correlation"))
    if launch is None:
        return None, None
    t = launch["ts"]
    labels = [lab for a, b, lab in ann.get((launch["pid"], launch["tid"]), ()) if a <= t <= b]
    return labels, launch


def busy(intervals):
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def eager_profile(path):
    """[(category, name, ts, dur)] of an annotated eager trace, plus copies."""
    kernels, launches, ann, copies = load(path)
    out = []
    for k in kernels:
        stack, _ = stack_of(k, launches, ann)
        out.append((classify(stack or [], k["name"]) if stack is not None else
                    ("comm:?" if "nccl" in k["name"].lower() else "unlaunched"), k["name"], k["ts"], k["dur"]))
    return out, copies


def graph_replays(path):
    kernels, launches, _, copies = load(path)
    groups = collections.OrderedDict()
    for k in kernels:
        corr = k.get("args", {}).get("correlation")
        launch = launches.get(corr)
        if launch is not None and "GraphLaunch" in launch["name"]:
            groups.setdefault(corr, []).append(k)
    return [sorted(g, key=lambda e: e["ts"]) for g in groups.values()], kernels


def short(name, n=70):
    name = re.sub(r"\(.*", "", name.replace("(anonymous namespace)", "anon"))
    return name if len(name) <= n else name[:n - 3] + "..."


def summarize_case(run, case, ranks):
    """Per rank: {category: ms per forward}, span, busy, gaps, kernel table, notes."""
    res = {"case": case, "ranks": {}, "notes": []}
    is_prefill = case.startswith("pf")
    twin = {}
    if not is_prefill:
        for r in ranks:
            p = os.path.join(run, f"trace-{case}-eager-r{r}.json")
            if os.path.exists(p):
                twin[r] = eager_profile(p)[0]
    nccl_by_rank = {}
    for r in ranks:
        p = os.path.join(run, f"trace-{case}-r{r}.json")
        if not os.path.exists(p):
            continue
        if is_prefill:
            items, _ = eager_profile(p)
            forwards = [items]
            gaps = []
        else:
            reps, _ = graph_replays(p)
            tw = twin.get(r, [])
            names_major = collections.defaultdict(collections.Counter)
            for cat, name, _, _ in tw:
                names_major[name][cat] += 1
            forwards = []
            mismatched = 0
            graph_only = collections.Counter()
            twin_names = [t[1] for t in tw]
            for g in reps:
                names = [k["name"] for k in g]
                matcher = difflib.SequenceMatcher(None, twin_names, names, autojunk=False)
                mapped = [None] * len(g)
                for a, b, size in matcher.get_matching_blocks():
                    for i in range(size):
                        mapped[b + i] = tw[a + i][0]
                extra = sum(m is None for m in mapped)
                if tw and extra <= max(8, len(g) // 10):
                    # graph-only nodes (captured D2D copies become memcpy kernels): the preceding kernel's category
                    last = "misc"
                    for i, m in enumerate(mapped):
                        if m is None:
                            mapped[i] = last
                            graph_only[(last, short(names[i]))] += 1
                        last = mapped[i]
                    forwards.append([(m, k["name"], k["ts"], k["dur"]) for m, k in zip(mapped, g)])
                else:
                    mismatched += 1
                    forwards.append([((names_major[k["name"]].most_common(1)[0][0] if names_major[k["name"]] else
                                       "unmapped"), k["name"], k["ts"], k["dur"]) for k in g])
            if mismatched:
                res["notes"].append(f"rank {r}: {mismatched}/{len(reps)} replays mapped by name "
                                    f"(replay kernels {[len(g) for g in reps]}, twin {len(tw)})")
            if graph_only and r == 0:
                res["notes"].append("graph-only kernels (per 1 forward), given the preceding kernel's category: " +
                                    ", ".join(f"{c}/{k} x{n // max(1, len(reps))}" for (c, k), n in graph_only.items()))
            gaps = [(reps[i + 1][0]["ts"] - (reps[i][-1]["ts"] + reps[i][-1]["dur"])) / 1e3
                    for i in range(len(reps) - 1)]
            if r == 0:
                res["twin_span_ms"] = ((tw[-1][2] + tw[-1][3] - tw[0][2]) / 1e3) if tw else None
        if not forwards or not forwards[0]:
            continue
        n = len(forwards)
        cats = collections.Counter()
        kern = collections.Counter()
        kcount = collections.Counter()
        spans, busys = [], []
        nccl = []
        for f in forwards:
            spans.append((f[-1][2] + f[-1][3] - f[0][2]) / 1e3)
            busys.append(busy([(ts, ts + d) for _, _, ts, d in f]) / 1e3)
            seq = []
            for cat, name, ts, d in f:
                cats[cat] += d / 1e3 / n
                kern[(cat, short(name))] += d / 1e3 / n
                kcount[(cat, short(name))] += 1 / n
                if "nccl" in name.lower():
                    seq.append((cat, d / 1e3))
            nccl.append(seq)
        nccl_by_rank[r] = nccl
        res["ranks"][r] = {"cats": dict(cats), "span_ms": sum(spans) / n, "busy_ms": sum(busys) / n,
                           "kernels_per_forward": len(forwards[0]), "inter_round_gap_ms": gaps,
                           "top": sorted(([c, k, ms, kcount[(c, k)]] for (c, k), ms in kern.items()),
                                         key=lambda x: -x[2])[:25]}
    # NCCL: per collective, min over ranks = transfer estimate
    if nccl_by_rank and len({len(v) for v in nccl_by_rank.values()}) == 1:
        rs = sorted(nccl_by_rank)
        nf = len(nccl_by_rank[rs[0]])
        if all(len(nccl_by_rank[r][j]) == len(nccl_by_rank[rs[0]][j]) for r in rs for j in range(nf)):
            transfer = collections.Counter()
            count = collections.Counter()
            for j in range(nf):
                for i, (cat, _) in enumerate(nccl_by_rank[rs[0]][j]):
                    transfer[cat] += min(nccl_by_rank[r][j][i][1] for r in rs) / nf
                    count[cat] += 1 / nf
            res["nccl_transfer_ms"] = dict(transfer)
            res["nccl_count"] = dict(count)
    comm = collections.defaultdict(lambda: [0, 0])
    cpath = os.path.join(run, f"comm-{case}{'' if is_prefill else '-eager'}-r0.json")
    if os.path.exists(cpath):
        for c in json.load(open(cpath)):
            cat = classify(c["stack"], "nccl")
            key = f"{cat} {c['kind']} {c['dtype'].replace('torch.', '')}"
            comm[key][0] += 1
            comm[key][1] += c["bytes"]
    res["comm_calls"] = {k: v for k, v in sorted(comm.items())}
    return res


def table(res):
    lines = []
    ranks = res["ranks"]
    if not ranks:
        return f"## {res['case']}: no traces\n"
    r0 = ranks.get(0) or next(iter(ranks.values()))
    cats = [c for c in ORDER if any(c in v["cats"] for v in ranks.values())]
    cats += sorted({c for v in ranks.values() for c in v["cats"]} - set(cats))
    total0 = sum(r0["cats"].values())
    lines.append(f"## {res['case']}  (rank0: span {r0['span_ms']:.2f} ms, kernel busy {r0['busy_ms']:.2f} ms, "
                 f"device idle {r0['span_ms'] - r0['busy_ms']:.2f} ms, {r0['kernels_per_forward']} kernels/forward)")
    if r0["inter_round_gap_ms"]:
        g = r0["inter_round_gap_ms"]
        lines.append(f"inter-round GPU gap rank0 (host: wait, sampling, announce, stage): "
                     f"median {sorted(g)[len(g) // 2]:.2f} ms, all {[round(x, 2) for x in g]}")
    if res.get("twin_span_ms"):
        lines.append(f"eager twin span (annotated, rank0): {res['twin_span_ms']:.2f} ms")
    lines.append("| category | rank0 ms | rank0 % | max-rank ms | mean ms | nccl transfer (min-rank) ms | calls |")
    lines.append("|---|---|---|---|---|---|---|")
    tr = res.get("nccl_transfer_ms", {})
    cnt = res.get("nccl_count", {})
    for c in cats:
        vals = [v["cats"].get(c, 0.0) for v in ranks.values()]
        x0 = r0["cats"].get(c, 0.0)
        lines.append(f"| {c} | {x0:.3f} | {100 * x0 / total0:.1f} | {max(vals):.3f} | {sum(vals) / len(vals):.3f} | "
                     f"{tr.get(c, ''):{'.3f' if c in tr else ''}} | {cnt.get(c, ''):{'.0f' if c in cnt else ''}} |")
    lines.append(f"| total kernel time | {total0:.3f} | 100 | | | | |")
    if res["comm_calls"]:
        lines.append("collectives per forward (host log of the annotated eager run, rank0): " + "; ".join(
            f"{k}: {n} x, {b / n / 1024:.1f} KiB avg, {b / 2**20:.2f} MiB" for k, (n, b) in res["comm_calls"].items()))
    lines.append("top kernels rank0 (category, kernel, ms/forward, launches/forward):")
    for c, k, ms, n in r0["top"][:15]:
        lines.append(f"  {c:12s} {ms:8.3f} ms  x{n:5.0f}  {k}")
    for note in res["notes"]:
        lines.append("NOTE " + note)
    return "\n".join(lines) + "\n"


def main():
    run = sys.argv[1]
    files = glob.glob(os.path.join(run, "trace-*-r*.json"))
    cases = sorted({re.sub(r"-r\d+\.json$", "", os.path.basename(f))[6:] for f in files})
    cases = [c for c in cases if not c.endswith("-eager")]
    ranks = sorted({int(re.search(r"-r(\d+)\.json$", f).group(1)) for f in files})
    results = [summarize_case(run, c, ranks) for c in cases]
    summary = os.path.join(run, "summary.json")
    if os.path.exists(summary):
        s = json.load(open(summary))
        print("# phases", json.dumps({k: v for k, v in s.items() if k != "rounds"}))
        for name, r in s.get("rounds", {}).items():
            print(f"# {name}: median round {r['median_round_ms']:.2f} ms, {r['decode_tok_s']:.1f} tok/s, "
                  f"eager twin {r['eager_twin_s'] * 1e3:.1f} ms")
    for res in results:
        print(table(res))
    if "--json" in sys.argv:
        json.dump(results, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
