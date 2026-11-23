"""DeepSeek-V4.1 SM70 per-kernel profile driver (development tool; one model load serves every case).

usage: ds41_profile.py MODEL_DIR OUT_DIR [--cases 16384x32,16384x8,...] [--corpus TEXT_FILE]
       [--kv-dtype rotorquant-iso64-norm4-h64] [--yarn-factor 2]

Eight rank processes (CUDA_VISIBLE_DEVICES=r, as tools/launch_cuda_tp.py and harness/m2.py) build DS41Engine
(serve=False, 32 lanes, 65536 context, RotorQuant Q4 KV, YaRN 2, graphs on, host prefix tier off).
Rank 0 drives through the control pipes; profiler / clone / eager-twin control frames use operation codes >= 900
handled by a driver-side ``_run`` wrapper on every rank (no production change).

Phases (rank 0 ``summary.json`` records each one's wall time):
  1. warm prefill: one unprofiled 8192-row chunk on slot 1 (allocator / cuBLAS first-use costs);
  2. prefill profiles (eager, annotated): slot 0 chunk [0, 8192) ("pf8k") then [8192, 16384) ("pf16k", the second
     chunk of a 16K prompt, last row's logits gathered) of real corpus text;
  3. slots 1..31 take copies of slot 0's KV pages and window rings (same tokens); before each decode case every
     stream's Engram n-gram history / C2 partial is slot 0's and its committed position the case's start
     (the trap-checked continuation invariant);
  4. per decode case (context x streams): 3 warm rounds, 10 timed rounds (no profiler), 5 profiled rounds of the
     startup graph replays, then one profiled eager twin (``Shape.body`` of the same bucket: the captured code,
     annotated) and one timed unannotated eager twin. A round is the Decoder's greedy round: replay with the
     compact packet (F_COMPACT), one wait, ``greedy_tokens``, next launch.
Each profile writes trace-<case>-r<rank>.json (Chrome trace) and comm-<case>-r<rank>.json (collective calls of
annotated phases: label stack, kind, dtype, elements). tools/ds41_profile_report.py turns them into tables.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import subprocess
import sys
import time

OP_PROF_START, OP_PROF_STOP, OP_CLONE, OP_TWIN, OP_SETPOS = 900, 901, 902, 903, 904

ANNOTATE = [False]          # wrappers annotate (record_function + collective log) only inside annotated profiles
STACK: list[str] = []
COMM: list = []


def _wrap(owner, name, label):
    import torch
    raw = owner.__dict__.get(name) if isinstance(owner, type) else None
    original = raw if raw is not None else getattr(owner, name)
    if isinstance(original, staticmethod):
        original = original.__func__
        static = True
    else:
        static = False

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if not ANNOTATE[0]:
            return original(*args, **kwargs)
        STACK.append(label)
        try:
            with torch.profiler.record_function("ds41:" + label):
                return original(*args, **kwargs)
        finally:
            STACK.pop()

    setattr(owner, name, staticmethod(wrapper) if static else wrapper)


def _comm_logger(kind, fn):
    def wrapper(tensor, *rest, **kwargs):
        if ANNOTATE[0]:
            COMM.append({"stack": list(STACK), "kind": kind, "dtype": str(tensor.dtype), "numel": tensor.numel(),
                         "bytes": tensor.numel() * tensor.element_size()})
            STACK.append("AG")
            try:
                with __import__("torch").profiler.record_function("ds41:AG"):
                    return fn(tensor, *rest, **kwargs)
            finally:
                STACK.pop()
        return fn(tensor, *rest, **kwargs)
    return wrapper


def install_wrappers():
    from tensorfold.families.deepseek_v4.cuda import target as T, engram as E
    from tensorfold.families.deepseek_v4.cuda.kernels import (mu_packed, quant_rope, mhc, mhc_fused, router,
                                                              mu_state, norm, engram as kengram, attention_ops)
    from tensorfold.cuda.sm70 import onecat_ds41, ds41_dense, ds41_decode_moe, ds41_lattice_moe
    for name, label in [("begin", "begin"), ("end", "head"), ("attention", "attn"), ("_compress", "compress"),
                        ("_select", "indexer"), ("_attend", "attn.core"), ("moe", "moe"),
                        ("engram_layer", "engram"), ("embed", "embed"), ("_norm", "norm"), ("_linear", "linear")]:
        _wrap(T.Target, name, label)
    for name, label in [("insert_rows", "kv_insert"), ("insert_ring", "ring_insert"),
                        ("select_positions", "idx.select"), ("index_scores", "idx.scores"), ("_topk", "idx.topk"),
                        ("select_compressed_positions", "idx.prefill"), ("unpack_request_index", "idx.unpack"),
                        ("gather_attention", "attn.gather"), ("_attend_fused", "attn.fused")]:
        _wrap(mu_packed, name, label)
    _wrap(quant_rope, "apply_rope", "rope")
    _wrap(quant_rope, "attention_qdq", "qdq")
    for name in ("hc_mixes", "collapse_hc", "expand_hc"):
        _wrap(mhc, name, "hc")
    _wrap(mhc, "fp32_projection", "router.proj")
    _wrap(mhc_fused, "hc_transition", "hc")
    _wrap(router, "route_topk", "router.topk")
    _wrap(norm, "short_rms_norm", "norm")
    _wrap(onecat_ds41, "v41_experts", "routed")
    _wrap(onecat_ds41, "shared_swiglu", "shared.act")
    _wrap(ds41_decode_moe.Workspace, "experts", "routed")
    _wrap(ds41_lattice_moe, "experts", "routed")
    _wrap(ds41_dense.Group, "__call__", "linear")
    _wrap(E.HostEngram, "lookup", "engram.gather")
    _wrap(kengram, "apply_engram_gate", "engram.gate")
    _wrap(mu_state.MuEngramHashState, "__call__", "hash")
    _wrap(mu_state.MuEngramHashState, "commit_all", "hash")
    _wrap(mu_state.MuC2State, "__call__", "c2")
    _wrap(mu_state.MuC2State, "commit_all", "c2")
    _wrap(T.Collectives, "_all_reduce", "AR")
    _wrap(T.Collectives, "_reduce_half", "comm")
    _wrap(T.Collectives, "gather", "AGcat")
    original = T.Collectives._all_reduce

    def logged(self, tensor):
        if ANNOTATE[0]:
            COMM.append({"stack": list(STACK), "kind": "all_reduce", "dtype": str(tensor.dtype),
                         "numel": tensor.numel(), "bytes": tensor.numel() * tensor.element_size()})
        return original(self, tensor)
    T.Collectives._all_reduce = logged
    del attention_ops


class Prof:
    def __init__(self, engine, out):
        self.e, self.out, self.p, self.case = engine, out, None, None

    def run(self, op, a, b, c, d, ids):
        import torch
        e = self.e
        if op == OP_PROF_START:
            self.case = CASES_BY_ID[a]
            ANNOTATE[0] = bool(b)
            COMM.clear()
            torch.cuda.synchronize()
            self.p = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            self.p.start()
            return None
        if op == OP_PROF_STOP:
            torch.cuda.synchronize()
            self.p.stop()
            ANNOTATE[0] = False
            self.p.export_chrome_trace(os.path.join(self.out, f"trace-{self.case}-r{e.rank}.json"))
            json.dump(COMM, open(os.path.join(self.out, f"comm-{self.case}-r{e.rank}.json"), "w"))
            COMM.clear()
            self.p = None
            return None
        if op == OP_CLONE:                  # slot a's first c tokens of KV pages and every window ring -> slot b
            pool = e.target.pool
            n = -(-c // 64)
            src = torch.tensor(pool.owned[a][:n], dtype=torch.long, device=e.target.device)
            dst = torch.tensor(pool.owned[b][:n], dtype=torch.long, device=e.target.device)
            pool.cache.index_copy_(0, dst, pool.cache.index_select(0, src))
            for ring in pool.rings:
                ring[b].copy_(ring[a])
            return None
        if op == OP_SETPOS:                 # slots [0, a) continue at position b: Engram n-gram history and C2
            t = e.target                    # partials copied from slot 0, committed positions set to b
            hist = t.hash_state.history
            for slot in range(a):
                hist[slot, :3] = hist[0, :3]
                hist[slot, 3] = b
                for state in t.c2.values():
                    state.partial[slot].copy_(state.partial[0])
                    state.committed[slot] = b
            return None
        if op == OP_TWIN:                   # eager Shape.body of bucket a for entries/ids in the payload
            n = 3 * c
            entries = [tuple(ids[i:i + 3]) for i in range(0, n, 3)]
            shape = e.graphs.shapes[(a, e.graphs.index_class(a, entries))]
            shape.stage(entries, ids[n:n + c])
            out = shape.body()
            return out[:c].clone() if e.rank == 0 else None
        raise RuntimeError(f"unknown profile op {op}")


CASES_BY_ID = {}


def case_names(spec):
    names = ["pf8k", "pf16k"]
    for item in spec:
        ctx, streams = item
        names += [f"d{ctx // 1024}k-c{streams}", f"d{ctx // 1024}k-c{streams}-eager"]
    return names


def rank_main(rank, model, out, port, spec, kv_dtype, yarn_factor):
    import torch
    from tensorfold.families.deepseek_v4.cuda import engine as EN
    install_wrappers()
    for i, name in enumerate(case_names(spec)):
        CASES_BY_ID[i] = name
    holder = {}
    original_run = EN.DS41Engine._run

    def run(self, op, a, b, c, d, ids):
        if op >= 900:
            return holder["prof"].run(op, a, b, c, d, ids)
        return original_run(self, op, a, b, c, d, ids)
    EN.DS41Engine._run = run
    t_start = time.perf_counter()
    e = EN.DS41Engine(model, rank=rank, tp=8, master="127.0.0.1", master_port=port, context=65536, parallel=32,
                      serve=False, prefix_host_bytes=0, kv_dtype=kv_dtype, yarn_factor=yarn_factor)
    holder["prof"] = Prof(e, out)
    e.comm.all_gather = _comm_logger("all_gather", e.comm.all_gather)   # Collectives.world is this object
    if rank:
        e.follow()
        e.close()
        return
    phases = {"kv_dtype": e.model_kv_dtype, "yarn_factor": e.yarn_factor,
              "engine_init_s": time.perf_counter() - t_start, "load_seconds": e.load_seconds,
              "graph_capture_s": e.graphs.seconds if e.graphs else None,
              "graph_bytes": e.graphs.bytes if e.graphs else None}
    print(f"[prof] engine up in {phases['engine_init_s']:.1f} s", flush=True)
    try:
        drive(e, out, spec, phases)
    finally:
        json.dump(phases, open(os.path.join(out, "summary.json"), "w"), indent=1)
        e.close()


def drive(e, out, spec, phases):
    import torch
    from tensorfold.families.deepseek_v4.cuda.engine import F_COMPACT
    from tensorfold.families.deepseek_v4.cuda.round_sampling import greedy_tokens
    from tensorfold.families.gemma4.sm70.ops import OP_ATTACH
    ids_all = json.load(open(os.path.join(out, "prompt-ids.json")))
    names = {v: k for k, v in CASES_BY_ID.items()}
    max_ctx = max(c for c, _ in spec) if spec else 16384
    max_streams = max(s for _, s in spec) if spec else 1
    span = max(max_ctx, 16384) + 64
    prompt = ids_all[:16384]

    def timed(label, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        phases[label] = time.perf_counter() - t0
        print(f"[prof] {label}: {phases[label]:.3f} s", flush=True)
        return result

    def profiled(name, annotate, fn):
        e.announce(OP_PROF_START, names[name], int(annotate))
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            torch.cuda.synchronize()
            phases[f"{name}_profiled_s"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            e.announce(OP_PROF_STOP)
            phases[f"{name}_export_s"] = time.perf_counter() - t1

    with torch.inference_mode():
        # 1. warm prefill (slot 1)
        e.announce(OP_ATTACH, 1, 8192 + 64)
        timed("warm_pf8k_s", lambda: e.step([(1, 0, 8192)], prompt[:8192], []))
        # 2. prefill profiles on slot 0
        e.announce(OP_ATTACH, 0, span)
        profiled("pf8k", True, lambda: e.step([(0, 0, 8192)], prompt[:8192], []))
        logits = profiled("pf16k", True, lambda: e.step([(0, 8192, 8192)], prompt[8192:16384], [8191]))
        first = int(logits[-1].argmax())
        # 3. KV copies
        t0 = time.perf_counter()
        for slot in range(1, max_streams):
            e.announce(OP_ATTACH, slot, span)
            e.announce(OP_CLONE, 0, slot, 16384)
        torch.cuda.synchronize()
        phases["clone_s"] = time.perf_counter() - t0
        # 4. decode cases
        rounds = {}
        for ctx, streams in spec:
            name = f"d{ctx // 1024}k-c{streams}"
            pos = [ctx] * streams
            e.announce(OP_SETPOS, streams, ctx)
            tokens = [first] * streams

            def one_round():                # the Decoder's greedy round (K9): compact packet, one host wait
                entries = [(s, pos[s], 1) for s in range(streams)]
                compact, _ = e.step(entries, tokens, list(range(streams)), F_COMPACT)
                done = torch.cuda.Event()
                done.record()
                done.synchronize()
                for g, (token, error) in enumerate(greedy_tokens(compact, streams)):
                    if error is not None:
                        raise error
                    tokens[g] = token
                    pos[g] += 1

            for _ in range(3):
                one_round()
            walls = []
            for _ in range(10):
                t0 = time.perf_counter()
                one_round()
                walls.append(time.perf_counter() - t0)

            def prof_rounds():
                w = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    one_round()
                    w.append(time.perf_counter() - t0)
                return w
            pwalls = profiled(name, False, prof_rounds)
            bucket = e.graphs.bucket(streams)

            def twin():
                entries = [(s, pos[s], 1) for s in range(streams)]
                payload = [v for x in entries for v in x] + list(tokens)
                out = e.announce(OP_TWIN, bucket, 0, streams, 0, payload)
                for g in range(streams):        # the twin committed this position (Engram/C2 trap invariant)
                    pos[g] += 1
                return out
            profiled(name + "-eager", True, twin)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            twin()
            torch.cuda.synchronize()
            rounds[name] = {"ctx": ctx, "streams": streams, "bucket": bucket, "round_wall_s": walls,
                            "profiled_round_wall_s": pwalls, "eager_twin_s": time.perf_counter() - t0,
                            "median_round_ms": sorted(walls)[len(walls) // 2] * 1e3,
                            "decode_tok_s": streams / (sorted(walls)[len(walls) // 2])}
            print(f"[prof] {name}: median round {rounds[name]['median_round_ms']:.2f} ms "
                  f"({rounds[name]['decode_tok_s']:.1f} tok/s), eager twin {rounds[name]['eager_twin_s']*1e3:.1f} ms",
                  flush=True)
            phases["rounds"] = rounds


def tokenize(model, corpus, count):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
    text = open(corpus, encoding="utf-8").read(400_000)
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) < count:
        raise ValueError("corpus slice too short")
    return ids[:count]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        _, _, rank, model, out, port, cases, kv_dtype, yarn_factor = sys.argv
        spec = [tuple(int(v) for v in x.split("x")) for x in cases.split(",") if x]
        rank_main(int(rank), model, out, int(port), spec, kv_dtype, float(yarn_factor))
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--cases", default="16384x32,16384x8,16384x1,4096x32,4096x8,4096x1")
    parser.add_argument("--corpus", default="/home/kklouzal/build-cache/matrix-ds41/corpus-0913d721-ds41.txt")
    parser.add_argument("--kv-dtype", choices=("rotorquant-iso64-norm4-h64", "bf16"), default="rotorquant-iso64-norm4-h64")
    parser.add_argument("--yarn-factor", type=float, default=2.0)
    args = parser.parse_args()
    model, out, cases, corpus = args.model, args.out, args.cases, args.corpus
    os.makedirs(out, exist_ok=True)
    json.dump(tokenize(model, corpus, 16384), open(os.path.join(out, "prompt-ids.json"), "w"))
    port = 29700
    procs = []
    t0 = time.perf_counter()
    for r in range(8):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(r))
        cmd = [sys.executable, __file__, "--rank", str(r), model, out, str(port), cases,
               args.kv_dtype, str(args.yarn_factor)]
        node = "0" if r < 4 else "1"
        if subprocess.run(["which", "numactl"], capture_output=True).returncode == 0:
            cmd = (["numactl", f"--cpunodebind={node}", f"--membind={node}"] if r else
                   ["numactl", f"--cpunodebind={node}"]) + cmd
        procs.append(subprocess.Popen(cmd, env=env))
    while any(p.poll() is None for p in procs):
        if any(p.returncode not in (None, 0) for p in procs):
            time.sleep(20)
            for p in procs:
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(2)
    codes = [p.wait() for p in procs]
    print(f"[prof] exit codes {codes}, wall {time.perf_counter() - t0:.1f} s", flush=True)
    sys.exit(max(codes))
