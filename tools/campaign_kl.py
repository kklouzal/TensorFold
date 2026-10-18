"""Teacher-forced next-token distributions of a TensorFold CUDA configuration, and KL / greedy agreement between two.

The HTTP server returns log-probabilities only for engines with ``supports_logprobs`` (Flash Next) and at most 20 per
token, so this tool measures in process: it runs ``tensorfold serve`` with the exact flags under test, intercepts the
CLI's final ``serve`` call (the engine and App are built exactly as in production, then no HTTP request is served),
and replaces the CUDA target sampler while it teacher-forces fixed token sequences through the serial path
(``draft=False``). At each forced position it records the full-vocabulary log-softmax of the logits the engine
handed its sampler: the top-K ids and log-probabilities, the forced token's log-probability, the entropy and, given a
reference dump, the log-probabilities at the reference's top-K ids.

  dump     [--build] --eval EVAL.json --out OUT.npz [--reference REF.npz] -- SERVE_ARGS...
  compare  REF.npz CAND.npz [--out RESULT.json]

``dump --build`` first writes EVAL.json when it is missing: chat prompts rendered by the served template (fixed short
prompts plus corpus-backed long ones) and each prompt's continuation sampled by this configuration (model-card
sampling, fixed seeds). Run it once with the reference configuration; every candidate dumps the same EVAL.json.

Precision limits (stated with every comparison): KL(ref || cand) is computed exactly on the reference's top-K ids and
with the remaining mass of both distributions merged into one tail bucket. That coarse-grained KL is a lower bound of
the true KL (data processing inequality); the reported tail mass p_T bounds what it cannot see. Log-probabilities are
computed in float32 from the engine's own logits dtype (recorded), so they carry that dtype's rounding. Each tail
mass is a float32 log-sum-exp over the ids outside the reference's top-K, taken in the dump: one minus a float32 sum
of the top-K probabilities would cancel to rounding noise (about 1e-7) when the tail is that small, and a noise-sized
or clipped tail can push the bound far above the true KL. The bound therefore holds to float32 log-softmax rounding,
about 1e-7 nats per position. Only the serial (one-row) path is measured; drafted verification is covered by the
drafted-vs-serial identity gate. Positions are conditioned on the reference continuation (teacher forcing), not on
each configuration's own samples.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROMPTS = (
    "Write a thread-safe LRU cache in Python with unit tests.",
    "Explain how a sliding-window rate limiter works and when to prefer a token bucket.",
    "Compute the expected queue length of an M/M/4 queue at 500 requests/s with 3 ms mean service time.",
    "Write a shell pipeline that lists the five largest files in a folder as a Markdown table.",
    "Summarize the causes of the 2008 financial crisis in five bullet points.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Go HTTP handler that streams server-sent events and explain its error handling.",
    "Explain the CAP theorem with a concrete example for each trade-off.",
    "Translate this to French and explain two idioms: 'It is raining cats and dogs; break a leg.'",
    "Write a SQL query that finds the top three customers by revenue per month, with window functions.",
    "Describe how photosynthesis works to a ten-year-old.",
    "Review this Python for bugs: def avg(xs): return sum(xs) / len(xs) if xs else 0 / 0",
)
LONG_ASK = "Summarize the document above file by file, noting anything unusual."


# ---------------------------------------------------------------------------------------------------------------
# in-process capture


class Capture:
    """The CUDA sampler stand-in: records each forced position's distribution and returns the forced tokens."""

    def __init__(self, torch, top_k: int):
        self.torch, self.top_k = torch, top_k
        self.active = False

    def begin(self, prompt_len: int, forced: list[int], reference_ids=None) -> None:
        self.prompt_len, self.forced, self.reference_ids = prompt_len, forced, reference_ids
        self.next, self.rows, self.dtype = 0, [], None
        self.active = True

    def __call__(self, logits, positions, sampling, *args, **kwargs):
        if not self.active:
            raise RuntimeError("target sampler called outside a teacher-forced pass")
        torch = self.torch
        out = []
        self.dtype = str(logits.dtype)
        for row, position in enumerate(positions):
            index = int(position) - self.prompt_len
            if index != self.next or not 0 <= index < len(self.forced):
                raise RuntimeError(f"sampler position {position} is not the next forced position "
                                   f"{self.prompt_len + self.next} (rows {len(positions)}): uncovered engine path")
            lp = torch.log_softmax(logits[row].float(), dim=-1)
            values, ids = torch.topk(lp, self.top_k)
            entry = {"ids": ids.to(torch.int32).cpu(), "lp": values.cpu(), "tail_lp": self._tail(lp, ids),
                     "forced": float(lp[self.forced[index]]),
                     "entropy": float(-(lp.exp() * lp).sum())}
            if self.reference_ids is not None:
                at = torch.as_tensor(self.reference_ids[index], device=lp.device, dtype=torch.int64)
                entry["at_ref"] = lp.index_select(0, at).cpu()
                entry["at_ref_tail_lp"] = self._tail(lp, at)
            self.rows.append(entry)
            out.append(int(self.forced[index]))
            self.next += 1
        return out

    def _tail(self, lp, ids) -> float:
        """log of the probability mass outside ``ids`` (log-sum-exp, no cancellation; -inf when ``ids`` is the
        whole vocabulary)."""

        outside = self.torch.ones(lp.shape[0], dtype=self.torch.bool, device=lp.device)
        outside[ids.to(self.torch.int64)] = False
        return float(self.torch.logsumexp(lp.masked_fill(~outside, float("-inf")), 0))

    def end(self) -> list[dict]:
        self.active = False
        if self.next != len(self.forced):
            raise RuntimeError(f"teacher-forced pass stopped at {self.next} of {len(self.forced)} positions")
        return self.rows


@contextlib.contextmanager
def sampler_replaced(replacement):
    """Point every loaded TensorFold reference to the CUDA ``sample_rows`` at ``replacement`` for the block (this
    tool's capture, campaign_trace's marker), then restore the original."""

    import tensorfold.cuda.sampling as sampling

    original = sampling.sample_rows
    modules = [module for name, module in list(sys.modules.items())
               if name.startswith("tensorfold") and getattr(module, "sample_rows", None) is original]
    for module in modules:
        module.sample_rows = replacement
    try:
        yield
    finally:
        for module in modules:
            module.sample_rows = original


def build_eval(app, args) -> dict:
    """Prompts rendered by the served template and continuations sampled by this configuration."""

    corpus = Path(args.corpus).read_text(encoding="utf-8") if args.corpus else None
    items = [{"name": f"short-{i}", "messages": [{"role": "user", "content": text}]} for i, text in enumerate(PROMPTS)]

    def long_messages(chars: int) -> list[dict]:
        return [{"role": "user", "content": "<document>\n" + corpus[:chars] + "\n</document>\n\n" + LONG_ASK}]

    for depth in [int(d) for d in args.long_depths.split(",") if d] if corpus else []:
        lo, hi = 0, min(len(corpus), depth * 8)
        while hi - lo > 64:                          # bisection on characters: the largest document within depth
            mid = (lo + hi) // 2
            body = {"messages": long_messages(mid), "chat_template_kwargs": {"enable_thinking": False}}
            lo, hi = (mid, hi) if len(app.prepare(body, True).prompt) <= depth else (lo, mid)
        items.append({"name": f"long-{depth}", "messages": long_messages(lo)})
    engine = app.engine
    for k, item in enumerate(items):
        body = {"messages": item["messages"], "chat_template_kwargs": {"enable_thinking": False},
                **args.card, "seed": 7000 + k}
        prepared = app.prepare(body, True)
        out: list[int] = []
        engine.generate(prepared.prompt, args.answer_tokens, prepared.sampling, lambda new: out.extend(new) or False)
        item.update(prompt=list(prepared.prompt), answer=out[:args.answer_tokens], sampling=body)
        print(json.dumps({"item": item["name"], "prompt": len(item["prompt"]), "answer": len(item["answer"])}),
              flush=True)
    return {"version": 1, "created_unix_s": int(time.time()), "items": items}


def dump(app, args) -> None:
    import numpy as np
    import torch

    eval_path = Path(args.eval)
    if not eval_path.exists():
        if not args.build:
            raise SystemExit(f"{eval_path} does not exist (pass --build with the reference configuration)")
        eval_path.write_text(json.dumps(build_eval(app, args)))
    data = json.loads(eval_path.read_text())
    reference = np.load(args.reference) if args.reference else None
    capture = Capture(torch, args.top_k)
    rows, offsets, started = [], [], time.perf_counter()
    with sampler_replaced(capture):
        position = 0
        for item in data["items"]:
            n = len(item["answer"])
            ref_ids = reference["ids"][position:position + n] if reference is not None else None
            if ref_ids is not None and len(ref_ids) != n:
                raise ValueError("reference dump does not match this eval set")
            capture.begin(len(item["prompt"]), item["answer"], ref_ids)
            app.engine.generate(list(item["prompt"]), n, None, lambda new: False, draft=False, stop_eos=False)
            rows.extend(capture.end())
            offsets.append(position)
            position += n
            print(json.dumps({"item": item["name"], "positions": n,
                              "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    arrays = {"ids": np.stack([r["ids"].numpy() for r in rows]), "lp": np.stack([r["lp"].numpy() for r in rows]),
              "tail_lp": np.array([r["tail_lp"] for r in rows], dtype=np.float64),
              "forced": np.array([r["forced"] for r in rows], dtype=np.float64),
              "entropy": np.array([r["entropy"] for r in rows], dtype=np.float64),
              "offsets": np.array(offsets, dtype=np.int64)}
    if reference is not None:
        arrays["at_ref"] = np.stack([r["at_ref"].numpy() for r in rows])
        arrays["at_ref_tail_lp"] = np.array([r["at_ref_tail_lp"] for r in rows], dtype=np.float64)
    meta = {"eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(), "serve_args": args.serve,
            "top_k": args.top_k, "logits_dtype": capture.dtype, "names": [it["name"] for it in data["items"]],
            "reference": args.reference, "source_revision": os.environ.get("TENSORFOLD_SOURCE_REVISION"),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    np.savez(args.out, meta=json.dumps(meta), **arrays)
    print(json.dumps({"out": args.out, "positions": len(rows), **{k: meta[k] for k in ("logits_dtype", "top_k")}}))


def with_app(serve_args: list[str], port: int, work) -> None:
    """Build the engine and App through the production CLI (``tensorfold serve`` with these flags), then call
    ``work(app)`` in place of serving HTTP; the CLI closes the engine afterwards. The CLI binds its socket on
    127.0.0.1:``port`` before serving, so the port must be free; no request is accepted on it."""

    import tensorfold.cuda.server as server
    from tensorfold import cli

    argv = list(serve_args)
    for flag, value in (("--host", "127.0.0.1"), ("--port", str(port))):
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv += [flag, value]
    done = []

    def hooked(app, host, port, server=None):
        work(app)
        done.append(True)

    server.serve = hooked
    cli.main(argv)
    if not done:
        raise SystemExit("the serve hook never ran")


# ---------------------------------------------------------------------------------------------------------------
# comparison (numpy only)


def compare(ref_path: str, cand_path: str) -> dict:
    import numpy as np

    ref, cand = np.load(ref_path), np.load(cand_path)
    ref_meta, cand_meta = json.loads(str(ref["meta"])), json.loads(str(cand["meta"]))
    if ref_meta["eval_sha256"] != cand_meta["eval_sha256"]:
        raise ValueError("the dumps were made from different eval sets")
    if "at_ref" not in cand.files:
        raise ValueError("the candidate dump needs --reference (log-probabilities at the reference's top-K ids)")
    if "tail_lp" not in ref.files or "at_ref_tail_lp" not in cand.files:
        raise ValueError("a dump without log-sum-exp tail masses (made before they were recorded): dump it again")
    lp_p = ref["lp"].astype(np.float64)
    lp_q = cand["at_ref"].astype(np.float64)
    if lp_p.shape != lp_q.shape:
        raise ValueError("reference and candidate positions differ")
    p = np.exp(lp_p)
    kl_top = (p * (lp_p - lp_q)).sum(-1)
    tail_p, tail_q = ref["tail_lp"].astype(np.float64), cand["at_ref_tail_lp"].astype(np.float64)
    p_tail = np.exp(tail_p)
    # an empty tail (top-K = vocabulary) contributes nothing; log masses never subtract -inf from -inf
    kl = kl_top + p_tail * np.subtract(tail_p, tail_q, out=np.zeros_like(tail_p), where=p_tail > 0)
    agree = ref["ids"][:, 0] == cand["ids"][:, 0]
    in_top5 = (cand["ids"][:, :5] == ref["ids"][:, :1]).any(-1)
    names, offsets = ref_meta["names"], list(ref["offsets"]) + [len(kl)]
    per_item = {name: {"positions": int(offsets[i + 1] - offsets[i]),
                       "kl_mean": float(kl[offsets[i]:offsets[i + 1]].mean()),
                       "top1_agreement": float(agree[offsets[i]:offsets[i + 1]].mean())}
                for i, name in enumerate(names)}
    return {"reference": ref_path, "candidate": cand_path, "positions": int(len(kl)), "top_k": ref_meta["top_k"],
            "logits_dtype": {"reference": ref_meta["logits_dtype"], "candidate": cand_meta["logits_dtype"]},
            "kl": {"mean": float(kl.mean()), "median": float(np.median(kl)), "p95": float(np.quantile(kl, 0.95)),
                   "p99": float(np.quantile(kl, 0.99)), "max": float(kl.max()),
                   "over_0.1": int((kl > 0.1).sum()), "over_1": int((kl > 1).sum()), "kind": "lower bound"},
            "tail_mass_ref": {"mean": float(p_tail.mean()), "max": float(p_tail.max())},
            "top1_agreement": float(agree.mean()), "ref_top1_in_cand_top5": float(in_top5.mean()),
            "nll_forced": {"reference": float(-ref["forced"].mean()), "candidate": float(-cand["forced"].mean())},
            "entropy_mean": {"reference": float(ref["entropy"].mean()), "candidate": float(cand["entropy"].mean())},
            "identical_logprobs": bool(np.array_equal(ref["lp"], cand["lp"]) and np.array_equal(ref["ids"], cand["ids"])
                                       and np.array_equal(ref["tail_lp"], cand["tail_lp"])),
            "per_item": per_item,
            "method": "KL(ref||cand) on the reference top-K ids plus one merged tail bucket (float32 log-sum-exp "
                      "masses): a lower bound of the true KL to float32 log-softmax rounding (~1e-7 nats); serial "
                      "path; teacher-forced on the reference continuation"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--eval", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--reference")
    d.add_argument("--build", action="store_true", help="write --eval first when it is missing")
    d.add_argument("--corpus", help="corpus file for the long items (with --build)")
    d.add_argument("--long-depths", default="32768", help="long item prompt lengths (with --build)")
    d.add_argument("--answer-tokens", type=int, default=256)
    d.add_argument("--profile", help="bench profile: --build continuations use its non-thinking card sampling")
    d.add_argument("--card", type=json.loads, help="JSON sampling fields for --build (overrides --profile)")
    d.add_argument("--top-k", type=int, default=1024)
    d.add_argument("--port", type=int, default=8899, help="the CLI binds its HTTP socket here (nothing is served)")
    d.add_argument("serve", nargs=argparse.REMAINDER, help="-- serve MODEL [production flags]")
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--out")
    args = ap.parse_args()
    if args.command == "dump":
        if args.serve and args.serve[0] == "--":
            args.serve = args.serve[1:]
        if not args.serve or args.serve[0] != "serve":
            ap.error("dump needs -- serve MODEL [flags]")
        if args.card is None:
            args.card = (json.loads(Path(args.profile).read_text())["card_sampling"]["nothink"] if args.profile
                         else {})
        with_app(args.serve, args.port, lambda app: dump(app, args))
    else:
        result = compare(args.reference, args.candidate)
        text = json.dumps(result, indent=1)
        if args.out:
            Path(args.out).write_text(text + "\n")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
