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
  subset   GOLDEN.npz EVAL.json OUT_DIR [--max-prompt-tokens N]   (a golden reference cut to the shorter items)
  calibrate GOLDEN.npz BASE.npz NAME=VARIANT.npz ... [--out CALIBRATION.json]   (non-inferiority margins)
  gate     GOLDEN.npz OLD.npz NEW.npz [--calibration CALIBRATION.json] [--out RESULT.json]
           (exit 0 pass, 1 fail, 3 incomplete or uncalibrated)

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
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__:
    from .campaign_bench import LONG_PREFILL_TOKENS
else:
    from campaign_bench import LONG_PREFILL_TOKENS

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
CALIBRATION_VARIANTS = (4, 6)          # reorder-only variants per calibration (campaign decision): min, max


# ---------------------------------------------------------------------------------------------------------------
# in-process capture


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_stats(torch, logits, top_k: int, forced: int, reference_ids=None) -> dict:
    """One position's record from its logits row: float32 log-softmax top-K ids and log-probabilities, the log of
    the mass outside them, the forced token's log-probability, the entropy and, given the reference's top-K ids, the
    log-probabilities there and the log of the mass outside them (shared by the engine capture and campaign_golden)."""

    lp = torch.log_softmax(logits.float(), dim=-1)
    values, ids = torch.topk(lp, top_k)
    entry = {"ids": ids.to(torch.int32).cpu(), "lp": values.cpu(), "tail_lp": _tail(torch, lp, ids),
             "forced": float(lp[forced]), "entropy": float(-(lp.exp() * lp).sum())}
    if reference_ids is not None:
        at = torch.as_tensor(reference_ids, device=lp.device, dtype=torch.int64)
        entry["at_ref"] = lp.index_select(0, at).cpu()
        entry["at_ref_tail_lp"] = _tail(torch, lp, at)
    return entry


def _tail(torch, lp, ids) -> float:
    """log of the probability mass outside ``ids`` (log-sum-exp, no cancellation; -inf when ``ids`` is the whole
    vocabulary)."""

    outside = torch.ones(lp.shape[0], dtype=torch.bool, device=lp.device)
    outside[ids.to(torch.int64)] = False
    return float(torch.logsumexp(lp.masked_fill(~outside, float("-inf")), 0))


def save_dump(path: str, rows: list[dict], offsets: list[int], meta: dict) -> None:
    """The dump format every comparison reads (engine dumps and campaign_golden references)."""

    import numpy as np

    arrays = {"ids": np.stack([r["ids"].numpy() for r in rows]), "lp": np.stack([r["lp"].numpy() for r in rows]),
              "tail_lp": np.array([r["tail_lp"] for r in rows], dtype=np.float64),
              "forced": np.array([r["forced"] for r in rows], dtype=np.float64),
              "entropy": np.array([r["entropy"] for r in rows], dtype=np.float64),
              "offsets": np.array(offsets, dtype=np.int64)}
    if "at_ref" in rows[0]:
        arrays["at_ref"] = np.stack([r["at_ref"].numpy() for r in rows])
        arrays["at_ref_tail_lp"] = np.array([r["at_ref_tail_lp"] for r in rows], dtype=np.float64)
    np.savez(path, meta=json.dumps(meta), **arrays)


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
            self.rows.append(row_stats(torch, logits[row], self.top_k, self.forced[index],
                                       None if self.reference_ids is None else self.reference_ids[index]))
            out.append(int(self.forced[index]))
            self.next += 1
        return out

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

    for depth in long_depths(args) if corpus else []:      # within the long-prefill hold: eval_set checked them
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


def long_depths(args) -> list[int]:
    """The prompt lengths of the long items ``--build`` writes (none without --corpus)."""

    return [int(d) for d in args.long_depths.split(",") if d] if args.corpus else []


def read_eval(path: Path) -> tuple[dict, str]:
    """An eval set and the sha256 of the very bytes it was parsed from (the dump records what it teacher-forced)."""

    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def eval_set(args) -> tuple[dict, str] | None:
    """The eval set ``dump`` teacher-forces (read_eval), or None when ``--build`` is to write it; checked before the
    engine loads, so a refusal costs no model load. During the campaign's long-prefill hold (lifted by
    --allow-long-prefill) an eval set with items above LONG_PREFILL_TOKENS prompt tokens, or a build of long items
    above it, is refused."""

    path = Path(args.eval)
    if not path.exists():
        if not args.build:
            raise SystemExit(f"{path} does not exist (pass --build with the reference configuration)")
        depths = long_depths(args)
        if any(d > LONG_PREFILL_TOKENS for d in depths) and not args.allow_long_prefill:
            raise SystemExit(f"long items {depths} exceed {LONG_PREFILL_TOKENS} tokens: held by the campaign's "
                             "long-prefill hold (pass --allow-long-prefill once it is lifted)")
        return None
    data, digest = read_eval(path)
    long_items = [it["name"] for it in data["items"] if len(it["prompt"]) > LONG_PREFILL_TOKENS]
    if long_items and not args.allow_long_prefill:
        raise SystemExit(f"eval items {long_items} exceed {LONG_PREFILL_TOKENS} prompt tokens: held by the campaign's "
                         "long-prefill hold (use a subset eval set, campaign_kl.py subset, or --allow-long-prefill "
                         "once the hold is lifted)")
    return data, digest


def dump(app, args, loaded: tuple[dict, str] | None) -> None:
    """Teacher-force eval_set's ``loaded`` set (None: build EVAL.json with this configuration first) and save it."""

    import numpy as np
    import torch

    eval_path = Path(args.eval)
    if loaded is None:
        eval_path.write_text(json.dumps(build_eval(app, args)))
        loaded = read_eval(eval_path)
    data, eval_sha256 = loaded
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
    meta = {"eval_sha256": eval_sha256, "serve_args": args.serve,
            "top_k": args.top_k, "logits_dtype": capture.dtype, "names": [it["name"] for it in data["items"]],
            "reference": args.reference,
            "reference_sha256": file_sha256(args.reference) if args.reference else None,
            "source_revision": os.environ.get("TENSORFOLD_SOURCE_REVISION"),
            "tool_sha256": file_sha256(__file__)}
    save_dump(args.out, rows, offsets, meta)
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


def _load_pair(ref_path: str, cand_path: str):
    import numpy as np

    ref, cand = np.load(ref_path), np.load(cand_path)
    ref_meta, cand_meta = json.loads(str(ref["meta"])), json.loads(str(cand["meta"]))
    if ref_meta["eval_sha256"] != cand_meta["eval_sha256"]:
        raise ValueError("the dumps were made from different eval sets")
    if "at_ref" not in cand.files:
        raise ValueError("the candidate dump needs --reference (log-probabilities at the reference's top-K ids)")
    if "tail_lp" not in ref.files or "at_ref_tail_lp" not in cand.files:
        raise ValueError("a dump without log-sum-exp tail masses (made before they were recorded): dump it again")
    if ref["lp"].shape != cand["at_ref"].shape:
        raise ValueError("reference and candidate positions differ")
    if cand_meta.get("reference_sha256") != file_sha256(ref_path):
        raise ValueError(f"{cand_path} was dumped against another reference than {ref_path} (or before dumps "
                         "recorded their reference's sha256: dump it again)")
    return ref, cand, ref_meta, cand_meta


def position_kl(ref, cand):
    """Per position KL(ref || cand) on the reference's top-K ids plus the merged tail bucket, and the reference's
    tail mass."""

    import numpy as np

    lp_p = ref["lp"].astype(np.float64)
    lp_q = cand["at_ref"].astype(np.float64)
    p = np.exp(lp_p)
    kl_top = (p * (lp_p - lp_q)).sum(-1)
    tail_p, tail_q = ref["tail_lp"].astype(np.float64), cand["at_ref_tail_lp"].astype(np.float64)
    p_tail = np.exp(tail_p)
    # an empty tail (top-K = vocabulary) contributes nothing; log masses never subtract -inf from -inf
    return kl_top + p_tail * np.subtract(tail_p, tail_q, out=np.zeros_like(tail_p), where=p_tail > 0), p_tail


def compare(ref_path: str, cand_path: str) -> dict:
    import numpy as np

    ref, cand, ref_meta, cand_meta = _load_pair(ref_path, cand_path)
    kl, p_tail = position_kl(ref, cand)
    agree = ref["ids"][:, 0] == cand["ids"][:, 0]
    in_top5 = (cand["ids"][:, :5] == ref["ids"][:, :1]).any(-1)
    names, offsets = ref_meta["names"], [*ref["offsets"], len(kl)]
    per_item = {name: {"positions": int(offsets[i + 1] - offsets[i]),
                       "kl_mean": float(kl[offsets[i]:offsets[i + 1]].mean()),
                       "top1_agreement": float(agree[offsets[i]:offsets[i + 1]].mean())}
                for i, name in enumerate(names)}
    return {"reference": ref_path, "candidate": cand_path, "positions": len(kl), "top_k": ref_meta["top_k"],
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


def block_bootstrap(values, offsets, block: int, resamples: int, seed: int):
    """Means of ``values`` (columns resampled together, rows = positions) over circular blocks of ``block``
    positions drawn with replacement within the eval items: no block spans two items, so the serial correlation of
    positions inside a reply is kept, and a block may wrap from an item's end to its start, so every position lies in
    equally many blocks. An item of at most ``block`` positions is one block, drawn ``block`` times as often. Every
    position therefore has the same expected weight and the resampled means centre on the sample mean (plain moving
    blocks under-weight each item's first and last positions and short items). Returns [resamples, columns]."""

    import numpy as np

    if block < 1 or resamples < 1:
        raise ValueError("bootstrap block and resamples must be positive")
    values = np.asarray(values, dtype=np.float64)
    bounds = [*offsets, len(values)]
    sums, sizes, weights = [], [], []
    for first, end in itertools.pairwise(bounds):
        item = values[first:end]
        if len(item) == 0:
            continue
        if len(item) <= block:
            sums.append(item.sum(0)[None])
            sizes.append([len(item)])
            weights.append([block])
        else:
            ring = np.concatenate([item, item[:block - 1]])
            sums.append(np.lib.stride_tricks.sliding_window_view(ring, block, axis=0).sum(-1))
            sizes.append([block] * len(item))
            weights.append([1] * len(item))
    sums, sizes, weights = np.concatenate(sums), np.concatenate(sizes), np.concatenate(weights).astype(np.float64)
    count = max(1, round(weights.sum() / block))     # blocks per resample: about as many positions as the eval set
    picks = np.random.default_rng(seed).choice(len(sums), size=(resamples, count), p=weights / weights.sum())
    return sums[picks].sum(1) / sizes[picks].sum(1)[:, None]


def paired(golden_path: str, old_path: str, new_path: str, *, block: int, resamples: int, seed: int) -> dict:
    """KL(golden || x) and golden top-1 agreement of OLD and NEW, their paired per-position differences and 95%
    circular-block bootstrap intervals (dumps checked against the golden reference)."""

    import numpy as np

    golden, old, golden_meta, old_meta = _load_pair(golden_path, old_path)
    if not golden_meta.get("golden"):
        raise ValueError(f"{golden_path} is not a campaign_golden reference")
    _, new, _, new_meta = _load_pair(golden_path, new_path)
    kl_old, p_tail = position_kl(golden, old)
    kl_new, _ = position_kl(golden, new)
    for path, values in ((old_path, kl_old), (new_path, kl_new)):
        # NaN compares false with every bound, so a NaN interval would pass; NaN/-inf log-probabilities (a broken
        # kernel, zero mass on a golden top-K id) leave the comparison undefined
        if not np.isfinite(values).all():
            raise ValueError(f"KL(golden || {path}) is not finite at {int((~np.isfinite(values)).sum())} positions "
                             "(NaN or -inf log-probabilities in the dump or the golden reference)")
    top = golden["ids"][:, 0]
    agree_old = (old["ids"][:, 0] == top).astype(np.float64)
    agree_new = (new["ids"][:, 0] == top).astype(np.float64)
    columns = np.stack([kl_old, kl_new, kl_new - kl_old, agree_old, agree_new, agree_new - agree_old], 1)
    boot = block_bootstrap(columns, golden["offsets"], block, resamples, seed)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    names = ("kl_old", "kl_new", "kl_delta", "agreement_old", "agreement_new", "agreement_delta")
    stats = {name: {"mean": float(columns[:, i].mean()), "ci95": [float(low[i]), float(high[i])]}
             for i, name in enumerate(names)}
    return {"golden": golden_path, "golden_sha256": file_sha256(golden_path),
            "golden_subset_of": golden_meta.get("subset_of"), "old": old_path, "new": new_path,
            "old_sha256": file_sha256(old_path), "new_sha256": file_sha256(new_path),
            "positions": len(columns), "golden_source": golden_meta.get("source"),
            "eval_sha256": golden_meta["eval_sha256"], "old_source": old_meta.get("source_revision"),
            "new_source": new_meta.get("source_revision"), **stats,
            "golden_tail_mass": {"mean": float(p_tail.mean()), "max": float(p_tail.max())},
            "bootstrap": {"block": block, "resamples": resamples, "seed": seed, "numpy": np.__version__,
                          "kind": "circular blocks within items, equal expected weight per position"}}


MARGIN_METHOD = "ci_bound"     # calibration records name their margin rule; a gate refuses any other


def margins(rows: dict[str, dict]) -> dict:
    """Non-inferiority margins from reorder-only variants' paired comparisons with the base: the largest deviation a
    numerically equivalent build shows at this eval size and confidence, i.e. the largest absolute 95% bound of the
    KL(golden || x) change (delta_kl) and of the top-1 agreement change (delta_agreement) over the variants."""

    return {"delta_kl": max(abs(b) for r in rows.values() for b in r["kl_delta"]["ci95"]),
            "delta_agreement": max(abs(b) for r in rows.values() for b in r["agreement_delta"]["ci95"])}


def verdict(kl_ci: list[float], agreement_ci: list[float], delta_kl: float, delta_agreement: float) -> tuple[str, list]:
    """Non-inferiority verdict of one paired comparison: "fail" when an interval lies wholly beyond its margin (lower
    KL bound > delta_kl, or upper agreement bound < -delta_agreement: a regression is established), "pass" when the
    upper KL bound <= delta_kl and the lower agreement bound >= -delta_agreement, else "incomplete" (an interval
    straddles its margin: uncertainty, resolved by more evidence, never a pass). Improvements never fail. Returns the
    status and the measures that failed or straddled."""

    (kl_low, kl_high), (agree_low, agree_high) = kl_ci, agreement_ci
    beyond = [n for n, bad in (("kl", kl_low > delta_kl), ("agreement", agree_high < -delta_agreement)) if bad]
    if beyond:
        return "fail", beyond
    straddling = [n for n, bad in (("kl", kl_high > delta_kl), ("agreement", agree_low < -delta_agreement)) if bad]
    return ("incomplete", straddling) if straddling else ("pass", [])


def calibrate(golden_path: str, base_path: str, variants: dict[str, str], *, block: int = 64,
              resamples: int = 4000, seed: int = 0) -> dict:
    """Non-inferiority margins (see ``margins``) from 4 to 6 reorder-only variants (the same declared math, another
    summation order) of the base build, each dumped against the golden reference, fixed before any candidate is seen.
    Every variant's differences and intervals are recorded, delta_kl also as a percentage of the base's KL (None for
    a base whose KL is 0), and a leave-one-out check (each variant judged with the margins of the others) counts the
    gate's false alarms."""

    import numpy as np

    if not CALIBRATION_VARIANTS[0] <= len(variants) <= CALIBRATION_VARIANTS[1]:
        raise ValueError(f"calibration takes {CALIBRATION_VARIANTS[0]} to {CALIBRATION_VARIANTS[1]} reorder-only "
                         f"variants, not {len(variants)} (the margins are the largest of their deviations)")
    base = np.load(base_path)
    for name, path in variants.items():
        # positive control: the engine is deterministic, so a bit-identical dump means the patch did not load or
        # changed no bits; it carries no reorder information
        if np.array_equal(np.load(path)["at_ref"], base["at_ref"]):
            raise ValueError(f"variant {name} ({path}) is bit-identical to the base: not a reorder sample")
    rows = {name: paired(golden_path, base_path, path, block=block, resamples=resamples, seed=seed)
            for name, path in variants.items()}
    first = next(iter(rows.values()))
    found = margins(rows)
    base_kl = first["kl_old"]["mean"]
    held_out = {}
    for name, row in rows.items():
        others = margins({k: r for k, r in rows.items() if k != name})
        status, measures = verdict(row["kl_delta"]["ci95"], row["agreement_delta"]["ci95"], others["delta_kl"],
                                   others["delta_agreement"])
        held_out[name] = {**others, "status": status, "measures": measures}
    return {"golden": golden_path, "golden_sha256": first["golden_sha256"], "eval_sha256": first["eval_sha256"],
            "golden_subset_of": first["golden_subset_of"],
            "golden_source": first["golden_source"], "base": base_path, "base_sha256": first["old_sha256"],
            "positions": first["positions"], "bootstrap": first["bootstrap"], "margin_method": MARGIN_METHOD,
            **found, "delta_kl_pct_of_base_kl": 100 * found["delta_kl"] / base_kl if base_kl > 0 else None,
            "variants": {name: {k: r[k] for k in ("new", "new_sha256", "new_source", "kl_new", "kl_delta",
                                                   "agreement_new", "agreement_delta")} for name, r in rows.items()},
            "base_values": {k: first[k] for k in ("kl_old", "agreement_old")},
            "leave_one_out": {"variants": held_out,
                              "false_alarms": sorted(n for n, v in held_out.items() if v["status"] != "pass")},
            "method": "delta = max over reorder-only variants of the absolute 95% bounds of their paired difference "
                      "from the base (KL(golden||x) change; golden top-1 agreement change); gate: fail when an "
                      "interval lies wholly beyond its margin, pass when the KL upper bound <= delta_kl and the "
                      "agreement lower bound >= -delta_agreement, otherwise incomplete"}


def gate(golden_path: str, old_path: str, new_path: str, *, calibration: dict | None = None, block: int = 64,
         resamples: int = 4000, seed: int = 0) -> dict:
    """Golden-reference non-inferiority gate of NEW against OLD under the model's calibration record (margins from
    reorder-only variants, ``margins``; the verdict rule is ``verdict``). Without a calibration the status is
    "uncalibrated" (intervals reported, no verdict; the CLI exits 3 as for incomplete). A calibration belongs to one
    golden reference (eval set and positions) and bootstrap setting, and must name MARGIN_METHOD and carry finite
    non-negative margins (refused otherwise); delta_kl is reported in nats and as a percentage of the calibration
    base's KL."""

    result = paired(golden_path, old_path, new_path, block=block, resamples=resamples, seed=seed)
    result["ci_width"] = {k: result[k]["ci95"][1] - result[k]["ci95"][0] for k in ("kl_delta", "agreement_delta")}
    if calibration is None:
        result.update(status="uncalibrated", margins=None)
        return result
    if calibration.get("margin_method") != MARGIN_METHOD:
        raise ValueError(f"calibration margin method {calibration.get('margin_method')!r} is not {MARGIN_METHOD!r}: "
                         "recalibrate (campaign_kl.py calibrate) from the variant dumps")
    for key in ("golden_sha256", "eval_sha256", "positions"):
        if calibration[key] != result[key]:
            raise ValueError(f"calibration {key} does not match this gate's golden reference")
    for key in ("block", "resamples", "seed"):
        if calibration["bootstrap"][key] != result["bootstrap"][key]:
            raise ValueError(f"bootstrap {key} {result['bootstrap'][key]} differs from the calibration's "
                             f"{calibration['bootstrap'][key]}")
    delta_kl, delta_agree = calibration["delta_kl"], calibration["delta_agreement"]
    # the record crossed a file boundary (json reads NaN and Infinity): every comparison with a NaN margin is false and
    # an infinite one is never exceeded, so either would pass any candidate; calibrate writes finite bounds >= 0
    if not all(isinstance(m, (int, float)) and 0 <= m < math.inf for m in (delta_kl, delta_agree)):
        raise ValueError(f"calibration margins delta_kl={delta_kl!r}, delta_agreement={delta_agree!r} are not finite "
                         "non-negative numbers: recalibrate (campaign_kl.py calibrate) from the variant dumps")
    status, measures = verdict(result["kl_delta"]["ci95"], result["agreement_delta"]["ci95"], delta_kl, delta_agree)
    result["margins"] = {"delta_kl": delta_kl, "delta_kl_pct_of_base_kl": calibration["delta_kl_pct_of_base_kl"],
                         "delta_agreement": delta_agree, "method": calibration["margin_method"],
                         "calibration_base": calibration["base"], "variants": sorted(calibration["variants"]),
                         "leave_one_out_false_alarms": calibration["leave_one_out"]["false_alarms"]}
    result.update(status=status, failed=measures if status == "fail" else [],
                  straddling=measures if status == "incomplete" else [],
                  method="non-inferiority against the golden reference: fail when an interval lies wholly beyond "
                         "its margin (a regression is established); pass iff the upper 95% bound of KL(golden||new) "
                         "- KL(golden||old) <= delta_kl and the lower bound of the top-1 agreement change >= "
                         "-delta_agreement; otherwise incomplete (uncertainty, never a pass)")
    return result


def subset(golden_path: str, eval_path: str, out_dir: str, *, max_prompt_tokens: int) -> dict:
    """A golden reference and eval set restricted to the items of at most ``max_prompt_tokens`` prompt tokens, sliced
    from existing ones (no GPU): OUT/eval.json and OUT/golden/{kl.npz, meta.json}, the layout the runner's --kl-from
    reads. Both record what they were cut from (sha256 of the full golden reference and eval set, item names), so a
    gate names the set it used; the dumps of a candidate must use this subset's eval.json. OUT must not be the full
    set's own directory: the subset would replace the files it is cut from."""

    import numpy as np

    out = Path(out_dir)
    inputs = {Path(golden_path).resolve(), Path(eval_path).resolve()}
    if any(path.resolve() in inputs for path in (out / "eval.json", out / "golden" / "kl.npz")):
        raise ValueError(f"{out_dir} holds the full set's own eval.json or golden/kl.npz: the subset would replace "
                         "them (choose another OUT_DIR)")
    data, eval_sha256 = read_eval(Path(eval_path))
    golden = np.load(golden_path)
    meta = json.loads(str(golden["meta"]))
    if meta.get("eval_sha256") != eval_sha256 or not meta.get("golden"):
        raise ValueError(f"{golden_path} is not a campaign_golden reference of {eval_path}")
    offsets = [*golden["offsets"], len(golden["lp"])]
    keep = [i for i, item in enumerate(data["items"]) if len(item["prompt"]) <= max_prompt_tokens]
    if not keep:
        raise ValueError(f"no eval item has at most {max_prompt_tokens} prompt tokens")
    rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in keep])
    (out / "golden").mkdir(parents=True, exist_ok=True)
    names = [data["items"][i]["name"] for i in keep]
    origin = {"golden_sha256": file_sha256(golden_path), "eval_sha256": eval_sha256,
              "max_prompt_tokens": max_prompt_tokens, "items": names}
    (out / "eval.json").write_text(json.dumps({**{k: v for k, v in data.items() if k != "items"},
                                               "items": [data["items"][i] for i in keep], "subset_of": origin}))
    sizes = [offsets[i + 1] - offsets[i] for i in keep]
    arrays = {k: golden[k][rows] for k in golden.files if k not in ("meta", "offsets")}
    arrays["offsets"] = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    new_meta = {**meta, "eval_sha256": file_sha256(str(out / "eval.json")), "names": names, "subset_of": origin}
    np.savez(out / "golden" / "kl.npz", meta=json.dumps(new_meta), **arrays)
    (out / "golden" / "meta.json").write_text(json.dumps({"golden_dir": meta.get("model_dir"), "subset_of": origin,
                                                          "source": meta.get("source")}, indent=1))
    return {"out": str(out), "items": names, "positions": len(rows), **origin}


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
    d.add_argument("--allow-long-prefill", action="store_true",
                   help=f"build or dump items above {LONG_PREFILL_TOKENS} prompt tokens (held otherwise)")
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
    g = sub.add_parser("gate", help="golden-reference gate: GOLDEN OLD NEW (OLD, NEW dumped with --reference GOLDEN)")
    g.add_argument("golden")
    g.add_argument("old")
    g.add_argument("new")
    g.add_argument("--calibration", help="the model's calibration record (margins); without it: uncalibrated")
    u = sub.add_parser("subset", help="slice a golden reference + eval set to items of at most N prompt tokens")
    u.add_argument("golden")
    u.add_argument("eval")
    u.add_argument("out_dir", help="writes eval.json and golden/{kl.npz,meta.json}; not the full set's directory")
    u.add_argument("--max-prompt-tokens", type=int, default=LONG_PREFILL_TOKENS)
    u.add_argument("--out", help="also write the result summary here")
    k = sub.add_parser("calibrate", help="margins from reorder-only variants: GOLDEN BASE NAME=DUMP ...")
    k.add_argument("golden")
    k.add_argument("base")
    k.add_argument("variants", nargs="+",
                   help=f"NAME=DUMP of {CALIBRATION_VARIANTS[0]} to {CALIBRATION_VARIANTS[1]} reorder-only variants "
                        "of BASE (dumped with --reference GOLDEN)")
    for parser in (g, k):
        parser.add_argument("--block", type=int, default=64, help="bootstrap block length in positions")
        parser.add_argument("--resamples", type=int, default=4000)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--out")
    args = ap.parse_args()
    if args.command == "dump":
        if args.serve and args.serve[0] == "--":
            args.serve = args.serve[1:]
        if not args.serve or args.serve[0] != "serve":
            ap.error("dump needs -- serve MODEL [flags]")
        if args.card is None:
            args.card = (json.loads(Path(args.profile).read_text())["card_sampling"]["nothink"] if args.profile
                         else {})
        loaded = eval_set(args)                      # refusals before the engine loads
        with_app(args.serve, args.port, lambda app: dump(app, args, loaded))
    else:
        if args.command == "compare":
            result = compare(args.reference, args.candidate)
        elif args.command == "subset":
            result = subset(args.golden, args.eval, args.out_dir, max_prompt_tokens=args.max_prompt_tokens)
        else:
            boot = {"block": args.block, "resamples": args.resamples, "seed": args.seed}
            if args.command == "calibrate":
                pairs = [v.split("=", 1) for v in args.variants]
                if any(len(p) != 2 or not all(p) for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
                    ap.error("calibrate variants must be distinct NAME=DUMP arguments")
                result = calibrate(args.golden, args.base, dict(pairs), **boot)
            else:
                record = json.loads(Path(args.calibration).read_text()) if args.calibration else None
                result = gate(args.golden, args.old, args.new, calibration=record, **boot)
        text = json.dumps(result, indent=1)
        if args.out:
            Path(args.out).write_text(text + "\n")
        print(text)
        if args.command == "gate":
            # no verdict (incomplete, uncalibrated) is never a pass
            return {"pass": 0, "fail": 1}.get(result["status"], 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
