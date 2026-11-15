#!/usr/bin/env python3
"""Teacher-forced output-quality evaluation of a served model, and A/B comparison of two runs.

  tokenize --corpus corpus-v1.json --tokenizer DIR --out corpus-v1.tokens.json
      Freeze a corpus into token IDs (needs ``transformers`` and ``tensorfold``; run it once per corpus version,
      e.g. inside the serving image). Chat documents use TensorFold's served template renderer, thinking off.
  score --tokens corpus-v1.tokens.json --out RUN.json [--url http://127.0.0.1:8000] [--backend tensorfold|vllm]
        [--top 20] [--repeat 1] [--model NAME]
      TensorFold: POST /v1/tensorfold/score (the server must run with TENSORFOLD_PRIVATE_SCORE=1). Every
      document's prompt tokens are scored by prompt-pass (prefill) rows and its continuation by teacher-forced
      decode/MTP-verify rows. vLLM: /v1/completions with ``prompt_logprobs`` over prompt+continuation (one pass).
  compare A.json B.json [--margin 0.005] [--out RESULT.json]
      Paired per-token statistics of B against A per path and kind, and a verdict (see ``compare``'s doc).

Standard library only for ``score`` and ``compare``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

SCHEMA_TOKENS = "tensorfold.quality-corpus-tokens"
SCHEMA_RUN = "tensorfold.quality-run"
SCHEMA_RESULT = "tensorfold.quality-comparison"
CONTINUATION_TOKENS = 192          # text documents: at most this many final tokens are scored by decode rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- tokenize

def tokenize(args) -> None:
    from transformers import AutoTokenizer
    from tensorfold.server.text import CHANNEL_MARKERS, render_prompt_ids

    corpus_path, tok_dir = Path(args.corpus), Path(args.tokenizer)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    # Exported padding/truncation are batch settings, not the model's tokenization (as in
    # tensorfold.token_metadata.load_tokenizer); AEON Gemma 4 ships left padding to 4096.
    tok.backend_tokenizer.no_padding()
    tok.backend_tokenizer.no_truncation()
    docs = []
    for doc in corpus["documents"]:
        if doc["kind"] == "chat":
            messages = doc["messages"]
            if messages[-1]["role"] != "assistant":
                raise SystemExit(f"{doc['id']}: a chat document ends with the assistant reply it scores")
            prompt = render_prompt_ids(tok, messages[:-1], enable_thinking=False, add_generation_prompt=True)
            full = render_prompt_ids(tok, messages, enable_thinking=False, add_generation_prompt=False)
            if full[:len(prompt)] == prompt:
                continuation = full[len(prompt):]
            else:
                # Gemma 4 (thinking off) opens the reply with an empty thought channel that its
                # history rendering omits: score the history reply after the served prompt.
                at = next(i for i, (a, b) in enumerate(zip(prompt, full)) if a != b)
                opener, closer = CHANNEL_MARKERS
                if tok.decode(prompt[at:]) != f"{opener}\n{closer}":
                    raise SystemExit(f"{doc['id']}: the rendered conversation does not extend its generation prompt")
                continuation = full[at:]
        else:
            ids = [int(t) for t in tok(doc["text"], add_special_tokens=False)["input_ids"]]
            cut = len(ids) - min(CONTINUATION_TOKENS, len(ids) // 3)
            prompt, continuation = ids[:cut], ids[cut:]
        if len(prompt) < 2 or not continuation:
            raise SystemExit(f"{doc['id']}: too short")
        docs.append({"id": doc["id"], "kind": doc["kind"], "prompt": prompt, "continuation": continuation})
    out = {"schema": SCHEMA_TOKENS, "version": 1, "corpus": corpus_path.name, "corpus_sha256": _sha(corpus_path),
           "corpus_version": corpus["version"],
           "tokenizer_sha256": _sha(tok_dir / "tokenizer.json"),
           "tokenizer_config_sha256": _sha(tok_dir / "tokenizer_config.json"), "documents": docs}
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    total = sum(len(d["prompt"]) + len(d["continuation"]) for d in docs)
    print(f"{len(docs)} documents, {total} tokens "
          f"(prompt {sum(len(d['prompt']) for d in docs)}, continuation {sum(len(d['continuation']) for d in docs)})")


# ----------------------------------------------------------------------------------------------- score

def _post(url: str, body: dict, timeout: float = 1800.0) -> dict:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.load(reply)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{url}: HTTP {error.code}: {error.read()[:2000]!r}") from None


def _score_tensorfold(url: str, doc: dict, top: int, model: str | None) -> dict:
    body = {"prompt": doc["prompt"], "continuation": doc["continuation"], "top_logprobs": top}
    if model:
        body["model"] = model
    got = _post(url.rstrip("/") + "/v1/tensorfold/score", body)
    prompt = got["prompt"]
    if [row["id"] for row in prompt] != doc["prompt"][1:] or [row["id"] for row in got["continuation"]] != doc["continuation"]:
        raise SystemExit(f"{doc['id']}: the server scored different tokens than the corpus")
    reply = got["continuation"]
    if reply[0].get("origin") != "prompt-end" or any(row.get("origin") == "prompt-end" for row in reply[1:]):
        raise SystemExit(f"{doc['id']}: unexpected row origins")
    # The reply's first token comes from the prompt pass's head, so it belongs to the prefill path.
    return {"prefill": prompt + reply[:1], "decode": reply[1:], "stats": got.get("stats")}


def _score_vllm(url: str, doc: dict, top: int, model: str | None) -> dict:
    ids = doc["prompt"] + doc["continuation"]
    body = {"prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": top}
    if model:
        body["model"] = model
    got = _post(url.rstrip("/") + "/v1/completions", body)
    rows = got["choices"][0]["prompt_logprobs"]
    if len(rows) != len(ids) or rows[0] is not None:
        raise SystemExit(f"{doc['id']}: vLLM returned {len(rows)} prompt rows for {len(ids)} tokens")
    scored = []
    for i in range(1, len(ids)):
        entries = rows[i]
        own = entries.get(str(ids[i]))
        if own is None:
            raise SystemExit(f"{doc['id']}: vLLM omitted the prompt token's own logprob at {i}")
        ranked = sorted(((int(k), v) for k, v in entries.items() if v.get("rank") is not None and v["rank"] <= top),
                        key=lambda kv: kv[1]["rank"])
        scored.append({"id": ids[i], "logprob": float(own["logprob"]),
                       "top": [[k, float(v["logprob"])] for k, v in ranked]})
    cut = len(doc["prompt"])                          # prompt rows plus the reply's first token, as TensorFold
    return {"prefill": scored[:cut], "decode": scored[cut:], "stats": None}


def score(args) -> None:
    tokens_path = Path(args.tokens)
    corpus = json.loads(tokens_path.read_text(encoding="utf-8"))
    if corpus.get("schema") != SCHEMA_TOKENS:
        raise SystemExit("not a frozen corpus token file")
    scorer = _score_tensorfold if args.backend == "tensorfold" else _score_vllm
    runs = []
    for repeat in range(args.repeat):
        docs, t0 = [], time.perf_counter()
        for doc in corpus["documents"]:
            got = scorer(args.url, doc, args.top, args.model)
            docs.append({"id": doc["id"], "kind": doc["kind"], **got})
        seconds = time.perf_counter() - t0
        runs.append({"seconds": round(seconds, 3), "documents": docs})
        summary = _summary(docs)
        rows = [tuple(row["origin"]) for doc in docs for row in doc["decode"] if isinstance(row.get("origin"), list)]
        if rows:
            print("decode rows scored by (verify row, window rows): " + ", ".join(
                f"{k}: {rows.count(k)}" for k in sorted(set(rows))))
        print(f"repeat {repeat}: {seconds:.1f}s " + "  ".join(
            f"{path}: nll {s['nll']:.5f} ppl {s['ppl']:.4f} n {s['n']}" for path, s in summary.items()))
    if len(runs) > 1:
        same = all(_bitwise(runs[0]["documents"], run["documents"]) for run in runs[1:])
        print("repeats bitwise identical" if same else "REPEATS DIFFER")
    out = {"schema": SCHEMA_RUN, "version": 1, "backend": args.backend, "label": args.label, "url": args.url,
           "top": args.top, "tokens_file": tokens_path.name, "tokens_sha256": _sha(tokens_path),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": runs}
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


def _bitwise(a: list, b: list) -> bool:
    return all(x["prefill"] == y["prefill"] and x["decode"] == y["decode"] for x, y in zip(a, b, strict=True))


def _summary(docs: list) -> dict:
    out = {}
    for path in ("prefill", "decode"):
        values = [-row["logprob"] for doc in docs for row in doc[path]]
        nll = math.fsum(values) / len(values)
        out[path] = {"n": len(values), "nll": nll, "ppl": math.exp(nll)}
    return out


# ----------------------------------------------------------------------------------------------- compare

def _rows(run: dict) -> dict:
    """(doc, path, index) -> row of the first repeat."""
    return {(doc["id"], path, i): (doc["kind"], row)
            for doc in run["runs"][0]["documents"] for path in ("prefill", "decode")
            for i, row in enumerate(doc[path])}


def _coarse_kl(p_top: list, q_top: list) -> tuple[float, float] | None:
    """KL(P||Q) on the partition {ids in both top lists} + one tail bucket: a lower bound of the true KL (data
    processing inequality). Also returns P's tail mass, which bounds what the partition cannot see."""
    if not p_top or not q_top:
        return None
    q = dict((int(k), v) for k, v in q_top)
    shared = [(int(k), v) for k, v in p_top if int(k) in q]
    kl = math.fsum(math.exp(lp) * (lp - q[k]) for k, lp in shared)
    p_tail = max(0.0, 1.0 - math.fsum(math.exp(lp) for _, lp in shared))
    q_tail = max(0.0, 1.0 - math.fsum(math.exp(q[k]) for k, _ in shared))
    if p_tail > 0.0:
        kl += p_tail * (math.log(p_tail) - math.log(max(q_tail, 1e-30)))
    return max(kl, 0.0), p_tail


def _paired(diffs: list, rng: random.Random, groups: list, boot: int) -> dict:
    """Mean of B-A per-token NLL differences; SE and percentile CI by a document-blocked bootstrap (tokens within a
    document are correlated, so resampling tokens alone would understate the uncertainty)."""
    n = len(diffs)
    mean = math.fsum(diffs) / n
    var = math.fsum((d - mean) ** 2 for d in diffs) / max(1, n - 1)
    by_group: dict = {}
    for d, g in zip(diffs, groups):
        by_group.setdefault(g, []).append(d)
    blocks = [(math.fsum(v), len(v)) for v in by_group.values()]
    means = []
    for _ in range(boot):
        picks = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        means.append(math.fsum(s for s, _ in picks) / sum(c for _, c in picks))
    means.sort()
    lo, hi = means[int(0.025 * boot)], means[min(boot - 1, int(0.975 * boot))]
    boot_mean = math.fsum(means) / boot
    boot_se = math.sqrt(math.fsum((m - boot_mean) ** 2 for m in means) / max(1, boot - 1))
    return {"n": n, "documents": len(blocks), "mean_diff": mean, "token_sd": math.sqrt(var),
            "naive_se": math.sqrt(var / n), "block_se": boot_se, "ci95": [lo, hi],
            "identical_tokens": sum(1 for d in diffs if d == 0.0)}


def compare(args) -> None:
    """Verdict per path (prefill, decode) on mean per-token NLL difference d = NLL(B) - NLL(A), in nats/token.

    Predeclared sensitivity (policy D14/F4): ``--margin`` M (default 0.005 nats/token, about 0.5% perplexity) is
    the smallest difference treated as material; it is fixed before looking at results. With the 95% document-
    blocked bootstrap CI [lo, hi] of d:
      better      hi < 0              (B's NLL is lower: a significant improvement, however small)
      worse       lo > 0 and hi > M   (significantly higher NLL that cannot be excluded from being material)
      equivalent  -M < lo and hi < M  (TOST-style: the whole CI lies inside the margin)
      worse-within-margin lo > 0 and hi <= M (a real but immaterial loss: reported, never called equivalent)
      inconclusive otherwise          (non-significance is not equivalence: rerun with more tokens or keep A)
    Identical tokens (bitwise equal scores) give d = 0 exactly and an 'identical' verdict.
    """
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    if a["tokens_sha256"] != b["tokens_sha256"]:
        raise SystemExit("the runs scored different frozen corpora")
    ra, rb = _rows(a), _rows(b)
    if set(ra) != set(rb):
        raise SystemExit("the runs scored different positions")
    rng = random.Random(args.seed)
    result = {"schema": SCHEMA_RESULT, "version": 1, "a": {"file": args.a, "label": a.get("label"), "backend": a["backend"]},
              "b": {"file": args.b, "label": b.get("label"), "backend": b["backend"]}, "margin": args.margin,
              "bootstrap": args.bootstrap, "seed": args.seed, "paths": {}}
    for path in ("prefill", "decode"):
        keys = sorted(k for k in ra if k[1] == path)
        if not keys:
            continue
        per_kind: dict = {}
        diffs, groups, agree, kls, tails, nll_a, nll_b = [], [], 0, [], [], [], []
        for key in keys:
            kind, x = ra[key]
            _, y = rb[key]
            if x["id"] != y["id"]:
                raise SystemExit(f"{key}: different target tokens")
            d = (-y["logprob"]) - (-x["logprob"])
            diffs.append(d)
            groups.append(key[0])
            nll_a.append(-x["logprob"])
            nll_b.append(-y["logprob"])
            per_kind.setdefault(kind, []).append(d)
            if x["top"] and y["top"]:
                agree += int(x["top"][0][0] == y["top"][0][0])
                got = _coarse_kl(x["top"], y["top"])
                if got is not None:
                    kls.append(got[0])
                    tails.append(got[1])
        stats = _paired(diffs, rng, groups, args.bootstrap)
        lo, hi = stats["ci95"]
        m = args.margin
        if stats["identical_tokens"] == len(diffs):
            verdict = "identical"
        elif hi < 0:
            verdict = "better"
        elif lo > 0 and hi > m:
            verdict = "worse"
        elif -m < lo and hi < m and lo <= 0:
            verdict = "equivalent"
        elif lo > 0:
            verdict = "worse-within-margin"
        else:
            verdict = "inconclusive"
        mean_a, mean_b = math.fsum(nll_a) / len(nll_a), math.fsum(nll_b) / len(nll_b)
        entry = {"verdict": verdict, "nll_a": mean_a, "nll_b": mean_b, "ppl_a": math.exp(mean_a),
                 "ppl_b": math.exp(mean_b), **stats,
                 "per_kind_mean_diff": {k: math.fsum(v) / len(v) for k, v in sorted(per_kind.items())}}
        if kls:
            kls_sorted = sorted(kls)
            entry.update({"top1_agreement": agree / len(kls), "kl_lower_bound_mean": math.fsum(kls) / len(kls),
                          "kl_lower_bound_p99": kls_sorted[min(len(kls) - 1, int(0.99 * len(kls)))],
                          "kl_lower_bound_max": kls_sorted[-1], "tail_mass_mean": math.fsum(tails) / len(tails),
                          "kl_positions": len(kls)})
        result["paths"][path] = entry
        print(f"{path:8s} {verdict:20s} nll A {mean_a:.5f} B {mean_b:.5f}  d {stats['mean_diff']:+.5f} "
              f"CI95 [{lo:+.5f}, {hi:+.5f}] (block SE {stats['block_se']:.5f}, n {stats['n']}, "
              f"{stats['documents']} docs)"
              + (f"  top1 {entry['top1_agreement']:.4f}  KL>= {entry['kl_lower_bound_mean']:.2e} "
                 f"(p99 {entry['kl_lower_bound_p99']:.2e})" if kls else ""))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("tokenize")
    p.add_argument("--corpus", required=True)
    p.add_argument("--tokenizer", required=True, help="directory holding tokenizer.json and tokenizer_config.json")
    p.add_argument("--out", required=True)
    p.set_defaults(run=tokenize)
    p = sub.add_parser("score")
    p.add_argument("--tokens", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--backend", choices=("tensorfold", "vllm"), default="tensorfold")
    p.add_argument("--model")
    p.add_argument("--label", default="")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--repeat", type=int, default=1)
    p.set_defaults(run=score)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--margin", type=float, default=0.005)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.set_defaults(run=compare)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
