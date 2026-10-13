"""Replay finished requests' token traces under deterministic draft policies, with a modeled round cost.

    python tools/draft_replay.py TRACE_OR_DIR... [--json OUT] [--t0 34.6] [--row-ms 6.5] [--step-ms 1.6]
        [--anchors 4,6,8] [--windows 5,8,16] [--recent 32] [--longest 64] [--corpus-tokens 1000000]
        [--corpus-segment 262144] [--mtp-drafts 2] [--mtp-accept 0.75] [--seed 0]
        [--log DRAFT_LOG...] [--tables FIT.json] [--lam 0.0466]

Input: the files ``TENSORFOLD_DRAFT_TRACE`` writes (``tensorfold.cuda.draft_log``), one per finished request, and
optionally (``--log``) the ``TENSORFOLD_DRAFT_LOG`` lines of the same run for the MTP stop-rule comparison below.

Why replay is exact: emitted tokens do not depend on the drafter (keyed sampling), so a deterministic drafter whose
round starts at reply index i with drafts D keeps exactly lcp(D, out[i:]) of them (at most the reply's remaining
tokens but one) and emits one more token. Copy and suffix drafters therefore replay exactly, round by round. MTP
drafts need the model: an MTP round reuses the recorded MTP round that started at the same index (exact when the MTP
chain at a round start is a function of the committed tokens), else samples the recorded MTP rounds of its context
bucket (seeded), else Tier-0 aggregates (``--mtp-drafts`` drafts, each kept with ``--mtp-accept`` given the ones
before). A prompt's first round drafts as the server's rule drafts a joining prompt: the A/B baseline's rule with
MTP drafts only (every policy but the ``-new`` ones: the recorded first round), the cost-aware rule (the ``-new``
policies) also from the request's own text when it pays. The server's session drafts need the prompt's index: a
prompt of ``session_index.BULK`` (4096) tokens or more builds it in the background, and until it is in, the server
drafts MTP only, so the replay's suffix rows may draft from text in a long prompt's first rounds where the server
could not (a recorded round's ``src`` shows what the server drafted; ``CopyIndex`` builds on first use, never late).

Policies (one report row each, per context bucket and overall):

    recorded            the trace's own rounds (``measured_ms`` is their wall time; streams share rounds)
    copy8               today's copy drafts: ``CopyIndex`` loaded from the source with the trace's needle
                        (``copy_match``, 8) and cap (the decoder's depth); on a trace recorded with copies on it must
                        reproduce ``recorded`` round for round (``parity``)
    session-k{K}-w{W}   design 4.2 S1 over this request's prompt and reply: anchor K, the newest ``--recent``
                        occurrences, the longest backward match (up to ``--longest``), then the newest; the continuation
                        (periodic when it reaches the current end), up to W-1 drafts; MTP when nothing matches
    corpus-k{K}-w{W}    S1 over the session, else over a causal in-process corpus (design 4.3): the replies of
                        the same decoder's foreground requests that finished before this one joined (its ``after``),
                        in segments of ``--corpus-segment`` tokens, the oldest segment dropped past
                        ``--corpus-tokens``; the longer match wins, the session on ties
    oracle-w{W}         the true continuation, W-1 drafts: the upper bound of a window of W rows
    suffix-old          the server's session drafts under switch "draft_rule" old (``MultiDecoder``): S1 with the
                        trace's anchor, ``recent`` and ``longest``, taken whole (up to the decoder's depth) when the
                        match is at least ``COPY_MATCH`` (8) long; on a trace recorded so ("text" suffix, every round's
                        rule old) it must reproduce ``recorded`` round for round (``parity_suffix``), except a
                        background request that gave way and replayed (its owed drafts are not modelled; rounds
                        with source "owed" show it); a long prompt's index built in the background may also miss
                        a request's first rounds
    suffix-new-w{W}     the session drafts under the cost-aware rule (``tensorfold.drafters.draft_rule``): the verified
                        prefix of S1's continuation whose chances (``--tables``, else the prior of the trace's
                        confidence) pay for their rows at ``--lam`` x ``--row-ms``, taken when worth more than an
                        MTP chain valued at its optimistic chances (the live rule also knows the first MTP pick's;
                        exact for the prior, an approximation for fitted tables); W-1 drafts at most (W above the
                        decoder's depth + 1 is beyond what the server captures: a what-if)
    copy8-new           today's copy drafts (``CopyIndex``) judged the same way (as a match of 8)
    sam-old, sam-new-w{W}
                        as suffix-old and suffix-new-w{W}, with the longest earlier match anywhere (an online suffix
                        automaton, design 4.2 S2, pure Python: the coverage oracle S1's newest-``recent`` limit is
                        measured against)

Round cost: ``t0 + (rows - 1) * row_ms + steps * step_ms``, steps = the MTP absorb (1) plus an MTP chain's further
steps (an MTP round with J drafts: max(1, J)). Context buckets: prompt plus reply before the round, <50k, 50-300k,
300-600k, >600k tokens. Only TensorFold-owned sources (the session, and requests this server served) are simulated.
Serial and grammar-constrained requests are reported under ``recorded`` only.

MTP stop rules (``--log``, the ``rules`` section): every probe round's MTP chain is labelled to its room, so any stop
rule's outcome on it is exact (design 3.6): a rule that verifies k of its drafts keeps the leading labelled ones among
them and runs as many MTP head steps as the picks it computes. Rules: ``confidence`` (today's: the first draft
always, later ones while at or above the decoder's confidence), ``prior`` (the cost-aware rule with the prior's
tables) and ``fit`` (with ``--tables``), at ``--lam`` x (``--row-ms``, ``--step-ms``) and steps forced alone (the
shared-step option of concurrent rounds is not modelled), and ``allocate`` (``engine.allocate``: the prefix with the
most expected tokens a modelled millisecond, design 3.1's comparison, on the prior's chances with the whole chain
computed). Rounds whose labels did not resolve, and streams whose log predates the "greedy" field, are counted and
left out.
"""

from __future__ import annotations

import argparse
import ast
from bisect import bisect_right
from collections.abc import Sequence
import json
from pathlib import Path
import random
import sys

BUCKETS = ((50_000, "<50k"), (300_000, "50-300k"), (600_000, "300-600k"), (None, ">600k"))
SOURCE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen3_5/cuda/decode.py"
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:                             # the rule's host math (numpy only), as the server runs it
    sys.path.insert(0, str(SRC))
from tensorfold.drafters import draft_rule  # noqa: E402
from tensorfold.engine.allocate import allocate  # noqa: E402


def copy_index_class():
    """The production ``CopyIndex``, compiled from its source alone (its module imports torch)."""

    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CopyIndex")
    namespace = {"Sequence": Sequence}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["CopyIndex"]


class Prefix(Sequence):
    """The first ``n`` tokens of ``seq`` without copying them (``CopyIndex`` reads a growing context)."""

    def __init__(self, seq: list[int], n: int) -> None:
        self.seq, self.n = seq, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.n)
            return self.seq[start:stop:step]
        if key < 0:
            key += self.n
        if not 0 <= key < self.n:
            raise IndexError(key)
        return self.seq[key]


def bucket(context: int) -> str:
    return next(name for limit, name in BUCKETS if limit is None or context < limit)


def lcp(drafts: Sequence[int], truth: Sequence[int], at: int) -> int:
    """How many leading drafts equal ``truth[at:]``."""

    n = 0
    for d in drafts:
        if at + n >= len(truth) or truth[at + n] != d:
            break
        n += 1
    return n


class Index:
    """Where each k-gram of a token sequence ends, ascending: S1's anchor index (exact; no hashing)."""

    def __init__(self, k: int) -> None:
        self.k, self.seq, self.ends = k, [], {}

    def extend(self, tokens: Sequence[int]) -> None:
        seq, k = self.seq, self.k
        for t in tokens:
            seq.append(int(t))
            if len(seq) >= k:
                gram = tuple(seq[-k:])
                if -1 not in gram:                       # a corpus separator never matches
                    self.ends.setdefault(gram, []).append(len(seq) - 1)


def match(index: Index, needle: Sequence[int], n: int, before: int, recent: int, longest: int) -> tuple[int, int]:
    """The best earlier occurrence of ``needle[:n]``'s last k tokens among ``index``'s ends < ``before``: the newest
    ``recent`` ones, by backward match length (at most ``longest``), then the newest. Returns (length, end) or
    (0, -1)."""

    k = index.k
    if n < k:
        return 0, -1
    ends = index.ends.get(tuple(needle[n - k:n]))
    if not ends:
        return 0, -1
    stop = bisect_right(ends, before - 1)
    seq, best = index.seq, (0, -1)
    for e in reversed(ends[max(0, stop - recent):stop]):
        length = 0
        while length < longest and e - length >= 0 and seq[e - length] == needle[n - 1 - length]:
            length += 1
        if length > best[0]:
            best = (length, e)
    return best if best[0] >= k else (0, -1)


def continuation(seq: Sequence[int], e: int, n: int, count: int) -> list[int]:
    """``count`` tokens after ``seq[e]``, ``seq[:n]`` known: past n-1 the text repeats with period n-1-e."""

    period, out = n - 1 - e, []
    for j in range(count):
        x = e + 1 + j
        while x >= n:
            x -= period
        out.append(seq[x])
    return out


class Automaton:
    """An online suffix automaton over a token sequence (design 4.2 S2's reference): after each token, the longest
    suffix that occurred before, and an end of such an occurrence (the newest one this automaton recorded)."""

    def __init__(self) -> None:
        self.len, self.link, self.last_end, self.next = [0], [-1], [-1], [{}]
        self.last, self.n = 0, 0

    def extend(self, c: int) -> tuple[int, int]:
        i, self.n = self.n, self.n + 1
        cur = len(self.len)
        self.len.append(self.len[self.last] + 1)
        self.link.append(-1)
        self.last_end.append(i)
        self.next.append({})
        p = self.last
        while p != -1 and c not in self.next[p]:
            self.next[p][c] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][c]
            if self.len[p] + 1 == self.len[q]:
                self.link[cur] = q
            else:
                clone = len(self.len)
                self.len.append(self.len[p] + 1)
                self.link.append(self.link[q])
                self.last_end.append(self.last_end[q])
                self.next.append(dict(self.next[q]))
                while p != -1 and self.next[p].get(c) == q:
                    self.next[p][c] = clone
                    p = self.link[p]
                self.link[q] = self.link[cur] = clone
        self.last = cur
        r = self.link[cur]
        found = (self.len[r], self.last_end[r]) if r > 0 else (0, -1)
        if r > 0:
            self.last_end[r] = i                         # this end is now the newest occurrence of that suffix
        return found


def corpus_continuation(seq: Sequence[int], e: int, count: int) -> list[int]:
    out = []
    for x in range(e + 1, min(len(seq), e + 1 + count)):
        if seq[x] == -1:
            break
        out.append(seq[x])
    return out


class Corpus:
    """Finished replies in segments of up to ``segment`` tokens, oldest first; past ``tokens`` in all, the oldest
    segment goes (design 4.3's ring of frozen segments). Replies are separated by -1, which never matches."""

    def __init__(self, anchors: Sequence[int], tokens: int, segment: int) -> None:
        self.anchors, self.cap, self.segment = list(anchors), tokens, segment
        self.segments: list[tuple[list[int], dict[int, Index]]] = []      # (sizes of its replies, index per anchor)
        self.size = 0

    def add(self, reply: Sequence[int]) -> None:
        if self.cap <= 0 or not reply:
            return
        if not self.segments or sum(self.segments[-1][0]) >= self.segment:
            self.segments.append(([], {k: Index(k) for k in self.anchors}))
        sizes, indexes = self.segments[-1]
        sizes.append(len(reply))
        for index in indexes.values():
            index.extend([-1, *reply])
        self.size += len(reply)
        while self.size > self.cap:
            self.size -= sum(self.segments.pop(0)[0])

    def match(self, k: int, needle: Sequence[int], n: int, recent: int, longest: int) -> tuple[int, Index, int]:
        """The best match over every segment: the longest, then the newest. Returns (length, index, end)."""

        best = (0, None, -1)
        for _, indexes in reversed(self.segments):
            index = indexes[k]
            length, e = match(index, needle, n, len(index.seq), recent, longest)
            if length > best[0]:
                best = (length, index, e)
        return best


REQUIRED = ("rid", "prompt", "out", "count", "draft", "constrained", "background", "process", "order", "after",
            "dropped", "decoder", "rounds")


class Mtp:
    """MTP rounds: the recorded one at the same reply index, else a recorded one of the bucket, else Tier-0."""

    def __init__(self, traces: list[dict], drafts: int, accept: float) -> None:
        self.by_bucket: dict[str, list[tuple[int, int]]] = {}
        for tr in traces:
            for r in tr["rounds"]:
                if r["src"] == "mtp":
                    self.by_bucket.setdefault(bucket(len(tr["prompt"]) + r["n"]), []).append((r["rows"], r["acc"]))
        self.drafts, self.accept = drafts, accept

    def round(self, at: dict, i: int, context: int, room: int, left: int,
              rng: random.Random) -> tuple[int, int, str, str]:
        """(drafts, kept, source, how) for an MTP round at reply index ``i`` with ``room`` drafts allowed and
        ``left`` tokens to emit; how: exact | bucket | tier0. ``at``: the recorded rounds without copies, by index."""

        if i in at:
            rows, acc, src = at[i]
            return rows - 1, acc, src, "exact"
        pool = self.by_bucket.get(bucket(context))
        if pool:
            rows, acc = pool[rng.randrange(len(pool))]
            drafts, how = min(rows - 1, room), "bucket"
            acc = min(acc, drafts)
        else:
            drafts, acc, how = min(self.drafts, room), 0, "tier0"
            while acc < drafts and rng.random() < self.accept:
                acc += 1
        return drafts, min(acc, left - 1), "mtp" if drafts else "", how


def _stats() -> dict:
    return {"requests": 0, "rounds": 0, "tokens": 0, "rows": 0, "drafted": 0, "accepted": 0, "suffix_tokens": 0,
            "ms": 0.0, "measured_ms": 0.0, "mtp_exact": 0, "mtp_bucket": 0, "mtp_tier0": 0}


class Report:
    def __init__(self, cost) -> None:
        self.cost, self.data = cost, {}

    def add(self, policy: str, context: int, rows: int, acc: int, steps: int, src: str, measured: float = 0.0,
            how: str | None = None) -> None:
        for name in (bucket(context), "all"):
            st = self.data.setdefault(policy, {}).setdefault(name, _stats())
            st["rounds"] += 1
            st["tokens"] += acc + 1
            st["rows"] += rows
            st["drafted"] += rows - 1
            st["accepted"] += acc
            st["suffix_tokens"] += acc + 1 if src not in ("mtp", "") else 0
            st["ms"] += self.cost(rows, steps)
            st["measured_ms"] += measured
            if how is not None:
                st[f"mtp_{how}"] += 1

    def request(self, policy: str) -> None:
        self.data.setdefault(policy, {}).setdefault("all", _stats())["requests"] += 1

    def summary(self) -> dict:
        out = {}
        for policy, buckets in self.data.items():
            out[policy] = {}
            for name, st in buckets.items():
                row = dict(st)
                row["tokens_per_round"] = st["tokens"] / st["rounds"] if st["rounds"] else None
                row["acceptance"] = st["accepted"] / st["drafted"] if st["drafted"] else None
                row["coverage"] = st["suffix_tokens"] / st["tokens"] if st["tokens"] else None
                row["ms_per_round"] = st["ms"] / st["rounds"] if st["rounds"] else None
                row["tok_s"] = st["tokens"] / (st["ms"] / 1000) if st["ms"] else None
                row["measured_tok_s"] = st["tokens"] / (st["measured_ms"] / 1000) if st["measured_ms"] else None
                out[policy][name] = row
        return out


def simulate(tr: dict, policy, mtp: Mtp, report: Report, name: str, rng: random.Random,
             joins: bool = False) -> list[tuple]:
    """One request under ``policy(i, context_length, room) -> (drafts, source) | None`` (None: an MTP round).
    ``joins``: the policy drafts the first round too (the cost-aware rule drafts a joining prompt from its text), else
    the first round is MTP's. Returns its rounds (i, rows, kept, source)."""

    prompt, out, count = tr["prompt"], tr["out"], tr["count"]
    at = {r["n"]: (r["rows"], r["acc"], r["src"]) for r in tr["rounds"] if r["src"] in ("mtp", "")}
    rounds, i, first = [], 1, True
    report.request(name)
    while i < len(out):
        context = len(prompt) + i
        room = max(0, count - i)
        proposal = None if first and not joins else policy(i, context, room)
        how = None                                       # an MTP round's model: exact | bucket | tier0
        if proposal is None:
            drafts, acc, src, how = mtp.round(at, i, context, room, len(out) - i, rng)
            steps = max(1, drafts)
        else:
            proposal, src = proposal
            drafts, steps = len(proposal), 1
            acc = min(lcp(proposal, out, i), len(out) - i - 1)
        first = False
        report.add(name, context, 1 + drafts, acc, steps, src, how=how)
        rounds.append((i, 1 + drafts, acc, src))
        i += acc + 1
    return rounds


def thresholds(args) -> draft_rule.Thresholds:
    """The cost-aware rule's thresholds (tokens) at the modeled costs: ``--lam`` tokens a ms."""

    row, step = args.lam * args.row_ms, args.lam * args.step_ms
    return draft_rule.Thresholds(row, row + step, row + draft_rule.STEP_SHARED * step)


def is_greedy(sampling: dict | None) -> bool:
    return sampling is None or sampling["temperature"] <= 0


def text_plan(pred, greedy: bool, length: int, tokens: list[int], depth: int, room: int,
              th: draft_rule.Thresholds) -> list[int] | None:
    """The cost-aware rule's verified prefix of a text continuation, or None when an MTP chain is worth more (valued
    at its optimistic chances: the first pick's own chance is not in a trace)."""

    given = [pred.suffix(greedy, k, length) for k in range(len(tokens))]
    kept, value = draft_rule.suffix_plan(given, th)
    abar = [pred.abar("mtp", greedy, j) for j in range(max(1, min(depth, room)))]
    if value <= draft_rule.mtp_value(abar[0], abar, min(depth, room), th):
        return None
    return tokens[:len(kept)]


def _parity() -> dict:
    return {"checked": 0, "equal": 0, "differ": []}


def run(traces: list[dict], args) -> dict:
    cost = lambda rows, steps: args.t0 + (rows - 1) * args.row_ms + steps * args.step_ms  # noqa: E731
    report = Report(cost)
    mtp = Mtp(traces, args.mtp_drafts, args.mtp_accept)
    CopyIndex = copy_index_class()
    parity, parity_suffix = _parity(), _parity()
    th = thresholds(args)
    fitted = draft_rule.Predictor.load(args.tables) if args.tables else None     # else each decoder's prior
    by_process: dict[str, list[dict]] = {}
    for tr in traces:
        by_process.setdefault(tr["process"], []).append(tr)
    # per decoder process: its in-process corpus, and its finished requests still to add, in finish order
    corpora = {process: (Corpus(args.anchors, args.corpus_tokens, args.corpus_segment),
                         sorted(group, key=lambda t: t["order"])) for process, group in by_process.items()}
    for tr in sorted(traces, key=lambda t: (t["process"], t["after"], t["order"])):
        corpus, finished = corpora[tr["process"]]
        while finished and finished[0]["order"] <= tr["after"]:            # finished before this one joined
            done = finished.pop(0)
            if not done["background"]:
                corpus.add(done["out"])
        prompt, out = tr["prompt"], tr["out"]
        seq = list(prompt) + list(out)
        decoder = tr["decoder"]
        depth, needle = int(decoder["depth"]), int(decoder["copy_match"])
        anchor, recent, longest = (int(decoder.get(key, default)) for key, default in
                                   (("anchor", 4), ("recent", 32), ("longest", 64)))
        pred = fitted or draft_rule.Predictor.prior(float(decoder["confidence"]))
        greedy = is_greedy(tr["sampling"])
        old_rule = all(r.get("rule", "old") == "old" for r in tr["rounds"])
        recorded = []
        report.request("recorded")
        for r in tr["rounds"]:
            report.add("recorded", len(prompt) + r["n"], r["rows"], r["acc"],
                       max(1, r["rows"] - 1) if r["src"] == "mtp" else 1, r["src"], r["ms"])
            recorded.append((r["n"], r["rows"], r["acc"], r["src"]))
        if not tr["draft"] or tr["constrained"]:
            continue                                     # serial or grammar-pruned replies: recorded only

        copies = CopyIndex(needle)

        def copy8(i, context, room):
            cap = min(depth, room)
            if cap <= 0:
                return None
            drafts = copies.propose(Prefix(seq, context), max(cap, needle))[:cap]
            return (drafts, "copy") if drafts else None

        copies_new = CopyIndex(needle)

        def copy8_new(i, context, room):
            cap = min(depth, room)
            if cap <= 0:
                return None
            drafts = copies_new.propose(Prefix(seq, context), max(cap, needle))[:cap]
            kept = text_plan(pred, greedy, needle, drafts, depth, room, th) if drafts else None
            return (kept, "copy") if kept else None

        runtime = Index(anchor)
        runtime.extend(seq)

        def suffix_old(i, context, room):
            cap = min(depth, room)
            length, e = match(runtime, seq, context, context - 1, recent, longest) if cap > 0 else (0, -1)
            return (continuation(seq, e, context, cap), "suffix") if length >= draft_rule.COPY_MATCH else None

        def suffix_new(window, find):
            def propose(i, context, room):
                cap = min(window - 1, room)
                length, e = find(context) if cap > 0 else (0, -1)
                if not length:
                    return None
                kept = text_plan(pred, greedy, length, continuation(seq, e, context, cap), depth, room, th)
                return (kept, "suffix") if kept else None
            return propose

        s1 = lambda context: match(runtime, seq, context, context - 1, recent, longest)  # noqa: E731
        automaton, found = Automaton(), [(0, -1)]
        for t in seq:                                    # found[n]: the longest earlier match of seq[:n]'s end
            found.append(automaton.extend(t))
        s2 = lambda context: found[context] if found[context][0] >= anchor else (0, -1)  # noqa: E731

        def sam_old(i, context, room):
            cap = min(depth, room)
            length, e = s2(context) if cap > 0 else (0, -1)
            return (continuation(seq, e, context, cap), "suffix") if length >= draft_rule.COPY_MATCH else None

        def oracle(window):
            return lambda i, context, room: (out[i:i + min(window - 1, room)], "oracle")

        session = {}
        for k in args.anchors:
            index = Index(k)
            index.extend(seq)
            session[k] = index

        def suffix(k, window, use_corpus):
            def propose(i, context, room):
                cap = min(window - 1, room)
                if cap <= 0:
                    return None
                length, e = match(session[k], seq, context, context - 1, args.recent, args.longest)
                c_length = 0
                if use_corpus and corpus.size:
                    c_length, c_index, c_e = corpus.match(k, seq, context, args.recent, args.longest)
                if c_length > length:
                    drafts = corpus_continuation(c_index.seq, c_e, cap)
                    return (drafts, "corpus") if drafts else None
                if length:
                    return continuation(seq, e, context, cap), "session"
                return None
            return propose

        policies = [("copy8", copy8), ("copy8-new", copy8_new), ("suffix-old", suffix_old), ("sam-old", sam_old)]
        policies += [(f"suffix-new-w{w}", suffix_new(w, s1)) for w in args.windows]
        policies += [(f"sam-new-w{w}", suffix_new(w, s2)) for w in args.windows]
        rule_new = {"copy8-new", *(f"{p}-new-w{w}" for p in ("suffix", "sam") for w in args.windows)}  # join with text
        policies += [(f"session-k{k}-w{w}", suffix(k, w, False)) for k in args.anchors for w in args.windows]
        policies += [(f"corpus-k{k}-w{w}", suffix(k, w, True)) for k in args.anchors for w in args.windows]
        policies += [(f"oracle-w{w}", oracle(w)) for w in args.windows]
        text = tr.get("text", "copy" if decoder["copy"] else None)       # version 1 traces before "text": copies
        for name, policy in policies:
            rng = random.Random(f"{args.seed}:{tr['rid']}:{name}")
            got = simulate(tr, policy, mtp, report, name, rng, joins=name in rule_new)
            check = parity if name == "copy8" and text == "copy" else \
                parity_suffix if name == "suffix-old" and text == "suffix" else None
            if check is not None and old_rule and recorded:
                check["checked"] += 1
                if got == recorded:
                    check["equal"] += 1
                else:
                    check["differ"].append(tr["rid"])
    return {"params": {k: v for k, v in vars(args).items() if k != "traces"}, "requests": len(traces),
            "traces_dropped": sum(max(t["dropped"] for t in group) for group in by_process.values()),
            "parity": parity, "parity_suffix": parity_suffix,
            "results": report.summary()}


def _score(x):
    """A logged feature: "inf"/"-inf"/"nan" strings back to floats."""

    return float(x) if isinstance(x, str) else x


def rules(lines: list[dict], args) -> dict:
    """Stop rules on the probe rounds' labelled MTP chains (``--log``): exact per-round counterfactuals."""

    cost = lambda rows, steps: args.t0 + (rows - 1) * args.row_ms + steps * args.step_ms  # noqa: E731
    report, th = Report(cost), thresholds(args)
    preds = {"prior": draft_rule.Predictor.prior(args.confidence),
             **({"fit": draft_rule.Predictor.load(args.tables)} if args.tables else {})}
    skipped = {"unlabelled": 0, "no_greedy_field": 0, "no_features": 0}
    for line in lines:
        for st in line["streams"]:
            drafts = st["drafts"]
            if st["src"] != "mtp" or not st["probe"] or not drafts:
                continue
            if any(d["label"] is None for d in drafts):
                skipped["unlabelled"] += 1
                continue
            labels = [d["label"] for d in drafts]
            for name, k, steps in _decide(drafts, args.confidence, preds, st.get("greedy"), th, skipped, cost):
                acc = 0
                while acc < k and labels[acc] == 1:
                    acc += 1
                report.add(name, st["ctx"], 1 + k, acc, steps, "mtp" if k else "")
    return {"skipped": skipped, "results": report.summary()}


def _decide(drafts: list[dict], confidence: float, preds: dict, greedy, th, skipped: dict, cost):
    """(rule, drafts it verifies, MTP steps it runs) for one labelled probe chain (``cost(rows, steps)``: the
    modelled round milliseconds, for allocate)."""

    k = steps = 0
    for j, d in enumerate(drafts):
        steps = j + 1                                    # the absorb computes pick 0; each further pick a step
        low = confidence > 0 and d["p"] < confidence
        if not low or j == 0:
            k += 1
        if low:
            break
    yield "confidence", k, steps
    if greedy is None:
        skipped["no_greedy_field"] += 1
        return
    for name, pred in preds.items():
        feature = pred.tables["mtp", "greedy" if greedy else "sampled"].feature
        if any(d[feature] is None for d in drafts):
            skipped["no_features"] += 1
            continue
        c, k, steps = 1.0, 0, 0
        for j, d in enumerate(drafts):
            steps = j + 1
            a = pred.mtp(greedy, j, _score(d["p"]), _score(d["m"]))
            c, include, full, _ = draft_rule.chain_step(c, a, pred.abar("mtp", greedy, j + 1), j + 1 < len(drafts), th)
            k += include
            if not full:
                break
        yield name, k, steps
        if name == "prior":                              # allocate: the whole chain's chances, then its best prefix
            chances, c = [], 1.0
            for j, d in enumerate(drafts):
                c *= pred.mtp(greedy, j, _score(d["p"]), _score(d["m"]))
                chances.append(c)
            costs = {rows: cost(rows, len(drafts)) for rows in range(1, len(drafts) + 2)}
            yield "allocate", allocate([1], [chances], costs, 0.0, 1 + len(drafts))[0], len(drafts)


def load_log(paths: Sequence[str]) -> list[dict]:
    lines = []
    for path in paths:
        for text in Path(path).read_text().splitlines():
            if text.strip():
                lines.append(json.loads(text))
    return lines


def load(paths: Sequence[str]) -> list[dict]:
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    traces = []
    for f in files:
        tr = json.loads(f.read_text())
        missing = [key for key in REQUIRED if key not in tr]
        if tr.get("v") != 1 or missing or any(key not in tr["decoder"] for key in ("depth", "copy", "copy_match")):
            raise ValueError(f"{f}: not a version 1 draft trace (missing {missing or 'decoder fields'})")
        traces.append(tr)
    return traces


def table(result: dict) -> str:
    lines = _rows(result["results"])
    p = result["parity"]
    lines.append(f"copy8 parity with recorded rounds: {p['equal']}/{p['checked']} requests")
    p = result.get("parity_suffix")
    if p is not None:
        lines.append(f"suffix-old parity with recorded rounds: {p['equal']}/{p['checked']} requests")
    if result["traces_dropped"]:
        lines.append(f"warning: the server dropped {result['traces_dropped']} traces (a full queue)")
    if "rules" in result:
        lines.append("MTP stop rules on probe rounds (exact per round):")
        lines += _rows(result["rules"]["results"])
        lines.append(f"left out: {result['rules']['skipped']}")
    return "\n".join(lines)


def _rows(results: dict) -> list[str]:
    lines = [f"{'policy':<22} {'bucket':<9} {'rounds':>7} {'tokens':>8} {'tok/rnd':>7} {'accept':>6} {'cover':>6} "
             f"{'ms/rnd':>7} {'tok/s':>7}"]
    fmt = lambda x, f: "-" if x is None else format(x, f)  # noqa: E731
    for policy, buckets in results.items():
        for name in ["all"] + [b for _, b in BUCKETS]:
            st = buckets.get(name)
            if st is None or not st["rounds"]:
                continue
            lines.append(f"{policy:<22} {name:<9} {st['rounds']:>7} {st['tokens']:>8} "
                         f"{fmt(st['tokens_per_round'], '7.3f')} {fmt(st['acceptance'], '6.3f')} "
                         f"{fmt(st['coverage'], '6.3f')} {fmt(st['ms_per_round'], '7.2f')} {fmt(st['tok_s'], '7.2f')}")
    return lines


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="trace files or directories of them")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--t0", type=float, default=34.6, help="ms a round costs with one row (default 34.6)")
    ap.add_argument("--row-ms", type=float, default=6.5, help="ms each further verify row adds (default 6.5)")
    ap.add_argument("--step-ms", type=float, default=1.6, help="ms each MTP head step adds (default 1.6)")
    ap.add_argument("--anchors", type=_ints, default=[4, 6, 8], help="S1 anchor lengths (default 4,6,8)")
    ap.add_argument("--windows", type=_ints, default=[5, 8, 16], help="window caps in rows (default 5,8,16)")
    ap.add_argument("--recent", type=int, default=32, help="S1: the newest anchor occurrences compared (32)")
    ap.add_argument("--longest", type=int, default=64, help="S1: the longest backward match measured (64)")
    ap.add_argument("--corpus-tokens", type=int, default=1_000_000, help="corpus cap in tokens (0: no corpus)")
    ap.add_argument("--corpus-segment", type=int, default=262_144, help="corpus segment size in tokens")
    ap.add_argument("--mtp-drafts", type=int, default=2, help="Tier-0 MTP drafts a round (default 2)")
    ap.add_argument("--mtp-accept", type=float, default=0.75, help="Tier-0 MTP draft acceptance (default 0.75)")
    ap.add_argument("--seed", type=int, default=0, help="seeds the modeled MTP rounds")
    ap.add_argument("--log", nargs="*", default=[], help="TENSORFOLD_DRAFT_LOG files: compare MTP stop rules")
    ap.add_argument("--tables", help="a fitted draft rule (tools/fit_draft_rule.py) for the cost-aware rule's chances")
    ap.add_argument("--lam", type=float, default=draft_rule.LAMBDA_PRIOR,
                    help=f"lambda*: tokens a ms the rule's thresholds use (default {draft_rule.LAMBDA_PRIOR:.4f})")
    ap.add_argument("--confidence", type=float, default=0.5,
                    help="the confidence rule's threshold for log lines (default 0.5, production's)")
    args = ap.parse_args(argv)
    result = run(load(args.traces), args)
    if args.log:
        result["rules"] = rules(load_log(args.log), args)
    print(table(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
