"""Opt-in draft telemetry for concurrent decoding (``MultiDecoder``); every part is off by default.

Off (``TENSORFOLD_DRAFT_LOG``, ``TENSORFOLD_DRAFT_PROBE`` and ``TENSORFOLD_DRAFT_TRACE`` unset), the decoder holds
``None`` and each hook costs one ``is not None`` test: no margins, threads, files or per-stream state. On, records leave
the engine thread through ``record_writer``'s bounded queues (the oldest is dropped, and counted, when a writer falls
behind) and daemon threads write them; ``MultiDecoder.close_telemetry`` (the engine's shutdown) drains them. Nothing
here changes a draft that is verified or a token that is emitted, but the hooks run on the engine thread: an error in
one fails the round's requests like any other error in the round.

``TENSORFOLD_DRAFT_LOG=<path>`` appends one JSON line a round, to calibrate draft acceptance. It holds no token ids
or text. A round's line is written once every draft it verified, and every shadow draft beside them, has its label:
when the stream's reply reaches the draft's position, or the stream ends (then the label stays null)::

    {"t": unix seconds, "live": streams verified, "rows": verify rows, "mixed": a prompt pass shared the forward,
     "pass_rows": its prompt rows, "steps": MTP head steps that drafted the next round (absorb, then the chain),
     "probed": that drafting was a probe, "joined": prompts that joined since the previous line, "ms": the round's
     wall time (from after any prompt passes that ran alone),
     "dropped": lines lost to a full queue so far, "forced": lines written before their labels resolved so far,
     "streams": [{"rid": a random id per request, "ctx": committed context tokens, "n": reply tokens before the
                  round, "src": "mtp" | "copy" | "" (no drafts), "probe": drafted in a probe, "join": drafted as
                  its prompt joined (single-stream drafting: no probability or margins), "constrained": a grammar
                  prunes its drafts (``inc`` false then also marks pruned ones), "verified": drafts verified,
                  "accepted": drafts kept, "done": the stream ended in this round,
                  "drafts": [{"j": depth, "p": the draft head's temperature-1 probability, "m": keyed margin,
                              "e": edge margin, "g": top-two logit gap, "z": log-sum-exp minus the top logit
                              (``MultiDecoder._picks``), "inc": verified, "hit": equals the reply's token at its
                              position, "label": hit, as did every earlier draft of its chain}]}]}

Copy and joined drafts carry null ``p``, ``m``, ``e``, ``g`` and ``z``; a margin of +inf is the string "inf" (NaN,
which valid scores never give, would be "nan"). Shadow drafts (``inc`` false) were computed but not verified: dropped
by the confidence rule, or chained past it in a probe.
Labels are exact because the reply does not depend on what was drafted. A draft's verified acceptance is its label
unless the stream's room or an end token cut the round first.

``TENSORFOLD_DRAFT_PROBE=<0..1>`` (with the log): the share of rounds whose MTP chain runs to full depth whatever the
confidence rule decides; the rule's drafts alone are verified, the rest are labelled shadows. A probe costs MTP head
steps, never tokens: the chained entries are trimmed before the stream's next absorb. Which rounds probe follows a
fixed-seed generator, so the same sequence of rounds probes the same ones.

``TENSORFOLD_DRAFT_TRACE=<dir>`` writes one file a finished request, ``<dir>/<start>-<rid>.json``, mode 0600. IT
PERSISTS CONTENT, the prompt's and the reply's token ids: enable it only for benchmark runs over public prompts. It is
the input of ``tools/draft_replay.py``::

    {"v": 1, "rid", "start", "end": unix seconds (first token, finish), "prompt": [ids], "out": [ids],
     "count": the token limit, "stop_eos", "sampling": {"seed", "temperature", "top_k", "top_p", "min_p"} or null
     (greedy), "draft": drafts verified (false: serial), "constrained": a grammar pruned drafts, "cached": prompt
     tokens resumed, "background", "error": its type's name or null,
     "process": a random id per decoder, "order": this request's place among the decoder's traced finishes (1, 2,
     ...), "after": how many had finished when it joined (an in-process corpus could hold exactly those),
     "decoder": {"depth", "confidence", "copy", "copy_match"}, "dropped": earlier traces lost to a full queue,
     "rounds": [{"n": reply tokens before it, "rows": verify rows, "acc": drafts kept, "src", "live", "mixed",
                 "ms"}]}

A stream's rounds are held until it finishes: one small record a round, at most its token limit.
"""

from __future__ import annotations

import math
import os
import random
import secrets
import time

from tensorfold.cuda.record_writer import JsonFiles, JsonLines, close_all

LOG_ENV, PROBE_ENV, TRACE_ENV = "TENSORFOLD_DRAFT_LOG", "TENSORFOLD_DRAFT_PROBE", "TENSORFOLD_DRAFT_TRACE"
OPEN = 4096          # round lines waiting for labels; past this the oldest is written with its open labels null


def _num(x: float | None, digits: int = 5):
    """JSON for a feature: rounded; +inf (nothing to compare against) as the string "inf"."""

    if x is None:
        return None
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return round(float(x), digits)


class _Line:
    """A round's log line until every draft in it has its label."""

    __slots__ = ("key", "record", "open", "written")

    def __init__(self, key: int, record: dict) -> None:
        self.key, self.record, self.open, self.written = key, record, 0, False


class _Chain:
    """One stream's drafting for a round: its tokens from reply index ``base`` on, and their entries in a line."""

    __slots__ = ("base", "tokens", "entries", "src", "probe", "join", "line", "done")

    def __init__(self, base: int, tokens: list[int], entries: list[dict], src: str, probe: bool, join: bool) -> None:
        self.base, self.tokens, self.entries = base, tokens, entries
        self.src, self.probe, self.join = src, probe, join
        self.line: _Line | None = None
        self.done = 0                                    # entries labelled (a prefix: positions ascend)


class DraftWatch:
    """The decoder's hooks: ``joined``, ``drafted``, ``verified``, then ``round_end`` a round; ``finished`` a stream."""

    def __init__(self, *, log: str = "", probe: float = 0.0, trace: str = "", decoder: dict | None = None,
                 seed: int = 0, queue_size: int = 4096) -> None:
        if not (math.isfinite(probe) and 0.0 <= probe <= 1.0):
            raise ValueError(f"{PROBE_ENV}={probe}: a probability from 0 to 1")
        if probe > 0.0 and not log:
            raise ValueError(f"{PROBE_ENV} labels its extra drafts in {LOG_ENV}: set both, or neither")
        self.rate = probe
        self.rng = random.Random(seed)                   # which rounds probe: telemetry only, never reaches a reply
        self.decoder = dict(decoder or {})
        self.log = JsonLines(log, name="tensorfold-draft-log", stamp=self._stamp, queue_size=queue_size) \
            if log else None
        self.trace = JsonFiles(trace, name="tensorfold-draft-trace") if trace else None
        self.rids: dict[int, str] = {}                   # stream id -> its random request id
        self.pending: dict[int, _Chain] = {}             # stream id -> the drafts its next round verifies
        self.chains: dict[int, list[_Chain]] = {}        # stream id -> verified chains with labels still open
        self.lines: dict[int, _Line] = {}                # lines waiting for labels, oldest first
        self.serial = 0
        self.forced = 0
        self.traces: dict[int, dict] = {}                # stream id -> its trace so far
        self.process = secrets.token_hex(8)
        self.finishes = 0                                # traced requests finished so far
        self.joins = 0                                   # prompts that joined since the last line
        self.begin()

    @classmethod
    def from_env(cls, decoder: dict | None = None) -> DraftWatch | None:
        log, trace = os.environ.get(LOG_ENV, ""), os.environ.get(TRACE_ENV, "")
        raw = os.environ.get(PROBE_ENV, "").strip()
        try:
            probe = float(raw) if raw else 0.0
        except ValueError:
            raise ValueError(f"{PROBE_ENV}={raw}: a probability from 0 to 1") from None
        if not (log or trace or probe):
            return None
        return cls(log=log, probe=probe, trace=trace, decoder=decoder)

    @property
    def margins(self) -> bool:
        """Drafts need their keyed margins (``MultiDecoder._picks``): only the log reads them."""

        return self.log is not None

    def writers(self) -> list:
        return [w for w in (self.log, self.trace) if w is not None]

    def close(self, timeout: float = 10.0) -> None:
        """Write what is queued (lines still waiting for labels stay unwritten), stop every writer, then raise their
        failures (``record_writer.close_all``)."""

        close_all(self.writers(), timeout)

    # -- a round -------------------------------------------------------------------------------------------------

    def begin(self) -> None:
        """A round starts (again, after passes run alone): its clock, and no entries left from a failed round."""

        self.t0 = time.perf_counter()
        self.round: list[tuple[object, dict, _Chain | None]] = []
        self.trace_rounds: list[dict] = []
        self.probed = False

    def probing(self) -> bool:
        """Whether this round's MTP chain runs to full depth (asked once a round, by its drafting)."""

        self.probed = self.rate > 0.0 and self.rng.random() < self.rate
        return self.probed

    def joined(self, s) -> None:
        """A prompt joined the rounds with its first token; ``s.drafts`` are its first drafts (single-stream)."""

        self.rids[s.sid] = secrets.token_hex(8)
        self.joins += 1
        if self.trace is not None:
            self.traces[s.sid] = {"start": time.time(), "after": self.finishes, "rounds": []}
        if s.drafts:
            self.drafted(s, len(s.out), "mtp", [(d, j, None, None) for j, d in enumerate(s.drafts)], join=True)

    def drafted(self, s, base: int, src: str, items: list, *, probe: bool = False, join: bool = False) -> None:
        """The drafts ``items`` ``[(token, depth, probability, margins)]`` the stream computed for its next round, in
        chain order from reply index ``base``: the ones its drafts hold first, then any the rule dropped or a probe
        chained past it."""

        entries = []
        if self.log is not None:
            for _, j, p, scores in items:
                scores = scores or {}
                entries.append({"j": j, "p": None if p is None else float(f"{p:.6g}"), "m": _num(scores.get("m")),
                                "e": _num(scores.get("e")), "g": _num(scores.get("g")), "z": _num(scores.get("z")),
                                "inc": False, "hit": None, "label": None})
        self.pending[s.sid] = _Chain(base, [int(item[0]) for item in items], entries, src, probe, join)

    def verified(self, s, ctx: int, rows: int, accepted: int) -> None:
        """The stream's window of ``rows`` rows (its pending token, then its drafts) kept ``accepted`` drafts; called
        before its new tokens are taken, at committed context length ``ctx``."""

        chain = self.pending.pop(s.sid, None)
        src = chain.src if chain is not None else ""
        if self.log is not None:
            drafts = chain.entries if chain is not None else []
            for i, entry in enumerate(drafts):
                entry["inc"] = i < rows - 1
            self.round.append((s, {"rid": self.rids[s.sid], "ctx": ctx, "n": len(s.out), "src": src,
                                   "probe": chain is not None and chain.probe, "join": chain is not None and chain.join,
                                   "constrained": s.constraint is not None, "verified": rows - 1,
                                   "accepted": accepted, "done": False, "drafts": drafts}, chain))
        trace = self.traces.get(s.sid)
        if trace is not None:
            entry = {"n": len(s.out), "rows": rows, "acc": accepted, "src": src}
            trace["rounds"].append(entry)
            self.trace_rounds.append(entry)

    def round_end(self, live: list, *, rows: int, mixed: bool, pass_rows: int, steps: int) -> None:
        """The round's streams took their tokens (and prompts that ended in its pass joined): label, then write."""

        ms = round((time.perf_counter() - self.t0) * 1e3, 3)
        for entry in self.trace_rounds:
            entry.update(live=len(live), mixed=mixed, ms=ms)
        if self.log is not None:
            self.serial += 1
            line = self.lines[self.serial] = _Line(self.serial, {
                "t": time.time(), "live": len(live), "rows": rows, "mixed": mixed, "pass_rows": pass_rows,
                "steps": steps, "probed": self.probed, "joined": self.joins, "ms": ms, "streams": []})
            for s, entry, chain in self.round:
                entry["done"] = bool(s.done)
                line.record["streams"].append(entry)
                if chain is not None and chain.tokens:
                    chain.line = line
                    line.open += len(chain.tokens)
                    self.chains.setdefault(s.sid, []).append(chain)
            for s in live:
                self._label(s)
            if line.open == 0:
                self._write(line)
            while len(self.lines) > OPEN:                # a stream that stays unlabelled must not hold memory
                self.forced += 1
                self._write(next(iter(self.lines.values())))
            self.joins = 0
        self.begin()

    def finished(self, s) -> None:
        """The stream left the decoder (finished, cancelled, yielded or dropped): what its reply did not reach stays
        unlabelled; a finished stream's trace is written."""

        self.pending.pop(s.sid, None)
        if self.log is not None:
            self._label(s)
            for chain in self.chains.pop(s.sid, []):
                if not chain.line.written:
                    chain.line.open -= len(chain.tokens) - chain.done
                    if chain.line.open == 0:
                        self._write(chain.line)
        rid = self.rids.pop(s.sid, None)
        trace = self.traces.pop(s.sid, None)
        if trace is not None and s.done and s.out:
            sampling = s.sampling
            self.finishes += 1
            self.trace.put((f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(trace['start']))}-{rid}.json", {
                "v": 1, "rid": rid, "start": trace["start"], "end": time.time(),
                "prompt": list(s.prompt), "out": list(s.out), "count": s.count, "stop_eos": s.stop_eos,
                "sampling": None if sampling is None else {
                    "seed": sampling.seed, "temperature": sampling.temperature, "top_k": sampling.top_k,
                    "top_p": sampling.top_p, "min_p": sampling.min_p},
                "draft": s.draft, "constrained": s.constraint is not None, "cached": s.cached,
                "background": s.background, "error": None if s.error is None else type(s.error).__name__,
                "process": self.process, "order": self.finishes, "after": trace["after"],
                "decoder": self.decoder, "rounds": trace["rounds"]}))

    # -- labels --------------------------------------------------------------------------------------------------

    def _label(self, s) -> None:
        """Label every draft the stream's reply now reaches: ``hit`` its token, ``label`` its whole chain so far."""

        chains = self.chains.get(s.sid)
        if not chains:
            return
        out, n = s.out, len(s.out)
        left = []
        for chain in chains:
            if chain.line.written:                       # forced out unlabelled: its entries are the writer's now
                continue
            while chain.done < len(chain.tokens) and chain.base + chain.done < n:
                entry = chain.entries[chain.done]
                hit = out[chain.base + chain.done] == chain.tokens[chain.done]
                entry["hit"] = int(hit)
                entry["label"] = int(hit and (chain.done == 0 or chain.entries[chain.done - 1]["label"] == 1))
                chain.done += 1
                chain.line.open -= 1
            if chain.line.open == 0:
                self._write(chain.line)
            if chain.done < len(chain.tokens):
                left.append(chain)
        if left:
            self.chains[s.sid] = left
        else:
            del self.chains[s.sid]

    def _write(self, line: _Line) -> None:
        if not line.written:
            line.written = True
            del self.lines[line.key]
            self.log.put(line.record)

    def _stamp(self, record: dict) -> None:
        record["dropped"] = self.log.dropped
        record["forced"] = self.forced
