"""A history-dependent fake target for lane engine and lane server tests.

The next token depends on the WHOLE absorbed history, so any rollback, refeed or checkpoint mistake shows up as a
byte divergence from the fake's own serial decode. ``FakeFamily`` serves it as a lane-engine family.
"""

from __future__ import annotations

import time
from typing import Any

from tensorfold.engine.lane_engine import LaneEngine

VOCAB = 97


def fake_next(history: list[int]) -> int:
    return (sum(history) * 7 + len(history) * 3) % VOCAB


def fake_serial(prompt: list[int], max_new: int, eos: set[int]) -> list[int]:
    history = list(prompt)
    out: list[int] = []
    while len(out) < max_new:
        token = fake_next(history)
        out.append(token)
        history.append(token)
        if token in eos:
            break
    return out


class PatternProposer:
    """Proposes the true continuation for ``good`` tokens, then a wrong one; every proposal is confident (the family
    round verifies it as a copy)."""

    last_match = 1 << 30

    def __init__(self, pattern: list[int]) -> None:
        self.pattern = pattern
        self.calls = 0
        self.observed: list[tuple[int, int]] = []

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        good = self.pattern[self.calls % len(self.pattern)]
        self.calls += 1
        history = list(context)
        out: list[int] = []
        for j in range(max_draft):
            token = fake_next(history)
            if j >= good:
                token = (token + 1) % VOCAB
            out.append(token)
            history.append(token)
        return out

    def observe(self, proposed: int, accepted: int) -> None:
        self.observed.append((proposed, accepted))


class FakeBatchItem:
    """Absorbed history standing in for a cache layer (one row)."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = [list(r) for r in rows]

    @property
    def state(self) -> list[Any]:
        return []


class FakeFamily:
    """The fake target as a family: a cache holds the absorbed history, each row's hidden state carries the token the
    target picks after it, and a rollback trims the history. ``delay``: seconds a forward takes."""

    lane_family = True
    streams_exact = True
    exact_width = 16
    mtp = None
    gpu_tokens = False

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = float(delay)
        try:
            import mlx.core as mx
        except ModuleNotFoundError as error:
            if error.name not in ("mlx", "mlx.core"):
                raise
            self.cpu_test_arrays = True
        else:
            if not callable(mx.array):
                raise TypeError("the MLX test runtime requires its array constructor")
            self.cpu_test_arrays = False

    def make_cache(self) -> list[Any]:
        return [FakeBatchItem([[]])]

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> Any:
        import numpy as np

        if not self.cpu_test_arrays:
            import mlx.core as mx

        if self.delay:
            time.sleep(self.delay)
        history = cache[0].rows[0]
        out = []
        for token in np.array(inputs).reshape(-1).tolist():
            history.append(int(token))
            out.append(fake_next(history))
        return (np.asarray(out, dtype=np.float32) if self.cpu_test_arrays else mx.array(out, dtype=mx.float32)).reshape(
            1, -1, 1
        )

    def hidden_rows(self, windows: list[Any], caches: list[list[Any]], parents: Any = None) -> Any:
        if self.cpu_test_arrays:
            import numpy as np

            return np.concatenate(
                [self.hidden(np.asarray([w], dtype=np.uint32), c) for w, c in zip(windows, caches)], axis=1
            )
        import mlx.core as mx

        return mx.concatenate([self.hidden(w, c) for w, c in zip(windows, caches)], axis=1)

    def head(self, hidden: Any) -> Any:
        import numpy as np

        if not self.cpu_test_arrays:
            import mlx.core as mx

        picks = np.array(hidden).reshape(-1).astype(np.int64)
        logits = np.zeros((1, len(picks), VOCAB), dtype=np.float32)
        logits[0, np.arange(len(picks)), picks] = 10.0
        return logits if self.cpu_test_arrays else mx.array(logits)

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        kept = keep if isinstance(keep, int) else len(keep)
        if kept < rows:
            del cache[0].rows[0][kept - rows :]

    def keep_rows_streams(self, caches: list[list[Any]], lengths: Any, keeps: Any) -> None:
        for cache, rows, keep in zip(caches, lengths, keeps):
            self.keep_rows(cache, int(rows), keep)


class FakeEngine(LaneEngine):
    """LaneEngine over ``FakeFamily`` without a prefill plan; records each prefill's resume and checks its prefix."""

    prefill_plan = None

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        super().__init__(model if model is not None else FakeFamily(), **kwargs)
        self.prefill_calls: list[tuple[str, int]] = []

    def _family_prefill_steps(
        self, stream: Any, *, cache: list[Any] | None, cached_tokens: int, checkpoints_at: Any
    ) -> Any:
        cached = int(cached_tokens) if cache is not None else 0
        if cache is not None and list(cache[0].rows[0]) != list(stream.prompt_ids[:cached]):
            raise AssertionError("checkpoint was not a prefix of the prompt")
        self.prefill_calls.append((stream.stream_id, cached))
        return (
            yield from super()._family_prefill_steps(
                stream, cache=cache, cached_tokens=cached_tokens, checkpoints_at=checkpoints_at
            )
        )

    def _family_feed_steps(
        self, tokens, cache, chunks, prompt_data=None, wide=False, widths=None, raised=None, whole=None
    ):
        """Eager CPU test arrays; production prefill orchestration/checkpoints stay exercised.

        This fixture represents the deterministic fake history target only. It
        supplies no production MLX/Metal compatibility or numerical claim.
        """
        if not getattr(self.model, "cpu_test_arrays", False):
            return (
                yield from super()._family_feed_steps(tokens, cache, chunks, prompt_data, wide, widths, raised, whole)
            )
        if prompt_data is not None:
            raise ValueError("the CPU history fixture has no vision model")
        import numpy as np

        last = None
        feed = getattr(self.model, "prefill", None) or self.model.hidden
        passes = wide and getattr(self.model, "prompt_pass", True)
        together = getattr(self.model, "hidden_pass", None) if passes else None
        reach = max(1, int(self.prefill_pass)) if together is not None else 1
        self._fed_rows = 0
        chunks = list(chunks)
        ahead = getattr(self.model, "prefetch_prompt", None)
        if ahead is not None and chunks:
            ahead(tokens, chunks[0][0], chunks[min(reach, len(chunks)) - 1][1])
        n = 0
        while n < len(chunks):
            if n:
                yield
            width = self._pass_width(chunks, n, cache) if together is not None else 1
            span = chunks[n : n + width]
            begin, end = span[0][0], span[-1][1]
            n += width
            if widths is not None:
                widths.append(width)
            if ahead is not None and n < len(chunks):
                ahead(tokens, chunks[n][0], chunks[min(n + reach, len(chunks)) - 1][1])
            rows = [int(t) for t in tokens[begin:end]]
            if self.prefill_guard is not None:
                self.prefill_guard.before_chunk(cache, len(rows))
            if whole is not None:
                whole[0] = None
            inputs = np.asarray([rows], dtype=np.uint32)
            sizes = [b - a for a, b in span]
            if raised is not None:
                # Eager NumPy arrays have no MLX freed-buffer cache to raise.
                raised.append(False)
            hidden = together(inputs, cache, sizes) if width > 1 else feed(inputs, cache)
            self._fed_rows = len(rows)
            self.prefill_chunks += width
            self.prefill_tokens += len(rows)
            last = hidden[:, -1:, :]
            if getattr(self.model, "mtp", None) is not None:
                for a, b in span:
                    nxt = [int(t) for t in tokens[a + 1 : b + 1]]
                    if nxt:
                        at = a - begin
                        self.model.absorb_draft_context(
                            hidden[:, at : at + len(nxt)], np.asarray(nxt, dtype=np.uint32), cache, start=at
                        )
            if whole is not None:
                whole[0] = end
            if self.prefill_guard is not None:
                self.prefill_guard.after_chunk(cache, len(rows))
        return last

    def _draw(self, logits, sampling, positions):
        if not getattr(self.model, "cpu_test_arrays", False):
            return super()._draw(logits, sampling, positions)
        import numpy as np

        # The fake target's contract is deterministic greedy, also at the
        # server's requested temperature; native model sampling is separate.
        return np.asarray(logits).reshape(-1, VOCAB).argmax(-1).astype(np.uint32)

    def _family_first(self, stream, work, hidden, cached_tokens, row):
        if not getattr(self.model, "cpu_test_arrays", False):
            return super()._family_first(stream, work, hidden, cached_tokens, row)
        import numpy as np

        stream.emitted, stream.pending = [], []
        stream.cache_len, stream.cached_tokens = len(stream.prompt_ids), int(cached_tokens)
        stream.started_at = time.perf_counter()
        logits = self.model.head(hidden)
        if stream.constraint is not None:
            logits = stream.constraint.mask(logits)
        token = self._draw(logits, stream.sampling, [stream.cache_len])
        forced = self._forced_next(stream, token)
        return np.asarray([forced], dtype=np.uint32) if forced is not None else token

    def _fake_window(self, stream, copied=None):
        if stream.force:
            width = min(self.base_width, self.batch_rows) if stream.drafts else 1
            forced = list(stream.force[: width - 1])
            del stream.force[: len(forced)]
            return "forced", [int(stream.pending[-1]), *forced], forced
        drafts = copied if copied is not None else self._copy_proposal(stream) if stream.drafts else []
        return "copy" if drafts else "none", [int(stream.pending[-1]), *drafts], []

    def _fake_finish(self, stream, cache, kind, window, forced, logits):
        """Independent chain verification against the history oracle; no GPU arrays."""
        if stream.constraint is not None:
            logits = stream.constraint.mask(logits)
        sampled = self._draw(
            logits, stream.sampling, list(range(stream.cache_len + 1, stream.cache_len + 1 + len(window)))
        ).tolist()
        if kind == "forced":
            keep = len(window)
            committed = [*forced, stream.force.pop(0) if stream.force else sampled[-1]]
        else:
            keep = 1
            while keep < len(window) and window[keep] == sampled[keep - 1]:
                keep += 1
            committed = [*window[1:keep], sampled[keep - 1]]
            if len(window) > 1:
                accepted = keep - 1
                self.drafted += len(window) - 1
                self.accepted += accepted
                stream.drafted += len(window) - 1
                stream.accepted += accepted
                observe = getattr(stream.proposer, "observe", None)
                if callable(observe):
                    observe(len(window) - 1, accepted)
                if self._alone:
                    width = self._copy_width.get(stream.stream_id, self.first_copy)
                    self._copy_width[stream.stream_id] = (
                        min(self.max_copy, 2 * width + 1) if accepted == len(window) - 1 else self.first_copy
                    )
        cut = stream.think_cut(committed)
        if cut is not None:
            keep = min(keep, cut + 1)
            committed = [*committed[:cut], stream.start_close()]
        elif stream.call_gate is not None and (hit := stream.call_gate.cut(committed)) is not None:
            cut, fix = hit
            keep = min(keep, cut + 1)
            committed = [*committed[:cut], fix[0]]
            stream.force = list(fix[1:])
        stream.rounds += 1
        got = stream.commit(committed)
        if stream.finished and len(got) < keep:
            keep = min(keep, len(got) + 1)
        stream.cache_len += keep
        stream.pending = [committed[keep - 1]]
        self.model.keep_rows(cache, len(window), keep)
        return got, len(window), keep

    def _family_round(self, stream, cache, copied=None):
        if not getattr(self.model, "cpu_test_arrays", False):
            return super()._family_round(stream, cache, copied)
        import numpy as np

        kind, window, forced = self._fake_window(stream, copied)
        inputs = np.asarray([window], dtype=np.uint32)
        return self._fake_finish(
            stream, cache, kind, window, forced, self.model.head(self.model.hidden(inputs, cache))
        )

    def _family_round_streams(self, entries):
        if not getattr(self.model, "cpu_test_arrays", False):
            return super()._family_round_streams(entries)
        from tensorfold.engine.lane_engine import RoundStats

        started = time.perf_counter()
        plans = [(stream, cache, *self._fake_window(stream)) for stream, cache in entries]
        hidden = self.model.hidden_rows([p[3] for p in plans], [p[1] for p in plans])
        logits = self.model.head(hidden)
        results, offset = {}, 0
        for stream, cache, kind, window, forced in plans:
            results[stream.stream_id] = self._fake_finish(
                stream, cache, kind, window, forced, logits[:, offset : offset + len(window), :]
            )
            offset += len(window)
        elapsed = (time.perf_counter() - started) * 1000
        lengths = [len(p[3]) for p in plans]
        self.round_stats.append(
            RoundStats(
                streams=len(plans),
                width=max(lengths),
                rows=sum(lengths),
                ragged=len(set(lengths)) > 1,
                rollbacks=sum(result[2] < result[1] for result in results.values()),
                committed=sum(len(result[0]) for result in results.values()),
                forward_ms=elapsed,
                finalize_ms=0.0,
                rollback_ms=0.0,
                total_ms=elapsed,
                started_at=started,
            )
        )
        return results

    def _label_logits(self, hidden, label_ids):
        if not getattr(self.model, "cpu_test_arrays", False):
            return super()._label_logits(hidden, label_ids)
        import numpy as np

        row = self.model.head(hidden).reshape(-1, VOCAB)[-1].astype(np.float64)
        peak = row.max()
        return row[list(label_ids)].tolist(), float(peak + np.log(np.exp(row - peak).sum()))

    @staticmethod
    def copy_single_cache(cache: list[Any]) -> list[Any]:
        return [FakeBatchItem([list(r) for r in item.rows]) for item in cache]
