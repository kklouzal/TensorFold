"""Cost estimates for the DeepSeek-V4.1 SM70 Decoder's prompt layer groups (rank 0, host-side, pure Python).

A prompt chunk running while streams decode pauses between units (``Target.layers``: each layer's attention, then
its experts) so a decoding stream waits about one group plus one decode round. The groups are cut from these
estimates; they change no kernel or row, so estimate errors only move the decode gap and the prefill share."""

from __future__ import annotations

DEFAULT_ROUND_MS = 50.0     # a decode round's GPU time before any round of a comparable batch was measured


class UnitCosts:
    """Per-unit GPU time (ms) of a prompt chunk as a function of its rows, from two measured anchors: a small chunk
    (first the startup warm-up) and a large one, at least twice its rows. Between them, linear; beyond them, the
    bound the measurement gives (a unit's time never falls with more rows, and per row never rises, a part of it
    being fixed per launch): below the small anchor its time, above the large one its time scaled by the rows.
    The anchors' slope is never extrapolated: from two small chunks (fixed launch cost dominates, so their times
    are flat within noise) it would put a large chunk near the small one's cost and run it in one group."""

    def __init__(self, rows: int, ms: list[float]):
        if rows < 1 or not ms:
            raise ValueError("a unit cost sample needs rows >= 1 and one time per unit")
        self.small = (rows, list(ms))
        self.large = None

    def add(self, rows: int, ms: list[float]) -> None:
        """A completed chunk's per-unit times replace the anchor of its size: the large one for a chunk of at least
        half its rows (and twice the small one's), the small one for a chunk of at most twice its rows (and half the
        large one's). Without a large anchor, a chunk at least twice the small anchor's rows becomes it, and one at
        most half of them becomes the small anchor, the old one the large. Chunks in between change neither."""
        if rows < 1 or len(ms) != len(self.small[1]):
            raise ValueError("a unit cost sample must time every unit")
        r0 = self.small[0]
        if self.large is None:
            if rows >= 2 * r0:
                self.large = (rows, list(ms))
            elif 2 * rows <= r0:
                self.large, self.small = self.small, (rows, list(ms))
            else:
                self.small = (rows, list(ms))
            return
        r1 = self.large[0]
        if 2 * rows >= r1 and rows >= 2 * r0:
            self.large = (rows, list(ms))
        elif rows <= 2 * r0 and 2 * rows <= r1:
            self.small = (rows, list(ms))

    def predict(self, rows: int) -> list[float]:
        r0, m0 = self.small
        if rows <= r0:
            return list(m0)
        if self.large is None or rows >= self.large[0]:
            r1, m1 = self.large or self.small
            return [ms * rows / r1 for ms in m1]
        r1, m1 = self.large
        return [a + max(0.0, b - a) * (rows - r0) / (r1 - r0) for a, b in zip(m0, m1)]


class RoundCosts:
    """GPU time (ms) of a decode round per batch bucket: an exponential average of the measured rounds; for a
    bucket not measured yet, the tightest bound the measured buckets give (a wider bucket's time, or a narrower
    one's scaled by the width ratio: a round's time grows at most in proportion to its rows)."""

    def __init__(self):
        self.ms: dict[int, float] = {}

    def add(self, bucket: int, ms: float) -> None:
        old = self.ms.get(bucket)
        self.ms[bucket] = ms if old is None else 0.8 * old + 0.2 * ms

    def estimate(self, bucket: int) -> float:
        if bucket in self.ms:
            return self.ms[bucket]
        bounds = [ms if b > bucket else ms * bucket / b for b, ms in self.ms.items()]
        return min(bounds) if bounds else DEFAULT_ROUND_MS
