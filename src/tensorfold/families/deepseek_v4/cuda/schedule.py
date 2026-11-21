"""Cost estimates for the DeepSeek-V4.1 SM70 Decoder's prompt layer groups (rank 0, host-side, pure Python).

A prompt chunk running while streams decode pauses between units (``Target.layers``: each layer's attention, then
its experts) so a decoding stream waits about one group plus one decode round. The groups are cut from these
estimates; they change no kernel or row, so estimate errors only move the decode gap and the prefill share."""

from __future__ import annotations

DEFAULT_ROUND_MS = 50.0     # a decode round's GPU time before any round of a comparable batch was measured


class UnitCosts:
    """Per-unit GPU time (ms) of a prompt chunk as a function of its rows: linear through two measured anchors
    (a small chunk, the startup warm-up, and the latest large one); with only the small anchor, proportional to
    the rows (an overestimate for larger chunks, since part of a unit's time is fixed per launch)."""

    def __init__(self, rows: int, ms: list[float]):
        if rows < 1 or not ms:
            raise ValueError("a unit cost sample needs rows >= 1 and one time per unit")
        self.small = (rows, list(ms))
        self.large = None

    def add(self, rows: int, ms: list[float]) -> None:
        """A completed chunk's per-unit times. Chunks of at least twice the small anchor's rows replace the large
        anchor; smaller ones replace the small anchor while staying at most half the large anchor's rows."""
        if len(ms) != len(self.small[1]):
            raise ValueError("a unit cost sample must time every unit")
        if rows >= 2 * self.small[0]:
            self.large = (rows, list(ms))
        elif rows >= 1 and (self.large is None or 2 * rows <= self.large[0]):
            self.small = (rows, list(ms))

    def predict(self, rows: int) -> list[float]:
        r0, m0 = self.small
        if self.large is None:
            return [ms * rows / r0 for ms in m0]
        r1, m1 = self.large
        out = []
        for a, b in zip(m0, m1):
            slope = max(0.0, (b - a) / (r1 - r0))
            out.append(max(0.0, a + slope * (rows - r0)))
        return out


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
