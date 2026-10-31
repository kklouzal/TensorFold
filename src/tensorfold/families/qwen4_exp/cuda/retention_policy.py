"""Explicit administrator retention promises, separate from logical API storage.

An unset policy preserves request-scoped execution and bounded best-effort
prefix reuse. RAM budgets cover backing; host transfer progress is separate.
Overlap is additional GPU bytes per rank and does not replace device floors.
"""

from dataclasses import dataclass
import math
import sys
import threading


SCHEMA = "tensorfold.retention-policy-v1"


def _integer(value, name, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= sys.maxsize:
        raise ValueError(f"retention {name} must be an explicit {'nonnegative' if zero else 'positive'} integer within the host size range")


@dataclass(frozen=True)
class RetentionPolicy:
    tier: str
    max_sessions: int
    lifetime_s: float
    ram_bytes: int | None = None
    chunk_bytes: int | None = None
    progress_bytes: int | None = None
    overlap_bytes: int | None = None

    def __post_init__(self):
        if self.tier not in ("gpu", "ram"):
            raise ValueError("retention tier must explicitly select gpu or ram")
        _integer(self.max_sessions, "session maximum")
        try:
            finite = type(self.lifetime_s) in (int, float) and math.isfinite(self.lifetime_s)
        except OverflowError:
            finite = False
        if not finite or not 0 < self.lifetime_s <= threading.TIMEOUT_MAX:
            raise ValueError("retention absolute lifetime must be positive and within the runtime wait range")
        if self.tier == "gpu":
            if any(value is not None for value in (self.ram_bytes, self.chunk_bytes, self.progress_bytes, self.overlap_bytes)):
                raise ValueError("GPU-only retention does not accept unused RAM/transfer bounds")
            return
        for name in ("ram_bytes", "chunk_bytes", "progress_bytes"):
            _integer(getattr(self, name), name)
        _integer(self.overlap_bytes, "overlap_bytes", zero=True)
        self.validate_world(1)

    def validate_world(self, world):
        _integer(world, "rank count")
        if self.tier == "ram":
            if self.chunk_bytes > sys.maxsize // (2 * world) or self.progress_bytes < 2 * world * self.chunk_bytes:
                raise ValueError("RAM retention progress must cover one pinned and one ordinary transfer chunk per rank")

    def record(self):
        return {"schema": SCHEMA, "tier": self.tier, "max_sessions": self.max_sessions,
                "lifetime_s": self.lifetime_s, "ram_bytes": self.ram_bytes, "chunk_bytes": self.chunk_bytes,
                "progress_bytes": self.progress_bytes, "overlap_bytes": self.overlap_bytes,
                "ram_scope": "global-host-backing", "progress_scope": "global-host-transfer",
                "overlap_scope": "per-rank-additional-gpu", "lifetime": "absolute-not-inactivity"}

    @classmethod
    def from_record(cls, record):
        fields = {"schema", "tier", "max_sessions", "lifetime_s", "ram_bytes", "chunk_bytes", "progress_bytes",
                  "overlap_bytes", "ram_scope", "progress_scope", "overlap_scope", "lifetime"}
        if (type(record) is not dict or set(record) != fields or record["schema"] != SCHEMA
                or record["ram_scope"] != "global-host-backing" or record["progress_scope"] != "global-host-transfer"
                or record["overlap_scope"] != "per-rank-additional-gpu" or record["lifetime"] != "absolute-not-inactivity"):
            raise ValueError("unsupported or ambiguous retention policy record")
        return cls(*(record[name] for name in ("tier", "max_sessions", "lifetime_s", "ram_bytes", "chunk_bytes",
                                               "progress_bytes", "overlap_bytes")))


def parse_retention_policy(*, retention_tier=None, retained_sessions=None, retention_lifetime_s=None,
                           retention_ram_bytes=None, retention_chunk_bytes=None,
                           retention_progress_bytes=None, retention_overlap_bytes=None):
    values = (retention_tier, retained_sessions, retention_lifetime_s, retention_ram_bytes,
              retention_chunk_bytes, retention_progress_bytes, retention_overlap_bytes)
    if all(value is None for value in values):
        return None
    if retention_tier is None:
        raise ValueError("retention bounds require an explicitly selected retention tier")
    return RetentionPolicy(retention_tier, retained_sessions, retention_lifetime_s, retention_ram_bytes,
                           retention_chunk_bytes, retention_progress_bytes, retention_overlap_bytes)
