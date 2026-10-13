"""Benchmark-only variant switches for interleaved A/B runs; removed once the A/B has selected (R5).

``TENSORFOLD_AB_CONTROL=<json file>`` names a JSON object of switches. ``refresh()`` (once a round, and as a request
is admitted) re-reads it when its modification time changes, so a benchmark flips variants between requests without
restarting the engine.
Unset, ``get`` returns the default and ``refresh`` does nothing.
"""

from __future__ import annotations

import json
import os

ENV = "TENSORFOLD_AB_CONTROL"
_path = os.environ.get(ENV, "")
_values: dict = {}
_stamp: int | None = None


def refresh() -> None:
    global _values, _stamp
    if not _path:
        return
    try:
        stamp = os.stat(_path).st_mtime_ns
    except FileNotFoundError:
        _values, _stamp = {}, None
        return
    if stamp != _stamp:
        with open(_path, encoding="utf-8") as f:
            values = json.load(f)
        if not isinstance(values, dict):
            raise ValueError(f"{ENV} must name a JSON object")
        _values, _stamp = values, stamp


def get(key: str, default):
    return _values.get(key, default) if _path else default


def active() -> bool:
    """Whether a benchmark control file is configured (an A/B run): variants may need both sides prepared."""

    return bool(_path)


def new(key: str) -> bool:
    """A stage's switch: ``"new"`` (the default, the stage's behaviour) or ``"old"`` (the exact path it replaces)."""

    variant = get(key, "new")
    if variant not in ("new", "old"):
        raise ValueError(f'ab_switch "{key}" must be "new" or "old", not {variant!r}')
    return variant == "new"


def snapshot() -> dict:
    return dict(_values) if _path else {}
