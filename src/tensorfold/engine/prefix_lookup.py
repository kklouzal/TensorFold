"""Strict-prefix lookup for engines owning ordered (token IDs, snapshot) caches."""

from __future__ import annotations


def _resume(self, prompt: list[int]):
    """The longest kept state the prompt extends (with at least one new token), or None."""

    best = None
    for ids, snap in self.cache:
        if len(ids) < len(prompt) and prompt[:len(ids)] == ids and (best is None or len(ids) > len(best[0])):
            best = (ids, snap)
    return best

