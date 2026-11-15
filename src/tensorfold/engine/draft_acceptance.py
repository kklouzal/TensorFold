"""Target-choice verification of anchored draft chains."""

from __future__ import annotations


def accepted_prefix(sampled, inputs, room: int, eos) -> list[int]:
    """Sample after the anchor; a proposal survives only its parent's choice."""
    chosen = []
    for row, token in enumerate(sampled):
        chosen.append(token)
        if len(chosen) >= room or token in eos or row + 1 >= len(inputs) or token != inputs[row + 1]:
            break
    return chosen

