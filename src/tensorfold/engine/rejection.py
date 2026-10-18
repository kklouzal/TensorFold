"""Keyed, distribution-exact speculative rejection sampling: the measured competitor of coupled verification.

Standard speculative sampling (Leviathan, Kalman and Matias 2023; Chen et al. 2023) accepts a draft ``x ~ q`` with
probability ``min(1, p(x) / q(x))`` and otherwise draws from the residual ``norm(max(p - q, 0))``; its output is
distributed exactly as ``p``. A node with several children drafted without replacement (the Gumbel-top-k order of
``q``) takes recursive rejection sampling (SpecInfer, Miao et al. 2024; without replacement: Jeon et al. 2024,
"Recursive Speculative Decoding"): child ``i`` is tested against the running residual ``r_i`` with the proposal ``q_i``
(``q`` without the earlier children, renormalized); a rejection sets ``r_{i+1} = norm(max(r_i - q_i, 0))``; when every
child is rejected the token is drawn from the last residual. A node without children emits the serial keyed draw.

Every random number is keyed, so a request's tokens are a fixed function of its seed and the proposals: acceptance
uniforms by (seed ^ ACCEPT, position, child index), residual Gumbel noise by (seed ^ RESIDUAL, position, token id),
independent of the target's own noise (seed, position, token id) that coupled drafts and serial draws use.

Preconditions for exactness, which callers own: each node's children are a prefix of the Gumbel-top-k order of the
``q`` passed for that node (sampled with noise independent of the acceptance and residual streams), and which nodes
exist (the tree's shape) does not depend on the noise that sampled them. Unlike coupled verification, the output then
depends on the proposals: same request, same proposals, same tokens.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from tensorfold.engine.exact_sampling import uniform

ACCEPT = 0x6A09E667F3BCC909      # first 64 fractional bits of sqrt(2): the acceptance uniforms' domain
RESIDUAL = 0xBB67AE8584CAA73B    # ... of sqrt(3): the residual draws' domain
_MASK = (1 << 64) - 1


def distribution(ids: np.ndarray, scaled: np.ndarray) -> dict[int, float]:
    """A filtered row (``tensorfold.engine.speculative.filtered``) as {token: probability}: the softmax of the kept
    scaled scores, the distribution the keyed Gumbel argmax over them samples."""

    keep = np.isfinite(scaled)
    if not keep.any():
        raise ValueError("a distribution needs at least one kept token")
    top = scaled[keep].max()
    weights = np.exp(scaled[keep] - top)
    return dict(zip((int(t) for t in ids[keep]), (weights / weights.sum()).tolist()))


def _normalized(weights: Mapping[int, float]) -> dict[int, float]:
    """``norm(max(w, 0))``: the positive part, renormalized ({} when nothing is positive)."""

    positive = {t: w for t, w in weights.items() if w > 0}
    total = sum(positive.values())
    return {t: w / total for t, w in positive.items()} if total > 0 else {}


def node(p: Mapping[int, float], q: Mapping[int, float], children: Sequence[int], seed: int,
         position: int) -> tuple[int, int]:
    """Recursive rejection sampling at one node whose children were drafted from ``q`` without replacement.

    Returns ``(i, token)``: ``i`` the accepted child's index (its token), or -1 with the residual draw. ``position``
    is the absolute position of the children's token (the node's keyed sampling position).
    """

    residual = dict(p)
    proposal = _normalized(q)
    for i, child in enumerate(children):
        qc = proposal.get(child, 0.0)
        if qc <= 0.0:
            raise ValueError("a child must be a token its proposal distribution can draw")
        u = float(uniform((seed ^ ACCEPT) & _MASK, position, np.array([i], dtype=np.int64))[0])
        if u * qc <= residual.get(child, 0.0):          # probability min(1, r(child) / q(child))
            return i, int(child)
        lifted = _normalized({t: residual.get(t, 0.0) - proposal.get(t, 0.0) for t in residual})
        # p == q leaves no residual mass: rejection then had probability zero, a rounding event; keep the residual
        residual = lifted or residual
        proposal.pop(child)
        proposal = _normalized(proposal)
    ids = np.array(sorted(residual), dtype=np.int64)
    weights = np.array([residual[int(t)] for t in ids], dtype=np.float64)
    noise = -np.log(-np.log(uniform((seed ^ RESIDUAL) & _MASK, position, ids)))
    return -1, int(ids[int(np.argmax(np.log(weights) + noise))])


def walk(tokens: Sequence[int], parents: Sequence[int], positions: Sequence[int], seed: int,
         p_rows: Sequence[Mapping[int, float] | None], q_rows: Sequence[Mapping[int, float] | None],
         serial: Sequence[int], room: int | None = None, eos: Sequence[int] = ()) -> tuple[list[int], int]:
    """Verify a draft tree: ``(path, terminal)``, the accepted rows from the root and the token after them.

    Row ``r`` inputs ``tokens[r]`` under parent ``parents[r]`` (-1 for the root, row 0); ``positions[r]`` is the
    absolute position its children / next token occupy. ``p_rows[r]`` is the target's filtered distribution at row
    ``r`` and ``q_rows[r]`` the one row ``r``'s children were drafted from (both needed only where ``r`` has children);
    ``serial[r]`` is the serial keyed draw at row ``r``, the token after a childless row. Children are taken in row
    order. ``room`` caps the tokens a round may emit (path drafts + terminal); an ``eos`` token ends the path.
    """

    if not (len(tokens) == len(parents) == len(positions) == len(p_rows) == len(q_rows) == len(serial)):
        raise ValueError("one parent, position, distribution pair and serial draw per row required")
    children: dict[int, list[int]] = {}
    for row in range(1, len(tokens)):
        if not 0 <= parents[row] < row:
            raise ValueError("tree parents must precede their rows")
        children.setdefault(parents[row], []).append(row)
    path, row = [0], 0
    while True:
        kids = children.get(row, [])
        if room is not None and len(path) >= room:        # the terminal fills the last slot
            kids = []
        if not kids:
            return path, int(serial[row])
        if p_rows[row] is None or q_rows[row] is None:
            raise ValueError("a row with children needs its target and proposal distributions")
        i, token = node(p_rows[row], q_rows[row], [tokens[k] for k in kids], seed, int(positions[row]))
        if i < 0:
            return path, token
        row = kids[i]
        path.append(row)
        if token in eos:
            return path[:-1], token


__all__ = ["ACCEPT", "RESIDUAL", "distribution", "node", "walk"]
