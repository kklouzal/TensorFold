"""Block-drafter proposals for one launch: the drafter's argmax, penalized as the target's rows are, read back once.

GB10 card-sampling sweeps (2026-10-18) found no decode-speed gain over the DFlash drafters' argmax: proposals
drawn under the target's keyed noise (``tensorfold.engine.speculative.coupled``) ran 4-13% slower exactly coupled and
within -4..+4% sharpened, and the best sharpened coupled or rejection-sampled variants gained at most 0.4 points of
per-row acceptance offline, so block drafters propose their argmax for every request. A request with
presence/frequency/repetition penalties proposes the argmax of its penalized candidates instead: row ``j`` counts the
reply and proposals 0..j-1, exactly as the target's verify row ``j`` will, so penalized requests keep their
acceptance. ``Chains`` wraps one launch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from tensorfold.cuda.sampling import valid_scores
from tensorfold.engine.exact_sampling import MARGIN

# candidates a penalized stream reads per row: the top plus MARGIN, plus room for the tokens penalties push past it
PENALIZED_CANDIDATES = 1 + MARGIN + 32


class Chains:
    """A block drafter's launched chains, one per live stream, and their readback (one host wait, first ``finish``).

    ``logits`` [len(histories) * rows, V] holds stream-major rows. ``histories`` holds each stream's penalty
    history (``tensorfold.engine.penalties.History``) or None; ``finish`` refuses any other one. ``owners`` (tensors
    the launch's kernels read through pointers) stay alive until the readback completes. Fresh readback confidences
    are checked before publication; cached host data remains immutable for this launch's lifetime. Producer and
    readback use the current CUDA stream; callers crossing streams must establish their execution dependency.
    """

    def __init__(self, logits: torch.Tensor, histories: Sequence, rows: int, owners=None):
        if logits.ndim != 2 or logits.shape[0] != len(histories) * rows:
            raise ValueError("draft rows must be stream-major, one block of rows per stream")
        best, tokens = logits.max(-1)
        scores = (torch.logsumexp(logits, -1) - best).view(len(histories), rows).cumsum(-1).view(-1)
        parts = [tokens[:, None], scores.float().view(torch.int32).to(torch.int64)[:, None]]
        if any(h is not None for h in histories):
            values, cols = torch.topk(logits.float(), min(PENALIZED_CANDIDATES, logits.shape[1]), dim=-1,
                                      sorted=False)
            parts += [values.contiguous().view(torch.int32).to(torch.int64), cols]
        self.words = torch.cat(parts, dim=1)
        self.histories, self.owners, self.host, self.rows = list(histories), owners, None, rows

    def finish(self, row: int, max_nodes: int, history=None, *, stop: float | None = None
               ) -> tuple[list[int], list[int], list[float]]:
        """Stream ``row``'s chain: (tokens, parents, cumulative ``-log`` confidences).

        ``stop`` is None or a finite nonnegative threshold validated by the owning drafter at configuration time.
        After the node cap, keep rows before the first confidence strictly above it, retaining one if any fit.
        """

        if self.histories[row] is not history:
            raise ValueError("a draft block must finish under the penalties it was launched with")
        if self.host is None:
            words = self.words.cpu().numpy()
            c = (words.shape[1] - 2) // 2
            confidence = words[:, 1].astype(np.int32).view(np.float32).reshape(-1, self.rows)
            if not np.isfinite(confidence).all():
                raise ValueError("draft confidence scores must be finite")
            self.host = (words[:, 0].reshape(-1, self.rows).tolist(),
                         confidence.tolist(),
                         (words[:, 2:2 + c].astype(np.int32).view(np.float32), words[:, 2 + c:]) if c else None)
            self.owners = self.words = None
        if history is None:
            ids, scores = self.host[0][row], self.host[1][row]
        else:
            ids, scores = penalized(*(part[row * self.rows:(row + 1) * self.rows] for part in self.host[2]), history)
        ids, scores = ids[:max_nodes], scores[:max_nodes]
        if stop is not None and ids:
            keep = max(1, next((i for i, score in enumerate(scores) if score > stop), len(scores)))
            ids, scores = ids[:keep], scores[:keep]
        return ids, list(range(-1, len(ids) - 1)), scores


def penalized(values: np.ndarray, ids: np.ndarray, history) -> tuple[list[int], list[float]]:
    """A chain of the argmax of each row's penalized candidates, row ``j`` counting the proposals before it.

    Returns (tokens, cumulative ``-log`` temperature-1 shares over the candidates, like the argmax path's scores).
    """

    # NaN, +inf or a row without a finite score is a drafter defect: refuse it rather than propose a non-token. The
    # candidates are the row's top, so a finite top candidate is the complete row's condition.
    valid_scores(values)

    def pick(scores: np.ndarray, rows: slice) -> tuple[np.ndarray, np.ndarray]:
        best = np.argmax(scores, axis=-1)          # any maximum: a draft's tie order cannot change output
        at = np.arange(len(best))
        row = scores.astype(np.float64)
        return ids[rows][at, best], -np.log(np.exp(row - row[at, best][:, None]).sum(axis=-1))

    # every row penalized by the reply at once; a row whose candidates include an earlier proposal is redone with
    # the proposals before it counted, as the target's verify row will count them
    picks, logconf = pick(history.candidates(values, ids), slice(None))
    tokens: list[int] = []
    for j in range(len(picks)):
        if tokens and not set(tokens).isdisjoint(ids[j].tolist()):
            redone = pick(history.candidates(values[j:j + 1], ids[j:j + 1], tokens), slice(j, j + 1))
            picks[j], logconf[j] = redone[0][0], redone[1][0]
        tokens.append(int(picks[j]))
    confidence = np.cumsum(-logconf)
    if not np.isfinite(confidence).all():
        raise ValueError("draft confidence scores must be finite")
    return tokens, confidence.tolist()


__all__ = ["Chains", "PENALIZED_CANDIDATES", "penalized"]
