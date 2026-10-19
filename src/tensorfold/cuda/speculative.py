"""One owned block-drafter readback: greedy argmax or sampled sparse-q proposals.

Greedy packets preserve their cumulative confidence and cutoff. Sampled packets hold a bounded candidate set,
draw under independent proposal noise and retain the actual normalized q for standard rejection verification.
They do not build unused greedy confidences. Both apply penalties under the reply and prior proposals on the path.
Historical2026-10-18 coupled/sharpened/offline rejection observations concern earlier implementations; they do not
select the current standard-rejection proposal schedule. ``Chains`` wraps one launch and retains foreign owners
until its actual synchronous readback completes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from tensorfold.cuda.sampling import candidate_fields, canonical_candidates, valid_scores
from tensorfold.engine.exact_sampling import MARGIN

# candidates a penalized stream reads per row: the top plus MARGIN, plus room for the tokens penalties push past it
PENALIZED_CANDIDATES = 1 + MARGIN + 32
PROPOSAL_CANDIDATES = 256  # explicit sparse proposal support; target probabilities keep complete-support fallback


def chain_workspace(rows: int, vocab: int, *, streams: int = 1, penalized: bool = True) -> dict[str, int]:
    """Preparation charge for a live sampled packet and its CPU FP64 rows.

    Source logits/foreign launch owners remain charged to their producing operation. The device sum includes the
    packet beside all candidate extraction/conversion inputs; host includes immutable q, decoded readback, Python
    headers and the sum of one row's normalization/sort/draw intermediates. Runtime/library reserves remain owned
    by the caller's admission policy; native peak checks validate that integration.
    """
    import sys

    from tensorfold.engine.rejection import ChainPacket, ProbabilityRow

    if any(type(v) is not int or v < 1 for v in (rows, vocab, streams)):
        raise ValueError("positive row, vocabulary and stream counts required")
    n, s = rows * streams, min(vocab, PROPOSAL_CANDIDATES + 1)
    kept = min(vocab, PROPOSAL_CANDIDATES)
    words = n * (1 + 2 * s) * 8
    allocated = lambda size: -(-size // 512) * 512
    device = sum(allocated(size) for size in [words, 4 * n * s, 4 * n * s, 8 * n * s, 8 * n * s,
                                             4 * n, 4 * n, 8 * n, 4 * vocab])
    example = ProbabilityRow([0], [1.0])
    row_header = sys.getsizeof(example) + sum(sys.getsizeof(a) + sys.getsizeof(a.base) - a.nbytes
                                             for a in (example.ids, example.probs))
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    packet_headers = streams * (sys.getsizeof(ChainPacket((), (), 0)) + 2 * sys.getsizeof(()))
    # A tied row can convert one complete BF16 source row to FP32 before host repair (4*vocab device bytes).
    # Seventeen FP64/int64 arrays and three Boolean masks upper-bound the explicit row preparation expressions;
    # they are charged together even where their lifetimes do not overlap. Drawing uses fewer buffers.
    scratch = max((17 * 8 + 3 + (16 if penalized else 0)) * s, (4 + 17 * 8 + 3) * vocab)
    host = (words + 4 * n * s + n * (16 * kept + row_header + sys.getsizeof(0) + 2 * pointer)
            + packet_headers + scratch)
    return {"device_bytes": device, "host_bytes": host}


class Chains:
    """A block drafter's launched chains, one per live stream, and their readback (one host wait, first ``finish``).

    ``logits`` [len(histories) * rows, V] holds stream-major rows. ``histories`` holds each stream's penalty
    history (``tensorfold.engine.penalties.History``) or None; ``finish`` refuses any other one. ``owners`` (tensors
    the launch's kernels read through pointers) stay alive until the readback completes. Fresh readback confidences
    are checked before publication; cached host data remains immutable for this launch's lifetime. Producer and
    readback use the current CUDA stream; callers crossing streams must establish their execution dependency.
    """

    def __init__(self, logits: torch.Tensor, histories: Sequence, rows: int, owners=None, *, sampled: bool = False):
        if logits.ndim != 2 or logits.shape[0] != len(histories) * rows:
            raise ValueError("draft rows must be stream-major, one block of rows per stream")
        self._sampled = sampled
        self._penalty_width = self._proposal_width = 0
        self._source = None
        if sampled:
            values, cols, flags = candidate_fields(logits, PROPOSAL_CANDIDATES)
            self._proposal_width = values.shape[1]
            self._source = logits
            self.words = torch.cat([flags[:, None], values.contiguous().view(torch.int32).to(torch.int64),
                                    cols], dim=1)
            self.histories, self.owners, self.host, self.rows = list(histories), owners, None, rows
            return
        best, tokens = logits.max(-1)
        scores = (torch.logsumexp(logits, -1) - best).view(len(histories), rows).cumsum(-1).view(-1)
        parts = [tokens[:, None], scores.float().view(torch.int32).to(torch.int64)[:, None]]
        if any(h is not None for h in histories):
            values, cols = torch.topk(logits.float(), min(PENALIZED_CANDIDATES, logits.shape[1]), dim=-1,
                                      sorted=False)
            self._penalty_width = values.shape[1]
            parts += [values.contiguous().view(torch.int32).to(torch.int64), cols]
        self.words = torch.cat(parts, dim=1)
        self.histories, self.owners, self.host, self.rows = list(histories), owners, None, rows

    def finish(self, row: int, max_nodes: int, history=None, *, stop: float | None = None
               ) -> tuple[list[int], list[int], list[float]]:
        """Stream ``row``'s greedy chain: (tokens, parents, cumulative ``-log`` confidences).

        ``stop`` is None or a finite nonnegative threshold validated by the owning drafter at configuration time.
        After the node cap, keep rows before the first confidence strictly above it, retaining one if any fit.
        """

        if self.histories[row] is not history:
            raise ValueError("a draft block must finish under the penalties it was launched with")
        if self._sampled:
            raise ValueError("a sampled-only packet must finish with sample(), not greedy finish()")
        self._readback()
        if history is None:
            ids, scores = self.host[0][row], self.host[1][row]
        else:
            ids, scores = penalized(*(part[row * self.rows:(row + 1) * self.rows] for part in self.host[2]), history)
        ids, scores = ids[:max_nodes], scores[:max_nodes]
        if stop is not None and ids:
            keep = max(1, next((i for i, score in enumerate(scores) if score > stop), len(scores)))
            ids, scores = ids[:keep], scores[:keep]
        return ids, list(range(-1, len(ids) - 1)), scores

    def _readback(self):
        if self.host is not None:
            return
        words = self.words.cpu().numpy()
        c, s = self._penalty_width, self._proposal_width
        if self._sampled:
            if words[:, 0].any():
                raise ValueError("proposal scores contain NaN or positive infinity")
            proposal = (words[:, 1:1 + s].astype(np.int32).view(np.float32), words[:, 1 + s:1 + 2 * s])
            def full_row(row):
                return self._source[row].float().cpu().numpy(), np.arange(self._source.shape[1], dtype=np.int64)

            proposal = canonical_candidates(*proposal, PROPOSAL_CANDIDATES, self._source.shape[1], full_row)
            self.host = None, None, None, proposal
            self.owners = self.words = self._source = None
            return
        confidence = words[:, 1].astype(np.int32).view(np.float32).reshape(-1, self.rows)
        if not np.isfinite(confidence).all():
            raise ValueError("draft confidence scores must be finite")
        self.host = (words[:, 0].reshape(-1, self.rows).tolist(), confidence.tolist(),
                     (words[:, 2:2 + c].astype(np.int32).view(np.float32), words[:, 2 + c:2 + 2 * c]) if c else None,
                     None)
        self.owners = self.words = None  # actual synchronous readback and validation completed

    def sample(self, row: int, max_nodes: int, sampling, first_position: int, history=None):
        """An immutable stochastic chain; logical width is chosen by the request, never remaining output room."""
        from tensorfold.engine.rejection import ChainPacket, draw, proposal_row

        if self.histories[row] is not history:
            raise ValueError("a draft block must finish under the penalties it was launched with")
        if not self._proposal_width:
            raise ValueError("a stochastic chain must be prepared with sampled=True")
        if type(max_nodes) is not int or max_nodes < 0:
            raise ValueError("proposal node count must be a nonnegative integer")
        self._readback()
        values, ids = self.host[3]
        start = row * self.rows
        tokens, distributions = [], []
        for j in range(min(max_nodes, self.rows)):
            at = start + j
            adjusted = values[at] if history is None else history.candidates(values[at:at + 1], ids[at:at + 1], tokens)[0]
            q = proposal_row(adjusted, ids[at], sampling)
            token = draw(q, sampling.seed, first_position + j)
            tokens.append(token)
            distributions.append(q)
        return ChainPacket(tuple(tokens), tuple(distributions), first_position)


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


__all__ = ["PENALIZED_CANDIDATES", "PROPOSAL_CANDIDATES", "Chains", "chain_workspace", "penalized"]
