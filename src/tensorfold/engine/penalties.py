"""Presence, frequency and repetition penalties (OpenAI / vLLM semantics), applied before the keyed sampler.

For each row of next-token scores ``z`` (the model's final logits) with ``c[t]`` = how often token ``t`` occurs in
the reply generated so far and ``prompt`` = the distinct tokens of the request's prompt:

  repetition (Hugging Face ``RepetitionPenaltyLogitsProcessor``, vLLM ``apply_repetition_penalties``): for every ``t``
      in the prompt or the reply, ``z[t] = z[t] / r`` if ``z[t] > 0`` else ``z[t] * r`` (computed as vLLM does, in
      float32 with ``/ r`` a multiply by the float32 reciprocal ``1 / r``; where an extreme ``r`` takes that product
      out of float32's finite range, which would hand vLLM's sampler +-inf or NaN, a finite score saturates at
      float32's largest magnitude and a zero stays zero, so penalties never turn a score into a non-score);
  then frequency and presence (OpenAI API; vLLM ``apply_penalties``): ``z[t] -= f * c[t] + p * (c[t] > 0)``.

vLLM applies these to the float32 logits before temperature, min-p, top-k and top-p (``Sampler``), and so does
TensorFold, ahead of its keyed draw; greedy requests take the argmax of the penalized scores. Ranges follow vLLM
``SamplingParams``: presence and frequency in [-2, 2], repetition finite and > 0; (0, 0, 1) is off.

A speculative verify row is penalized with the tokens before it: the committed reply plus the drafts on its path
(vLLM's rejection sampler likewise appends draft tokens to the output tokens it penalizes), so drafted output stays
the serial output. ``History`` keeps one request's counts on the host; ``apply`` penalizes CUDA score rows (in place
when they are float32), ``candidates`` a drafter's host candidate rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

F32_MAX = float(np.finfo(np.float32).max)


def _factors(repetition: float) -> tuple[np.float32, np.float32]:
    """vLLM's float32 repetition factors ``(1 / r, r)`` (0 or inf where ``r`` passes float32's limits)."""

    with np.errstate(over="ignore", divide="ignore"):
        r = np.float32(repetition)
        return np.float32(1) / r, r


@dataclass(frozen=True)
class Penalties:
    presence: float = 0.0
    frequency: float = 0.0
    repetition: float = 1.0
    # tokens of the engine's prompt that are the request's prompt; any after them are reply tokens a gated
    # continuation re-reads as prompt (call_gate), counted as reply
    prompt_length: int | None = None

    def __post_init__(self) -> None:
        for name in ("presence", "frequency", "repetition"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} penalty must be a finite number")
        if not -2.0 <= self.presence <= 2.0 or not -2.0 <= self.frequency <= 2.0:
            raise ValueError("presence and frequency penalties must be between -2 and 2")
        if self.repetition <= 0.0:
            raise ValueError("repetition penalty must be greater than 0")
        if self.prompt_length is not None and (type(self.prompt_length) is not int or self.prompt_length < 0):
            raise ValueError("penalty prompt length must be a nonnegative integer")

    @property
    def active(self) -> bool:
        return self.presence != 0.0 or self.frequency != 0.0 or self.repetition != 1.0


def penalize(values: np.ndarray, counts: np.ndarray, in_prompt: np.ndarray, p: Penalties) -> np.ndarray:
    """The rule on float32 score arrays (broadcasting): the host reference ``History.apply`` matches."""

    values = np.asarray(values, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.float32)
    seen = counts > 0
    if p.repetition != 1.0:
        inverse, r = _factors(p.repetition)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = np.where(values > 0, values * inverse, values * r)
        scaled = np.where(np.isfinite(values), np.where(values == 0, values, np.clip(scaled, -F32_MAX, F32_MAX)),
                          values)         # finite scores saturate, zeros stay, -inf masks stay masks
        values = np.where(in_prompt | seen, scaled, values)
    return (values - np.float32(p.frequency) * counts - np.float32(p.presence) * seen.astype(np.float32)
            ).astype(np.float32)


class History:
    """One request's penalty state: the prompt's distinct tokens (repetition only) and the reply's token counts.

    ``extend`` after every committed token (the first sampled one included). Not thread-safe: the request's decode
    loop owns it. ``_ids`` (sorted distinct penalized ids), ``_counts`` (reply counts) and ``_in_prompt`` stay aligned.
    """

    def __init__(self, penalties: Penalties, prompt: Sequence[int]):
        if not penalties.active:
            raise ValueError("a penalty history needs an active penalty")
        self.penalties = penalties
        cut = len(prompt) if penalties.prompt_length is None else penalties.prompt_length
        if cut > len(prompt):
            raise ValueError("penalty prompt length exceeds the engine prompt")
        prompt_ids = np.asarray(prompt[:cut], dtype=np.int64)
        self._ids = np.unique(prompt_ids) if penalties.repetition != 1.0 else np.empty(0, np.int64)
        self._counts = np.zeros(len(self._ids), dtype=np.float32)
        self._in_prompt = np.ones(len(self._ids), dtype=bool)
        self.extend(prompt[cut:])

    def extend(self, tokens: Sequence[int]) -> None:
        fresh: dict[int, list[int]] = {}            # ids new to the history -> [insert position, count]
        for token in tokens:
            at = int(np.searchsorted(self._ids, token))
            if at < len(self._ids) and self._ids[at] == token:
                self._counts[at] += 1
            else:
                fresh.setdefault(int(token), [at, 0])[1] += 1
        if fresh:                                   # inserted together, once per call (positions in the old arrays)
            new = sorted(fresh)
            at = [fresh[t][0] for t in new]
            self._ids = np.insert(self._ids, at, new)
            self._counts = np.insert(self._counts, at, [fresh[t][1] for t in new])
            self._in_prompt = np.insert(self._in_prompt, at, False)

    def count(self, token: int) -> int:
        at = int(np.searchsorted(self._ids, token))
        return int(self._counts[at]) if at < len(self._ids) and self._ids[at] == token else 0

    def _lookup(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reply counts and prompt flags of arbitrary ids (zeros/False for ids without a penalty)."""

        if not len(self._ids):
            return np.zeros(ids.shape, np.float32), np.zeros(ids.shape, bool)
        at = np.searchsorted(self._ids, ids).clip(max=len(self._ids) - 1)
        hit = self._ids[at] == ids
        return np.where(hit, self._counts[at], np.float32(0)), hit & self._in_prompt[at]

    def candidates(self, values: np.ndarray, ids: np.ndarray, path: Sequence[int] = ()) -> np.ndarray:
        """Penalized float32 candidate scores, as if ``path`` (uncommitted drafts) followed the reply."""

        counts, in_prompt = self._lookup(ids)
        for token in path:
            counts = counts + (ids == token)
        return penalize(values, counts, in_prompt, self.penalties)

    def apply(self, logits, tokens: Sequence[int] = (), parents: Sequence[int] | None = None, *,
              copy: bool = False):
        """Penalize CUDA score rows; returns the float32 rows to sample (``logits`` itself unless a copy is needed).

        Row ``r`` follows the reply plus ``tokens[a]`` for every ``a`` on row ``r``'s path from the root
        (``a`` >= 1, ``r`` included): ``tokens[0]`` is the committed anchor the rows extend; without ``tokens``
        every row follows the reply alone. ``parents`` gives each row's parent (-1 for the root) for a tree; None
        is a chain. ``copy`` keeps an immutable source unchanged. The host builds each row's counts over the
        penalized columns and the device gathers, penalizes and scatters only those columns.
        """

        import torch

        rows = logits.shape[0]
        if logits.ndim != 2 or (tokens and len(tokens) != rows) or (parents is not None and len(parents) != rows):
            raise ValueError("penalties need one input token and parent per score row")
        if logits.dtype != torch.float32 or copy:
            logits = logits.float().clone() if logits.dtype == torch.float32 else logits.float()
        known = len(self._ids)
        # the distinct drafts, each draft row's index among them (``slot``) and each draft's column (``at``): its
        # column among the penalized ids, else a new one after them
        drafted, slot = np.unique(np.asarray(tokens[1:], dtype=np.int64), return_inverse=True)
        at = np.searchsorted(self._ids, drafted)
        old = at < known
        old[old] = self._ids[at[old]] == drafted[old]
        extra = drafted[~old]
        at[~old] = known + np.arange(len(extra))
        cols = np.concatenate([self._ids, extra]) if len(extra) else self._ids
        if not len(cols):
            return logits
        # rows [0, rows): each row's counts; row ``rows``: the prompt flags (repetition)
        table = np.zeros((rows + 1, len(cols)), dtype=np.float32)
        table[:rows, :known] = self._counts
        table[rows, :known] = self._in_prompt
        if len(drafted):
            # each row's drafts are its parent's plus its own: one pass in row order (parents precede their rows)
            path = np.zeros((rows, len(drafted)), dtype=np.float32)
            for row in range(1, rows):
                parent = row - 1 if parents is None else int(parents[row])
                if parent >= row:
                    raise ValueError("tree parents must precede their rows")
                if parent > 0:
                    path[row] = path[parent]
                path[row, slot[row - 1]] += 1
            table[:rows, at] += path
        device = logits.device
        ids = torch.from_numpy(cols).to(device)
        both = torch.from_numpy(table).to(device)
        c, in_prompt = both[:rows], both[rows] > 0
        z = logits.index_select(1, ids)
        seen = c > 0
        p = self.penalties
        if p.repetition != 1.0:                     # float32 scalars: exact casts, one rounding per multiply
            inverse, r = _factors(p.repetition)
            scaled = torch.where(z > 0, z * float(inverse), z * float(r))
            scaled = torch.where(torch.isfinite(z), torch.where(z == 0, z, scaled.clamp(-F32_MAX, F32_MAX)), z)
            z = torch.where(in_prompt[None, :] | seen, scaled, z)
        if p.frequency != 0.0:
            z = z - p.frequency * c
        if p.presence != 0.0:
            z = z - p.presence * seen.float()
        logits.index_copy_(1, ids, z)
        return logits


__all__ = ["History", "Penalties", "penalize"]
