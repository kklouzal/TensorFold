"""One decode round's token choices for every DeepSeek-V4.1 SM70 stream with one host wait (rank 0).

Positive-temperature streams with top_k off and no grammar, penalties, probabilities or forced reply draw on the
device in a decode graph round (``device_sampling``: the host rule below, bit for bit, on each rank's vocabulary
shard); their tokens land in the same wait. Eager rounds keep the host path for them.

Plain greedy streams (no sampling, grammar, penalties, probabilities or forced reply) choose from the decode graph's
compact packet: per rank and row the shard maximum, its first column (global ID) and a NaN/+inf flag, all-gathered in
the graph (8 x [rows, 3] FP32) and copied to pinned host memory behind it (``graphs.Shape``). The first rank holding
the global maximum gives its first column: exactly ``argmax`` of the concatenated full row (the vocabulary shards are
contiguous in rank order, every value is the same FP32 word), with ``sample_rows``' greedy errors.

Every other stream needs its full-vocabulary row (the round's eager gather after the graph): its grammar mask and
penalties are applied when the round launches (their state already holds every decided token), then the rows'
device work is queued by kind with ``tensorfold.cuda.sampling``'s stream packets (greedy and top-k groups:
``_group_packet`` / ``_group_choose``, ``sample_streams``' exact path) and a batched top-k-off nucleus packet whose
per-row arithmetic is ``nucleus_rows``' (one rank: FP64 scores centered on the row maximum, divided by the row's
temperature, fixed-point masses, the top ``NUCLEUS`` candidates), so every choice keeps ``sample_rows``' bits. The
packets reach the host in the same wait as the round (copied before the round's done event); only rare tie repairs
and complete-support fallbacks read again. A failure is the stream's own (it fails alone, as ``_choose``)."""

from __future__ import annotations

import numpy as np
import torch

from tensorfold.cuda.sampling import (MARGIN, MASS, NUCLEUS, _draw_support, _group_choose, _group_packet, _shares,
                                      _support, one_rank, score_flags, validate_rows)

GREEDY, TOPK, NUCLEUS_KIND, FORCED = "greedy", "topk", "nucleus", "forced"


def _unmodified(s) -> bool:
    return s.forced is None and s.constraint is None and s.history is None and s.probabilities is None


def plain(s) -> bool:
    """A stream whose token is the argmax of its unmodified row (the compact packet decides it)."""
    return _unmodified(s) and (s.sampling is None or s.sampling.temperature <= 0)


def device_sampled(s) -> bool:
    """A positive-temperature stream the decode graph's keyed draw serves (``device_sampling``)."""
    from .device_sampling import eligible

    return _unmodified(s) and eligible(s.sampling)


def device_decided(s) -> bool:
    """A stream whose token the decode graph decides on the device (run-ahead may follow it)."""
    return plain(s) or device_sampled(s)


def needs_row(s, graph: bool = True) -> bool:
    """A stream that needs its full-vocabulary row this round (sampling, masks/penalties or a probabilities sink);
    ``graph``: a decode graph round, whose keyed draw serves ``device_sampled`` streams."""
    return (not plain(s) and not (graph and device_sampled(s))
            and not (s.forced is not None and s.probabilities is None))


def compact_packet(local: torch.Tensor, rank: int) -> torch.Tensor:
    """This rank's [rows, 3] FP32 packet of its contiguous vocabulary shard ``local`` [rows, vocab / world]: the
    maximum, its first column as a global ID (torch.max's documented first-maximum rule; IDs < 2**24 are exact
    in FP32) and a NaN/+inf flag."""
    values, index = local.max(dim=-1)
    bad = (torch.isnan(local) | torch.isposinf(local)).any(dim=-1)
    return torch.stack([values, (index + rank * local.shape[1]).float(), bad.float()], -1).contiguous()


def greedy_tokens(host: torch.Tensor, n: int):
    """The compact packet [world, bucket, 3] (pinned, landed) -> per real row (token, error or None)."""
    best = host.numpy()[:, :n]
    choice = best[:, :, 0].argmax(axis=0)                  # first (lowest) rank among equal maxima
    rows = np.arange(n)
    tokens = best[choice, rows, 1].astype(np.int64)
    maxima = best[choice, rows, 0]
    bad = best[:, :, 2].any(axis=0)
    out = []
    for r in range(n):
        if bad[r]:
            out.append((None, ValueError("sampling scores contain NaN or positive infinity")))
        elif not np.isfinite(maxima[r]):
            out.append((None, ValueError("a complete vocabulary row needs at least one finite score")))
        else:
            out.append((int(tokens[r]), None))
    return out


class _Nucleus:
    """``nucleus_rows`` (one rank, no ID map) over rows of several policies: the device part queued at once."""

    def __init__(self, rows: torch.Tensor, temperatures):
        width = rows.shape[1]
        self.width = width
        scores = rows.float().double()
        top = scores.max(dim=-1, keepdim=True).values
        # ``nucleus_rows`` divides by a Python scalar, which CUDA computes as a multiply by its FP64 reciprocal
        # (div_true's CPU-scalar path): the same reciprocal per row keeps those bits
        inverse = torch.tensor([1.0 / max(float(t), 1e-6) for t in temperatures], dtype=torch.float64).to(
            rows.device, non_blocking=True)[:, None]
        header = torch.cat([top.view(torch.int64), score_flags(scores).to(torch.int64)[:, None]], dim=1)
        self.scaled = (scores - top) * inverse
        self.mass = torch.floor(torch.exp(self.scaled) * MASS).to(torch.int64)
        count = min(NUCLEUS, width)
        vals, cols = torch.topk(self.scaled, count, dim=-1)
        kept = self.mass.gather(1, cols)
        shard = torch.full((rows.shape[0], 1), width, dtype=torch.int64, device=rows.device)
        packed = torch.cat([header, vals.view(torch.int64), cols, kept, self.mass.sum(dim=-1, keepdim=True), shard],
                           dim=1)
        self.count = count
        self.host = packed.to("cpu", non_blocking=True)

    def choose(self, r: int, position: int, sampling) -> int:
        words = self.host.numpy()[r:r + 1]
        if words[:, 1].any():
            raise ValueError("a rank's sampling scores contain NaN or positive infinity")
        if self.width * int(MASS) > np.iinfo(np.int64).max:
            raise ValueError("fixed-point vocabulary mass exceeds signed64")
        if not np.isfinite(np.ascontiguousarray(words[:, 0]).view(np.float64)).all():
            raise ValueError("a complete vocabulary row needs at least one finite score")
        c, p = self.count, words[None, :, 2:]                  # [world 1, row, ...] as _shares returns
        got = (np.ascontiguousarray(p[:, :, :c]).view(np.float64), p[:, :, c:2 * c], p[:, :, 2 * c:3 * c],
               p[:, :, 3 * c], p[:, :, 3 * c + 1])
        support = _support(got, sampling)
        if support is None:                     # the nucleus runs past the candidates: the whole row
            support = _support(_shares(one_rank, self.scaled[r:r + 1], self.mass[r:r + 1], int(got[4].max()), 0,
                                       None), sampling)
        return _draw_support(support, [position], sampling)[0][0]


class RowSampling:
    """The full-row streams of one round: ``queue`` at launch (device work, no host wait), ``choose`` after the
    round's done event. Members are row indices of the round's full logits."""

    def __init__(self, live, positions, rows: torch.Tensor, members):
        self.rows, self.errors, self.scored = {}, {}, {}
        self.kind, self.local = {}, {}
        greedy, topk, nucleus = [], {}, []
        width = rows.shape[1]
        for g in members:
            s = live[g]
            try:
                row = rows[g:g + 1]
                if s.forced is not None:
                    self.kind[g] = FORCED
                else:                           # ``Decoder._choose``'s order: grammar mask, then penalties
                    if s.constraint is not None:
                        row = row.clone()
                        s.constraint.mask(row)
                    if s.history is not None:
                        row = s.history.apply(row, copy=s.constraint is None)
                    validate_rows(row, [positions[g]], s.sampling)
                    sampling = s.sampling
                    if sampling is None or sampling.temperature <= 0:
                        self.kind[g] = GREEDY
                        greedy.append(g)
                    elif not sampling.top_k:
                        self.kind[g] = NUCLEUS_KIND
                        nucleus.append(g)
                    else:
                        self.kind[g] = TOPK
                        topk.setdefault(min(width, int(sampling.top_k) + MARGIN), []).append(g)
                self.scored[g] = row
            except Exception as error:          # noqa: BLE001  (this request fails alone)
                self.errors[g] = error
        self.groups = []
        for count, group in ([(0, greedy)] if greedy else []) + sorted(topk.items()):
            stacked = torch.cat([self.scored[g] for g in group])
            starts = list(range(len(group) + 1))
            count, _, scores, packet, kept = _group_packet(stacked, starts, count, list(range(len(group))))
            self.groups.append((count, group, scores, packet.to("cpu", non_blocking=True), kept))
        self.nucleus = None
        if nucleus:
            self.nucleus = (nucleus, _Nucleus(torch.cat([self.scored[g] for g in nucleus]),
                                              [live[g].sampling.temperature for g in nucleus]))
        self.live, self.positions = live, positions

    def choose(self) -> dict:
        """{row g: token or Exception} after the round's packets landed."""
        out = dict(self.errors)
        for count, group, scores, packet, kept in self.groups:
            words = packet.numpy()
            for i, g in enumerate(group):
                got = [[]]
                try:
                    _group_choose(got, count, [0], scores[i:i + 1], words[i:i + 1], kept, [0, 1],
                                  [[self.positions[g]]], [self.live[g].sampling])
                    out[g] = got[0][0]
                except Exception as error:      # noqa: BLE001
                    out[g] = error
        if self.nucleus is not None:
            group, nucleus = self.nucleus
            for i, g in enumerate(group):
                try:
                    out[g] = nucleus.choose(i, self.positions[g], self.live[g].sampling)
                except Exception as error:      # noqa: BLE001
                    out[g] = error
        for g, kind in self.kind.items():
            if kind == FORCED and g not in out:
                out[g] = None                   # the caller takes the forced token
        return out
