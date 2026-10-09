"""Position-keyed CUDA target sampling with the Metal engine's host-side rule, for every CUDA family."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
import math

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows, uniform

MASS = 2.0 ** 40        # a token's share of the mass in fixed point: shard sums are exact, so ranks agree bit for bit
NUCLEUS = 1024          # candidates a rank reads for a top_k-off draw; a row they don't cover reads whole shards


def validate_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> None:
    """Validate metadata before allocation; callers publish scores on their current CUDA stream.

    Scores, positions and policy remain immutable until the sampler returns. The host readbacks finish
    every borrowed source use; returned Python tokens retain no tensor storage.
    """

    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or not logits.is_cuda
            or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or logits.shape[1] <= 0 or len(positions) != logits.shape[0]):
        raise ValueError("expected floating CUDA logits [rows, nonempty vocab] and one position per row")
    validate_policy(positions, sampling)


def validate_policy(positions: Sequence[int], sampling: Sampling | None) -> None:
    """Validate keyed positions and filtering policy independently of tensor allocation."""

    for position in positions:
        if not isinstance(position, (int, np.integer)) or isinstance(position, (bool, np.bool_)) \
                or not 0 <= int(position) <= (1 << 64) - 1:
            raise ValueError("sampling positions must be unsigned64 absolute integers")
    if sampling is not None:
        if (not isinstance(sampling, Sampling) or type(sampling.seed) is not int
                or type(sampling.top_k) is not int or sampling.top_k < 0):
            raise ValueError("sampling requires an integer keyed seed")
        if (not math.isfinite(float(sampling.temperature))
                or not math.isfinite(float(sampling.top_p)) or not 0.0 <= sampling.top_p <= 1.0
                or not math.isfinite(float(sampling.min_p)) or not 0.0 <= sampling.min_p <= 1.0):
            raise ValueError("finite temperature and top_p/min_p in [0, 1] required")


def token_ids(id_map: torch.Tensor | np.ndarray | None, width: int, *, offset: int = 0) -> np.ndarray:
    """Own a validated immutable global-ID table; stable model maps should call this at initialization."""

    if type(width) is not int or width <= 0:
        raise ValueError("token map needs a positive vocabulary width")
    if isinstance(id_map, TokenMap):
        return id_map.ids(width)
    if type(offset) is not int or offset < 0 or offset + width - 1 > np.iinfo(np.int64).max:
        raise ValueError("global token IDs must fit nonnegative signed64")
    if id_map is None:
        result = np.arange(width, dtype=np.int64) + np.int64(offset)
    else:
        source = id_map.detach().cpu().numpy() if isinstance(id_map, torch.Tensor) else np.asarray(id_map)
        if (source.shape != (width,) or source.dtype.kind not in "iu"
                or bool((source < 0).any()) or bool((source > np.iinfo(np.int64).max).any())):
            raise ValueError("one nonnegative signed64 global token ID per vocabulary column required")
        result = source.astype(np.int64, copy=True)
        if len(np.unique(result)) != width:
            raise ValueError("global token IDs must be distinct")
    result.flags.writeable = False
    return result


class TokenMap:
    """Model-owned immutable ID snapshot; validate once outside per-token work."""

    def __init__(self, id_map: torch.Tensor | np.ndarray, width: int, *, device=None):
        self._ids = token_ids(id_map, width)
        device = id_map.device if isinstance(id_map, torch.Tensor) and device is None else device
        self._tensor = torch.tensor(self._ids, dtype=torch.int64, device=device) if device is not None else None

    def ids(self, width: int) -> np.ndarray:
        if self._ids.shape != (width,):
            raise ValueError("token-map owner differs from the vocabulary geometry")
        return self._ids

    def tensor(self, device) -> torch.Tensor:
        if self._tensor is None or self._tensor.device != device:
            raise ValueError("token-map owner differs from the model's CUDA device")
        return self._tensor


def score_flags(scores: torch.Tensor) -> torch.Tensor:
    """One invalid-domain bit per complete row, including values absent from top-k."""

    return (torch.isnan(scores) | torch.isposinf(scores)).any(dim=-1)


def valid_scores(values: np.ndarray, *, require_finite: bool = True) -> None:
    """Contain malformed scores before logarithms and keyed selection; -inf is a mask."""

    if (np.isnan(values).any() or np.isposinf(values).any()
            or (require_finite and not np.isfinite(values).any(axis=-1).all())):
        raise ValueError("sampling needs finite scores or -inf masks, with one finite score per complete row")


def ambiguous_rows(values: np.ndarray, top_k: int, full_width: int) -> np.ndarray:
    """A strictly lower retained edge proves all top-k boundary ties fit in the candidates."""

    rows, width = values.shape
    k = min(int(top_k), full_width)
    if not 0 < k <= width <= full_width:
        raise ValueError("positive top-k and complete candidate geometry required")
    if width == full_width:
        return np.empty(0, dtype=np.int64)
    edge = values.min(axis=1)
    cutoff = np.partition(values, width - k, axis=1)[:, width - k]
    return np.nonzero(np.isfinite(edge) & (cutoff == edge))[0]


def repaired_choose(values: np.ndarray, ids: np.ndarray, positions: Sequence[int], sampling: Sampling,
                    full_width: int, full_row: Callable) -> list[int]:
    """Preserve the FP64 chooser; read ambiguous complete rows from the same immutable score evaluation."""

    if values.shape != ids.shape or len(positions) != values.shape[0]:
        raise ValueError("one global-ID table and position per candidate row required")
    valid_scores(values)
    # Explicit integer storage preserves every keyed position bit, including
    # mixed Python integers at UINT64_MAX that inferred float64 cannot retain.
    keyed_positions = np.asarray(positions, dtype=np.uint64)
    repair = ambiguous_rows(values, sampling.top_k, full_width)
    out = choose_rows(values, ids, keyed_positions, sampling)
    for row in repair:
        row = int(row)
        complete, global_ids = full_row(row)
        if complete.shape != (full_width,) or global_ids.shape != complete.shape:
            raise ValueError("tie repair must supply the complete original row and global IDs")
        valid_scores(complete[None])
        out[row] = choose_rows(complete[None], global_ids[None], keyed_positions[row:row + 1], sampling)[0]
    return out


def sample_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
                id_map: torch.Tensor | np.ndarray | TokenMap | None = None, with_prob: bool = False):
    """Sample immutable complete rows; keyed choices use real IDs and exact top-k boundary ties.

    ``with_prob`` returns temperature-1 probabilities using the original FP32
    log-sum-exp followed by FP64 exp. The unsharded draft probability path
    retains its complete-row FP64 filtering when top_k is off; ordinary target
    sampling uses fixed-point nucleus mass. A mapped greedy row selects its
    first maximum. Native greedy scores may be finite FP64 values outside
    FP32 range; probabilities retain FP32 normalization and fail explicitly
    when it cannot produce a finite result.
    """

    validate_rows(logits, positions, sampling)
    rows, width = logits.shape
    if rows == 0:
        return ([], []) if with_prob else []
    mapping = token_ids(id_map, width) if id_map is not None else None
    if (sampling is not None and sampling.temperature > 0 and not sampling.top_k
            and id_map is None and not with_prob):
        return nucleus_rows(logits, positions, sampling)
    greedy = sampling is None or sampling.temperature <= 0
    if greedy and not with_prob:
        cols = logits.argmax(dim=-1, keepdim=True)
        packet = torch.cat([cols, score_flags(logits).to(torch.int64)[:, None],
                            torch.isfinite(logits.gather(1, cols)).to(torch.int64)], dim=1).cpu().numpy()
        if packet[:, 1].any():
            raise ValueError("sampling scores contain NaN or positive infinity")
        if not packet[:, 2].all():
            raise ValueError("a complete vocabulary row needs at least one finite score")
        columns = packet[:, 0]
        return [int(token) for token in (mapping[columns] if mapping is not None else columns)]
    scores = logits.float()
    count = 1 if greedy else min(width, int(sampling.top_k) + MARGIN) if sampling.top_k else width
    if greedy:
        cols = (scores if with_prob else logits).argmax(dim=-1, keepdim=True)
        values = scores.gather(1, cols)
    elif count < width:
        values, cols = torch.topk(scores, count, dim=-1, sorted=False)
    else:
        values = scores
        cols = torch.arange(width, dtype=torch.int64, device=scores.device).expand(rows, width)
    # Raw score words, integer columns and full-row flags share one readback.
    # No floating-point ID encoding or score conversion changes their bits.
    parts = [values.contiguous().view(torch.int32).to(torch.int64), cols,
             score_flags(logits if greedy else scores).to(torch.int64)[:, None]]
    if with_prob:
        parts.append(torch.logsumexp(scores, dim=-1, keepdim=True).view(torch.int32).to(torch.int64))
    metadata = torch.cat(parts, dim=1).cpu().numpy()
    if metadata[:, 2 * count].any():
        raise ValueError("sampling scores contain NaN or positive infinity")
    if with_prob:
        normalizer = metadata[:, 2 * count + 1].astype(np.int32).view(np.float32)
        if not np.isfinite(normalizer).all():
            raise ValueError("FP32 confidence normalization is nonfinite")
    values_np = metadata[:, :count].astype(np.int32).view(np.float32)
    valid_scores(values_np)
    cols_np = metadata[:, count:2 * count]
    ids = mapping[cols_np] if mapping is not None else cols_np
    if greedy:
        chosen = [int(t) for t in ids[:, 0]]
    else:
        complete_ids = mapping
        complete_rows = {}

        def full_row(row):
            nonlocal complete_ids
            if complete_ids is None:
                complete_ids = np.arange(width, dtype=np.int64)
            complete_rows[row] = scores[row].cpu().numpy()
            return complete_rows[row], complete_ids

        chosen = repaired_choose(values_np, ids, positions, sampling, width, full_row) if sampling.top_k \
            else choose_rows(values_np, ids, np.asarray(positions, dtype=np.uint64), sampling)
    if not with_prob:
        return chosen
    lse = normalizer
    # A repaired choice may be outside the original candidate set. Read its
    # original value without changing the normal candidate probability math.
    probabilities = []
    for row, token in enumerate(chosen):
        hit = np.nonzero(ids[row] == token)[0]
        value = values_np[row, hit[0]] if len(hit) \
            else complete_rows[row][int(np.nonzero(complete_ids == token)[0][0])]
        probability = float(np.exp(float(value) - float(lse[row])))
        if not math.isfinite(probability):
            raise ValueError("FP32 confidence normalization produced a nonfinite probability")
        probabilities.append(probability)
    return chosen, probabilities


def sample_streams(logits: torch.Tensor, starts: Sequence[int], positions: Sequence[Sequence[int]],
                   samplings: Sequence[Sampling | None]) -> list[list[int]]:
    """Group stream rows by candidate geometry while preserving each stream's policy and absolute positions."""

    if (len(starts) != len(samplings) + 1 or len(positions) != len(samplings)
            or not starts or starts[0] != 0
            or any(type(start) is not int for start in starts)
            or any(left > right for left, right in zip(starts, starts[1:]))):
        raise ValueError("ordered stream boundaries and one position list/policy per stream required")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or starts[-1] > logits.shape[0]:
        raise ValueError("stream boundaries must fit the complete logits rows")
    validate_rows(logits[:0], [], None)
    groups: dict[int, list[int]] = {}
    width = logits.shape[1]
    out: list[list[int]] = [[] for _ in samplings]
    for stream, sampling in enumerate(samplings):
        validate_rows(logits[starts[stream]:starts[stream + 1]], positions[stream], sampling)
        if starts[stream] == starts[stream + 1]:
            continue
        greedy = sampling is None or sampling.temperature <= 0
        if not greedy and not sampling.top_k:
            out[stream] = nucleus_rows(logits[starts[stream]:starts[stream + 1]], positions[stream], sampling)
        else:
            groups.setdefault(0 if greedy else min(width, int(sampling.top_k) + MARGIN), []).append(stream)
    # Preserve the original launch-all-groups pipeline. Sources stay strongly
    # owned until every candidate/full-row readback has completed.
    launched = []
    for count, members in groups.items():
        scores = torch.cat([logits[starts[s]:starts[s + 1]] for s in members]) if len(members) > 1 \
            else logits[starts[members[0]]:starts[members[0] + 1]]
        source = scores
        if count == 0:
            cols = source.argmax(dim=-1, keepdim=True)
            packet = torch.cat([cols, score_flags(source).to(torch.int64)[:, None],
                                torch.isfinite(source.gather(1, cols)).to(torch.int64)], dim=1)
            launched.append((count, members, source, packet, 1))
            continue
        scores = scores.float()
        if count < width:
            values, cols = torch.topk(scores, count, dim=-1, sorted=False)
        else:
            values = scores
            cols = torch.arange(width, dtype=torch.int64, device=scores.device).expand(scores.shape[0], width)
        kept = cols.shape[1]
        packet = torch.cat([values.contiguous().view(torch.int32).to(torch.int64), cols,
                            score_flags(scores).to(torch.int64)[:, None]], dim=1)
        launched.append((count, members, scores, packet, kept))
    # Different policies sharing a count still use their own FP64 chooser.
    for count, members, scores, packet, kept in launched:
        packet = packet.cpu().numpy()
        if count == 0:
            if packet[:, 1].any():
                raise ValueError("sampling scores contain NaN or positive infinity")
            if not packet[:, 2].all():
                raise ValueError("a complete vocabulary row needs at least one finite score")
            columns, values_np = packet[:, :1], None
        else:
            if packet[:, 2 * kept].any():
                raise ValueError("sampling scores contain NaN or positive infinity")
            values_np = packet[:, :kept].astype(np.int32).view(np.float32)
            columns = packet[:, kept:2 * kept]
            valid_scores(values_np)
        row = 0
        complete_ids = None

        def complete_row(index, base):
            nonlocal complete_ids
            if complete_ids is None:
                complete_ids = np.arange(width, dtype=np.int64)
            return scores[base + index].cpu().numpy(), complete_ids

        for stream in members:
            n = starts[stream + 1] - starts[stream]
            if count == 0:
                out[stream] = [int(token) for token in columns[row:row + n, 0]]
            else:
                out[stream] = repaired_choose(values_np[row:row + n], columns[row:row + n, :count],
                    positions[stream], samplings[stream], width,
                    lambda index, base=row: complete_row(index, base))
            row += n

    return out


def _stacked(gather: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor) -> torch.Tensor:
    """Every rank's copy of ``t`` [world, *t.shape], this rank's included, through ``gather`` of its float32 words."""

    words = t.contiguous().view(torch.float32).view(-1)
    got = gather(words)
    if (not isinstance(got, torch.Tensor) or got.dtype != torch.float32 or got.device != words.device
            or words.numel() == 0 or got.numel() < words.numel() or got.numel() % words.numel()):
        raise ValueError("gather must return complete rank copies of the original CUDA float32 words")
    return got.contiguous().reshape(-1, words.numel()).view(t.dtype).reshape(-1, *t.shape)


def one_rank(words: torch.Tensor) -> torch.Tensor:
    return words[None]


def comm_gather(comm) -> Callable[[torch.Tensor], torch.Tensor]:
    """``nucleus_rows``'s gather over a family's NCCL ``comm`` (``tensorfold.cuda.comm``)."""

    def gather(words: torch.Tensor) -> torch.Tensor:
        got = torch.empty((comm.world * words.numel(),), dtype=words.dtype, device=words.device)
        comm.all_gather(words, got)
        return got.view(comm.world, -1)

    return gather


def dist_gather(words: torch.Tensor) -> torch.Tensor:
    """``nucleus_rows``'s gather over torch.distributed (the 27B's two ranks)."""

    import torch.distributed as dist

    got = torch.empty((dist.get_world_size(), words.numel()), dtype=words.dtype, device=words.device)
    dist.all_gather_into_tensor(got, words)
    return got


def nucleus_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling, *, offset: int = 0,
                 id_map: torch.Tensor | None = None, gather: Callable = one_rank,
                 probs: list[float] | None = None) -> list[int]:
    """top_k off: the keyed draw over the top_p nucleus then min_p, cut by fixed-point mass, the same on each shape."""

    validate_rows(logits, positions, sampling)
    if sampling.temperature <= 0 or sampling.top_k:
        raise ValueError("nucleus_rows requires positive temperature and top_k off")
    if type(offset) is not int or offset < 0 or offset + logits.shape[1] - 1 > np.iinfo(np.int64).max:
        raise ValueError("global token IDs must fit nonnegative signed64")
    if logits.shape[0] == 0:
        return []
    if id_map is not None:
        mapping = token_ids(id_map, logits.shape[1])
        if isinstance(id_map, TokenMap):
            id_map = id_map.tensor(logits.device)
        elif not isinstance(id_map, torch.Tensor):
            id_map = torch.tensor(mapping, dtype=torch.int64, device=logits.device)
        elif id_map.device != logits.device:
            raise ValueError("token map and logits must share a CUDA device")
    scaled = logits.float().double() / max(float(sampling.temperature), 1e-6)
    maxima = scaled.max(dim=-1, keepdim=True).values
    header = torch.cat([maxima.view(torch.int64),
                        torch.full_like(maxima, logits.shape[1], dtype=torch.int64),
                        score_flags(scaled).to(torch.int64)[:, None]], dim=1)
    gathered = _stacked(gather, header)
    summary = gathered.cpu().numpy()
    if summary[:, :, 2].any():
        raise ValueError("a rank's sampling scores contain NaN or positive infinity")
    widths = summary[:, :, 1]
    if np.any(widths.sum(axis=0, dtype=object) * int(MASS) > np.iinfo(np.int64).max):
        raise ValueError("fixed-point vocabulary mass exceeds signed64")
    maximum_values = np.ascontiguousarray(summary[:, :, 0]).view(np.float64)
    if not np.isfinite(maximum_values).any(axis=0).all():
        raise ValueError("a complete vocabulary row needs at least one finite score")
    top = gathered[:, :, 0].contiguous().view(torch.float64).max(dim=0).values
    mass = torch.floor(torch.exp(scaled - top[:, None]) * MASS).to(torch.int64)
    got = _shares(gather, scaled, mass, NUCLEUS, offset, id_map)
    drawn = _draw(got, positions, sampling)
    if drawn is None:                           # some row's nucleus runs past the candidates: every whole shard
        drawn = _draw(_shares(gather, scaled, mass, int(got[4].max()), offset, id_map), positions, sampling)
    if probs is not None:
        probs.extend(share for _, share in drawn)
    return [token for token, _ in drawn]


def _shares(gather, scaled, mass, count, offset, id_map):
    """Every rank's padded top (value, id, mass) per row, plus the shard's mass and width sums."""

    rows, width = scaled.shape
    vals, cols = torch.topk(scaled, min(count, width), dim=-1)
    ids = id_map[cols].to(torch.int64) if id_map is not None else cols + int(offset)
    pad = count - vals.shape[1]
    if pad:
        vals = torch.cat([vals, vals.new_full((rows, pad), float("-inf"))], dim=1)
        ids = torch.cat([ids, ids.new_full((rows, pad), -1)], dim=1)
    kept = mass.gather(1, cols)
    kept = torch.cat([kept, kept.new_zeros((rows, pad))], dim=1) if pad else kept
    shard = torch.tensor([[width]], dtype=torch.int64, device=scaled.device).expand(rows, 1)
    packed = torch.cat([vals.view(torch.int64), ids, kept, mass.sum(dim=-1, keepdim=True), shard], dim=1)
    both = _stacked(gather, packed).cpu().numpy()
    return (np.ascontiguousarray(both[:, :, :count]).view(np.float64), both[:, :, count:2 * count],
            both[:, :, 2 * count:3 * count], both[:, :, 3 * count], both[:, :, 3 * count + 1])


def _draw(got, positions, s: Sampling) -> list[tuple[int, float]] | None:
    """Each row's (token, its share of the mass) from every rank's candidates; None if a row needs whole shards."""

    vals, ids, mass, sums, widths = got
    cut = 0.0 < s.top_p < 1.0
    drawn = []
    for r, position in enumerate(positions):
        total = sum(int(part) for part in sums[:, r])
        if total <= 0 or total > np.iinfo(np.int64).max:
            raise ValueError("nucleus row mass must fit positive signed64")
        need = math.ceil(Fraction(s.top_p) * total) if cut else None
        floor = vals[:, r, :].max() + s.min_log                   # the min_p cut (-inf when off)
        for k in range(vals.shape[0]):                            # a rank's share must end inside its candidates
            real = int((ids[k, r] >= 0).sum())
            if real == widths[k, r]:                              # its whole shard
                continue
            order = np.lexsort((ids[k, r], -vals[k, r]))
            order = order[ids[k, r][order] >= 0]                  # the padding goes
            v = vals[k, r][order]
            if cut:                                               # its mass reaches the need above its last one
                at = np.nonzero(np.cumsum(mass[k, r][order]) >= need)[0]
                covered = len(at) > 0 and v[int(at[0])] > v[-1]
            else:                                                 # min_p alone: a candidate below its cut
                covered = s.min_p > 0.0 and bool((v < floor).any())
            if not covered:
                return None
        v, i, m = vals[:, r].reshape(-1), ids[:, r].reshape(-1), mass[:, r].reshape(-1)
        order = np.lexsort((i, -v))
        order = order[i[order] >= 0]
        v, i, m = v[order], i[order], m[order]
        keep = len(v)
        if cut:
            keep = int((np.cumsum(m) < need).sum()) + 1
        if s.min_p > 0.0:
            keep = min(keep, int((v >= floor).sum()))
        if keep <= 0:
            raise ValueError("sampling policy removed every finite candidate")
        score = v[:keep] - np.log(-np.log(uniform(s.seed, int(position), i[:keep])))
        best = int(np.argmax(score))
        drawn.append((int(i[best]), float(m[best]) / total))
    return drawn


class ShardSampler:
    """Own stable TP geometry and token identity, validated collectively at startup.

    All ranks initialize plans in the same order. A plan owns immutable mapped
    IDs; changing a head/map/communicator requires a new plan. The regular
    sampler creates a bounded one-call plan when no model-owned plan is supplied.
    """

    def __init__(self, width: int, device, *, gather: Callable = one_rank, world: int = 1,
                 offset: int = 0, id_map=None, identity=None, native_greedy: bool = True):
        if type(world) is not int or world <= 0 or not callable(gather):
            raise ValueError("a positive shard world and word-preserving gather are required")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("sampling shard owner requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        mapping, error = None, None
        try:
            if type(width) is not int or width <= 0:
                raise ValueError("a nonempty TP vocabulary shard is required")
            if id_map is not None:
                if isinstance(id_map, torch.Tensor) and id_map.device != device:
                    raise ValueError("token map and communicator must share their CUDA device")
                mapping = token_ids(id_map, width)
            elif type(offset) is not int or offset < 0 or offset + width - 1 > np.iinfo(np.int64).max:
                raise ValueError("global token IDs must fit nonnegative signed64")
        except ValueError as exc:
            error = exc
        header = torch.tensor([width if type(width) is int and 0 < width <= (1 << 63) - 1 else 0,
                               int(error is not None), int(id_map is not None),
                               offset if type(offset) is int and 0 <= offset < (1 << 63) else 0],
                              dtype=torch.int64, device=device)
        metadata = _stacked(gather, header).cpu().numpy()
        if metadata.shape[0] != world:
            raise ValueError("sampling gather differs from its declared shard world")
        if metadata[:, 1].any():
            raise ValueError("a rank's TP vocabulary geometry or token map is invalid") from error
        if not (metadata[:, 2] == metadata[0, 2]).all():
            raise ValueError("all TP ranks must use the same global-ID mapping mode")
        widths = metadata[:, 0].copy()
        if sum(int(part) for part in widths) > np.iinfo(np.int64).max:
            raise ValueError("TP vocabulary width exceeds signed64")
        if mapping is None:
            spans = sorted((int(base), int(base) + int(size)) for size, base in zip(widths, metadata[:, 3]))
            if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
                raise ValueError("TP vocabulary shards must have distinct global token IDs")
        else:
            # Arbitrary global maps require whole-map validation once. Model
            # loaders already sort their maps, but neither order nor a local
            # uniqueness proof establishes disjointness across ranks.
            owned = torch.tensor(mapping, dtype=torch.int64, device=device)
            pad = int(widths.max()) - width
            packet = torch.cat([owned, owned.new_full((pad,), -1)]) if pad else owned
            all_ids = _stacked(gather, packet).cpu().numpy()
            real = all_ids[all_ids >= 0]
            if len(np.unique(real)) != len(real):
                raise ValueError("TP global token IDs must be distinct across ranks")
        widths.flags.writeable = False
        self.widths, self.width, self.offset = widths, width, offset
        self.gather, self.world, self.device, self.identity = gather, world, device, identity
        self.native_greedy = native_greedy
        self.max_width, self.total_width = int(widths.max()), sum(int(part) for part in widths)
        self.token_map = TokenMap(mapping, width, device=device) if mapping is not None else None
        self.mapping = self.token_map.ids(width) if self.token_map is not None else None
        self.device_map = self.token_map.tensor(device) if self.token_map is not None else None
        self.sorted_mapping = mapping is None or bool((mapping[1:] > mapping[:-1]).all())
        self.graph_ids_safe = int(real.max()) <= np.iinfo(np.int32).max if mapping is not None \
            else spans[-1][1] - 1 <= np.iinfo(np.int32).max

    def check(self, logits: torch.Tensor, *, identity=None) -> None:
        if (logits.shape[1] != self.width or logits.device != self.device
                or (identity is not None and identity is not self.identity)):
            raise ValueError("sampling shard owner differs from the source geometry/device/communicator")


def _tp_candidates(plan: ShardSampler, scores: torch.Tensor, count: int, mapping: np.ndarray | None,
                   offset: int, *, greedy: bool = False, device_map: torch.Tensor | None = None,
                   sorted_mapping: bool = True, source: torch.Tensor | None = None, with_norm: bool = False,
                   domain_source: torch.Tensor | None = None):
    """Gather equal-sized padded packets; complete source scores stay borrowed through the readback."""

    rows, width = scores.shape
    kept = min(count, width)
    if greedy and count == 1:
        source = scores if source is None else source
        cols = source.argmax(dim=-1, keepdim=True)
        values = scores.gather(1, cols)
        if mapping is not None and not sorted_mapping:
            # TP greedy breaks ties by global ID. Ordinary sorted checkpoint
            # maps keep the original argmax path and its first-column rule.
            global_ids = device_map
            ids = torch.where(source == source.gather(1, cols), global_ids[None],
                              np.iinfo(np.int64).max).min(dim=-1).values[:, None]
        else:
            ids = device_map[cols] if mapping is not None \
                else cols + offset
    else:
        values, cols = torch.topk(scores, kept, dim=-1, sorted=False)
        ids = device_map[cols] if mapping is not None \
            else cols + offset
    pad = count - values.shape[1]
    if pad:
        values = torch.cat([values, values.new_full((rows, pad), -float("inf"))], dim=-1)
        ids = torch.cat([ids, ids.new_full((rows, pad), -1)], dim=-1)
    # Integer transport retains every ID bit; FP32 values and FP32 log-sum-exp
    # use raw int32 words, without numerical casting or float ID sentinels.
    parts = [values.contiguous().view(torch.int32).to(torch.int64), ids.to(torch.int64)]
    if with_norm:
        parts.append(torch.logsumexp(scores, dim=-1, keepdim=True).view(torch.int32).to(torch.int64))
    domain = scores if domain_source is None else domain_source
    status = score_flags(domain).to(torch.int64)
    native_finite_proof = domain_source is not None
    if native_finite_proof:
        status = status | (torch.isfinite(domain_source).any(dim=-1).to(torch.int64) << 1)
    parts.append(status[:, None])
    packet = torch.cat(parts, dim=-1)
    gathered = _stacked(plan.gather, packet).cpu().numpy()
    if gathered.shape[0] != plan.world:
        raise ValueError("sampling gather differs from its owned shard world")
    if (gathered[:, :, -1] & 1).any():
        raise ValueError("a rank's sampling scores contain NaN or positive infinity")
    vals = gathered[:, :, :count].astype(np.int32).view(np.float32)
    ids = gathered[:, :, count:2 * count]
    norms = gathered[:, :, 2 * count].astype(np.int32).view(np.float32) if with_norm else None
    finite = ((gathered[:, :, -1] & 2).any(axis=0) if native_finite_proof
              else np.isfinite(vals).any(axis=(0, 2)))
    if not finite.all():
        raise ValueError("a complete vocabulary row needs at least one finite score")
    return vals, ids, norms


def _tp_choose(values, tokens, positions, sampling, *, narrowed_greedy: bool = False):
    chosen = []
    greedy = sampling is None or sampling.temperature <= 0
    for row, position in enumerate(positions):
        v, ids = values[:, row].reshape(-1), tokens[:, row].reshape(-1)
        real = ids >= 0
        v, ids = v[real], ids[real]
        if not (greedy and narrowed_greedy):
            valid_scores(v[None])
        elif np.isnan(v).any():
            raise ValueError("greedy candidate transport produced NaN")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("gathered global token IDs must be distinct")
        if greedy:
            order = np.lexsort((ids, -v))
            chosen.append(int(ids[order[0]]))
        else:
            chosen.append(choose_rows(v[None], ids[None], np.asarray([position], dtype=np.uint64), sampling)[0])
    return chosen


def sample_shards(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
                  plan: ShardSampler, with_prob: bool = False, probability: Callable | None = None,
                  greedy_count: int = 1):
    """Exact global-ID sampling over an operation's immutable original shards.

    All ranks use the same positions/policy/probability mode. NaN/+inf and
    globally all-masked rows fail after the same collective. A zero-mass local
    shard remains valid. ``probability`` preserves a family's existing host
    confidence rule; otherwise probabilities use FP32 local log-sum-exps and
    the original FP64 rank merge. Greedy confidence retains its original
    per-shard candidate cap, including after exact ID tie repair. Greedy
    domain checks inspect the original representation even when the legacy
    rank merge narrows values to FP32, including derived infinities from
    finite FP64 values. That legacy merge compares narrowed values and IDs;
    it does not recompute a higher-precision global maximum. Nonfinite
    confidence fails identically
    on all ranks after their shared score packet is complete.
    """

    validate_rows(logits, positions, sampling)
    plan.check(logits)
    rows, width = logits.shape
    if type(greedy_count) is not int or greedy_count <= 0:
        raise ValueError("positive greedy candidate count required")
    if rows == 0:
        return ([], []) if with_prob else []
    mapping, widths, offset = plan.mapping, plan.widths, plan.offset
    greedy = sampling is None or sampling.temperature <= 0
    if not greedy and not sampling.top_k:
        probs: list[float] | None = [] if with_prob else None
        chosen = nucleus_rows(logits, positions, sampling, offset=offset, id_map=plan.token_map, gather=plan.gather,
                              probs=probs)
        return (chosen, probs) if with_prob else chosen
    scores = logits.float()
    count = min(plan.max_width, greedy_count) if greedy else min(plan.max_width, int(sampling.top_k) + MARGIN)
    values, tokens, norms = _tp_candidates(plan, scores, count, mapping, offset, greedy=greedy, device_map=plan.device_map,
                                         sorted_mapping=plan.sorted_mapping, source=logits if plan.native_greedy else scores,
                                         with_norm=with_prob and probability is None,
                                         domain_source=logits if greedy else None)
    repair = set()
    if not greedy or count > 1:
        global_k = 1 if greedy else min(int(sampling.top_k), plan.total_width)
        for row in range(rows):
            v = values[:, row].reshape(-1)
            ids = tokens[:, row].reshape(-1)
            real = ids >= 0
            v = v[real]
            cutoff = np.partition(v, len(v) - global_k)[len(v) - global_k]
            for rank, shard_width in enumerate(widths):
                if count < shard_width and np.isfinite(values[rank, row].min()) \
                        and values[rank, row].min() >= cutoff:
                    repair.add(row)
                    break
    narrowed_greedy = greedy
    chosen = _tp_choose(values, tokens, positions, sampling, narrowed_greedy=narrowed_greedy)
    probability_values = values
    probability_tokens = tokens
    if repair:
        # Every rank derives this ordered list from the same gathered packet.
        # Full shards retain the original score evaluation and mapping.
        indices = sorted(repair)
        complete, complete_ids, _ = _tp_candidates(plan, scores[indices], plan.max_width, mapping, offset,
                                                      device_map=plan.device_map)
        if greedy and probability is not None:
            # Confidence normalizes the original capped candidate set. Repair
            # IDs within each shard's cap, rather than expanding that measure
            # to the entire vocabulary when a maximum tie was truncated.
            kept_values = np.full((plan.world, len(indices), count), -np.inf, dtype=np.float32)
            kept_ids = np.full((plan.world, len(indices), count), -1, dtype=np.int64)
            for rank, shard_width in enumerate(widths):
                for index in range(len(indices)):
                    v, ids = complete[rank, index], complete_ids[rank, index]
                    order = np.lexsort((ids, -v))
                    order = order[ids[order] >= 0][:count]
                    kept_values[rank, index, :len(order)] = v[order]
                    kept_ids[rank, index, :len(order)] = ids[order]
            complete, complete_ids = kept_values, kept_ids
        repaired = _tp_choose(complete, complete_ids, [positions[row] for row in indices], sampling)
        for row, token in zip(indices, repaired):
            chosen[row] = token
        repaired_values = {row: (complete[:, index], complete_ids[:, index]) for index, row in enumerate(indices)}
    if not with_prob:
        return chosen
    if probability is not None:
        probs = []
        for row, token in enumerate(chosen):
            v, ids = (repaired_values[row] if row in repair
                      else (probability_values[:, row], probability_tokens[:, row]))
            real = ids.reshape(-1) >= 0
            result = probability(v.reshape(-1)[real][None], ids.reshape(-1)[real][None], [token], sampling)
            if len(result) != 1 or not math.isfinite(result[0]) or not 0.0 <= result[0] <= 1.0:
                raise ValueError("candidate confidence normalization produced an invalid probability")
            probs.append(result[0])
        return chosen, probs
    lse = norms.astype(np.float64)
    top = lse.max(axis=0)
    total = top + np.log(np.exp(lse - top).sum(axis=0))
    if not np.isfinite(total).all():
        raise ValueError("FP32 confidence normalization is nonfinite across vocabulary shards")
    probs = []
    for row, token in enumerate(chosen):
        v, ids = probability_values[:, row], probability_tokens[:, row]
        if row in repair:
            v, ids = repaired_values[row]
        hit = np.nonzero(ids.reshape(-1) == token)[0]
        if not len(hit):
            raise ValueError("selected TP token is absent from its exact score packet")
        probability = float(np.exp(float(v.reshape(-1)[hit[0]]) - total[row]))
        if not math.isfinite(probability):
            raise ValueError("FP32 confidence normalization produced a nonfinite probability")
        probs.append(probability)
    return chosen, probs
