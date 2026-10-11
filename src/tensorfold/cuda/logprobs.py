"""Raw target log probabilities with the same FP32 reduction tree at every batch size."""

from numbers import Integral

import torch
import triton as tr
import triton.language as tl

from tensorfold.engine.probabilities import LabelProbabilities


@tr.jit
def _parts(X, P, STRIDE: tl.constexpr, V: tl.constexpr, T: tl.constexpr, B: tl.constexpr, WIDE: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    if WIDE:
        row, part = row.to(tl.int64), part.to(tl.int64)
    cols = part * B + tl.arange(0, B)
    x = tl.load(X + row * STRIDE + cols, cols < V, other=-float("inf")).to(tl.float32)
    peak = tl.max(x, 0)
    mass = tl.sum(tl.exp(x - tl.where(peak == -float("inf"), 0.0, peak)), 0)
    tl.store(P + (row * T + part) * 2, peak)
    tl.store(P + (row * T + part) * 2 + 1, mass)


@tr.jit
def _finish(P, L, T: tl.constexpr, B: tl.constexpr, WIDE: tl.constexpr):
    row = tl.program_id(0)
    if WIDE:
        row = row.to(tl.int64)
    at = tl.arange(0, B)
    peak = tl.load(P + (row * T + at) * 2, at < T, other=-float("inf"))
    mass = tl.load(P + (row * T + at) * 2 + 1, at < T, other=0.0)
    maximum = tl.max(peak, 0)
    total = tl.sum(mass * tl.exp(peak - maximum), 0)
    tl.store(L + row, maximum + tl.log(total))


def _validate_capture(logits, tokens, positions, probabilities, rows):
    """Prove host metadata before allocating or issuing device indexing operations."""

    if (not isinstance(logits, torch.Tensor) or not logits.is_cuda or logits.ndim != 2
            or logits.shape[1] <= 0 or logits.stride(1) != 1):
        raise ValueError("probabilities need CUDA target rows with a nonempty contiguous vocabulary")
    count, vocab = len(tokens), logits.shape[1]
    if len(positions) != count:
        raise ValueError("probabilities need one position per accepted token")
    if any(isinstance(token, bool) or not isinstance(token, Integral) or not 0 <= token < vocab
           for token in tokens):
        raise ValueError("accepted tokens must be integer target vocabulary IDs")
    if any(isinstance(pos, bool) or not isinstance(pos, Integral) or pos < 0 for pos in positions):
        raise ValueError("accepted positions must be nonnegative integers")
    if rows is None:
        if logits.shape[0] != count:
            raise ValueError("probabilities need one target row per accepted token")
    elif (len(rows) != count
          or any(isinstance(row, bool) or not isinstance(row, Integral)
                 or not 0 <= row < logits.shape[0] for row in rows)):
        raise ValueError("accepted rows must be valid integer source row indices")
    if isinstance(probabilities.top, bool) or not isinstance(probabilities.top, Integral) or probabilities.top < 0:
        raise ValueError("the number of top probabilities must be a nonnegative integer")
    labels = getattr(probabilities, "labels", None)
    if labels is not None:
        if type(probabilities) is _LabelCapture:
            return labels                     # private, checked tuple; recursive chunks preserve vocabulary
        # Request-owned labels are mutable; indexing uses this checked snapshot.
        try:
            labels = tuple(labels)
        except TypeError as error:
            raise ValueError("decision labels must be integer target vocabulary IDs") from error
        if any(isinstance(label, bool) or not isinstance(label, Integral) or not 0 <= label < vocab
               for label in labels):
            raise ValueError("decision labels must be integer target vocabulary IDs")
    return labels


def _wide_offsets(rows, stride, tiles):
    """Include masked padding lanes when proving signed32 pointer arithmetic."""

    finish_lanes = 1 << (tiles - 1).bit_length()
    return max((rows - 1) * stride + tiles * 1024 - 1, rows * tiles * 2 - 1,
               (rows - 1) * tiles * 2 + (finish_lanes - 1) * 2 + 1) > 0x7FFFFFFF


def _check_label_identity(probabilities, labels):
    """Callbacks may not reconfigure the request whose label values are being published."""

    try:
        current = tuple(probabilities.labels)
    except (AttributeError, TypeError) as error:
        raise RuntimeError("decision labels changed during capture") from error
    if (len(current) != len(labels)
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value != expected
                   for value, expected in zip(current, labels))):
        raise RuntimeError("decision labels changed during capture")


class _LabelCapture:
    """Bind checked labels and collector callbacks for one logical capture."""

    def __init__(self, probabilities, labels):
        self.labels, self.top = labels, probabilities.top
        self._owner = probabilities
        self._add, self._add_labels = probabilities.add, probabilities.add_labels
        if (type(probabilities) is LabelProbabilities
                and getattr(self._add, "__func__", None) is LabelProbabilities.add
                and getattr(self._add_labels, "__func__", None) is LabelProbabilities.add_labels):
            # These concrete callbacks only store the supplied values. Keep
            # their direct calls; unchanged private metadata needs no rescan.
            self.add, self.add_labels = self._add, self._add_labels

    def add_labels(self, *args):
        _check_label_identity(self._owner, self.labels)
        self._add_labels(*args)
        _check_label_identity(self._owner, self.labels)

    def add(self, *args):
        _check_label_identity(self._owner, self.labels)
        self._add(*args)
        _check_label_identity(self._owner, self.labels)


@torch.no_grad()
def capture(logits, tokens, positions, probabilities, rows=None):
    """Collect read-only target rows with request-exclusively owned collector metadata.

    Labels may be configured between calls, but cannot change during a capture,
    including its callbacks. One checked snapshot governs all bounded chunks.
    """

    if probabilities is None or not tokens:
        return
    labels = _validate_capture(logits, tokens, positions, probabilities, rows)
    if labels and type(probabilities) is not _LabelCapture:
        probabilities = _LabelCapture(probabilities, labels)
    # Bound all vocabulary-sized sorting temporaries, including selected source rows.
    width = max(1, (128 * 1024**2) // (48 * logits.shape[1]))
    if len(tokens) > width:
        for start in range(0, len(tokens), width):
            end = start + width
            capture(logits[start:end] if rows is None else logits, tokens[start:end], positions[start:end],
                    probabilities, None if rows is None else rows[start:end])
        return
    if rows is not None:
        logits = logits.index_select(0, torch.tensor(rows, dtype=torch.long, device=logits.device))
    n, vocab = logits.shape
    tiles = tr.cdiv(vocab, 1024)
    wide = _wide_offsets(n, logits.stride(0), tiles)
    parts = torch.empty((n, tiles, 2), dtype=torch.float32, device=logits.device)
    lse = torch.empty((n,), dtype=torch.float32, device=logits.device)
    _parts[(n, tiles)](logits, parts, logits.stride(0), vocab, tiles, 1024, wide, num_warps=4)
    _finish[(n,)](parts, lse, tiles, tr.next_power_of_2(tiles), wide, num_warps=4)
    ids = torch.tensor(tokens, dtype=torch.long, device=logits.device)[:, None]
    chosen = (logits.gather(1, ids).float()[:, 0] - lse).cpu().tolist()
    if labels:                                   # a decision: the label logits at the prompt's last position
        picked = logits.index_select(1, torch.tensor(labels, dtype=torch.long, device=logits.device)).float()
        for row, pos in enumerate(positions):
            probabilities.add_labels(pos, picked[row].cpu().tolist(), float(lse[row].item()))
    count = min(probabilities.top, vocab)
    if count:
        values = logits.float()
        bits = values.view(torch.int32).to(torch.int64)
        bits = torch.where(values == 0, 0, bits)
        ordered = torch.where(bits < 0, ~bits, bits ^ 0x80000000) - 0x80000000
        token_ids = torch.arange(vocab, dtype=torch.int64, device=logits.device)
        keys = (ordered << 32) | (0xFFFFFFFF - token_ids)
        top_ids = keys.topk(count, dim=-1, sorted=True).indices
        scores = (logits.gather(1, top_ids).float() - lse[:, None]).cpu().tolist()
        alternatives = top_ids.cpu().tolist()
    else:
        alternatives, scores = [[] for _ in tokens], [[] for _ in tokens]
    probabilities.add(positions, tokens, chosen, alternatives, scores)
