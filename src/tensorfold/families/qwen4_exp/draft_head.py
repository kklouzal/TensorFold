"""Sample drafts over listed token ids with the target's keyed rule; committed tokens still use its full vocabulary."""

from __future__ import annotations

import codecs
import os
import sys
from operator import index
from pathlib import Path
import unicodedata
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

VOCAB_FILE = Path(__file__).with_name("cuda") / "draft_vocab.txt"


def _drain_ids(scope, primary):
    """Finish owned file work; retain failures and owner in ordinary traceback frames."""
    native_cause = BaseException.__cause__.__get__(primary) if primary is not None else None
    native_context = BaseException.__context__.__get__(primary) if primary is not None else None
    try:
        errors = scope.drain()
        if not scope.retired:
            errors.append(RuntimeError('draft ID-file owner did not retire'))
        if not errors:
            return
        if primary is None:
            primary = errors[0]
            native_cause = BaseException.__cause__.__get__(primary)
            native_context = BaseException.__context__.__get__(primary)
        statuses = []
        for error in (native_cause, native_context, *errors):
            if error is not None and error is not primary and all(error is not previous for previous in statuses):
                statuses.append(error)
        if len(statuses) == 1:
            raise primary from statuses[0]
        if statuses:
            raise primary from BaseExceptionGroup('draft ID-file retained cleanup failures', statuses)
        raise primary
    except BaseException as failure:
        if primary is None or failure is primary:
            raise
        raise primary from failure

def _listed_ids(path: Path, *, vocab: int, multiple: int) -> list[int]:
    """Legacy int-token syntax, deduplicated/sorted with bounded current-vocabulary padding.

    The one opened regular file is borrowed unchanged. Consume at most its
    admitted byte size plus one, with <=64KiB read/decoded chunks and constant
    token state; no full-token or full-file string. The set/output has <=vocab
    entries, and padding scans no more than vocab candidates. Unicode decimal
    digits, signs and single between-digit underscores retain int(s) syntax.
    The current CPython decimal digit limit is borrowed unchanged for the
    operation (zero means unlimited). No gzip/comments/decimal/scientific syntax is added to the old MLX interface.
    """
    from tensorfold.file_io import FileStreams

    try:
        if isinstance(vocab, bool) or isinstance(multiple, bool):
            raise TypeError('boolean count')
        vocab, multiple = index(vocab), index(multiple)
    except TypeError as error:
        raise ValueError('draft vocabulary and padding counts must be integers') from error
    if not 1 <= multiple <= vocab <= 1 << 32:
        raise ValueError('draft counts require positive uint32 vocabulary and padding within it')
    scope, primary = FileStreams(max_files=1), None
    found = set()
    digit_limit, digit_count = sys.get_int_max_str_digits(), 0
    value = 0
    present = digit_seen = last_digit = negative = False

    def finish():
        nonlocal value, digit_count, present, digit_seen, last_digit, negative
        if present:
            if not digit_seen or not last_digit:
                raise ValueError('draft IDs require integer token syntax')
            found.add(value)
        value = digit_count = 0
        present = digit_seen = last_digit = negative = False

    def consume(text):
        nonlocal value, digit_count, present, digit_seen, last_digit, negative
        for char in text:
            if char.isspace():
                finish()
            elif char.isdecimal():
                digit_count += 1
                if digit_limit and digit_count > digit_limit:
                    raise ValueError("draft ID token exceeds the configured CPython integer digit limit")
                present = digit_seen = last_digit = True
                value = value * 10 + unicodedata.decimal(char)
                if value >= vocab or negative and value:
                    raise ValueError('draft IDs must address nonnegative current vocabulary rows')
            elif char in '+-' and not present:
                present, negative = True, char == '-'
            elif char == '_' and last_digit:
                present, last_digit = True, False
            else:
                raise ValueError('draft IDs require integer token syntax')

    try:
        record, before = scope.open(path, os.O_RDONLY | os.O_NONBLOCK, 'rb')
        stream, remaining = record.stream, before.st_size
        decoder = codecs.getincrementaldecoder('utf8')()
        while remaining:
            block = stream.read(min(remaining, 64 << 10))
            if not block:
                raise ValueError('draft ID-file changed during its borrow')
            remaining -= len(block)
            consume(decoder.decode(block))
        if stream.read(1):
            raise ValueError('draft ID-file grew during its borrow')
        consume(decoder.decode(b'', final=True))
        finish()
        if not found:
            raise ValueError('draft ID-file contains no token IDs')
        target = ((len(found) + multiple - 1) // multiple) * multiple
        if target > vocab:
            raise ValueError('draft ID padding cannot fit the current vocabulary')
        candidate = 0
        while len(found) < target:
            if candidate not in found:
                found.add(candidate)
            candidate += 1
        return sorted(found)
    except BaseException as error:
        primary = error
        raise
    finally:
        _drain_ids(scope, primary)


def draft_ids(path: Path = VOCAB_FILE, multiple: int = 64, *, vocab: int = 1 << 32) -> np.ndarray:
    """Sort listed current-vocabulary IDs and pad with the smallest unlisted IDs.

    The regular UTF8 file uses Python decimal integer tokens, including Unicode
    decimal digits and between-digit underscores, with the configured Python
    digit limit. Counts and IDs must fit the current vocabulary and uint32;
    ``multiple`` is positive and padding must fit. Parsing borrows the opened
    file unchanged and uses bounded chunks before the original output conversion.
    """

    return np.array(_listed_ids(path, vocab=vocab, multiple=multiple), dtype=np.uint32)


def cut_head(lm_head: Any, ids: np.ndarray) -> nn.QuantizedLinear:
    """The rows of a 4-bit quantized vocabulary head for ``ids`` as a quantized linear of their own."""

    from tensorfold.families.qwen4_exp.decode import _PreparedLinear
    from tensorfold.kernels.qwen.flash_next.v1.base import QWeights

    ids = np.asarray(ids)
    if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
        raise ValueError("draft-head IDs must be a nonempty one-dimensional integer array")
    if ids.min() < 0 or ids.max() >= lm_head.weight.shape[0] or ids.max() > 2**32 - 1:
        raise ValueError("draft-head IDs must address current vocabulary rows within uint32 indexing")
    index = mx.array(ids.astype(np.uint32))
    source = QWeights.of(lm_head)
    part = QWeights(mx.take(source.weight, index, axis=0), mx.take(source.scales, index, axis=0),
                    mx.take(source.biases, index, axis=0), source.bits, source.group)
    return _PreparedLinear([part], source.group)


def sample(logits: mx.array, ids: mx.array, sampling: Any, positions: Sequence[int] | mx.array) -> mx.array:
    """Return lazy uint32 token ids [R] using keyed sampling over ``ids``, or greedy selection when ``sampling`` is None."""

    from tensorfold.engine.gpu_sampling import sample as gpu_sample

    return gpu_sample(logits.reshape(-1, logits.shape[-1]), sampling, positions, ids=ids)
