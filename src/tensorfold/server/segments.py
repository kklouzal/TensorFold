"""Chat prompts encoded a message at a time, each message's tokens cached by its text.

A conversation resends its whole history every turn; encoding it whole held the GIL for seconds at long contexts
(about 2 s at 724k tokens on a GB10) while the engine thread waited. Messages already seen now come from a cache and
new ones are encoded together with the GIL released.
"""

from __future__ import annotations

import sys
import threading
from array import array
from collections import OrderedDict
from typing import Any

LIMIT = 64 << 20         # bytes the cache holds: segment texts and their ids (about 7M tokens of English text)


class SegmentEncoder:
    """``tok.encode(text, add_special_tokens=False).ids`` for chat text, a segment (from one ``opener`` to the next) at
    a time. Exact when ``opener`` is an added token the tokenizer splits its input at before normalizing and
    pre-tokenizing (``qualifies``): no merge, normalization or other added token crosses it. Thread-safe."""

    def __init__(self, tok: Any, opener: str, limit: int = LIMIT) -> None:
        if not self.qualifies(tok, opener):
            raise ValueError(f"{opener!r} is not an added token the tokenizer splits on exactly")
        self.tok, self.opener, self.limit = tok, opener, int(limit)
        self.cache: OrderedDict[str, array] = OrderedDict()
        self.held = 0
        self.lock = threading.Lock()

    @staticmethod
    def qualifies(tok: Any, opener: str) -> bool:
        """``opener`` is an added token matched on the raw text (no strip, single-word or normalized flags; special
        tokens split, as ``encode_special_tokens`` off leaves them); no other added token can match across its start
        (none ends with a prefix of it or holds it past its own first character, which leftmost-longest matching
        would prefer to it); and no truncation or padding acts on each segment's encoding apart."""

        added = list(tok.get_added_tokens_decoder().values())
        token = next((t for t in added if t.content == opener), None)
        if token is None or token.lstrip or token.rstrip or token.normalized or token.single_word:
            return False
        if token.special and tok.encode_special_tokens or tok.truncation is not None or tok.padding is not None:
            return False
        return not any(opener in t.content[1:] or any(opener.startswith(t.content[i:])
                                                      for i in range(1, len(t.content)))
                       for t in added if t.content != opener)

    def encode(self, text: str) -> list[int]:
        parts = text.split(self.opener)
        segments = ([parts[0]] if parts[0] else []) + [self.opener + p for p in parts[1:]]
        with self.lock:
            known = {s: self.cache[s] for s in segments if s in self.cache}
            for segment in known:
                self.cache.move_to_end(segment)
        missing = list(dict.fromkeys(s for s in segments if s not in known))
        fresh = {segment: array("i", encoding.ids)                    # the GIL is free while these encode
                 for segment, encoding in zip(missing, self.tok.encode_batch(missing, add_special_tokens=False))}
        out: list[int] = []
        for segment in segments:
            ids = known.get(segment)
            out.extend(ids if ids is not None else fresh[segment])
        with self.lock:
            try:
                for segment, ids in fresh.items():
                    if segment not in self.cache:
                        held = self.held + _size(segment, ids)
                        self.cache[segment] = ids
                        self.held = held
                while self.held > self.limit and len(self.cache) > 1:   # the newest entry stays, however large
                    self.held -= _size(*self.cache.popitem(last=False))
            except MemoryError:
                # Owned string/array entries have no destructor callbacks.
                # Restore this optional cache's charge invariant before failure escapes.
                self.cache.clear()
                self.held = 0
                raise
        return out


def _size(segment: str, ids: array) -> int:
    """An entry's bytes: its text (1, 2 or 4 bytes a character) and its ids, objects included."""

    return sys.getsizeof(segment) + sys.getsizeof(ids)
