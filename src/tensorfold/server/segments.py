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
from dataclasses import dataclass
from typing import Any

from tensorfold.server.prompt_origin import PromptEncodingError

LIMIT = 64 << 20         # bytes the cache holds: segment texts and their ids (about 7M tokens of English text)


@dataclass(frozen=True, slots=True)
class _Segment:
    ids: array
    # Original-tokenizer control positions: token index, start, end, automatic.
    # Origin decisions are request-local and NEVER cached with the text.
    controls: array

    def __iter__(self):
        return iter(self.ids)


class SegmentEncoder:
    """``tok.encode(text, add_special_tokens=False).ids`` for chat text, a segment (from one ``opener`` to the next) at
    a time. Exact when ``opener`` is an added token the tokenizer splits its input at before normalizing and
    pre-tokenizing (``qualifies``): no merge, normalization or other added token crosses it. Thread-safe."""

    def __init__(self, tok: Any, opener: str, limit: int = LIMIT) -> None:
        if not self.qualifies(tok, opener):
            raise ValueError(f"{opener!r} is not an added token the tokenizer splits on exactly")
        self.tok, self.opener, self.limit = tok, opener, int(limit)
        self.cache: OrderedDict[str, _Segment] = OrderedDict()
        self.held = 0
        self.lock = threading.Lock()
        self._controls = frozenset()

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
        return self._encode(text)

    def encode_origin(self, text, controls, replacements, content_bounds):
        return self._encode(text, frozenset(controls), replacements, content_bounds)

    def _encode(self, text, controls=None, replacements=None, content_bounds=None):
        origin = text if controls is not None else None
        raw = str.__str__(text)
        parts = raw.split(self.opener)
        segments = ([parts[0]] if parts[0] else []) + [self.opener + p for p in parts[1:]]
        with self.lock:
            if controls is not None and controls != self._controls:
                # A new reserved-ID authority invalidates cached position data.
                self.cache.clear()
                self.held = 0
                self._controls = controls
            authority = self._controls
            known = {s: self.cache[s] for s in segments if s in self.cache}
            for segment in known:
                self.cache.move_to_end(segment)
        missing = list(dict.fromkeys(s for s in segments if s not in known))
        fresh = {}
        for segment, encoded in zip(missing, self.tok.encode_batch(missing, add_special_tokens=False)):
            native_ids = encoded.ids
            positions = array("i")
            content = content_bounds(segment, native_ids) if content_bounds is not None else None
            if authority:
                for index, token in enumerate(native_ids):
                    if token in authority:
                        offsets = encoded.token_to_chars(index) or (0, 0)
                        automatic = encoded.special_tokens_mask[index] if offsets[0] == offsets[1] else 0
                        if content is not None:
                            offsets = content(token, *offsets)
                        positions.extend((index, *offsets, automatic))
            fresh[segment] = _Segment(array("i", native_ids), positions)
        out: list[int] = []
        base = 0
        for segment in segments:
            ids = known.get(segment)
            entry = ids if ids is not None else fresh[segment]
            if origin is None:
                out.extend(entry.ids)
            else:
                at = 0
                for offset in range(0, len(entry.controls), 4):
                    index, start, end, automatic = entry.controls[offset:offset + 4]
                    if not 0 <= start <= end <= len(segment):
                        raise PromptEncodingError("cached tokenizer offsets exceed their segment")
                    token = entry.ids[index]
                    if start == end and not automatic:
                        raise PromptEncodingError("reserved input token has no source origin")
                    out.extend(entry.ids[at:index])
                    if start != end and any(origin._mask[base + start:base + end]):
                        out.extend(replacements[token])
                    else:
                        out.append(token)
                    at = index + 1
                out.extend(entry.ids[at:])
            base += len(segment)
        with self.lock:
            for segment, ids in fresh.items() if self._controls == authority else ():
                size = _size(segment, ids)
                if segment not in self.cache and size <= self.limit:
                    self.cache[segment] = ids
                    self.held += size
            while self.held > self.limit and self.cache:
                self.held -= _size(*self.cache.popitem(last=False))
        return out


def _size(segment: str, ids: array) -> int:
    """An entry's bytes: its text (1, 2 or 4 bytes a character) and its ids, objects included."""

    size = sys.getsizeof(segment) + sys.getsizeof(ids)
    if isinstance(ids, _Segment):
        size += sys.getsizeof(ids.ids) + sys.getsizeof(ids.controls)
    return size
