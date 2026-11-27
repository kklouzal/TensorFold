"""Chat prompts encoded a message at a time, each message's tokens cached by its text.

A conversation resends its whole history every turn; encoding it whole held the GIL for seconds at long contexts
(about 2 s at 724k tokens on a GB10) while the engine thread waited. Messages already seen now come from a cache and
new ones are encoded together with the GIL released.
"""

from __future__ import annotations

import json
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


class UnsupportedSegmentOpener(ValueError):
    """The current tokenizer cannot preserve encoding at this split."""


class SegmentEncoder:
    """``tok.encode(text, add_special_tokens=False).ids`` for chat text, a segment (from one ``opener`` to the next) at
    a time. Exact when the opener isolates the two sides before pre-tokenizing. The default requires a raw
    AddedToken. An explicit identity-normalizer route may qualify a normalized one. Thread-safe."""

    def __init__(self, tok: Any, opener: str, limit: int = LIMIT, *, identity_normalized: bool = False) -> None:
        if not self.qualifies(tok, opener, identity_normalized=identity_normalized):
            raise UnsupportedSegmentOpener(f"{opener!r} is not an added token the tokenizer splits on exactly")
        self.tok, self.opener, self.limit = tok, opener, int(limit)
        self.cache: OrderedDict[str, _Segment] = OrderedDict()
        self.held = 0
        self.lock = threading.Lock()
        self._controls = frozenset()

    @staticmethod
    def qualifies(tok: Any, opener: str, *, identity_normalized: bool = False) -> bool:
        """Check that an added token isolates the text on either side before pre-tokenizing.

        No token can overlap its start, where leftmost-longest matching would prefer the earlier one.
        Strip flags, padding and truncation would make separately encoded segments differ.
        """

        added = list(tok.get_added_tokens_decoder().values())
        token = next((t for t in added if t.content == opener), None)
        if token is None or token.lstrip or token.rstrip or token.single_word:
            return False
        if token.special and tok.encode_special_tokens or tok.truncation is not None or tok.padding is not None:
            return False
        others = [t for t in added if t.content != opener]
        if any(opener in t.content[1:] or any(opener.startswith(t.content[i:]) for i in range(1, len(t.content)))
               for t in others):
            return False
        if identity_normalized:
            return token.normalized and SegmentEncoder._identity_normalized_opener(tok, opener, others)
        return not token.normalized

    @staticmethod
    def _identity_normalized_opener(tok, opener, others):
        from importlib.metadata import PackageNotFoundError, version

        try:
            if version("tokenizers") != "0.23.3":
                return False
        except PackageNotFoundError:
            return False
        data = json.loads(tok.to_str())
        if data.get("normalizer") != {"type": "Sequence", "normalizers": []}:
            return False
        model = data.get("model", {})
        if model.get("type") != "BPE" or model.get("dropout") is not None:
            return False
        if data.get("post_processor") != {"type": "ByteLevel", "add_prefix_space": True,
                                          "trim_offsets": False, "use_regex": True}:
            return False
        # The raw AddedToken pass precedes the normalized pass. It cannot
        # consume any part of this opener, and matching at its start must
        # have no longer alternative or neighbor-dependent whitespace.
        for token in others:
            text = token.content
            if token.single_word or token.lstrip or token.rstrip or text.startswith(opener):
                return False
            if not token.normalized and (text in opener or any(
                    opener.endswith(text[:i]) for i in range(1, min(len(text), len(opener)) + 1))):
                return False
        return True

    def encode(self, text: str, *, ids_credit=None, cache_credit=None) -> list[int]:
        return self._encode(text, ids_credit=ids_credit, cache_credit=cache_credit)

    def encode_origin(self, text, controls, replacements, content_bounds, *, ids_credit=None, cache_credit=None):
        return self._encode(text, frozenset(controls), replacements, content_bounds,
                            ids_credit=ids_credit, cache_credit=cache_credit)

    def _encode(self, text, controls=None, replacements=None, content_bounds=None, *, ids_credit=None, cache_credit=None):
        origin = text if controls is not None else None
        if cache_credit is not None:
            # OriginText -> base str can copy: reserve before that conversion too.
            cache_credit(self.limit, str.count(text, self.opener) + 1, str.__len__(text))
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
        held_ids = held_positions = previous_native = 0
        entries = 4 * len(missing) + len(known)
        if ids_credit is not None:
            ids_credit(0, entries=entries)  # Before batch Encoding wrappers/fresh containers exist.
        for segment, encoded in zip(missing, self.tok.encode_batch(missing, add_special_tokens=False)):
            if ids_credit is not None:
                count = len(encoded)
                ids_credit(held_ids + previous_native + 3 * count, entries=entries,
                           positions=held_positions + 4 * count)
            native_ids = encoded.ids
            positions = array("i")
            if ids_credit is None:
                content = content_bounds(segment, native_ids) if content_bounds is not None else None
            else:
                content = None  # Previous segment's lookup has no remaining consumer.
                if content_bounds is not None:
                    content = content_bounds(segment, native_ids, ids_credit=ids_credit,
                                             credit_base=held_ids + 3 * count,
                                             positions_base=held_positions + 4 * count, entries_base=entries)
            if authority:
                for index, token in enumerate(native_ids):
                    if token in authority:
                        offsets = encoded.token_to_chars(index) or (0, 0)
                        automatic = encoded.special_tokens_mask[index] if offsets[0] == offsets[1] else 0
                        if content is not None:
                            offsets = content(token, *offsets)
                        positions.extend((index, *offsets, automatic))
            fresh[segment] = _Segment(array("i", native_ids), positions)
            if ids_credit is not None:
                held_ids += count
                held_positions += len(positions)
                previous_native = count
                content = None
        if ids_credit is not None:
            # Known cache arrays are separately pinned by cache_credit. Pay all
            # fresh arrays, last getter, mapped output and one source slice.
            output_count = maximum_slice = 0
            expansion = max((len(ids) for ids in (replacements or {}).values()), default=1)
            for segment in segments:
                entry = known.get(segment) or fresh[segment]
                maximum_slice = max(maximum_slice, len(entry.ids))
                output_count += len(entry.ids) + (len(entry.controls) // 4 * expansion if origin is not None else 0)
            ids_credit(held_ids + previous_native + output_count + maximum_slice,
                       entries=entries, positions=held_positions + 4)
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
            try:
                for segment, ids in fresh.items() if self._controls == authority else ():
                    size = _size(segment, ids)
                    if segment not in self.cache and size <= self.limit:
                        held = self.held + size
                        self.cache[segment] = ids
                        self.held = held
                while self.held > self.limit and self.cache:
                    self.held -= _size(*self.cache.popitem(last=False))
            except MemoryError:
                # Drop only this cache's owners before a partial publication or
                # eviction can escape with an invalid byte charge.
                self.cache.clear()
                self.held = 0
                raise
        return out


def _size(segment: str, ids: array) -> int:
    """An entry's bytes: its text (1, 2 or 4 bytes a character) and its ids, objects included."""

    size = sys.getsizeof(segment) + sys.getsizeof(ids)
    if isinstance(ids, _Segment):
        size += sys.getsizeof(ids.ids) + sys.getsizeof(ids.controls)
    return size
