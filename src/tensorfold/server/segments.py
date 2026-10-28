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

LIMIT = 64 << 20         # bytes the cache holds: segment texts and their ids (about 7M tokens of English text)


def _fast_ids_batch(tok):
    native = sys.modules.get("tokenizers")
    if (native is not None and getattr(native, "__version__", None) == "0.23.3"
            and type(tok) is getattr(native, "Tokenizer", None)):
        model = tok.model
        if type(model) is native.models.BPE and model.dropout is None:
            method = getattr(tok, "encode_batch_fast", None)
            if callable(method):
                return method
    return None


@dataclass(frozen=True, slots=True)
class IDsEncoder:
    """One immutable profile's actual method, tokenizer lifetime and identity."""
    tokenizer: Any
    method: Any
    codec: str
    tokenizer_identity: str | None

    def __call__(self, text: str, *, add_special_tokens: bool) -> list[int]:
        # The singleton containers/released GIL need the batch resource mode.
        encoded = (self.method([text], add_special_tokens=add_special_tokens)[0]
                   if self.codec == "encode_batch_fast" else
                   self.method(text, add_special_tokens=add_special_tokens))
        return encoded.ids


def bind_ids_encoder(tok):
    """Resolve a private file-loaded tokenizer once, before profile publication.

    Only the pinned deterministic native BPE API receives fast selection. Other
    providers retain their ordinary call. Tokenizer mutation after binding is
    outside this immutable profile contract; errors never trigger a retry.
    """
    fast = _fast_ids_batch(tok)
    if fast is None:
        return IDsEncoder(tok, tok.encode, "encode", None)
    from tensorfold.token_metadata import tokenizer_identity

    return IDsEncoder(tok, fast, "encode_batch_fast", tokenizer_identity(tok))


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

    @property
    def tok(self):
        return self._batch_binding[0]

    @tok.setter
    def tok(self, value):
        # Only the inspected native API supplies the IDs-only guarantee.
        # Keep the established lazy ordinary call for older/custom providers.
        fast = _fast_ids_batch(value)
        self._batch_binding = (value, fast, "encode_batch_fast" if fast is not None else "encode_batch")

    @property
    def batch_codec(self):
        """Actual batch API, for the separately validated resource contract."""
        return self._batch_binding[2]

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
        binding = self._batch_binding
        batch = binding[1] if binding[1] is not None else binding[0].encode_batch
        fresh = {segment: array("i", encoding.ids)                    # the GIL is free while these encode
                 for segment, encoding in zip(missing, batch(missing, add_special_tokens=False))}
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
