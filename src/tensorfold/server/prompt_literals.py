"""Keep native template evaluation and distinguish request text at token encoding.

Jinja receives the original logical values with out-of-band character origins.
Only the final rendered media view uses slots; text encoding retains the native
sequence except reserved IDs whose source overlaps request-origin characters.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections import OrderedDict
from array import array
from bisect import bisect_left

from tensorfold.server.prompt_origin import (OriginText, PromptEncodingError, origin_concat, origin_environment,
                                            request_values, trust_message_roles, trust_policy_strings,
                                            trusted, trusted_values)

MAX_RENDER_BYTES = 96 << 20


def _slot_pattern(literals):
    if not literals:
        return re.compile(r"(?!)")
    head = literals[0][0].rsplit("_", 2)[0] + "_"
    if all(marker.startswith(head) and marker[len(head):-1].isdigit() and marker.endswith("_")
           for marker, _ in literals):
        return re.compile(re.escape(head) + r"[0-9]+_")
    return re.compile("|".join(re.escape(k) for k, _ in sorted(literals, key=lambda p: -len(p[0]))))


def restore_literals(text, literals):
    if not literals:
        return text
    values = dict(literals)
    return _slot_pattern(literals).sub(lambda match: values.get(match.group(), match.group()), text)


def _restored_origin(text, literals):
    """Media output: expanded trusted syntax plus the same opaque literal slots."""
    values, pieces, at = dict(literals), [], 0
    if not values:
        return trusted(text)
    for match in _slot_pattern(literals).finditer(text):
        if match.group() not in values:
            continue
        pieces.extend((trusted(text[at:match.start()]), OriginText(values[match.group()])))
        at = match.end()
    pieces.append(trusted(text[at:]))
    return origin_concat(pieces)


def _media_view(text, authority):
    # Quarantine ALL request-origin spans after native evaluation. Otherwise a
    # processor normalizer could create an image/control spelling before the
    # final token-offset origin check. Trusted media syntax stays visible.
    raw = str.__str__(text)
    root = (authority[-1][1] if authority else "tfl_") + "data_"
    used = set(re.findall(re.escape(root) + r"([0-9]+)_", raw))
    nonce = 0
    while str(nonce) in used:
        nonce += 1
    prefix = root + str(nonce) + "_"
    parts, slots, known, at = [], [], {}, 0
    for match in re.finditer(b"\1+", text._mask):
        value = raw[match.start():match.end()]
        marker = known.get(value)
        if marker is None:
            marker = prefix + str(len(slots)) + "_"
            known[value] = marker
            slots.append((marker, value))
        parts.extend((raw[at:match.start()], marker))
        at = match.end()
    parts.append(raw[at:])
    return "".join(parts), tuple(slots)


class RenderedPrompt(str):
    """Immutable native logical output, origin, and processor-safe media view."""
    __slots__ = ("encoded_text", "literals", "control_literals", "origin", "_preencoded", "_renderer", "_ordinary_codec",
                 "_slot_bytes")

    def __new__(cls, text, literals, *, control_literals=None):
        literals = tuple(literals)
        authority = literals if control_literals is None else tuple(control_literals)
        origin = text if isinstance(text, OriginText) else _restored_origin(text, literals)
        encoded, slots = _media_view(origin, authority)
        obj = str.__new__(cls, str.__str__(origin))
        object.__setattr__(obj, "origin", origin)
        object.__setattr__(obj, "literals", slots)
        object.__setattr__(obj, "control_literals", authority)
        object.__setattr__(obj, "encoded_text", encoded)
        object.__setattr__(obj, "_slot_bytes", {marker: len(value.encode("utf-8")) for marker, value in slots})
        object.__setattr__(obj, "_preencoded", None)
        object.__setattr__(obj, "_renderer", None)
        object.__setattr__(obj, "_ordinary_codec", None)
        return obj

    def __setattr__(self, name, value):
        raise AttributeError("rendered prompt provenance is immutable")

    def __delattr__(self, name):
        raise AttributeError("rendered prompt provenance is immutable")

    def expanded_text_bytes(self, text):
        """Logical pre-tokenization byte work, not opaque-slot representation."""
        size = len(text.encode("utf-8"))
        for match in _slot_pattern(self.literals).finditer(text):
            if match.group() in self._slot_bytes:
                size += self._slot_bytes[match.group()] - len(match.group())
        return size


def trusted_prompt_text(text):
    """Syntax discovery/processors cannot interpret request control strings."""
    return text.encoded_text if isinstance(text, RenderedPrompt) else text


def append_trusted(text, suffix):
    if not isinstance(suffix, str):
        raise TypeError("trusted prompt suffix must be text")
    if not isinstance(text, RenderedPrompt):
        return text + suffix
    result = RenderedPrompt(origin_concat((text.origin, trusted(suffix))), text.literals,
                            control_literals=text.control_literals)
    object.__setattr__(result, "_renderer", text._renderer)
    object.__setattr__(result, "_ordinary_codec", text._ordinary_codec)
    return result


def native_tokenizer(tokenizer):
    """The owned Rust tokenizer behind the supported HF/MLX facades."""
    pending, seen = [(tokenizer, 0)], set()
    while pending:
        item, depth = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if callable(getattr(item, "to_str", None)) and callable(getattr(item, "get_added_tokens_decoder", None)):
            return item
        if depth < 3:
            for name in ("backend_tokenizer", "_tokenizer", "tokenizer", "_inner"):
                child = getattr(item, name, None)
                if child is not None and child is not item:
                    pending.append((child, depth + 1))
    return None


def bounded_render(template, kwargs, max_render_bytes=MAX_RENDER_BYTES, *, render_credit=None):
    if type(max_render_bytes) is not int or max_render_bytes <= 0:
        raise PromptEncodingError("render byte budget must be a positive integer")
    if render_credit is not None:
        render_credit(0, 0)
    parts, used, chars = [], 0, 0
    for part in template.generate(**kwargs):
        if render_credit is not None:
            chars += len(part)
            render_credit(chars, len(parts) + 1)  # Before owned origin/UTF8/list/concat materialization.
        # Owned origin compiler marks every dynamic output. Raw strings here
        # are static template output (also useful for a plain Jinja caller).
        part = part if isinstance(part, OriginText) else trusted(part)
        used += len(part.encode("utf-8"))
        if used > max_render_bytes:
            raise PromptEncodingError("rendered prompt exceeds the chat-render byte budget")
        parts.append(part)
    return origin_concat(parts)


class PromptLiterals:
    """Captured syntax/role authority with a bounded tokenizer-local codec cache."""
    def __init__(self, tokenizer_data, template_sources, specials=()):
        added = tokenizer_data.get("added_tokens", [])
        spellings = {t["content"] for t in added if t.get("special") is True and t.get("content")}
        spellings.update(s for s in specials if isinstance(s, str) and s)
        self.spellings = tuple(sorted(spellings, key=lambda s: (-len(s), s)))
        source = "\0".join(template_sources)
        prefix = "tfl_" + hashlib.sha256(source.encode()).hexdigest()[:12] + "_"
        while prefix in source or any(prefix in t.get("content", "") for t in added):
            prefix += "x"
        self.prefix = prefix
        self.literals = tuple((prefix + str(i) + "_", value) for i, value in enumerate(self.spellings))
        self.literals += ((prefix + "e_", prefix),)
        self._encoders = []
        self._lock = threading.Lock()

    def protect(self, value, *, max_bytes=MAX_RENDER_BYTES):
        # Compatibility name: no input character is substituted before Jinja.
        used = 0
        def measure(item, depth=0):
            nonlocal used
            if depth > 128:
                raise PromptEncodingError("chat values exceed the rendering nesting limit")
            if isinstance(item, str):
                used += len(item.encode("utf-8"))
                if used > max_bytes:
                    raise PromptEncodingError("chat values exceed the render-work byte budget")
            elif isinstance(item, dict):
                for k, v in item.items():
                    measure(k, depth + 1)
                    measure(v, depth + 1)
            elif isinstance(item, (list, tuple)):
                for v in item:
                    measure(v, depth + 1)
        measure(value)
        return request_values(value), True

    def encode(self, tokenizer, text, *, segments=None, ids_credit=None, cache_credit=None):
        options = {}
        if ids_credit is not None:
            options["ids_credit"] = ids_credit
        if cache_credit is not None:
            options["cache_credit"] = cache_credit
        def plain(value):
            if segments is not None:
                return segments.encode(value, **options)
            if ids_credit is None:
                return tokenizer.encode(value, add_special_tokens=False).ids
            encoded = tokenizer.encode(value, add_special_tokens=False)
            ids_credit(len(encoded), entries=1)
            return encoded.ids
        if not isinstance(text, RenderedPrompt):
            return plain(text)
        if not self.spellings:
            # No reserved token authority is declared by this model metadata.
            # Ordinary lexical codecs need no Rust tokenizer clone or slots.
            logical = str.__str__(text)
            return plain(logical)
        actual = native_tokenizer(tokenizer)
        if actual is None:
            raise PromptEncodingError("reserved prompt tokens require an origin-aware native tokenizer")
        with self._lock:
            encoder = next((e for owner, e in self._encoders if owner is actual), None)
            if encoder is None:
                encoder = LiteralEncoder(actual, self.literals, **({"ids_credit": ids_credit} if ids_credit is not None else {}))
                self._encoders.append((actual, encoder))
                if len(self._encoders) > 2:
                    self._encoders.pop(0)
        # Equal visible strings can have different control-token provenance.
        return encoder.encode_origin(text.origin, segments=segments, **options)


def _without_prepend(value):
    if isinstance(value, list):
        return [_without_prepend(v) for v in value if not isinstance(v, dict) or v.get("type") != "Prepend"]
    if not isinstance(value, dict):
        return value
    result = {k: _without_prepend(v) for k, v in value.items()}
    if result.get("type") == "ByteLevel":
        result["add_prefix_space"] = False
    if result.get("type") == "Metaspace":
        if "prepend_scheme" in result:
            result["prepend_scheme"] = "never"
        if "add_prefix_space" in result:
            result["add_prefix_space"] = False
    return result


class LiteralEncoder:
    """One authoritative reserved-ID set; no synthetic IDs or tokenizer mutation."""
    def __init__(self, tokenizer, literals, *, ids_credit=None):
        from tokenizers import Tokenizer

        if ids_credit is None:
            data = json.loads(tokenizer.to_str())
        else:
            ids_credit(0, metadata_snapshot=True)
            source = tokenizer.to_str()
            ids_credit(0, metadata_chars=2 * len(source))  # Source graph and lexical clone coexist.
            data = json.loads(source)
        self.tokenizer = tokenizer
        control = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
        # Configured role tokens need not carry the added-token special flag.
        for _, spelling in literals[:-1]:
            token_id = tokenizer.token_to_id(spelling)
            if token_id is None:
                raise PromptEncodingError("declared control spelling has no exact tokenizer ID")
            control.add(token_id)
        self.control = frozenset(control)
        self._strip_ids = frozenset(t["id"] for t in data.get("added_tokens", [])
                                   if t["id"] in control and (t.get("lstrip") or t.get("rstrip")))
        self._unstripped = None
        self._strip_lock = threading.Lock()
        lexical = copy.deepcopy(data)
        lexical["added_tokens"] = [t for t in lexical.get("added_tokens", []) if t["id"] not in control]
        model = lexical["model"]
        if model.get("type") == "BPE":
            blocked = {piece for piece, index in model["vocab"].items() if index in control}
            model["ignore_merges"] = False
            model["merges"] = [pair for pair in model.get("merges", [])
                               if "".join(pair.split(" ") if isinstance(pair, str) else pair) not in blocked]
        elif model.get("type") == "Unigram":
            prefix = literals[-1][0]
            for index in control:
                if index < len(model["vocab"]) and index != model.get("unk_id"):
                    model["vocab"][index][0] = prefix + "disabled_" + str(index)
        lexical["normalizer"] = _without_prepend(lexical.get("normalizer"))
        if isinstance(lexical["normalizer"], dict) and lexical["normalizer"].get("type") == "Prepend":
            lexical["normalizer"] = None
        lexical["pre_tokenizer"] = _without_prepend(lexical.get("pre_tokenizer"))
        ordinary = Tokenizer.from_str(json.dumps(lexical))
        ordinary.no_padding()
        ordinary.no_truncation()
        ordinary.encode_special_tokens = True
        self.replacements = {}
        for token_id in control:
            spelling = tokenizer.decode([token_id], skip_special_tokens=False)
            if ids_credit is None:
                ids = tuple(ordinary.encode(spelling, add_special_tokens=False).ids)
            else:
                encoded = ordinary.encode(spelling, add_special_tokens=False)
                ids_credit(sum(map(len, self.replacements.values())) + len(encoded), entries=len(control))
                ids = tuple(encoded.ids)
            if not ids or control.intersection(ids):
                raise PromptEncodingError("tokenizer cannot represent a control spelling as ordinary text")
            if tokenizer.decode(list(ids), skip_special_tokens=False) != spelling:
                raise PromptEncodingError("ordinary literal tokenization does not preserve its text")
            self.replacements[token_id] = ids
        self.literals = tuple(literals)

    def content_bounds(self, text, native_ids, *, ids_credit=None, credit_base=0, positions_base=0, entries_base=0):
        """Native content offsets, excluding AddedToken strip-only whitespace.

        One immutable alternate matcher per codec retains the actual normalizer
        and model; only strip flags differ. Its request-local positions are not
        retained by the preparation owner. No bespoke whitespace/normalizer map.
        """
        if not self._strip_ids or self._strip_ids.isdisjoint(native_ids):
            return None
        with self._strip_lock:
            if self._unstripped is None:
                from tokenizers import Tokenizer
                if ids_credit is None:
                    data = json.loads(self.tokenizer.to_str())
                else:
                    ids_credit(credit_base, metadata_snapshot=True)
                    source = self.tokenizer.to_str()
                    ids_credit(credit_base, metadata_chars=2 * len(source))
                    data = json.loads(source)
                for token in data.get("added_tokens", []):
                    token["lstrip"] = token["rstrip"] = False
                other = Tokenizer.from_str(json.dumps(data))
                other.no_padding()
                other.no_truncation()
                self._unstripped = other
            other = self._unstripped
        encoded = other.encode(str.__str__(text), add_special_tokens=False)
        if ids_credit is not None:
            ids_credit(credit_base + len(encoded), entries=entries_base + 3 * len(self._strip_ids) + 1,
                       positions=positions_base + 2 * len(encoded))
        positions = {}
        for index, token_id in enumerate(encoded.ids):
            if token_id in self._strip_ids:
                extent = encoded.token_to_chars(index)
                if extent is not None:
                    pair = positions.get(token_id)
                    if pair is None:
                        pair = positions[token_id] = (array("i"), array("i"))
                    starts, ends = pair
                    starts.append(extent[0])
                    ends.append(extent[1])

        def content(token_id, start, end):
            if token_id not in self._strip_ids or start == end:
                return start, end
            starts, ends = positions.get(token_id, ((), ()))
            index = bisect_left(starts, start)
            if index >= len(starts) or not start <= starts[index] < ends[index] <= end:
                raise PromptEncodingError("native stripped control content has no unambiguous source extent")
            return starts[index], ends[index]
        return content

    def encode_origin(self, text, *, segments=None, ids_credit=None, cache_credit=None):
        if segments is not None:
            if segments.tok is not self.tokenizer:
                raise PromptEncodingError("segment cache belongs to a different tokenizer")
            options = {}
            if ids_credit is not None:
                options["ids_credit"] = ids_credit
            if cache_credit is not None:
                options["cache_credit"] = cache_credit
            return segments.encode_origin(text, self.control, self.replacements, self.content_bounds, **options)
        encoded = self.tokenizer.encode(str.__str__(text), add_special_tokens=False)
        budget = 0
        if ids_credit is not None:
            # Pay native ID/mask conversions before the first Python getter.
            budget = 2 * len(encoded)
            ids_credit(budget, entries=1, positions=4 * len(encoded))
        native_ids = encoded.ids
        if ids_credit is not None:
            # Only actual control hits can expand; reserve one bulk mapped-output
            # upper before any append, preserving native normalization and IDs.
            mapped = len(native_ids) + sum(len(self.replacements[token]) - 1
                                          for token in native_ids if token in self.control)
            budget += mapped
            ids_credit(budget, entries=1, positions=4 * len(encoded))
        content = (self.content_bounds(text, native_ids) if ids_credit is None else
                   self.content_bounds(text, native_ids, ids_credit=ids_credit, credit_base=budget,
                                       positions_base=4 * len(encoded), entries_base=1))
        out = []
        for index, token_id in enumerate(native_ids):
            if token_id not in self.control:
                out.append(token_id)
                continue
            start, end = encoded.token_to_chars(index) or (0, 0)
            automatic = encoded.special_tokens_mask[index] if start == end else 0
            if content is not None:
                start, end = content(token_id, start, end)
            if not 0 <= start <= end <= len(text):
                raise PromptEncodingError("tokenizer offsets exceed native rendered text")
            if token_id in self.control and start == end and not automatic:
                raise PromptEncodingError("reserved input token has no source origin")
            if token_id in self.control and start != end and any(text._mask[start:end]):
                out.extend(self.replacements[token_id])
            else:
                out.append(token_id)
        return out

    def encode(self, text):
        return self.encode_origin(_restored_origin(text, self.literals))


def encode_hf_prompt(tokenizer, text):
    if not isinstance(text, RenderedPrompt):
        return [int(t) for t in tokenizer.encode(text)]
    if text._preencoded is not None:
        owner, ids = text._preencoded
        if owner is not tokenizer:
            raise PromptEncodingError("ordinary prompt IDs belong to a different tokenizer")
        return list(ids)
    if text._ordinary_codec is not None:
        owner, reserved, add_special = text._ordinary_codec
        if owner is not tokenizer:
            raise PromptEncodingError("ordinary prompt belongs to a different tokenizer")
        # Re-encode the WHOLE new logical prompt: suffix IDs may merge.
        value = (tokenizer.encode(str.__str__(text)) if add_special is None
                 else tokenizer.encode(str.__str__(text), add_special_tokens=add_special))
        ids = tuple(int(t) for t in (value.ids if hasattr(value, "ids") else value))
        if reserved.intersection(ids):
            raise PromptEncodingError("ordinary appended prompt emitted controls without an origin adapter")
        return list(ids)
    if text._renderer is not None:
        return text._renderer.encode(tokenizer, text)
    backend = native_tokenizer(tokenizer)
    if backend is None:
        raise PromptEncodingError("tokenizer does not expose ordinary literal encoding")
    return LiteralEncoder(backend, text.control_literals).encode_origin(text.origin)


def _declared_controls(tokenizer):
    """Authority exposed by an ordinary codec; never infer controls by spelling."""
    ids, spellings = set(), set()
    decoder = getattr(tokenizer, "get_added_tokens_decoder", None)
    if callable(decoder):
        for token_id, token in decoder().items():
            if token.special:
                ids.add(int(token_id))
                spellings.add(token.content)
    roles = getattr(tokenizer, "special_tokens_map", {})
    for value in roles.values():
        spellings.update(value if isinstance(value, (list, tuple)) else [value])
    for role in ("bos", "eos", "pad", "unk"):
        value = getattr(tokenizer, role + "_token", None)
        if isinstance(value, str) and value:
            spellings.add(value)
        value = getattr(tokenizer, role + "_token_id", None)
        if type(value) is int:
            ids.add(value)
    stops = getattr(tokenizer, "eos_token_ids", ())
    if stops is not None:
        values = (stops,) if type(stops) is int else stops
        ids.update(value for value in values if type(value) is int)
    lookup = getattr(tokenizer, "token_to_id", None)
    if lookup is None:
        lookup = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(lookup):
        for spelling in spellings:
            token_id = lookup(spelling)
            if type(token_id) is int:
                ids.add(token_id)
    return ids, tuple(word for word in spellings if isinstance(word, str) and word)


def _ordinary_render(tokenizer, messages, kwargs):
    """Preserve an ordinary lexical/counting codec without a fabricated HF path.

    This is NOT an origin adapter for an opaque control-producing renderer. Its
    already-produced IDs must contain none of the codec's declared controls.
    Codecs that do not declare any reserved authority make no protection claim.
    """
    if not all(callable(getattr(tokenizer, name, None)) for name in ("apply_chat_template", "encode", "decode")):
        raise PromptEncodingError("custom chat renderer needs its owned literal-origin adapter")
    reserved, spellings = _declared_controls(tokenizer)
    def contains(value, depth=0):
        if depth > 128:
            raise PromptEncodingError("chat values exceed the rendering nesting limit")
        if isinstance(value, str):
            return any(word in value for word in spellings)
        if isinstance(value, dict):
            return any(contains(k, depth + 1) or contains(v, depth + 1) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return any(contains(v, depth + 1) for v in value)
        return False
    if contains((messages, kwargs)):
        raise PromptEncodingError("custom chat renderer needs its owned literal-origin adapter")
    options = dict(kwargs)
    options["tokenize"] = False
    value = tokenizer.apply_chat_template(messages, **options)
    if isinstance(value, str):
        logical = value
        encoded = tokenizer.encode(value)
        ids = encoded.ids if hasattr(encoded, "ids") else encoded
        add_special = None
    else:
        ids = list(value)
        logical = tokenizer.decode(ids, skip_special_tokens=False)
        add_special = False
    ids = tuple(int(value) for value in ids)
    if reserved.intersection(ids):
        raise PromptEncodingError("custom chat renderer emitted controls without its owned literal-origin adapter")
    if len(logical.encode("utf-8")) > MAX_RENDER_BYTES:
        raise PromptEncodingError("rendered prompt exceeds the chat-render byte budget")
    result = RenderedPrompt(OriginText(logical), ())
    object.__setattr__(result, "_preencoded", (tokenizer, ids))
    object.__setattr__(result, "_ordinary_codec", (tokenizer, frozenset(reserved), add_special))
    return result


class HFPromptRenderer:
    """App-owned, synchronized preparation cache; at most two native routes.

    No state is attached to the tokenizer or HF's shared environment. Entries
    capture immutable model metadata, not request values. Eviction cannot revoke
    an in-flight prompt's literal authority. App teardown releases this owner.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def _entry(self, backend, sources, specials, *, jinja=False):
        key = (jinja, tuple(sources), tuple(specials))
        with self._lock:
            found = self._entries.get(key)
            if found is not None and found[0] is backend:
                self._entries.move_to_end(key)
                return found
            protection = PromptLiterals(json.loads(backend.to_str()), sources, specials)
            template = None
            if jinja:
                from transformers.utils.chat_template_utils import _compile_jinja_template
                native = _compile_jinja_template(sources[0])
                template = origin_environment(native.environment).from_string(sources[0])
            found = (backend, protection, template)
            self._entries[key] = found
            if len(self._entries) > 2:
                self._entries.popitem(last=False)
            return found

    def protection(self, backend, sources, specials=()):
        return self._entry(backend, sources, specials)[1]

    def encode(self, tokenizer, text):
        backend = native_tokenizer(tokenizer)
        if backend is None or backend is not native_tokenizer(self.tokenizer):
            raise PromptEncodingError("prompt preparation belongs to a different tokenizer")
        with self._lock:
            protection = next((entry[1] for entry in self._entries.values()
                               if entry[0] is backend and entry[1].literals == text.control_literals), None)
        if protection is None:
            return LiteralEncoder(backend, text.control_literals).encode_origin(text.origin)
        return protection.encode(backend, text)


def render_hf_prompt(tokenizer, messages, *, renderer=None, **kwargs):
    """Reuse HF's selected native source/environment, with owned output origins.

    Server callers use a single normalized conversation, explicit tools and
    generation mode, tokenize=False. No installed/HF environment is modified.
    """
    if renderer is None:
        renderer = HFPromptRenderer(tokenizer)
    custom = getattr(type(tokenizer), "render_prompt_literals", None)
    if custom is not None:
        result = custom(tokenizer, messages, renderer=renderer, **kwargs)
        object.__setattr__(result, "_renderer", renderer)
        return result
    # Owned custom wrappers must opt in explicitly. __getattr__ may forward a
    # valid HF getter even though its own apply_chat_template is authoritative.
    own_apply = vars(type(tokenizer)).get("apply_chat_template")
    own_get = getattr(type(tokenizer), "get_chat_template", None)
    if own_apply is not None and own_get is None:
        module = type(tokenizer).__module__
        if not module.startswith("mlx_lm.") or vars(tokenizer).get("_chat_template") is not None:
            return _ordinary_render(tokenizer, messages, kwargs)
    pending, seen, receiver = [(tokenizer, 0)], set(), None
    while pending:
        item, depth = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if callable(getattr(type(item), "get_chat_template", None)):
            receiver = item
            break
        if depth < 3:
            for name in ("_inner", "_tokenizer", "tokenizer"):
                child = getattr(item, name, None)
                if child is not None and child is not item:
                    pending.append((child, depth + 1))
    if receiver is None:
        return _ordinary_render(tokenizer, messages, kwargs)
    backend = native_tokenizer(receiver)
    if backend is None:
        raise PromptEncodingError("tokenizer does not expose ordinary literal encoding")
    kwargs = dict(kwargs)
    if kwargs.pop("tokenize", False):
        raise PromptEncodingError("native protected rendering returns text, not tokenizer output")
    source = receiver.get_chat_template(kwargs.pop("chat_template", None), tools=kwargs.get("tools"))
    specials = getattr(receiver, "special_tokens_map", {})
    declared = []
    for value in specials.values():
        declared.extend(value if isinstance(value, (list, tuple)) else [value])
    _, protection, template = renderer._entry(backend, [source], declared, jinja=True)
    values, _ = protection.protect({"messages": messages, "kwargs": kwargs})
    context = trusted_values(specials)
    context.update(values["kwargs"])
    context["messages"] = trust_message_roles(values["messages"])
    trust_policy_strings(context)
    context.setdefault("tools", None)
    context.setdefault("documents", None)
    result = RenderedPrompt(bounded_render(template, context), protection.literals)
    object.__setattr__(result, "_renderer", renderer)
    return result


def repair_media_tokens(prompt, tokenizer, tokens, protector=None):
    if not isinstance(prompt, RenderedPrompt):
        return tokens
    expanded = tokenizer.decode(list(tokens), skip_special_tokens=False)
    original = tokenizer.encode(expanded, add_special_tokens=False).ids
    if list(tokens) != original:
        raise PromptEncodingError("media tokenizer cannot round-trip protected expanded text")
    pattern = _slot_pattern(prompt.literals)
    def counts(value):
        found = {}
        for match in pattern.finditer(value):
            found[match.group()] = found.get(match.group(), 0) + 1
        return found
    if counts(prompt.encoded_text) != counts(expanded):
        raise PromptEncodingError("media tokenizer changed a request literal slot")
    protected = RenderedPrompt(expanded, prompt.literals, control_literals=prompt.control_literals)
    object.__setattr__(protected, "_renderer", prompt._renderer)
    return (protector.encode(tokenizer, protected) if protector is not None
            else prompt._renderer.encode(tokenizer, protected) if prompt._renderer is not None
            else LiteralEncoder(tokenizer, prompt.control_literals).encode_origin(protected.origin))
