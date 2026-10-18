"""Model-owned reply-stop metadata, independent from training/segmentation IDs.

Resolve once during preparation. Only the tokenizer's named EOS role adds a
token stop; other special tokens have no implied end-of-reply meaning.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from tensorfold.checkpoint_metadata import read_json, read_text

# tokenizer.json sections that only shape text <-> pieces; the rest (the vocabulary model and the added tokens) fixes
# what every token ID means and must equal the checkpoint's.
TEXT_SECTIONS = ("normalizer", "pre_tokenizer", "post_processor", "decoder", "truncation", "padding", "version")


@dataclass(frozen=True)
class TokenizerOverride:
    """``--tokenizer``: a tokenizer.json read in place of the checkpoint's own (same token IDs, other text handling)."""

    path: Path
    text: str
    sha256: str
    differs: tuple[str, ...]          # TEXT_SECTIONS that differ from the checkpoint's

    def line(self) -> str:
        what = ", ".join(self.differs) if self.differs else "nothing"
        return (f"[tensorfold] tokenizer: {self.path} (sha256 {self.sha256[:12]}) in place of the checkpoint's "
                f"tokenizer.json; same vocabulary, merges and added tokens, differs in: {what}")


_override: TokenizerOverride | None = None     # set once at startup (``use_tokenizer``); one process serves one model


def _meaning(model: dict | None) -> dict | None:
    """The vocabulary model with BPE merges in one spelling: tokenizers writes a merge as "a b" (legacy) or [a, b]."""

    if not isinstance(model, dict) or not isinstance(model.get("merges"), list):
        return model
    merges = []
    for merge in model["merges"]:
        pair = merge.split(" ") if isinstance(merge, str) else merge
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(t, str) for t in pair):
            raise ValueError("tokenizer.json BPE merges must be token pairs")
        merges.append(tuple(pair))
    return dict(model, merges=merges)


def use_tokenizer(path, model_dir) -> TokenizerOverride:
    """Encode and decode with ``path`` (a tokenizer.json) instead of the checkpoint's, from now on in this process.

    Token IDs must keep the meaning the checkpoint was trained with: the vocabulary model (its type, vocabulary,
    merges in either spelling, and options) and the added tokens must equal the checkpoint's own tokenizer.json, so
    only the text handling in ``TEXT_SECTIONS`` may differ. Call once at startup, before any tokenizer, template or grammar reads it.
    """
    global _override

    if _override is not None:
        raise RuntimeError("the tokenizer override is set once, at startup")
    from tensorfold.cuda.tensor_file import _metadata_object

    path = Path(path).expanduser().resolve()
    text = read_text(path.parent, path.name, required=True)   # one bounded regular read, as for --chat-template FILE
    raw = text.encode("utf-8")
    value = _metadata_object(raw, path)
    own = read_json(model_dir, "tokenizer.json", required=True)
    for key, same in (("model", _meaning(value.get("model")) == _meaning(own.get("model"))),
                      ("added_tokens", value.get("added_tokens") == own.get("added_tokens"))):
        if not same:
            raise ValueError(f"--tokenizer {path}: its {key} differs from the checkpoint's tokenizer.json, so token "
                             "IDs would change meaning; only text handling (" + ", ".join(TEXT_SECTIONS[:4]) +
                             ") may differ")
    unknown = sorted(set(value) - set(own) - set(TEXT_SECTIONS))
    if unknown:
        raise ValueError(f"--tokenizer {path}: unknown tokenizer.json sections {unknown}")
    differs = tuple(key for key in TEXT_SECTIONS if value.get(key) != own.get(key))
    _override = TokenizerOverride(path, text, hashlib.sha256(raw).hexdigest(), differs)
    return _override


def tokenizer_override() -> TokenizerOverride | None:
    return _override


def tokenizer_json(model_dir) -> str:
    """The tokenizer.json text this process encodes with: the ``--tokenizer`` override, else the checkpoint's."""

    return _override.text if _override is not None else read_text(model_dir, "tokenizer.json", required=True)


def load_tokenizer(model_dir):
    """Prepare exact CUDA sequences, leaving length admission to the server.

    Exported padding/truncation are batch preparation settings, not the model's
    tokenization. Keep its vocabulary, added tokens and postprocessor unchanged;
    neither counting nor generation may pad or silently discard prompt tokens.
    This private instance is configured once before concurrent encoding starts.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(tokenizer_json(model_dir))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def token_ids(value, *, field="eos_token_id", vocab_size=None, required=False):
    """Validate native token IDs without accepting booleans, coercions or empty overrides."""
    if vocab_size is not None and (type(vocab_size) is not int or vocab_size <= 0):
        raise ValueError("vocabulary size must be a positive native integer")
    values = () if value is None else value if type(value) in (list, tuple) else (value,)
    if any(type(item) is not int or item < 0 or (vocab_size is not None and item >= vocab_size)
           for item in values):
        raise ValueError(f"{field} must contain nonnegative native integer IDs within the model vocabulary")
    out = tuple(dict.fromkeys(values))
    if required and not out:
        raise ValueError(f"{field} requires at least one token ID")
    return out


def _content(value, field, *, required=False, allow_empty=False):
    if value is None and not required:
        return None
    if isinstance(value, dict):
        value = value.get("content")
    if type(value) is not str or (not value and not allow_empty):
        expected = "a token string" if allow_empty else "a nonempty token string"
        raise ValueError(f"{field} must be {expected} or AddedToken content")
    return value


def resolve_token_roles(model_dir, *, config=None):
    """Resolve existing named roles with HF's modern/legacy loading precedence.

    An added_tokens_decoder key selects modern tokenizer_config roles, even for
    an empty decoder. Otherwise legacy special_tokens_map roles replace those
    declarations, including an explicit null. Unused legacy metadata is not read.
    """
    config = (read_json(model_dir, "tokenizer_config.json") or {}) if config is None else config
    keys = ("bos_token", "eos_token", "pad_token", "unk_token")
    roles = {key: config[key] for key in keys if key in config}
    if "added_tokens_decoder" in config:
        if not isinstance(config["added_tokens_decoder"], dict):
            raise ValueError("added_tokens_decoder must be an object")
    else:
        legacy = read_json(model_dir, "special_tokens_map.json") or {}
        roles.update((key, legacy[key]) for key in keys if key in legacy)
    return {key: content for key, value in roles.items()
            if (content := _content(value, f"tokenizer {key}", allow_empty=key != "eos_token")) is not None}


def _lookup_eos(model_dir, config, token, vocab_size):
    """Resolve exact token content, including added tokens; never encode to an unknown-token fallback."""
    tokenizer = read_json(model_dir, "tokenizer.json")
    actual = []
    if tokenizer is not None:
        added = tokenizer.get("added_tokens", [])
        if not isinstance(added, list):
            raise ValueError("tokenizer.json added_tokens must be a list")
        for entry in added:
            if not isinstance(entry, dict):
                raise ValueError("tokenizer.json added tokens must be objects")  # noqa: TRY004 - external JSON schema
            if entry.get("content") == token:
                actual.extend(token_ids(entry.get("id"), field="tokenizer EOS added-token ID",
                                        vocab_size=vocab_size, required=True))
        model = tokenizer.get("model")
        model = {} if model is None else model
        if not isinstance(model, dict):
            raise ValueError("tokenizer.json model must be an object")
        vocab = model.get("vocab", {})
        if isinstance(vocab, dict):
            if token in vocab:
                actual.extend(token_ids(vocab[token], field="tokenizer EOS vocabulary ID",
                                        vocab_size=vocab_size, required=True))
        elif isinstance(vocab, list):  # Unigram stores [token, score] entries in ID order.
            for index, entry in enumerate(vocab):
                if not isinstance(entry, list) or len(entry) != 2 or type(entry[0]) is not str:
                    raise ValueError("tokenizer Unigram vocabulary must contain token/score pairs")
                if entry[0] == token:
                    actual.extend(token_ids(index, field="tokenizer EOS vocabulary ID", vocab_size=vocab_size))
        else:
            raise ValueError("tokenizer vocabulary must be a mapping or Unigram token/score list")
    declared = []
    decoder = config.get("added_tokens_decoder")
    decoder = {} if decoder is None else decoder
    if not isinstance(decoder, dict):
        raise ValueError("added_tokens_decoder must be an object")  # noqa: TRY004 - external JSON schema
    for key, entry in decoder.items():
        if _content(entry, "added_tokens_decoder content", required=True) == token:
            try:
                index = int(key)
            except (ValueError, TypeError):
                raise ValueError("added_tokens_decoder EOS key must be an integer ID") from None
            declared.extend(token_ids(index, field="declared tokenizer EOS ID", vocab_size=vocab_size))
    actual, declared = tuple(dict.fromkeys(actual)), tuple(dict.fromkeys(declared))
    if len(actual) > 1 or len(declared) > 1 or (actual and declared and actual != declared):
        raise ValueError("tokenizer EOS content has conflicting token IDs")
    if tokenizer is not None and not actual:
        raise ValueError("declared tokenizer EOS content is absent from tokenizer.json")
    found = actual or declared
    if not found:
        raise ValueError("declared tokenizer EOS content has no authoritative token ID")
    return found


def resolve_eos(model_dir, configured=None, *, vocab_size=None, override=None, fallback=None, tokenizer=None):
    """Stable model, generation and named tokenizer EOS union, or explicit replacement.

    ``configured`` is the caller's authoritative model EOS; an optional family
    fallback applies only when no metadata declares an EOS. ``override=None`` is
    omitted, while any explicit override must be nonempty and valid. Training
    and n-gram segmentation EOS are intentionally outside this reply contract.
    """
    if override is not None:
        return token_ids(override, field="eos_token_ids override", vocab_size=vocab_size, required=True)
    found = list(token_ids(configured, field="model eos_token_id", vocab_size=vocab_size))
    generation = read_json(model_dir, "generation_config.json") or {}
    found.extend(token_ids(generation.get("eos_token_id"), field="generation eos_token_id", vocab_size=vocab_size))
    config = read_json(model_dir, "tokenizer_config.json") or {}
    token = resolve_token_roles(model_dir, config=config).get("eos_token")
    if token is not None:
        if tokenizer is None:
            found.extend(_lookup_eos(model_dir, config, token, vocab_size))
        else:
            vocab = tokenizer.get_vocab()
            if token not in vocab:
                raise ValueError("declared tokenizer EOS content is absent from the loaded tokenizer")
            found.extend(token_ids(vocab[token], field="loaded tokenizer EOS ID", vocab_size=vocab_size, required=True))
    if not found and fallback is not None:
        found.extend(token_ids(fallback, field="family default eos_token_id", vocab_size=vocab_size))
    return token_ids(found, vocab_size=vocab_size, required=True)
