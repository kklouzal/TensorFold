"""``--tokenizer``: another tokenizer.json is admitted only when every token ID keeps its meaning."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from tensorfold import token_metadata

tokenizers = pytest.importorskip("tokenizers")

LETTERS = r"[^\r\n\p{L}\p{N}]?[\p{L}]+"              # the AEON export's: combining marks split from their letters
MARKS = r"[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+"      # upstream Qwen3.8's: marks stay inside the word


def _tokenizer(pattern: str) -> dict:
    vocab = {ch: i for i, ch in enumerate(["a", "b", "क", "ि", "कि", " ", "!"])}
    return {"version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [{"id": 7, "content": "<|im_end|>", "single_word": False, "lstrip": False,
                              "rstrip": False, "normalized": False, "special": True}],
            "normalizer": None,
            "pre_tokenizer": {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated",
                              "invert": False},
            "post_processor": None, "decoder": None,
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                      "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                      "vocab": vocab, "merges": [["क", "ि"]]}}


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(token_metadata, "_override", None)
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text(json.dumps(_tokenizer(LETTERS)))
    return model


def _write(tmp_path, value) -> str:
    path = tmp_path / "override.json"
    path.write_text(json.dumps(value))
    return str(path)


def test_text_handling_override_is_used_for_every_encode(checkpoint, tmp_path):
    text = "कि"                                                  # a Devanagari letter and its vowel sign
    before = token_metadata.load_tokenizer(checkpoint).encode(text).ids
    upstream = _tokenizer(MARKS)
    upstream["model"]["merges"] = [" ".join(m) for m in upstream["model"]["merges"]]   # the legacy merge spelling
    got = token_metadata.use_tokenizer(_write(tmp_path, upstream), checkpoint)
    assert got.differs == ("pre_tokenizer",) and "pre_tokenizer" in got.line()
    assert got.sha256 == hashlib.sha256((tmp_path / "override.json").read_bytes()).hexdigest()   # the file as read
    after = token_metadata.load_tokenizer(checkpoint).encode(text).ids
    assert before == [2, 3] and after == [4]                            # the merge applies inside one pre-token
    assert token_metadata.tokenizer_override() is got
    with pytest.raises(RuntimeError, match="once"):
        token_metadata.use_tokenizer(_write(tmp_path, _tokenizer(MARKS)), checkpoint)


@pytest.mark.parametrize("change", ["vocab", "merges", "model_type", "added", "section"])
def test_token_meaning_changes_are_refused(checkpoint, tmp_path, change):
    value = deepcopy(_tokenizer(MARKS))
    if change == "vocab":
        value["model"]["vocab"]["a"], value["model"]["vocab"]["b"] = 1, 0
    elif change == "merges":
        value["model"]["merges"] = []
    elif change == "model_type":
        value["model"]["byte_fallback"] = True
    elif change == "added":
        value["added_tokens"][0]["id"] = 6
    else:
        value["extra_mapping"] = {}
    with pytest.raises(ValueError, match="--tokenizer"):
        token_metadata.use_tokenizer(_write(tmp_path, value), checkpoint)
    assert token_metadata.tokenizer_override() is None
    assert token_metadata.tokenizer_json(checkpoint) == (checkpoint / "tokenizer.json").read_text()
