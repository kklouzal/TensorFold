"""``--tokenizer``: another tokenizer.json is admitted only when every token ID keeps its meaning."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("change", ["vocab", "merges", "merge_shape", "model_type", "added", "section"])
def test_token_meaning_changes_are_refused(checkpoint, tmp_path, change):
    value = deepcopy(_tokenizer(MARKS))
    if change == "vocab":
        value["model"]["vocab"]["a"], value["model"]["vocab"]["b"] = 1, 0
    elif change == "merges":
        value["model"]["merges"] = []
    elif change == "merge_shape":                                       # named as the override's, not the checkpoint's
        value["model"]["merges"] = [["क", "ि", "a"]]
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


def test_serve_admits_the_override_before_the_engine_loads(checkpoint, tmp_path, capsys):
    """The engine (resume markers), the server and its grammars all read the tokenizer after startup admission."""

    from tensorfold import cli
    from tensorfold.cuda import precision, prompt_precision

    seen = []

    def engine(*args, **kwargs):
        seen.append(token_metadata.tokenizer_override())
        raise RuntimeError("engine acquired")

    override = _write(tmp_path, _tokenizer(MARKS))
    family = SimpleNamespace(title="Test family", model_type="test", package=SimpleNamespace(cuda_engine=engine))
    args = cli.build_parser().parse_args(["serve", str(checkpoint), "--backend", "cuda", "--no-drafts",
                                          "--tokenizer", override])
    try:
        with pytest.raises(RuntimeError, match="engine acquired"):
            cli._serve_cuda(args, family, checkpoint, 8192)
    finally:
        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)
        precision.set_mode(precision.CHECKPOINT)
    assert len(seen) == 1 and seen[0] is not None and seen[0].path == Path(override).resolve()
    assert "differs in: pre_tokenizer" in capsys.readouterr().out


def _byte_level(decoder: dict | None) -> dict:
    value = _tokenizer(MARKS)
    value["added_tokens"][0]["id"] = 5
    value["decoder"] = decoder
    value["model"].update(vocab={ch: i for i, ch in enumerate(["a", "b", "Ġ", "Ġa", "!"])}, merges=[["Ġ", "a"]])
    return value


def test_grammars_use_the_admitted_text_not_the_file_again(checkpoint, tmp_path):
    """Structured-output token tables follow the served decoder (byte-level "Ġa" is " a"), from the bytes admitted at
    startup even when the file changes afterwards."""

    pytest.importorskip("xgrammar")
    pytest.importorskip("transformers")
    from tensorfold.engine.grammar import Grammars

    (checkpoint / "tokenizer.json").write_text(json.dumps(_byte_level(None)))
    assert Grammars.for_model(checkpoint, 8, [5]).info.decoded_vocab[3] == "Ġa".encode()
    decoder = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    override = _write(tmp_path, _byte_level(decoder))
    assert token_metadata.use_tokenizer(override, checkpoint).differs == ("decoder",)
    Path(override).write_text("{}")                                    # admitted once: never read again
    assert Grammars.for_model(checkpoint, 8, [5]).info.decoded_vocab[3] == b" a"
    assert token_metadata.load_tokenizer(checkpoint).decode([3]) == " a"
