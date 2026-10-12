"""Chat text encoded a message at a time, each message's tokens cached, equals the text encoded whole."""

import random
import threading

import pytest

tokenizers = pytest.importorskip("tokenizers")

from tensorfold.server.segments import LIMIT, SegmentEncoder, _size  # noqa: E402

OPEN, END = "<|im_start|>", "<|im_end|>"
PIECES = ["hello", " world", "\n", "  ", "\t", "é", "é", "日本語", "🙂", "<think>", "</think>", "<tool_call>",
          '{"a": 1}', END, "\n\n", " ", "x" * 40, "def f():\n    return 1\n", "İ", "\r\n", "1234567"]


def bpe():
    """A byte-level BPE with an NFC normalizer (Qwen's shape), trained here, and Qwen's added tokens."""

    tok = tokenizers.Tokenizer(tokenizers.models.BPE())
    tok.normalizer = tokenizers.normalizers.NFC()
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    rng = random.Random(1)
    corpus = ["".join(rng.choice(PIECES) for _ in range(30)) for _ in range(300)]
    tok.train_from_iterator(corpus, tokenizers.trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet()))
    tok.add_special_tokens([OPEN, END])
    tok.add_tokens(["<think>", "</think>", "<tool_call>"])
    return tok


def chat(rng):
    text = "".join(f"{OPEN}{rng.choice(['system', 'user', 'assistant', 'tool'])}\n"
                   + "".join(rng.choice(PIECES) for _ in range(rng.randrange(0, 30))) + f"{END}\n"
                   for _ in range(rng.randrange(1, 10)))
    return (rng.choice(PIECES) if rng.random() < 0.2 else "") + text + (f"{OPEN}assistant\n" if rng.random() < .7 else "")


@pytest.mark.parametrize("limit", [30, 20_000, LIMIT])                # bytes: one entry, a few, all of them
def test_segments_equal_the_whole_text_with_any_cache_bound(limit):
    tok, rng = bpe(), random.Random(7)
    enc = SegmentEncoder(tok, OPEN, limit)
    convo = ""
    for turn in range(300):                                  # a growing conversation, sometimes edited
        convo = convo + chat(rng) if rng.random() < .8 else convo[: len(convo) // 2]
        assert enc.encode(convo) == tok.encode(convo, add_special_tokens=False).ids, turn
        other = chat(rng)
        assert enc.encode(other) == tok.encode(other, add_special_tokens=False).ids, turn
    assert enc.held == sum(_size(text, ids) for text, ids in enc.cache.items())
    assert enc.held <= max(limit, max(_size(text, ids) for text, ids in enc.cache.items()))


def test_threads_share_the_cache_and_each_gets_its_own_text():
    tok, rng = bpe(), random.Random(3)
    enc, texts, bad = SegmentEncoder(tok, OPEN, 500), [chat(rng) * 3 for _ in range(40)], []

    def work(offset):
        for i in range(200):
            text = texts[(offset + i) % len(texts)]
            if enc.encode(text) != tok.encode(text, add_special_tokens=False).ids:
                bad.append(text)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert bad == []


def test_an_opener_the_tokenizer_does_not_split_on_exactly_is_refused():
    tok = bpe()
    tok.add_special_tokens([tokenizers.AddedToken("<|lstrip|>", lstrip=True)])
    assert SegmentEncoder.qualifies(tok, OPEN)
    for opener in ("<|lstrip|>", "<|missing|>", "<think>"):   # <think> is added but normalized
        if not SegmentEncoder.qualifies(tok, opener):
            with pytest.raises(ValueError):
                SegmentEncoder(tok, opener)
    assert not SegmentEncoder.qualifies(tok, "<|lstrip|>") and not SegmentEncoder.qualifies(tok, "<|missing|>")


@pytest.mark.parametrize("change", ["ends with its prefix", "holds it", "special tokens encoded", "truncation",
                                    "padding"])
def test_a_tokenizer_whose_encoding_could_cross_the_opener_is_refused(change):
    tok = bpe()
    assert SegmentEncoder.qualifies(tok, OPEN)
    text = f"{OPEN}user\nxy{OPEN}assistant\n"
    if change == "ends with its prefix":                     # leftmost-longest: 'xy<|im' wins over the opener
        tok.add_tokens([tokenizers.AddedToken("xy<|im", normalized=False)])
    elif change == "holds it":
        tok.add_tokens([tokenizers.AddedToken(f"y{OPEN}", normalized=False)])
    elif change == "special tokens encoded":
        tok.encode_special_tokens = True
    elif change == "truncation":
        tok.enable_truncation(8)
    else:
        tok.enable_padding(length=64)
    assert not SegmentEncoder.qualifies(tok, OPEN)
    with pytest.raises(ValueError):
        SegmentEncoder(tok, OPEN)
    if change in ("ends with its prefix", "holds it"):     # the reason: split, the encoding differs from the whole one
        parts = [tok.encode(part, add_special_tokens=False).ids for part in (f"{OPEN}user\nxy", f"{OPEN}assistant\n")]
        assert parts[0] + parts[1] != tok.encode(text, add_special_tokens=False).ids


def test_a_qwen_template_gets_a_message_encoder_that_equals_the_whole_encode(tmp_path):
    pytest.importorskip("jinja2")
    from tensorfold.cuda.server import ChatTemplate, _segment_encoder
    from tests.test_cuda_markers import TEMPLATE

    tok = bpe()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text(TEMPLATE.replace("KEPT", "true"))
    template = ChatTemplate(tmp_path)
    enc = _segment_encoder(tok, template)
    assert enc is not None and enc.opener == OPEN
    rng = random.Random(5)
    talk = [{"role": "system", "content": "".join(rng.choice(PIECES) for _ in range(40))}]
    for turn in range(12):
        talk.append({"role": ("user", "assistant")[turn % 2], "content": "".join(rng.choice(PIECES) for _ in range(30))})
        text = template.render(talk, tools=None, enable_thinking=True)
        assert enc.encode(text) == tok.encode(text, add_special_tokens=False).ids, turn
