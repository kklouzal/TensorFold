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


@pytest.mark.parametrize("limit", [30, 20_000, LIMIT])                # bytes: none, a few, all of them
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
    assert enc.held <= limit


def test_oversized_segment_is_encoded_but_not_retained():
    tok = bpe()
    encoder = SegmentEncoder(tok, OPEN, 64)
    text = OPEN + "user\n" + "hello world " * 1000 + END
    assert encoder.encode(text) == tok.encode(text, add_special_tokens=False).ids
    assert encoder.held <= 64 and not encoder.cache


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


def test_credits_precede_split_cache_lock_batch_and_python_getters():
    tok, calls, cache_calls = bpe(), [], []
    class Encoding:
        def __init__(self, value):
            self.value = value
        def __len__(self):
            return len(self.value)
        @property
        def ids(self):
            assert calls[-1][0] >= len(self.value)
            return self.value.ids
    class Guarded:
        def __getattr__(self, name):
            return getattr(tok, name)
        def encode_batch(self, missing, **kwargs):
            assert cache_calls and calls[-1][1]['entries'] >= len(missing)
            return [Encoding(value) for value in tok.encode_batch(missing, **kwargs)]
    encoder = SegmentEncoder(Guarded(), OPEN, limit=4096)
    text = f'{OPEN}user\né🙂{END}{OPEN}assistant\nanswer{END}'
    def cache_credit(limit, entries, chars):
        assert not encoder.lock.locked()
        cache_calls.append((limit, entries, chars))
    actual = encoder.encode(text, ids_credit=lambda n, **kw: calls.append((n, kw)), cache_credit=cache_credit)
    assert actual == tok.encode(text, add_special_tokens=False).ids
    assert cache_calls == [(4096, text.count(OPEN) + 1, len(text))]
    assert max(n for n, kw in calls) >= len(actual)
    # Cached arrays still require payment before constructing the output list.
    calls.clear()
    assert encoder.encode(text, ids_credit=lambda n, **kw: calls.append((n, kw)), cache_credit=cache_credit) == actual
    assert max(n for n, kw in calls) >= len(actual)


def test_fresh_segment_credits_cover_cumulative_arrays_not_only_largest_segment():
    tok, records = bpe(), []
    encoder = SegmentEncoder(tok, OPEN, limit=0)
    text = ''.join(f'{OPEN}user\n{value}{END}' for value in ['é🙂', 'long ' * 17, 'third 日本語'])
    result = encoder.encode(text, ids_credit=lambda n, **kw: records.append((n, kw)))
    assert result == tok.encode(text, add_special_tokens=False).ids
    counts = [len(tok.encode(OPEN + part, add_special_tokens=False)) for part in text.split(OPEN)[1:]]
    assert records[-1][0] >= 2 * sum(counts)
    assert not encoder.cache


def test_cache_credit_failure_happens_before_cache_or_native_batch():
    encoder = SegmentEncoder(bpe(), OPEN)
    primary = RuntimeError('cache pins denied')
    class Forbidden:
        def __enter__(self):
            pytest.fail('cache lock entered before credit')
        def __exit__(self, *args):
            pass
    encoder.lock = Forbidden()
    def denied(*args):
        raise primary
    with pytest.raises(RuntimeError) as caught:
        encoder.encode(OPEN + 'text', cache_credit=denied)
    assert caught.value is primary


def test_concurrent_publication_keeps_old_lock_entry_eligibility_and_lru_suffix():
    """Preserve the insert-all/trim semantics; naive per-insert eviction fails here."""
    from array import array
    from tensorfold.server.segments import _Segment
    tok = bpe()
    a, x, y = (OPEN + letter for letter in 'AXY')
    def entry(text):
        return _Segment(array('i', tok.encode(text, add_special_tokens=False).ids), array('i'))
    ea, ey = entry(a), entry(y)
    assert _size(a, ea) == _size(y, ey)
    class Concurrent:
        def __getattr__(self, name):
            return getattr(tok, name)
        def encode_batch(self, missing, **kwargs):
            assert missing == [x, y]
            with encoder.lock:
                encoder.cache[y] = ey
                encoder.cache.move_to_end(a)
                encoder.held += _size(y, ey)
            return tok.encode_batch(missing, **kwargs)
    encoder = SegmentEncoder(Concurrent(), OPEN, limit=2 * _size(a, ea))
    encoder.cache[a], encoder.held = ea, _size(a, ea)
    text = x + y
    assert encoder.encode(text, ids_credit=lambda n, **kw: None, cache_credit=lambda *args: None) == tok.encode(text, add_special_tokens=False).ids
    assert list(encoder.cache) == [a, x]
    assert encoder.held <= encoder.limit
