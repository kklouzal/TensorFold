"""Chat text encoded a message at a time, each message's tokens cached, equals the text encoded whole."""

import random
import threading
import weakref
from collections import OrderedDict

import pytest

tokenizers = pytest.importorskip("tokenizers")

from tensorfold.server.segments import LIMIT, SegmentEncoder, _size, bind_ids_encoder  # noqa: E402

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


def small_encoder(limit=LIMIT):
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(vocab={"a": 0, "b": 1, OPEN: 2}, merges=[]))
    tok.add_tokens([tokenizers.AddedToken(OPEN, normalized=False)])
    return SegmentEncoder(tok, OPEN, limit)


@pytest.mark.parametrize("cut", ["size", "counter", "map-before", "map-after"])
def test_cache_publication_exhaustion_clears_charge_and_preserves_primary(monkeypatch, cut):
    from tensorfold.server import segments

    enc = small_encoder()
    enc.encode(OPEN + "a" * 512)
    new, primary = OPEN + "b" * 512, MemoryError("controlled cache publication failure")
    with monkeypatch.context() as patch:
        if cut == "size":
            def size(text, ids):
                if text == new:
                    assert text not in enc.cache
                    raise primary
                return _size(text, ids)
            patch.setattr(segments, "_size", size)
        elif cut == "counter":
            class Counter(int):
                def __add__(self, other):
                    assert new not in enc.cache
                    raise primary
            enc.held = Counter(enc.held)
        else:
            class Publishing(OrderedDict):
                def __setitem__(self, key, value):
                    if key == new:
                        if cut == "map-after":
                            super().__setitem__(key, value)
                        raise primary
                    super().__setitem__(key, value)
            enc.cache = Publishing(enc.cache)
        with pytest.raises(MemoryError) as caught:
            enc.encode(new)
        assert caught.value is primary and not enc.cache and enc.held == 0
    recovered = OPEN + "ab"
    assert enc.encode(recovered) == enc.tok.encode(recovered, add_special_tokens=False).ids
    assert enc.held == sum(_size(text, ids) for text, ids in enc.cache.items()) > 0


@pytest.mark.parametrize("cut", ["size", "counter", "map-before", "map-after"])
def test_cache_eviction_exhaustion_cannot_leave_an_invalid_charge(monkeypatch, cut):
    from tensorfold.server import segments

    enc = small_encoder(limit=0)
    old, new = OPEN + "a" * 512, OPEN + "b" * 512
    enc.encode(old)  # The selected newest-entry policy permits this oversized entry.
    primary = MemoryError("controlled cache eviction failure")
    with monkeypatch.context() as patch:
        if cut in ("size", "counter"):
            class Size(int):
                def __rsub__(self, other):
                    raise primary
            def size(text, ids):
                if text == old:
                    assert text not in enc.cache
                    if cut == "size":
                        raise primary
                    return Size(_size(text, ids))
                return _size(text, ids)
            patch.setattr(segments, "_size", size)
        else:
            class Evicting(OrderedDict):
                def popitem(self, last=True):
                    if cut == "map-after":
                        super().popitem(last=last)
                    raise primary
            enc.cache = Evicting(enc.cache)
        with pytest.raises(MemoryError) as caught:
            enc.encode(new)
        assert caught.value is primary and not enc.cache and enc.held == 0
    assert enc.encode(new) == enc.tok.encode(new, add_special_tokens=False).ids
    assert list(enc.cache) == [new] and enc.held == _size(new, enc.cache[new]) > enc.limit


def test_batch_deduplication_hit_order_and_newest_oversized_entry_are_preserved():
    enc = small_encoder()
    native, batches = enc.tok, []
    class Batch:
        def encode_batch(self, missing, **options):
            batches.append(list(missing))
            return native.encode_batch(missing, **options)
    enc.tok = Batch()
    first, second, newest = OPEN + "a", OPEN + "b", OPEN + "ab" * 512
    text = first + second + first
    assert enc.encode(text) == native.encode(text, add_special_tokens=False).ids
    assert batches == [[first, second]] and list(enc.cache) == [first, second]
    assert enc.encode(first) == native.encode(first, add_special_tokens=False).ids
    assert batches[-1] == [] and list(enc.cache) == [second, first]
    enc.limit = 0
    assert enc.encode(newest) == native.encode(newest, add_special_tokens=False).ids
    assert list(enc.cache) == [newest] and enc.held == _size(newest, enc.cache[newest]) > enc.limit
    assert enc.encode(newest) == native.encode(newest, add_special_tokens=False).ids
    assert batches[-1] == [] and list(enc.cache) == [newest]


def test_containment_does_not_retire_another_callers_borrowed_hit(monkeypatch):
    from tensorfold.server import segments

    enc = small_encoder()
    old, new = OPEN + "a" * 512, OPEN + "b" * 512
    expected = enc.encode(old)
    owner = weakref.ref(enc.cache[old])
    native, entered, released, results, errors = enc.tok, threading.Event(), threading.Event(), [], []
    class Batch:
        def encode_batch(self, missing, **options):
            if not missing:
                entered.set()
                assert released.wait(3)
            return native.encode_batch(missing, **options)
    enc.tok = Batch()
    def read():
        try:
            results.append(enc.encode(old))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=read)
    thread.start()
    primary = MemoryError("controlled competing publication failure")
    def size(text, ids):
        if text == new:
            raise primary
        return _size(text, ids)
    try:
        assert entered.wait(3)
        monkeypatch.setattr(segments, "_size", size)
        with pytest.raises(MemoryError) as caught:
            enc.encode(new)
        assert caught.value is primary and not enc.cache and enc.held == 0
        assert owner() is not None  # The outside-lock reader still owns the array it borrowed.
    finally:
        released.set()
        thread.join(3)
    assert not thread.is_alive() and errors == [] and results == [expected]
    assert owner() is None


def test_pinned_native_misses_use_public_fast_ids_and_preserve_order_and_flags():
    enc = small_encoder()
    native, fast, calls = enc.tok, enc.tok.encode_batch_fast, []
    def selected(parts, **options):
        calls.append((list(parts), options))
        return fast(parts, **options)
    native.encode_batch_fast = selected
    native.encode_batch = lambda *a, **k: pytest.fail("ordinary batch called after native fast selection")
    enc = SegmentEncoder(native, OPEN)
    first, second = OPEN + "a", OPEN + "b"
    assert enc.batch_codec == "encode_batch_fast"
    assert enc.encode(first + second + first) == native.encode(first + second + first, add_special_tokens=False).ids
    assert calls == [([first, second], {"add_special_tokens": False})]
    assert enc.encode(first) == native.encode(first, add_special_tokens=False).ids
    assert calls[-1] == ([], {"add_special_tokens": False})


def test_fast_failure_is_not_retried_as_ordinary_or_published():
    native = small_encoder().tok
    primary = MemoryError("controlled fast API failure")
    def selected(*args, **kwargs):
        raise primary
    native.encode_batch_fast = selected
    native.encode_batch = lambda *a, **k: pytest.fail("fast failure triggered an ordinary retry")
    enc = SegmentEncoder(native, OPEN)
    with pytest.raises(MemoryError) as caught:
        enc.encode(OPEN + "ab")
    assert caught.value is primary and not enc.cache and enc.held == 0


def test_custom_provider_keeps_ordinary_protocol_without_probing_fast_property():
    native, calls = small_encoder().tok, []
    class Custom:
        def __getattr__(self, name):
            return getattr(native, name)
        @property
        def encode_batch_fast(self):
            pytest.fail("a custom fast capability was inferred")
        def encode_batch(self, parts, **options):
            calls.append((list(parts), options))
            return native.encode_batch(parts, **options)
        def encode(self, text, **options):
            calls.append((text, options))
            return native.encode(text, **options)
    custom = Custom()
    enc = SegmentEncoder(custom, OPEN)
    assert enc.batch_codec == "encode_batch"
    assert enc.encode(OPEN + "ab") == native.encode(OPEN + "ab", add_special_tokens=False).ids
    ids = bind_ids_encoder(custom)
    assert ids.codec == "encode" and ids.tokenizer_identity is None
    assert ids("ab", add_special_tokens=True) == native.encode("ab", add_special_tokens=True).ids
    assert calls == [([OPEN + "ab"], {"add_special_tokens": False}), ("ab", {"add_special_tokens": True})]


@pytest.mark.parametrize("fallback", ["older", "dropout", "other_model"])
def test_unqualified_or_stochastic_native_tokenizers_keep_ordinary_call(monkeypatch, fallback):
    native = small_encoder().tok
    if fallback == "older":
        monkeypatch.setattr(tokenizers, "__version__", "0.23.2")
    elif fallback == "dropout":
        native.model.dropout = 0.1
    else:
        native.model = tokenizers.models.WordLevel({"a": 0, "[UNK]": 1}, unk_token="[UNK]")
    native.encode_batch_fast = lambda *a, **k: pytest.fail("unqualified native fast API selected")
    enc = SegmentEncoder(native, OPEN)
    assert enc.batch_codec == "encode_batch"
    enc.encode(OPEN + "a")
    assert bind_ids_encoder(native).codec == "encode"


def test_ids_callback_is_bound_once_with_identity_and_preserves_special_token_flags(monkeypatch):
    from tensorfold import token_metadata

    native = small_encoder().tok
    native.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A " + OPEN, special_tokens=[(OPEN, native.token_to_id(OPEN))])
    ordinary, original_identity, calls = native.encode, token_metadata.tokenizer_identity, []
    def identity(tok):
        calls.append(tok)
        return original_identity(tok)
    monkeypatch.setattr(token_metadata, "tokenizer_identity", identity)
    ids = bind_ids_encoder(native)
    assert ids.codec == "encode_batch_fast" and calls == [native]
    assert ids.tokenizer_identity == original_identity(native)
    native.encode = lambda *a, **k: pytest.fail("profile-bound callback reselected ordinary encode")
    for special in (False, True):
        for text in ("a", "b", "ab", "", "é", "e\u0301"):
            assert ids(text, add_special_tokens=special) == ordinary(text, add_special_tokens=special).ids
    assert calls == [native]


@pytest.mark.parametrize("inputs", [[None], ["\ud800"]])
def test_public_fast_and_ordinary_batch_errors_keep_their_type_and_message(inputs):
    native = small_encoder().tok
    outcomes = []
    for method in (native.encode_batch, native.encode_batch_fast):
        with pytest.raises(BaseException) as caught:
            method(inputs, add_special_tokens=False)
        outcomes.append((type(caught.value), caught.value.args))
    assert outcomes[0] == outcomes[1]
