"""The real GLM template adapter preserves bounded native prompt authority."""

import json

import pytest
from jinja2.exceptions import TemplateError
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.families.glm5_next import prompts as glm_prompts
from tensorfold.families.glm5_next.cuda import app as glm
from tensorfold.families.glm5_next.prompts import EFFORT_LINE, OPENED, GlmTokenizer
from tensorfold.server.prompt_literals import HFPromptRenderer, RenderedPrompt, encode_hf_prompt, render_hf_prompt
from tensorfold.server.segments import SegmentEncoder

PREFIX = "[gMASK]<sop>"
SOURCE = (PREFIX + EFFORT_LINE
          + "{% for m in messages %}<|user|>{{ m.content }}{% endfor %}"
          + "{% if add_generation_prompt %}" + OPENED + "{% endif %}")
SPECIALS = ["<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<think>", "</think>"]


@pytest.fixture
def checkpoint(tmp_path):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["ordinary user assistant thinking text café"],
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=SPECIALS),
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    def prepare(source=SOURCE):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
        inner = ChatTemplate(tmp_path)
        return tokenizer, inner, glm.ThinkingOffTemplate(inner, clear=False)

    return prepare


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("generation", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
def test_native_ids_keep_literal_controls_and_count_generation_parity(checkpoint, thinking, generation, segmented):
    tokenizer, inner, wrapper = checkpoint()
    literal = "quoted " + EFFORT_LINE + " and " + OPENED + " and </think> café"
    render_calls, ids_calls, cache_calls = [], [], []
    text = wrapper.render([{"role": "user", "content": literal}], tools=None, enable_thinking=thinking,
                          add_generation_prompt=generation,
                          render_credit=lambda *args: render_calls.append(args))
    expected = PREFIX + (EFFORT_LINE if thinking else "") + "<|user|>" + literal
    if generation:
        expected += OPENED + ("" if thinking else "</think>")
    assert isinstance(text, RenderedPrompt) and str(text) == expected
    assert text.control_literals == inner.literals.literals
    at = str(text).index(literal)
    assert text.origin._mask[at:at + len(literal)] == b"\1" * len(literal)
    if generation:
        assert not any(text.origin._mask[-len(OPENED + ("" if thinking else "</think>")):])
    cache = SegmentEncoder(tokenizer, "<|user|>", limit=4096) if segmented else None
    options = {"segments": cache, "ids_credit": lambda n, **kw: ids_calls.append((n, kw)),
               "cache_credit": lambda *args: cache_calls.append(args)}
    counted = wrapper.encode(tokenizer, text, **options)
    generated = wrapper.encode(tokenizer, text, **options)
    assert counted == generated
    assert tokenizer.decode(counted, skip_special_tokens=False) == expected
    controls = [tokenizer.token_to_id("<|system|>")] if thinking else []
    controls.append(tokenizer.token_to_id("<|user|>"))
    if generation:
        controls.extend(tokenizer.token_to_id(token) for token in ("<|assistant|>", "<think>"))
        if not thinking:
            controls.append(tokenizer.token_to_id("</think>"))
    reserved = {tokenizer.token_to_id(token) for token in SPECIALS}
    assert [token for token in counted if token in reserved] == controls
    assert render_calls and ids_calls
    assert bool(cache_calls) is segmented


@pytest.mark.parametrize("source", ["{{ messages[0].content }}", "{{ messages[0].content }}" + EFFORT_LINE])
def test_request_effort_and_opened_markers_are_never_transformed(checkpoint, source):
    tokenizer, _inner, wrapper = checkpoint(source)
    literal = EFFORT_LINE + " quoted " + OPENED
    text = wrapper.render([{"role": "user", "content": literal}], tools=None, enable_thinking=False,
                          add_generation_prompt=False)
    assert str(text) == literal
    assert text.origin._mask == b"\1" * len(literal)
    ids = wrapper.encode(tokenizer, text)
    assert tokenizer.decode(ids, skip_special_tokens=False) == literal
    reserved = {tokenizer.token_to_id(token) for token in SPECIALS}
    assert not reserved.intersection(ids)


def test_current_render_options_and_encode_credits_reach_the_same_inner_owner(checkpoint, monkeypatch):
    tokenizer, inner, wrapper = checkpoint("{{ clear_thinking }}{% for m in messages %}{{ m.content }}{% endfor %}"
                                           "{% if add_generation_prompt %}" + OPENED + "{% endif %}")
    renders, encodes = [], []
    original_render, original_encode = inner.render, inner.encode

    def render(*args, **kwargs):
        renders.append(kwargs)
        return original_render(*args, **kwargs)

    def encode(*args, **kwargs):
        value = original_encode(*args, **kwargs)
        encodes.append((args, kwargs, value))
        return value

    monkeypatch.setattr(inner, "render", render)
    monkeypatch.setattr(inner, "encode", encode)
    extra, render_credit = {"clear_thinking": True}, lambda *args: None
    text = wrapper.render([{"role": "user", "content": "hello"}], tools=[], enable_thinking=True,
                          extra=extra, allow_images=True, allow_audio=True, max_render_bytes=4096,
                          add_generation_prompt=False, render_credit=render_credit)
    assert str(text) == "Truehello" and extra == {"clear_thinking": True}
    assert renders == [{"tools": [], "enable_thinking": True, "extra": extra,
                        "allow_images": True, "allow_audio": True, "max_render_bytes": 4096,
                        "add_generation_prompt": False, "render_credit": render_credit}]
    cache = SegmentEncoder(tokenizer, "<|user|>", limit=4096)
    ids_credit, cache_credit = lambda *args, **kwargs: None, lambda *args: None
    ids = wrapper.encode(tokenizer, text, segments=cache, ids_credit=ids_credit, cache_credit=cache_credit)
    assert encodes[0][0] == (tokenizer, text) and ids is encodes[0][2]
    assert encodes[0][1] == {"segments": cache, "ids_credit": ids_credit, "cache_credit": cache_credit}
    for tools in (False, True):
        assert wrapper.selected(tools=tools) is inner.selected(tools=tools)
        assert wrapper.provenance(tools=tools) == inner.provenance(tools=tools)


def test_transformation_retains_renderer_authority_and_invalidates_old_preencoding(checkpoint, monkeypatch):
    _tokenizer, inner, wrapper = checkpoint()
    original = inner.render
    renderer, ordinary = object(), object()

    def render(*args, **kwargs):
        text = original(*args, **kwargs)
        object.__setattr__(text, "_renderer", renderer)
        object.__setattr__(text, "_ordinary_codec", ordinary)
        object.__setattr__(text, "_preencoded", (999,))
        return text

    monkeypatch.setattr(inner, "render", render)
    text = wrapper.render([{"role": "user", "content": "hello"}], tools=None, enable_thinking=False)
    assert text._renderer is renderer and text._ordinary_codec is ordinary
    assert text._preencoded is None and text.control_literals == inner.literals.literals


def test_final_thinking_off_suffix_obeys_utf8_render_ceiling(checkpoint):
    _tokenizer, inner, wrapper = checkpoint(" " * 128 + "{{ messages[0].content }}" + OPENED)
    messages = [{"role": "user", "content": "é" * 128}]
    original = inner.render(messages, tools=None, enable_thinking=False, extra={"clear_thinking": False})
    needed = len(str(original).encode("utf-8")) + len("</think>")
    with pytest.raises(TemplateError, match="chat-render byte budget"):
        wrapper.render(messages, tools=None, enable_thinking=False, max_render_bytes=needed - 1)
    text = wrapper.render(messages, tools=None, enable_thinking=False, max_render_bytes=needed)
    assert len(str(text).encode("utf-8")) == needed and str(text).endswith(OPENED + "</think>")


def test_transform_credit_precedes_new_origin_and_denial_keeps_primary(checkpoint, monkeypatch):
    _tokenizer, inner, wrapper = checkpoint()
    messages = [{"role": "user", "content": "hello"}]
    original = inner.render(messages, tools=None, enable_thinking=False, extra={"clear_thinking": False})
    primary, records = RuntimeError("transform credit denied"), []

    def reserve(chars, parts):
        records.append((chars, parts))
        if chars > len(original):
            raise primary

    def unexpected(*args, **kwargs):
        pytest.fail("transformed prompt allocated before its credit")

    monkeypatch.setattr(glm_prompts, "RenderedPrompt", unexpected)
    with pytest.raises(RuntimeError) as caught:
        wrapper.render(messages, tools=None, enable_thinking=False, render_credit=reserve)
    assert caught.value is primary and records[-1][0] > 2 * len(original)


def test_wide_literal_runs_fund_both_media_views_and_build_one_final_prompt(checkpoint, monkeypatch):
    source = EFFORT_LINE + "{% for c in messages[0].content %}{{ c }}🧪{% endfor %}" + OPENED
    _tokenizer, inner, wrapper = checkpoint(source)
    value = "".join(chr(0x10000 + i) for i in range(100))
    messages = [{"role": "user", "content": value}]
    original = inner.render(messages, tools=None, enable_thinking=False, extra={"clear_thinking": False})
    records, constructed = [], []
    constructor = glm_prompts.RenderedPrompt

    def construct(*args, **kwargs):
        assert records[-1][1] >= len(original.literals) + 100
        result = constructor(*args, **kwargs)
        constructed.append(result)
        return result

    monkeypatch.setattr(glm_prompts, "RenderedPrompt", construct)
    text = wrapper.render(messages, tools=None, enable_thinking=False,
                          render_credit=lambda *args: records.append(args))
    assert constructed == [text] and len(text.literals) == len(original.literals) == 100
    assert str(text) == "".join(c + "🧪" for c in value) + OPENED + "</think>"
    assert records[-1][0] >= len(original) + len(original.encoded_text) + len(text) + len(text.encoded_text)
    assert records[-1][1] >= len(original.literals) + len(text.literals)


@pytest.mark.parametrize("phase", ["ids", "cache"])
def test_encode_credit_denial_propagates_without_fallback(checkpoint, phase):
    tokenizer, _inner, wrapper = checkpoint()
    text = wrapper.render([{"role": "user", "content": OPENED}], tools=None, enable_thinking=False)
    primary = RuntimeError(phase + " credit denied")

    def denied(*args, **kwargs):
        raise primary

    with pytest.raises(RuntimeError) as caught:
        wrapper.encode(tokenizer, text, segments=SegmentEncoder(tokenizer, "<|user|>", limit=4096),
                       ids_credit=denied if phase == "ids" else lambda *args, **kwargs: None,
                       cache_credit=denied if phase == "cache" else lambda *args: None)
    assert caught.value is primary


@pytest.mark.parametrize("extra", [[], False, "invalid", {"messages": []}, {"add_generation_prompt": False}])
def test_shared_render_validation_survives_the_adapter(checkpoint, extra):
    _tokenizer, _inner, wrapper = checkpoint()
    with pytest.raises(TemplateError, match="chat template kwargs"):
        wrapper.render([{"role": "user", "content": "hello"}], tools=None, enable_thinking=False, extra=extra)


@pytest.mark.parametrize("source", [SOURCE, "{{ messages[0].content }}",
                                   "{{ messages[0].content }}" + EFFORT_LINE])
@pytest.mark.parametrize("generation", [False, True])
def test_shared_hf_wrapper_keeps_the_same_literal_authority_and_renderer(checkpoint, source, generation):
    from transformers import PreTrainedTokenizerFast

    tokenizer, _inner, cuda = checkpoint(source)
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, chat_template=source)
    shared = GlmTokenizer(hf, clear=False)
    renderer = HFPromptRenderer(shared)
    literal = EFFORT_LINE + " quoted " + OPENED
    messages = [{"role": "user", "content": literal}]
    text = render_hf_prompt(shared, messages, tools=None, enable_thinking=False,
                            add_generation_prompt=generation, tokenize=False, renderer=renderer)
    if source == SOURCE:
        expected = PREFIX + "<|user|>" + literal + (OPENED + "</think>" if generation else "")
        controls = [tokenizer.token_to_id("<|user|>")]
        if generation:
            controls.extend(tokenizer.token_to_id(token) for token in ("<|assistant|>", "<think>", "</think>"))
    else:
        expected, controls = literal, []
    assert isinstance(text, RenderedPrompt) and str(text) == expected
    assert text._renderer is renderer
    at = str(text).index(literal)
    assert text.origin._mask[at:at + len(literal)] == b"\1" * len(literal)
    counted = encode_hf_prompt(shared, text)
    assert counted == encode_hf_prompt(shared, text)
    native = cuda.render(messages, tools=None, enable_thinking=False, add_generation_prompt=generation)
    assert str(native) == expected and counted == cuda.encode(tokenizer, native)
    reserved = {tokenizer.token_to_id(token) for token in SPECIALS}
    assert [token for token in counted if token in reserved] == controls
