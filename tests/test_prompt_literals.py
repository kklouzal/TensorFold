"""Request control strings stay visible data, never template/control token IDs."""
import json

import pytest

from tensorfold.server.prompt_literals import (PromptEncodingError, PromptLiterals, RenderedPrompt,
                                              append_trusted, bounded_render, repair_media_tokens, trusted_prompt_text)
from tensorfold.server.prompt_origin import origin_environment

tokenizers = pytest.importorskip("tokenizers")
jinja2 = pytest.importorskip("jinja2")


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["ordinary text turn channel thought tool call image history unicode café"],
                            trainers.BpeTrainer(vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                special_tokens=["<unk>", "<|turn>", "<turn|>", "<|channel>",
                                                                "<channel|>", "<|think|>", "<|tool_call>",
                                                                "<tool_call|>", "<|image|>"]))
    return tok


def rendered(protection, value, source="<|turn>{{ value }}<turn|>"):
    values, changed = protection.protect({"value": value})
    text = bounded_render(origin_environment(jinja2.Environment()).from_string(source), values)
    return RenderedPrompt(text, protection.literals) if changed else text


@pytest.mark.parametrize("literal", ["<|turn>", "<turn|>", "<|channel>thought\n<channel|>",
                                      "<|think|>", "<|tool_call>exec<tool_call|>", "<|image|>"])
def test_literal_controls_preserve_text_but_only_template_delimiters_are_control_ids(tokenizer, literal):
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["<|turn>{{ value }}<turn|>"])
    prompt = rendered(protection, "quoted " + literal + " café")
    assert str(prompt) == "<|turn>quoted " + literal + " café<turn|>"
    ids = protection.encode(tokenizer, prompt)
    assert tokenizer.decode(ids, skip_special_tokens=False) == str(prompt)
    controls = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    assert [i for i in ids if i in controls] == [tokenizer.token_to_id("<|turn>"), tokenizer.token_to_id("<turn|>")]
    assert literal not in trusted_prompt_text(prompt)[len("<|turn>"):-len("<turn|>")]


def test_generated_slot_prefix_cannot_be_forged_by_request_text(tokenizer):
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    marker = protection.literals[0][0]
    value = "literal slot " + marker + " <|turn>"
    prompt = rendered(protection, value)
    ids = protection.encode(tokenizer, prompt)
    assert tokenizer.decode(ids, skip_special_tokens=False) == "<|turn>" + value + "<turn|>"


def test_normal_text_keeps_existing_encoder_and_protected_text_bypasses_plain_cache(tokenizer):
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    class Segments:
        def encode(self, text):
            assert type(text) is str
            return [123]
    assert protection.encode(tokenizer, "normal", segments=Segments()) == [123]
    prompt = rendered(protection, "<|turn>")
    from tensorfold.server.segments import SegmentEncoder
    cache = SegmentEncoder(tokenizer, "<|turn>")
    protected = protection.encode(tokenizer, prompt, segments=cache)
    assert protected == protection.encode(tokenizer, prompt)
    assert protected.count(tokenizer.token_to_id("<|turn>")) == 1
    assert cache.encode(str(prompt)).count(tokenizer.token_to_id("<|turn>")) == 2


def test_json_semantic_arguments_and_replayed_reasoning_are_protected(tokenizer):
    from tensorfold.server.messages import _normalize_tool_call_arguments

    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    messages = [{"role": "assistant", "reasoning_content": "<|channel>old",
                 "tool_calls": [{"function": {"name": "f", "arguments": '{"x":"\\u003c|turn>"}'}}]},
                {"role": "tool", "content": "<|tool_call>untrusted"}]
    normalized = _normalize_tool_call_arguments(messages)
    protected, changed = protection.protect(normalized)
    assert changed
    # Protection is out-of-band: native filters/branches must see the original
    # semantic data, including decoded tool-argument strings.
    assert "<|turn>" in json.dumps(protected)
    assert normalized[0]["tool_calls"][0]["function"]["arguments"]["x"] == "<|turn>"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"x":"\\u003c|turn>"}'


def test_provenance_is_immutable_and_render_work_is_bounded(tokenizer):
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    prompt = rendered(protection, "<|turn>")
    with pytest.raises(AttributeError):
        prompt.encoded_text = "forged"
    with pytest.raises(AttributeError):
        del prompt.literals
    with pytest.raises(PromptEncodingError):
        protection.protect("<|turn>" * 100, max_bytes=32)
    template = jinja2.Environment().from_string("{% for n in range(1000) %}abc{% endfor %}")
    with pytest.raises(PromptEncodingError, match="byte budget"):
        bounded_render(template, {}, 128)


def test_media_expansion_preserves_trusted_media_and_not_quoted_media(tokenizer):
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    prompt = rendered(protection, "quoted <|image|>")
    expanded = prompt.encoded_text.replace("<turn|>", "<|image|><turn|>")
    raw = tokenizer.encode(expanded, add_special_tokens=False).ids
    tokens = repair_media_tokens(prompt, tokenizer, raw, protection)
    assert tokens.count(tokenizer.token_to_id("<|image|>")) == 1
    assert tokenizer.decode(tokens, skip_special_tokens=False) == str(prompt).replace("<turn|>", "<|image|><turn|>")


def test_generated_decision_label_suffix_preserves_literal_encoding(tokenizer):
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    prompt = rendered(protection, "literal <|turn>")
    labelled = append_trusted(prompt, "A")
    ids = protection.encode(tokenizer, labelled)
    assert tokenizer.decode(ids, skip_special_tokens=False) == str(prompt) + "A"
    assert ids.count(tokenizer.token_to_id("<|turn>")) == 1


def test_actual_chat_template_path_keeps_literal_text_and_trusted_delimiters(tokenizer, tmp_path):
    from tensorfold.cuda.chat_template import ChatTemplate

    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<|turn>", "eos_token": "<turn|>"}))
    (tmp_path / "chat_template.jinja").write_text(
        "{{ bos_token }}{% for m in messages %}{{ m.content }}{{ m.reasoning_content|default('') }}"
        "{% for call in m.tool_calls|default([]) %}{{ call.function.arguments|tojson }}{% endfor %}"
        "{% endfor %}{{ eos_token }}")
    chat = ChatTemplate(tmp_path)
    messages = [{"role": "assistant", "content": "quoted <|turn>", "reasoning_content": "old <|think|>",
                 "tool_calls": [{"function": {"name": "f", "arguments": '{"x":"\\u003c|channel>"}'}}]},
                {"role": "tool", "content": "result <|tool_call>"}]
    prompt = chat.render(messages, tools=None, enable_thinking=True)
    tokens = chat.encode(tokenizer, prompt)
    assert tokenizer.decode(tokens, skip_special_tokens=False) == str(prompt)
    controls = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    assert [i for i in tokens if i in controls] == [tokenizer.token_to_id("<|turn>"), tokenizer.token_to_id("<turn|>")]
    with pytest.raises(jinja2.exceptions.TemplateError, match="cannot replace prepared"):
        chat.render(messages, tools=None, enable_thinking=True, extra={"messages": []})


@pytest.mark.parametrize("quoted", ["quoted <|image|>", "quoted ＜|image|>"])
def test_native_gemma_processor_keeps_quoted_image_separate_from_real_image(tokenizer, quoted):
    import numpy as np
    from transformers import GemmaTokenizer
    from test_gemma_cuda_vision_processor import standard_processor, get_aspect_ratio_preserving_size
    from tensorfold.families.gemma4.cuda.vision import GemmaVisionProcessor
    from tensorfold.vision.images import ImageInput

    raw, processor = standard_processor()
    names = {"image_token": "<|image|>", "boi_token": "<|image>", "eoi_token": "<image|>",
             "audio_token": "<|audio|>", "boa_token": "<|audio>", "eoa_token": "<audio|>"}
    native = GemmaTokenizer(tokenizer_object=tokenizer, model_specific_special_tokens=names,
                            extra_special_tokens=["<|video|>"])
    processor = type(processor)(processor.feature_extractor, processor.image_processor, native,
                                processor.video_processor, image_seq_length=70)
    raw["image_token_id"] = processor.tokenizer.convert_tokens_to_ids(names["image_token"])
    raw["video_token_id"] = processor.tokenizer.convert_tokens_to_ids("<|video|>")
    backend = processor.tokenizer.backend_tokenizer
    from tokenizers import AddedToken, normalizers
    backend.normalizer = normalizers.NFKC()
    backend.add_special_tokens([AddedToken("<|image|>", special=True, normalized=True)])
    protection = PromptLiterals(json.loads(backend.to_str()), ["template"])
    value, _ = protection.protect(quoted)
    from tensorfold.server.prompt_origin import origin_concat, trusted

    prompt = RenderedPrompt(origin_concat((value, trusted(" <|image|>"))), protection.literals)
    frontend = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    image = ImageInput(8, 4, pixels.tobytes())
    counted = frontend.prepare(prompt, [image], max_prompt_tokens=None)
    generated = frontend.prepare(prompt, [image], max_prompt_tokens=len(counted.token_ids))
    assert counted.token_ids == generated.token_ids and len(counted.frames) == 1
    assert counted.token_ids.count(raw["image_token_id"]) == counted.frames[0].soft_tokens
    assert "quoted <|image|>" in backend.decode(list(counted.token_ids), skip_special_tokens=False)
    assert quoted in str(prompt)


@pytest.mark.parametrize("expression,value", [
    ("value|upper", "ßéΣ<|turn>"),
    ("value|lower", "ΟΣİ<|TURN>"),
    ("value[:7]", "<|turn>suffix"),
    ("value|replace('X', '')", "<|tuXrn>"),
    ("value|length", "<|turn>"),
    ("value|trim", "  <|turn>  "),
])
def test_native_transforms_see_original_values_and_created_controls_are_literal(tokenizer, expression, value):
    from tokenizers import AddedToken

    tokenizer.add_special_tokens([AddedToken("<|TURN>", special=True, normalized=False)])
    source = "<|turn>{{ " + expression + " }}<turn|>"
    baseline = jinja2.Environment().from_string(source).render(value=value)
    protection = PromptLiterals(json.loads(tokenizer.to_str()), [source])
    prompt = rendered(protection, value, source)
    assert str(prompt) == baseline
    ids = protection.encode(tokenizer, prompt)
    assert tokenizer.decode(ids, skip_special_tokens=False) == baseline
    controls = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    assert [i for i in ids if i in controls] == [tokenizer.token_to_id("<|turn>"), tokenizer.token_to_id("<turn|>")]


@pytest.mark.parametrize("source", [
    "{{ '<|turn>' ~ (value|replace('X', '')) ~ '<turn|>' }}",
    "{% macro wrap(v) %}<|turn>{{ v|replace('X', '') }}<turn|>{% endmacro %}{{ wrap(value) }}",
    "{% set body %}<|turn>{{ value|replace('X', '') }}<turn|>{% endset %}{{ body }}",
    "{% if value|length == 8 %}<|turn>{{ value|replace('X', '') }}<turn|>{% else %}wrong{% endif %}",
    "{{ ['<|turn>', value|replace('X', ''), '<turn|>']|join('') }}",
    "{{ '<|turn>{}<turn|>'.format(value|replace('X', '')) }}",
    "{{ '<|turn>%s<turn|>'|format(value|replace('X', '')) }}",
    "{{ '<|turn>%(text)s<turn|>'|format(text=value|replace('X', '')) }}",
    "{{ '<|turn>%(name2)s<turn|>'|format(name2=value|replace('X', '')) }}",
])
def test_native_mixed_trusted_syntax_macros_capture_and_length_branches(tokenizer, source):
    value = "<|tuXrn>"
    protection = PromptLiterals(json.loads(tokenizer.to_str()), [source])
    prompt = rendered(protection, value, source)
    baseline = jinja2.Environment().from_string(source).render(value=value)
    assert baseline == "<|turn><|turn><turn|>"
    assert str(prompt) == baseline
    ids = protection.encode(tokenizer, prompt)
    assert tokenizer.decode(ids, skip_special_tokens=False) == baseline
    assert ids.count(tokenizer.token_to_id("<|turn>")) == 1
    assert ids.count(tokenizer.token_to_id("<turn|>")) == 1


def test_same_text_replace_changes_origin_and_partial_control_crossing_trusted_suffix(tokenizer):
    from tensorfold.server.prompt_origin import OriginText, trusted

    value = trusted("<|turn>").replace("<|turn>", OriginText("<|turn>"))
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    prompt = RenderedPrompt(value, protection.literals)
    assert tokenizer.token_to_id("<|turn>") not in protection.encode(tokenizer, prompt)
    partial = RenderedPrompt(OriginText("<|tu"), protection.literals)
    completed = append_trusted(partial, "rn>")
    assert str(completed) == "<|turn>"
    assert tokenizer.token_to_id("<|turn>") not in protection.encode(tokenizer, completed)


@pytest.mark.parametrize("value,old", [("MARK", "MARK"), ("", "")])
def test_trusted_replacement_spelling_inside_request_value_stays_data(tokenizer, value, old):
    source = "<|turn>{{ value|replace(old, '<|turn>') }}<turn|>"
    protection = PromptLiterals(json.loads(tokenizer.to_str()), [source])
    values, _ = protection.protect({"value": value, "old": old})
    prompt = RenderedPrompt(bounded_render(origin_environment(jinja2.Environment()).from_string(source), values),
                            protection.literals)
    assert str(prompt) == "<|turn><|turn><turn|>"
    ids = protection.encode(tokenizer, prompt)
    assert ids.count(tokenizer.token_to_id("<|turn>")) == 1
    # An operation exclusively on template-authored values remains syntax.
    pure = rendered(protection, "", "{{ 'MARK'|replace('MARK', '<|turn>') }}")
    assert protection.encode(tokenizer, pure) == [tokenizer.token_to_id("<|turn>")]


@pytest.mark.parametrize("model_role", [False, True])
def test_configured_role_id_is_protected_without_added_special_flag(tokenizer, model_role):
    from tokenizers import AddedToken, Tokenizer

    spelling = "ROLE"
    if model_role:
        data = json.loads(tokenizer.to_str())
        for left, right in [("R", "O"), ("RO", "L"), ("ROL", "E")]:
            word = left + right
            assert word not in data["model"]["vocab"]
            data["model"]["vocab"][word] = len(data["model"]["vocab"])
            data["model"]["merges"].append([left, right])
        tokenizer = Tokenizer.from_str(json.dumps(data))
    else:
        tokenizer.add_tokens([AddedToken(spelling, special=False, normalized=False)])
    role_id = tokenizer.token_to_id(spelling)
    assert role_id is not None
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"], [spelling])
    prompt = rendered(protection, spelling)
    ids = protection.encode(tokenizer, prompt)
    assert role_id not in ids
    assert tokenizer.decode(ids, skip_special_tokens=False) == str(prompt)


def test_unicode_normalization_control_id_uses_original_character_origin(tokenizer):
    from tokenizers import AddedToken, normalizers

    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.add_special_tokens([AddedToken("<|turn>", special=True, normalized=True)])
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    prompt = rendered(protection, "é😀＜|turn>")
    ids = protection.encode(tokenizer, prompt)
    assert ids.count(tokenizer.token_to_id("<|turn>")) == 1
    assert tokenizer.decode(ids, skip_special_tokens=False) == "<|turn>é😀<|turn><turn|>"


def test_default_join_of_model_tokens_does_not_invent_request_dependency(tokenizer):
    from tensorfold.server.prompt_origin import trusted

    source = "{{ [bos, eos]|join|replace('turn', 'turn') }}"
    protection = PromptLiterals(json.loads(tokenizer.to_str()), [source])
    context = {"bos": trusted("<|turn>"), "eos": trusted("<turn|>")}
    prompt = RenderedPrompt(bounded_render(origin_environment(jinja2.Environment()).from_string(source), context),
                            protection.literals)
    assert protection.encode(tokenizer, prompt) == [tokenizer.token_to_id("<|turn>"), tokenizer.token_to_id("<turn|>")]


def test_segment_cache_reuses_native_offsets_without_caching_request_origin(tokenizer):
    from tensorfold.server.prompt_origin import OriginText, origin_concat, trusted
    from tensorfold.server.segments import SegmentEncoder

    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    cache = SegmentEncoder(tokenizer, "<|turn>", limit=4096)
    fragment = "<|turn>x<turn|>"
    first = RenderedPrompt(origin_concat((trusted(fragment), OriginText(fragment))), protection.literals)
    second = RenderedPrompt(origin_concat((OriginText(fragment), trusted(fragment))), protection.literals)
    for prompt in [first, second, first]:
        ids = protection.encode(tokenizer, prompt, segments=cache)
        assert tokenizer.decode(ids, skip_special_tokens=False) == fragment * 2
        assert ids.count(tokenizer.token_to_id("<|turn>")) == 1
        assert ids.count(tokenizer.token_to_id("<turn|>")) == 1
        assert ids == protection.encode(tokenizer, prompt)
    assert cache.cache and cache.held <= cache.limit
    assert all(type(text) is str for text in cache.cache)


def test_app_owned_hf_preparation_reuses_two_bounded_routes_without_mutating_tokenizer(tokenizer, monkeypatch):
    from transformers import PreTrainedTokenizerFast
    from tensorfold.server import prompt_literals as literals

    native = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                    chat_template="<|turn>{{ messages[0].content }}<turn|>")
    owner = literals.HFPromptRenderer(native)
    original = tokenizer.to_str()
    calls = []
    prepare = literals.origin_environment
    def observed(environment):
        calls.append(environment)
        return prepare(environment)
    monkeypatch.setattr(literals, "origin_environment", observed)
    for value in ["SECRET_one", "<|turn>", "SECRET_two"]:
        prompt = literals.render_hf_prompt(native, [{"role": "user", "content": value}], renderer=owner)
        ids = literals.encode_hf_prompt(native, prompt)
        assert tokenizer.decode(ids, skip_special_tokens=False) == str(prompt)
    assert len(calls) == 1
    assert "SECRET" not in repr(owner._entries)
    for suffix in ["one", "two", "three"]:
        literals.render_hf_prompt(native, [{"role": "user", "content": "x"}], renderer=owner,
                                 chat_template="{{ messages[0].content }}" + suffix)
        assert len(owner._entries) <= 2
    assert tokenizer.to_str() == original


def test_ordinary_codec_append_reencodes_whole_prompt_and_checks_reserved_ids_without_rerender():
    from tensorfold.server.prompt_literals import encode_hf_prompt, render_hf_prompt

    class Lexical:
        eos_token_id = 99
        def __init__(self):
            self.calls = 0
        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            return messages[0]["content"]
        def encode(self, value, **kwargs):
            if value.startswith("xy"):
                return [7, *map(ord, value[2:])]
            return list(map(ord, value))
        def decode(self, ids, **kwargs):
            return "".join("xy" if token == 7 else chr(token) for token in ids)
    codec = Lexical()
    prompt = render_hf_prompt(codec, [{"role": "user", "content": "x"}])
    assert encode_hf_prompt(codec, prompt) == [ord("x")]
    completed = append_trusted(prompt, "y")
    assert encode_hf_prompt(codec, completed) == [7]
    assert codec.calls == 1
    with pytest.raises(PromptEncodingError, match="emitted controls"):
        encode_hf_prompt(codec, append_trusted(completed, "c"))
    assert codec.calls == 1


def test_model_role_lists_are_trusted_but_request_override_is_literal(tokenizer):
    from transformers import PreTrainedTokenizerFast
    from tensorfold.server.prompt_literals import encode_hf_prompt, render_hf_prompt

    class ModelRoles(PreTrainedTokenizerFast):
        @property
        def special_tokens_map(self):
            return {**super().special_tokens_map, "additional_special_tokens": ["<|image|>"]}
    native = ModelRoles(tokenizer_object=tokenizer,
                        chat_template="{{ additional_special_tokens[0] }}{{ messages[0].content }}")
    messages = [{"role": "user", "content": "x"}]
    prompt = render_hf_prompt(native, messages)
    assert encode_hf_prompt(native, prompt).count(tokenizer.token_to_id("<|image|>")) == 1
    overridden = render_hf_prompt(native, messages, additional_special_tokens=["<|image|>"])
    assert str(overridden) == str(prompt)
    assert encode_hf_prompt(native, overridden).count(tokenizer.token_to_id("<|image|>")) == 0


@pytest.mark.parametrize("source", ["{{ '<|turn>%.*s<turn|>'|format(-1, value) }}",
                                    "{{ '<|turn>%*.*s<turn|>'|format(3, -1, value) }}"])
def test_negative_percent_precision_matches_native_semantics(tokenizer, source):
    protection = PromptLiterals(json.loads(tokenizer.to_str()), [source])
    prompt = rendered(protection, "abc", source)
    assert str(prompt) == jinja2.Environment().from_string(source).render(value="abc")
    ids = protection.encode(tokenizer, prompt)
    assert ids.count(tokenizer.token_to_id("<|turn>")) == 1
    assert ids.count(tokenizer.token_to_id("<turn|>")) == 1


def test_percent_star_budget_precedes_native_field_materialization():
    from tensorfold.server.prompt_origin import MAX_ORIGIN_CHARS, trusted

    calls = []
    class Field:
        def __str__(self):
            calls.append(True)
            raise AssertionError("field must not be materialized")
    with pytest.raises(PromptEncodingError, match="character budget"):
        trusted("%*s") % (MAX_ORIGIN_CHARS + 1, Field())
    assert not calls


@pytest.mark.parametrize("edge", ["closing_lstrip", "opening_rstrip"])
def test_control_content_origin_excludes_native_stripped_request_whitespace(tokenizer, edge):
    from tokenizers import AddedToken
    from tensorfold.server.segments import SegmentEncoder

    token = (AddedToken("<turn|>", special=True, normalized=False, lstrip=True)
             if edge == "closing_lstrip"
             else AddedToken("<|turn>", special=True, normalized=False, rstrip=True))
    tokenizer.add_special_tokens([token])
    protection = PromptLiterals(json.loads(tokenizer.to_str()), ["template"])
    prompt = rendered(protection, " ordinary ")
    expected = tokenizer.encode(str(prompt), add_special_tokens=False).ids
    assert protection.encode(tokenizer, prompt) == expected
    if edge == "closing_lstrip":
        cache = SegmentEncoder(tokenizer, "<|turn>")
        assert protection.encode(tokenizer, prompt, segments=cache) == expected
    assert expected.count(tokenizer.token_to_id("<|turn>")) == 1
    assert expected.count(tokenizer.token_to_id("<turn|>")) == 1


def test_validated_role_enum_builds_framing_but_nested_argument_role_stays_data(tokenizer):
    from transformers import PreTrainedTokenizerFast
    from tensorfold.server.prompt_literals import encode_hf_prompt, render_hf_prompt

    tokenizer.add_special_tokens(["<|assistant|>", "<|user|>"])
    native = PreTrainedTokenizerFast(tokenizer_object=tokenizer, chat_template=(
        "{{ '<|' ~ messages[0].role ~ '|>' }}"
        "{{ '<|' ~ messages[0].tool_calls[0].function.arguments.role ~ '|>' }}"))
    messages = [{"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "echo", "arguments": {"role": "user"}}}]}]
    prompt = render_hf_prompt(native, messages)
    assert str(prompt) == "<|assistant|><|user|>"
    ids = encode_hf_prompt(native, prompt)
    assert ids.count(tokenizer.token_to_id("<|assistant|>")) == 1
    assert ids.count(tokenizer.token_to_id("<|user|>")) == 0
    with pytest.raises(PromptEncodingError, match="role enum"):
        render_hf_prompt(native, [{**messages[0], "role": "<|user|>"}])


def test_resolved_policy_facts_keep_native_types_and_nested_primitive_arguments_stay_data(tokenizer):
    from transformers import PreTrainedTokenizerFast
    from tensorfold.server.prompt_literals import encode_hf_prompt, render_hf_prompt

    tokenizer.add_special_tokens(["<|think_True|>", "<|effort_high|>"])
    native = PreTrainedTokenizerFast(tokenizer_object=tokenizer, chat_template=(
        "{% if enable_thinking is not boolean %}{{ raise_exception('bool type changed') }}{% endif %}"
        "{{ '<|think_' ~ enable_thinking ~ '|>' }}{{ '<|effort_' ~ reasoning_effort ~ '|>' }}"
        "{{ '<|think_' ~ messages[0].tool_calls[0].function.arguments.flag ~ '|>' }}"
        "{{ '<|effort_' ~ messages[0].tool_calls[0].function.arguments.reasoning_effort ~ '|>' }}"))
    messages = [{"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "echo", "arguments": {"flag": True, "reasoning_effort": "high"}}}]}]
    prompt = render_hf_prompt(native, messages, enable_thinking=True, reasoning_effort="high")
    assert str(prompt) == "<|think_True|><|effort_high|><|think_True|><|effort_high|>"
    ids = encode_hf_prompt(native, prompt)
    assert ids.count(tokenizer.token_to_id("<|think_True|>")) == 1
    assert ids.count(tokenizer.token_to_id("<|effort_high|>")) == 1


@pytest.mark.parametrize("body", [
    "{% set enable_thinking = messages[0].tool_calls[0].function.arguments.flag %}"
    "{{ '<|think_' ~ enable_thinking ~ '|>' }}",
    "{% set enable_thinking, unused = messages[0].tool_calls[0].function.arguments.flag, 0 %}"
    "{{ '<|think_' ~ enable_thinking ~ '|>' }}",
    "{% with enable_thinking = messages[0].tool_calls[0].function.arguments.flag %}"
    "{{ '<|think_' ~ enable_thinking ~ '|>' }}{% endwith %}",
    "{% macro header(enable_thinking) %}{{ '<|think_' ~ enable_thinking ~ '|>' }}{% endmacro %}"
    "{{ header(messages[0].tool_calls[0].function.arguments.flag) }}",
])
def test_shadowed_policy_binding_does_not_trust_equal_primitive_argument(tokenizer, body):
    from transformers import PreTrainedTokenizerFast
    from tensorfold.server.prompt_literals import encode_hf_prompt, render_hf_prompt

    tokenizer.add_special_tokens(["<|think_True|>"])
    native = PreTrainedTokenizerFast(tokenizer_object=tokenizer, chat_template=body)
    messages = [{"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "echo", "arguments": {"flag": True}}}]}]
    prompt = render_hf_prompt(native, messages, enable_thinking=True)
    assert str(prompt) == "<|think_True|>"
    assert tokenizer.token_to_id("<|think_True|>") not in encode_hf_prompt(native, prompt)


@pytest.mark.parametrize("body", [
    "{% set alias = enable_thinking %}{{ '<|think_' ~ alias ~ '|>' }}",
    "{% set enable_thinking = true %}{{ '<|think_' ~ enable_thinking ~ '|>' }}",
    "{{ '<|think_' ~ (enable_thinking|string)|upper ~ '|>' }}",
])
def test_owned_policy_alias_and_template_constant_remain_native_framing(tokenizer, body):
    from transformers import PreTrainedTokenizerFast
    from tensorfold.server.prompt_literals import encode_hf_prompt, render_hf_prompt

    tokenizer.add_special_tokens(["<|think_True|>", "<|think_TRUE|>"])
    native = PreTrainedTokenizerFast(tokenizer_object=tokenizer, chat_template=body)
    prompt = render_hf_prompt(native, [{"role": "user", "content": "x"}], enable_thinking=True)
    assert encode_hf_prompt(native, prompt) == tokenizer.encode(str(prompt), add_special_tokens=False).ids
