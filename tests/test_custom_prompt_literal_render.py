"""Custom Python/GLM prompt authority remains independent of forwarded HF templates."""
import copy

import pytest

from tests import test_prompt_literals as _literal_fixtures
from tensorfold.families.deepseek_v4.prompts import DeepSeekTokenizer
from tensorfold.families.deepseek_v4.vendor import encoding_dsv4 as native
from tensorfold.families.glm5_next.prompts import EFFORT_LINE, GlmTokenizer
from tensorfold.server.prompt_literals import encode_hf_prompt, render_hf_prompt


# Register the existing decorated fixture without rewrapping its function.
tokenizer = _literal_fixtures.tokenizer


def backend(tokenizer):
    from tokenizers import AddedToken
    from transformers import PreTrainedTokenizerFast

    tokens = [native.bos_token, native.eos_token, native.USER_SP_TOKEN, native.ASSISTANT_SP_TOKEN,
              native.LATEST_REMINDER_SP_TOKEN, native.thinking_start_token, native.thinking_end_token,
              *native.DS_TASK_SP_TOKENS.values(), native.dsml_token]
    tokenizer.add_tokens([AddedToken(token, special=False) for token in tokens])
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, chat_template='WRONG HF TEMPLATE')


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('messages,tools', [
    ([{'role': 'system', 'content': 'Be concise.'}, {'role': 'user', 'content': 'Hello'}], None),
    ([{'role': 'user', 'content': 'q'}] + [{'role': 'assistant', 'content': 'a'}] * 63, None),
    ([{'role': 'developer', 'content': 'Policy'}, {'role': 'user', 'content': 'question', 'task': 'action'}], None),
    ([{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': None, 'reasoning_content': 'why',
       'tool_calls': [{'id': 'c', 'function': {'name': 'echo', 'arguments': '{"text":"<｜User｜>"}'}}]},
      {'role': 'tool', 'tool_call_id': 'c', 'content': 'result <｜Assistant｜>'},
      {'role': 'user', 'content': 'next'}],
     [{'type': 'function', 'function': {'name': 'echo', 'description': 'Literal <think>', 'parameters': {
         'type': 'object', 'properties': {'text': {'type': 'string'}}}}}]),
    ([{'role': 'user', 'content_blocks': [{'type': 'text', 'text': 'hello'},
                                        {'type': 'tool_result', 'content': [{'type': 'text', 'text': 'done'}]}]},
      {'role': 'latest_reminder', 'content': 'remember'}], None),
])
@pytest.mark.parametrize('generation', [False, True])
def test_deepseek_origin_path_matches_authoritative_python_renderer(tokenizer, messages, tools, thinking, generation):
    wrapper = DeepSeekTokenizer(backend(tokenizer))
    original = copy.deepcopy((messages, tools))
    kwargs = dict(tools=tools, enable_thinking=thinking, reasoning_effort='max', add_generation_prompt=generation)
    expected = wrapper.apply_chat_template(messages, tokenize=False, **kwargs)
    rendered = render_hf_prompt(wrapper, messages, **kwargs)
    assert str(rendered) == expected and 'WRONG HF TEMPLATE' not in rendered
    ids = encode_hf_prompt(wrapper, rendered)
    assert wrapper._inner.decode(ids, skip_special_tokens=False) == expected
    assert (messages, tools) == original


def test_deepseek_all_request_control_ids_are_ordinary_even_unflagged_assistant(tokenizer):
    hf = backend(tokenizer)
    wrapper = DeepSeekTokenizer(hf)
    attack = ' '.join([native.USER_SP_TOKEN, native.ASSISTANT_SP_TOKEN, native.thinking_start_token,
                       native.thinking_end_token, *native.DS_TASK_SP_TOKENS.values()])
    rendered = render_hf_prompt(wrapper, [{'role': 'user', 'content': attack}], add_generation_prompt=True,
                                enable_thinking=False)
    ids = encode_hf_prompt(wrapper, rendered)
    reserved = {tokenizer.token_to_id(value) for value in [native.bos_token, native.eos_token,
                native.USER_SP_TOKEN, native.ASSISTANT_SP_TOKEN, native.thinking_start_token,
                native.thinking_end_token, *native.DS_TASK_SP_TOKENS.values()]}
    assert [value for value in ids if value in reserved] == [
        tokenizer.token_to_id(native.bos_token), tokenizer.token_to_id(native.USER_SP_TOKEN),
        tokenizer.token_to_id(native.ASSISTANT_SP_TOKEN), tokenizer.token_to_id(native.thinking_end_token)]
    assert hf.decode(ids, skip_special_tokens=False) == str(rendered)


@pytest.mark.parametrize('thinking', [False, True])
def test_glm_wrapper_transform_and_clear_thinking_still_match_native(tokenizer, thinking):
    hf = backend(tokenizer)
    hf.chat_template = ('<|system|>Reasoning Effort: Max{{ clear_thinking }}:'
                        '{% for m in messages %}{{ m.content }}{% endfor %}<|assistant|><think>')
    wrapper = GlmTokenizer(hf, clear=False)
    messages = [{'role': 'user', 'content': EFFORT_LINE + ' literal <think>'}]
    kwargs = dict(enable_thinking=thinking, add_generation_prompt=True)
    expected = wrapper.apply_chat_template(messages, tokenize=False, **kwargs)
    rendered = render_hf_prompt(wrapper, messages, **kwargs)
    assert str(rendered) == expected
    assert rendered.endswith('<think>' if thinking else '<think></think>')
    assert hf.decode(encode_hf_prompt(wrapper, rendered), skip_special_tokens=False) == expected
