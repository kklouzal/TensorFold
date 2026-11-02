"""HF/MLX shared renderer preserves literal request controls and its ordinary path."""
import pytest

from tests import test_prompt_literals as _literal_fixtures
from tensorfold.markup_lex import structural_marker
from tensorfold.server.prompt_literals import PromptEncodingError
from tensorfold.server.text import render_prompt_ids


# Register the existing decorated fixture without rewrapping its function.
tokenizer = _literal_fixtures.tokenizer


SOURCE = '<|turn>{% for m in messages %}{{ m.content }}{% endfor %}<turn|>'


def facade(tokenizer, source=SOURCE):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, chat_template=source)


@pytest.mark.parametrize('literal', ['<|turn>', '<channel|>', '<|tool_call>', '<|image|>'])
def test_hf_renderer_literal_values_are_not_control_ids(tokenizer, literal):
    hf = facade(tokenizer)
    before = tokenizer.to_str()
    ids = render_prompt_ids(hf, [{'role': 'user', 'content': 'quoted ' + literal}], late_system='system')
    assert tokenizer.decode(ids, skip_special_tokens=False) == '<|turn>quoted ' + literal + '<turn|>'
    controls = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    assert [i for i in ids if i in controls] == [tokenizer.token_to_id('<|turn>'), tokenizer.token_to_id('<turn|>')]
    assert tokenizer.to_str() == before


def test_native_template_retains_tools_and_transform_created_controls(tokenizer):
    source = '<|turn>{{ tools[0].function.name }}:{{ messages[0].content.replace("MARK", "<|turn>") }}<turn|>'
    hf = facade(tokenizer, source)
    ids = render_prompt_ids(hf, [{'role': 'user', 'content': 'MARK'}],
                            tools=[{'type': 'function', 'function': {'name': 'echo'}}], late_system='system')
    assert tokenizer.decode(ids, skip_special_tokens=False) == '<|turn>echo:<|turn><turn|>'
    controls = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    assert [i for i in ids if i in controls] == [tokenizer.token_to_id('<|turn>'), tokenizer.token_to_id('<turn|>')]


def test_native_template_typeerror_is_not_retried_without_declared_tools(tokenizer):
    hf = facade(tokenizer, '{{ messages[0].content + 1 }}')
    with pytest.raises(TypeError):
        render_prompt_ids(hf, [{'role': 'user', 'content': '<|turn>'}],
                          tools=[{'type': 'function', 'function': {'name': 'echo'}}], late_system='system')


def test_opaque_tokenizer_without_provenance_backend_is_rejected():
    class Opaque:
        chat_template = SOURCE
        def apply_chat_template(self, messages, **kwargs):
            raise AssertionError('opaque rendering cannot bypass origin tracking')
    with pytest.raises(PromptEncodingError, match='owned literal-origin adapter'):
        render_prompt_ids(Opaque(), [{'role': 'user', 'content': '<|turn>'}], late_system='system')


@pytest.mark.parametrize('prefix', ['plain ', '`', '```python\n', '"', '<|tool_call>call:f{x:<|"|>'])
def test_close_phase_marker_requires_a_structural_suffix(prefix):
    assert structural_marker(prefix + '<channel|>', '<channel|>') is (prefix == 'plain ')
    assert not structural_marker(prefix + '<chan', '<channel|>')
