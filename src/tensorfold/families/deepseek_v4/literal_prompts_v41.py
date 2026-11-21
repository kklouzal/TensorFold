"""Origin boundary for DeepSeek's V4.1 Python encoder; the vendored module stays authoritative.

Owned adaptation of ``encode_messages``/``render_message`` from DeepSeek-V4.1-Flash's MIT
``encoding/encoding.py`` (vendor/encoding_dsv41.py; LICENSES/DeepSeek-MIT.txt). The body
follows upstream statement for statement; only string construction goes through the origin
view, so encoder-written controls stay trusted and request strings stay data. Its
preprocessing helpers run unchanged. The last-user index is a pure function of the
message list the loop renders, so it is computed once instead of once per message.
"""
from __future__ import annotations

import json

from tensorfold.server.prompt_origin import OriginText, trusted
from .literal_prompts import _NativeOrigin, _Template
from .vendor import encoding_dsv41 as native


class _V41Origin(_NativeOrigin):
    def __init__(self):
        super().__init__(native)

    def render_tools(self, tools):
        return _Template(native.TOOLS_TEMPLATE).format(
            tool_schemas=self.join('\n', [self.to_json(tool) for tool in tools]),
            dsml_token=self.dsml_token, tc_block_name=self.tool_calls_block_name,
            tool_call_tag_name=self.tool_call_tag_name, tool_parameter_tag_name=self.tool_parameter_tag_name,
            thinking_start_token=self.thinking_start_token, thinking_end_token=self.thinking_end_token)

    def encode_arguments_to_dsml(self, tool_call):
        arguments = tool_call['arguments']
        if not isinstance(arguments, dict):
            for _ in range(2):                       # upstream: JSON strings, including double-encoded ones
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        break
                else:
                    break
            if not isinstance(arguments, dict):
                arguments = {'arguments': tool_call['arguments']}
        template = _Template('<{dsml_token}{tool_parameter_tag_name} name="{key}" string="{is_str}">'
                             '{value}</{dsml_token}{tool_parameter_tag_name}>')
        parts = [template.format(dsml_token=self.dsml_token, tool_parameter_tag_name=self.tool_parameter_tag_name,
                                 key=OriginText(key), is_str=trusted('true' if isinstance(value, str) else 'false'),
                                 value=OriginText(value) if isinstance(value, str) else self.to_json(value))
                 for key, value in arguments.items()]
        return self.join('\n', parts)

    @staticmethod
    def render_reasoning_effort(index, thinking_mode, effort):
        # The closed, validated policy value: upstream renders and asserts it, the text is model-owned.
        return trusted(native.render_reasoning_effort(index, thinking_mode, effort))


def _render_message(o, index, messages, thinking_mode, drop_thinking, reasoning_effort, last_user_idx):
    assert 0 <= index < len(messages)
    assert thinking_mode in ['chat', 'thinking'], f'Invalid thinking_mode `{thinking_mode}`'
    msg = messages[index]
    role = msg.get('role')
    content = msg.get('content')
    tools = msg.get('tools')
    response_format = msg.get('response_format')
    tool_calls = msg.get('tool_calls')
    reasoning_content = msg.get('reasoning_content')
    wo_eos = msg.get('wo_eos', False)
    if tools:
        tools = native.tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = native.tool_calls_from_openai_format(tool_calls)
    reasoning_effort_prompt = o.render_reasoning_effort(index, thinking_mode, reasoning_effort)
    prompt = o.SYSTEM_SP_TOKEN if index == 0 and (reasoning_effort_prompt or role == 'system') else trusted('')
    prompt += reasoning_effort_prompt
    if role == 'system':
        if index > 0:
            prompt += o.SYSTEM_SP_TOKEN
        prompt += o.system_msg_template.format(content=content or '')
        if tools:
            prompt += trusted('\n\n') + o.render_tools(tools)
        if response_format:
            prompt += trusted('\n\n') + o.response_format_template.format(schema=o.to_json(response_format))
    elif role == 'user':
        prompt += o.USER_SP_TOKEN
        content_blocks = msg.get('content_blocks')
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get('type')
                if block_type == 'text':
                    parts.append(block.get('text', ''))
                elif block_type == 'tool_result':
                    tool_content = block.get('content', '')
                    if isinstance(tool_content, list):
                        text_parts = []
                        for b in tool_content:
                            if b.get('type') == 'text':
                                text_parts.append(b.get('text', ''))
                            else:
                                text_parts.append(o.unsupported(b.get('type')))
                        tool_content = o.join('\n\n', text_parts)
                    parts.append(o.tool_output_template.format(content=tool_content))
                else:
                    parts.append(o.unsupported(block_type))
            prompt += o.join('\n\n', parts)
        else:
            prompt += content or ''
    elif role == 'latest_reminder':
        prompt += o.LATEST_REMINDER_SP_TOKEN + o.latest_reminder_msg_template.format(content=content)
    elif role == 'tool':
        raise NotImplementedError('deepseek_v41 merges tool messages into user; please preprocess with merge_tool_messages()')
    elif role == 'assistant':
        thinking_part = trusted('')
        tc_content = trusted('')
        if tool_calls:
            tc_list = [o.tool_call_template.format(dsml_token=o.dsml_token, tool_call_tag_name=o.tool_call_tag_name,
                                                   name=native._tool_name_for_encoding(tc),
                                                   arguments=o.encode_arguments_to_dsml(tc))
                       for tc in tool_calls]
            tc_content += trusted('\n\n') + o.tool_calls_template.format(
                dsml_token=o.dsml_token, tool_calls=o.join('\n', tc_list), tc_block_name=o.tool_calls_block_name)
        summary_content = content or ''
        rc = reasoning_content or ''
        prev_has_task = index - 1 >= 0 and messages[index - 1].get('task') is not None
        if thinking_mode == 'thinking' and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = o.thinking_template.format(reasoning_content=rc) + o.thinking_end_token
        template = o.assistant_msg_wo_eos_template if wo_eos else o.assistant_msg_template
        prompt += template.format(reasoning=thinking_part, content=summary_content, tool_calls=tc_content)
    else:
        raise NotImplementedError(f'Unknown role: {role}')
    if index + 1 < len(messages) and messages[index + 1].get('role') not in ['assistant', 'latest_reminder']:
        return prompt
    task = messages[index].get('task')
    if task is not None:
        assert task in native.VALID_TASKS, f"Invalid task: '{task}'. Valid tasks are: {list(native.VALID_TASKS)}"
        task_sp_token = o.DS_TASK_SP_TOKENS[task]
        if task != 'action':
            prompt += task_sp_token
        else:
            prompt += o.ASSISTANT_SP_TOKEN
            prompt += o.thinking_end_token if thinking_mode != 'thinking' else o.thinking_start_token
            prompt += task_sp_token
    elif role == 'user' or (role == 'system' and index > 0):
        prompt += o.ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == 'thinking':
            prompt += o.thinking_start_token
        elif drop_thinking and thinking_mode == 'thinking' and index >= last_user_idx:
            prompt += o.thinking_start_token
        else:
            prompt += o.thinking_end_token
    return prompt


def encode(messages, *, thinking_mode, reasoning_effort, drop_thinking=True):
    """``encode_messages`` of the vendored encoder (no context, BOS added), with output origins."""
    o = _V41Origin()
    for msg in messages:
        native._validate_no_image_sp_tokens(msg)
    messages = native.merge_tool_messages(messages)
    messages = native.sort_tool_results_by_call_order(messages)
    messages, _ = native.process_image_messages(messages)
    prompt = o.bos_token
    effective_drop_thinking = drop_thinking and not any(m.get('tools') for m in messages)
    if thinking_mode == 'thinking' and effective_drop_thinking:
        messages = native._drop_thinking_messages(messages)
    last_user_idx = native.find_last_user_index(messages)
    for idx in range(len(messages)):
        prompt += _render_message(o, idx, messages, thinking_mode, effective_drop_thinking, reasoning_effort,
                                  last_user_idx)
    return prompt


def reserved_tokens(backend):
    values = [native.bos_token, native.eos_token, native.thinking_start_token, native.thinking_end_token,
              native.SYSTEM_SP_TOKEN, native.USER_SP_TOKEN, native.ASSISTANT_SP_TOKEN,
              native.LATEST_REMINDER_SP_TOKEN, native.IMAGE_PLACEHOLDER, *native.DS_TASK_SP_TOKENS.values(),
              native.dsml_token]
    return [value for value in values if backend.token_to_id(value) is not None]
