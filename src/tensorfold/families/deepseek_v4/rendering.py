"""Owned two-function adaptation of DeepSeek's MIT encoder, upstream60d8d70.

The supplied module owns its unchanged helpers/templates. This callable borrows
the normalized builtin-data message graph synchronously; callers must not mutate
it during rendering. Non-inert/custom data follows the original renderer.
Only a long rendered suffix without nearby user/developer benefits from the
precomputed index. No parsing, tokenizer, precision or model math changes.
"""
from __future__ import annotations

# Derived from DeepSeek's MIT encoder (copyright2023DeepSeek), retained in
# LICENSES/DeepSeek-MIT.txt. Helpers/templates remain authoritative in that unchanged
# module; this maintained adaptation owns only render_message/encode_messages.
from .vendor import encoding_dsv4 as _original


def _eligible(messages, count):
    # Size specialization is a candidate policy, never an input restriction.
    if count < 64 or type(messages) is not list:
        return False
    for message in messages[-63:]:
        if type(message) is not dict:
            return False
        role = message.get('role')
        if type(role) is not str or role not in ('assistant', 'latest_reminder'):
            return False
    # Native builtin containers/scalars cannot run callbacks while the copied
    # render body traverses them. Unknown payloads preserve the original path.
    # Bound the probe before expanding any container; oversized graphs fall back.
    pending, seen, consumed = [messages], set(), 0
    while pending:
        value = pending.pop()
        consumed += 1
        if consumed > 16384:
            return False
        kind = type(value)
        if kind in (str, int, float, bool, type(None)):
            continue
        if kind not in (dict, list, tuple):
            return False
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if len(value) > 16384 - consumed - len(pending):
            return False
        if kind is dict:
            if any(type(key) is not str for key in value):
                return False
            pending.extend(value.values())
        else:
            pending.extend(value)
    return True


def _create_encoder(_upstream):

    def _render_message(index: int, messages: _upstream.List[_upstream.Dict[str, _upstream.Any]], thinking_mode: str, drop_thinking: bool=True, reasoning_effort: _upstream.Optional[str]=None, *, _last_user_index: int) -> str:
        """
    Render a single message at the given index into its encoded string form.

    This is the core function that converts each message in the conversation
    into the DeepSeek-V4 format.

    Args:
        index: Index of the message to render.
        messages: Full list of messages in the conversation.
        thinking_mode: Either "chat" or "thinking".
        drop_thinking: Whether to drop reasoning content from earlier turns.
        reasoning_effort: Optional reasoning effort level ("max", "high", or None).

    Returns:
        Encoded string for this message.
    """
        assert 0 <= index < len(messages)
        assert thinking_mode in ['chat', 'thinking'], f'Invalid thinking_mode `{thinking_mode}`'
        prompt = ''
        msg = messages[index]
        last_user_idx = _last_user_index
        role = msg.get('role')
        content = msg.get('content')
        tools = msg.get('tools')
        response_format = msg.get('response_format')
        tool_calls = msg.get('tool_calls')
        reasoning_content = msg.get('reasoning_content')
        wo_eos = msg.get('wo_eos', False)
        if tools:
            tools = _upstream.tools_from_openai_format(tools)
        if tool_calls:
            tool_calls = _upstream.tool_calls_from_openai_format(tool_calls)
        assert reasoning_effort in ['max', None, 'high'], f'Invalid reasoning effort: {reasoning_effort}'
        if index == 0 and thinking_mode == 'thinking' and (reasoning_effort == 'max'):
            prompt += _upstream.REASONING_EFFORT_MAX
        if role == 'system':
            prompt += _upstream.system_msg_template.format(content=content or '')
            if tools:
                prompt += '\n\n' + _upstream.render_tools(tools)
            if response_format:
                prompt += '\n\n' + _upstream.response_format_template.format(schema=_upstream.to_json(response_format))
        elif role == 'developer':
            assert content, f'Invalid message for role `{role}`: {msg}'
            content_developer = _upstream.USER_SP_TOKEN
            content_developer += content
            if tools:
                content_developer += '\n\n' + _upstream.render_tools(tools)
            if response_format:
                content_developer += '\n\n' + _upstream.response_format_template.format(schema=_upstream.to_json(response_format))
            prompt += _upstream.user_msg_template.format(content=content_developer)
        elif role == 'user':
            prompt += _upstream.USER_SP_TOKEN
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
                                    text_parts.append(f"[Unsupported {b.get('type')}]")
                            tool_content = '\n\n'.join(text_parts)
                        parts.append(_upstream.tool_output_template.format(content=tool_content))
                    else:
                        parts.append(f'[Unsupported {block_type}]')
                prompt += '\n\n'.join(parts)
            else:
                prompt += content or ''
        elif role == 'latest_reminder':
            prompt += _upstream.LATEST_REMINDER_SP_TOKEN + _upstream.latest_reminder_msg_template.format(content=content)
        elif role == 'tool':
            raise NotImplementedError('deepseek_v4 merges tool messages into user; please preprocess with merge_tool_messages()')
        elif role == 'assistant':
            thinking_part = ''
            tc_content = ''
            if tool_calls:
                tc_list = [_upstream.tool_call_template.format(dsml_token=_upstream.dsml_token, name=tc.get('name'), arguments=_upstream.encode_arguments_to_dsml(tc)) for tc in tool_calls]
                tc_content += '\n\n' + _upstream.tool_calls_template.format(dsml_token=_upstream.dsml_token, tool_calls='\n'.join(tc_list), tc_block_name=_upstream.tool_calls_block_name)
            summary_content = content or ''
            rc = reasoning_content or ''
            prev_has_task = index - 1 >= 0 and messages[index - 1].get('task') is not None
            if thinking_mode == 'thinking' and (not prev_has_task):
                if not drop_thinking or index > last_user_idx:
                    thinking_part = _upstream.thinking_template.format(reasoning_content=rc) + _upstream.thinking_end_token
                else:
                    thinking_part = ''
            if wo_eos:
                prompt += _upstream.assistant_msg_wo_eos_template.format(reasoning=thinking_part, content=summary_content, tool_calls=tc_content)
            else:
                prompt += _upstream.assistant_msg_template.format(reasoning=thinking_part, content=summary_content, tool_calls=tc_content)
        else:
            raise NotImplementedError(f'Unknown role: {role}')
        if index + 1 < len(messages) and messages[index + 1].get('role') not in ['assistant', 'latest_reminder']:
            return prompt
        task = messages[index].get('task')
        if task is not None:
            assert task in _upstream.VALID_TASKS, f"Invalid task: '{task}'. Valid tasks are: {list(_upstream.VALID_TASKS)}"
            task_sp_token = _upstream.DS_TASK_SP_TOKENS[task]
            if task != 'action':
                prompt += task_sp_token
            else:
                prompt += _upstream.ASSISTANT_SP_TOKEN
                prompt += _upstream.thinking_end_token if thinking_mode != 'thinking' else _upstream.thinking_start_token
                prompt += task_sp_token
        elif messages[index].get('role') in ['user', 'developer']:
            prompt += _upstream.ASSISTANT_SP_TOKEN
            if not drop_thinking and thinking_mode == 'thinking':
                prompt += _upstream.thinking_start_token
            elif drop_thinking and thinking_mode == 'thinking' and (index >= last_user_idx):
                prompt += _upstream.thinking_start_token
            else:
                prompt += _upstream.thinking_end_token
        return prompt

    def encode_messages(messages: _upstream.List[_upstream.Dict[str, _upstream.Any]], thinking_mode: str, context: _upstream.Optional[_upstream.List[_upstream.Dict[str, _upstream.Any]]]=None, drop_thinking: bool=True, add_default_bos_token: bool=True, reasoning_effort: _upstream.Optional[str]=None) -> str:
        """
    Encode a list of messages into the DeepSeek-V4 prompt format.

    This is the main entry point for encoding conversations. It handles:
    - BOS token insertion
    - Thinking mode with optional reasoning content dropping
    - Tool message merging into user messages
    - Multi-turn conversation context

    Args:
        messages: List of message dicts to encode.
        thinking_mode: Either "chat" or "thinking".
        context: Optional preceding context messages (already encoded prefix).
        drop_thinking: If True, drop reasoning_content from earlier assistant turns
                      (only keep reasoning for messages after the last user message).
        add_default_bos_token: Whether to prepend BOS token at conversation start.
        reasoning_effort: Optional reasoning effort level ("max", "high", or None).

    Returns:
        The encoded prompt string.
    """
        context = context if context else []
        messages = _upstream.merge_tool_messages(messages)
        messages = _upstream.sort_tool_results_by_call_order(context + messages)[len(context):]
        if context:
            context = _upstream.merge_tool_messages(context)
            context = _upstream.sort_tool_results_by_call_order(context)
        full_messages = context + messages
        prompt = _upstream.bos_token if add_default_bos_token and len(context) == 0 else ''
        effective_drop_thinking = drop_thinking
        if any((m.get('tools') for m in full_messages)):
            effective_drop_thinking = False
        if thinking_mode == 'thinking' and effective_drop_thinking:
            full_messages = _upstream._drop_thinking_messages(full_messages)
            num_to_render = len(full_messages) - len(_upstream._drop_thinking_messages(context))
            context_len = len(full_messages) - num_to_render
        else:
            num_to_render = len(messages)
            context_len = len(context)
        _renderer, _render_kwargs = (_upstream.render_message, {})
        if _eligible(full_messages, num_to_render):
            _renderer = _render_message
            _render_kwargs = {'_last_user_index': _upstream.find_last_user_index(full_messages)}
        for idx in range(num_to_render):
            prompt += _renderer(idx + context_len, full_messages, thinking_mode=thinking_mode, drop_thinking=effective_drop_thinking, reasoning_effort=reasoning_effort, **_render_kwargs)
        return prompt
    return encode_messages


# Both eligibility checks are outside message rendering. The canonical caller
# selects copied inert data; upstream preprocessing changes structure, so the
# normalized size/roles/probe budget are checked again before index reuse.
encode_long_messages = _create_encoder(_original)
