"""Origin boundary for the owned DeepSeek Python encoder; vendor semantics stay authoritative.

Only fixed native templates and the three flattening helper boundaries are
adapted. Request strings retain their logical values through preprocessing;
JSON serialized request values remain wholly data. No vendor globals mutate.
"""
from __future__ import annotations

import json
from string import Formatter

from tensorfold.server.prompt_origin import OriginText, as_origin, origin_concat, trusted
from . import rendering
from .vendor import encoding_dsv4 as native


class _Template(str):
    def format(self, **values):
        parts = []
        for literal, name, spec, conversion in Formatter().parse(self):
            parts.append(trusted(literal))
            if name is not None:
                # All pinned encoder templates use simple named fields only.
                if spec or conversion or name not in values:
                    raise ValueError('unsupported native DeepSeek template format field')
                parts.append(as_origin(values[name]))
        return origin_concat(parts)


class _NativeOrigin:
    """A private helper view, never installed into the native module namespace."""
    def __getattr__(self, name):
        value = getattr(native, name)
        if isinstance(value, str):
            return _Template(value) if '{' in value else trusted(value)
        if name == 'DS_TASK_SP_TOKENS':
            return {key: trusted(value) for key, value in value.items()}
        return value

    @staticmethod
    def join(separator, values):
        return trusted(separator).join(values)

    @staticmethod
    def unsupported(value):
        return origin_concat((trusted('[Unsupported '), as_origin(value), trusted(']')))

    @staticmethod
    def to_json(value):
        # Match the native serialization/error policy; serialized semantic data
        # cannot authorize any model control, including decoded JSON escapes.
        return OriginText(native.to_json(value))

    def render_tools(self, tools):
        return _Template(native.TOOLS_TEMPLATE).format(
            tool_schemas=self.join('\n', [self.to_json(tool) for tool in tools]),
            dsml_token=self.dsml_token, thinking_start_token=self.thinking_start_token,
            thinking_end_token=self.thinking_end_token)

    def encode_arguments_to_dsml(self, tool_call):
        try:
            arguments = json.loads(tool_call['arguments'])
        except Exception:
            arguments = {'arguments': tool_call['arguments']}
        parts = []
        template = _Template('<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>')
        for key, value in arguments.items():
            parts.append(template.format(dsml_token=self.dsml_token, key=OriginText(key),
                                         is_str=trusted('true' if isinstance(value, str) else 'false'),
                                         value=OriginText(value) if isinstance(value, str) else self.to_json(value)))
        return self.join('\n', parts)


def encode(messages, *, thinking_mode, reasoning_effort):
    adapter = _NativeOrigin()
    # Reuse the maintained full native rendering body for every row. Its
    # original default/selected specialization remains unchanged for other users.
    encoder = rendering._create_encoder(adapter, origin=adapter)
    return encoder(messages, thinking_mode=thinking_mode, reasoning_effort=reasoning_effort)


def reserved_tokens(backend):
    values = [native.bos_token, native.eos_token, native.thinking_start_token, native.thinking_end_token,
              native.USER_SP_TOKEN, native.ASSISTANT_SP_TOKEN, native.LATEST_REMINDER_SP_TOKEN,
              *native.DS_TASK_SP_TOKENS.values(), native.dsml_token]
    return [value for value in values if backend.token_to_id(value) is not None]
