"""GLM-5.3's prompts, the same on both servers: earlier turns keep their reasoning, and with thinking off no effort
line and no opened think block."""

from __future__ import annotations

import os
from typing import Any

from tensorfold.server.prompt_literals import MAX_RENDER_BYTES, PromptEncodingError, RenderedPrompt
from tensorfold.server.prompt_origin import origin_concat, trusted

EFFORT_LINE = "<|system|>Reasoning Effort: Max"
EFFORT_PREFIX = "<|system|>Reasoning Effort: "
EFFORT_LINES = (EFFORT_LINE, EFFORT_PREFIX+"Low", EFFORT_PREFIX+"High")
_EFFORT_FOLLOWERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")
OPENED = "<|assistant|><think>"
CLEAR_THINKING = "TF_GLM_CLEAR_THINKING"


def clear_thinking() -> bool:
    """The template's ``clear_thinking`` when a request leaves it unset: False unless ``TF_GLM_CLEAR_THINKING=1``.

    zai-org's GLM-5.3-Flash template defaults it to false, keeping every assistant turn's reasoning in the prompt
    (its model card: pass ``clear_thinking=true`` for chat). Checkpoints that carry the template's first revision
    (``Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`` among them) clear it unless told otherwise, so the reasoning before the
    last user message renders as ``<think></think>``: each new user message changes the earlier turns' tokens, and
    an agent's whole previous tool loop is prefilled again. Passing false renders those checkpoints as the current
    template does; on the current template it changes nothing."""

    value = os.environ.get(CLEAR_THINKING, "0").strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{CLEAR_THINKING}={value!r}: expected 0 or 1")
    return value == "1"


def thinking_off(text: str) -> str:
    """The checkpoint text as the thinking-off template renders it: no effort line, an empty think block."""

    text = text.replace(EFFORT_LINE, "", 1)
    return text + "</think>" if text.endswith(OPENED) else text


def thinking_off_prompt(rendered, *, max_render_bytes=MAX_RENDER_BYTES, render_credit=None):
    """Transform native template controls while retaining request origins and owners."""
    origin = rendered.origin
    at, removed = origin.find(EFFORT_PREFIX), 0
    while at >= 0:
        for line in EFFORT_LINES:
            if origin.startswith(line, at) and not any(origin._mask[at:at + len(line)]):
                end = at + len(line)
                # A source label ends at whitespace/EOF or a complete trusted
                # GLM role delimiter. A transition into request data also ends
                # its authority; that data remains literal, even without space.
                boundary = (end == len(origin) or bool(origin._mask[end]) or origin[end].isspace()
                            or any(origin.startswith(role, end)
                                   and not any(origin._mask[end:end + len(role)])
                                   for role in _EFFORT_FOLLOWERS))
                if boundary:
                    removed = len(line)
                    break
        if removed:
            break  # Exactly the earliest fully trusted known header, never request data.
        at = origin.find(EFFORT_PREFIX, at + 1)
    close = origin.endswith(OPENED) and not any(origin._mask[-len(OPENED):])
    if at < 0 and not close:
        return rendered
    remaining = len(origin) - removed
    final = remaining + (len("</think>") if close else 0)
    if render_credit is not None:
        # Both prompts retain origins and opaque media views. Each unsafe run
        # can need its own marker/value/slot entry in the new view, whose text
        # can be much longer (and wider) than the logical prompt.
        runs = origin._mask.count(b"\0\1") + int(origin._mask.startswith(b"\1"))
        root_chars = len(rendered.control_literals[-1][1]) if rendered.control_literals else len("tfl_")
        marker_chars = root_chars + len("data_") + len(str(len(origin))) + len(str(max(0, runs - 1))) + 2
        media_chars = final + runs * marker_chars
        render_credit(len(origin) + len(rendered.encoded_text) + final + media_chars,
                      len(rendered.literals) + runs + 3)
    if close and (len(origin.encode("utf-8")) - removed
                  + len("</think>") > max_render_bytes):
        raise PromptEncodingError("rendered prompt exceeds the chat-render byte budget")
    parts = [origin] if at < 0 else [origin[:at], origin[at + removed:]]
    if close:
        parts.append(trusted("</think>"))
    result = RenderedPrompt(origin_concat(parts), rendered.literals, control_literals=rendered.control_literals)
    object.__setattr__(result, "_renderer", rendered._renderer)
    object.__setattr__(result, "_ordinary_codec", rendered._ordinary_codec)
    return result


class GlmTokenizer:
    """The checkpoint's tokenizer (the Mac's), its chat template read through ``thinking_off`` when thinking is off
    and given ``clear_thinking`` (``clear_thinking()``) when the call leaves it unset."""

    def __init__(self, inner: Any, clear: bool | None = None) -> None:
        self._inner = inner
        self._clear = clear_thinking() if clear is None else clear

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def render_prompt_literals(self, messages, **kwargs):
        """Preserve the native wrapper's post-render transformation and origins."""
        from tensorfold.server.prompt_literals import render_hf_prompt

        kwargs.setdefault("clear_thinking", self._clear)
        rendered = render_hf_prompt(self._inner, messages, **kwargs)
        if kwargs.get("enable_thinking", True) is not False:
            return rendered
        return thinking_off_prompt(rendered)

    def apply_chat_template(self, messages: Any, *args: Any, tokenize: bool = True, **kwargs: Any) -> Any:
        kwargs.setdefault("clear_thinking", self._clear)
        if kwargs.get("enable_thinking", True) is not False:
            return self._inner.apply_chat_template(messages, *args, tokenize=tokenize, **kwargs)
        text = thinking_off(self._inner.apply_chat_template(messages, *args, tokenize=False, **kwargs))
        return list(self._inner.encode(text, add_special_tokens=False)) if tokenize else text
