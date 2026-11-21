"""DeepSeek-V4.1-Flash (``deepseek_v41``) chat prompts through DeepSeek's own V4.1 encoder.

V4.1 differs from V4 (prompts.py, unchanged): a leading system message and the thinking-mode
"Reasoning Effort: N (range 1-100, ...)" line open with <｜System｜>; a later system message is
its own turn; DSML tags are "<｜DSML｜ calls>" / " invoke" / " parameter". Efforts follow
1Cat-vLLM's ``deepseek_v41`` tokenizer: low/high/max or an int 1-100 as the encoder takes them,
medium 75, xhigh 100, none turns thinking off, unset is the encoder's default (high, 75).
Tools join the first system message (one is added if needed), as DeepSeek's ``load_cases`` and
the V4 wrapper attach them. Text and tool results only: this lane serves no images.
"""

from __future__ import annotations

import copy
from typing import Any

from .prompts import DeepSeekTokenizer
from .vendor.encoding_dsv41 import ASSISTANT_SP_TOKEN as ASSISTANT
from .vendor.encoding_dsv41 import encode_messages

EFFORTS = {"low": "low", "medium": 75, "high": "high", "xhigh": 100, "max": "max"}
SOURCE = "DeepSeek-V4.1 native Python encoder 13231c59"


def message_points(token_id: Any) -> Any:
    """Where a V4.1 prompt's token ids end a message (``token_id``: the tokenizer's control-token lookup) ->
    ``points(ids)``: the sorted positions p in (0, len(ids)) at which the encoder's framing starts the next one.

      * a ``<｜User｜>`` (a user turn or a group of tool results), a later ``<｜System｜>`` or a
        ``<｜latest_reminder｜>`` at p: the system prompt, the assistant reply (thinking and DSML calls through its
        EOS) or the message before it ends at p;
      * an ``<｜Assistant｜>`` at a, with its ``<think>`` / ``</think>`` at a + 1: the user turn and its generation
        header end at a + 2 (else a + 1), where the reply starts; a prompt that ends there (the generation prompt)
        ends at this point, so its next turn (which re-renders that header, thinking kept or not) resumes on it.

    Request text never encodes as a control token (``literal_prompts_v41``), so these ids are the encoder's own
    framing. Every point is a function of the ids before it and the id at it, never of the ids after it."""

    import numpy as np

    from .vendor import encoding_dsv41 as native

    def need(text: str) -> int:
        found = token_id(text)
        if found is None:
            raise ValueError(f"the tokenizer has no {text} control token")
        return int(found)

    openers = np.array([need(native.USER_SP_TOKEN), need(native.SYSTEM_SP_TOKEN),
                        need(native.LATEST_REMINDER_SP_TOKEN)], dtype=np.int64)
    assistant = need(ASSISTANT)
    thinks = np.array([need(native.thinking_start_token), need(native.thinking_end_token)], dtype=np.int64)

    def points(ids: Any) -> list[int]:
        arr = np.asarray(ids, dtype=np.int64)
        starts = np.flatnonzero(np.isin(arr, openers))
        heads = np.flatnonzero(arr[:-1] == assistant)
        ends = heads + 1 + np.isin(arr[heads + 1], thinks)
        found = np.union1d(starts, ends)
        return [int(p) for p in found if 0 < p < len(arr)]

    return points


def _text_only(value: Any) -> None:
    if isinstance(value, list):
        for block in value:
            if not isinstance(block, dict) or block.get("type") not in ("text", "tool_result"):
                raise ValueError("DeepSeek-V4.1 prompts admit text and tool results only (no images)")
            _text_only(block.get("content"))


def _prepare(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, thinking: bool,
             reasoning_effort: Any) -> tuple[list[dict[str, Any]], bool, Any]:
    """Copied messages with tools attached, the thinking switch and the encoder's effort."""

    if thinking and reasoning_effort is not None:
        if reasoning_effort == "none":
            thinking, reasoning_effort = False, None
        elif type(reasoning_effort) is int:
            if not 1 <= reasoning_effort <= 100:
                raise ValueError(f"DeepSeek-V4.1 reasoning_effort {reasoning_effort} is outside 1-100")
        elif isinstance(reasoning_effort, str) and str.__str__(reasoning_effort) in EFFORTS:
            reasoning_effort = EFFORTS[str.__str__(reasoning_effort)]
        else:
            raise ValueError(f"DeepSeek-V4.1 reasoning_effort must be none, low, medium, high, xhigh, max or an "
                             f"integer 1-100, not {reasoning_effort!r}")
    effort = reasoning_effort if thinking else None
    msgs = [copy.deepcopy(m) for m in messages]
    for m in msgs:
        _text_only(m.get("content"))
        _text_only(m.get("content_blocks"))
        if m.get("role") == "assistant" and m.get("content") is None:
            m["content"] = ""
    if tools:
        if not msgs or msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = [t if "function" in t else {"type": "function", "function": t} for t in tools]
    return msgs, thinking, effort


def render(messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, thinking: bool = False,
           reasoning_effort: Any = None) -> str:
    """The V4.1 prompt text for OpenAI-style messages."""

    msgs, thinking, effort = _prepare(messages, tools, thinking, reasoning_effort)
    return encode_messages(msgs, thinking_mode="thinking" if thinking else "chat", reasoning_effort=effort)


def _generation(text: Any, messages: list[dict[str, Any]], thinking: bool, add_generation_prompt: bool,
                suffix: Any) -> Any:
    """The encoder adds the reply prefix after a last user/system turn itself; without a generation prompt it
    goes, and after a last assistant turn a generation prompt adds it (as the V4 wrapper does)."""

    prefix = ASSISTANT + ("<think>" if thinking else "</think>")
    last = messages[-1].get("role") if messages else None
    if add_generation_prompt and last == "assistant":
        text += suffix(prefix)
    elif not add_generation_prompt and last != "assistant" and text.endswith(prefix):
        text = text[:-len(prefix)]
    return text


def render_literals(messages: list[dict[str, Any]], protection: Any, *, tools: list[dict[str, Any]] | None = None,
                    thinking: bool = False, reasoning_effort: Any = None, add_generation_prompt: bool = True,
                    max_bytes: int | None = None) -> Any:
    """The V4.1 prompt as a ``RenderedPrompt``: encoder-written controls trusted, request strings data
    (``protection``: the tokenizer's ``PromptLiterals``; ``max_bytes`` bounds the request values it reads)."""
    from tensorfold.server.prompt_literals import RenderedPrompt
    from tensorfold.server.prompt_origin import trusted
    from .literal_prompts_v41 import encode

    msgs, thinking, effort = _prepare(messages, tools, thinking, reasoning_effort)
    values, _ = protection.protect(msgs, **({"max_bytes": max_bytes} if max_bytes is not None else {}))
    text = encode(values, thinking_mode="thinking" if thinking else "chat", reasoning_effort=effort)
    return RenderedPrompt(_generation(text, messages, thinking, add_generation_prompt, trusted), protection.literals)


class DeepSeekV41Tokenizer(DeepSeekTokenizer):
    """The checkpoint's tokenizer with ``apply_chat_template`` rendered by DeepSeek's V4.1 encoder."""

    def render_prompt_literals(self, messages, tools=None, add_generation_prompt=True,
                               enable_thinking=False, thinking_mode=None, reasoning_effort=None, renderer=None, **_):
        """Keep Python encoder authority while tracking semantic input origins."""
        import json
        from tensorfold.server.prompt_literals import PromptEncodingError, PromptLiterals, native_tokenizer
        from .literal_prompts_v41 import reserved_tokens

        backend = native_tokenizer(self._inner)
        if backend is None:
            raise PromptEncodingError("DeepSeek tokenizer does not expose ordinary literal encoding")
        thinking = thinking_mode == "thinking" if thinking_mode is not None else bool(enable_thinking)
        source = [SOURCE]
        reserved = reserved_tokens(backend)
        protection = (renderer.protection(backend, source, reserved) if renderer is not None else
                      PromptLiterals(json.loads(backend.to_str()), source, reserved))
        return render_literals(messages, protection, tools=tools, thinking=thinking, reasoning_effort=reasoning_effort,
                               add_generation_prompt=add_generation_prompt)

    def apply_chat_template(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                            add_generation_prompt: bool = True, tokenize: bool = True, enable_thinking: bool = False,
                            thinking_mode: str | None = None, reasoning_effort: Any = None, **_: Any) -> Any:
        thinking = thinking_mode == "thinking" if thinking_mode is not None else bool(enable_thinking)
        msgs, thinking, effort = _prepare(messages, tools, thinking, reasoning_effort)
        text = encode_messages(msgs, thinking_mode="thinking" if thinking else "chat", reasoning_effort=effort)
        text = _generation(text, messages, thinking, add_generation_prompt, str)
        if not tokenize:
            return text
        return list(self._inner.encode(text, add_special_tokens=False))
