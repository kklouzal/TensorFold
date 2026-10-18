"""The server's thinking notes: thinking is on at startup, a reply that ran out of tokens while thinking, and a reply
that ended at an end token while its think block was still open (a model failure; ``failure_capture`` records it)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OFF = 'chat_template_kwargs {"enable_thinking": false}'


def template_thinks(model_dir: Path) -> bool:
    """Whether the checkpoint's chat template reads ``enable_thinking`` (its default decides a silent request)."""

    template = Path(model_dir) / "chat_template.jinja"
    if template.is_file() and "enable_thinking" in template.read_text(errors="ignore"):
        return True
    config = Path(model_dir) / "tokenizer_config.json"
    try:
        return "enable_thinking" in json.dumps(json.loads(config.read_text()).get("chat_template") or "")
    except (OSError, ValueError):
        return False


def startup(model_dir: Path, thinking: bool, *, template_source: str | None = None) -> str | None:
    """The startup line when replies think by default, or None."""

    from tensorfold.server.request_options import fixed_thinking
    from jinja2 import TemplateError

    try:
        if template_source is not None:
            template = template_source
        else:
            source = Path(model_dir) / "chat_template.jinja"
            template = (source.read_text(errors="ignore") if source.is_file() else
                        json.loads((Path(model_dir) / "tokenizer_config.json").read_text()).get("chat_template"))
        fixed = fixed_thinking(template) if isinstance(template, str) else None
    except (OSError, ValueError, TemplateError):
        return None                    # notes are observational; the captured runtime template owns validation
    if fixed is True:
        return ("[tensorfold] thinking on (fixed by the checkpoint's chat template): replies reason in "
                "reasoning_content before the answer in content, and max_tokens counts both. This template "
                "overrides enable_thinking; disabling thinking requires selecting a different chat template.")

    if not thinking or "enable_thinking" not in (template if isinstance(template, str) else json.dumps(template)):
        return None
    return ("[tensorfold] thinking on (server default): replies reason in reasoning_content before the "
            f"answer in content, and max_tokens counts both. --no-thinking turns it off; a request can send {OFF}")


def unanswered(finish: str, thinking: bool, content: Any, calls: Any = None, *, can_disable: bool = True) -> str | None:
    """A warning when a reply ended at max_tokens with nothing visible: its tokens all went to thinking."""

    if finish != "length" or not thinking or calls or str(content or "").strip():
        return None
    action = f"raise max_tokens, or send {OFF} (server: --no-thinking)" if can_disable else (
        "raise max_tokens; the selected chat template fixes thinking on")
    return ("[tensorfold] warning: a reply reached max_tokens while still thinking, so its content is empty and its "
            f"text is all in reasoning_content; {action}")


def ended_thinking(finish: str, thinking: bool, content: Any, calls: Any, reasoning: Any, end_token: bool,
                   closed: bool) -> bool:
    """Whether a reply ended at an end token (``end_token``) inside its think block: thinking on, finish "stop", no
    answer and no call, reasoning written, and no closing think marker (``closed``) in the reply."""

    return (finish == "stop" and thinking and end_token and not closed and not calls
            and not str(content or "").strip() and bool(str(reasoning or "").strip()))


def ended_thinking_note(request_id: str, prompt: int, reply: int, token: int, text: str | None) -> str:
    """The warning line for ``ended_thinking``: the request, its sizes and the end token that closed it."""

    return (f"[tensorfold] warning: {request_id} ended at end token {text or '?'} (id {token}) inside its think "
            f"block, so it has no answer: prompt={prompt} reply={reply} tokens")


__all__ = ["ended_thinking", "ended_thinking_note", "startup", "template_thinks", "unanswered"]
