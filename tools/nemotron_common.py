"""Shared tokenizer preparation for the Nemotron measurement tools."""

from typing import Any


def chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    """The chat template's token ids for one user message, thinking off (a flat list under transformers 4 or 5)."""

    messages = [{"role": "user", "content": prompt}]
    kwargs = {"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}
    try:
        out = tokenizer.apply_chat_template(messages, return_dict=False, **kwargs)
    except TypeError:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]

