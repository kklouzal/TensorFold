"""Message-start resume points for CUDA engines, found by the Mac planner's rule through the CUDA chat template."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

MIN_GAP = 256            # a snapshot this close to the prompt start or to the previous snapshot saves too little


class TemplateTokens:
    """The tokenizer surface ``message_markers`` reads, over a CUDA server's tokenizer and chat template."""

    def __init__(self, tok: Any, template: Any, *, tools=False) -> None:
        self.tok, self.template, self.tools = tok, template, tools
        self.added_tokens_decoder = tok.get_added_tokens_decoder()
        self.all_special_ids = [int(i) for i, t in self.added_tokens_decoder.items() if t.special]

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = True,
                            add_generation_prompt: bool = False, **kwargs: Any):
        tools = kwargs.pop("tools", [{"type": "function", "function": {"name": "tf_template_probe",
                           "parameters": {"type": "object", "properties": {}}}}] if self.tools else None)
        text = self.template.selected(tools=tools).template.render(**self.template.specials, messages=messages, tools=tools,
                                             add_generation_prompt=add_generation_prompt, **kwargs)
        return self.tok.encode(text, add_special_tokens=False).ids if tokenize else text


def snapshot_points(openers: Sequence[int], assistant: Sequence[int], *,
                    turn: bool = False) -> Callable[[Sequence[int]], list[int]]:
    """Where a prefill keeps states: the second message's start (a shared system block) and the last assistant start;
    with ``turn`` also the last message that is not the assistant's (the user's turn, a tool result, a late system note:
    a client that rewrites it next time, as Harness does its per-run status block, still resumes before it).

    A point ends a prefill chunk unless the engine's passes hold several (Flash Next's concurrent decoder, up to
    ``prompt_plan.CUTS`` a piece), so only that decoder asks for ``turn``: elsewhere it adds a chunk to every turn."""

    from tensorfold.engine.prefill_plan import PrefillPlan

    plan = PrefillPlan(openers=openers, assistant=assistant, min_chunk=max(MIN_GAP, len(assistant)))

    def points(ids: Sequence[int]) -> list[int]:
        arr = np.asarray(ids, dtype=np.int64)
        found = plan.points(arr)                     # assistant starts and the second message's
        starts = np.flatnonzero(np.isin(arr, plan.openers)) if plan.openers else []
        wanted = ([int(starts[1])] if len(starts) > 1 else []) + ([found[-1]] if found else [])
        if turn:
            taken = set(found)
            wanted += [int(p) for p in starts if int(p) not in taken][-1:]
        out: list[int] = []
        for p in sorted(set(wanted)):
            if MIN_GAP <= p < len(arr) and (not out or p - out[-1] >= MIN_GAP):
                out.append(p)
        return out

    return points


def resume_points(model_dir: str | Path, *, turn: bool = False, template=None) -> Callable[[Sequence[int]], list[int]] | None:
    """``snapshot_points`` (``turn`` as there) for this checkpoint's chat template, or None when it marks no message
    starts."""

    from tensorfold.engine.prefill_plan import message_markers
    from tensorfold.token_metadata import load_tokenizer

    from .chat_template import prepare_chat_template

    model_dir = Path(model_dir)
    if not (model_dir / "tokenizer.json").is_file():
        return None
    try:
        template = template if template is not None else prepare_chat_template(model_dir)
        if template is None:
            return None
        tok = load_tokenizer(model_dir)
        openers, assistant = message_markers(TemplateTokens(tok, template))
        other = message_markers(TemplateTokens(tok, template, tools=True))
        if (openers, assistant) != other:
            return None  # one planner cannot claim a single message layout for differing authoritative templates
    except (OSError, ValueError, KeyError):
        return None
    return snapshot_points(openers, assistant, turn=turn) if openers or assistant else None
