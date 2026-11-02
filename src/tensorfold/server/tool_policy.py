from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.engine.grammar import SPECS


def visible_tool_prose(text: str, *, calls: bool | None = None) -> str:
    """Withhold ambiguous whitespace; known no-call output remains exact data."""
    return '' if calls is not False and text.isspace() else text


class _ToolTextFilter:
    """Quote-aware cumulative presentation for one admitted bounded reply.

    Retains the admitted raw reply and its visible prefix (each bounded by
    that request's output allowance), not an
    independent/unbounded stream. One lexical interpretation is shared by
    incremental and final tool presentation instead of repairing emitted bytes.
    """
    def __init__(self):
        self.text = ''
        self.shown = ''

    def feed(self, text: str) -> str:
        from tensorfold.server.text import hide_tool_calls
        self.text += text
        visible = hide_tool_calls(self.text, finished=False)
        if not visible.startswith(self.shown):
            raise ValueError('tool presentation regressed after committed output')
        fresh = visible[len(self.shown):]
        self.shown = visible
        return fresh

    def finish(self) -> str:
        from tensorfold.server.text import hide_tool_calls
        visible = hide_tool_calls(self.text, finished=True)
        if not visible.startswith(self.shown):
            raise ValueError('final tool presentation regressed after committed output')
        tail = visible[len(self.shown):]
        self.text, self.shown = '', ''
        return tail


class ToolCallPolicy:
    def __init__(self, body: dict[str, Any]):
        parallel = body.get("parallel_tool_calls")
        parallel = True if parallel is None else parallel
        if type(parallel) is not bool:
            raise RequestError("parallel_tool_calls must be a boolean")
        self.single = not parallel
        self._text = _ToolTextFilter() if self.single else None
        self._literal_data = any(body.get(key) is not None for key in SPECS)
        self._prose_pending = []
        self._prose_started = False
        self._prose_calls = None

    @property
    def max_calls(self) -> int | None:
        return 1 if self.single else None

    def limit(self, calls):
        return calls[:1] if self.single and calls else calls

    def _prose(self, text):
        if self._prose_started or not text:
            return text
        if text.isspace():
            self._prose_pending.append(text)
            return ''
        self._prose_pending.append(text)
        result = ''.join(self._prose_pending)
        self._prose_pending = []
        self._prose_started = True
        return result

    def delta(self, delta):
        if self._literal_data:
            return delta
        if isinstance(delta, str):
            return self._prose(self._text.feed(delta) if self.single else delta)
        if isinstance(delta, dict):
            if not self.single and (self._prose_started or not isinstance(delta.get('content'), str)):
                return delta
            result = {key: value for key, value in delta.items() if not self.single or key != 'tool_calls'}
            if isinstance(result.get('content'), str):
                text = result.pop('content')
                content = self._prose(self._text.feed(text) if self.single else text)
                if content:
                    result['content'] = content
            return result
        return delta

    def flush(self) -> str:
        if self._literal_data:
            return ''
        tail = self._prose(self._text.finish()) if self.single else ''
        pending = ''.join(self._prose_pending) if self._prose_calls is not True else ''
        self._prose_pending = []
        return tail + pending

    def content(self, text: str, *, finished: bool = True) -> str:
        if self.single and not self._literal_data:
            filtered = _ToolTextFilter()
            return filtered.feed(text) + (filtered.finish() if finished else "")
        return text

    def finish(self, reply, tools, parse):
        """The reply's calls, read by ``parse`` (the lenient parsers), or by the reply's ``strict_calls``
        (``server.strict_tools.Plan``) when its grammar wrote them."""

        if not tools or self._literal_data:
            return reply
        text, plan = str(reply.get("content") or ""), reply.get("strict_calls")
        # Direct complete-reply callers retain that contract; an engine reply
        # supplies actual terminal authority so a cap/user stop cannot resolve
        # ambiguous raw quoted values into calls.
        finished = reply.get("tool_parse_finished", True)
        content, calls = (plan.parse(text, max_calls=self.max_calls) if plan is not None
                          else parse(text, tools, max_calls=self.max_calls, finished=finished))
        calls = calls or reply.get("tool_calls")
        self._prose_calls = bool(calls)
        if not calls:
            return {**reply, "content": self.content(content)} if self.single else reply
        result = {**reply, "content": visible_tool_prose(self.content(content), calls=True),
                  "tool_calls": self.limit(calls)}
        if plan is not None or reply.get("tool_calls_complete", True):
            result["finish_reason"] = "tool_calls"
        if self.single:
            result["tool_calls_streamed"] = False
        return result
