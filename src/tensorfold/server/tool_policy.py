from typing import Any

from tensorfold.server.errors import RequestError


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

    @property
    def max_calls(self) -> int | None:
        return 1 if self.single else None

    def limit(self, calls):
        return calls[:1] if self.single and calls else calls

    def delta(self, delta):
        if self.single:
            if isinstance(delta, str):
                return self._text.feed(delta)
            if isinstance(delta, dict):
                result = {k: v for k, v in delta.items() if k != "tool_calls"}
                if isinstance(result.get("content"), str):
                    content = self._text.feed(result.pop("content"))
                    if content:
                        result["content"] = content
                return result
        return delta

    def flush(self) -> str:
        return self._text.finish() if self.single else ""

    def content(self, text: str, *, finished: bool = True) -> str:
        if self.single:
            filtered = _ToolTextFilter()
            return filtered.feed(text) + (filtered.finish() if finished else "")
        return text

    def finish(self, reply, tools, parse):
        """The reply's calls, read by ``parse`` (the lenient parsers), or by the reply's ``strict_calls``
        (``server.strict_tools.Plan``) when its grammar wrote them."""

        if not tools:
            return reply
        text, plan = str(reply.get("content") or ""), reply.get("strict_calls")
        # Direct complete-reply callers retain that contract; an engine reply
        # supplies actual terminal authority so a cap/user stop cannot resolve
        # ambiguous raw quoted values into calls.
        finished = reply.get("tool_parse_finished", True)
        content, calls = (plan.parse(text, max_calls=self.max_calls) if plan is not None
                          else parse(text, tools, max_calls=self.max_calls, finished=finished))
        calls = calls or reply.get("tool_calls")
        if not calls:
            return {**reply, "content": self.content(content)} if self.single else reply
        result = {**reply, "content": self.content(content), "tool_calls": self.limit(calls)}
        if plan is not None or reply.get("tool_calls_complete", True):
            result["finish_reason"] = "tool_calls"
        if self.single:
            result["tool_calls_streamed"] = False
        return result
