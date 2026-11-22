"""Compile strict tool schemas into the DeepSeek V4.1 DSML envelope."""
from __future__ import annotations

import json
import re
from uuid import uuid4

from tensorfold.engine.grammar import Spec
from tensorfold.markup_lex import DSML41_CALLS
from tensorfold.server import strict_tools as common
from tensorfold.server.errors import RequestError
from tensorfold.server.text import _partial_tag
from tensorfold.server.tools import stream_tool_call_deltas, tool_spec_name


OPEN, CLOSE = DSML41_CALLS
INVOKE = '<｜DSML｜ invoke name="'
INVOKE_END = '</｜DSML｜ invoke>'
PARAM = '<｜DSML｜ parameter name="'
PARAM_END = '</｜DSML｜ parameter>'
_NAME_PATTERN = r'[^ \t\r\n"<>]+'
NAME = re.compile(_NAME_PATTERN)
_QUOTE = re.compile(r'"')
_BLANKS = re.compile(r"[ \t\r\n]*")


def _parameter(name, value, string):
    return common._seq(common._const(f'{PARAM}{name}" string="{string}">'), value, common._const(PARAM_END + "\n"))


def _strict_parameters(parameters, where, remaining):
    reserved = {"$ref", "anyOf", "oneOf", "enum", "const"}
    if reserved & parameters.keys() or parameters.get("type", "object") != "object":
        raise common._refuse(where, "strict tool parameters must be an object schema")
    if parameters.get("additionalProperties", False) is not False:
        raise common._refuse(where, "strict DSML tools require a fixed set of parameter names")
    top = {"type": "object", **{key: val for key, val in parameters.items() if key != "$schema"}}
    schema = common._Schema(top, where)
    schema.check()
    required = set(top.get("required", ()))
    parts = []
    emitted = 0
    for name, node in top.get("properties", {}).items():
        at = f"{where}.properties.{name}"
        if not NAME.fullmatch(name):
            raise common._refuse(at, "this name cannot be written as a DSML parameter")
        types = schema.admits(node)
        alternatives = []
        if "string" in types:
            alternatives.append(_parameter(name, schema.raw(node, at, delimiter=PARAM_END), "true"))
        if types - {"string"}:
            value, copied = schema.json(node, at)
            if emitted + copied > remaining:
                raise common._too_large()
            alternatives.append(_parameter(name, schema.json_format(value, at), "false"))
        if not alternatives:
            raise common._refuse(at, "the parameter schema has no DSML representation")
        part = alternatives[0] if len(alternatives) == 1 else {"type": "or", "elements": alternatives}
        part = part if name in required else {"type": "optional", "content": part}
        emitted += _size(part)
        if emitted > remaining:
            raise common._too_large()
        parts.append(part)
    return common._seq(*parts) if parts else common._const("")


def _size(format):
    return len(json.dumps(format, ensure_ascii=False, separators=(",", ":")).encode())


def _ordinary_parameters():
    name = {"type": "regex", "pattern": _NAME_PATTERN}
    raw = {"type": "any_text", "excludes": [PARAM_END]}
    value = {"type": "json_schema", "json_schema": {}, "style": "json",
             "max_whitespace_cnt": common.BLANKS}
    typed = {"type": "or", "elements": [common._seq(common._const('" string="true">'), raw),
                                        common._seq(common._const('" string="false">'), value)]}
    return {"type": "star", "content": common._seq(common._const(PARAM), name, typed, common._const(PARAM_END + "\n"))}


class Plan:
    """The DSML grammar and reader owned by a strictly constrained reply."""

    opener = OPEN

    def __init__(self, tools, *, required, parallel, offered=None):
        self.parallel = parallel
        self.tools = set()
        tags = []
        indices = {id(tool): i for i, tool in reversed(list(enumerate(offered or ())))}
        size = 0
        for i, tool in enumerate(tools):
            where = f"tools[{indices.get(id(tool), i)}].function"
            name = tool_spec_name(tool)
            if not NAME.fullmatch(name) or name in self.tools:
                raise RequestError(f"{where}.name cannot identify a unique DSML tool")
            parameters = common._function(tool).get("parameters")
            parameters = {} if parameters is None else parameters
            if not isinstance(parameters, dict):
                raise RequestError(f"{where}.parameters must be a JSON schema object")
            strict = common.strict_flag(tool, where)
            content = (_strict_parameters(parameters, f"{where}.parameters", common.MAX_GRAMMAR_BYTES - size)
                       if strict else _ordinary_parameters())
            # DeepSeek's encoder leaves a blank line for an empty parameter list.
            content = common._seq(content, {"type": "optional", "content": common._const("\n")})
            tag = {"type": "tag", "begin": f'{INVOKE}{name}">\n', "content": content, "end": INVOKE_END}
            size += _size(tag)
            if size > common.MAX_GRAMMAR_BYTES:
                raise common._too_large()
            tags.append(tag)
            self.tools.add(name)
        invokes = {"type": "tags_with_separator", "tags": tags, "separator": "\n",
                   "at_least_one": True, "stop_after_first": not parallel}
        block = {"type": "tag", "begin": OPEN + "\n", "content": common._seq(invokes, common._const("\n")), "end": CLOSE}
        if required:
            blanks = {"type": "regex", "pattern": f"[ \\t\\r\\n]{{0,{common.BLANKS}}}"}
            body = common._seq(blanks, block, blanks)
        else:
            body = {"type": "triggered_tags", "triggers": [OPEN], "tags": [block], "stop_after_first": not parallel}
        text = json.dumps({"type": "structural_tag", "format": body}, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode()) > common.MAX_GRAMMAR_BYTES:
            raise common._too_large()
        self.spec = Spec("tools", text, "tools")

    @property
    def thinking_headers(self):
        return tuple(f'{OPEN}\n{INVOKE}{name}">\n' for name in sorted(self.tools))

    def reader(self):
        return Reader(self)

    def parse_finished(self, answer, *, max_calls=None):
        reader = self.reader()
        reader.read(answer, finished=True)
        parsed = reader.calls if max_calls is None else reader.calls[:max_calls]
        calls = [_call(name, arguments) for name, arguments in parsed]
        return "".join(reader.text).strip(), calls or None, not reader.unfinished

    def content(self, answer, *, finished):
        reader = self.reader()
        reader.read(answer, finished=finished)
        return "".join(reader.text)


class _Incomplete(Exception):
    pass


def _take(text, start, literal):
    if text.startswith(literal, start):
        return start + len(literal)
    if len(text) - start < len(literal) and literal.startswith(text[start:]):
        raise _Incomplete
    raise common.CallMarkupError("the reply left its native DSML grammar")


def _quoted_name(text, start):
    final = _QUOTE.search(text, start)
    if final is None:
        raise _Incomplete
    name = text[start:final.start()]
    if not NAME.fullmatch(name):
        raise common.CallMarkupError("the reply has an invalid DSML name")
    return name, final.end()


def _non_json_number(_):
    raise common.CallMarkupError("the reply has a non-JSON DSML value")


def _ignore_number(_):
    """Consume a number's JSON spelling without rounding or applying a digit limit."""
    return None


_JSON = json.JSONDecoder(parse_int=_ignore_number, parse_float=_ignore_number, parse_constant=_non_json_number)


def _call(name, arguments):
    args = ",".join(json.dumps(key, ensure_ascii=False) + ":" + value for key, value in arguments.items())
    return {"id": f"call_{uuid4().hex[:24]}", "type": "function",
            "function": {"name": name, "arguments": "{" + args + "}"}}


def _block(text, start, tools):
    """Read one block. JSON values can contain literal markup inside strings."""
    at = _take(text, start, OPEN + "\n")
    calls = []
    while not text.startswith(CLOSE, at):
        name, at = _quoted_name(text, _take(text, at, INVOKE))
        if name not in tools:
            raise common.CallMarkupError("the reply called an unoffered DSML tool")
        at = _take(text, at, ">\n")
        args = {}
        while not text.startswith(INVOKE_END, at) and not text.startswith("\n" + INVOKE_END, at):
            key, at = _quoted_name(text, _take(text, at, PARAM))
            if key in args:
                raise common.CallMarkupError("the reply repeated a DSML parameter")
            at = _take(text, at, ' string="')
            if text.startswith('true">', at):
                first = at + len('true">')
                at = text.find(PARAM_END, first)
                if at < 0:
                    raise _Incomplete
                value = json.dumps(text[first:at], ensure_ascii=False)
            else:
                first = _BLANKS.match(text, _take(text, at, 'false">')).end()
                try:
                    _, end = _JSON.raw_decode(text, first)
                except json.JSONDecodeError:
                    if text.endswith("\n" + INVOKE_END + "\n" + CLOSE):
                        raise common.CallMarkupError("the reply has malformed JSON in a closed DSML call") from None
                    raise _Incomplete from None
                except RecursionError:
                    raise common.CallMarkupError("the reply has deeply nested JSON in a DSML call") from None
                value = text[first:end]
                at = _BLANKS.match(text, end).end()
                # Before the complete closing tag, even a number that parses can be a partial exponent.
                if not text.startswith(PARAM_END, at) and text.find(PARAM_END, at) < 0:
                    raise _Incomplete
            at = _take(text, at, PARAM_END + "\n")
            args[key] = value
        if text.startswith("\n", at):
            at += 1
        at = _take(text, at, INVOKE_END + "\n")
        calls.append((name, args))
    if not calls:
        raise common.CallMarkupError("the reply contained an empty DSML tool block")
    return at + len(CLOSE), calls


class Reader:
    """Publish each complete block once. Incomplete blocks never publish a call."""

    def __init__(self, plan):
        self.plan = plan
        self.offset = 0
        self.calls = []
        self.text = []
        self.unfinished = False
        self.index = -1
        self.streamed = False

    def read(self, answer, *, finished=False):
        self.unfinished = False
        while self.offset < len(answer):
            start = answer.find(OPEN, self.offset)
            if start < 0:
                tail = 0 if finished else _partial_tag(answer, OPEN)
                end = max(self.offset, len(answer) - tail)
                self.text.append(answer[self.offset:end])
                self.offset = end
                return
            self.text.append(answer[self.offset:start])
            self.offset = start
            try:
                if answer.find(CLOSE, start + len(OPEN)) < 0:
                    raise _Incomplete
                end, calls = _block(answer, start, self.plan.tools)
            except _Incomplete:
                self.unfinished = True
                if finished:
                    self.text.append(answer[start:])
                return
            if not self.plan.parallel and len(self.calls) + len(calls) != 1:
                raise common.CallMarkupError("the reply exceeded its one-call DSML grammar")
            self.calls.extend(calls)
            self.offset = end

    def feed(self, answer):
        before = len(self.calls)
        self.read(answer)
        deltas = stream_tool_call_deltas([_call(name, args) for name, args in self.calls[before:]])
        for delta in deltas:
            for part in delta["tool_calls"]:
                part["index"] += before
        if deltas:
            self.index = len(self.calls) - 1
            self.streamed = True
        return deltas
