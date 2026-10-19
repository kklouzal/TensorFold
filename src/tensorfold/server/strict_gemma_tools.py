"""Exact native Gemma tool calls, with schema constraints enforced before publication.

The captured profile writes sorted bare keys, raw sentinel strings, true/false
and the template's null word (``template_null``) recursively. Owned EBNF expresses
that representation; native xgrammar conversion remains authoritative for numeric
constraints and regexes.
"""
from __future__ import annotations

import json
import re
import string
import uuid
from copy import deepcopy

from tensorfold.engine.grammar import Spec
from tensorfold.server import strict_tools as xml
from tensorfold.server.gemma_values import (
    GEMMA_CLOSE, GEMMA_KEY, GEMMA_NULLS, GEMMA_OPEN, GEMMA_STRING, GemmaJSONStream, gemma_value,
)
from tensorfold.server.text import _partial_tag
from tensorfold.server.tools import tool_spec_name


PROBE = deepcopy(xml.PROBE)
PROBE[1]["tool_calls"][0]["function"]["arguments"]["tfprobe_x"] = 'tfprobe_"\\\n<tool_call|>value'
PROBE_TOOLS = xml.PROBE_TOOLS


def false_probe():
    messages = deepcopy(PROBE)
    messages[1]["tool_calls"][0]["function"]["arguments"]["tfprobe_bool"] = False
    return messages


def _probe_text(messages, null):
    return "".join(GEMMA_OPEN + "call:" + call["function"]["name"]
                   + gemma_value(call["function"]["arguments"], null=null)
                   + GEMMA_CLOSE for call in messages[1]["tool_calls"])


def template_null(rendered: str, rendered_false: str) -> str | None:
    """The null word of a template whose probe calls match exactly, else None.

    Both booleans, recursively typed values (nested nulls included), key
    ordering and adjacent call envelopes must match one ``GEMMA_NULLS`` spelling.
    """
    return next((null for null in GEMMA_NULLS
                 if _probe_text(PROBE, null) in rendered and _probe_text(false_probe(), null) in rendered_false), None)


def _q(text):
    return json.dumps(text, ensure_ascii=False)


class _Grammar:
    """One checked schema graph, emitted once per node, including recursive local references."""
    def __init__(self, parameters, where, null):
        self.null = null
        self.schema = xml._Schema(parameters, where)
        self.schema.check()
        self.rules, self.nodes, self.pending = {}, {}, []
        self.any = {}
        self.rules["number"] = r'"-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?'
        self.rules["integer"] = r'"-"? ("0" | [1-9] [0-9]*)'
        self.rules["key_rest"] = r'[A-Za-z0-9_.\-]*'
        # A small DFA excludes the complete raw string delimiter while allowing
        # every proper prefix, including a final partial delimiter as data.
        marker = GEMMA_STRING
        for i in range(len(marker)):
            choices = ['""']
            groups = {}
            for char in set(marker):
                suffix = marker[:i] + char
                if suffix.endswith(marker):
                    continue
                nxt = max(k for k in range(len(marker)) if suffix.endswith(marker[:k]))
                groups.setdefault(nxt, []).append(char)
            excluded = "".join("\\" + c if c in '\\]-^' else c for c in sorted(set(marker)))
            choices.append(f'[^{excluded}] string_0')
            choices.extend(f'({_q(c)}) string_{nxt}' for nxt, chars in groups.items() for c in sorted(chars))
            self.rules[f"string_{i}"] = " | ".join(choices)
        root = self.node(parameters, where)
        while self.pending:
            name, node, at = self.pending.pop()
            self.rules[name] = self.build(node, at)
        # Native numeric rules can end in a rule-level lookahead assertion;
        # enclosing that body in parentheses changes the EBNF parser context.
        self.text = "root ::= " + root + "\n" + "\n".join(f"{name} ::= {body}" for name, body in self.rules.items())
        if len(self.text.encode()) > xml.MAX_GRAMMAR_BYTES:
            raise xml._too_large()

    def node(self, node, at):
        if id(node) not in self.nodes:
            if len(self.nodes) >= xml.MAX_SUBSCHEMAS:
                raise xml._refuse(at, f"a native Gemma tool schema is limited to {xml.MAX_SUBSCHEMAS} subschemas")
            name = f"value_{len(self.nodes)}"
            self.nodes[id(node)] = name
            self.pending.append((name, node, at))
        return self.nodes[id(node)]

    def imported(self, source):
        """Namespace lexical rule identifiers only; terminals/classes retain their exact bytes."""
        prefix = f"import_{len(self.rules)}_"
        names = re.findall(r"^([A-Za-z_][A-Za-z_0-9-]*)\s*::=", source, re.M)
        rename = {name: prefix + name for name in names}
        token = re.compile(r'"(?:[^"\\]|\\.)*"|\[(?:[^\]\\]|\\.)*\]|[A-Za-z_][A-Za-z_0-9-]*')
        lines = token.sub(lambda m: rename.get(m[0], m[0]), source)
        self.rules[prefix + "entry"] = rename["root"]
        # Imported source is already a verified native grammar, not untrusted
        # input concatenated into one of our productions.
        for line in lines.splitlines():
            name, body = line.split(" ::= ", 1)
            self.rules[name] = body
        return prefix + "entry"

    def native(self, node, *, regex=False):
        try:
            import xgrammar as xgr
        except ImportError:
            from tensorfold.engine.grammar import EXTRA
            raise xml._refuse(self.schema.where, f"strict function tools need xgrammar: pip install '{EXTRA}'") from None
        try:
            source = xgr.Grammar.from_regex(node) if regex else xgr.Grammar.from_json_schema(
                node, max_whitespace_cnt=xml.BLANKS)
            return self.imported(str(source))
        except (RuntimeError, ValueError, TypeError) as exc:
            from tensorfold.engine.grammar import _message
            raise xml._refuse(self.schema.where, f"the grammar cannot be enforced: {_message(exc)}") from None

    def literal(self, value, at):
        try:
            return _q(gemma_value(value, null=self.null))
        except (ValueError, RecursionError) as exc:
            raise xml._refuse(at, str(exc)) from None

    def build(self, node, at):
        if "$ref" in node:
            return self.node(self.schema.resolve(node["$ref"], at), at)
        if "anyOf" in node:
            return " | ".join(self.node(sub, f"{at}.anyOf[{i}]") for i, sub in enumerate(node["anyOf"]))
        if "const" in node or "enum" in node:
            values = [node["const"]] if "const" in node else node["enum"]
            return " | ".join(dict.fromkeys(self.literal(value, at) for value in values))
        kinds = self.schema.types(node)
        kinds = kinds - {"integer"} if "number" in kinds else kinds
        alternatives = []
        for kind in sorted(kinds):
            if kind == "null":
                alternatives.append(_q(self.null))
            elif kind == "boolean":
                alternatives.append('("true" | "false")')
            elif kind in ("number", "integer"):
                constraints = {key: node[key] for key in (*xml._RANGES, "multipleOf") if key in node}
                alternatives.append(self.native({"type": kind, **constraints}) if constraints else kind)
            elif kind == "string":
                pattern = node.get("pattern", xml.FORMATS.get(node.get("format")))
                if pattern is None:
                    content = "string_0"
                else:
                    tree = xml._regex(pattern, at)
                    if all(xml._may_match(tree, char) for char in GEMMA_STRING):
                        raise xml._refuse(at, "the string pattern may spell Gemma's <|\"|> delimiter: narrow it")
                    content = self.native(pattern, regex=True)
                alternatives.append(f'{_q(GEMMA_STRING)} {content} {_q(GEMMA_STRING)}')
            elif kind == "array":
                alternatives.append(self.array(node, at))
            else:
                alternatives.append(self.object(node, at))
        if not alternatives:
            raise xml._refuse(at, "the schema admits no native Gemma value")
        return " | ".join(alternatives)

    def array(self, node, at):
        low, high = node.get("minItems", 0), node.get("maxItems")
        if high is not None and low > high:
            raise xml._refuse(at, "minItems exceeds maxItems")
        item = node.get("items", self.any)
        if node.get("uniqueItems"):
            seen = set()
            while "$ref" in item and id(item) not in seen:
                seen.add(id(item))
                item = self.schema.resolve(item["$ref"], at)
            values = item.get("enum")
            if not values or not all(isinstance(value, str) for value in values):
                raise xml._refuse(at, "uniqueItems needs a string enum with at most 10 distinct values")
            values = list(dict.fromkeys(value for value in values if "const" not in item or value == item["const"]))
            if 2 ** len(values) > xml.MAX_SUBSCHEMAS:
                raise xml._refuse(at, "uniqueItems needs a string enum with at most 10 distinct values")
            from tensorfold.server.strict_grammar import unique_array
            source = unique_array(values, low, high, 0)
            # This independently generated oracle language contains only these
            # known JSON string literals, replaced as whole lexical terminals.
            mapping = {_q(json.dumps(value, ensure_ascii=False)): self.literal(value, at) for value in values}
            source = re.sub(r'"(?:[^"\\]|\\.)*"', lambda m: mapping.get(m[0], m[0]), source)
            return self.imported(source)
        value = self.node(item, at + ".items")
        options = ['"[]"'] if low == 0 else []
        if high is None or high:
            lower, upper = max(1, low) - 1, "" if high is None else str(high - 1)
            options.append(f'"[" {value} ("," {value}){{{lower},{upper}}} "]"')
        return " | ".join(options)

    def keys(self, excluded):
        """Bare ASCII additional keys, excluding every declared key to preserve its stricter schema."""
        root, trie = {}, {}
        for key in excluded:
            current = root
            for char in key:
                current = current.setdefault(char, {})
            current[None] = True
        prefix = f"key_{len(self.rules)}_"
        pending = [(root, 0, False)]
        trie[id(root)] = 0
        while pending:
            current, index, nonempty = pending.pop()
            alphabet = set(string.ascii_letters + "_" + (string.digits + ".-" if nonempty else ""))
            choices = ['""'] if nonempty and None not in current else []
            fallback = alphabet - set(current)
            if fallback:
                chars = "".join("\\-" if char == "-" else char for char in sorted(fallback))
                choices.append(f'[{chars}] key_rest')
            for char in sorted(alphabet & set(current)):
                child = current[char]
                if id(child) not in trie:
                    trie[id(child)] = len(trie)
                    pending.append((child, trie[id(child)], True))
                choices.append(f'{_q(char)} {prefix}{trie[id(child)]}')
            self.rules[prefix + str(index)] = " | ".join(choices)
        return prefix + "0"

    def object(self, node, at):
        props = node.get("properties", {})
        required = set(node.get("required", ()))
        for key in props:
            if not GEMMA_KEY.fullmatch(key):
                raise xml._refuse(at + ".properties." + key, "this key cannot be written in Gemma's bare-key markup")
        ordered = sorted(props, key=str.lower)
        values = [self.node(props[key], at + ".properties." + key) for key in ordered]
        prefix = f"fields_{len(self.rules)}_"
        extra = node.get("additionalProperties", True)
        tail = None
        if extra is not False:
            value = self.node(extra if isinstance(extra, dict) else self.any, at + ".additionalProperties")
            key = self.keys(props)
            tail = f'{key} ":" {value} ("," {key} ":" {value})*'
        for seen in (False, True):
            self.rules[prefix + f"{len(ordered)}_{int(seen)}"] = (
                f'({_q("," if seen else "")} {tail})?' if tail else '""')
        for i in reversed(range(len(ordered))):
            key, value = ordered[i], values[i]
            for seen in (False, True):
                following = prefix + f"{i + 1}_1"
                field = f'{_q(("," if seen else "") + key + ":")} {value} {following}'
                if key not in required:
                    field += " | " + prefix + f"{i + 1}_{int(seen)}"
                self.rules[prefix + f"{i}_{int(seen)}"] = field
        return f'"{{" {prefix}0_0 "}}"'


class Plan(xml.Plan):
    """A native Gemma structural tag with the ordinary strict Plan reader protocol.

    ``null`` is the selected template's null word (``template_null``).
    """
    def __init__(self, tools, *, required, parallel, null, offered=None):
        if null not in GEMMA_NULLS:
            raise ValueError(f"a Gemma null word is one of {GEMMA_NULLS}")
        self.required, self.parallel, self.tools, self.null = bool(required), bool(parallel), {}, null
        positions = {id(tool): i for i, tool in reversed(list(enumerate(offered or ())))}
        tags = []
        for i, tool in enumerate(tools):
            where = f"tools[{positions.get(id(tool), i)}].function"
            name = tool_spec_name(tool)
            if not GEMMA_KEY.fullmatch(name):
                raise xml._refuse(where + ".name", "the name cannot be written in Gemma call markup")
            if name in self.tools:
                raise xml._refuse(where + ".name", "the function is offered twice")
            parameters = xml._function(tool).get("parameters")
            parameters = {} if parameters is None else parameters
            if not isinstance(parameters, dict):
                raise xml._refuse(where + ".parameters", "must be a JSON schema object")
            strict = xml.strict_flag(tool, where)
            if strict:
                if {"anyOf", "$ref", "enum", "const"} & set(parameters) or parameters.get("type", "object") != "object":
                    raise xml._refuse(where + ".parameters", "a strict tool's parameters must be an object schema")
                top = {"type": "object", **{k: v for k, v in parameters.items() if k != "$schema"}}
                grammar = _Grammar(top, where + ".parameters", null)
            else:
                grammar = _Grammar({"type": "object"}, where + ".parameters", null)
            self.tools[name] = xml.Tool(name, strict)
            tags.append({"type": "tag", "begin": GEMMA_OPEN + "call:" + name,
                         "content": {"type": "grammar", "grammar": grammar.text}, "end": GEMMA_CLOSE})
        if required:
            blanks = {"type": "regex", "pattern": f"[ \\t\\r\\n]{{0,{xml.BLANKS}}}"}
            body = xml._seq(blanks, {"type": "tags_with_separator", "tags": tags, "separator": "",
                                    "at_least_one": True, "stop_after_first": not parallel}, blanks)
        else:
            body = {"type": "triggered_tags", "triggers": [GEMMA_OPEN], "tags": tags,
                    "stop_after_first": not parallel}
        text = json.dumps({"type": "structural_tag", "format": body}, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode()) > xml.MAX_GRAMMAR_BYTES:
            raise xml._too_large()
        self.spec = Spec("tools", text, "tools")

    def reader(self):
        return Reader(self)

    def parse(self, answer, *, max_calls=None):
        content, calls, _ = self.parse_finished(answer, max_calls=max_calls)
        return content, calls

    def content(self, answer, *, finished):
        reader = self.reader()
        reader.read(answer, finished=finished)
        return "".join(reader.text)


class Reader:
    """Incrementally transcode grammar-admitted calls; retain truncated raw markup as text."""
    def __init__(self, plan):
        self.plan = plan
        self.text, self.calls, self.deltas = [], [], []
        self.index, self.streamed, self.pos = -1, False, 0
        self._state, self._unfinished = "text", False
        self._at = self._sent = 0
        self._quoted = False

    def feed(self, answer):
        before = len(self.deltas)
        self.read(answer, finished=False)
        return self.deltas[before:]

    def emit(self, fragment):
        if fragment:
            self._args.append(fragment)
            self.deltas.append({"tool_calls": [{"index": self.index, "function": {"arguments": fragment}}]})

    def read(self, answer, *, finished):
        while True:
            if self._state == "text":
                at = answer.find(GEMMA_OPEN, self.pos)
                end = at if at >= 0 else len(answer) - (0 if finished else _partial_tag(answer[self.pos:], GEMMA_OPEN))
                if end > self.pos:
                    self.text.append(answer[self.pos:end])
                    self.pos = end
                if at < 0:
                    return
                self._at, self._state = at, "name"
            if self._state == "name":
                begin = GEMMA_OPEN + "call:"
                if not answer.startswith(begin, self.pos):
                    if begin.startswith(answer[self.pos:]):
                        return self.wait(answer, finished)
                    raise xml.CallMarkupError("the reply's Gemma call opener left its grammar")
                end = answer.find("{", self.pos + len(begin))
                if end < 0:
                    return self.wait(answer, finished)
                name = answer[self.pos + len(begin):end]
                if name not in self.plan.tools:
                    raise xml.CallMarkupError(f"the reply called {name!r}, which its grammar does not offer")
                self.index += 1
                self.streamed = True
                self._head = {"index": self.index, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                              "function": {"name": name, "arguments": ""}}
                self.deltas.append({"tool_calls": [self._head]})
                self._args, self._codec = [], GemmaJSONStream(null=self.plan.null)
                self.pos = self._sent = end
                self._quoted, self._state = False, "value"
            if self._state == "value":
                while self.pos < len(answer):
                    if answer.startswith(GEMMA_STRING, self.pos):
                        self._quoted = not self._quoted
                        self.pos += len(GEMMA_STRING)
                    elif not self._quoted and answer.startswith(GEMMA_CLOSE, self.pos):
                        self.transcode(answer[self._sent:self.pos], finished=True)
                        arguments = "".join(self._args)
                        try:
                            value = json.loads(arguments, parse_float=str)
                        except (ValueError, RecursionError) as exc:
                            raise xml.CallMarkupError("the reply's Gemma values left their grammar") from exc
                        if not isinstance(value, dict):
                            raise xml.CallMarkupError("Gemma call arguments are not an object")
                        self.calls.append({"id": self._head["id"], "type": "function",
                                           "function": {"name": self._head["function"]["name"], "arguments": arguments}})
                        self.pos += len(GEMMA_CLOSE)
                        self._state = "text"
                        break
                    elif len(answer) - self.pos < len(GEMMA_CLOSE) and (
                            GEMMA_STRING.startswith(answer[self.pos:])
                            or not self._quoted and GEMMA_CLOSE.startswith(answer[self.pos:])):
                        break
                    else:
                        self.pos += 1
                else:
                    pass
                if self._state == "text":
                    continue
                self.transcode(answer[self._sent:self.pos])
                self._sent = self.pos
                return self.wait(answer, finished)

    def transcode(self, text, *, finished=False):
        try:
            self.emit(self._codec.feed(text, finished=finished))
        except ValueError as exc:
            raise xml.CallMarkupError("the reply's Gemma values left their template grammar") from exc

    def wait(self, answer, finished):
        if finished:
            self._unfinished = True
            self.text.append(answer[self._at:])
            self.pos, self._state = len(answer), "text"
