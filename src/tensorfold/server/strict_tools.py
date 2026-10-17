"""OpenAI ``strict`` function tools: the reply grammar that holds each call to its tool's schema, and its reader.

When a request's active tools include ``"strict": true``, the reply decodes under one xgrammar structural tag
(``engine.grammar`` kind ``"tools"``), from the token after ``</think>`` when thinking. A call is the chat template's own
markup, which ``template_supported`` checks from a rendered probe (Qwen's XML calls)::

    <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>

Parameters come in the schema's declared order (optional ones may be left out); a string parameter's VALUE is its raw
text (the template's ``|string``), any other value is JSON (``|tojson``), and a string-or-null one is raw text or
``null``. ``tool_choice`` "auto" leaves text free around calls; "required" or a named function allows only blanks and
calls; ``parallel_tool_calls`` false ends the reply after the first call. Tools that are not strict keep their names
and parameter markup well formed, with values unchecked (decoded as before).

``Reader`` reads that language exactly (case-sensitive, by structure, never the lenient parsers): the streamed content,
streamed call deltas and the final calls, whose JSON arguments keep each JSON value's text as written.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from tensorfold.engine.grammar import BLANKS, Spec   # BLANKS: blanks around required calls and in JSON values
from tensorfold.server.errors import RequestError
from tensorfold.server.text import _partial_tag
from tensorfold.server.tools import tool_spec_name
from tensorfold.tool_parameters import decode_parameter

# -- the template's call markup (Qwen's XML calls) ------------------------------------------------------------------
OPEN = "<tool_call>"                         # every call begins with it; free text never holds it otherwise
FUNCTION = "<tool_call>\n<function="         # then NAME and NAME_END
NAME_END = ">\n"
PARAMETER = "<parameter="                    # then KEY and NAME_END
VALUE_END = "</parameter>"                   # a value's text never holds it: its line ends the value
VALUE_CLOSE = "\n</parameter>\n"
CLOSE = "</function>\n</tool_call>"
SEPARATOR = "\n"                             # between calls

PROBE = [{"role": "user", "content": "x"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call_0", "type": "function",
              "function": {"name": "tfprobe_a", "arguments": {"tfprobe_x": "tfprobe_v", "tfprobe_y": 7,
                                                            "tfprobe_bool": True, "tfprobe_null": None,
                                                         "tfprobe_nested": [True, None, {"key": "value"}]}}},
             {"id": "call_1", "type": "function", "function": {"name": "tfprobe_b", "arguments": {}}}]}]
PROBE_TOOLS = [{"type": "function", "function": {"name": "tfprobe_a", "parameters": {
    "type": "object", "properties": {"tfprobe_x": {"type": "string"}, "tfprobe_y": {"type": "integer"},
    "tfprobe_bool": {"type": "boolean"}, "tfprobe_null": {"type": "null"},
    "tfprobe_nested": {"type": "array", "items": {}}}}}},
    {"type": "function", "function": {"name": "tfprobe_b", "parameters": {"type": "object", "properties": {}}}}]
_PROBED = (f"{FUNCTION}tfprobe_a{NAME_END}{PARAMETER}tfprobe_x{NAME_END}tfprobe_v{VALUE_CLOSE}"
           f"{PARAMETER}tfprobe_y{NAME_END}7{VALUE_CLOSE}"
           f"{PARAMETER}tfprobe_bool{NAME_END}true{VALUE_CLOSE}"
           f"{PARAMETER}tfprobe_null{NAME_END}null{VALUE_CLOSE}"
           f'{PARAMETER}tfprobe_nested{NAME_END}[true, null, {{"key": "value"}}]{VALUE_CLOSE}'
           f"{CLOSE}{SEPARATOR}{FUNCTION}tfprobe_b{NAME_END}{CLOSE}")


def template_supported(rendered: str) -> bool:
    """Whether a chat template rendered ``PROBE``'s calls exactly as this module writes and reads them."""

    return _PROBED in rendered


# -- the supported JSON-schema subset ---------------------------------------------------------------------------------
TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})
_ANNOTATIONS = frozenset({"title", "description", "default", "examples", "$comment", "deprecated", "readOnly",
                          "writeOnly"})
_KEYWORDS = _ANNOTATIONS | {"type", "enum", "const", "anyOf", "$ref", "$defs", "definitions", "properties", "required",
                            "additionalProperties", "items", "minItems", "maxItems", "pattern", "format", "minimum",
                            "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "uniqueItems"}
# OpenAI's strict-mode limits (developers.openai.com/api/docs/guides/structured-outputs#supported-schemas); the nesting
# limit also bounds anyOf and $defs nested in one another between two property or items levels
MAX_NESTING, MAX_PROPERTIES, MAX_ENUM_VALUES, MAX_SCHEMA_CHARS = 10, 5000, 1000, 120_000
# The most grammar text one request's strict tools may make: the CUDA HTTP body cap, which is also the two-rank
# transport envelope (``cuda.rank_protocol.MAX_MESSAGE_BYTES``, kept equal by a test). Definitions and root references
# are copied into every JSON parameter's schema, so the text can outgrow the request; past this it is refused.
MAX_GRAMMAR_BYTES = 96 << 20
# The most subschemas one JSON parameter's grammar holds, the definitions and root it refers to included. xgrammar
# 0.2.8 compiles a schema recursively along its references, and a long chain overflows the thread's stack and kills the
# process: in an 8 MB worker thread (amd64 verification image) a chain of objects did so at 8,000 subschemas (4,000
# objects, compiled at 6,000), plain references at 20,000, anyOf references not yet at 21,000. A quarter of the
# smallest: other builds (the GB10's ARM64) and stack sizes were not measured.
MAX_SUBSCHEMAS = 2000
_LENGTHS = {"minLength", "maxLength"}
_RANGES = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
# keywords that constrain one JSON type: xgrammar 0.2.8 applies them only beside an explicit type
_TYPED = {"pattern", "format", "multipleOf", "items", "minItems", "maxItems", "properties", "required",
          "additionalProperties", "uniqueItems", *_RANGES}

_HEX = "[0-9a-fA-F]"
_OCTET = "(25[0-5]|(2[0-4]|1?[0-9])?[0-9])"
_IPV4_TAIL = f"({_OCTET}\\.){{3}}{_OCTET}"
_DATE = "[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
_TIME = "([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\\.[0-9]+)?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])"
_TIME_PART = "([0-9]+S|[0-9]+M([0-9]+S)?|[0-9]+H([0-9]+M([0-9]+S)?)?)"
_ATOM = "[a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+"
# A raw string's format, as xgrammar's own JSON-schema formats spell it (email without the quoted local part, which
# could hold ``</parameter>``); a format regex never matches ``<``.
FORMATS = {
    "date-time": f"{_DATE}T{_TIME}",
    "date": _DATE,
    "time": _TIME,
    "duration": (f"P(([0-9]+D|[0-9]+M([0-9]+D)?|[0-9]+Y([0-9]+M([0-9]+D)?)?)(T{_TIME_PART})?|T{_TIME_PART}"
                 f"|[0-9]+W)"),
    "email": f"{_ATOM}(\\.{_ATOM})*@[A-Za-z0-9]([-A-Za-z0-9]*[A-Za-z0-9])?(\\.[A-Za-z0-9][-A-Za-z0-9]*[A-Za-z0-9])*",
    "hostname": "[a-z0-9]([a-z0-9-]*[a-z0-9])?(\\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*",
    "ipv4": _IPV4_TAIL,
    "ipv6": (f"(({_HEX}{{1,4}}:){{7}}{_HEX}{{1,4}}|({_HEX}{{1,4}}:){{1,7}}:|({_HEX}{{1,4}}:){{1,6}}:{_HEX}{{1,4}}"
             f"|({_HEX}{{1,4}}:){{1,5}}(:{_HEX}{{1,4}}){{1,2}}|({_HEX}{{1,4}}:){{1,4}}(:{_HEX}{{1,4}}){{1,3}}"
             f"|({_HEX}{{1,4}}:){{1,3}}(:{_HEX}{{1,4}}){{1,4}}|({_HEX}{{1,4}}:){{1,2}}(:{_HEX}{{1,4}}){{1,5}}"
             f"|{_HEX}{{1,4}}:(:{_HEX}{{1,4}}){{1,6}}|:((:{_HEX}{{1,4}}){{1,7}}|:)|::(ffff(:0{{1,4}})?:)?{_IPV4_TAIL}"
             f"|({_HEX}{{1,4}}:){{1,4}}:{_IPV4_TAIL})"),
    "uuid": f"{_HEX}{{8}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{12}}",
}


def _refuse(where: str, why: str) -> RequestError:
    return RequestError(f"{where}: {why}")


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return "array" if isinstance(value, list) else "object"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _value_type(value: Any) -> str:
    """A JSON value's type, integers (1.0 included) apart from other numbers."""

    if isinstance(value, float) and value.is_integer():
        return "integer"
    return "integer" if type(value) is int else _kind(value)


def _within(types: set[str], declared: set[str]) -> bool:
    return all(t in declared or (t == "integer" and "number" in declared) for t in types)


def _literals(node: dict[str, Any]) -> list[list[Any]]:
    """The value lists ``node``'s ``enum`` and ``const`` each allow (an unchecked enum that is not a list: none)."""

    enum = [node["enum"]] if isinstance(node.get("enum"), list) else []
    return enum + ([[node["const"]]] if "const" in node else [])


def _side(kind: str) -> str:
    """A JSON type as the raw call markup sees it: string, null or other."""

    return kind if kind in ("string", "null") else "other"


class _Schema:
    """One tool's ``parameters``: checked against the supported subset, refs resolved within it.

    Every walk is bounded by the request: ``check`` recurses only along the schema's own nesting (at most
    ``MAX_NESTING`` property/items levels, each holding at most ``MAX_NESTING`` anyOf/$defs levels), and whatever follows
    ``$ref`` (``types``, ``raw``, ``json``) runs without recursion, each node or definition once, so shared and long ref
    chains cost linear time and cycles end."""

    def __init__(self, root: dict[str, Any], where: str) -> None:
        self.root, self.where = root, where
        self.properties = self.chars = self.enums = 0
        self._unique = False
        self._typed_anyof: list[tuple[dict[str, Any], set[str], str]] = []   # checked once all nodes are
        self._types: dict[int, frozenset[str]] = {}       # solved node types by id (the request holds the nodes)
        self._copies: dict[tuple[str, str], tuple[dict[str, Any], set[str], int, int]] = {}   # rebased defs, root

    def resolve(self, ref: Any, where: str) -> dict[str, Any]:
        """The subschema a local ``$ref`` names, its JSON pointer read through schema keywords only: data such as
        ``default``, ``examples``, ``enum`` or ``const`` is never checked as a schema, so no ref may name it."""

        if not isinstance(ref, str) or not ref.startswith("#"):
            raise _refuse(where, "$ref must point into the same schema (#...)")
        if ref in ("#", "#/"):
            return self.root
        if not ref.startswith("#/"):
            raise _refuse(where, f"$ref {ref!r} is not a JSON pointer")
        parts = [_unescape(part) for part in ref[2:].split("/")]
        node: Any = self.root
        while parts and isinstance(node, dict):
            keyword = parts.pop(0)
            inner = node.get(keyword)
            if keyword in ("items", "additionalProperties"):
                node = inner
            elif keyword in ("properties", "$defs", "definitions") and parts and isinstance(inner, dict):
                node = inner.get(parts.pop(0))
            elif (keyword == "anyOf" and parts and isinstance(inner, list) and parts[0].isascii() and parts[0].isdigit()
                  and int(parts[0]) < len(inner)):
                node = inner[int(parts.pop(0))]
            else:
                node = None
        if parts or not isinstance(node, dict):
            raise _refuse(where, f"$ref {ref!r} does not resolve to a subschema")
        return node

    def check(self) -> None:
        """The whole schema supported, well formed and within OpenAI's limits (RequestError naming the path)."""

        self._check(self.root, self.where, 0, 0)
        for node, declared, where in self._typed_anyof:      # types follow refs: every node is checked by now
            if not all(_within(self.types(b), declared) for b in node["anyOf"]):
                raise _refuse(where, "each anyOf branch must admit only the types listed beside it")

    def _check(self, node: Any, where: str, depth: int, chain: int) -> None:
        """``node`` and its subschemas; ``depth`` counts property/items levels, ``chain`` anyOf/$defs levels within
        the current one."""

        if not isinstance(node, dict):
            raise _refuse(where, "a schema must be a JSON object")
        if chain > MAX_NESTING:
            raise _refuse(where, f"anyOf and $defs nest at most {MAX_NESTING} levels in one another here")
        unknown = sorted(k for k in node if k not in _KEYWORDS)
        if unknown:
            hint = (" (strict schemas follow OpenAI's subset: put length limits in the description)"
                    if _LENGTHS & set(unknown) else "")
            raise _refuse(where, f"strict function schemas do not support {', '.join(unknown)}{hint}")
        kinds = node.get("type")
        declared = set(TYPES)
        if kinds is not None:
            listed = kinds if isinstance(kinds, list) else [kinds]
            if not listed or any(not isinstance(k, str) or k not in TYPES for k in listed) or len(set(listed)) < len(
                    listed):
                raise _refuse(where, "type must be one JSON type name or a list of distinct ones")
            declared = set(listed)
        # xgrammar 0.2.8 lets enum, const, $ref and anyOf override what stands beside them, format override pattern, and
        # drops type-specific keywords without a type: those shapes are refused rather than left unenforced
        own = set(node) - _ANNOTATIONS - {"$defs", "definitions"}
        if "$ref" in node and own - {"$ref"}:
            raise _refuse(where, "$ref takes no constraints beside it here: put them in the schema it names")
        if ("enum" in node or "const" in node) and own - {"enum", "const", "type"}:
            raise _refuse(where, "enum and const take only type beside them here")
        if "anyOf" in node:
            if own - {"anyOf", "type"}:
                raise _refuse(where, "anyOf takes only type beside it here: put the other constraints in each branch")
            if kinds is not None:
                self._typed_anyof.append((node, declared, where))
        if "pattern" in node and "format" in node:
            raise _refuse(where, "pattern and format are not enforced together here: use one")
        typed = own & _TYPED
        if typed and kinds is None:
            raise _refuse(where, f"{', '.join(sorted(typed))} need a type beside them")
        if typed and len(declared - {"null"}) > 1:
            raise _refuse(where, f"{', '.join(sorted(typed))} need a single type (besides null)")
        for values in _literals(node):
            if kinds is not None and not _within({_value_type(v) for v in values}, declared):
                raise _refuse(where, "every enum and const value must have a type listed beside it")
        if "enum" in node:
            if not isinstance(node["enum"], list) or not node["enum"]:
                raise _refuse(where, "enum must be a non-empty list")
            self.enums += len(node["enum"])
            enum_chars = sum(len(v) for v in node["enum"] if isinstance(v, str))
            if len(node["enum"]) > 250 and enum_chars > 15_000:
                raise _refuse(where, "an enum with more than 250 values is limited to 15000 string characters")
            self.chars += enum_chars
            # xgrammar 0.2.8 enforces const alone when both stand: const must be one of the enum's values (written
            # alike, so 1 beside 1.0 is refused too)
            try:
                outside = "const" in node and _canonical(node["const"]) not in {_canonical(v) for v in node["enum"]}
            except RecursionError:
                raise _refuse(where, "enum and const values nest too deeply") from None
            if outside:
                raise _refuse(where, "const must be one of the enum values here (no value satisfies both)")
        if isinstance(node.get("const"), str):
            self.chars += len(node["const"])
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key in node and not _number(node[key]):
                raise _refuse(where, f"{key} must be a number")
        if "multipleOf" in node:
            if not _number(node["multipleOf"]) or node["multipleOf"] <= 0:
                raise _refuse(where, "multipleOf must be a positive number")
            if kinds not in ("integer", ["integer"]) or _RANGES & set(node):
                raise _refuse(where, "multipleOf is enforced for type integer without minimum or maximum only")
        for key in ("minItems", "maxItems"):
            if key in node and (type(node[key]) is not int or node[key] < 0):
                raise _refuse(where, f"{key} must be a non-negative integer")
        if "uniqueItems" in node:
            if type(node["uniqueItems"]) is not bool:
                raise _refuse(where, "uniqueItems must be a boolean")
            self._unique |= node["uniqueItems"]
        if "pattern" in node:
            _regex(node["pattern"], where)
        if "format" in node and (not isinstance(node["format"], str) or node["format"] not in FORMATS):
            raise _refuse(where, f"format must be one of {', '.join(FORMATS)}")
        if "$ref" in node:
            self.resolve(node["$ref"], where)
        for key in ("$defs", "definitions"):
            if key in node:
                if not isinstance(node[key], dict):
                    raise _refuse(where, f"{key} must be an object of schemas")
                for name, sub in node[key].items():
                    self.chars += len(name)
                    self._check(sub, f"{where}.{key}.{name}", depth, chain + 1)
        if "anyOf" in node:
            if not isinstance(node["anyOf"], list) or not node["anyOf"]:
                raise _refuse(where, "anyOf must be a non-empty list of schemas")
            for i, sub in enumerate(node["anyOf"]):
                self._check(sub, f"{where}.anyOf[{i}]", depth, chain + 1)
        nested = "properties" in node or "items" in node or isinstance(node.get("additionalProperties"), dict)
        if nested and depth >= MAX_NESTING:
            raise _refuse(where, f"strict schemas nest at most {MAX_NESTING} levels")
        if "properties" in node:
            props = node["properties"]
            if not isinstance(props, dict):
                raise _refuse(where, "properties must be an object of schemas")
            self.properties += len(props)
            for name, sub in props.items():
                self.chars += len(name)
                self._check(sub, f"{where}.properties.{name}", depth + 1, 0)
        if "required" in node:
            required = node["required"]
            if (not isinstance(required, list) or any(not isinstance(r, str) for r in required)
                    or len(set(required)) < len(required)):
                raise _refuse(where, "required must be a list of distinct property names")
            missing = [r for r in required if r not in (node.get("properties") or {})]
            if missing:
                raise _refuse(where, f"required names {', '.join(missing)}, which properties does not declare")
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            self._check(extra, f"{where}.additionalProperties", depth + 1, 0)
        elif extra is not None and not isinstance(extra, bool):
            raise _refuse(where, "additionalProperties must be a boolean or a schema")
        if "items" in node:
            self._check(node["items"], f"{where}.items", depth + 1, 0)
        if self.properties > MAX_PROPERTIES or self.enums > MAX_ENUM_VALUES or self.chars > MAX_SCHEMA_CHARS:
            raise _refuse(self.where, f"strict schemas are limited to {MAX_PROPERTIES} properties, {MAX_ENUM_VALUES} "
                                      f"enum values and {MAX_SCHEMA_CHARS} characters of names and values")

    def _next(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        """The schemas a checked ``node``'s value must also satisfy: its ref's target, else its anyOf branches."""

        return [self.resolve(node["$ref"], self.where)] if "$ref" in node else list(node.get("anyOf") or ())

    def types(self, node: dict[str, Any]) -> frozenset[str]:
        """The JSON types a checked ``node`` admits (integer apart from number), refs and anyOf followed.

        The greatest solution over the ref graph, so a ref cycle constrains nothing by itself (a type is excluded only
        when the node's own type, enum or const, its ref's target, or every anyOf branch excludes it). Solved without
        recursion, each node and type at most once, and kept for later calls."""

        known = self._types
        if id(node) in known:
            return known[id(node)]
        order: list[dict[str, Any]] = []            # the unsolved nodes ``node`` reaches, and what each reads
        reads: dict[int, list[dict[str, Any]]] = {}
        stack = [node]
        while stack:
            n = stack.pop()
            if id(n) in reads or id(n) in known:
                continue
            reads[id(n)] = self._next(n)
            order.append(n)
            stack.extend(reads[id(n)])
        readers: dict[int, list[dict[str, Any]]] = {}   # who reads each unsolved node
        for n in order:
            for m in reads[id(n)]:
                if id(m) not in known:
                    readers.setdefault(id(m), []).append(n)
        excluded: dict[int, set[str]] = {id(n): set() for n in order}
        open_branches = {id(n): dict.fromkeys(TYPES, len(reads[id(n)])) for n in order if "anyOf" in n}
        queue: list[tuple[dict[str, Any], str]] = []

        def exclude(n: dict[str, Any], t: str) -> None:
            if t not in excluded[id(n)]:
                excluded[id(n)].add(t)
                queue.append((n, t))

        def lost(n: dict[str, Any], t: str) -> None:     # one more of an anyOf's branches excludes ``t``
            if "anyOf" not in n:                      # a ref: its target's exclusion is its own
                exclude(n, t)
                return
            open_branches[id(n)][t] -= 1
            if not open_branches[id(n)][t]:
                exclude(n, t)

        for n in order:
            for t in TYPES - _own_types(n):
                exclude(n, t)
            for m in reads[id(n)]:
                if id(m) in known:
                    for t in TYPES - known[id(m)]:
                        lost(n, t)
        while queue:
            m, t = queue.pop()
            for n in readers.get(id(m), ()):
                lost(n, t)
        for n in order:
            known[id(n)] = TYPES - excluded[id(n)]
        return known[id(node)]

    def admits(self, node: dict[str, Any]) -> set[str]:
        """What a checked ``node`` admits as the raw call markup sees it (string, null, other), its refs followed."""

        return {_side(t) for t in self.types(node)}

    def raw(self, node: dict[str, Any], where: str) -> dict[str, Any]:
        """The grammar of the strings a checked ``node`` admits, as raw text inside a parameter: the alternatives of its
        string branches (``check`` refused constraints beside ``$ref``, ``anyOf``, ``enum`` or ``const``, and pattern
        beside format), each once; a schema that refers back to itself is refused."""

        options: dict[str, dict[str, Any]] = {}     # by grammar text: the same alternative once, in schema order
        done: set[int] = set()
        path: set[int] = set()
        stack: list[tuple[dict[str, Any], str, bool]] = [(node, where, False)]
        while stack:
            n, at, leaving = stack.pop()
            if leaving:
                path.discard(id(n))
                done.add(id(n))
                continue
            if id(n) in path:
                raise _refuse(at, "a string parameter's schema may not refer back to itself")
            if id(n) in done:
                continue
            path.add(id(n))
            stack.append((n, at, True))
            if "$ref" in n:
                stack.append((self.resolve(n["$ref"], at), at, False))
            elif "anyOf" in n:
                stack.extend((b, f"{at}.anyOf[{i}]", False) for i, b in reversed(list(enumerate(n["anyOf"])))
                             if "string" in self.admits(b))
            else:
                for option in _raw_options(n, at):
                    options.setdefault(json.dumps(option, sort_keys=True), option)
        found = list(options.values())
        return found[0] if len(found) == 1 else {"type": "or", "elements": found}

    def json_format(self, node: dict[str, Any], where: str) -> dict[str, Any]:
        """JSON values, replacing finite uniqueness constraints with exact bounded owned EBNF rules.

        xgrammar's schema conversion remains authoritative for every other checked keyword. A marker absent from the
        source schema locates each replacement in its emitted EBNF; it can never replace an actual property or value.
        """

        if not self._unique:
            return {"type": "json_schema", "json_schema": node, "style": "json", "max_whitespace_cnt": BLANKS}
        from tensorfold.server.strict_grammar import unique_array

        local = _Schema(node, where)
        source = json.dumps(node, ensure_ascii=False, separators=(",", ":"))
        marker = _namespace(source, "__tensorfold_constraint_")
        replacements: list[tuple[str, str]] = []

        def convert(current: dict[str, Any]) -> dict[str, Any]:
            special = None
            if current.get("uniqueItems") is True and "array" in _own_types(current):
                item = current.get("items")
                seen = set()
                while isinstance(item, dict) and "$ref" in item and id(item) not in seen:
                    seen.add(id(item))
                    item = local.resolve(item["$ref"], where)
                values = item.get("enum") if isinstance(item, dict) else None
                if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
                    raise _refuse(where, "uniqueItems needs a string enum with at most 10 distinct values")
                values = list(dict.fromkeys(v for v in values if "const" not in item or v == item["const"]))
                if 2 ** len(values) > MAX_SUBSCHEMAS:
                    raise _refuse(where, "uniqueItems needs a string enum with at most 10 values")
                # Checked enum schemas have only type/enum/const and annotations beside them; filtering const here
                # preserves the intersection, and duplicate enum entries denote the same JSON string value.
                special = unique_array(values, current.get("minItems", 0), current.get("maxItems"), BLANKS)
                if "null" in _own_types(current):
                    special = special.replace("root ::= ", 'root ::= "null" | ', 1)
            if special is not None:
                word = marker + str(len(replacements))
                replacements.append((word, special))
                return {"const": word}
            out = dict(current)
            for keyword, key, child in _subschemas(current):
                if key is None:
                    out[keyword] = convert(child)
                else:
                    if out[keyword] is current[keyword]:
                        out[keyword] = dict(current[keyword]) if isinstance(current[keyword], dict) else list(
                            current[keyword])
                    out[keyword][key] = convert(child)
            return out

        converted = convert(node)
        if not replacements:
            return {"type": "json_schema", "json_schema": node, "style": "json", "max_whitespace_cnt": BLANKS}
        try:
            import xgrammar as xgr

            text = str(xgr.Grammar.from_json_schema(converted, max_whitespace_cnt=BLANKS))
        except ImportError:
            from tensorfold.engine.grammar import EXTRA

            raise RequestError(f"strict function tools need xgrammar on the server: pip install '{EXTRA}'") from None
        except (RuntimeError, ValueError, TypeError) as exc:
            from tensorfold.engine.grammar import _message

            raise _refuse(where, f"the grammar cannot be enforced: {_message(exc)}") from None
        prefix = _namespace(text, "tf_helpers_")
        for i, (word, replacement) in enumerate(replacements):
            # EBNF identifiers are lexical tokens; rewrite only them, never a terminal or a character class.
            rules = {}
            for line in replacement.splitlines():
                name, body = line.split(" ::= ", 1)
                rules[name] = body
            rename = {name: f"{prefix}{i}_{name}" for name in rules}
            token = re.compile(r'"(?:[^"\\]|\\.)*"|\[(?:[^\]\\]|\\.)*\]|[A-Za-z_][A-Za-z_0-9]*')
            lines = [rename[name] + " ::= " + token.sub(lambda m: rename.get(m[0], m[0]), body)
                     for name, body in rules.items()]
            literal = json.dumps(json.dumps(word, ensure_ascii=False), ensure_ascii=False)
            if literal not in text:
                # The copied tool root can contain unreferenced schemas. The native converter emits only reachable
                # definitions; a marker it never emits needs neither substitution nor retained helper rules.
                continue
            text = text.replace(literal, rename["root"])
            text += "\n" + "\n".join(lines)
        return {"type": "grammar", "grammar": text}

    def json(self, node: dict[str, Any], where: str) -> tuple[dict[str, Any], int]:
        """``node`` as a schema of its own (refs into the tool's schema kept), for a JSON value's grammar, and how many
        characters of the tool's definitions and root it copies in: the definitions it refers to (all of them with the
        root, when it refers to the root); RequestError past ``MAX_SUBSCHEMAS``."""

        refs = _refs(node)
        names: dict[str, list[str]] = {}            # the root definitions it needs, by keyword, in schema order
        rooted = False
        while refs and not rooted:
            ref = refs.pop()
            if not ref.startswith(("#/$defs/", "#/definitions/")):     # "#" or "#/properties/...": the root
                rooted = True
                break
            keyword, name = ref[2:].split("/", 2)[:2]
            name = _unescape(name)
            if name not in names.setdefault(keyword, []):
                names[keyword].append(name)
                refs |= self._copy(keyword, name)[1]
        own = {key: self.root[key] for key in ("$defs", "definitions") if isinstance(self.root.get(key), dict)}
        if rooted:
            names = {key: list(d) for key, d in own.items()}
        defs = {key: {name: self._copy(key, name)[0] for name in own[key] if name in names.get(key, ())}
                for key in own}
        copied = sum(self._copy(key, name)[2] for key in defs for name in defs[key])
        nodes = _count(node) + sum(self._copy(key, name)[3] for key in defs for name in defs[key])
        if rooted:
            if "__tf_root" in defs.get("$defs", {}):
                raise _refuse(self.where, "$defs may not name __tf_root")
            root = self._copy("", "")
            defs["$defs"] = {**defs.get("$defs", {}), "__tf_root": root[0]}
            copied, nodes = copied + root[2], nodes + root[3]
        if nodes > MAX_SUBSCHEMAS:
            raise _refuse(where, f"this parameter's schema holds {nodes} subschemas with the definitions it refers to: "
                                 f"strict schemas are limited to {MAX_SUBSCHEMAS} per parameter")
        sub = _rebase(node)
        for key, value in defs.items():             # the tool's own definitions win: its refs were written for them
            if value:
                sub[key] = {**sub[key], **value} if isinstance(sub.get(key), dict) else value
        return sub, copied

    def _copy(self, keyword: str, name: str) -> tuple[dict[str, Any], set[str], int, int]:
        """A root definition (``keyword`` ``$defs``/``definitions``), or the root itself (``""``), rebased once: (copy,
        the refs it holds, its grammar characters, its subschemas)."""

        key = (keyword, name)
        if key not in self._copies:
            node = self.root[keyword][name] if keyword else self.root
            copy = _rebase(node)
            self._copies[key] = (copy, _refs(node), len(json.dumps(copy, ensure_ascii=False, separators=(",", ":"))),
                                 _count(node))
        return self._copies[key]


def validate_response_schema(root: dict[str, Any], where: str) -> None:
    """Reject schemas whose requested strict response-format constraints the pinned compiler cannot enforce.

    Tool-specific grammar substitutions are not applied to response formats; those capabilities are explicitly
    refused here. Ordinary non-strict/guided schemas keep their existing compiler behavior.
    """

    root = {key: value for key, value in root.items() if key != "$schema"}
    schema = _Schema(root, where)
    schema.check()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.get("uniqueItems") is True:
            raise _refuse(where, "uniqueItems cannot be enforced in strict response_format schemas")
        stack.extend(sub for _, _, sub in _subschemas(node))


def _unescape(part: str) -> str:
    """A JSON pointer segment's text (RFC 6901)."""

    return part.replace("~1", "/").replace("~0", "~")


def _namespace(text: str, prefix: str) -> str:
    """A numbered prefix absent from ``text``, found in linear input work without incrementally rescanning it."""

    used = set(re.findall(re.escape(prefix) + r"([0-9]+)_", text))
    number = 0
    while str(number) in used:
        number += 1
    return f"{prefix}{number}_"


def _canonical(value: Any) -> str:
    """A JSON value's text with sorted keys: equal texts are equal values (booleans apart from numbers)."""

    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _own_types(node: dict[str, Any]) -> frozenset[str]:
    """The JSON types a checked ``node``'s own ``type``, ``enum`` and ``const`` admit (integer apart from number)."""

    found = TYPES
    if "type" in node:
        listed = node["type"] if isinstance(node["type"], list) else [node["type"]]
        found = frozenset(t for t in found if t in listed or (t == "integer" and "number" in listed))
    for values in _literals(node):
        found &= {_value_type(v) for v in values}
    return found


def _raw_options(node: dict[str, Any], where: str) -> list[dict[str, Any]]:
    """The raw-text grammar of a checked string ``node`` without ``$ref`` or ``anyOf``: its literals, its pattern or
    format, or any text that does not end the value."""

    if "enum" in node or "const" in node:
        allowed = node["enum"] if "enum" in node else [node["const"]]
        literals = [v for v in allowed if isinstance(v, str) and ("const" not in node or v == node["const"])]
        if not literals:
            raise _refuse(where, "no string satisfies this parameter's enum/const")
        if any(VALUE_END in v for v in literals):
            raise _refuse(where, f"a string parameter's value cannot contain {VALUE_END}")
        return [_const(v) for v in dict.fromkeys(literals)]
    if "pattern" in node:
        tree = _regex(node["pattern"], where)
        if all(_may_match(tree, ch) for ch in VALUE_END):
            raise _refuse(where, f"this string parameter's pattern can spell {VALUE_END}, which ends a raw value in this "
                                 "model's call markup: narrow the pattern")
        return [{"type": "regex", "pattern": node["pattern"]}]
    if "format" in node:
        return [{"type": "regex", "pattern": FORMATS[node["format"]]}]
    return [{"type": "any_text", "excludes": [VALUE_END]}]


def _subschemas(node: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(keyword, key or index, subschema) of each schema directly inside a checked ``node``; other values are data."""

    found = []
    for keyword in ("properties", "$defs", "definitions"):
        found += [(keyword, name, sub) for name, sub in (node.get(keyword) or {}).items()]
    found += [("anyOf", i, sub) for i, sub in enumerate(node.get("anyOf") or ())]
    found += [(keyword, None, node[keyword]) for keyword in ("items", "additionalProperties")
              if isinstance(node.get(keyword), dict)]
    return found


def _count(node: dict[str, Any]) -> int:
    """How many schemas a checked schema holds, itself included."""

    return 1 + sum(_count(sub) for _, _, sub in _subschemas(node))


def _refs(node: dict[str, Any]) -> set[str]:
    """Every ``$ref`` in a checked schema."""

    found = {node["$ref"]} if "$ref" in node else set()
    for _, _, sub in _subschemas(node):
        found |= _refs(sub)
    return found


def _rebase(node: dict[str, Any]) -> dict[str, Any]:
    """A copy of a checked schema whose refs into the tool's schema root point at ``#/$defs/__tf_root`` instead."""

    out = dict(node)
    ref = node.get("$ref")
    if ref is not None and not ref.startswith(("#/$defs/", "#/definitions/")):
        out["$ref"] = "#/$defs/__tf_root" + ("" if ref in ("#", "#/") else ref[1:])
    for keyword in ("properties", "$defs", "definitions"):
        if keyword in node:
            out[keyword] = {name: _rebase(sub) for name, sub in node[keyword].items()}
    if "anyOf" in node:
        out["anyOf"] = [_rebase(sub) for sub in node["anyOf"]]
    for keyword in ("items", "additionalProperties"):
        if isinstance(node.get(keyword), dict):
            out[keyword] = _rebase(node[keyword])
    return out


def _regex(pattern: Any, where: str) -> Any:
    """``pattern``'s syntax tree from CPython's regex parser (3.11+); RequestError when it is not a string or not a
    regular expression Python reads. Python's parser recurses once per group, so this also bounds the nesting xgrammar
    0.2.8's regex compiler sees (it overflowed an 8 MB thread stack on 50,000 nested groups and killed the server)."""

    from re import _parser as P

    if not isinstance(pattern, str):
        raise _refuse(where, "pattern must be a string")
    try:
        return P.parse(pattern)
    except RecursionError:
        raise _refuse(where, "pattern nests its groups too deeply") from None
    except Exception as exc:  # noqa: BLE001 - re.error and the parser's other refusals of a malformed pattern
        raise _refuse(where, f"pattern is not a regular expression this server reads: {exc}") from None


def _may_match(tree: Any, char: str) -> bool:
    """Whether the regex ``tree`` (``_regex``) could match a string holding ``char`` (True when it cannot tell)."""

    from re import _constants as C

    code = ord(char)
    digit, space, word, line = char.isdigit(), char.isspace(), char.isalnum() or char == "_", char in "\n\r"
    categories = {C.CATEGORY_DIGIT: digit, C.CATEGORY_NOT_DIGIT: not digit, C.CATEGORY_SPACE: space,
                  C.CATEGORY_NOT_SPACE: not space, C.CATEGORY_WORD: word, C.CATEGORY_NOT_WORD: not word,
                  C.CATEGORY_LINEBREAK: line, C.CATEGORY_NOT_LINEBREAK: not line}

    def within(items: Any) -> bool:
        hit, negate = False, False
        for op, av in items:
            if op is C.NEGATE:
                negate = True
            elif op is C.LITERAL:
                hit |= av == code
            elif op is C.RANGE:
                hit |= av[0] <= code <= av[1]
            elif op is C.CATEGORY:
                hit |= categories.get(av, True)
            else:
                return True
        return hit != negate

    def walk() -> bool:                        # every item of the tree, without recursion (groups nest deeply)
        stack = [tree]
        while stack:
            for op, av in stack.pop():
                if op is C.LITERAL and av == code or op is C.NOT_LITERAL and av != code or op is C.ANY:
                    return True
                if op is C.IN and within(av):
                    return True
                if op in (C.GROUPREF, C.GROUPREF_EXISTS, C.ASSERT, C.ASSERT_NOT):
                    return True
                if op is C.BRANCH:
                    stack.extend(av[1])
                elif op is C.SUBPATTERN:
                    stack.append(av[-1])
                elif op in (C.MAX_REPEAT, C.MIN_REPEAT, C.POSSESSIVE_REPEAT):
                    stack.append(av[2])
                elif op is C.ATOMIC_GROUP:
                    stack.append(av)
        return False

    found = walk()
    if not found and char.swapcase() != char:      # a case-insensitive pattern matches either case: assume it may
        code = ord(char.swapcase())
        found = walk()
    return found


# -- a request's strict tools -------------------------------------------------------------------------------------------
RAW, NULLABLE, JSON = "raw", "nullable", "json"


@dataclass(frozen=True)
class Param:
    name: str
    required: bool
    mode: str                    # RAW, NULLABLE (raw or null), JSON, or "" (a tool that is not strict: as before)


@dataclass(frozen=True, eq=False)
class Tool:
    name: str
    strict: bool
    params: tuple[Param, ...] = ()
    schemas: dict[str, Any] = field(default_factory=dict)   # a tool that is not strict: its properties, decoded as before


def _function(tool: dict[str, Any]) -> dict[str, Any]:
    """A tool's function object (``tools[].function``, else the tool itself), as ``server.tools`` reads it."""

    return tool["function"] if isinstance(tool.get("function"), dict) else tool


def strict_flag(tool: dict[str, Any], where: str) -> bool:
    """A tool's ``function.strict`` (true, false or absent/null); RequestError for any other value."""

    value = _function(tool).get("strict")
    if value is not None and not isinstance(value, bool):
        raise RequestError(f"{where}.strict must be a boolean")
    return value is True


def wanted(tools: Sequence[dict[str, Any]]) -> bool:
    """Whether any active tool asks for strict arguments (malformed ``strict`` values: RequestError)."""

    return any([strict_flag(t, f"tools[{i}].function") for i, t in enumerate(tools)])


_NAME = re.compile(r"[^\s<>]+")


class Plan:
    """A request's call grammar (``spec``, an ``engine.grammar.Spec``) and its reader, from the active tools."""

    def __init__(self, tools: Sequence[dict[str, Any]], *, required: bool, parallel: bool,
                 offered: Sequence[Any] | None = None) -> None:
        """``tools``: the active ones (after ``tool_choice``); ``offered``: the request's list, for error paths."""

        self.required, self.parallel = bool(required), bool(parallel)
        self.tools: dict[str, Tool] = {}
        tags = []
        index = {id(t): k for k, t in reversed(list(enumerate(offered or ())))}    # a tool's first place in the request
        copied = 0                                       # grammar characters of definitions copied into parameters
        for i, spec in enumerate(tools):
            where = f"tools[{index.get(id(spec), i)}].function"
            name = tool_spec_name(spec)
            if not _NAME.fullmatch(name):
                raise RequestError(f"{where}.name {name!r} cannot be written in this model's call markup")
            if name in self.tools:
                raise RequestError(f"{where}.name {name!r} is offered twice")
            parameters = _function(spec).get("parameters")
            parameters = {} if parameters is None else parameters
            if not isinstance(parameters, dict):
                raise RequestError(f"{where}.parameters must be a JSON schema object")
            if strict_flag(spec, where):
                tool, content, copies = self._strict(name, parameters, f"{where}.parameters")
                copied += copies
                if copied > MAX_GRAMMAR_BYTES:
                    raise _too_large()
            else:
                props = parameters.get("properties")
                tool = Tool(name, False, schemas={k: v for k, v in props.items() if isinstance(v, dict)}
                            if isinstance(props, dict) else {})
                content = {"type": "star", "content": _seq(
                    _const(PARAMETER), {"type": "regex", "pattern": "[^ \\t\\r\\n<>]+"}, _const(NAME_END),
                    {"type": "any_text", "excludes": [VALUE_END]}, _const(VALUE_CLOSE))}
            self.tools[name] = tool
            tags.append({"type": "tag", "begin": f"{FUNCTION}{name}{NAME_END}", "content": content, "end": CLOSE})
        if self.required:
            blanks = {"type": "regex", "pattern": f"[ \\t\\r\\n]{{0,{BLANKS}}}"}
            body = _seq(blanks, {"type": "tags_with_separator", "tags": tags, "separator": SEPARATOR,
                                 "at_least_one": True, "stop_after_first": not self.parallel}, blanks)
        else:
            body = {"type": "triggered_tags", "triggers": [OPEN], "tags": tags,
                    "stop_after_first": not self.parallel}
        text = json.dumps({"type": "structural_tag", "format": body}, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode()) > MAX_GRAMMAR_BYTES:
            raise _too_large()
        self.spec = Spec("tools", text, "tools")

    @staticmethod
    def _strict(name: str, parameters: dict[str, Any], where: str) -> tuple[Tool, dict[str, Any], int]:
        """A strict tool, its parameters' grammar, and how many characters of its definitions that grammar copies."""

        if {"anyOf", "$ref", "enum", "const"} & set(parameters):
            raise _refuse(where, "a strict tool's parameters must be an object schema (not anyOf, $ref, enum or const)")
        if parameters.get("type", "object") != "object":
            raise _refuse(where, 'a strict tool\'s parameters must have type "object"')
        top = {"type": "object", **{k: v for k, v in parameters.items() if k != "$schema"}}
        schema = _Schema(top, where)
        schema.check()
        required = set(top.get("required") or ())
        params, parts, copied = [], [], 0
        for key, sub in (top.get("properties") or {}).items():
            at = f"{where}.properties.{key}"
            if not _NAME.fullmatch(key):
                raise _refuse(at, "this parameter name cannot be written in this model's call markup")
            admitted = schema.admits(sub)
            if {"string", "other"} <= admitted:
                raise _refuse(at, "a parameter that admits strings and non-string values is ambiguous in this model's "
                                  "call markup, which writes strings raw (only string-or-null is supported)")
            if "string" in admitted:
                value = schema.raw(sub, at)
                mode = NULLABLE if "null" in admitted else RAW
                if mode == NULLABLE:
                    value = {"type": "or", "elements": [value, _const("null")]}
            else:
                own, copies = schema.json(sub, at)
                value = schema.json_format(own, at)
                mode, copied = JSON, copied + copies
            part = _seq(_const(f"{PARAMETER}{key}{NAME_END}"), value, _const(VALUE_CLOSE))
            parts.append(part if key in required else {"type": "optional", "content": part})
            params.append(Param(key, key in required, mode))
        content = _seq(*parts) if parts else _const("")
        return Tool(name, True, tuple(params)), content, copied

    # -- reading replies in this grammar --------------------------------------------------------------------------------
    def reader(self) -> "Reader":
        return Reader(self)

    def parse(self, answer: str, *, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
        """(content, calls) of a finished answer; an unfinished last call (the reply ran out) stays text."""

        reader = Reader(self)
        reader.read(answer, finished=True)
        calls = reader.calls if max_calls is None else reader.calls[:max_calls]
        return "".join(reader.text).strip(), calls or None

    def content(self, answer: str, *, finished: bool) -> str:
        """The text outside calls so far (a possible opener or an unfinished call held back while streaming)."""

        reader = Reader(self)
        reader.read(answer, finished=finished)
        return "".join(reader.text)


def _too_large() -> RequestError:
    return RequestError(f"tools: the strict tools' call grammar exceeds {MAX_GRAMMAR_BYTES >> 20} MiB (each JSON "
                        "parameter copies the definitions and root schema it refers to): send fewer or smaller tools")


def _const(value: str) -> dict[str, Any]:
    return {"type": "const_string", "value": value}


def _seq(*elements: dict[str, Any]) -> dict[str, Any]:
    return {"type": "sequence", "elements": list(elements)}


class CallMarkupError(RuntimeError):
    """A reply under the strict grammar left its markup (an internal fault): the request ends with this error."""


class Reader:
    """Follow an answer written under a ``Plan``'s grammar: its text, its calls and their streamed deltas.

    ``read`` takes the whole answer so far (each call extends the last); ``text``, ``calls`` and ``deltas`` (OpenAI
    ``tool_calls`` chunks, in order) only grow. ``feed`` returns the new deltas (the CUDA server's streamer protocol);
    ``index`` is the newest call's index and ``streamed`` whether any call has begun, as ``ToolCallStreamer``'s.
    """

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.text: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.deltas: list[dict[str, Any]] = []
        self.index = -1
        self.streamed = False
        self.pos = 0                           # the answer is read up to here
        self._state = "text"                   # text, name, params or value
        self._at = 0                           # where the current call's opener begins
        self._head: dict[str, Any] = {}        # the current call's first delta
        self._tool: Tool | None = None
        self._next = 0                         # the strict tool's next declared parameter
        self._args: list[str] = []             # the current call's argument fragments
        self._loose: dict[str, Any] | None = None   # a tool that is not strict: its arguments, sent whole at its close
        self._param: Param | None = None
        self._start = 0                        # where the current value's text begins
        self._sent = 0                         # how much of a raw value has gone out as deltas

    def feed(self, answer: str) -> list[dict[str, Any]]:
        before = len(self.deltas)
        self.read(answer, finished=False)
        return self.deltas[before:]

    def read(self, answer: str, *, finished: bool) -> None:
        while True:
            if self._state == "text":
                at = answer.find(OPEN, self.pos)
                end = at if at >= 0 else len(answer) - (0 if finished else _partial_tag(answer[self.pos:], OPEN))
                if end > self.pos:
                    self.text.append(answer[self.pos:end])
                    self.pos = end
                if at < 0:
                    return
                self._state, self._at = "name", at
            elif self._state == "name":
                end = answer.find(NAME_END, self.pos + len(FUNCTION))
                if not answer.startswith(FUNCTION, self.pos) or end < 0:
                    return self._wait(answer, finished, FUNCTION)
                name = answer[self.pos + len(FUNCTION):end]
                tool = self.plan.tools.get(name)
                if tool is None:
                    raise CallMarkupError(f"the reply called {name!r}, which its grammar does not offer")
                self.index += 1
                self.streamed = True
                self._head = {"index": self.index, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                              "function": {"name": name, "arguments": ""}}
                self.deltas.append({"tool_calls": [self._head]})
                self._tool, self._next, self._args = tool, 0, []
                self._loose = None if tool.strict else {}
                self.pos, self._state = end + len(NAME_END), "params"
            elif self._state == "params":
                if answer.startswith(CLOSE, self.pos):
                    if self._loose is not None:        # as the lenient parser: a repeated name keeps its last value
                        self._emit(json.dumps(self._loose, ensure_ascii=False, separators=(",", ":")))
                    else:
                        self._emit("}" if self._args else "{}")
                    self.calls.append({"id": self._head["id"], "type": "function",
                                       "function": {"name": self._head["function"]["name"],
                                                    "arguments": "".join(self._args)}})
                    self.pos, self._state = self.pos + len(CLOSE), "text"
                    continue
                end = answer.find(NAME_END, self.pos + len(PARAMETER))
                if not answer.startswith(PARAMETER, self.pos) or end < 0:
                    return self._wait(answer, finished, PARAMETER, CLOSE)
                self._open(answer[self.pos + len(PARAMETER):end])
                self.pos = self._start = end + len(NAME_END)
                self._state = "value"
            elif not self._value(answer, finished):
                return

    def _emit(self, fragment: str) -> None:
        self._args.append(fragment)
        self.deltas.append({"tool_calls": [{"index": self.index, "function": {"arguments": fragment}}]})

    def _wait(self, answer: str, finished: bool, *marks: str) -> None:
        """Markup still arriving: wait; a finished answer's unfinished call (it ran out) stays text; other markup is a
        fault."""

        rest = answer[self.pos:]
        if marks and not any(mark.startswith(rest) or rest.startswith(mark) for mark in marks):
            raise CallMarkupError("the reply's call markup left its grammar")
        if not finished:
            return
        self.text.append(answer[self._at:])
        self.pos, self._state = len(answer), "text"

    def _open(self, key: str) -> None:
        tool = self._tool
        assert tool is not None
        if tool.strict:
            names = [p.name for p in tool.params]
            at = names.index(key, self._next) if key in names[self._next:] else -1
            if at < 0 or any(p.required for p in tool.params[self._next:at]):
                raise CallMarkupError(f"the reply wrote parameter {key!r} of {tool.name!r} out of its grammar")
            self._param, self._next = tool.params[at], at + 1
        else:
            self._param = Param(key, False, "")
            return
        raw = self._param.mode == RAW
        self._emit(("," if self._args else "{") + json.dumps(key, ensure_ascii=False) + (':"' if raw else ":"))
        self._sent = 0

    def _value(self, answer: str, finished: bool) -> bool:
        """Read the current value as far as the answer goes; True once its closing line has been read."""

        param = self._param
        assert param is not None and self._tool is not None
        if param.mode == JSON:                 # JSON writes no raw newline before "<": the first such line closes it
            end = answer.find("\n" + VALUE_END, self._start)
            close = end + 1 + len(VALUE_END) if end >= 0 else -1
        else:                                  # raw text never holds VALUE_END; the newline before it is framing
            close = answer.find(VALUE_END, self._start)
            end = close - 1 if close >= 0 else -1
            if close >= 0 and (end < self._start or answer[end] != "\n"):
                raise CallMarkupError("a raw parameter value did not end with its framing newline")
            close = close + len(VALUE_END) if close >= 0 else -1
        if close < 0 or close >= len(answer):
            if param.mode == RAW:              # stream raw text, holding back what may begin its closing line
                tail = answer[max(self._start, len(answer) - len(VALUE_END)):]
                upto = end if close >= 0 else len(answer) - _partial_tag(tail, "\n" + VALUE_END)
                if upto - self._start > self._sent:
                    self._emit(json.dumps(answer[self._start + self._sent:upto], ensure_ascii=False)[1:-1])
                    self._sent = upto - self._start
            if finished:
                self._wait(answer, True)
            return False
        if answer[close] != "\n":
            raise CallMarkupError("a parameter's closing line did not end")
        value = answer[self._start:end]
        if param.mode == RAW:
            rest = value[self._sent:]
            self._emit((json.dumps(rest, ensure_ascii=False)[1:-1] if rest else "") + '"')
        elif param.mode == NULLABLE:
            self._emit("null" if value == "null" else json.dumps(value, ensure_ascii=False))
        elif param.mode == JSON:
            self._emit(value)
        else:                                  # a tool that is not strict: decoded as the lenient parser does
            assert self._loose is not None
            self._loose[param.name] = decode_parameter(value, self._tool.schemas.get(param.name, {}))
        self.pos, self._state = close + 1, "params"
        return True


__all__ = ["BLANKS", "CallMarkupError", "FORMATS", "JSON", "NULLABLE", "PROBE", "Param", "Plan", "RAW", "Reader",
           "Tool", "strict_flag", "template_supported", "wanted"]
