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

from tensorfold.server.errors import RequestError
from tensorfold.tool_parameters import decode_parameter

BLANKS = 32                      # blank characters allowed around required calls and in JSON values (as response_format)

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
              "function": {"name": "tfprobe_a", "arguments": {"tfprobe_x": "tfprobe_v", "tfprobe_y": 7}}},
             {"id": "call_1", "type": "function", "function": {"name": "tfprobe_b", "arguments": {}}}]}]
_PROBED = (f"{FUNCTION}tfprobe_a{NAME_END}{PARAMETER}tfprobe_x{NAME_END}tfprobe_v{VALUE_CLOSE}"
           f"{PARAMETER}tfprobe_y{NAME_END}7{VALUE_CLOSE}{CLOSE}{SEPARATOR}{FUNCTION}tfprobe_b{NAME_END}{CLOSE}")


def template_supported(rendered: str) -> bool:
    """Whether a chat template rendered ``PROBE``'s calls exactly as this module writes and reads them."""

    return _PROBED in rendered


# -- the supported JSON-schema subset ---------------------------------------------------------------------------------
TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})
_ANNOTATIONS = frozenset({"title", "description", "default", "examples", "$comment", "deprecated", "readOnly",
                          "writeOnly"})
_KEYWORDS = _ANNOTATIONS | {"type", "enum", "const", "anyOf", "$ref", "$defs", "definitions", "properties", "required",
                            "additionalProperties", "items", "minItems", "maxItems", "pattern", "format", "minimum",
                            "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
# OpenAI's strict-mode limits (developers.openai.com/api/docs/guides/structured-outputs#supported-schemas)
MAX_NESTING, MAX_PROPERTIES, MAX_ENUM_VALUES, MAX_SCHEMA_CHARS = 10, 5000, 1000, 120_000
_LENGTHS = {"minLength", "maxLength"}
_RANGES = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
# keywords that constrain one JSON type: xgrammar 0.2.8 applies them only beside an explicit type
_TYPED = {"pattern", "format", "multipleOf", "items", "minItems", "maxItems", "properties", "required",
          "additionalProperties", *_RANGES}

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


class _Schema:
    """One tool's ``parameters``: checked against the supported subset, refs resolved within it."""

    def __init__(self, root: dict[str, Any], where: str) -> None:
        self.root, self.where = root, where
        self.properties = self.chars = self.enums = 0

    def resolve(self, ref: Any, where: str) -> dict[str, Any]:
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise _refuse(where, "$ref must point into the same schema (#...)")
        node: Any = self.root
        if ref not in ("#", "#/"):
            if not ref.startswith("#/"):
                raise _refuse(where, f"$ref {ref!r} is not a JSON pointer")
            for part in ref[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                if isinstance(node, dict) and part in node:
                    node = node[part]
                elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                    node = node[int(part)]
                else:
                    raise _refuse(where, f"$ref {ref!r} does not resolve")
        if not isinstance(node, dict):
            raise _refuse(where, f"$ref {ref!r} does not name a schema object")
        return node

    def check(self, node: Any, where: str, depth: int) -> None:
        """Every keyword of ``node`` (and its subschemas) supported, well formed and within OpenAI's limits."""

        if not isinstance(node, dict):
            raise _refuse(where, "a schema must be a JSON object")
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
        # xgrammar 0.2.8 lets enum, const, $ref and anyOf override what stands beside them, and drops type-specific
        # keywords without a type: those shapes are refused rather than left unenforced
        own = set(node) - _ANNOTATIONS - {"$defs", "definitions"}
        if "$ref" in node and own - {"$ref"}:
            raise _refuse(where, "$ref takes no constraints beside it here: put them in the schema it names")
        if ("enum" in node or "const" in node) and own - {"enum", "const", "type"}:
            raise _refuse(where, "enum and const take only type beside them here")
        if "anyOf" in node:
            if own - {"anyOf", "type"}:
                raise _refuse(where, "anyOf takes only type beside it here: put the other constraints in each branch")
            if kinds is not None and isinstance(node["anyOf"], list) and not all(
                    isinstance(b, dict) and _within(self.types(b), declared) for b in node["anyOf"]):
                raise _refuse(where, "each anyOf branch must admit only the types listed beside it")
        typed = own & _TYPED
        if typed and kinds is None:
            raise _refuse(where, f"{', '.join(sorted(typed))} need a type beside them")
        if typed and len(declared - {"null"}) > 1:
            raise _refuse(where, f"{', '.join(sorted(typed))} need a single type (besides null)")
        for values in ([node["enum"]] if isinstance(node.get("enum"), list) else []) + (
                [[node["const"]]] if "const" in node else []):
            if kinds is not None and not _within({_value_type(v) for v in values}, declared):
                raise _refuse(where, "every enum and const value must have a type listed beside it")
        if "enum" in node:
            if not isinstance(node["enum"], list) or not node["enum"]:
                raise _refuse(where, "enum must be a non-empty list")
            self.enums += len(node["enum"])
            self.chars += sum(len(v) for v in node["enum"] if isinstance(v, str))
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
        if "pattern" in node and not isinstance(node["pattern"], str):
            raise _refuse(where, "pattern must be a string")
        if "format" in node and node["format"] not in FORMATS:
            raise _refuse(where, f"format must be one of {', '.join(FORMATS)}")
        if "$ref" in node:
            self.resolve(node["$ref"], where)
        for key in ("$defs", "definitions"):
            if key in node:
                if not isinstance(node[key], dict):
                    raise _refuse(where, f"{key} must be an object of schemas")
                for name, sub in node[key].items():
                    self.chars += len(name)
                    self.check(sub, f"{where}.{key}.{name}", depth)
        if "anyOf" in node:
            if not isinstance(node["anyOf"], list) or not node["anyOf"]:
                raise _refuse(where, "anyOf must be a non-empty list of schemas")
            for i, sub in enumerate(node["anyOf"]):
                self.check(sub, f"{where}.anyOf[{i}]", depth)
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
                self.check(sub, f"{where}.properties.{name}", depth + 1)
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
            self.check(extra, f"{where}.additionalProperties", depth + 1)
        elif extra is not None and not isinstance(extra, bool):
            raise _refuse(where, "additionalProperties must be a boolean or a schema")
        if "items" in node:
            self.check(node["items"], f"{where}.items", depth + 1)
        if self.properties > MAX_PROPERTIES or self.enums > MAX_ENUM_VALUES or self.chars > MAX_SCHEMA_CHARS:
            raise _refuse(self.where, f"strict schemas are limited to {MAX_PROPERTIES} properties, {MAX_ENUM_VALUES} "
                                      f"enum values and {MAX_SCHEMA_CHARS} characters of names and values")

    def types(self, node: dict[str, Any], seen: frozenset[int] = frozenset()) -> set[str]:
        """The JSON types ``node`` admits (integer apart from number), its refs followed."""

        if id(node) in seen:
            return set(TYPES)
        seen = seen | {id(node)}
        found = set(TYPES)
        if "$ref" in node:
            found &= self.types(self.resolve(node["$ref"], self.where), seen)
        if "type" in node:
            listed = set(node["type"] if isinstance(node["type"], list) else [node["type"]])
            found = {t for t in found if t in listed or (t == "integer" and "number" in listed)}
        for values in ([node["enum"]] if isinstance(node.get("enum"), list) else []) + (
                [[node["const"]]] if "const" in node else []):
            found &= {_value_type(v) for v in values} | ({"number"} if any(
                _value_type(v) == "number" for v in values) else set())
        if isinstance(node.get("anyOf"), list):
            found &= set().union(*(self.types(b, seen) for b in node["anyOf"] if isinstance(b, dict)))
        return found

    def admits(self, node: dict[str, Any], seen: frozenset[int] = frozenset()) -> tuple[bool, bool, bool]:
        """Whether ``node`` admits (strings, null, any other JSON value), its refs followed."""

        if id(node) in seen:                                  # a ref cycle without a type: anything
            return True, True, True
        seen = seen | {id(node)}
        result = (True, True, True)

        def meet(other: tuple[bool, bool, bool]) -> None:
            nonlocal result
            result = tuple(a and b for a, b in zip(result, other))  # type: ignore[assignment]

        if "$ref" in node:
            meet(self.admits(self.resolve(node["$ref"], self.where), seen))
        if "type" in node:
            types = set(node["type"] if isinstance(node["type"], list) else [node["type"]])
            meet(("string" in types, "null" in types, bool(types - {"string", "null"})))
        for values in ([node["enum"]] if "enum" in node else []) + ([[node["const"]]] if "const" in node else []):
            kinds = {_kind(v) for v in values}
            meet(("string" in kinds, "null" in kinds, bool(kinds - {"string", "null"})))
        if "anyOf" in node:
            branches = [self.admits(b, seen) for b in node["anyOf"]]
            meet(tuple(any(b[i] for b in branches) for i in range(3)))  # type: ignore[arg-type]
        return result

    def raw(self, node: dict[str, Any], where: str) -> dict[str, Any]:
        """The grammar of the strings ``node`` admits, as raw text inside a parameter."""

        if "$ref" in node:
            if set(node) - _ANNOTATIONS - {"$ref"}:
                raise _refuse(where, "a string parameter's $ref cannot have sibling constraints")
            return self.raw(self.resolve(node["$ref"], where), where)
        if "anyOf" in node:
            if {"enum", "const", "pattern", "format"} & set(node):
                raise _refuse(where, "a string parameter cannot combine anyOf with enum, const, pattern or format")
            branches = [self.raw(b, f"{where}.anyOf[{i}]") for i, b in enumerate(node["anyOf"])
                        if self.admits(b)[0]]
            return branches[0] if len(branches) == 1 else {"type": "or", "elements": branches}
        literals = None
        if "enum" in node:
            literals = [v for v in node["enum"] if isinstance(v, str)]
        if isinstance(node.get("const"), str):
            literals = [v for v in (literals if literals is not None else [node["const"]]) if v == node["const"]]
        shapes = [k for k in ("pattern", "format") if k in node] + (["enum/const"] if literals is not None else [])
        if len(shapes) > 1:
            raise _refuse(where, f"a string parameter takes one of enum/const, pattern or format, not {shapes}")
        if literals is not None:
            if not literals:
                raise _refuse(where, "no string satisfies this parameter's enum/const")
            if any(VALUE_END in v for v in literals):
                raise _refuse(where, f"a string parameter's value cannot contain {VALUE_END}")
            unique = list(dict.fromkeys(literals))
            options = [{"type": "const_string", "value": v} for v in unique]
            return options[0] if len(options) == 1 else {"type": "or", "elements": options}
        if "pattern" in node:
            if all(_may_match(node["pattern"], ch) for ch in set(VALUE_END)):
                raise _refuse(where, f"this string parameter's pattern can spell {VALUE_END}, which ends a raw value "
                                     "in this model's call markup: narrow the pattern")
            return {"type": "regex", "pattern": node["pattern"]}
        if "format" in node:
            return {"type": "regex", "pattern": FORMATS[node["format"]]}
        return {"type": "any_text", "excludes": [VALUE_END]}

    def json(self, node: dict[str, Any]) -> dict[str, Any]:
        """``node`` as a schema of its own (refs into the tool's schema kept), for a JSON value's grammar."""

        defs = {key: _rebase(self.root[key]) for key in ("$defs", "definitions") if isinstance(self.root.get(key), dict)}
        refs = _refs(node).union(*(_refs(sub) for d in defs.values() for sub in d.values()))
        if any(not r.startswith(("#/$defs/", "#/definitions/")) for r in refs):     # "#" or "#/properties/...": the root
            if "__tf_root" in defs.get("$defs", {}):
                raise _refuse(self.where, "$defs may not name __tf_root")
            defs["$defs"] = {**defs.get("$defs", {}), "__tf_root": _rebase(self.root)}
        sub = _rebase(node)
        for key, value in defs.items():             # the tool's own definitions win: its refs were written for them
            sub[key] = {**sub[key], **value} if isinstance(sub.get(key), dict) else value
        return sub


def _subschemas(node: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(keyword, key or index, subschema) of each schema directly inside a checked ``node``; other values are data."""

    found = []
    for keyword in ("properties", "$defs", "definitions"):
        found += [(keyword, name, sub) for name, sub in (node.get(keyword) or {}).items()]
    found += [("anyOf", i, sub) for i, sub in enumerate(node.get("anyOf") or ())]
    found += [(keyword, None, node[keyword]) for keyword in ("items", "additionalProperties")
              if isinstance(node.get(keyword), dict)]
    return found


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


def _may_match(pattern: str, char: str) -> bool:
    """Whether the regex ``pattern`` could match a string holding ``char`` (True when it cannot be analysed)."""

    try:
        from re import _constants as C, _parser as P   # CPython's regex parser: the pattern's syntax tree
        tree = P.parse(pattern)
    except Exception:  # noqa: BLE001 - a pattern Python cannot read: assume the worst
        return True
    code = ord(char)

    def category(cat: Any) -> bool:
        tests = {C.CATEGORY_DIGIT: char.isdigit(), C.CATEGORY_SPACE: char.isspace(),
                 C.CATEGORY_WORD: char.isalnum() or char == "_", C.CATEGORY_LINEBREAK: char in "\n\r"}
        for name, value in list(tests.items()):
            tests[getattr(C, str(name).replace("CATEGORY_", "CATEGORY_NOT_"), None)] = not value
        return tests.get(cat, True)

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
                hit |= category(av)
            else:
                return True
        return hit != negate

    def walk(items: Any) -> bool:
        for op, av in items:
            if op is C.LITERAL and av == code or op is C.NOT_LITERAL and av != code or op is C.ANY:
                return True
            if op is C.IN and within(av):
                return True
            if op is C.BRANCH and any(walk(b) for b in av[1]):
                return True
            if op is C.SUBPATTERN and walk(av[-1]):
                return True
            if op in (C.MAX_REPEAT, C.MIN_REPEAT, getattr(C, "POSSESSIVE_REPEAT", None)) and walk(av[2]):
                return True
            if op is getattr(C, "ATOMIC_GROUP", None) and walk(av):
                return True
            if op in (C.GROUPREF, C.GROUPREF_EXISTS, C.ASSERT, C.ASSERT_NOT):
                return True
        return False

    found = walk(tree)
    if not found and char.swapcase() != char:      # a case-insensitive pattern matches either case: assume it may
        code = ord(char.swapcase())
        found = walk(tree)
    return found


# -- a request's strict tools -------------------------------------------------------------------------------------------
RAW, NULLABLE, JSON = "raw", "nullable", "json"


@dataclass(frozen=True)
class Param:
    name: str
    required: bool
    mode: str                    # RAW, NULLABLE (raw or null) or JSON


@dataclass(frozen=True, eq=False)
class Tool:
    name: str
    strict: bool
    params: tuple[Param, ...] = ()
    schemas: dict[str, Any] = field(default_factory=dict)   # a tool that is not strict: its properties, decoded as before


def _tool_name(tool: Any) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    return str((function if isinstance(function, dict) else tool).get("name") or "").strip()


def strict_flag(tool: Any, where: str) -> bool:
    """A tool's ``function.strict`` (true, false or absent/null); RequestError for any other value."""

    function = tool.get("function") if isinstance(tool, dict) else None
    function = function if isinstance(function, dict) else tool
    value = function.get("strict") if isinstance(function, dict) else None
    if value is not None and not isinstance(value, bool):
        raise RequestError(f"{where}.strict must be a boolean")
    return value is True


def wanted(tools: Sequence[dict[str, Any]]) -> bool:
    """Whether any active tool asks for strict arguments (malformed ``strict`` values: RequestError)."""

    return any([strict_flag(t, f"tools[{i}].function") for i, t in enumerate(tools)])


_NAME = re.compile(r"[^\s<>]+")


class Plan:
    """A request's call grammar (``spec``, an ``engine.grammar.Spec``) and its reader, from the active tools."""

    def __init__(self, tools: Sequence[dict[str, Any]], *, required: bool, parallel: bool) -> None:
        from tensorfold.engine.grammar import Spec

        self.required, self.parallel = bool(required), bool(parallel)
        self.tools: dict[str, Tool] = {}
        tags = []
        for i, spec in enumerate(tools):
            where = f"tools[{i}].function"
            name = _tool_name(spec)
            if not _NAME.fullmatch(name):
                raise RequestError(f"{where}.name {name!r} cannot be written in this model's call markup")
            if name in self.tools:
                raise RequestError(f"{where}.name {name!r} is offered twice")
            function = spec["function"] if isinstance(spec.get("function"), dict) else spec
            parameters = function.get("parameters")
            parameters = {} if parameters is None else parameters
            if not isinstance(parameters, dict):
                raise RequestError(f"{where}.parameters must be a JSON schema object")
            if strict_flag(spec, where):
                tool, content = self._strict(name, parameters, f"{where}.parameters")
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
        self.spec = Spec("tools", text, "tools")

    @staticmethod
    def _strict(name: str, parameters: dict[str, Any], where: str) -> tuple[Tool, dict[str, Any]]:
        if {"anyOf", "$ref", "enum", "const"} & set(parameters):
            raise _refuse(where, "a strict tool's parameters must be an object schema (not anyOf, $ref, enum or const)")
        if parameters.get("type", "object") != "object":
            raise _refuse(where, 'a strict tool\'s parameters must have type "object"')
        top = {"type": "object", **{k: v for k, v in parameters.items() if k != "$schema"}}
        schema = _Schema(top, where)
        schema.check(top, where, 0)
        required = set(top.get("required") or ())
        params, parts = [], []
        for key, sub in (top.get("properties") or {}).items():
            at = f"{where}.properties.{key}"
            if not _NAME.fullmatch(key):
                raise _refuse(at, "this parameter name cannot be written in this model's call markup")
            strings, null, other = schema.admits(sub)
            if strings and other:
                raise _refuse(at, "a parameter that admits strings and non-string values is ambiguous in this model's "
                                  "call markup, which writes strings raw (only string-or-null is supported)")
            if strings:
                value = schema.raw(sub, at)
                mode = NULLABLE if null else RAW
                if null:
                    value = {"type": "or", "elements": [value, _const("null")]}
            else:
                value = {"type": "json_schema", "json_schema": schema.json(sub), "style": "json",
                         "max_whitespace_cnt": BLANKS}
                mode = JSON
            part = _seq(_const(f"{PARAMETER}{key}{NAME_END}"), value, _const(VALUE_CLOSE))
            parts.append(part if key in required else {"type": "optional", "content": part})
            params.append(Param(key, key in required, mode))
        content = _seq(*parts) if parts else _const("")
        return Tool(name, True, tuple(params)), content

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


def _const(value: str) -> dict[str, Any]:
    return {"type": "const_string", "value": value}


def _seq(*elements: dict[str, Any]) -> dict[str, Any]:
    return {"type": "sequence", "elements": list(elements)}


def _partial(text: str, tag: str) -> int:
    """How many characters at the end of ``text`` could begin ``tag``."""

    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


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
                end = at if at >= 0 else len(answer) - (0 if finished else _partial(answer[self.pos:], OPEN))
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
                self.pos, self._state = end + len(NAME_END), "params"
            elif self._state == "params":
                if answer.startswith(CLOSE, self.pos):
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
                upto = end if close >= 0 else len(answer) - _partial(tail, "\n" + VALUE_END)
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
            schema = self._tool.schemas.get(param.name, {})
            self._emit(json.dumps(decode_parameter(value, schema), ensure_ascii=False, separators=(",", ":")))
        self.pos, self._state = close + 1, "params"
        return True


__all__ = ["BLANKS", "CallMarkupError", "FORMATS", "JSON", "NULLABLE", "PROBE", "Param", "Plan", "RAW", "Reader",
           "Tool", "strict_flag", "template_supported", "wanted"]
