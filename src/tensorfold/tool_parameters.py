"""Schema-aware decoding of Qwen XML parameter text."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any


def parameter_schemas(tools: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Each offered tool's parameter schemas by lowercase name; a spec that is not an object reads as untyped."""

    result = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool["function"] if isinstance(tool.get("function"), dict) else tool
        parameters = function.get("parameters") or function.get("input_schema") or {}
        properties = parameters.get("properties") or {} if isinstance(parameters, dict) else {}
        result[str(function.get("name", "")).lower()] = {
            name: schema for name, schema in properties.items() if isinstance(schema, dict)
        } if isinstance(properties, dict) else {}
    return result


_JSON_TYPES = frozenset({'string', 'array', 'object', 'boolean', 'integer', 'number', 'null'})


def _parameter_types(schema: dict[str, Any]) -> frozenset[str] | None:
    """Conservative type bounds, not a replacement JSON Schema validator.

    Compositions union their branch types; an explicit enclosing type narrows
    that bound. Unknown branches remain unknown. Iterative traversal handles
    deep compositions and refuses cyclic programmatic schemas without guessing.
    """
    if not isinstance(schema, dict):
        return None
    pending, active, resolved = [(schema, False)], set(), {}
    while pending:
        node, finish = pending.pop()
        identity = id(node)
        if not finish:
            if identity in resolved:
                continue
            if identity in active:
                return None
            active.add(identity)
            pending.append((node, True))
            for name in ('anyOf', 'oneOf'):
                if name not in node:
                    continue
                branches = node[name]
                if not isinstance(branches, list) or not branches or any(not isinstance(b, dict) for b in branches):
                    return None
                pending.extend((branch, False) for branch in reversed(branches))
            continue
        allowed = None
        if 'type' in node:
            kind = node['type']
            kinds = [kind] if isinstance(kind, str) else kind
            if (not isinstance(kinds, list) or not kinds
                    or any(not isinstance(k, str) or k not in _JSON_TYPES for k in kinds)):
                return None
            # JSON Schema number includes integer: intersect semantic domains,
            # not disjoint spelling labels, when an enclosing type narrows a union.
            allowed = frozenset(kinds) | ({'integer'} if 'number' in kinds else set())
        for name in ('anyOf', 'oneOf'):
            if name in node:
                branches = [resolved[id(branch)] for branch in node[name]]
                if all(branch is not None for branch in branches):
                    union = frozenset().union(*branches)
                    allowed = union if allowed is None else allowed & union
                # An unknown branch cannot widen an explicit enclosing type.
        resolved[identity] = allowed
        active.remove(identity)
    return resolved[id(schema)]


def typed_parameter(schema: dict[str, Any]) -> bool:
    kinds = _parameter_types(schema)
    # XML writes strings without a JSON delimiter. If both raw text and a
    # converted value are admitted, preserve the text instead of guessing.
    return bool(kinds) and 'string' not in kinds


def closed_json(text: str) -> str | None:
    """``text`` with the arrays and objects it left open closed, or None if it doesn't only stop short of them."""

    closers, in_string, escaped = [], False, False
    for ch in text:
        if in_string:
            escaped, in_string = (False, True) if escaped else (ch == "\\", ch != '"')
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            closers.append("]" if ch == "[" else "}")
        elif ch in "]}" and (not closers or closers.pop() != ch):
            return None
    return text.rstrip() + "".join(reversed(closers)) if closers and not in_string else None


_PY_WORDS = {"true": True, "false": False, "none": None, "null": None}


def _python_literal(text: str) -> Any:
    """``text`` as a Python literal (``True``, ``['a']``, ``{'k': None}``), tuples as lists; ValueError if not one."""

    word = _PY_WORDS.get(text.strip().lower(), ...)
    if word is not ...:
        return word

    def lists(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return [lists(x) for x in v]
        if isinstance(v, dict):
            return {k: lists(x) for k, x in v.items()}
        return v

    try:
        return lists(ast.literal_eval(text.strip()))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(str(exc)) from None


def decode_parameter(value: str, schema: dict[str, Any], *, python: bool = True) -> Any:
    """``value`` as the schema's type when it spells one, else the text; ``python`` also reads Python's spelling."""

    kinds = _parameter_types(schema)
    if not kinds or 'string' in kinds:
        return value
    predicates = {
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: type(v) is int,
        "number": lambda v: type(v) in (int, float),
        "null": lambda v: v is None,
    }
    def valid(parsed):
        return any(predicates[kind](parsed) for kind in kinds)
    # a model can end an object or array value one closer short (#87): the value closed is what it meant
    for text in (value, closed_json(value) if kinds & {"array", "object"} else None):
        if text is None:
            continue
        try:
            parsed = json.loads(text)
            json.dumps(parsed, allow_nan=False)
        except (ValueError, TypeError):
            continue
        return parsed if valid(parsed) else value
    # not JSON: Python's spelling (True, None, single-quoted strings), as vLLM's Qwen parsers also accept
    if not python:
        return value
    try:
        parsed = _python_literal(value)
        json.dumps(parsed, allow_nan=False)
    except (ValueError, TypeError):
        return value
    return parsed if valid(parsed) else value
