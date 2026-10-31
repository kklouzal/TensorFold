"""Exact finite-enum uniqueness for strict JSON function arguments."""

from __future__ import annotations

import json
from typing import Any

from tensorfold.server.errors import RequestError


def _literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def unique_array(values: list[Any], low: int, high: int | None, blanks: int) -> str:
    """An array over a finite enum, with one rule per reachable remaining subset and no duplicate JSON values."""

    high = len(values) if high is None else min(high, len(values))
    if low > high:
        raise RequestError("uniqueItems and the array bounds admit no value")
    words = [_literal(json.dumps(v, ensure_ascii=False, separators=(",", ":"))) for v in values]
    rules = {}

    def state(mask: int) -> str:
        name = f"subset_{mask}"
        if name in rules:
            return name
        used = mask.bit_count()
        parts = ['""'] if used and used >= low else []
        rules[name] = ""                     # child masks strictly add an item, so recursion is bounded by enum size
        if used < high:
            for i, word in enumerate(words):
                if not mask & (1 << i):
                    child = state(mask | (1 << i))
                    parts.append((('blank "," blank ' if used else "") + word + " " + child))
        rules[name] = " | ".join(parts)
        return name

    roots = ['"[" blank "]"'] if low == 0 else []
    if high:
        roots.append(f'"[" blank {state(0)} blank "]"')
    return ('root ::= ' + " | ".join(roots) + f'\nblank ::= [ \\t\\r\\n]{{0,{blanks}}}\n'
            + "\n".join(f"{name} ::= ({body})" for name, body in rules.items()))
