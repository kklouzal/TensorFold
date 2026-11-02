"""Bounded owned EBNF for constraints the pinned JSON converter cannot preserve."""

from __future__ import annotations

import json
from typing import Any

from tensorfold.server.errors import RequestError


def _literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _counted_paths(edges: list[dict[int, str]], endings: dict[int, str], low: int, high: int | None) -> str:
    """Small character-DFA paths with logarithmic bound decomposition.

    Each edge consumes one decoded character. Binary concatenation preserves
    entry/exit states; no advisory lookahead or per-character range expansion.
    """
    size, rules = len(edges), {}
    for i, targets in enumerate(edges):
        for j, body in targets.items():
            rules[f"edge_{i}_{j}"] = body
    exact_cache, upto_cache = {}, {}

    def combine(parts):
        return " | ".join(" ".join(part) for part in parts) or None

    def exact(n, i, j):
        if n == 0:
            return '""' if i == j else None
        if n == 1:
            return f"edge_{i}_{j}" if j in edges[i] else None
        key = (n, i, j)
        if key not in exact_cache:
            name = f"exact_{n}_{i}_{j}"
            parts = []
            for k in range(size):
                left, right = exact(n // 2, i, k), exact(n - n // 2, k, j)
                if left is not None and right is not None:
                    parts.append((left, right))
            body = combine(parts)
            exact_cache[key] = name if body is not None else None
            if body is not None:
                rules[name] = body
        return exact_cache[key]

    def upto(n, i, j):
        if n == 0:
            return '""' if i == j else None
        key = (n, i, j)
        if key not in upto_cache:
            name = f"upto_{n}_{i}_{j}"
            power = 1 << (n.bit_length() - 1)
            first = upto(power - 1, i, j)
            parts = [(first,)] if first is not None else []
            for k in range(size):
                left, right = exact(power, i, k), upto(n - power, k, j)
                if left is not None and right is not None:
                    parts.append((left, right))
            body = combine(parts)
            upto_cache[key] = name if body is not None else None
            if body is not None:
                rules[name] = body
        return upto_cache[key]

    if high is None:
        for i in range(size):
            parts = [(endings[i],)] if i in endings else []
            parts.extend((f"edge_{i}_{j}", f"tail_{j}") for j in edges[i])
            rules[f"tail_{i}"] = combine(parts)
    roots = []
    for i in range(size):
        prefix = exact(low, 0, i)
        if prefix is None:
            continue
        if high is None:
            roots.append((prefix, f"tail_{i}"))
        else:
            for j, ending in endings.items():
                tail = upto(high - low, i, j)
                if tail is not None:
                    roots.append((prefix, tail, ending))
    return ("root ::= " + combine(roots) + "\n"
            + "\n".join(f"{name} ::= {body}" for name, body in rules.items()))


def string_length(low: int, high: int | None, *, exclude: str | None = None) -> str:
    """Count JSON-decoded characters, or actual raw tool characters.

    The JSON states remember a preceding high surrogate escape. Its following
    low escape is a zero-cost prefix of the next character or an end suffix;
    it cannot instead become a second character. Lone escapes count once.
    Raw states remember delimiter prefixes and omit its completing transition.
    """
    if exclude is not None:
        alphabet = set(exclude)
        edges = []
        escaped = "".join("\\" + char if char in "\\]-^" else char for char in sorted(alphabet))
        for i in range(len(exclude)):
            targets = {0: [f"[^{escaped}]"]}
            for char in sorted(alphabet):
                suffix = exclude[:i] + char
                if suffix.endswith(exclude):
                    continue
                nxt = max(k for k in range(len(exclude)) if suffix.endswith(exclude[:k]))
                targets.setdefault(nxt, []).append(_literal(char))
            edges.append({j: " | ".join(words) for j, words in targets.items()})
        return _counted_paths(edges, dict.fromkeys(range(len(edges)), '""'), low, high)
    edges = [{0: "atom | low", 1: "high"},
             {0: "atom | low atom | low low", 1: "high | low high"}]
    text = _counted_paths(edges, {0: '""', 1: '("" | low)'}, low, high)
    text = text.replace("root ::= ", 'root ::= "\\\"" body "\\\""\nbody ::= ', 1)
    return text + r'''
atom ::= [^"\\\x00-\x1f] | "\\" ["\\/bfnrt] | "\\u" plain
hex ::= [0-9a-fA-F]
plain ::= [0-9a-cA-Ce-fE-F] hex hex hex | [dD] [0-7] hex hex
high ::= "\\u" [dD] [89aAbB] hex hex
low ::= "\\u" [dD] [c-fC-F] hex hex'''


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
