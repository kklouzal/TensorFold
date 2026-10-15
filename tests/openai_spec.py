"""Check reply bodies and stream frames against the pinned OpenAI schemas in fixtures/openai/spec_subset.json.

The subset (tools/openai_spec_subset.py) keeps only constraining JSON Schema keywords; this validator implements exactly
those: type, properties, required, enum, const, items, anyOf, oneOf, allOf, $ref, additionalProperties, minimum,
maximum, minItems and maxItems. Unknown properties are allowed unless a schema says otherwise, as the spec allows the
server's own extensions (``tensorfold``, ``reasoning_content``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SUBSET = Path(__file__).parent / "fixtures" / "openai" / "spec_subset.json"
_TYPES = {"null": lambda v: v is None, "boolean": lambda v: isinstance(v, bool),
          "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
          "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "string": lambda v: isinstance(v, str), "array": lambda v: isinstance(v, list),
          "object": lambda v: isinstance(v, dict)}


@lru_cache(maxsize=1)
def schemas() -> dict[str, Any]:
    return json.loads(SUBSET.read_text())["schemas"]


def errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Every violation of ``schema`` by ``value`` (empty when it conforms)."""

    ref = schema.get("$ref")
    if ref is not None:
        return errors(value, schemas()[ref.rsplit("/", 1)[1]], path)
    found: list[str] = []
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(_TYPES[k](value) for k in kinds):
            return [f"{path}: {type(value).__name__} is not {kinds}"]
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{path}: {value!r} not in {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        found.append(f"{path}: {value!r} is not {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            found.append(f"{path}: {value} < {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            found.append(f"{path}: {value} > {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                found.append(f"{path}: missing required {key!r}")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                found += errors(item, properties[key], f"{path}.{key}")
            elif extra is False:
                found.append(f"{path}: unexpected {key!r}")
            elif isinstance(extra, dict):
                found += errors(item, extra, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            found.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            found.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                found += errors(item, schema["items"], f"{path}[{i}]")
    for part in schema.get("allOf", ()):
        found += errors(value, part, path)
    if "anyOf" in schema and all(errors(value, part, path) for part in schema["anyOf"]):
        found.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema and sum(not errors(value, part, path) for part in schema["oneOf"]) != 1:
        found.append(f"{path}: does not match exactly one of oneOf")
    return found


def check(value: Any, name: str) -> None:
    """Assert ``value`` conforms to the named spec schema."""

    problems = errors(value, schemas()[name])
    assert not problems, f"{name}: " + "; ".join(problems)


def check_error(body: Any) -> None:
    """An error reply: the spec's ErrorResponse, every Error field present."""

    check(body, "ErrorResponse")


__all__ = ["check", "check_error", "errors", "schemas"]
