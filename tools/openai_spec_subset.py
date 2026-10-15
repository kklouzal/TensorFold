"""Extract the OpenAI OpenAPI schemas this server's replies are checked against, for tests/fixtures/openai.

usage: python tools/openai_spec_subset.py OPENAPI_JSON COMMIT > tests/fixtures/openai/spec_subset.json

The output keeps each named schema's transitive ``$ref`` closure with only the keywords that constrain a value (types,
properties, required keys, enums, constants, items, combinators and ranges); prose, examples and vendor keys are
dropped. ``Response`` is too large to copy whole: its required top-level keys are kept, under ``Response.required``.
The provenance block names the spec commit and the source file's SHA-256, so a regenerated subset is comparable.
"""

from __future__ import annotations

import hashlib
import json
import sys

ROOTS = ("ErrorResponse", "Model", "ListModelsResponse", "CreateChatCompletionResponse",
         "CreateChatCompletionStreamResponse", "CreateCompletionResponse", "TokenCountsResource")
KEEP = {"type", "properties", "required", "enum", "const", "items", "anyOf", "oneOf", "allOf", "$ref",
        "additionalProperties", "minimum", "maximum", "minItems", "maxItems", "nullable"}


def _strip(value):
    if isinstance(value, list):
        return [_strip(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key == "properties":
            out[key] = {name: _strip(schema) for name, schema in item.items()}
        elif key in KEEP:
            out[key] = _strip(item)
    return out


def _required(schemas: dict, name: str) -> list[str]:
    schema = schemas[name]
    keys = list(schema.get("required", []))
    for part in schema.get("allOf", []):
        ref = part.get("$ref")
        keys += _required(schemas, ref.rsplit("/", 1)[1]) if ref else part.get("required", [])
    return keys


def subset(spec: dict) -> dict:
    schemas = spec["components"]["schemas"]
    found: dict[str, dict] = {}

    def walk(value):
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                name = ref.rsplit("/", 1)[1]
                if name not in found:
                    found[name] = schemas[name]
                    walk(schemas[name])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for name in ROOTS:
        found[name] = schemas[name]
        walk(schemas[name])
    out = {name: _strip(found[name]) for name in sorted(found)}
    deleted = spec["paths"]["/responses/{response_id}"]["delete"]["responses"]["200"]["content"]["application/json"]
    out["DeleteResponseResponse"] = _strip(deleted["schema"])
    out["Response"] = {"required": sorted(set(_required(schemas, "Response")))}
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    raw = open(argv[1], "rb").read()
    spec = json.loads(raw)
    document = {"provenance": {"source": "https://github.com/openai/openai-openapi (openapi.json)",
                               "commit": argv[2], "version": spec["info"]["version"],
                               "sha256": hashlib.sha256(raw).hexdigest(), "generator": "tools/openai_spec_subset.py"},
                "schemas": subset(spec)}
    json.dump(document, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
