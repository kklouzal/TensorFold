"""Independent decoded-value oracles for strict length and finite-uniqueness grammars."""

import itertools
import json

import pytest

from tensorfold.engine import grammar
from tensorfold.server import strict_tools as st
from tensorfold.server.errors import RequestError


@pytest.fixture(scope="module")
def grammars():
    xgr = pytest.importorskip("xgrammar")
    return grammar.Grammars(xgr.TokenizerInfo([""] + [chr(i) for i in range(1, 128)], xgr.VocabType.RAW,
                                            stop_token_ids=[0]))


def _plan(schema):
    return st.Plan([{"type": "function", "function": {"name": "bounded", "strict": True, "parameters": {
        "type": "object", "properties": {"value": schema}, "required": ["value"], "additionalProperties": False}}}],
        required=True, parallel=False)


def _call(encoded):
    return '<tool_call>\n<function=bounded>\n<parameter=value>\n' + encoded + '\n</parameter>\n</function>\n</tool_call>'


def _accepted(grammars, compiled, text):
    matcher = grammars.xgr.GrammarMatcher(compiled)
    return bool(matcher.accept_string(text) and matcher.is_completed())


def test_unique_enum_arrays_accept_every_order_and_refuse_repetitions(grammars):
    values = ["Alt", "Control", "Meta", "Shift"]
    schema = {"type": "array", "items": {"type": "string", "enum": values}, "uniqueItems": True,
              "minItems": 1, "maxItems": 3}
    plan = _plan(schema)
    compiled = grammars.compile(plan.spec)
    for size in range(5):
        for items in itertools.product(values, repeat=size):
            encoded = json.dumps(items)
            assert _accepted(grammars, compiled, _call(encoded)) == (1 <= size <= 3 and len(set(items)) == size), items
    text = _call('["Meta", "Alt", "Shift"]')
    reader = plan.reader()
    for end in range(len(text) + 1):
        reader.feed(text[:end])
    reader.read(text, finished=True)
    assert json.loads(reader.calls[0]["function"]["arguments"]) == {"value": ["Meta", "Alt", "Shift"]}


def test_shared_compiler_refuses_hostile_schema_graphs_and_regex_nesting(grammars):
    schema = {"type": "object", "$defs": {str(i): {"$ref": f"#/$defs/{i + 1}"}
                                          for i in range(grammar.MAX_SCHEMA_NODES)}}
    for make in (lambda: grammar.request_spec({"guided_json": schema}),
                 lambda: grammars.compile(grammar.Spec("json_schema", json.dumps(schema)))):
        with pytest.raises(RequestError, match="limited to 2000 subschemas"):
            make()
    pattern = "(" * 50_000 + "a" + ")" * 50_000
    with pytest.raises(RequestError, match="nests its groups too deeply"):
        grammars.compile(grammar.Spec("regex", pattern))
    with pytest.raises(RequestError, match="nests its groups too deeply"):
        grammar.request_spec({"guided_json": {"type": "string", "pattern": pattern}})


@pytest.mark.parametrize("low,high", [(0, 0), (0, 2), (2, 2)])
def test_unique_bounds_refs_and_constraint_marker_data_are_preserved(grammars, low, high):
    enum = ["__tensorfold_constraint__0", "b"]
    schema = {"type": "array", "uniqueItems": True, "minItems": low, "maxItems": high,
              "$defs": {"choice": {"type": "string", "enum": enum}},
              "items": {"$ref": "#/properties/value/$defs/choice"}}
    plan = _plan(schema)
    compiled = grammars.compile(plan.spec)
    for items in ([], [enum[0]], enum, list(reversed(enum)), ["b", "b"]):
        expected = low <= len(items) <= high and len(set(items)) == len(items)
        assert _accepted(grammars, compiled, _call(json.dumps(items))) == expected
    if low == 0:
        assert not _accepted(grammars, compiled, _call("[" + " " * 33 + "]"))


@pytest.mark.parametrize("value", [[], {}, 1])
def test_malformed_formats_and_uniqueness_flags_are_request_errors(value):
    with pytest.raises(RequestError, match="format must be one of"):
        _plan({"type": "string", "format": value})
    with pytest.raises(RequestError, match="uniqueItems must be a boolean"):
        _plan({"type": "array", "uniqueItems": value, "items": {"type": "string", "enum": ["a"]}})


def test_large_enums_obey_the_published_string_character_budget():
    with pytest.raises(RequestError, match="more than 250 values"):
        _plan({"type": "string", "enum": [str(i) + "a" * 60 for i in range(251)]})


@pytest.mark.parametrize("schema,words", [
    ({"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": ["a", "b"]}}, "uniqueItems"),
    ({"type": "string", "maxLength": 3}, "maxLength"),
    ({"oneOf": [{"type": "integer"}, {"type": "string"}]}, "oneOf"),
    ({"type": "number", "multipleOf": 0.5}, "multipleOf"),
    ({"type": "integer", "enum": [1, 3], "minimum": 2}, "take only type"),
    ({"$ref": "#/$defs/n", "$defs": {"n": {"type": "integer"}}, "minimum": 2}, "$ref takes no constraints"),
])
def test_requested_strict_response_schemas_refuse_native_ignored_constraints(schema, words):
    body = {"response_format": {"type": "json_schema", "json_schema": {"schema": schema, "strict": True}}}
    with pytest.raises(RequestError) as error:
        grammar.request_spec(body)
    assert words in str(error.value)
    for strict in (False, None):
        body["response_format"]["json_schema"]["strict"] = strict
        assert grammar.request_spec(body).kind == "json_schema"
    assert grammar.request_spec({"guided_json": schema}).kind == "json_schema"
