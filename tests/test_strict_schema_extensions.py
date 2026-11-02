"""Owned schema adapters versus decoded JSON semantics, not emitted grammar spelling."""
import itertools
import json

import pytest

from tensorfold.server import strict_gemma_tools as gemma
from tensorfold.server import strict_scalar_tools as scalar
from tensorfold.server import strict_tools as st
from tensorfold.server.errors import RequestError
from tensorfold.server.gemma_values import GEMMA_CLOSE, GEMMA_OPEN, gemma_value
from tensorfold.server.strict_grammar import string_length
from tests.test_strict_constraints import _accepted, _call, _plan, grammars as shared_grammars


@pytest.fixture(scope="module", name="grammars")
def schema_grammars():
    return shared_grammars.__wrapped__()


@pytest.mark.parametrize("low,high", [(0, 0), (0, 3), (1, 1), (2, 3), (2, None)])
def test_json_bounds_count_decoded_unicode_with_all_escape_forms(grammars, low, high):
    schema = {"type": "string", "minLength": low}
    if high is not None:
        schema["maxLength"] = high
    spec = st.response_schema_spec(schema, "response_format")
    compiled = grammars.compile(spec)
    words = ['""', '"abc"', r'"\n"', r'"\t\r"', r'"\""', r'"\\"', r'"\/"',
             r'"\u0061"', r'"\ud83d\ude00"', r'"\uD83D\uDE00a"', r'"\ud800"',
             r'"\udc00"', r'"\ud800\u0041"', r'"\ud800\ud800\udc00"', '"é😀"']
    for size in range(4):
        words.extend('"' + "".join(parts) + '"' for parts in itertools.product(
            [r"\ud800", r"\udc00", r"\n", "a"], repeat=size))
    for encoded in words:
        size = len(json.loads(encoded))
        assert _accepted(grammars, compiled, encoded) == (size >= low and (high is None or size <= high)), encoded
    for malformed in ['"\n"', '"\x01"', r'"\q"', r'"\uZZZZ"']:
        assert not _accepted(grammars, compiled, malformed)


@pytest.mark.parametrize("profile", [st, scalar, gemma])
def test_raw_bounds_preserve_character_count_delimiter_prefixes_and_streaming(grammars, profile):
    sub = {"type": "string", "minLength": 1, "maxLength": 3}
    tool = {"type": "function", "function": {"name": "bounded", "strict": True, "parameters": {
        "type": "object", "properties": {"value": sub}, "required": ["value"], "additionalProperties": False}}}
    plan = profile.Plan([tool], required=True, parallel=False, **({"null": "None"} if profile is gemma else {}))
    compiled = grammars.compile(plan.spec)
    for value in ["", "<", "</", "<|", "abc", "abcd", "é😀", "\n\t", "\\n"]:
        encoded = (GEMMA_OPEN + "call:bounded" + gemma_value({"value": value}, null="None") + GEMMA_CLOSE
                   if profile is gemma else _call(value))
        expected = 1 <= len(value) <= 3
        assert _accepted(grammars, compiled, encoded) == expected, (profile, value)
        if expected:
            reader = plan.reader()
            for end in range(len(encoded) + 1):
                reader.feed(encoded[:end])
            reader.read(encoded, finished=True)
            assert json.loads(reader.calls[0]["function"]["arguments"]) == {"value": value}


def test_discriminator_oneof_is_exclusive_and_refs_keep_their_targets(grammars):
    branches = [{"type": "object", "properties": {"action": {"const": action}, "text": {
        "type": "string", "maxLength": 2}}, "required": ["action", "text"], "additionalProperties": False}
                for action in ["add", "remove"]]
    schema = {"oneOf": branches}
    compiled = grammars.compile(st.response_schema_spec(schema, "response_format"))
    for action, text in itertools.product(["add", "remove", "other"], ["", "ab", "abc", "\n"]):
        assert _accepted(grammars, compiled, json.dumps({"action": action, "text": text})) == (
            action in ("add", "remove") and len(text) <= 2)
    rooted = {"$defs": {"choice": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
              "$ref": "#/$defs/choice/oneOf/0"}
    compiled = grammars.compile(st.response_schema_spec(rooted, "response_format"))
    assert _accepted(grammars, compiled, "1")
    assert not _accepted(grammars, compiled, '"1"')


@pytest.mark.parametrize("branches", [[{"type": "integer"}, {"type": "number"}],
                                      [{"const": 1}, {"const": 1.0}],
                                      [{"type": "object"}, {"type": "object"}]])
def test_unproved_oneof_does_not_become_anyof(branches):
    with pytest.raises(RequestError, match="oneOf branches must"):
        st.response_schema_spec({"oneOf": branches}, "response_format")


def test_oneof_structural_budget_precedes_exclusivity_work(monkeypatch):
    def must_not_compare(*args):
        pytest.fail("oversized union reached the pairwise proof")
    monkeypatch.setattr(st._Schema, "_disjoint", must_not_compare)
    schema = {"oneOf": [{"const": chr(0x10000 + i)} for i in range(100_000)]}
    with pytest.raises(RequestError, match="limited to 2000 subschemas"):
        st._Schema(schema, "schema").check()


def test_large_admitted_finite_union_preserves_every_literal():
    schema = {"oneOf": [{"const": chr(0x1000 + i)} for i in range(st.MAX_SUBSCHEMAS - 1)]}
    checked = st._Schema(schema, "schema")
    checked.check()  # one aggregate finite-value overlap check; no native compilation needed
    assert checked.types(schema) == {"string"}
    schema["oneOf"][-1] = schema["oneOf"][0]
    with pytest.raises(RequestError, match="oneOf branches must"):
        st._Schema(schema, "schema").check()


@pytest.mark.parametrize("marker", [st.VALUE_END, gemma.GEMMA_STRING])
def test_counted_raw_language_excludes_whole_delimiter_but_preserves_every_prefix(grammars, marker):
    from tensorfold.engine.grammar import Spec

    source = string_length(0, 64, exclude=marker)
    source = source.replace("root ::= ", "root ::= body " + json.dumps(marker) + "\nbody ::= ", 1)
    compiled = grammars.compile(Spec("grammar", source))
    values = [marker[:i] for i in range(len(marker))]
    values += ["a" + marker + "b", marker, "é😀\n", "x" * 64, "x" * 65]
    for value in values:
        assert _accepted(grammars, compiled, value + marker) == (len(value) <= 64 and marker not in value)


def test_strict_response_unique_enum_arrays_share_the_owned_exact_adapter(grammars):
    schema = {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": ["a", "b"]}}
    compiled = grammars.compile(st.response_schema_spec(schema, "response_format"))
    for size in range(4):
        for values in itertools.product(["a", "b"], repeat=size):
            assert _accepted(grammars, compiled, json.dumps(values)) == (len(set(values)) == size)
    with pytest.raises(RequestError, match="uniqueItems needs"):
        st.response_schema_spec({"type": "array", "items": {"type": "integer"}, "uniqueItems": True}, "schema")


def test_length_json_parameters_references_and_marker_like_data(grammars):
    schema = {"type": "array", "items": {"$ref": "#/properties/value/$defs/s"}, "$defs": {
        "s": {"type": "string", "minLength": 1, "maxLength": 2}}, "maxItems": 2}
    compiled = grammars.compile(_plan(schema).spec)
    for values in [["\n"], ["😀"], ["é", "ab"], ["abc"], [""]]:
        assert _accepted(grammars, compiled, _call(json.dumps(values))) == all(1 <= len(v) <= 2 for v in values)


@pytest.mark.parametrize("value", [-1, True, 1.5, "2", None, 2**31])
def test_malformed_or_unrepresentable_length_bounds_refuse(value):
    with pytest.raises(RequestError, match="length|Length"):
        _plan({"type": "string", "maxLength": value})


@pytest.mark.parametrize("extra", [{"pattern": ".*"}, {"format": "date"}])
def test_unsupported_length_intersections_are_not_dropped(extra):
    with pytest.raises(RequestError, match="intersection"):
        _plan({"type": "string", "maxLength": 2, **extra})


@pytest.mark.parametrize("profile", [st, scalar, gemma])
@pytest.mark.parametrize("union", [False, True])
def test_counted_helper_aggregate_budget_precedes_retaining_all_inputs(monkeypatch, profile, union):
    from tensorfold.server import strict_grammar

    calls = []
    def bounded_helper(low, high, **kwargs):
        calls.append(low)
        return 'root ::= "' + "x" * (512 + low) + '"'
    monkeypatch.setattr(strict_grammar, "string_length", bounded_helper)
    monkeypatch.setattr(st, "MAX_GRAMMAR_BYTES", 4096)
    subs = [{"type": "string", "minLength": i, "maxLength": 100} for i in range(30)]
    props = {"value": {"anyOf": subs}} if union else {"value" + str(i): sub for i, sub in enumerate(subs)}
    tool = {"type": "function", "function": {"name": "bounded", "strict": True, "parameters": {
        "type": "object", "properties": props}}}
    with pytest.raises(RequestError, match="grammar exceeds"):
        profile.Plan([tool], required=True, parallel=False, **({"null": "None"} if profile is gemma else {}))
    assert 0 < len(calls) < len(subs)
