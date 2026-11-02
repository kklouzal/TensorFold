"""XML type hints honor JSON Schema type unions without guessing string intent."""
import json

import pytest

from tensorfold.tool_parameters import decode_parameter, typed_parameter
from tensorfold.server.tools import parse_tool_calls_from_content
from tensorfold.engine.tool_draft import ToolCallStreamer


@pytest.mark.parametrize('schema', [
    {'type': ['integer', 'null']},
    {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
    {'oneOf': [{'type': 'integer'}, {'type': 'null'}]},
    {'anyOf': [{'oneOf': [{'type': 'integer'}, {'type': 'null'}]}]},
])
@pytest.mark.parametrize('text,expected', [('60', 60), ('null', None), ('wrong', 'wrong'), ('true', 'true')])
def test_nullable_integer_types_are_derived_without_boolean_coercion(schema, text, expected):
    assert typed_parameter(schema)
    assert decode_parameter(text, schema, python=False) == expected


@pytest.mark.parametrize('schema', [
    {'type': ['string', 'integer']},
    {'anyOf': [{'type': 'string'}, {'type': 'integer'}]},
    {'oneOf': [{'type': 'string'}, {'type': 'null'}]},
    {'anyOf': [{'type': 'integer'}, {}]},
])
def test_ambiguous_or_unknown_string_intent_preserves_exact_wire_text(schema):
    assert not typed_parameter(schema)
    for text in ('60', 'null', ' true ', '{"x":1}'):
        assert decode_parameter(text, schema) == text


def test_enclosing_type_narrows_composition_and_python_literals_remain_data():
    schema = {'type': 'integer', 'anyOf': [{'type': 'string'}, {'type': 'integer'}]}
    assert decode_parameter('60', schema) == 60
    nullable = {'oneOf': [{'type': 'array'}, {'type': 'null'}]}
    assert decode_parameter("('a', 1)", nullable) == ['a', 1]
    assert decode_parameter("__import__('os').getcwd()", nullable) == "__import__('os').getcwd()"


def test_deep_compositions_and_programmatic_cycles_are_bounded_by_schema_graph():
    schema = {'type': 'integer'}
    for _ in range(1500):
        schema = {'anyOf': [schema]}
    assert decode_parameter('7', schema) == 7
    cyclic = {}
    cyclic['anyOf'] = [cyclic]
    assert not typed_parameter(cyclic)
    assert decode_parameter('7', cyclic) == '7'


@pytest.mark.parametrize('composition', ['type', 'anyOf', 'oneOf'])
@pytest.mark.parametrize('step', [1, 7, 10000])
def test_complete_and_streamed_xml_share_nullable_type_resolution(composition, step):
    schema = ({'type': ['integer', 'null']} if composition == 'type' else
              {composition: [{'type': 'integer'}, {'type': 'null'}]})
    tools = [{'type': 'function', 'function': {'name': 'wait', 'parameters': {
        'type': 'object', 'properties': {'timeout_seconds': schema}}}}]
    text = '<tool_call><function=wait><parameter=timeout_seconds>60</parameter></function></tool_call>'
    _, calls = parse_tool_calls_from_content(text, tools)
    assert json.loads(calls[0]['function']['arguments']) == {'timeout_seconds': 60}
    streamer = ToolCallStreamer(tools)
    deltas = []
    for end in range(1, len(text) + 1, step):
        deltas += streamer.feed(text[:end])
    deltas += streamer.feed(text, finished=True)
    arguments = ''.join(d['tool_calls'][0]['function'].get('arguments', '') for d in deltas)
    assert json.loads(arguments) == {'timeout_seconds': 60}


@pytest.mark.parametrize('schema', [
    {'type': 'number', 'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
    {'type': 'integer', 'anyOf': [{'type': 'number'}, {'type': 'null'}]},
    {'type': 'number', 'oneOf': [{'type': 'integer'}, {'type': 'null'}]},
    {'type': 'integer', 'oneOf': [{'type': 'number'}, {'type': 'null'}]},
])
def test_enclosing_numeric_types_intersect_integer_subtype_in_both_directions(schema):
    assert typed_parameter(schema)
    assert type(decode_parameter('3', schema, python=False)) is int
    assert decode_parameter('3', schema, python=False) == 3
    assert decode_parameter('3.5', schema, python=False) == '3.5'
    assert decode_parameter('null', schema, python=False) == 'null'
    assert decode_parameter('true', schema, python=False) == 'true'


@pytest.mark.parametrize('schema', [
    {'type': ['string', 'null']},
    {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
    {'oneOf': [{'type': 'string'}, {'type': 'null'}]},
])
def test_nullable_raw_string_does_not_guess_that_null_text_is_json_null(schema):
    assert not typed_parameter(schema)
    for text in ('null', 'hello', '"null"'):
        assert decode_parameter(text, schema) == text
