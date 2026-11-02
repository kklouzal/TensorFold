"""Derived-schema byte admission against independent stdlib JSON encodings."""
import copy
import json

import pytest

from tensorfold.server import strict_tools as st
from tensorfold.server.errors import RequestError


@pytest.mark.parametrize("value", [None, True, 1, 1.25, "", "é😀\n\t\x00\"\\", [], {},
    {"é\"\\": [None, False, 17, -2.5, {"data": "quote\" slash\\ newline\n"}]},
    {1: "integer key", None: ("tuple", True), 2.5: "float key"}])
def test_byte_count_and_admitted_text_match_stdlib_exactly(monkeypatch, value):
    expected = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    size = len(expected.encode("utf-8"))
    monkeypatch.setattr(st, "MAX_GRAMMAR_BYTES", size)
    assert st._schema_json_size(value) == size
    assert st._schema_json_text(value) == expected
    monkeypatch.setattr(st, "MAX_GRAMMAR_BYTES", size - 1)
    with pytest.raises(RequestError, match="grammar exceeds"):
        st._schema_json_text(value)


def test_shared_annotation_payload_charges_every_serialized_occurrence(monkeypatch):
    payload = {"description": "é\\\"\n" * 100}
    value = {"a": payload, "b": payload, "c": payload}
    expected = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
    assert st._schema_json_size(value) == expected
    monkeypatch.setattr(st, "MAX_GRAMMAR_BYTES", expected - 1)
    with pytest.raises(RequestError, match="grammar exceeds"):
        st._schema_json_text(value)


def test_native_stdlib_positive_deep_annotation_keeps_its_encoding_domain():
    nested = "annotation"
    for _ in range(1100):
        nested = [nested]
    schema = {"type": "string", "description": nested}
    try:
        expected = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    except RecursionError:
        pytest.skip("this stdlib cannot encode the positive-depth fixture (CPython 3.12 separates C/Python limits)")
    assert st._schema_json_size(schema) == len(expected.encode())
    assert st.response_schema_spec(schema, "response_format").text == expected


def test_cycle_refusal_matches_stdlib_and_terminates_without_recursion():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Circular reference detected"):
        json.dumps(cyclic)
    with pytest.raises(ValueError, match="Circular reference detected"):
        st._schema_json_size(cyclic)


def test_oversized_string_preflight_never_materializes_whole_json_or_string(monkeypatch):
    value = {"description": "😀\"\\\n" * 10_000}
    real, chunks = json.dumps, []
    def bounded_fragments(part, *args, **kwargs):
        assert isinstance(part, str), "whole document reached serializer before admission"
        assert len(part) <= 4096, "unbounded string chunk reached serializer"
        chunks.append(len(part))
        return real(part, *args, **kwargs)
    monkeypatch.setattr(st, "MAX_GRAMMAR_BYTES", 1024)
    monkeypatch.setattr(st.json, "dumps", bounded_fragments)
    with pytest.raises(RequestError, match="grammar exceeds"):
        st._schema_json_text(value)
    assert chunks


def alias_schema(annotation):
    branch = {"type": "string", "description": annotation}
    for _ in range(4):
        branch = {"oneOf": [branch]}
    return {"type": "object", "$defs": {"choice": branch}, "properties": {
        f"p{i}": {"$ref": "#/$defs/choice" + "/oneOf/0" * i} for i in range(1, 5)}}


def test_alias_expansion_refuses_before_any_subtree_copy(monkeypatch):
    annotation = 'é"\\\n' * 100
    schema = alias_schema(annotation)
    original = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    # Independent lower bound: original leaf plus four alias ancestors each
    # serialize the exact annotation again. No expanded large fixture is built.
    repeated_payload_floor = 5 * len(json.dumps(annotation, ensure_ascii=False).encode())
    limit = len(original.encode()) + 1
    assert limit < repeated_payload_floor
    monkeypatch.setattr(st, "MAX_GRAMMAR_BYTES", limit)
    def must_not_copy(*args, **kwargs):
        pytest.fail("alias subtree copied before byte admission")
    monkeypatch.setattr(copy, "deepcopy", must_not_copy)
    with pytest.raises(RequestError, match="grammar exceeds"):
        st.response_schema_spec(schema, "response_format")


def test_admitted_aliases_preserve_annotations_and_effective_reference_targets():
    annotation = {"unicode": "é😀", "escapes": '"\\\n', "nested": [1, True, None]}
    schema = alias_schema(annotation)
    before = copy.deepcopy(schema)
    spec = st.response_schema_spec(schema, "response_format")
    assert schema == before
    emitted = json.loads(spec.text)
    for prop in emitted["properties"].values():
        target = emitted["$defs"][prop["$ref"].removeprefix("#/$defs/")]
        while "anyOf" in target:
            target = target["anyOf"][0]
        assert target == {"type": "string", "description": annotation}


@pytest.mark.parametrize("native", [False, True])
def test_deep_array_pointer_alias_preserves_stdlib_positive_data_without_copy(native):
    annotation = "deep alias annotation"
    for _ in range(600):
        annotation = [annotation]
    schema = alias_schema(annotation)
    if native:
        pytest.importorskip("xgrammar")
        schema["properties"]["limited"] = {"type": "string", "maxLength": 2}
    expected = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    # Both paths exercise the public response adapter; the replacement path
    # additionally crosses the real native string consumer.
    spec = st.response_schema_spec(schema, "response_format")
    assert json.dumps(schema, ensure_ascii=False, separators=(",", ":")) == expected
    assert spec.kind == ("grammar" if native else "json_schema")
    if not native:
        emitted = json.loads(spec.text)
        annotation_text = json.dumps(annotation, ensure_ascii=False, separators=(",", ":"))
        for prop in emitted["properties"].values():
            target = emitted["$defs"][prop["$ref"].removeprefix("#/$defs/")]
            while "anyOf" in target:
                target = target["anyOf"][0]
            assert json.dumps(target["description"], ensure_ascii=False, separators=(",", ":")) == annotation_text


def test_native_converter_receives_admitted_exact_text_not_an_unbounded_dict(monkeypatch):
    xgr = pytest.importorskip("xgrammar")
    real, received = xgr.Grammar.from_json_schema, []
    annotation = {"unicode": "é😀", "escaped": '"\\\n', "samples": [True, None]}
    def capture(source, *args, **kwargs):
        assert isinstance(source, str)
        assert len(source.encode()) <= st.MAX_GRAMMAR_BYTES
        assert json.loads(source)["description"] == annotation
        received.append(source)
        return real(source, *args, **kwargs)
    monkeypatch.setattr(xgr.Grammar, "from_json_schema", staticmethod(capture))
    st.response_schema_spec({"type": "object", "description": annotation,
                             "properties": {"value": {"type": "string", "maxLength": 2}}}, "response_format")
    assert len(received) == 1
