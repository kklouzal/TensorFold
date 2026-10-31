"""Python-scalar XML constraints keep the template's typed history semantics."""
import json

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from tensorfold.server import strict_scalar_tools as scalar
from tensorfold.server import strict_tools as xml
from tensorfold.server.errors import RequestError
from tests.test_strict_tools import QWEN_CALLS, MarkupTokenizer, ScriptEngine, _app, _body, _reply, conforms, tool
from tests.test_strict_tools import grammars as grammars

SCALAR_CALLS = QWEN_CALLS.replace("v if v is string else v|tojson",
    "v|tojson if v is mapping or (v is sequence and v is not string) else v|string")


def render(values, name="f", template=SCALAR_CALLS):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    message = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": values}}]}
    text = env.from_string(template).render(messages=[message], enable_thinking=False)
    return text[text.index(xml.OPEN):text.index(xml.CLOSE) + len(xml.CLOSE)]


def parameters(value):
    return {"type": "object", "properties": {"value": value}, "required": ["value"],
            "additionalProperties": False}


def test_recognition_distinguishes_scalars_and_canonical_nested_containers():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    rendered = env.from_string(SCALAR_CALLS).render(messages=xml.PROBE, enable_thinking=False)
    rendered_false = env.from_string(SCALAR_CALLS).render(messages=scalar.false_probe(), enable_thinking=False)
    assert scalar.template_supported(rendered, rendered_false) and not xml.template_supported(rendered)
    assert not scalar.template_supported(rendered, rendered_false.replace("\nFalse\n", "\nfalse\n"))
    assert not scalar.template_supported(rendered.replace("[true, null,", "[True, None,"), rendered_false)
    standard = env.from_string(QWEN_CALLS).render(messages=xml.PROBE, enable_thinking=False)
    assert xml.template_supported(standard) and not scalar.template_supported(standard, rendered_false)
    assert xml.PROBE[1]["tool_calls"][0]["function"]["arguments"]["tfprobe_bool"] is True


@pytest.mark.parametrize("schema,values", [
    ({"type": "boolean"}, [True, False]), ({"type": "null"}, [None]),
    ({"type": ["boolean", "null"]}, [True, False, None]),
    ({"enum": [True, None, 7]}, [True, None, 7]), ({"const": False}, [False]), ({"const": None}, [None]),
    ({"anyOf": [{"type": "boolean"}, {"type": "integer", "minimum": 2, "maximum": 3}]}, [True, False, 2, 3]),
    ({"anyOf": [{"type": "null"}, {"type": "object", "properties": {"flag": {"type": "boolean"}},
                               "required": ["flag"], "additionalProperties": False}]}, [None, {"flag": True}]),
    ({"anyOf": [{"type": "boolean"}, {"type": "array", "items": {"type": ["boolean", "null"]}}]},
     [False, [True, None, False]])])
def test_scalar_and_json_alternatives_obey_grammar_and_round_trip(grammars, schema, values):
    plan = scalar.Plan([tool("f", parameters(schema))], required=True, parallel=True)
    compiled = grammars.compile(plan.spec)
    for value in values:
        call = render({"value": value})
        assert grammars.accepts(compiled, call)
        content, calls = plan.parse(call)
        assert content == "" and len(calls) == 1
        parsed = json.loads(calls[0]["function"]["arguments"])
        assert parsed == {"value": value} and conforms(parsed, parameters(schema))
        reader = plan.reader()
        deltas = []
        for end in range(1, len(call) + 1):
            deltas.extend(reader.feed(call[:end]))
        arguments = "".join(part["tool_calls"][0]["function"].get("arguments", "") for part in deltas)
        assert json.loads(arguments) == parsed
    assert not grammars.accepts(compiled, render({"value": "unoffered"}))


def test_references_preserve_nested_json_and_boolean_number_identity(grammars):
    schema = {"type": "object", "$defs": {"node": {"anyOf": [{"type": "null"}, {
        "type": "object", "properties": {"flag": {"type": "boolean"}, "next": {"$ref": "#/$defs/node"}},
        "required": ["flag", "next"], "additionalProperties": False}]}},
        "properties": {"value": {"$ref": "#/$defs/node"}}, "required": ["value"]}
    plan = scalar.Plan([tool("f", schema)], required=True, parallel=True)
    compiled = grammars.compile(plan.spec)
    value = {"flag": False, "next": {"flag": True, "next": None}}
    for selected in (None, value):
        call = render({"value": selected})
        assert grammars.accepts(compiled, call)
        assert json.loads(plan.parse(call)[1][0]["function"]["arguments"]) == {"value": selected}
        if selected is not None:
            assert not grammars.accepts(compiled, call.replace('"flag": false', '"flag": False'))
            assert not grammars.accepts(compiled, call.replace('"next": null', '"next": None'))
    boolean = scalar.Plan([tool("f", parameters({"type": "boolean", "enum": [True]}))], required=True, parallel=True)
    compiled = grammars.compile(boolean.spec)
    for bad in (False, 1, 0, None):
        assert not grammars.accepts(compiled, render({"value": bad}))
    wrong_profile = render({"value": True}).replace("\nTrue\n", "\ntrue\n")
    assert not grammars.accepts(compiled, wrong_profile)
    with pytest.raises(xml.CallMarkupError, match="outside its template profile"):
        boolean.parse(wrong_profile)


@pytest.mark.parametrize("value", ["True", "False", "None", "null"])
def test_raw_strings_remain_strings_and_get_distinct_cache_namespace(grammars, value):
    tools = [tool("f", parameters({"type": "string", "enum": [value]}))]
    plan, standard = scalar.Plan(tools, required=True, parallel=True), xml.Plan(tools, required=True, parallel=True)
    assert plan.spec.text != standard.spec.text
    call = render({"value": value})
    assert grammars.accepts(grammars.compile(plan.spec), call)
    assert grammars.accepts(grammars.compile(standard.spec), call)
    assert json.loads(plan.parse(call)[1][0]["function"]["arguments"])["value"] == value


def test_ambiguous_unions_refuse_and_unambiguous_nullable_literals_keep_types(grammars):
    for sub in ({"type": ["string", "boolean"]}, {"type": ["string", "null"]},
                {"enum": ["None", None]}, {"type": ["string", "null"], "pattern": "^[A-Za-z]+$"}):
        with pytest.raises(RequestError, match=r"ambiguous|nullable raw strings"):
            scalar.Plan([tool("f", parameters(sub))], required=True, parallel=True)
    for sub, values in (({"enum": ["null", "ok", None]}, ["null", "ok", None]),
                        ({"type": ["string", "null"], "pattern": "^a+$"}, ["aaa", None])):
        plan = scalar.Plan([tool("f", parameters(sub))], required=True, parallel=True)
        compiled = grammars.compile(plan.spec)
        for value in values:
            call = render({"value": value})
            assert grammars.accepts(compiled, call)
            parsed = json.loads(plan.parse(call)[1][0]["function"]["arguments"])["value"]
            assert parsed == value and type(parsed) is type(value)


def test_finite_unique_container_alternative_stays_canonical_json(grammars):
    sub = {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "string", "enum": ["a", "b"]},
                                        "uniqueItems": True, "minItems": 0, "maxItems": 2}]}
    plan = scalar.Plan([tool("f", parameters(sub))], required=True, parallel=True)
    compiled = grammars.compile(plan.spec)
    for value in (None, [], ["b", "a"]):
        call = render({"value": value})
        assert grammars.accepts(compiled, call)
        assert json.loads(plan.parse(call)[1][0]["function"]["arguments"]) == {"value": value}
    assert not grammars.accepts(compiled, render({"value": ["a", "a"]}))


def test_optional_scalar_and_explicit_non_strict_tool_keep_their_own_contracts(grammars):
    schema = {"type": "object", "properties": {"word": {"type": "string"}, "flag": {"type": "boolean"}},
              "required": ["word"], "additionalProperties": False}
    plan = scalar.Plan([tool("f", schema), tool("loose", parameters({"type": "boolean"}), strict=False)],
                       required=False, parallel=True)
    compiled = grammars.compile(plan.spec)
    for values in ({"word": "None"}, {"word": "True", "flag": False}):
        call = render(values)
        assert grammars.accepts(compiled, call)
        assert json.loads(plan.parse(call)[1][0]["function"]["arguments"]) == values
    loose = render({"value": True, "extra": "None"}, name="loose")
    assert grammars.accepts(compiled, loose)
    assert json.loads(plan.parse(loose)[1][0]["function"]["arguments"]) == {"value": True, "extra": "None"}


@pytest.mark.parametrize("required,parallel", [(False, False), (False, True), (True, False), (True, True)])
def test_required_single_parallel_and_incomplete_profiles(grammars, required, parallel):
    plan = scalar.Plan([tool("f", parameters({"type": "boolean"}))], required=required, parallel=parallel)
    compiled = grammars.compile(plan.spec)
    call = render({"value": True})
    assert grammars.accepts(compiled, call)
    assert grammars.accepts(compiled, "plain text") is (not required)
    assert grammars.accepts(compiled, call + "\n" + call) is parallel
    assert not grammars.accepts(compiled, call.replace("<function=f>", "<function=other>"))
    partial = call[:call.index("True") + 2]
    assert not grammars.accepts(compiled, partial)
    content, calls = plan.parse(partial)
    assert content == partial and calls is None


def test_profile_wrapper_and_reference_expansion_keep_resource_limits(monkeypatch):
    tools = [tool("f", parameters({"type": "boolean"}))]
    plan = scalar.Plan(tools, required=True, parallel=True)
    monkeypatch.setattr(xml, "MAX_GRAMMAR_BYTES", len(plan.spec.text.encode()) - 1)
    with pytest.raises(RequestError, match="exceeds"):
        scalar.Plan(tools, required=True, parallel=True)


def test_reference_node_bound_is_checked_before_root_scalar_splitting(monkeypatch):
    schema = {"type": "object", "properties": {"value": {"$ref": "#/$defs/a"}},
              "$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/c"},
                        "c": {"type": "boolean"}}}
    monkeypatch.setattr(xml, "MAX_SUBSCHEMAS", 3)
    with pytest.raises(RequestError, match="subschemas"):
        scalar.Plan([tool("f", schema)], required=True, parallel=True)


def test_mixed_false_spelling_is_refused_by_actual_cuda_admission(tmp_path, grammars):
    mixed = SCALAR_CALLS.replace("else v|string", "else ('false' if v is sameas false else v|string)")
    engine = ScriptEngine("not called")
    app = _app(tmp_path, engine, grammars, mixed)
    with pytest.raises(RequestError, match="writes calls otherwise"):
        app._prepare(_body(False, tools=[tool("f", parameters({"type": "boolean"}))]), True)
    assert not engine.calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", [True, None, 7])
def test_actual_cuda_app_selects_profile_and_returns_typed_history(tmp_path, grammars, stream, value):
    from tests.test_cuda_admission import http_server, post

    declared = parameters({"enum": [True, None, 7]})
    tools = [tool("f", declared), tool("unused", parameters({"type": "string"}))]
    call = render({"value": value})
    engine = ScriptEngine(call)
    app = _app(tmp_path, engine, grammars, SCALAR_CALLS)
    body = _body(stream, tools=tools, tool_choice={"type": "function", "function": {"name": "f"}},
                 parallel_tool_calls=False)
    prepared = app._prepare(body, True)
    assert isinstance(prepared.calls, scalar.Plan) and list(prepared.calls.tools) == ["f"]
    with http_server(app) as port:
        status, response = post(port, body, True)
    assert status == 200, response
    content, calls, finish = _reply(stream, response)
    assert not content and finish == "tool_calls" and len(calls) == 1 and calls[0][0] == "f"
    assert json.loads(calls[0][1]) == {"value": value}
    history = app.template.render([{"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "f", "arguments": calls[0][1]}}]}],
        tools=tools, enable_thinking=False)
    assert call in history


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("flag", [True, False])
def test_typed_scalar_http_arguments_and_each_stream_prefix_are_valid_json(tmp_path, grammars, stream, flag):
    from tests.test_cuda_admission import http_server, post

    props = {"flag": {"type": "boolean"}, "absent": {"type": "null"},
             "word_true": {"type": "string", "enum": ["True"]}, "word_none": {"type": "string", "enum": ["None"]}}
    schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    values = {"flag": flag, "absent": None, "word_true": "True", "word_none": "None"}
    app = _app(tmp_path, ScriptEngine(render(values)), grammars, SCALAR_CALLS)
    with http_server(app) as port:
        status, text = post(port, _body(stream, tools=[tool("f", schema)], parallel_tool_calls=True), True)
    assert status == 200, text
    _content, calls, finish = _reply(stream, text)
    assert finish == "tool_calls" and len(calls) == 1
    decoded = json.loads(calls[0][1])
    assert decoded == values and type(decoded["flag"]) is bool and decoded["absent"] is None
    if stream:
        compiled = grammars.compiler.compile_json_schema(schema, max_whitespace_cnt=xml.BLANKS)
        matcher = grammars.xgr.GrammarMatcher(compiled)
        fragments = []
        for line in text.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                for call in choice["delta"].get("tool_calls", []):
                    fragment = call.get("function", {}).get("arguments", "")
                    if fragment:
                        fragments.append(fragment)
                        assert matcher.accept_string(fragment), fragment
        assert matcher.is_completed() and len(fragments) > 1 and "".join(fragments) == calls[0][1]


@pytest.mark.parametrize("template", [QWEN_CALLS, SCALAR_CALLS], ids=["json-scalars", "python-scalars"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cut", ["later-budget", "first-stop"])
def test_incomplete_strict_calls_do_not_claim_a_tool_call_finish(tmp_path, grammars, template, stream, cut):
    from tests.test_cuda_admission import http_server, post

    first = render({"value": True}, template=template)
    if cut == "later-budget":
        second = render({"value": False}, template=template)
        word = "False" if template == SCALAR_CALLS else "false"
        partial = second[:second.index(word) + 2]
        script = first + "\n" + partial
        options = {"max_tokens": len(MarkupTokenizer().encode(script).ids)}
        want = "length"
    else:
        script = first
        options = {"stop": "ue", "max_tokens": len(MarkupTokenizer().encode(script).ids) + 1}
        want = "stop"
    app = _app(tmp_path, ScriptEngine(script), grammars, template)
    with http_server(app) as port:
        status, text = post(port, _body(stream, tools=[tool("f", parameters({"type": "boolean"}))],
                                       parallel_tool_calls=True, **options), True)
    assert status == 200, text
    content, calls, finish = _reply(stream, text)
    assert finish == want
    if cut == "later-budget":
        assert json.loads(calls[0][1]) == {"value": True}
        if stream:
            assert len(calls) == 2
            with pytest.raises(json.JSONDecodeError):
                json.loads(calls[1][1])
        else:
            assert len(calls) == 1 and content == partial
    elif stream:
        assert len(calls) == 1
        with pytest.raises(json.JSONDecodeError):
            json.loads(calls[0][1])
    else:
        assert not calls and xml.OPEN in content


@pytest.mark.parametrize("template,planner", [(QWEN_CALLS, xml.Plan), (SCALAR_CALLS, scalar.Plan)])
def test_completion_observer_does_not_confuse_call_limit_with_truncation(template, planner):
    plan = planner([tool("f", parameters({"type": "boolean"}))], required=True, parallel=True)
    call = render({"value": True}, template=template)
    content, calls, complete = plan.parse_finished(call + "\n" + call, max_calls=1)
    assert content == "" and len(calls) == 1 and complete
    partial = call[:call.index("True" if planner is scalar.Plan else "true") + 2]
    content, calls, complete = plan.parse_finished(call + "\n" + partial, max_calls=1)
    assert content == partial and len(calls) == 1 and not complete
