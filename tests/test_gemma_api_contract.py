"""Native Gemma API boundary contracts; no checkpoint or private artifact dependency."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from tensorfold.server import strict_gemma_tools as gemma
from tensorfold.server import strict_tools as xml
from tensorfold.server.errors import RequestError
from tensorfold.server.gemma_values import (
    GEMMA_CLOSE, GEMMA_OPEN, GEMMA_STRING, gemma_value, hide_gemma_calls, split_gemma_thinking,
)
from tensorfold.server.request_options import fixed_thinking
from tensorfold.server.tools import parse_tool_calls_from_content
from tests.test_strict_tools import (
    STOP, V, MarkupTokenizer, ScriptEngine, _app, _body, _reply, conforms, tool,
)

# A self-contained template of the captured publisher's native argument language.
# In particular, mapping keys are sorted/bare and the None branch is not JSON.
GEMMA_CALLS = r'''{% macro value(v) -%}
{%- if v is string -%}{{ '<|"|>' + v + '<|"|>' }}
{%- elif v is boolean -%}{{ 'true' if v else 'false' }}
{%- elif v is mapping -%}{ {% for k,x in v|dictsort %}{{ k }}:{{ value(x) }}{{ ',' if not loop.last }}{% endfor %} }
{%- elif v is sequence -%}[{% for x in v %}{{ value(x) }}{{ ',' if not loop.last }}{% endfor %}]
{%- else -%}{{ v }}{%- endif -%}
{%- endmacro -%}
{% for m in messages %}{{ m.content if m.role == 'user' else '' }}{% for call in m.tool_calls|default([]) %}<|tool_call>call:{{ call.function.name }}{{ value(call.function.arguments)|replace('{ ', '{')|replace(' }', '}') }}<tool_call|>{% endfor %}{% endfor %}'''


def render(arguments, name="f", template=GEMMA_CALLS):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}]}]
    return env.from_string(template).render(messages=messages).strip()


def schema():
    props = {"flag": {"type": "boolean"}, "absent": {"type": "null"}, "mode": {"enum": ["None", "true"]},
             "nested": {"type": "object", "properties": {"items": {"type": "array", "items": {
                 "anyOf": [{"type": "boolean"}, {"type": "null"}, {"type": "string"}]}}},
                        "required": ["items"], "additionalProperties": False}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def values(flag=True):
    return {"flag": flag, "absent": None, "mode": "None", "nested": {
        "items": [True, False, None, 'quotes " braces {x:None} slash \\ newline\n',
                  "a<tool_call|>b<|tool_call>c", "\x000\x00None",
                  "a<|channel>thought\nliteral<channel|>b"]}}


def plan(parameters=None, **kwargs):
    return gemma.Plan([tool("f", parameters or schema())], required=True, parallel=True, **kwargs)


def test_exact_render_discovery_checks_false_nested_null_and_order():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(GEMMA_CALLS)
    yes, no = [template.render(messages=probe) for probe in (gemma.PROBE, gemma.false_probe())]
    assert gemma.template_supported(yes, no)
    for changed in (no.replace("false", "False"), no.replace("None", "null"), no.replace("call:", ":")):
        assert not gemma.template_supported(yes, changed)


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("step", [1, 3, 31])
def test_native_reader_and_lenient_parser_preserve_typed_nested_values(flag, step):
    expected = values(flag)
    call = render(expected)
    assert call == GEMMA_OPEN + "call:f" + gemma_value(expected) + GEMMA_CLOSE
    reader = plan().reader()
    for end in range(0, len(call) + step, step):
        reader.feed(call[:end])
    reader.read(call, finished=True)
    arguments = reader.calls[0]["function"]["arguments"]
    assert json.loads(arguments) == expected
    fragments = [item.get("function", {}).get("arguments", "")
                 for delta in reader.deltas for item in delta.get("tool_calls", ())]
    assert len(fragments) > 2 and "".join(fragments) == arguments
    assert not reader.text and not reader._unfinished
    content, calls = parse_tool_calls_from_content("before " + call + " after", [tool("f")])
    assert content == "before  after" and json.loads(calls[0]["function"]["arguments"]) == expected
    assert hide_gemma_calls("before " + call + " after", finished=True) == "before  after"
    assert split_gemma_thinking(call, finished=True) == ("", call)


@pytest.mark.parametrize("cut", ["name", "string", "number", "close"])
def test_unfinished_native_call_retains_text_and_streamed_evidence(cut):
    call = render(values())
    stops = {"name": call.index("{"), "string": call.index("quotes") + 3,
             "number": len(call) - len(GEMMA_CLOSE), "close": len(call) - 2}
    partial = call[:stops[cut]]
    content, calls, complete = plan().parse_finished(partial)
    assert content == partial and calls is None and not complete
    reader = plan().reader()
    for end in range(len(partial) + 1):
        reader.feed(partial[:end])
    if cut != "name":
        assert reader.streamed and reader.deltas


@pytest.mark.parametrize("sub", [{"type": "string", "minLength": 1}, {"type": "string", "pattern": ".*"},
                                  {"const": GEMMA_STRING}, {"type": "object", "properties": {"bad,key": {}}}])
def test_unsupported_constraints_and_unrepresentable_literals_refuse(sub):
    with pytest.raises(RequestError):
        plan({"type": "object", "properties": {"value": sub}})


def test_recursive_schemas_emit_bounded_shared_rules_and_resource_limits(monkeypatch):
    declared = {"type": "object", "properties": {"next": {"anyOf": [{"type": "null"}, {"$ref": "#"}]}},
                "required": ["next"], "additionalProperties": False}
    assert len(plan(declared).spec.text) < 5000
    monkeypatch.setattr(xml, "MAX_SUBSCHEMAS", 2)
    with pytest.raises(RequestError, match="subschemas"):
        plan(declared)


def test_fixed_thinking_uses_unconditional_jinja_assignment_only(tmp_path):
    assert fixed_thinking("{% set enable_thinking = true %}{{enable_thinking}}") is True
    assert fixed_thinking("{% set enable_thinking = false %}{{enable_thinking}}") is False
    assert fixed_thinking("{# enable_thinking=true #}") is None
    assert fixed_thinking("{% if enable_thinking is not defined %}{% set enable_thinking = true %}{% endif %}") is None
    assert fixed_thinking("{% set enable_thinking = true %}{% if tools %}{% set enable_thinking = false %}{% endif %}") is None
    from tensorfold.server.thinking_notes import startup
    (tmp_path / "chat_template.jinja").write_text("{% malformed default %}")
    assert startup(tmp_path, False) is None
    selected = "{% set enable_thinking = true %}"
    assert "fixed by" in startup(tmp_path, False, template_source=selected)
    assert startup(tmp_path, False, template_source="{{ messages }}") is None


def test_qwen_missing_closer_and_native_string_literals_keep_outer_ownership():
    tools = [tool("f", {"type": "object", "properties": {"x": {"type": "string"}}}), tool("g")]
    literal = "<|tool_call>call:g{}<tool_call|>"
    qwen = "<tool_call><function=f><parameter=x>" + literal + "</parameter></function>"
    content, calls = parse_tool_calls_from_content(qwen, tools)
    assert not content and len(calls) == 1 and calls[0]["function"]["name"] == "f"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": literal}
    literal = "<tool_call><tool_call>\n<function=g>literal</function>"
    content, calls = parse_tool_calls_from_content(render({"x": literal}), tools)
    assert not content and json.loads(calls[0]["function"]["arguments"]) == {"x": literal}


def test_finished_initial_partial_thought_opener_is_held_without_hiding_invalid_answer():
    assert split_gemma_thinking("<|channel>", finished=True) == ("", "")
    assert split_gemma_thinking(" \n<|channel>tho", finished=True) == ("", " \n")
    invalid = "invalid answer<|channel>"
    assert split_gemma_thinking(invalid, finished=True) == ("", invalid)
    native = '<|tool_call>call:f{x:<|"|>literal<|channel>'
    assert split_gemma_thinking(native, finished=True) == ("", native)


def test_incompatible_explicit_thinking_template_is_refused_before_generation():
    from tensorfold.cuda.server import App
    result = App.__new__(App)
    result.tok = GemmaTokenizer()
    source = SimpleNamespace(source="{{ '<think>' if enable_thinking else '</think>' }}")
    result.template = SimpleNamespace(options=("explicit.jinja", None),
                                      selected=lambda **kwargs: SimpleNamespace(source=source))
    with pytest.raises(RequestError, match="Qwen thinking markers"):
        result._fixed_thinking(False)


class GemmaTokenizer(MarkupTokenizer):
    words = list(MarkupTokenizer.words)
    for index, word in enumerate((GEMMA_OPEN, GEMMA_CLOSE, GEMMA_STRING, "<|channel>", "<channel|>"), 180):
        words[index] = word
    words[185] = "\x00"  # NUL is data, distinct from this toy engine's stop token zero.
    ids = {word: i for i, word in enumerate(words) if len(word) > 1}
    characters = {word: i for i, word in enumerate(words) if len(word) == 1}

    def encode(self, text, **kwargs):
        result, at = [], 0
        specials = sorted(self.ids, key=len, reverse=True)
        while at < len(text):
            special = next((word for word in specials if text.startswith(word, at)), None)
            result.append(self.ids[special] if special else self.characters[text[at]])
            at += len(special) if special else 1
        return SimpleNamespace(ids=result)

    def token_to_id(self, text):
        return self.ids.get(text) if len(text) != 1 else self.characters.get(text)


class GemmaEngine(ScriptEngine):
    likes = {**ScriptEngine.likes, GemmaTokenizer.ids[GEMMA_CLOSE]: 2.0, GemmaTokenizer.ids[GEMMA_STRING]: 1.0}

    def __init__(self, text):
        self.script = GemmaTokenizer().encode(text).ids + [STOP]
        self.calls = []


@pytest.fixture(scope="module")
def grammars():
    xgr = pytest.importorskip("xgrammar")
    pytest.importorskip("torch")
    from tensorfold.engine import grammar
    return grammar.Grammars(xgr.TokenizerInfo(GemmaTokenizer.words, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[STOP]))


def app(tmp_path, text, grammars=None, template=GEMMA_CALLS):
    pytest.importorskip("tensorfold._fd_owner", reason="CUDA checkpoint boundary requires the installed native owner")
    result = _app(tmp_path, GemmaEngine(text), grammars, template)
    result.tok = GemmaTokenizer()
    return result


def test_cuda_fixed_template_default_and_off_refusal_before_generation(tmp_path):
    result = app(tmp_path, "answer", template="{% set enable_thinking = true %}" + GEMMA_CALLS)
    prepared = result._prepare({"messages": [{"role": "user", "content": "Hi"}]}, True)
    assert prepared.thinking is True and result.thinking_markers == ("<|channel>thought", "<channel|>")
    for options in ({"reasoning_effort": "none"}, {"chat_template_kwargs": {"enable_thinking": False}}):
        with pytest.raises(RequestError, match="fixes enable_thinking=true"):
            result._prepare({"messages": [{"role": "user", "content": "Hi"}], **options}, True)
    assert not result.engine.calls


def test_native_schema_grammar_holds_types_enums_nested_values_and_required_keys(grammars):
    expected = values()
    compiled = grammars.compile(plan().spec)
    call = render(expected)
    assert grammars.accepts(compiled, call)
    assert conforms(json.loads(plan().parse(call)[1][0]["function"]["arguments"]), schema())
    for bad in (call.replace("flag:true", "flag:True"), call.replace("absent:None", "absent:null"),
                call.replace("mode:<|\"|>None", "mode:<|\"|>other"), call.replace("flag:true,", ""),
                call.replace("call:f", "call:other"), call.replace("absent:None", "absent:true")):
        assert not grammars.accepts(compiled, bad), bad


def test_strict_whole_answer_prefix_cannot_be_bypassed_with_stray_or_literal_channel_marker(grammars):
    from tensorfold.cuda.server import App
    declared = {"type": "object", "properties": {"x": {"type": "string"}},
                "required": ["x"], "additionalProperties": False}
    native = plan(declared)
    prepared = SimpleNamespace(grammar=(native.spec, grammars.compile(native.spec)), thinking=True)
    result = App.__new__(App)
    result.tok, result.grammars, result.engine = GemmaTokenizer(), grammars, SimpleNamespace(eos=(STOP,))
    good = render({"x": "valid"})
    invalid = "<|tool_call>call:f{x:None}<tool_call|><channel|>" + good
    with pytest.raises(xml.CallMarkupError, match="strict tool grammar"):
        result._thought(prepared, invalid, True)
    with pytest.raises(xml.CallMarkupError, match="strict tool grammar"):
        result._thought(prepared, invalid, False)
    literal = render({"x": "literal<|channel>thought\nvalue<channel|>end"})
    assert result._thought(prepared, literal, True) == ("", literal)
    reasoning = "<|channel>thought\nreason<channel|>" + literal
    assert result._thought(prepared, reasoning, True) == ("reason", literal)
    assert result._thought(prepared, "<|channel>", True) == ("", "")


def test_native_recursive_refs_optional_keys_additional_schemas_numeric_and_formats(grammars):
    declared = {"type": "object", "properties": {
        "id": {"type": "string", "format": "uuid"}, "limit": {"type": "integer", "minimum": 1, "maximum": 3},
        "node": {"$ref": "#/$defs/node"}}, "required": ["id", "limit"], "additionalProperties": {"type": "boolean"},
        "$defs": {"node": {"type": "object", "properties": {"next": {"anyOf": [
            {"type": "null"}, {"$ref": "#/$defs/node"}]}}, "required": ["next"], "additionalProperties": False}}}
    compiled = grammars.compile(plan(declared).spec)
    args = {"id": "123e4567-e89b-12d3-a456-426614174000", "limit": 2, "node": {"next": {"next": None}}}
    call = render(args)
    assert grammars.accepts(compiled, call)
    for bad in (call.replace("limit:2", "limit:4"), call.replace("123e4567", "123x4567"),
                call.replace("limit:2", "limit:2.5"), call.replace("node:", "extra:")):
        assert not grammars.accepts(compiled, bad)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("flag", [True, False])
def test_cuda_http_native_strict_calls_and_sse_prefixes(tmp_path, grammars, stream, flag):
    from tests.test_cuda_admission import http_server, post
    expected = values(flag)
    result = app(tmp_path, render(expected), grammars)
    with http_server(result) as port:
        status, response = post(port, _body(stream, tools=[tool("f", schema())], tool_choice="required"), True)
    assert status == 200, response
    content, calls, finish = _reply(stream, response)
    assert not content and finish == "tool_calls" and json.loads(calls[0][1]) == expected
    if stream:
        ordered = json.loads(json.dumps(schema(), sort_keys=True))
        matcher = grammars.xgr.GrammarMatcher(grammars.compiler.compile_json_schema(ordered))
        for line in response.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                for call in choice["delta"].get("tool_calls", []):
                    fragment = call.get("function", {}).get("arguments", "")
                    if fragment:
                        assert matcher.accept_string(fragment)
        assert matcher.is_completed()


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_gemma_reasoning_ends_at_channel_marker_before_strict_call(tmp_path, grammars, stream):
    from tests.test_cuda_admission import http_server, post
    text = "<|channel>thought\nconsider carefully\n<channel|>" + render(values())
    result = app(tmp_path, text, grammars, "{% set enable_thinking = true %}" + GEMMA_CALLS)
    with http_server(result) as port:
        status, response = post(port, _body(stream, tools=[tool("f", schema())], tool_choice="required"), True)
    assert status == 200, response
    content, calls, finish = _reply(stream, response)
    assert not content and finish == "tool_calls" and json.loads(calls[0][1]) == values()
    if not stream:
        message = json.loads(response)["choices"][0]["message"]
        assert message["reasoning_content"] == "consider carefully\n"


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_http_truncated_native_call_keeps_stop_or_length(tmp_path, grammars, stream):
    from tests.test_cuda_admission import http_server, post
    text = render(values())
    result = app(tmp_path, text, grammars)
    cut = len(result.tok.encode(text[:text.index("quotes") + 3]).ids)
    with http_server(result) as port:
        status, response = post(port, _body(stream, tools=[tool("f", schema())], max_tokens=cut), True)
    assert status == 200, response
    _content, calls, finish = _reply(stream, response)
    assert finish == "length"
    if stream:
        assert calls and calls[0][0] == "f"
    else:
        assert not calls


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_http_unsupported_schema_refuses_before_engine_or_sse(tmp_path, grammars, stream):
    from tests.test_cuda_admission import http_server, post
    result = app(tmp_path, "should not run", grammars)
    bad = {"type": "object", "properties": {"x": {"type": "string", "minLength": 1}}}
    with http_server(result) as port:
        status, response = post(port, _body(stream, tools=[tool("f", bad)]), True)
    assert status == 400 and "minLength" in response and not result.engine.calls
