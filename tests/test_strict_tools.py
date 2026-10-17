"""Strict function tools (``server.strict_tools``): the supported schema subset, the call grammar, its reader, replies
fuzzed under the grammar and checked by an independent validator, and the CUDA server's strict requests (no GPU)."""

from __future__ import annotations

import datetime
import ipaddress
import json
import re
import threading
import uuid

import pytest

from tensorfold.server import strict_tools as st
from tensorfold.server.errors import RequestError

# -- an independent JSON-schema oracle for the supported subset (no xgrammar, no strict_tools code) -----------------


def _same(a, b) -> bool:
    """JSON equality (Python's 1 == True == 1.0 is not JSON's)."""

    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _format_ok(value: str, fmt: str) -> bool:
    try:
        if fmt == "uuid":
            return str(uuid.UUID(value)) == value.lower() and len(value) == 36
        if fmt == "date":
            return re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None and bool(
                datetime.date(int(value[:4]), int(value[5:7]), min(int(value[8:]), 28)))
        if fmt in ("date-time", "time"):
            return re.fullmatch(r"(\d{4}-\d\d-\d\dT)?\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:\d\d)", value) is not None
        if fmt == "ipv4":
            return isinstance(ipaddress.ip_address(value), ipaddress.IPv4Address)
        if fmt == "ipv6":
            return isinstance(ipaddress.ip_address(value), ipaddress.IPv6Address)
        if fmt == "email":
            local, _, domain = value.partition("@")
            return bool(local) and bool(domain) and "@" not in domain
        if fmt == "hostname":
            return all(re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", part) for part in value.split("."))
        if fmt == "duration":
            return re.fullmatch(r"P(\d+W|(\d+Y)?(\d+M)?(\d+D)?(T(\d+H)?(\d+M)?(\d+S)?)?)", value) is not None
    except ValueError:
        return False
    return False


def conforms(value, schema, root=None) -> bool:
    """Whether ``value`` satisfies ``schema`` (the JSON-schema keywords strict tools accept)."""

    root = schema if root is None else root
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/") if schema["$ref"] not in ("#",) else []:
            node = node[part]
        if not conforms(value, node, root):
            return False
    types = schema.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        kinds = {"null": value is None, "boolean": isinstance(value, bool), "string": isinstance(value, str),
                 "object": isinstance(value, dict), "array": isinstance(value, list),
                 "number": isinstance(value, (int, float)) and not isinstance(value, bool),
                 "integer": (isinstance(value, int) and not isinstance(value, bool))
                 or (isinstance(value, float) and value.is_integer())}
        if not any(kinds[n] for n in names):
            return False
    if "enum" in schema and not any(_same(value, v) for v in schema["enum"]):
        return False
    if "const" in schema and not _same(value, schema["const"]):
        return False
    if "anyOf" in schema and not any(conforms(value, b, root) for b in schema["anyOf"]):
        return False
    if isinstance(value, str):
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            return False
        if "format" in schema and not _format_ok(value, schema["format"]):
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            return False
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return False
        if "multipleOf" in schema and value % schema["multipleOf"]:
            return False
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            return False
        if "items" in schema and not all(conforms(v, schema["items"], root) for v in value):
            return False
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", ())):
            return False
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                if not conforms(v, props[k], root):
                    return False
            elif extra is False or (isinstance(extra, dict) and not conforms(v, extra, root)):
                return False
    return True


def test_the_oracle_rejects_what_it_should():
    assert conforms({"a": 1}, {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})
    assert not conforms({"a": True}, {"type": "object", "properties": {"a": {"type": "integer"}}})
    assert not conforms({"b": 1}, {"type": "object", "properties": {"a": {}}, "additionalProperties": False})
    assert not conforms("x", {"enum": ["y"]}) and not conforms(True, {"enum": [1]}) and conforms(None, {"enum": [None]})
    assert not conforms("2024-13-01", {"type": "string", "format": "date"})
    assert conforms("123e4567-e89b-12d3-a456-426614174000", {"type": "string", "format": "uuid"})


# -- tools, schemas and the probe ------------------------------------------------------------------------------------
WEATHER = {"type": "object", "additionalProperties": False, "required": ["city", "days", "unit", "opts", "note", "id"],
           "properties": {
               "city": {"type": "string", "description": "any text"},
               "days": {"type": "integer", "minimum": 1, "maximum": 7},
               "unit": {"type": "string", "enum": ["c", "f"]},
               "opts": {"type": "object", "additionalProperties": False, "required": ["a", "b"],
                        "properties": {"a": {"type": "string"},
                                       "b": {"type": "array", "maxItems": 2, "items": {"$ref": "#/$defs/level"}}}},
               "note": {"type": ["string", "null"]},
               "id": {"type": "string", "format": "uuid"},
               "flag": {"type": "boolean", "default": False},
           },
           "$defs": {"level": {"type": "integer", "enum": [1, 2, 3]}}}
LIST = {"type": "object", "additionalProperties": False, "required": ["head"],
        "properties": {"head": {"$ref": "#/$defs/node"}},
        "$defs": {"node": {"type": "object", "additionalProperties": False, "required": ["v", "next"],
                           "properties": {"v": {"type": "integer"},
                                          "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}}}}}


def tool(name, parameters=None, strict=True):
    function = {"name": name, **({} if parameters is None else {"parameters": parameters})}
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


TOOLS = [tool("get_weather", WEATHER), tool("chain", LIST), tool("now"), tool("loose", {
    "type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}}}, strict=False)]
SCHEMAS = {"get_weather": WEATHER, "chain": LIST, "now": {"type": "object", "properties": {}}}

QWEN_CALLS = ("{% for m in messages %}{{ m.role }}:{{ m.content or '' }}"
              "{% for tc in (m.tool_calls or []) %}{% if not loop.first %}{{ '\\n' }}{% endif %}"
              "{{ '<tool_call>\\n<function=' + tc.function.name + '>\\n' }}"
              "{% for k, v in tc.function.arguments|items %}{{ '<parameter=' + k + '>\\n' }}"
              "{{ v if v is string else v|tojson }}{{ '\\n</parameter>\\n' }}{% endfor %}"
              "{{ '</function>\\n</tool_call>' }}{% endfor %};{% endfor %}"
              "assistant:{% if enable_thinking %}<think>\\n{% endif %}")


def test_the_probe_recognizes_qwen_xml_calls_only():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    assert st.template_supported(env.from_string(QWEN_CALLS).render(messages=st.PROBE, enable_thinking=False))
    hermes = ("{% for m in messages %}{% for tc in (m.tool_calls or []) %}<tool_call>\n"
              "{{ {'name': tc.function.name, 'arguments': tc.function.arguments}|tojson }}\n</tool_call>"
              "{% endfor %}{% endfor %}")
    assert not st.template_supported(env.from_string(hermes).render(messages=st.PROBE))
    # Integer 7 alone cannot distinguish JSON from Python's True/None string formatting.
    python_values = QWEN_CALLS.replace("v|tojson", "v|string")
    assert not st.template_supported(env.from_string(python_values).render(messages=st.PROBE))
    assert not st.template_supported(env.from_string(QWEN_CALLS.replace("'\\n</parameter>", "'</parameter>")).render(
        messages=st.PROBE, enable_thinking=False))


def test_a_plan_reads_each_parameters_mode():
    plan = st.Plan(TOOLS, required=False, parallel=True)
    weather = plan.tools["get_weather"]
    assert [(p.name, p.required, p.mode) for p in weather.params] == [
        ("city", True, "raw"), ("days", True, "json"), ("unit", True, "raw"), ("opts", True, "json"),
        ("note", True, "nullable"), ("id", True, "raw"), ("flag", False, "json")]
    assert [p.mode for p in plan.tools["chain"].params] == ["json"] and plan.tools["now"].params == ()
    assert not plan.tools["loose"].strict
    assert plan.spec.kind == "tools" and plan.spec.field == "tools"
    tag = json.loads(plan.spec.text)
    assert tag["format"]["type"] == "triggered_tags" and tag["format"]["triggers"] == ["<tool_call>"]
    assert not tag["format"]["stop_after_first"]
    assert json.loads(st.Plan(TOOLS, required=False, parallel=False).spec.text)["format"]["stop_after_first"]
    forced = json.loads(st.Plan(TOOLS[:1], required=True, parallel=True).spec.text)["format"]
    assert forced["type"] == "sequence" and forced["elements"][1]["type"] == "tags_with_separator"
    # identical requests give identical grammar text (the compiled-grammar cache key)
    assert st.Plan(TOOLS, required=False, parallel=True).spec == plan.spec


@pytest.mark.parametrize("schema, words", [
    ({"type": "object", "properties": {"s": {"type": "string", "maxLength": 3}}}, "do not support maxLength"),
    ({"type": "object", "properties": {"s": {"type": "string", "minLength": 1}}}, "put length limits"),
    ({"type": "object", "properties": {"s": {"type": "array", "uniqueItems": True}}}, "uniqueItems needs"),
    ({"type": "object", "properties": {"s": {"oneOf": [{"type": "integer"}]}}}, "do not support oneOf"),
    ({"type": "object", "properties": {"s": {"allOf": [{"type": "integer"}]}}}, "do not support allOf"),
    ({"type": "object", "properties": {"s": {"type": "integer", "not": {"const": 1}}}}, "do not support not"),
    ({"type": "object", "properties": {"s": {"type": "number", "multipleOf": 0.5}}}, "type integer without"),
    ({"type": "object", "properties": {"s": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}, "ambiguous"),
    ({"type": "object", "properties": {"s": {}}}, "ambiguous"),
    ({"type": "object", "properties": {"s": {"type": "string", "pattern": "^.*$"}}}, "pattern can spell"),
    ({"type": "object", "properties": {"s": {"type": "string", "pattern": "^[a-z<>/]+$"}}}, "pattern can spell"),
    ({"type": "object", "properties": {"s": {"type": "string", "format": "uri"}}}, "format must be one of"),
    ({"type": "object", "properties": {"s": {"type": "string", "enum": ["a</parameter>"]}}}, "cannot contain"),
    ({"type": "object", "properties": {"s": {"type": "string", "enum": ["a"], "pattern": "a"}}}, "take only type"),
    ({"type": "object", "properties": {"s": {"$ref": "#/$defs/missing"}}}, "does not resolve"),
    ({"type": "object", "properties": {"s": {"$ref": "other.json#/x"}}}, "same schema"),
    ({"type": "object", "properties": {"s": {"type": "integer"}}, "required": ["t"]}, "does not declare"),
    ({"type": "array"}, 'type "object"'),
    ({"anyOf": [{"type": "object"}]}, "must be an object schema"),
    ({"type": "object", "properties": {"bad name": {"type": "integer"}}}, "cannot be written"),
    ({"type": "object", "properties": {"s": {"type": "integer", "minimum": "1"}}}, "must be a number"),
    # shapes xgrammar 0.2.8 would not enforce whole (it lets enum, const, $ref and anyOf override their neighbours)
    ({"type": "object", "properties": {"s": {"type": "integer", "anyOf": [{"minimum": 5}, {"maximum": 1}]}}},
     "need a type beside them"),
    ({"type": "object", "properties": {"s": {"type": "integer", "anyOf": [{"enum": [1, "a"]}]}}},
     "admit only the types"),
    ({"type": "object", "properties": {"s": {"type": "integer", "anyOf": [{"$ref": "#/$defs/t"}]}},
      "$defs": {"t": {"type": "string"}}}, "admit only the types"),
    ({"type": "object", "properties": {"s": {"anyOf": [{"type": "integer"}, {"minimum": 1}]}}},
     "need a type beside them"),
    ({"type": "object", "properties": {"s": {"type": "integer", "anyOf": [{"type": "string"}]}}},
     "admit only the types"),
    ({"type": "object", "properties": {"s": {"type": "object", "properties": {"a": {"type": "integer"}},
                                             "anyOf": [{"type": "object"}]}}}, "anyOf takes only type"),
    ({"type": "object", "properties": {"s": {"type": "string", "enum": ["a", 1]}}}, "every enum and const value"),
    ({"type": "object", "properties": {"s": {"type": "integer", "enum": [3, 9], "maximum": 5}}},
     "enum and const take only type"),
    ({"type": "object", "properties": {"s": {"$ref": "#/$defs/n", "minimum": 3}}, "$defs": {"n": {"type": "integer"}}},
     "$ref takes no constraints"),
    ({"type": "object", "properties": {"s": {"minimum": 3}}}, "need a type beside them"),
    ({"type": "object", "properties": {"s": {"type": ["integer", "array"], "minimum": 3}}}, "need a single type"),
    ({"type": "object", "properties": {"s": {"type": "integer", "multipleOf": 3, "minimum": 1}}},
     "without minimum or maximum"),
    ({"type": "object", "properties": {"s": {"type": "texty"}}}, "JSON type name"),
    ({"type": "object", "properties": {"s": {"type": "string", "enum": [1]}}}, "every enum and const value"),
    ({"type": "object", "properties": {"s": {"type": "integer", "anyOf": [{"type": {"x": 1}}]}}}, "JSON type name"),
    ({"type": "object", "properties": {"s": {"type": "array", "items": {"type": "string", "pattern": "^a+@a+$",
                                                                        "format": "email"}}}}, "pattern and format"),
    # a $ref names a subschema, never data the checks do not read as one
    ({"type": "object", "properties": {"a": {"type": "string", "default": {"type": "string", "minLength": 5}},
                                       "s": {"$ref": "#/properties/a/default"}}}, "does not resolve to a subschema"),
    ({"type": "object", "properties": {"a": {"type": "integer", "examples": [{"enum": 5}]},
                                       "s": {"$ref": "#/properties/a/examples/0"}}}, "does not resolve to a subschema"),
    ({"type": "object", "properties": {"s": {"$ref": "#/properties"}}}, "does not resolve to a subschema"),
    ({"type": "object", "properties": {"s": {"anyOf": [{"type": "integer"}]}, "t": {"$ref": "#/properties/s/anyOf/²"}}},
     "does not resolve to a subschema"),
])
def test_unsupported_or_malformed_schemas_are_refused_with_their_path(schema, words):
    with pytest.raises(RequestError, match=re.escape(words)) as err:
        st.Plan([tool("f", schema)], required=False, parallel=True)
    assert str(err.value).startswith("tools[0].function.parameters")


def test_openai_limits_and_name_rules_are_refused():
    deep = {"type": "integer"}
    for _ in range(11):
        deep = {"type": "object", "properties": {"x": deep}, "required": ["x"], "additionalProperties": False}
    with pytest.raises(RequestError, match="nest at most 10"):
        st.Plan([tool("f", deep)], required=False, parallel=True)
    many = {"type": "object", "properties": {"e": {"type": "integer", "enum": list(range(1001))}}}
    with pytest.raises(RequestError, match="1000 enum values"):
        st.Plan([tool("f", many)], required=False, parallel=True)
    with pytest.raises(RequestError, match="offered twice"):
        st.Plan([tool("f"), tool("f")], required=False, parallel=True)
    with pytest.raises(RequestError, match="cannot be written"):
        st.Plan([tool("a b")], required=False, parallel=True)
    with pytest.raises(RequestError, match="strict must be a boolean"):
        st.wanted([{"type": "function", "function": {"name": "f", "strict": "yes"}}])
    assert not st.wanted([tool("f", strict=False), tool("g", strict=None)]) and st.wanted([tool("f")])


def _defs_chain(n, step, leaf):
    """A schema whose one parameter ``x`` refers to ``d0``; ``d_k`` is ``step(k)``, which refers on to ``d_k+1``."""

    defs = {f"d{k}": step(k) for k in range(n)}
    defs[f"d{n}"] = leaf
    return {"type": "object", "$defs": defs, "properties": {"x": {"$ref": "#/$defs/d0"}}, "required": ["x"]}


def test_hostile_schemas_cost_linear_work_and_are_refused_explicitly(monkeypatch):
    """Untrusted schemas never cost exponential work, recurse without bound or crash: shared ref chains are solved once,
    a self-referring string parameter, nesting past the limits and unreadable patterns are refused with their path."""

    resolved = []
    real = st._Schema.resolve
    monkeypatch.setattr(st._Schema, "resolve", lambda self, ref, where: resolved.append(ref) or real(self, ref, where))

    def value(plan):                     # the one parameter's value grammar (tool f, parameter x, required)
        tag = json.loads(plan.spec.text)["format"]
        tags = tag["elements"][1]["tags"] if plan.required else tag["tags"]
        return tags[0]["content"]["elements"][0]["elements"][1]

    # each definition refers twice to the next: 2^40 paths, each definition solved and written once
    two = lambda k: {"anyOf": [{"$ref": f"#/$defs/d{k + 1}"}, {"$ref": f"#/$defs/d{k + 1}"}]}   # noqa: E731
    for leaf, mode in (({"type": "string", "pattern": "^a+$"}, "raw"), ({"type": "integer"}, "json")):
        resolved.clear()
        plan = st.Plan([tool("f", _defs_chain(40, two, leaf))], required=True, parallel=True)
        assert plan.tools["f"].params[0].mode == mode and len(resolved) <= 8 * 81, len(resolved)
        if mode == "raw":
            assert value(plan) == {"type": "regex", "pattern": "^a+$"}
    # 3,000 chained references (past Python's recursion limit): followed without recursion
    resolved.clear()
    one = lambda k: {"anyOf": [{"$ref": f"#/$defs/d{k + 1}"}, {"const": f"c{k}"}]}             # noqa: E731
    plan = st.Plan([tool("f", _defs_chain(3000, one, {"const": "end"}))], required=False, parallel=True)
    assert plan.tools["f"].params[0].mode == "raw" and len(resolved) <= 4 * 3001, len(resolved)
    assert {e["value"] for e in value(plan)["elements"]} == {"end", *(f"c{k}" for k in range(3000))}
    # a JSON parameter's grammar holds at most MAX_SUBSCHEMAS subschemas, its definitions included: xgrammar 0.2.8
    # compiles a reference chain recursively and overflowed its stack on longer ones
    ref = lambda k: {"$ref": f"#/$defs/d{k + 1}"}                                               # noqa: E731
    st.Plan([tool("f", _defs_chain(st.MAX_SUBSCHEMAS - 2, ref, {"type": "integer"}))], required=False, parallel=True)
    with pytest.raises(RequestError, match=rf"properties\.x: this parameter's schema holds {st.MAX_SUBSCHEMAS + 2} subs"):
        st.Plan([tool("f", _defs_chain(st.MAX_SUBSCHEMAS, ref, {"type": "integer"}))], required=False, parallel=True)
    looped = {"type": "object", "properties": {"x": {"$ref": "#/$defs/a"}},
              "$defs": {"a": {"type": "string", "anyOf": [{"$ref": "#/$defs/a"}, {"type": "string", "pattern": "^a$"}]}}}
    with pytest.raises(RequestError, match=r"properties\.x\.anyOf\[0\]: a string parameter's schema may not refer back"):
        st.Plan([tool("f", looped)], required=False, parallel=True)

    def nest(levels, wrap):
        node = {"type": "integer"}
        for _ in range(levels):
            node = wrap(node)
        return node

    def any_of(levels):                  # the innermost schema ``levels`` anyOf levels deep
        return {"type": "object", "properties": {"x": nest(levels, lambda s: {"anyOf": [s]})}}

    def defs(levels):                    # the innermost definition ``levels`` $defs levels deep
        inner = nest(levels - 1, lambda s: {"type": "integer", "$defs": {"a": s}})
        return {"type": "object", "properties": {"x": {"type": "integer"}}, "$defs": {"a": inner}}

    for shape in (any_of, defs):
        st.Plan([tool("f", shape(10))], required=False, parallel=True)
        with pytest.raises(RequestError, match="anyOf and \\$defs nest at most 10 levels"):
            st.Plan([tool("f", shape(11))], required=False, parallel=True)
    for pattern, words in (("(" * 600 + "a" + ")" * 600, "nests its groups too deeply"),
                           ("(a", "not a regular expression this server reads")):
        for where in ({"type": "string", "pattern": pattern},                       # a raw value
                      {"type": "array", "items": {"type": "string", "pattern": pattern}}):     # inside JSON
            with pytest.raises(RequestError, match=words):
                st.Plan([tool("f", {"type": "object", "properties": {"x": where}})], required=False, parallel=True)


def test_const_beside_an_enum_must_be_one_of_its_values():
    """xgrammar 0.2.8 enforces const alone when enum stands beside it: a const outside the enum is refused, one inside
    it keeps the grammar that both describe."""

    for value, enum in ((2, [1]), ("b", ["a"]), (1, [True]), ({"k": 1}, [{"k": 2}])):
        schema = {"type": "object", "properties": {"x": {"enum": enum, "const": value}}}
        with pytest.raises(RequestError, match="const must be one of the enum values"):
            st.Plan([tool("f", schema)], required=False, parallel=True)
    for value, enum in ((2, [1, 2]), ("b", ["a", "b"]), ({"k": 1, "j": 2}, [{"j": 2, "k": 1}])):
        st.Plan([tool("f", {"type": "object", "properties": {"x": {"enum": enum, "const": value}}})],
                required=False, parallel=True)


def test_definitions_are_copied_only_where_referred_and_the_grammar_size_is_capped(monkeypatch):
    """A JSON parameter's grammar copies the definitions it refers to (all of them, with the root, when it refers to
    the root), so shared definitions grow the grammar only where used; past ``MAX_GRAMMAR_BYTES`` it is refused, as
    the two-rank transport would refuse it later."""

    from tensorfold.cuda import rank_protocol

    assert st.MAX_GRAMMAR_BYTES == rank_protocol.MAX_MESSAGE_BYTES
    item = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
    defs = {f"d{k}": item for k in range(200)}
    props = {f"p{i}": {"type": "integer"} for i in range(200)}
    props["used"] = {"type": "array", "items": {"$ref": "#/$defs/d7"}}
    plan = st.Plan([tool("f", {"type": "object", "$defs": defs, "properties": props})], required=False, parallel=True)
    params = json.loads(plan.spec.text)["format"]["tags"][0]["content"]["elements"]
    schemas = [p["content"]["elements"][1]["json_schema"] for p in params]
    assert all("$defs" not in s for s in schemas[:200]) and list(schemas[200]["$defs"]) == ["d7"]
    rooted = {"type": "object", "$defs": {"n": {"type": "integer"}},
              "properties": {f"p{i}": {"type": "array", "items": {"$ref": "#"}, "maxItems": 1} for i in range(40)}}
    plan = st.Plan([tool("f", rooted)], required=False, parallel=True)
    size = len(plan.spec.text.encode())
    monkeypatch.setattr(st, "MAX_GRAMMAR_BYTES", size)
    st.Plan([tool("f", rooted)], required=False, parallel=True)
    monkeypatch.setattr(st, "MAX_GRAMMAR_BYTES", size - 1)
    with pytest.raises(RequestError, match=r"^tools: the strict tools' call grammar exceeds"):
        st.Plan([tool("f", rooted)], required=False, parallel=True)


def test_patterns_that_cannot_spell_the_value_close_are_kept():
    for pattern in ("^[0-9a-f]{64}$", "^g[1-9][0-9]*$", "^<[a-z]+>$", "^(a|b)c?$"):
        plan = st.Plan([tool("f", {"type": "object", "properties": {"s": {"type": "string", "pattern": pattern}},
                                   "required": ["s"], "additionalProperties": False})], required=False, parallel=True)
        assert plan.tools["f"].params[0].mode == "raw"


# -- reading replies ----------------------------------------------------------------------------------------------------
CALL = ('<tool_call>\n<function=get_weather>\n<parameter=city>\nPa\nris\n</parameter>\n'
        '<parameter=days>\n3\n</parameter>\n<parameter=unit>\nc\n</parameter>\n'
        '<parameter=opts>\n{"a": "x</parameter>y", "b": [1,\n 2]}\n</parameter>\n'
        '<parameter=note>\nnull\n</parameter>\n<parameter=id>\n123e4567-e89b-12d3-a456-426614174000\n</parameter>\n'
        '</function>\n</tool_call>')
ARGS = {"city": "Pa\nris", "days": 3, "unit": "c", "opts": {"a": "x</parameter>y", "b": [1, 2]}, "note": None,
        "id": "123e4567-e89b-12d3-a456-426614174000"}


def _streamed(plan, text, step=1):
    reader, out = plan.reader(), {}
    for i in range(step, len(text) + step, step):
        for delta in reader.feed(text[:i]):
            call = delta["tool_calls"][0]
            head = out.setdefault(call["index"], {"name": None, "arguments": ""})
            if "id" in call:
                head["name"] = call["function"]["name"]
            head["arguments"] += call["function"]["arguments"]
    return [out[i] for i in sorted(out)]


def test_the_reader_parses_calls_text_and_streams_the_same_arguments():
    plan = st.Plan(TOOLS, required=False, parallel=True)
    loose = '<tool_call>\n<function=loose>\n<parameter=n>\n5\n</parameter>\n<parameter=zz>\nhi\n</parameter>\n' \
            '</function>\n</tool_call>'
    now = '<tool_call>\n<function=now>\n</function>\n</tool_call>'
    text = "Let me check.\n\n" + CALL + "\n" + loose + "\n" + now
    content, calls = plan.parse(text)
    assert content == "Let me check."
    assert [c["function"]["name"] for c in calls] == ["get_weather", "loose", "now"]
    assert json.loads(calls[0]["function"]["arguments"]) == ARGS
    assert calls[0]["function"]["arguments"].index('"opts":{"a": "x</parameter>y", "b": [1,\n 2]}') > 0   # verbatim
    assert json.loads(calls[1]["function"]["arguments"]) == {"n": 5, "zz": "hi"}      # not strict: decoded as before
    assert calls[2]["function"]["arguments"] == "{}"
    assert all(c["id"].startswith("call_") and c["type"] == "function" for c in calls)
    for step in (1, 3, 17):
        streamed = _streamed(plan, text, step)
        assert [(s["name"], s["arguments"]) for s in streamed] == [
            (c["function"]["name"], c["function"]["arguments"]) for c in calls], step
    # the streamed content holds back a possible opener and a call still being written
    assert plan.content("Hi <tool_c", finished=False) == "Hi "
    assert plan.content("Hi <tool_c", finished=True) == "Hi <tool_c"
    assert plan.content("Hi " + CALL[:40], finished=False) == "Hi "


def test_a_tool_without_strict_keeps_the_lenient_arguments_and_errors_name_the_requests_tool():
    plan = st.Plan(TOOLS, required=False, parallel=True)
    repeated = ('<tool_call>\n<function=loose>\n<parameter=n>\n5\n</parameter>\n<parameter=s>\nx\n</parameter>\n'
                '<parameter=n>\n6\n</parameter>\n</function>\n</tool_call>')
    calls = plan.parse(repeated)[1]
    assert calls[0]["function"]["arguments"] == '{"n":6,"s":"x"}'              # the last value, as the lenient parser
    assert [(s["name"], s["arguments"]) for s in _streamed(plan, repeated)] == [("loose", '{"n":6,"s":"x"}')]
    bad = [tool("ok"), tool("f", {"type": "object", "properties": {"s": {"type": "string", "maxLength": 3}}})]
    with pytest.raises(RequestError, match=r"^tools\[1\]\.function\.parameters"):
        st.Plan(bad[1:], required=True, parallel=True, offered=bad)          # a named tool_choice's one active tool


def test_an_unfinished_call_stays_text_and_a_raw_null_is_null():
    plan = st.Plan(TOOLS, required=False, parallel=True)
    content, calls = plan.parse("ok " + CALL[:-30])
    assert calls is None and content == ("ok " + CALL[:-30]).strip()
    text = CALL.replace("<parameter=note>\nnull", "<parameter=note>\nnull or not")
    assert json.loads(plan.parse(text)[1][0]["function"]["arguments"])["note"] == "null or not"
    assert plan.parse("x", max_calls=1) == ("x", None)
    two = plan.parse(CALL + "\n" + CALL, max_calls=1)[1]
    assert len(two) == 1


def test_markup_outside_the_grammar_is_a_fault_not_a_call():
    plan = st.Plan(TOOLS, required=False, parallel=True)
    for bad in ('<tool_call>\n<function=nope>\n</function>\n</tool_call>',
                '<tool_call>{"name": "now"}</tool_call>',
                CALL.replace("<parameter=days>", "<parameter=unit>", 1),
                CALL.replace("<parameter=city>\nPa\nris\n", "<parameter=city>\nParis")):
        with pytest.raises(st.CallMarkupError):
            plan.parse(bad)


# -- the grammar: compiled by xgrammar, fuzzed with an independent validator ---------------------------------------------
SPECIALS = ["<tool_call>", "</tool_call>", "<think>", "</think>", "<parameter=", "\n</parameter>\n",
            "</function>\n</tool_call>", "<function=", ">\n", '": "', "null", "true", "false", "123e4567-e89b-12d3-"]
V = 256
STOP = 0


def _vocab() -> list[str]:
    words = [""] + [chr(i) for i in range(1, 128)]
    words += SPECIALS + [""] * (V - len(words) - len(SPECIALS))
    return words


@pytest.fixture(scope="module")
def grammars():
    xgr = pytest.importorskip("xgrammar")
    pytest.importorskip("torch")
    from tensorfold.engine import grammar

    return grammar.Grammars(xgr.TokenizerInfo(_vocab(), xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[STOP]))


def test_the_grammar_accepts_the_templates_calls_and_refuses_others(grammars):
    plan = st.Plan(TOOLS, required=False, parallel=True)
    compiled = grammars.compile(plan.spec)
    loose = '<tool_call>\n<function=loose>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>'
    good = [CALL, "text only", "Sure.\n\n" + CALL + "\n" + CALL, loose,
            CALL.replace("</function>", "<parameter=flag>\ntrue\n</parameter>\n</function>"),
            CALL.replace("\nnull\n", "\nany note at all\n"),
            '<tool_call>\n<function=chain>\n<parameter=head>\n{"v": 1, "next": {"v": 2, "next": null}}\n'
            '</parameter>\n</function>\n</tool_call>']
    bad = [CALL.replace("\n3\n", "\n9\n"), CALL.replace("\n3\n", "\n 3\n"), CALL.replace("\nc\n", "\n\nc\n"),
           CALL.replace("\nc\n", "\nk\n"), CALL.replace("[1,\n 2]", "[1, 4]"), CALL.replace("[1,\n 2]", "[1, 2, 3]"),
           CALL.replace("<parameter=days>\n3\n</parameter>\n", ""), CALL.replace("123e4567", "123x4567"),
           CALL.replace("\nPa\nris\n", "\nPa</parameter>ris\n"), CALL.replace('"b"', '"c"'),
           '<tool_call>\n<function=nope>\n</function>\n</tool_call>', 'hi <tool_call>{"name": "now"}',
           '<tool_call>\n<function=chain>\n<parameter=head>\n{"v": 1}\n</parameter>\n</function>\n</tool_call>']
    assert [grammars.accepts(compiled, t) for t in good] == [True] * len(good)
    assert [grammars.accepts(compiled, t) for t in bad] == [False] * len(bad)
    required = grammars.compile(st.Plan(TOOLS, required=True, parallel=True).spec)
    assert grammars.accepts(required, "\n\n" + CALL + "\n" + CALL + "\n")
    assert not grammars.accepts(required, "text " + CALL) and not grammars.accepts(required, "\n")
    single = grammars.compile(st.Plan(TOOLS, required=False, parallel=False).spec)
    assert grammars.accepts(single, "pre " + CALL) and not grammars.accepts(single, CALL + "\n" + CALL)
    assert not grammars.accepts(single, CALL + " post")


def test_the_longest_reference_chains_allowed_compile(grammars):
    """At ``MAX_SUBSCHEMAS`` a JSON parameter's reference chain still compiles in xgrammar 0.2.8 (its compiler recurses
    along references: much longer chains overflowed its stack and killed the process), and the grammar holds it."""

    levels = (st.MAX_SUBSCHEMAS - 2) // 3            # per level: the definition, its reference and its const branch
    step = lambda k: {"anyOf": [{"$ref": f"#/$defs/d{k + 1}"}, {"const": k}]}                  # noqa: E731
    for schema in (_defs_chain(levels, step, {"const": -1}),
                   _defs_chain(st.MAX_SUBSCHEMAS - 2, lambda k: {"$ref": f"#/$defs/d{k + 1}"}, {"type": "integer"})):
        plan = st.Plan([tool("f", schema)], required=True, parallel=True)
        compiled = grammars.compile(plan.spec)
        call = "<tool_call>\n<function=f>\n<parameter=x>\n{}\n</parameter>\n</function>\n</tool_call>"
        assert grammars.accepts(compiled, call.format(-1 if "anyOf" in schema["$defs"]["d0"] else 7))
        assert not grammars.accepts(compiled, call.format('"x"'))


def test_refs_into_the_tools_schema_resolve_inside_a_json_parameter(grammars):
    """A JSON parameter's grammar keeps refs to its tool's root (``#/properties/...``, recursion) and ``$defs``, also
    under a property whose name is a keyword."""

    tree = {"type": "object", "additionalProperties": False, "required": ["kids"],
            "properties": {"kids": {"type": "array", "maxItems": 2, "items": {"$ref": "#/properties/tree"}}}}
    schema = {"type": "object", "additionalProperties": False, "required": ["tree", "copy", "enum"],
              "properties": {"tree": tree, "copy": {"$ref": "#/properties/tree"},
                             "enum": {"type": "array", "items": {"$ref": "#/$defs/n"}}},
              "$defs": {"n": {"type": "integer", "enum": [1, 2]}}}
    plan = st.Plan([tool("t", schema)], required=False, parallel=True)
    compiled = grammars.compile(plan.spec)
    call = ('<tool_call>\n<function=t>\n<parameter=tree>\n{"kids": [{"kids": []}, {"kids": [{"kids": []}]}]}\n'
            '</parameter>\n<parameter=copy>\n{"kids": []}\n</parameter>\n<parameter=enum>\n[1, 2]\n</parameter>\n'
            '</function>\n</tool_call>')
    assert grammars.accepts(compiled, call)
    assert conforms(json.loads(plan.parse(call)[1][0]["function"]["arguments"]), schema)
    for bad in (call.replace("[1, 2]", "[3]"), call.replace('{"kids": []}\n</par', '{"kids": [], "x": 1}\n</par'),
                call.replace('[{"kids": []}, {"kids": [{"kids": []}]}]', '[{"kids": []}, {"kids": []}, {"kids": []}]')):
        assert not grammars.accepts(compiled, bad)


def test_definitions_keep_their_refs_to_the_tools_root_inside_a_json_parameter(grammars):
    """Inside a JSON parameter's grammar, a ``$defs`` entry's ``#`` still names the tool's schema (not the parameter's),
    and a definition may be named like a schema keyword."""

    schema = {"type": "object", "additionalProperties": False, "required": ["a", "b"],
              "properties": {"a": {"$ref": "#/$defs/wrap"}, "b": {"$ref": "#/$defs/properties"}},
              "$defs": {"wrap": {"type": "object", "additionalProperties": False, "required": ["inner"],
                                 "properties": {"inner": {"anyOf": [{"$ref": "#"}, {"type": "null"}]}}},
                        "properties": {"type": "integer", "enum": [1, 2]}}}
    plan = st.Plan([tool("t", schema)], required=False, parallel=True)
    compiled = grammars.compile(plan.spec)

    def call(a, b):
        return (f"<tool_call>\n<function=t>\n<parameter=a>\n{a}\n</parameter>\n<parameter=b>\n{b}\n</parameter>\n"
                "</function>\n</tool_call>")

    good = call('{"inner": {"a": {"inner": null}, "b": 2}}', "1")
    assert grammars.accepts(compiled, good)
    assert conforms(json.loads(plan.parse(good)[1][0]["function"]["arguments"]), schema)
    for bad in (call('{"inner": {"inner": null}}', "1"), call('{"inner": null}', "3")):
        assert not grammars.accepts(compiled, bad)


@pytest.mark.parametrize("fmt", sorted(st.FORMATS))
def test_raw_formats_match_xgrammars_json_formats(grammars, fmt):
    """The raw format regexes accept exactly what xgrammar's own JSON-schema formats accept (a corpus of samples)."""

    import xgrammar as xgr

    samples = ["2024-02-29T23:59:60.5+05:30", "2024-13-01T00:00:00Z", "1999-12-31T10:20:30Z", "2024-02-29",
               "2024-2-29", "23:59:60Z", "24:00:00Z", "12:30:00.123-08:00", "P1D", "PT1H30M", "P1Y2M3DT4H5M6S",
               "P3W", "P", "PT", "1 day", "a@b.co", "first.last+tag@sub.example.org", "a@b", "@b.co", "a.@b.co",
               "host", "a-b.example", "-a.example", "A.example", "1.2.3.4", "255.255.255.255", "256.1.1.1",
               "01.2.3.4", "::1", "2001:db8::1", "::ffff:1.2.3.4", "fe80::1:2:3:4:5", "1:2:3:4:5:6:7:8", "1::2::3",
               "123e4567-e89b-12d3-a456-426614174000", "123E4567-E89B-12D3-A456-426614174000", "123e4567e89b"]
    as_json = grammars.compiler.compile_json_schema(json.dumps({"type": "string", "format": fmt}))
    raw = grammars.compiler.compile_structural_tag(json.dumps(
        {"type": "structural_tag", "format": {"type": "regex", "pattern": st.FORMATS[fmt]}}))
    for s in samples:
        want = xgr.GrammarMatcher(as_json)
        want = want.accept_string(json.dumps(s)) and want.is_completed()
        assert grammars.accepts(raw, s) == want, (fmt, s)
        assert "<" not in s or not want


def _toy_model(seed: int, bonus: dict[int, float]):
    """A made-up model: seeded logits per position, closing markup favoured so replies end."""

    import torch

    gen = torch.Generator().manual_seed(seed)

    def logits(position: int):
        row = torch.randn(1, V, generator=gen) * 2
        for t, b in bonus.items():
            row[0, t] += b
        return row
    return logits


def _decode(grammars, compiled, model, count=700, *, think_end=None, prefix=()):
    c = grammars.constraint(compiled, think_end=think_end)
    c.advance(list(prefix))
    out = list(prefix)
    while len(out) < count and (not out or out[-1] != STOP):
        row = c.mask(model(len(out)))
        token = int(row.argmax())
        c.advance([token])
        out.append(token)
    return out


def _text(tokens) -> str:
    words = _vocab()
    return "".join(words[t] for t in tokens if t != STOP)


@pytest.mark.parametrize("required, parallel", [(False, True), (True, True), (True, False), (False, False)])
def test_every_reply_the_grammar_allows_parses_to_conforming_calls(grammars, required, parallel):
    """Fuzz: random replies under each tool_choice and parallel_tool_calls; every finished one parses to calls whose
    arguments satisfy their schemas (the oracle above), and the streamed deltas equal the final calls."""

    words = _vocab()
    favour = {words.index(w): 3.0 for w in SPECIALS[4:]}
    favour.update({STOP: 2.0, ord("}"): 2.5, ord("]"): 2.5, ord('"'): 1.5, ord("\n"): 0.5})
    favour[words.index("<tool_call>")] = 4.0
    plan = st.Plan(TOOLS[:3], required=required, parallel=parallel)
    compiled = grammars.compile(plan.spec)
    finished = called = 0
    for seed in range(40):
        tokens = _decode(grammars, compiled, _toy_model(seed, favour))
        if tokens[-1] != STOP:
            continue
        finished += 1
        text = _text(tokens)
        assert grammars.accepts(compiled, text), text
        content, calls = plan.parse(text)
        if required:
            assert calls and content == "", text
        if not parallel:
            assert len(calls or ()) <= 1
        for call in calls or ():
            called += 1
            name = call["function"]["name"]
            assert conforms(json.loads(call["function"]["arguments"]), SCHEMAS[name]), (name, text)
        streamed = _streamed(plan, text, step=2)
        assert [(s["name"], s["arguments"]) for s in streamed] == [
            (c["function"]["name"], c["function"]["arguments"]) for c in calls or ()]
    assert finished >= 20 and called >= 10, (finished, called)


def test_a_named_tool_is_the_only_call_and_thinking_stays_free(grammars):
    words = _vocab()
    think_end = words.index("</think>")
    plan = st.Plan([TOOLS[2]], required=True, parallel=False)
    compiled = grammars.compile(plan.spec)
    favour = {STOP: 2.0, words.index("</function>\n</tool_call>"): 3.0}
    # the model thinks freely (no grammar) until </think>, then can only call ``now``
    thought = [ord(c) for c in "I think <tool_call> is a fine word"] + [think_end]
    tokens = _decode(grammars, compiled, _toy_model(5, favour), think_end=think_end, prefix=thought)
    text = _text(tokens)
    answer = text.split("</think>", 1)[1]
    assert plan.parse(answer)[1][0]["function"]["name"] == "now" and plan.parse(answer)[0] == ""


def test_a_fresh_copy_starts_the_grammar_again_and_the_other_rank_follows_the_same_one(grammars):
    """``Constraint.fresh`` (a yielded stream's replay) and ``pack``/``follow`` (rank 1 of two) give the grammar of
    the original: the same masks along the same tokens."""

    import numpy as np

    from tensorfold.cuda import rank_protocol
    from tensorfold.engine import grammar

    plan = st.Plan(TOOLS[:3], required=False, parallel=True)
    words = _vocab()
    think_end = words.index("</think>")
    original = grammars.constraint(grammars.compile(plan.spec), think_end=think_end, spec=plan.spec)
    tokens = [ord("a"), think_end] + MarkupTokenizer().encode(CALL).ids + [STOP]
    original.advance(tokens[:20])
    again = original.fresh()
    packed = rank_protocol.grammar(grammar.pack(again), V)
    follower = grammars.follow(packed)
    assert again.spec == plan.spec and (follower.spec.kind, follower.spec.text) == (plan.spec.kind, plan.spec.text)
    assert again.think_end == follower.think_end == think_end
    assert not again.active and again.m is not original.m
    for t in tokens:
        a, b = again.window([t], [-1]), follower.window([t], [-1])
        assert a.rows == b.rows and (a.bits is None) == (b.bits is None)
        assert a.bits is None or np.array_equal(a.bits, b.bits)
        again.advance([t])
        follower.advance([t])
    assert again.finished and follower.finished


def test_identical_requests_decode_identical_tokens(grammars):
    words = _vocab()
    favour = {words.index("<tool_call>"): 4.0, STOP: 2.0, words.index("\n</parameter>\n"): 3.0}
    compiled = grammars.compile(st.Plan(TOOLS[:3], required=False, parallel=True).spec)
    again = grammars.compile(st.Plan(TOOLS[:3], required=False, parallel=True).spec)
    for seed in range(5):
        assert _decode(grammars, compiled, _toy_model(seed, favour)) == _decode(
            grammars, again, _toy_model(seed, favour))


# -- the CUDA server --------------------------------------------------------------------------------------------------
class MarkupTokenizer:
    """ASCII characters are their own ids; the toy vocabulary's markup tokens follow them (as the fixture's)."""

    words = _vocab()
    ids = {w: i for i, w in enumerate(words) if len(w) > 1}

    def encode(self, text, **kwargs):
        from types import SimpleNamespace

        out, i = [], 0
        while i < len(text):
            special = next((w for w in ("<tool_call>", "</think>", "<think>") if text.startswith(w, i)), None)
            out.append(self.ids[special] if special else ord(text[i]))
            i += len(special) if special else 1
        return SimpleNamespace(ids=out)

    def decode(self, ids, **kwargs):
        return "".join(self.words[i] for i in ids)

    def token_to_id(self, text):
        return self.ids.get(text, ord(text) if len(text) == 1 else None)

    def id_to_token(self, token):
        return self.words[token]


class ScriptEngine:
    """Writes ``script`` where its grammar lets it; elsewhere the allowed token it likes best: the end, closing markup,
    then the lowest id (any disallowed token would fail the reply)."""

    eos = (STOP,)
    likes = {STOP: 2.0, ord("}"): 1.5, ord("]"): 1.5, ord('"'): 1.0, MarkupTokenizer.ids["null"]: 1.0,
             MarkupTokenizer.ids["\n</parameter>\n"]: 2.0, MarkupTokenizer.ids["</function>\n</tool_call>"]: 2.0,
             MarkupTokenizer.ids["<tool_call>"]: 0.5}

    def __init__(self, script: str) -> None:
        self.script = MarkupTokenizer().encode(script).ids + [STOP]
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        import torch

        self.calls.append({"constraint": constraint, "prompt": list(prompt)})
        for i in range(max_tokens):
            want = self.script[i] if i < len(self.script) else STOP
            t = want
            if constraint is not None:
                logits = torch.zeros(1, V)
                for liked, bonus in self.likes.items():
                    logits[0, liked] = bonus
                logits[0, want] = 3.0
                t = int(constraint.mask(logits).argmax())
                constraint.advance([t])
            if on_tokens([t]) or t == STOP:
                break
        return {}


class PlainEngine(ScriptEngine):
    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft)


def _app(tmp_path, engine, grammars, template=QWEN_CALLS):
    from tensorfold.cuda import server

    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.model_dir = engine, "fake-cuda", tmp_path
    app.tok, app.template = MarkupTokenizer(), server.ChatTemplate(tmp_path)
    app.default_thinking, app.sampling, app.max_tokens = False, {"temperature": 0.0}, 2000
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    app.vision = None
    app.grammars = grammars
    return app


def _body(stream, tools=TOOLS[:3], **extra):
    return {"messages": [{"role": "user", "content": "Hi"}], "stream": stream, "tools": tools, **extra}


def _reply(stream: bool, text: str):
    if not stream:
        message = json.loads(text)["choices"][0]
        return message["message"].get("content"), [
            (c["function"]["name"], c["function"]["arguments"]) for c in message["message"].get("tool_calls") or ()
        ], message["finish_reason"]
    content, calls, finish = "", {}, None
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for choice in json.loads(line[6:]).get("choices", []):
                delta = choice["delta"]
                content += delta.get("content") or ""
                for call in delta.get("tool_calls") or ():
                    entry = calls.setdefault(call["index"], ["", ""])
                    entry[0] += (call.get("function") or {}).get("name") or ""
                    entry[1] += (call.get("function") or {}).get("arguments") or ""
                finish = choice.get("finish_reason") or finish
    return content or None, [tuple(calls[i]) for i in sorted(calls)], finish


@pytest.mark.parametrize("stream", [False, True])
def test_a_strict_request_returns_the_grammars_calls(tmp_path, grammars, stream):
    from tests.test_cuda_admission import http_server, post

    engine = ScriptEngine("Let me look.\n\n" + CALL)
    with http_server(_app(tmp_path, engine, grammars)) as port:
        status, text = post(port, _body(stream), True)
    assert status == 200, text
    content, calls, finish = _reply(stream, text)
    assert content.strip() == "Let me look." and finish == "tool_calls"
    assert [name for name, _ in calls] == ["get_weather"] and json.loads(calls[0][1]) == ARGS
    assert engine.calls[0]["constraint"].spec.kind == "tools" and engine.calls[0]["constraint"].finished


@pytest.mark.parametrize("stream", [False, True])
def test_the_grammar_holds_a_call_the_model_writes_wrong(tmp_path, grammars, stream):
    """The scripted model writes days 9 (maximum 7) and an unknown unit: the mask keeps the call valid."""

    from tests.test_cuda_admission import http_server, post

    wrong = CALL.replace("\n3\n", "\n9\n").replace("\nc\n", "\nk\n")
    with http_server(_app(tmp_path, ScriptEngine(wrong), grammars)) as port:
        status, text = post(port, _body(stream), True)
    assert status == 200, text
    _, calls, finish = _reply(stream, text)
    args = json.loads(calls[0][1])
    assert conforms(args, WEATHER) and args["days"] != 9 and args["unit"] in ("c", "f") and finish == "tool_calls"


@pytest.mark.parametrize("stream", [False, True])
def test_auto_allows_plain_text_and_required_forces_a_call(tmp_path, grammars, stream):
    from tests.test_cuda_admission import http_server, post

    with http_server(_app(tmp_path, ScriptEngine("Just words."), grammars)) as port:
        status, text = post(port, _body(stream), True)
        assert status == 200 and _reply(stream, text) == ("Just words.", [], "stop")
        status, text = post(port, _body(stream, tool_choice="required"), True)
        assert status == 200, text
        content, calls, finish = _reply(stream, text)
        assert calls and not (content or "").strip() and finish == "tool_calls"
        status, text = post(port, _body(stream, tool_choice={"type": "function", "function": {"name": "now"}}), True)
        assert status == 200 and _reply(stream, text)[1] == [("now", "{}")]


@pytest.mark.parametrize("stream", [False, True])
def test_parallel_tool_calls_false_ends_the_reply_after_one_call(tmp_path, grammars, stream):
    from tests.test_cuda_admission import http_server, post

    engine = ScriptEngine(CALL + "\n" + CALL)
    with http_server(_app(tmp_path, engine, grammars)) as port:
        status, text = post(port, _body(stream, parallel_tool_calls=False), True)
        assert status == 200 and len(_reply(stream, text)[1]) == 1
        status, text = post(port, _body(stream), True)
        assert status == 200 and len(_reply(stream, text)[1]) == 2


def test_thinking_stays_free_and_a_valid_call_in_an_open_think_block_is_promoted(tmp_path, grammars):
    from tests.test_cuda_admission import http_server, post

    thinking = {"chat_template_kwargs": {"enable_thinking": True}}
    free = ScriptEngine("hmm <tool_call> 9 days</think>\n\n" + CALL)
    with http_server(_app(tmp_path, free, grammars)) as port:
        status, text = post(port, _body(False, **thinking), True)
    message = json.loads(text)["choices"][0]["message"]
    assert status == 200 and message["reasoning_content"].startswith("hmm <tool_call> 9 days")
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == ARGS
    assert free.calls[0]["constraint"].think_end == MarkupTokenizer.ids["</think>"]
    # a call written without closing the think block: the answer when the grammar accepts it, else reasoning
    for body, called in ((CALL, True), (CALL.replace("\n3\n", "\n9\n"), False)):
        with http_server(_app(tmp_path, ScriptEngine("plan: " + body), grammars)) as port:
            status, text = post(port, _body(False, **thinking), True)
        message = json.loads(text)["choices"][0]["message"]
        assert status == 200 and bool(message.get("tool_calls")) == called, message
        if not called:
            assert "9" in message["reasoning_content"] and not message.get("content")


@pytest.mark.parametrize("body, words", [
    (_body(False, tools=[tool("f", {"type": "object", "properties": {"s": {"type": "string", "maxLength": 3}}})]),
     "do not support maxLength"),
    (_body(False, response_format={"type": "json_object"}), "cannot be combined with strict function tools"),
    (_body(False, tools=[tool("f", strict="yes")]), "strict must be a boolean"),
])
def test_unsupported_strict_requests_are_refused_before_generation(tmp_path, grammars, body, words):
    from tests.test_cuda_admission import http_server, post

    engine = ScriptEngine("x")
    for stream in (False, True):
        with http_server(_app(tmp_path, engine, grammars)) as port:
            status, text = post(port, {**body, "stream": stream}, True)
        assert status == 400 and words in json.loads(text)["error"]["message"], text
    assert engine.calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_reply_that_leaves_the_grammar_fails_the_request_instead_of_returning_a_call(tmp_path, grammars, stream):
    """An engine fault that writes markup the strict grammar never allows is an internal error (HTTP 500, or an error
    event once streaming), never a call read leniently or text passed off as the answer."""

    from tests.test_cuda_admission import http_server, post

    class Ignoring(ScriptEngine):                    # takes the constraint but never masks with it
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
            return super().generate(prompt, max_tokens, sampling, on_tokens, draft)

    with http_server(_app(tmp_path, Ignoring('<tool_call>\n<function=now>\n{"x": 1}</tool_call>'), grammars)) as port:
        status, text = post(port, _body(stream), True)
    if stream:                                       # the call's head went out before its body left the grammar
        assert status == 200 and '"type": "server_error"' in text and "left its grammar" in text, text
        assert '"finish_reason": "tool_calls"' not in text and '"arguments": "{' not in text, text
    else:
        assert status == 500 and "left its grammar" in json.loads(text)["error"]["message"], text


def test_engines_or_templates_that_cannot_hold_calls_refuse_strict(tmp_path, grammars):
    from tests.test_cuda_admission import http_server, post

    with http_server(_app(tmp_path, PlainEngine("x"), grammars)) as port:
        status, text = post(port, _body(False), True)
    assert status == 400 and "does not enforce strict function tools" in text
    hermes = QWEN_CALLS.replace("'<tool_call>\\n<function='", "'<tool_call><function='")
    with http_server(_app(tmp_path, ScriptEngine("x"), grammars, hermes)) as port:
        status, text = post(port, _body(False), True)
    assert status == 400 and "Qwen XML tool calls" in text and "writes calls otherwise" in text
    python_values = QWEN_CALLS.replace("v|tojson", "v|string")
    with http_server(_app(tmp_path, ScriptEngine("x"), grammars, python_values)) as port:
        status, text = post(port, _body(False), True)
    assert status == 400 and "writes calls otherwise" in text
    refusing = QWEN_CALLS.replace("{% for tc in (m.tool_calls or []) %}",
                                  "{% if m.tool_calls %}{{ raise_exception('no calls here') }}{% endif %}"
                                  "{% for tc in (m.tool_calls or []) %}")
    with http_server(_app(tmp_path, ScriptEngine("x"), grammars, refusing)) as port:
        status, text = post(port, _body(False), True)
    assert status == 400 and "could not render a call (no calls here)" in text


@pytest.mark.parametrize("tool_support", [False, True])
def test_strict_probe_uses_the_actual_tool_route_instead_of_default_markup(tmp_path, grammars, tool_support):
    from tensorfold.server.errors import RequestError

    sources = {"default": "ordinary chat" if tool_support else QWEN_CALLS,
               "tool_use": QWEN_CALLS if tool_support else "tool route writes other markup"}
    app = _app(tmp_path, ScriptEngine("x"), grammars, sources)
    if tool_support:
        assert app._prepare(_body(False), True).calls is not None
    else:
        with pytest.raises(RequestError, match="writes calls otherwise"):
            app._prepare(_body(False), True)


@pytest.mark.parametrize("stream", [False, True])
def test_requests_without_strict_tools_decode_as_before(tmp_path, grammars, stream):
    """No strict tool: no constraint reaches the engine, and the lenient parser reads the reply as it always has."""

    from tests.test_cuda_admission import http_server, post

    loose = [tool(t["function"]["name"], t["function"].get("parameters"), strict=None) for t in TOOLS[:3]]
    for tools in (loose, [{**t, "function": {**t["function"], "strict": False}} for t in loose]):
        engine = ScriptEngine("Let me look.\n\n" + CALL)
        with http_server(_app(tmp_path, engine, grammars)) as port:
            status, text = post(port, _body(stream, tools=tools), True)
        assert status == 200, text
        assert engine.calls[0]["constraint"] is None
        _, calls, finish = _reply(stream, text)
        assert [name for name, _ in calls] == ["get_weather"] and finish == "tool_calls"


# -- the Mac server's shared pieces (its MLX engine cannot run here: these are the parts Linux can exercise) ----------
class TemplateTokenizer(MarkupTokenizer):
    """A Hugging Face-like tokenizer over the toy vocabulary, rendering ``template`` as ``apply_chat_template``."""

    def __init__(self, template):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
        self.template = env.from_string(template)

    def apply_chat_template(self, messages, *, add_generation_prompt=False, enable_thinking=False, tools=None,
                            tokenize=True, **kwargs):
        text = self.template.render(messages=messages, tools=tools, enable_thinking=enable_thinking)
        return text if not tokenize else self.encode(text)

    def encode(self, text, **kwargs):
        return super().encode(text).ids

    def decode(self, ids, **kwargs):
        return super().decode(ids)


def _mac_owner(grammars, template=QWEN_CALLS):
    from tensorfold.server.request_options import RequestOptions

    owner = RequestOptions()
    owner.tokenizer, owner.tokenizer_lock = TemplateTokenizer(template), threading.Lock()
    owner.grammars, owner.model_dir, owner.stop_ids = grammars, None, (STOP,)
    return owner


def test_the_mac_refusal_checks_strict_tools_before_a_stream(grammars):
    from tensorfold.engine import grammar

    owner = _mac_owner(grammars)
    body = {"messages": [{"role": "user", "content": "Hi"}], "tools": TOOLS[:3]}
    assert grammar.refusal(body, owner) is None and grammar.refusal(body) is None
    assert "cannot be combined with strict" in grammar.refusal({**body, "response_format": {"type": "json_object"}},
                                                               owner)
    bad = {**body, "tools": [tool("f", {"type": "object", "properties": {"s": {"type": "string", "maxLength": 2}}})]}
    assert "do not support maxLength" in grammar.refusal(bad, owner)
    assert "Qwen XML tool calls" in grammar.refusal(body, _mac_owner(grammars, QWEN_CALLS.replace("'\\n</par", "'</par")))
    assert "does not enforce strict" in grammar.refusal(body, object())
    assert grammar.refusal({**body, "tools": [tool("f", strict=False)]}, object()) is None
    plan = owner.strict_calls(TOOLS[:3], required=True, parallel=False)
    assert plan.spec == st.Plan(TOOLS[:3], required=True, parallel=False).spec
    assert owner.strict_calls([tool("f", strict=False)], required=False, parallel=True) is None
    assert owner._strict_accepts(plan, CALL) and not owner._strict_accepts(plan, "text " + CALL)


def test_the_mac_policy_reads_strict_calls_with_their_plan_and_flush_shows_their_text():
    from tensorfold.server.stopping import StopPolicy
    from tensorfold.server.tool_policy import ToolCallPolicy
    from tensorfold.server.tools import parse_tool_calls_from_content

    plan = st.Plan(TOOLS, required=False, parallel=True)
    reply = {"content": "Let me look.\n\n" + CALL, "finish_reason": "stop", "strict_calls": plan}
    done = ToolCallPolicy({}).finish(reply, TOOLS, parse_tool_calls_from_content)
    assert done["finish_reason"] == "tool_calls" and done["content"] == "Let me look."
    assert json.loads(done["tool_calls"][0]["function"]["arguments"]) == ARGS
    lenient = ToolCallPolicy({}).finish({**reply, "strict_calls": None}, TOOLS, parse_tool_calls_from_content)
    assert json.loads(lenient["tool_calls"][0]["function"]["arguments"]) != ARGS    # a value holding </parameter>
    sent = []
    StopPolicy({}, None, None, frozenset()).flush(sent.append, "Hi" + CALL + " bye", None, "Hi", "",
                                                  lambda answer: plan.content(answer, finished=True))
    assert sent == [" bye"]
