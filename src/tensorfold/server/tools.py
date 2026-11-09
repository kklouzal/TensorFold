"""Normalize tool specs and choices, then parse model tool calls into complete or streamed OpenAI responses."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tensorfold.tool_parameters import decode_parameter, parameter_schemas
from tensorfold.server.gemma_values import GEMMA_OPEN, gemma_json
from tensorfold.markup_lex import markup_regions, xml_function_frames, xml_parameter_fields

def tool_spec_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(tool.get("name") or "").strip() if isinstance(tool, dict) else ""


def normalize_tool_specs(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    normalized: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{index}] must be an object")
        function = tool.get("function")
        definition = function if isinstance(function, dict) else tool
        name = definition.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tools[{index}] must include a function name")
        normalized.append(tool)
    return normalized


def tool_choice_disables_tools(tool_choice: Any) -> bool:
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "none"
    if isinstance(tool_choice, dict):
        value = tool_choice.get("type") or tool_choice.get("mode")
        return isinstance(value, str) and value.strip().lower() == "none"
    return False


def tool_choice_requires_call(tool_choice: Any) -> bool:
    """OpenAI's "required" or a named function: the reply must call a tool."""

    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "required"
    if isinstance(tool_choice, dict):
        value = str(tool_choice.get("type") or tool_choice.get("mode") or "").strip().lower()
        return value == "required" or (value == "function" and isinstance(tool_choice.get("function"), dict))
    return False


def validate_tool_choice(tools: list[dict[str, Any]], tool_choice: Any) -> None:
    if not isinstance(tool_choice, dict):
        return
    if str(tool_choice.get("type") or "").lower() != "function":
        return
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        raise ValueError("tool_choice function must include a function object")
    requested = str(function.get("name") or "").strip()
    if not requested:
        raise ValueError("tool_choice function must include a name")
    known = {tool_spec_name(tool) for tool in tools}
    if requested not in known:
        raise ValueError(f"tool_choice requested unknown tool '{requested}'")


def active_tool_specs(tools: Any, tool_choice: Any) -> list[dict[str, Any]]:
    """The tools the template offers: none for "none", only the named one for a named function."""

    specs = normalize_tool_specs(tools)
    if tool_choice_disables_tools(tool_choice):
        return []
    if not specs:
        if tool_choice_requires_call(tool_choice):
            raise ValueError("tool_choice requires a tool call, but the request offers no tools")
        return []
    validate_tool_choice(specs, tool_choice)
    if isinstance(tool_choice, dict) and str(tool_choice.get("type") or "").lower() == "function":
        named = str(tool_choice["function"].get("name") or "").strip()
        return [tool for tool in specs if tool_spec_name(tool) == named]
    return specs


_TOOL_FUNCTION_BLOCK_RE = re.compile(
    r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def normalize_qwen_tool_markup(text: str, *, finished: bool = True) -> str:
    """Repair framing slips only, outside code/quoted/native argument data.

    Repeated envelopes are one frame; distinct complete functions remain
    distinct calls. Argument bytes are never repaired or interpreted here.
    """
    parts, at = [], 0
    for region in markup_regions(text, finished=finished):
        if region.kind != 'call' or region.opener.lower() != '<tool_call>':
            continue
        parts.append(text[at:region.start])
        body = text[region.payload_start:region.payload_end]
        functions = xml_function_frames(body, finished=finished)
        if functions and any(xml_parameter_fields(_TOOL_FUNCTION_BLOCK_RE.fullmatch(fn).group(2), finished=finished)[1].strip()
                             for fn in functions):
            functions = None
        repeated = (region.payload_start != region.start + len(region.opener)
                    or region.complete and region.end != region.payload_end + len(region.closer))
        if functions and (len(functions) > 1 or not region.complete) and (region.complete or finished):
            parts.append('\n'.join('<tool_call>\n' + fn.strip() + '\n</tool_call>' for fn in functions))
        elif repeated:
            parts.append('<tool_call>' + body + ('</tool_call>' if region.complete else ''))
        else:
            parts.append(text[region.start:region.end])
        at = region.end
    parts.append(text[at:])
    return ''.join(parts)


# Gemma 4: <|tool_call>call:NAME{key:value,...}<tool_call|>, and the 26B's bare :NAME{...}. Keys bare, strings between <|"|>.
_GEMMA_CALL_RE = re.compile(r"^(?:call)?:([\w.-]+)\s*(\{.*\})$", re.DOTALL)
# DeepSeek-V4's DSML: one <｜DSML｜tool_calls> block holds invokes of named parameters, string="false" ones as JSON
_DSML_BLOCK_RE = re.compile(r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>", re.DOTALL)
_DSML_INVOKE_RE = re.compile(r'<｜DSML｜invoke name="([^"]*)">(.*?)</｜DSML｜invoke>', re.DOTALL)
_DSML_PARAM_RE = re.compile(r'<｜DSML｜parameter name="([^"]*)" string="(true|false)">(.*?)</｜DSML｜parameter>', re.DOTALL)
_MISSING = object()


def _tool_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        document = value.strip()
        if not document:
            return {}
        arguments = json.loads(document)
        if isinstance(arguments, dict):
            return arguments
    raise ValueError("tool_call arguments must be a JSON object")


def _loose_tool_arguments(payload: dict[str, Any], explicit: Any) -> Any:
    if explicit is not _MISSING:
        return explicit
    return {
        key: value
        for key, value in payload.items()
        if key not in {"name", "tool", "function", "call", "type"}
    }


# GLM-4.5 and later (GLM-5.3-Flash): <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>
_GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
_GLM_NAME_RE = re.compile(r"[\w.:-]+")


def parse_glm_tool_call_block(block: str, tools: Any, *, complete: bool = False) -> tuple[str, dict[str, Any]] | None:
    """GLM's call in a ``<tool_call>`` block, or None, values typed by the offered tool's schema (both servers)."""

    return _parse_glm_payload(block, parameter_schemas(tools), complete=complete)


def _parse_glm_payload(block: str, schemas: dict[str, dict[str, Any]] | None, *,
                       complete: bool = False) -> tuple[str, dict[str, Any]] | None:
    """A GLM call (the bare name without arguments), values decoded as Qwen's so a history renders as written."""

    name, found, rest = block.partition("<arg_key>")
    name = name.strip()
    if _GLM_NAME_RE.fullmatch(name) is None:
        return None
    rest = found + rest
    if complete and _GLM_ARG_RE.sub("", rest).strip():
        return None
    schema = (schemas or {}).get(name.lower(), {})
    return name, {key.strip(): decode_parameter(value, schema.get(key.strip(), {}), python=False)
                  for key, value in _GLM_ARG_RE.findall(rest)}


def _parse_gemma_call(block: str) -> tuple[str, dict[str, Any]] | None:
    """Gemma's calls, with native strings/keys/null decoded without interpreting text inside strings."""

    match = _GEMMA_CALL_RE.match(block)
    if match is None:
        return None
    return match.group(1), _tool_json_object(gemma_json(match.group(2)))




def _parse_tool_call_payload(block: str, schemas: dict[str, dict[str, Any]] | None = None, *, complete: bool = False) -> tuple[str, dict[str, Any]] | None:
    gemma = _parse_gemma_call(block) if block.startswith(("call:", ":")) else None
    if gemma is not None:
        return gemma
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        for item in payload:
            parsed = _parse_tool_call_payload(json.dumps(item, ensure_ascii=False), complete=complete)
            if parsed is not None:
                return parsed
        return None
    if isinstance(payload, dict):
        function = payload.get("function")
        if isinstance(function, dict):
            name = function.get("name") or function.get("tool") or function.get("function")
            explicit_arguments = function.get(
                "arguments",
                function.get("args", function.get("parameters", _MISSING)),
            )
            arguments = _loose_tool_arguments(function, explicit_arguments)
        else:
            name = (
                payload.get("name")
                or payload.get("tool")
                or payload.get("function")
                or payload.get("call")
            )
            explicit_arguments = payload.get(
                "arguments",
                payload.get("args", payload.get("parameters", _MISSING)),
            )
            arguments = _loose_tool_arguments(payload, explicit_arguments)
        name_text = str(name or "").strip()
        if not name_text:
            raise ValueError("tool_call is missing a function name")
        arguments = _tool_json_object(arguments)
        json.dumps(arguments, allow_nan=False)
        return name_text, arguments

    match = _TOOL_FUNCTION_BLOCK_RE.match(block)
    if match is None:
        if payload is None and not block.lstrip().startswith(("{", "[", "<")):
            return _parse_glm_payload(block, schemas, complete=complete)
        return None
    name = match.group(1).strip()
    arguments: dict[str, Any] = {}
    body = match.group(2)
    fields, residue = xml_parameter_fields(body)
    if complete and residue.strip():
        return None
    for key, value in fields:
        schema = (schemas or {}).get(name.lower(), {}).get(key, {})
        arguments[key] = decode_parameter(value, schema)
    if not name:
        raise ValueError("tool_call is missing a function name")
    return name, arguments


def _parse_dsml_calls(block: str) -> list[tuple[str, dict[str, Any]]] | None:
    """Every invoke of a DSML block as (name, arguments), or None when anything in it is not a well-formed invoke."""

    calls, at = [], 0
    for invoke in _DSML_INVOKE_RE.finditer(block):
        if block[at:invoke.start()].strip():
            return None
        at = invoke.end()
        arguments: dict[str, Any] = {}
        body = invoke.group(2)
        if _DSML_PARAM_RE.sub("", body).strip():
            return None
        for name, string, value in _DSML_PARAM_RE.findall(body):
            if string == "true":
                arguments[name] = value
                continue
            try:
                arguments[name] = json.loads(value)
            except json.JSONDecodeError:
                return None
        calls.append((invoke.group(1).strip(), arguments))
    return calls if calls and not block[at:].strip() else None




def _openai_tool_call(raw_name: str, arguments: dict[str, Any], known: dict[str, str]) -> dict[str, Any]:
    name = known.get(raw_name.lower())
    if name is None:
        raise ValueError(f"unknown tool '{raw_name}'")
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _parse_bare_json_tool_calls(text: str, known: dict[str, str], *, max_calls: int | None = None) -> list[dict[str, Any]] | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    payloads = payload if isinstance(payload, list) else [payload]
    calls: list[dict[str, Any]] = []
    for item in payloads:
        if max_calls is not None and len(calls) >= max_calls:
            break
        if not isinstance(item, dict):
            return None
        try:
            parsed = _parse_tool_call_payload(json.dumps(item, ensure_ascii=False), complete=max_calls is not None)
        except (ValueError, TypeError):
            if max_calls is None:
                return None           # JSON that is not a call (a structured answer) is the reply's content
            continue
        if parsed is None:
            return None
        raw_name, arguments = parsed
        if raw_name.lower() not in known:
            return None
        calls.append(_openai_tool_call(raw_name, arguments, known))
    return calls or None


def _envelopes(text: str, *, finished=True) -> list[tuple[int, int, str]]:
    """Complete actual lexical frames; code, strings and nested data stay opaque."""
    return [(region.start, region.end,
             text[region.start:region.end] if region.opener == '<｜DSML｜tool_calls>'
             else text[region.payload_start:region.payload_end].strip())
            for region in markup_regions(text, finished=finished) if region.kind == 'call' and region.complete]


def tool_calls_complete(text: str, tools, *, finished: bool = True) -> bool:
    """No offered named frame remains open at this actual parse boundary.

    Partial names/bare openers and quoted examples are not call intent. This
    reports lifecycle only; argument validity remains with the actual decoder.
    """
    if finished:
        text = normalize_qwen_tool_markup(text, finished=True)
    known = {tool_spec_name(tool).lower() for tool in tools or []}
    for region in markup_regions(text, finished=finished):
        if region.kind != 'call' or region.complete:
            continue
        body = text[region.payload_start:region.payload_end].lstrip()
        if region.opener == GEMMA_OPEN:
            match = re.match(r'^(?:call)?:([\w.-]+)\s*\{', body)
            if match is not None and match[1].lower() in known:
                return False
        elif region.opener == '<｜DSML｜tool_calls>':
            match = re.match(r'^<｜DSML｜invoke name="([^"]+)">', body)
            if match is not None and match[1].lower() in known:
                return False
        elif any(body.lower().startswith('<function=' + name + '>') for name in known):
            return False
    return True


def parse_tool_calls_from_content(
    text: str,
    tools: list[dict[str, Any]],
    *, max_calls: int | None = None, finished: bool = True,
) -> tuple[str, list[dict[str, Any]] | None]:
    if not tools:
        return text, None
    known = {tool_spec_name(tool).lower(): tool_spec_name(tool) for tool in tools}
    schemas = parameter_schemas(tools)
    text = normalize_qwen_tool_markup(text, finished=finished)
    envelopes = _envelopes(text, finished=finished)
    if not envelopes:
        bare_calls = _parse_bare_json_tool_calls(text, known, max_calls=max_calls)
        if bare_calls is not None:
            return "", bare_calls
        return text, None
    calls: list[dict[str, Any]] = []
    residue_parts: list[str] = []
    cursor = 0
    for start, end, block in envelopes:
        residue_parts.append(text[cursor:start])
        cursor = end
        if max_calls is not None and len(calls) >= max_calls:
            continue
        try:
            if block.startswith("<｜DSML｜tool_calls>"):
                parsed = _parse_dsml_calls(_DSML_BLOCK_RE.fullmatch(block).group(1))
            else:
                one = _parse_tool_call_payload(block, schemas, complete=max_calls is not None)
                parsed = None if one is None else [one]
        except (ValueError, TypeError):
            parsed = None
        if not parsed or any(name.lower() not in known for name, _ in parsed):
            # A malformed or unoffered call stays text: the reply is content, never an error or a client retry loop.
            if max_calls is None:
                residue_parts.append(text[start:end])
            continue
        for one in parsed:
            if max_calls is None or len(calls) < max_calls:
                calls.append(_openai_tool_call(*one, known))
    residue_parts.append(text[cursor:])
    content = "".join(residue_parts).strip()
    return content, calls or None


def stream_tool_call_deltas(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict):
            continue
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": str(tool_call.get("id") or f"call_{index}"),
                        "type": str(tool_call.get("type") or "function"),
                        "function": {
                            "name": str(function.get("name") or ""),
                            "arguments": "",
                        },
                    }
                ]
            }
        )
        arguments = str(function.get("arguments") or "")
        if arguments:
            deltas.append({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]})
    return deltas
