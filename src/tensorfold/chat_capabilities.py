"""Observed model-owned rendering capabilities, independent of model names/hashes.

Discovery uses the actual public renderer on both request routes at startup.
Stable rendered differences establish a rendering contract, not model obedience.
No weights, engine lane, tokenizer work or generation participates in discovery.
"""
from jinja2 import TemplateError

from tensorfold.server.errors import RequestError

_OLDER = ("tf_probe_older_reasoning_a93f", "tf_probe_older_reasoning_b75d")
_CURRENT = ("tf_probe_current_reasoning_a63c", "tf_probe_current_reasoning_b81e")


def _stable(render, messages, tools, thinking):
    try:
        first = render(messages, tools=tools, enable_thinking=thinking)
        second = render(messages, tools=tools, enable_thinking=thinking)
    except (TemplateError, RequestError):
        return None, "unsupported"
    if not isinstance(first, str) or first != second:
        return None, "unstable"
    return first, "stable"


def _history(variant):
    messages = [
        {"role": "user", "content": "An earlier question."},
        {"role": "assistant", "content": "An earlier answer."},
        {"role": "user", "content": "Use the probe tool for this question."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_tf_probe", "type": "function",
         "function": {"name": "tf_capability_probe", "arguments": {}}}]},
        {"role": "tool", "content": "The probe tool result.", "name": "tf_capability_probe",
         "tool_call_id": "call_tf_probe"},
    ]
    if variant is not None:
        messages[1]["reasoning_content"] = _OLDER[variant]
        messages[3]["reasoning_content"] = _CURRENT[variant]
    return messages


def discover_chat_capabilities(render, *, fixed=(None, None)):
    """Return common verified facts plus bounded, synthetic-probe diagnostics.

    Harness discovers without tools and applies the answer to tool requests.
    Therefore a shared fact must agree across BOTH captured render routes.
    Missing/unstable/unsupported evidence omits that fact, never guesses it.
    """
    if len(fixed) != 2 or any(value is not None and type(value) is not bool for value in fixed):
        raise ValueError("both template routes require a boolean or unknown fixed-thinking fact")
    tools_spec = [{"type": "function", "function": {"name": "tf_capability_probe",
                   "description": "A deterministic capability-discovery probe.",
                   "parameters": {"type": "object", "properties": {}, "required": [],
                                  "additionalProperties": False}, "strict": False}}]
    routes = []
    for tools, forced in zip((None, tools_spec), fixed):
        on, on_state = _stable(render, [{"role": "user", "content": "A capability probe."}], tools, True)
        off, off_state = _stable(render, [{"role": "user", "content": "A capability probe."}], tools, False)
        switch = forced is None and on is not None and off is not None and on != off
        samples = [_stable(render, _history(variant), tools, True) for variant in (0, 1, None)]
        first, second, absent = [item[0] for item in samples]
        replay = (all(value is not None for value in (first, second, absent))
                  and _CURRENT[0] in first and _CURRENT[1] in second
                  and _CURRENT[1] not in first and _CURRENT[0] not in second
                  and all(marker not in absent for marker in (*_OLDER, *_CURRENT)))
        preserve = None
        if replay:
            older = (_OLDER[0] in first, _OLDER[1] in second)
            if older[0] == older[1]:
                preserve = older[0]
        routes.append({"thinking_control": bool(switch), "fixed_thinking": forced,
                       "switch_samples": [on_state, off_state], "replay_samples": [item[1] for item in samples],
                       "replay_field_verified": bool(replay), "preserve_reasoning": preserve})
    capabilities = {}
    if len(routes) == 2 and all(route["thinking_control"] for route in routes):
        capabilities["thinking_control"] = {"keyword": "enable_thinking", "on_value": True,
                                            "off_value": False, "top_level": False}
    if (len(routes) == 2 and all(route["replay_field_verified"] for route in routes)
            and routes[0]["preserve_reasoning"] is not None
            and routes[0]["preserve_reasoning"] is routes[1]["preserve_reasoning"]):
        capabilities["reasoning_replay_field"] = "reasoning_content"
        capabilities["preserve_reasoning"] = routes[0]["preserve_reasoning"]
    return capabilities, {"method": "stable_model_render_probes_v1", "routes": routes}
