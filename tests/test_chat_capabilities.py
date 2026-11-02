"""Actual Jinja render contracts: shared control/replay facts, not model obedience."""
import pytest
from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from tensorfold.chat_capabilities import discover_chat_capabilities

SOURCE = ("{% if enable_thinking %}THINK{% else %}DIRECT{% endif %}"
          "{% set ns = namespace(last=-1) %}{% for m in messages %}"
          "{% if m.role == 'user' %}{% set ns.last = loop.index0 %}{% endif %}{% endfor %}"
          "{% for m in messages %}{{ m.content }}"
          "{% if m.role == 'assistant' and m.reasoning_content is defined "
          "and (PRESERVE or loop.index0 > ns.last) %}{{ m.reasoning_content }}{% endif %}{% endfor %}")


def renderer(default, tools=None):
    env = ImmutableSandboxedEnvironment()
    compiled = tuple(env.from_string(value) for value in (default, default if tools is None else tools))
    def render(messages, *, tools, enable_thinking):
        return compiled[int(bool(tools))].render(messages=messages, tools=tools, enable_thinking=enable_thinking)
    return render


@pytest.mark.parametrize('preserve', [False, True])
def test_both_real_render_routes_discover_switch_and_older_vs_current_tool_reasoning(preserve):
    source = SOURCE.replace('PRESERVE', 'true' if preserve else 'false')
    caps, evidence = discover_chat_capabilities(renderer(source, 'tool route:' + source))
    assert caps == {'thinking_control': {'keyword': 'enable_thinking', 'on_value': True, 'off_value': False,
                                        'top_level': False},
                    'reasoning_replay_field': 'reasoning_content', 'preserve_reasoning': preserve}
    assert all(route['replay_field_verified'] for route in evidence['routes'])


def test_mismatched_route_controls_and_replay_are_never_shared():
    default = SOURCE.replace('PRESERVE', 'true')
    tools = SOURCE.replace('PRESERVE', 'false').replace(
        '{% if enable_thinking %}THINK{% else %}DIRECT{% endif %}', 'fixed')
    caps, _ = discover_chat_capabilities(renderer(default, tools))
    assert caps == {}  # default alone would falsely advertise both capabilities


def test_unconditional_fixed_switch_is_not_advertised_even_if_other_output_varies():
    source = SOURCE.replace('PRESERVE', 'false')
    caps, _ = discover_chat_capabilities(renderer(source), fixed=(None, True))
    assert 'thinking_control' not in caps
    assert caps['reasoning_replay_field'] == 'reasoning_content'


def test_ignored_reasoning_field_is_not_replay_support():
    source = '{% if enable_thinking %}THINK{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}'
    caps, _ = discover_chat_capabilities(renderer(source))
    assert 'thinking_control' in caps and 'reasoning_replay_field' not in caps and 'preserve_reasoning' not in caps


def test_unstable_and_known_unsupported_renderers_omit_facts_but_resource_failure_propagates():
    sequence = iter(str(index) for index in range(32))
    caps, evidence = discover_chat_capabilities(lambda *a, **k: next(sequence))
    assert caps == {} and evidence['routes'][0]['switch_samples'] == ['unstable', 'unstable']
    def unsupported(*args, **kwargs):
        raise TemplateError('this role history is unsupported')
    assert discover_chat_capabilities(unsupported)[0] == {}
    primary = MemoryError('render allocation')
    def exhausted(*args, **kwargs):
        raise primary
    with pytest.raises(MemoryError) as caught:
        discover_chat_capabilities(exhausted)
    assert caught.value is primary


def test_fixed_route_shape_does_not_coerce_absent_false_and_invalid_values():
    with pytest.raises(ValueError):
        discover_chat_capabilities(renderer('text'), fixed=(None,))
    with pytest.raises(ValueError):
        discover_chat_capabilities(renderer('text'), fixed=(None, 'false'))
