"""Instructions keep their place in the conversation, so a resumed prompt's earlier tokens never change."""

import jinja2
import pytest

from tensorfold.server.messages import late_system_role, normalize_messages

# Qwen3.8's rule: one system message, and only first
_FIRST_ONLY = jinja2.Environment().from_string(
    "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('late system') }}{% endif %}"
    "<{{ m.role }}>{{ m.content }}{% endfor %}")
# Nemotron's rule: a system message renders wherever it is
_ANYWHERE = jinja2.Environment().from_string("{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}")


def render(template, messages):
    def fail(message):
        raise jinja2.TemplateError(message)
    return template.render(messages=messages, raise_exception=fail)


def test_the_template_decides_how_a_later_instruction_renders():
    assert late_system_role(lambda m: render(_FIRST_ONLY, m)) == "user"
    assert late_system_role(lambda m: render(_ANYWHERE, m)) == "system"
    assert late_system_role(lambda m: "a template that drops it") == "user"


@pytest.mark.parametrize("template", [_FIRST_ONLY, _ANYWHERE])
def test_a_later_instruction_leaves_the_earlier_conversation_unchanged(template):
    role = late_system_role(lambda m: render(template, m))
    history = [{"role": "developer", "content": "be terse"}, {"role": "system", "content": "use tools"},
               {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    later = history + [{"role": "developer", "content": "cwd changed"}, {"role": "user", "content": "go on"}]
    before = render(template, normalize_messages(history, late_system=role))
    after = render(template, normalize_messages(later, late_system=role))
    assert after.startswith(before)
    assert before.startswith("<system>be terse\n\nuse tools<user>hi")
    assert f"<{role}>cwd changed<user>go on" in after


def test_normalizing_twice_changes_nothing():
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"},
                {"role": "system", "content": "c"}]
    once = normalize_messages(messages, late_system="user")
    assert normalize_messages(once, late_system="user") == once
    assert normalize_messages(normalize_messages(messages), late_system="user") == once


_GATHER_SYSTEMS = jinja2.Environment().from_string(
    "{% for m in messages if m.role == 'system' %}<system>{{ m.content }}{% endfor %}"
    "{% for m in messages if m.role != 'system' %}<{{ m.role }}>{{ m.content }}{% endfor %}")


def test_hoisted_late_system_is_rejected_even_though_marker_is_present():
    seen = []
    def gather(messages):
        seen.extend(messages)
        return render(_GATHER_SYSTEMS, messages)
    assert late_system_role(gather) == 'user'
    rendered = render(_GATHER_SYSTEMS, seen)
    assert seen[3]['content'] in rendered
    assert rendered.index(seen[3]['content']) < rendered.index(seen[1]['content'])
    history = [{'role': 'system', 'content': 'opening'}, {'role': 'user', 'content': 'before'},
               {'role': 'assistant', 'content': 'answer'}]
    later = history + [{'role': 'system', 'content': 'late'}, {'role': 'user', 'content': 'after'}]
    assert render(_GATHER_SYSTEMS, normalize_messages(later, late_system='user')).startswith(
        render(_GATHER_SYSTEMS, normalize_messages(history, late_system='user')))


def test_probe_requires_all_unique_surrounding_turns_in_relative_order():
    assert late_system_role(lambda messages: ''.join(m['content'] for m in messages)) == 'system'
    assert late_system_role(lambda messages: ''.join(m['content'] for m in reversed(messages))) == 'user'
    assert late_system_role(lambda messages: messages[3]['content']) == 'user'
    assert late_system_role(lambda messages: ''.join(m['content'] * 2 for m in messages)) == 'user'


@pytest.mark.parametrize('primary', [MemoryError('probe allocation'), ValueError('unexpected template setup defect'),
                                   RuntimeError('unexpected renderer defect')])
def test_probe_does_not_persist_fallback_policy_after_resource_or_unexpected_failure(primary):
    def fail(messages):
        raise primary
    with pytest.raises(type(primary)) as caught:
        late_system_role(fail)
    assert caught.value is primary


def test_probe_only_handles_declared_template_refusal_not_opaque_nontext_result():
    def unsupported(messages):
        raise jinja2.TemplateError('late system history unsupported')
    assert late_system_role(unsupported) == 'user'
    with pytest.raises(TypeError, match='rendered text'):
        late_system_role(lambda messages: {'not': 'rendered text'})
