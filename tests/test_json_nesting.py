"""Untrusted JSON nesting refusal: the 3.12 C container-depth contract, enforced by the guard."""
import json
import threading

import pytest

from tensorfold import json_nesting


@pytest.fixture
def guard(monkeypatch):
    # Exercise the guard's own logic with a small limit.
    monkeypatch.setattr(json_nesting, "NESTING_LIMIT", 40)
    return json_nesting


def outcome(call, text, **kw):
    try:
        return "value", call(text, **kw)
    except (json.JSONDecodeError, RecursionError) as error:
        return type(error).__name__, str(error)


def test_nesting_limit_is_the_312_compiled_depth():
    assert json_nesting.NESTING_LIMIT == 10000


@pytest.mark.parametrize("text", [
    "[" * 40 + "]" * 40, '{"a":' * 40 + "1" + "}" * 40,
    '["' + "[" * 80 + '"]', '["a\\"[[' + "[" * 80 + '"]', '"' + "{" * 90 + '"',
    "[" * 30 + "]" * 30 + "[" * 30 + "]" * 30,
    "[1 x" + "[" * 60, '["abc' + "[" * 60, "]" + "[" * 60, "[" * 20 + "]" * 30 + "[" * 60,
    '{"k": [1, 2.5e3, null, true]}', " [ ] ", "",
])
def test_guard_agrees_with_the_scanner_within_the_limit(guard, text):
    for raw in (text, text.encode(), text.encode("utf-16"), text.encode("utf-32-le")):
        assert outcome(guard.loads, raw) == outcome(json.loads, raw)


@pytest.mark.parametrize(("text", "kind"), [("[" * 41, "array"), ("[" * 40 + "{", "object"),
                                            ('{"a":' * 41 + "1", "object"), ("[" * 39 + "{" * 2 + "]" * 41, "object")])
def test_guard_refuses_like_the_compiled_counter(guard, text, kind):
    with pytest.raises(RecursionError) as refused:
        guard.loads(text.encode())
    assert str(refused.value) == f"maximum recursion depth exceeded while decoding a JSON {kind} from a unicode string"


def test_earlier_syntax_error_wins_over_deep_tail(guard):
    text = '[1, "x" 2,' + "[" * 100
    with pytest.raises(json.JSONDecodeError) as refused:
        guard.loads(text)
    with pytest.raises(json.JSONDecodeError) as expected:
        json.loads(text[:9])
    assert (refused.value.msg, refused.value.pos, refused.value.doc) == (expected.value.msg, 8, text)


def test_guard_keeps_decoder_hooks_and_boms(guard):
    def constant(_value):
        raise ValueError("nonfinite")
    assert guard.loads(b'{"a": [1.5]}', parse_float=str) == {"a": ["1.5"]}
    with pytest.raises(ValueError, match="nonfinite"):
        guard.loads("[NaN]", parse_constant=constant)
    for raw in (b"\xef\xbb\xbf[1]", b"\xef\xbb\xbf\xef\xbb\xbf[1]", "﻿[1]", bytearray(b"[2]")):
        assert outcome(guard.loads, raw) == outcome(json.loads, raw)
    with pytest.raises(TypeError, match="must be str, bytes or bytearray"):
        guard.loads(1)


def test_actual_runtime_accepts_the_limit_and_refuses_beyond_it():
    limit = json_nesting.NESTING_LIMIT
    assert json_nesting.loads(("[" * limit + "]" * limit).encode()) is not None
    with pytest.raises(RecursionError, match="decoding a JSON array"):
        json_nesting.loads(b"[" * (limit + 1) + b"]" * (limit + 1))


def test_stack_check_reads_the_default_thread_stack(monkeypatch):
    assert json_nesting.require_stack_capacity(1000) in (True, False)
    previous = threading.stack_size()
    try:
        threading.stack_size(1 << 20)
        assert json_nesting.require_stack_capacity(1000) is False
    finally:
        threading.stack_size(previous)


def test_walker_raises_the_encoders_own_int_digit_error_first():
    from tensorfold.server import host_payload

    assert host_payload.json_wire_bound([10 ** 4299, -(10 ** 4299)], maximum=1 << 20) > 8600
    for value in ([10 ** 4300], {"k": -(10 ** 4300)}):
        with pytest.raises(ValueError) as refused:
            host_payload.json_wire_bound(value, maximum=1 << 20)
        with pytest.raises(ValueError) as native:
            json.dumps(value)
        assert str(refused.value) == str(native.value).split("\n")[0]


def nesting(value):
    if isinstance(value, dict):
        return 1 + max(map(nesting, value.values()), default=0)
    if isinstance(value, list):
        return 1 + max(map(nesting, value), default=0)
    return 0


@pytest.mark.parametrize("seed", range(40))
def test_guard_matches_actual_nesting_on_strings_with_escapes(guard, seed):
    import random

    chooser = random.Random(seed)
    pieces = ['"', "\\", "[", "]", "{", "}", "a", "é", "\U0001F600", "\n"]

    def text():
        return "".join(chooser.choice(pieces) for _ in range(chooser.randrange(0, 40)))

    def build(level):
        if level == 0 or chooser.random() < 0.15:
            return chooser.choice([text(), 1, None, 2.5])
        # One deep child per container keeps the tree linear in its depth.
        siblings = [chooser.choice([text(), 0]) for _ in range(chooser.randrange(0, 3))]
        if chooser.random() < 0.5:
            return [*siblings, build(level - 1)]
        return {**{text() + str(index): item for index, item in enumerate(siblings)}, text(): build(level - 1)}

    value = build(chooser.randrange(30, 60))
    encoded = json.dumps(value, ensure_ascii=chooser.random() < 0.5)
    block = chooser.choice([1, 7, 64, 1 << 16])  # Exercise carried string/escape state.
    if nesting(value) > 40:
        with pytest.raises(RecursionError):
            guard.loads(encoded.encode())
        with pytest.raises(RecursionError):
            guard._vector_scan(encoded, 0, 0, {}, block=block)
    else:
        assert guard.loads(encoded.encode()) == value
        assert guard._vector_scan(encoded, 0, 0, {}, block=block) is None


def test_string_dense_input_switches_to_blocks_with_the_same_answers(guard):
    dense = "[" + ",".join(['"[{"'] * 9000) + "," + "[" * 45 + "1" + "]" * 45 + "]"
    with pytest.raises(RecursionError, match="JSON array"):
        guard.loads(dense)
    shallow = "[" + ",".join([r'"[{\\"'] * 9000) + "," + "[" * 30 + "1" + "]" * 30 + "]"
    assert guard.loads(shallow) == json.loads(shallow)
