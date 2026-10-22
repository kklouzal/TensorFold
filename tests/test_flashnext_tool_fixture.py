"""The required Flash Next toy streams need room for their longest first call."""

import json

import pytest

from tests.cuda.toy_grammar import TOOL_ALPHA, TOOL_FIRST_CALL_TOKENS, tool_spell, toy_tools


def test_toy_non_stop_tokens_consume_at_least_one_ascii_character():
    words = [tool_spell(token) for token in range(1, 4096)]
    assert all(word.isascii() and 1 <= len(word) <= 3 for word in words)


@pytest.mark.parametrize("parallel", [False, True])
def test_required_first_call_fits_budget_and_shorter_prefix_is_incomplete(parallel):
    pytest.importorskip("xgrammar")
    grammars, plan, compiled = toy_tools(4096, required=True, parallel=parallel)
    blanks = " " * 32
    values = (("n", "99"), ("u", "ff"),
              ("o", "{" + blanks + '"k"' + blanks + ":" + blanks + "false" + blanks + "}"),
              ("z", "false"))
    call = (blanks + "<tool_call>\n<function=get>\n"
            + "".join(f"<parameter={name}>\n{value}\n</parameter>\n" for name, value in values)
            + "</function>\n</tool_call>")
    single = {tool_spell(token): token for token in range(1, len(TOOL_ALPHA))}
    tokens = [single[character] for character in call]
    assert len(tokens) == TOOL_FIRST_CALL_TOKENS == 343
    matcher = grammars.xgr.GrammarMatcher(compiled)
    assert all(matcher.accept_token(token) for token in tokens[:-1])
    assert plan.parse(call[:96])[1] is None       # the old budget can end inside a conforming call
    assert plan.parse(call[:-1])[1] is None
    assert matcher.accept_token(tokens[-1])
    _, calls = plan.parse(call)
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "get"
    assert json.loads(calls[0]["function"]["arguments"]) == {"n": 99, "u": "ff", "o": {"k": False}, "z": False}
