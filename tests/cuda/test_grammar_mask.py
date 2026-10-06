"""Production CUDA grammar masks match independent CPU matcher bits without model weights."""

from __future__ import annotations

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

xgr = pytest.importorskip("xgrammar")

from tensorfold.engine import grammar  # noqa: E402

V, STOP, THINK_END = 128, 0, 127
SCHEMA = {
    "type": "object",
    "properties": {
        "k": {"type": "integer"},
        "tag": {"type": "string", "enum": ["a", "bb"]},
    },
    "required": ["k"],
    "additionalProperties": False,
}
DTYPES = (torch.float32, torch.bfloat16, torch.float16)


@pytest.fixture(scope="module")
def grammars():
    info = xgr.TokenizerInfo(
        [""] + [chr(token) for token in range(1, V)],
        xgr.VocabType.RAW,
        vocab_size=V,
        stop_token_ids=[STOP],
    )
    return grammar.Grammars(info)


@pytest.fixture(scope="module")
def compiled(grammars):
    specs = {
        "json_schema": grammar.Spec("json_schema", json.dumps(SCHEMA)),
        "regex": grammar.Spec("regex", "(OK|NO)"),
        "choice": grammar.Spec("choice", json.dumps(["OK", "NO"])),
    }
    return {name: grammars.compile(spec) for name, spec in specs.items()}


def _ids(text):
    return [ord(char) for char in text]


def _oracle(compiled, kind, prefix):
    """Fresh CPU matcher, decoded bit by bit independently of Constraint.allowed."""
    matcher = xgr.GrammarMatcher(compiled[kind])
    for token in _ids(prefix):
        assert matcher.accept_token(token), (kind, prefix, token)
    bits = xgr.allocate_token_bitmask(1, V)
    assert bits.device.type == "cpu"
    matcher.fill_next_token_bitmask(bits, 0)
    return {token for token in range(V) if (int(bits[0, token // 32]) >> (token % 32)) & 1}


def _assert_mask(compiled, constraint, window, paths, *, dtype, offset=0, width=V):
    rows = len(paths)
    original = (torch.arange(rows * width, dtype=torch.float32).view(rows, width) / 16).to(dtype)
    logits = original.to("cuda")
    result = constraint.mask(logits, window, offset=offset)
    assert result is logits and result.device.type == "cuda"
    got = result.cpu()
    expected = original.clone()
    for row, path in enumerate(paths):
        tokens = None if path is None else _oracle(compiled, *path)
        allowed = {column for column in range(width) if tokens is None or offset + column in tokens}
        for column in range(width):
            if column not in allowed:
                expected[row, column] = float("-inf")
        # Exact comparison also checks that every allowed logit retains its value.
        assert torch.equal(got[row], expected[row]), (row, dtype, offset, width, allowed)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset,width", [(0, V + 7), (64, 80)])
def test_schema_masks_partitioned_logits_and_pads_past_vocabulary(grammars, compiled, dtype, offset, width):
    constraint = grammars.constraint(compiled["json_schema"])
    constraint.advance(_ids('{"k":'))
    _assert_mask(
        compiled, constraint, None, [("json_schema", '{"k":')], dtype=dtype, offset=offset, width=width
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_branched_verify_window_prunes_drafts_and_masks_each_kept_path(grammars, compiled, dtype):
    constraint = grammars.constraint(compiled["json_schema"])
    constraint.advance(_ids('{"k":'))
    window = constraint.window(_ids(":1x2}y") + [STOP], [-1, 0, 0, 1, 1, 2, 4])
    assert window.tokens == _ids(":12}") and window.parents == [-1, 0, 1, 1]
    assert window.rows == [0, 1, 2, 3]
    paths = [
        ("json_schema", '{"k":'),
        ("json_schema", '{"k":1'),
        ("json_schema", '{"k":12'),
        ("json_schema", '{"k":1}'),
    ]
    _assert_mask(compiled, constraint, window, paths, dtype=dtype, width=V + 7)
    assert _oracle(compiled, *paths[-1]) == {STOP}


@pytest.mark.parametrize("dtype", DTYPES)
def test_thinking_window_masks_only_rows_after_think_end(grammars, compiled, dtype):
    constraint = grammars.constraint(compiled["json_schema"], think_end=THINK_END)
    window = constraint.window([ord("a"), THINK_END, ord("x"), ord("{")], [-1, 0, 1, 1])
    assert window.tokens == [ord("a"), THINK_END, ord("{")] and window.rows == [1, 2]
    _assert_mask(
        compiled, constraint, window, [None, ("json_schema", ""), ("json_schema", "{")], dtype=dtype, width=V + 7
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_advancing_past_think_end_activates_the_grammar(grammars, compiled, dtype):
    constraint = grammars.constraint(compiled["json_schema"], think_end=THINK_END)
    constraint.advance(_ids("hmm") + [THINK_END])
    assert constraint.active
    _assert_mask(compiled, constraint, None, [("json_schema", "")], dtype=dtype)


@pytest.mark.parametrize("kind", ["regex", "choice"])
@pytest.mark.parametrize("prefix", ["", "OK"])
def test_regex_and_choice_masks_follow_prefix_and_terminate(grammars, compiled, kind, prefix):
    constraint = grammars.constraint(compiled[kind])
    constraint.advance(_ids(prefix))
    _assert_mask(compiled, constraint, None, [(kind, prefix)], dtype=torch.float32)
    if prefix:
        assert _oracle(compiled, kind, prefix) == {STOP}
        constraint.advance([STOP])
        assert constraint.finished
