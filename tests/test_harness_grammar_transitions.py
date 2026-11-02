"""Actual xgrammar state boundaries, no model/CUDA work or replacement q laws."""
import re

import pytest

xgr = pytest.importorskip('xgrammar')
torch = pytest.importorskip('torch')
from tensorfold.engine import grammar  # noqa: E402

TOKENS = ['', '</think>', '<tool_call>', 'why', '```', '{', '}', '"x"', ':', '1',
          '</tool_call>', ' ', '\n', '"', '`', 'text']
END, OPEN, WHY, FENCE, LEFT, RIGHT, KEY, COLON, ONE, CLOSE, SPACE, NL = range(1, 13)


def decode(ids, **kwargs):
    return ''.join(TOKENS[int(t)] for t in ids)


@pytest.fixture
def grammars():
    return grammar.Grammars(xgr.TokenizerInfo(TOKENS, xgr.VocabType.RAW, vocab_size=len(TOKENS),
                                             stop_token_ids=[0]), decode=decode)


def tool_constraint(grammars):
    spec = grammar.Spec('regex', re.escape('<tool_call>{"x":1}</tool_call>'))
    return grammars.constraint(grammars.compile(spec), think_end=END, answer_start=OPEN,
                               answer_opener='<tool_call>', answer_headers=('<tool_call>{',), spec=spec)


def test_omitted_opener_real_close_activates_without_reinterpreting_answer_bytes(grammars):
    spec = grammar.Spec('regex', re.escape('{"x":1}'))
    c = grammars.constraint(grammars.compile(spec), think_end=END, spec=spec)
    c.advance([WHY])  # no thought opener was emitted
    preview = c.window([WHY, END, LEFT], [-1, 0, 1])
    assert preview.tokens == [WHY, END, LEFT] and preview.rows == [1, 2]
    assert not c.active and c.answer_offset is None and c.observed == 1
    c.advance([END])
    assert c.active and c.answer_offset == 2
    c.advance([LEFT, KEY, COLON, ONE, RIGHT, 0])
    assert c.finished and c.answer_offset == 2
    fresh = c.fresh()
    assert not fresh.active and fresh.answer_offset is None and fresh.observed == 0


def test_structural_tool_inside_open_thinking_includes_its_own_opener(grammars):
    c = tool_constraint(grammars)
    c.advance([WHY])
    preview = c.window([WHY, OPEN, LEFT], [-1, 0, 1])
    assert preview.tokens == [WHY, OPEN, LEFT] and preview.rows == [2]
    assert not c.active  # speculative preview restored matcher and committed phase
    c.advance([OPEN])
    assert not c.active and c.pending_answer_offset == 1
    c.advance([LEFT])
    assert c.active and c.answer_offset == 1
    c.advance([KEY, COLON, ONE, RIGHT, CLOSE, 0])
    assert c.finished


def test_quoted_tool_opener_does_not_activate_or_force_a_phantom_call(grammars):
    c = tool_constraint(grammars)
    c.advance([FENCE, NL])
    preview = c.window([NL, OPEN, WHY], [-1, 0, 1])
    assert preview.rows == [] and preview.tokens == [NL, OPEN, WHY]
    c.advance([OPEN, NL, FENCE, NL])
    assert not c.active and c.answer_offset is None
    c.advance([OPEN, LEFT])
    assert c.active and c.answer_offset == 6


def test_thought_only_native_eos_does_not_invent_a_close_or_answer(grammars):
    c = tool_constraint(grammars)
    c.advance([WHY, 0])
    assert not c.active and c.answer_offset is None and c.observed == 2


def test_activation_wire_preserves_real_decoder_and_legacy_wire(grammars):
    c = tool_constraint(grammars)
    encoded = grammar.pack(c)
    assert encoded[0] == -2
    restored = grammars.follow(encoded)
    restored.advance([WHY, OPEN, LEFT])
    assert restored.answer_start == OPEN and restored.answer_opener == '<tool_call>'
    assert restored.active and restored.answer_offset == 1
    legacy = grammars.constraint(c.compiled, think_end=END, spec=c.spec)
    old = grammar.pack(legacy)
    assert old[0] >= 0 and grammars.follow(old).answer_start is None


def test_actual_illegal_q_is_rejected_by_masked_p_and_replays_without_q_substitution(grammars):
    import numpy as np
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.rejection import ChainPacket, ProbabilityRow, proposal_row, verify

    spec = grammar.Spec('regex', re.escape('{"x":1}'))
    c = grammars.constraint(grammars.compile(spec), spec=spec)
    logits = torch.zeros(1, len(TOKENS), dtype=torch.float32, device="cpu")
    logits[0, WHY] = 100  # highest unconstrained score is illegal under actual p
    c.mask(logits)
    assert torch.isneginf(logits[0, WHY]) and torch.isfinite(logits[0, LEFT])
    sampling = Sampling(temperature=1.0, top_k=0, top_p=1.0, seed=7)
    p = proposal_row(logits[0].numpy(), np.arange(len(TOKENS)), sampling)
    actual_q = ProbabilityRow([WHY], [1.0])
    packet = ChainPacket((WHY,), (actual_q,), 1)
    results = [verify(packet, [p, None], sampling.seed, room=1, eos=(0,)) for _ in range(2)]
    assert results[0] == results[1]
    assert list(results[0][0]) == [0] and results[0][1] == LEFT
    assert packet.tokens == (WHY,) and packet.q_rows[0] is actual_q
    assert actual_q.ids.tolist() == [WHY] and actual_q.probs.tolist() == [1.0]


@pytest.mark.parametrize('header', [
    [-1], [-1, 0, 1, 1, 257], [-1, 0, 1, 0, 1, 65], [-1, 0, 1, 1, 1, -1],
    [-1, 0, 1, 1, 2, 0xFF, 0xFF],
])
def test_rank_activation_header_refuses_malformed_fields_before_compile(header):
    from tensorfold.cuda.rank_protocol import grammar as validate
    with pytest.raises((ValueError, UnicodeError)):
        validate(header, len(TOKENS))


def test_rank_activation_header_accepts_exact_pack_without_raising_transport_budget(grammars):
    from tensorfold.cuda.rank_protocol import grammar as validate, MAX_MESSAGE_BYTES
    assert MAX_MESSAGE_BYTES == 96 << 20
    encoded = grammar.pack(tool_constraint(grammars))
    assert validate(encoded, len(TOKENS)) is encoded


def test_quoted_native_close_is_data_but_explicit_budget_close_is_authoritative(grammars):
    spec = grammar.Spec('regex', re.escape('{"x":1}'))
    c = grammars.constraint(grammars.compile(spec), think_end=END, spec=spec)
    c.advance([FENCE, NL, END])
    assert not c.active and c.answer_offset is None and c.observed == 3
    c.force_think_close(END)  # real caller event, not a string inference inside unclosed code
    assert c.active and c.answer_offset == 4 and c.thinking_close_offset == 3
    c.advance([LEFT, KEY, COLON, ONE, RIGHT, 0])
    assert c.finished


def test_boundary_observer_is_independent_of_producer_matcher_copies(grammars):
    c = tool_constraint(grammars)
    observer = c.fresh_boundary()
    producer = c.fresh()
    producer.advance([WHY, OPEN, LEFT])
    assert c.observed == observer.observed == 0 and producer.observed == 3
    observer.advance([WHY, OPEN])
    assert observer.answer_offset is None and observer.pending_answer_offset == 1
    assert observer.observed == 2 and c.observed == 0
    observer.advance([LEFT])
    assert observer.answer_offset == producer.answer_offset == 1


def test_boundary_decoder_cannot_silently_erase_native_close():
    with pytest.raises(grammar.GrammarError, match='native close marker'):
        grammar.AnswerBoundary(len(TOKENS), think_end=END, decode=lambda *a, **k: '')
    with pytest.raises(grammar.GrammarError, match='callable'):
        grammar.AnswerBoundary(len(TOKENS), think_end=END, decode=False)
