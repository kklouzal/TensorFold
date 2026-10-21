"""Request-owned plan reuse against fresh mixed GDN/attention forwards."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from qwen_engine_checks import same_state
from test_qwen27_prefill import _model

from tensorfold.families.qwen3_5.cuda.decode import draft_decode
from tensorfold.families.qwen3_5.cuda.eager_staging import _RequestStaging
from tensorfold.families.qwen3_5.cuda.forward import State, commit, grow, tree_forward


def states(w, position, width):
    a, b = State(w), State(w)
    gen = torch.Generator(device="cuda").manual_seed(43)
    for st in (a, b):
        st.limit = 2048
        grow(st, 1, position + width + 16)
        st.pos, st.rope_delta = position, -3
    for side in range(2):
        source = torch.randn(a.kv[1][side].shape, generator=gen, device="cuda").bfloat16()
        a.kv[1][side].copy_(source)
        b.kv[1][side].copy_(source)
    return a, b


def equal_forward(reference, candidate):
    assert torch.equal(reference[0], candidate[0])
    for a, b in zip(reference[1], candidate[1]):
        assert type(a) is type(b)
        for name in vars(a):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert torch.equal(reference[2], candidate[2])


@pytest.mark.parametrize("width,position", [(1, 0), (1, 511), (7, 504), (8, 0), (8, 504),
                                           (8, 505), (8, 511), (8, 512), (16, 1016), (128, 384)])
def test_reuse_boundaries_records_taps_and_commit_match_fresh(width, position):
    w = _model()
    w.bind_draft_taps((0, 1))
    reference, candidate = states(w, position, width)
    parents = list(range(-1, width - 1))
    with _RequestStaging(w) as owner:
        previous = None
        for turn in range(3):
            ids = (torch.arange(width, device="cuda", dtype=torch.int32) + 7 + turn) % w.config.vocab
            expected = tree_forward(w, ids, parents, reference, capture_taps=True)
            actual = tree_forward(w, ids, parents, candidate, capture_taps=True, _request_staging=owner)
            torch.cuda.synchronize()
            owner.completed()
            equal_forward(expected, actual)
            entry = owner.entry
            boundary = (candidate.pos // 512, -(-(candidate.pos + width) // 512))
            if previous is not None:
                assert (entry is previous) == (boundary == previous.boundary)
            stream = entry.aplan.streams.cpu().tolist()
            assert stream == [[0, width, candidate.pos, boundary[1]]]
            for row in range(width):
                assert entry.aplan.paths[row, :row + 1].cpu().tolist() == list(range(row + 1))
            commit(reference, expected[1], [0])
            commit(candidate, actual[1], [0])
            torch.cuda.synchronize()
            same_state(reference, candidate)
            previous = entry


def test_arbitrary_tree_view_replacement_and_nondefault_stream_keep_fresh_semantics():
    w = _model()
    w.bind_draft_taps((0, 1))
    reference, candidate = states(w, 31, 8)
    chain, branch = list(range(-1, 7)), [-1, 0, 0, 1, 1, 2, 2, 3]
    ids = torch.arange(7, 15, dtype=torch.int32, device="cuda")
    with _RequestStaging(w) as owner:
        for turn, parents in enumerate((chain, chain, branch, chain, chain)):
            if turn == 1:
                candidate.kv[1] = tuple(t.clone() for t in candidate.kv[1])
            expected = tree_forward(w, ids, parents, reference, capture_taps=True)
            actual = tree_forward(w, ids, parents, candidate, capture_taps=True, _request_staging=owner)
            torch.cuda.synchronize()
            owner.completed()
            equal_forward(expected, actual)
            assert (owner.entry is None) == (parents is branch)
        other = torch.cuda.Stream()
        with torch.cuda.stream(other):
            actual = tree_forward(w, ids, chain, candidate, capture_taps=True, _request_staging=owner)
            other.synchronize()
            owner.completed()
        equal_forward(expected, actual)
        assert owner.entry is None


def test_complete_serial_request_ids_and_state_match_ordinary_decoder():
    w = _model()
    reference, candidate = states(w, 16, 1)
    ordinary = draft_decode(w, reference, [7] * 16, 8, 16, None, None,
                            allow_copy=False, stop_eos=False, inplace=True)
    with _RequestStaging(w) as owner:
        reused = draft_decode(w, candidate, [7] * 16, 8, 16, None, None,
                              allow_copy=False, stop_eos=False, inplace=True, _staging=owner)
    assert reused.tokens == ordinary.tokens
    assert reused.widths == ordinary.widths and reused.rounds == ordinary.rounds
    same_state(reference, candidate)
