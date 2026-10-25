"""Logical reply bounds and warm stock-MTP state, independent of GPU numerics."""

import json
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry, checkpoint_method


@pytest.mark.parametrize("depth,rows", [(0, 1), (1, 2), (2, 3), (3, 4), (15, 16)])
def test_stock_mtp_allocated_geometry_does_not_disappear_at_zero_proposals(depth, rows):
    method = MethodGeometry("mtp", depth)
    assert method.target_rows(depth) == rows
    assert method.target_tail_rows == rows
    assert method.stock_mtp_cache_layers == 1
    assert method.target_rows(0) == 1
    assert method.receipt()["zero_proposal_maintains_conditioning"] is True


@pytest.mark.parametrize("field,limit", [("reply_tokens", 3), ("context_tokens", 4), ("target_rows", 2)])
def test_each_bound_includes_the_anchor_verification_row(field, limit):
    bounds = dict(reply_tokens=100, context_tokens=100, target_rows=100)
    bounds[field] = limit
    assert MethodGeometry("mtp", 15).cap_proposals(15, **bounds) == limit - 1


@pytest.mark.parametrize("bad", [-1, True, 1.0, None])
def test_geometry_rejects_invalid_counts(bad):
    with pytest.raises(ValueError, match="integer"):
        MethodGeometry("mtp", bad)
    with pytest.raises(ValueError, match="integer"):
        MethodGeometry("mtp", 3).cap_proposals(3, reply_tokens=bad, context_tokens=9, target_rows=4)


def test_none_does_not_claim_a_draft_cache_and_dspark_geometry_is_not_mtp():
    none = MethodGeometry("none", 0)
    assert none.stock_mtp_cache_layers == none.draft_tail_rows == 0
    assert none.target_tail_rows == 1
    assert MethodGeometry("dspark", 7).draft_tail_rows is None
    with pytest.raises(ValueError):
        MethodGeometry("none", 1)
    with pytest.raises(ValueError):
        none.target_rows(1)


def test_unknown_dflash_checkpoint_cannot_fall_through_to_stock_mtp(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DFlashDraftModel"]}))
    with pytest.raises(ValueError, match="pairing profile"):
        checkpoint_method(tmp_path)


def test_selected_dflash2_architecture_has_its_own_pairing_preflight(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DFlash2DraftModel"]}))
    assert checkpoint_method(tmp_path) == "dflash2"


def test_dspark_keeps_its_own_architecture_even_with_dflash_named_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DSparkDraftModel"],
                                                     "dflash_config": {"projector_type": "dspark"}}))
    assert checkpoint_method(tmp_path) == "dspark"


def test_stock_mtp_optional_null_architectures_remains_valid(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": None}))
    assert checkpoint_method(tmp_path) == "mtp"


@pytest.fixture
def decoder(monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorfold.families.qwen4_exp.cuda import decode

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    consumed, proposals, maintenance = [], [], []
    state = NS(pos=20, target_tentative_length=20)
    state.discard_target_tentative = lambda: setattr(state, "target_tentative_length", state.pos)
    engine = NS(w=NS(mtp=object(), cfg=NS(eos={12})), st=state, rows=16,
                buf=NS(streams=torch.zeros((16, 1))), mbuf=object(),
                last_streams=torch.zeros((1, 1)), logical_context=100)
    def forward(tokens):
        state.target_tentative_length = state.pos + len(tokens)
        return torch.tensor(tokens)[:, None]

    engine.forward = forward
    engine.sample = lambda logits, positions, sampling, **kw: [int(row) + 1 for row in logits[:, 0]]

    def commit(_w, st, _b, _rows, keep):
        # Independently record the exact consumed prefix; future rows are
        # deliberately absent from the retained recurrence oracle.
        consumed.append(keep)
        st.pos += keep
        st.target_tentative_length = st.pos

    def draft(_e, _streams, tokens, _position, count, *_args):
        proposals.append(count)
        return [tokens[-1] + 1 + index for index in range(count)]

    monkeypatch.setattr(decode, "commit", commit)
    monkeypatch.setattr(decode, "draft", draft)
    monkeypatch.setattr(decode, "absorb", lambda e, streams, tokens: maintenance.append(list(tokens)))
    return decode, engine, consumed, proposals, maintenance


@pytest.mark.parametrize("count", [2, 3, 4, 5, 17, 33])
def test_all_accepted_serial_speculation_never_commits_a_hidden_extra_reply(decoder, count):
    decode, engine, consumed, proposals, _ = decoder
    result = decode.mtp_decode(engine, 10, count, None, depth=3)
    assert result.tokens == list(range(10, 10 + count))
    assert result.committed == result.tokens[:-1]
    assert sum(consumed) == count - 1
    assert engine.st.pos == 20 + count - 1
    assert all(0 <= count <= 3 for count in proposals)
    assert result.widths[-1] <= result.keeps[-1]


def test_reply_crossing_logical_context_is_refused_before_any_gpu_work(decoder):
    decode, engine, consumed, proposals, _ = decoder
    engine.logical_context = 23
    with pytest.raises(ValueError, match="logical context"):
        decode.mtp_decode(engine, 10, 4, None, depth=3)
    assert consumed == proposals == []


def test_eos_commits_only_the_prefix_that_generated_the_stop_token(decoder):
    decode, engine, consumed, _, _ = decoder
    result = decode.mtp_decode(engine, 10, 10, None, depth=3, stop_eos=True)
    assert result.tokens == [10, 11, 12]
    assert consumed == [2]
    assert result.committed == [10, 11]


def test_zero_proposal_serial_steps_keep_stock_conditioning_current(decoder):
    decode, engine, consumed, _, maintenance = decoder
    result = decode.serial_decode(engine, 10, 4, None)
    assert result.tokens == [10, 11, 12, 13]
    assert consumed == [1, 1, 1]
    assert maintenance == [[10], [11], [12], [13]]


def test_bad_provider_cannot_hide_extra_proposals_by_slicing_the_response(decoder, monkeypatch):
    decode, engine, consumed, _, _ = decoder
    monkeypatch.setattr(decode, "draft", lambda *_args: [11, 12, 13])
    with pytest.raises(RuntimeError, match="more proposals"):
        decode.mtp_decode(engine, 10, 2, None, depth=3)
    assert consumed == []


@pytest.mark.parametrize("keep", [0, 1, 3])
def test_stop_prefix_is_resolved_before_target_state_commit(decoder, keep):
    decode, engine, consumed, _, _ = decoder
    seen = []

    def limit(new, prefix):
        seen.append((list(new), list(prefix)))
        return keep, True

    result = decode.mtp_decode(engine, 10, 10, None, depth=2, token_limit=limit)
    assert seen == [([11, 12, 13], [10])]
    assert result.tokens == [10] + [11, 12, 13][:keep]
    assert sum(consumed) == keep and engine.st.pos == 20 + keep
    assert engine.st.target_tentative_length == engine.st.pos
    assert result.committed == result.tokens[:-1]


def test_full_width_stop_does_not_start_another_target_step(decoder):
    decode, engine, consumed, _, _ = decoder
    result = decode.mtp_decode(engine, 10, 10, None, depth=3,
                               token_limit=lambda new, _prefix: (len(new), True))
    assert result.tokens == [10, 11, 12, 13, 14]
    assert consumed == [4] and result.rounds == 1


def test_zero_prefix_leaves_serial_target_state_unadvanced(decoder):
    decode, engine, consumed, _, _ = decoder
    result = decode.serial_decode(engine, 10, 10, None, token_limit=lambda *_args: (0, True))
    assert result.tokens == [10] and result.committed == []
    assert consumed == [] and engine.st.pos == 20
    assert engine.st.target_tentative_length == engine.st.pos


@pytest.mark.parametrize("decision", [True, (True, True), (-1, True), (4, True), (1, 1), (1, False)])
def test_invalid_precommit_stop_decisions_fail_before_state_publication(decoder, decision):
    decode, engine, consumed, _, _ = decoder
    with pytest.raises(ValueError, match="token_limit|shortened"):
        decode.mtp_decode(engine, 10, 10, None, depth=2, token_limit=lambda *_args: decision)
    assert consumed == [] and engine.st.pos == 20


@pytest.mark.parametrize("rank", [0, 1])
def test_serial_tensor_parallel_stop_decision_uses_only_leader_and_same_boundary(rank):
    torch = pytest.importorskip("torch")
    from tensorfold.families.qwen4_exp.cuda.speculation import collective_prefix_limit

    calls = []

    def gather(send, receive):
        calls.append(send.tolist())
        receive.zero_()
        receive[0].copy_(torch.tensor([2, 1, 0]))

    comm = NS(rank=rank, world=2, device="cpu", all_gather=gather)
    callback = ((lambda new, prefix: (2, True)) if rank == 0 else
                lambda *_args: pytest.fail("follower evaluated a private tokenizer"))
    limit = collective_prefix_limit(callback, comm)
    assert limit([11, 12, 13], [10]) == (2, True)
    assert len(calls) == 1


def test_leader_tokenizer_failure_publishes_error_status_before_propagation():
    pytest.importorskip("torch")
    from tensorfold.families.qwen4_exp.cuda.speculation import collective_prefix_limit

    sent = []

    def gather(send, receive):
        sent.append(send.tolist())
        receive.copy_(send.expand_as(receive))

    original = ValueError("tokenizer failed")

    def broken(*_args):
        raise original

    limit = collective_prefix_limit(broken, NS(rank=0, world=2, device="cpu", all_gather=gather))
    with pytest.raises(ValueError) as error:
        limit([11, 12], [10])
    assert error.value is original
    assert sent == [[0, 0, 1]]


def test_background_continuation_keeps_pure_stop_policy():
    from tensorfold.cuda.streams import Stream

    def policy(new, prefix):
        return len(new), False
    stream = Stream([1, 2], 10, token_limit=policy)
    stream.finished = stream.started = 1.0
    assert stream.continued().token_limit is policy
