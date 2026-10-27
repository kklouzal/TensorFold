"""Exact status/candidate transport, owned CPU results and existing sampler oracles."""

from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling, choose  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import decode, multi  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import KVWords, candidates  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort, workspace_bytes  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import CAND, KVNumericError  # noqa: E402
from test_flashnext_kv_status_cohort import states_for  # noqa: E402


def cohort(words, rows):
    world, slots, _ = words.shape
    states = states_for(words[0].tolist(), world)
    calls = []
    def gather(send, receive):
        calls.append(send.clone())
        torch.testing.assert_close(send, words[0], rtol=0, atol=0)
        receive.copy_(words.reshape(-1))
    comm = NS(rank=0, world=world, all_gather=gather)
    return KVStatusCohort(slots, comm, "cpu", candidate_rows=rows), states, comm, calls


def healthy(world, slots):
    words = torch.zeros((world, slots, 2), dtype=torch.int32)
    words[:, :, 1] = -2
    return words


@pytest.mark.parametrize("world,slots,rows", [(2, 1, 1), (4, 3, 5), (8, 26, 32)])
def test_combined_packet_preserves_every_bit_and_never_borrows_the_reusable_host(world, slots, rows):
    workspace, states, comm, calls = cohort(healthy(world, slots), rows)
    packet = torch.arange(world * rows * (2 * CAND + 1), dtype=torch.int32)
    packet[::13] = 0x7fc00001  # NaN payload is transport data until status is valid.
    packet[::17] = -2147483648  # Signed zero/global-ID bits must not be numerically converted.
    before = packet.numpy().copy()
    received = workspace.read_candidates(states, packet.view(torch.float32), rows, comm=comm)
    np.testing.assert_array_equal(received.view(np.int32).reshape(-1), before)
    assert received.flags.owndata and received.flags.c_contiguous and len(calls) == 1
    assert not np.shares_memory(received, workspace.host.numpy())
    workspace.host.fill_(71)
    packet.zero_()
    np.testing.assert_array_equal(received.view(np.int32).reshape(-1), before)
    assert all(not state._kv_pending for state in states)


@pytest.mark.parametrize("fault", ["remote", "local-and-remote"])
def test_every_peer_status_is_latched_before_candidate_interpretation(monkeypatch, fault):
    words = healthy(8, 3)
    words[7, 0] = torch.tensor([4, -1], dtype=torch.int32)
    if fault == "local-and-remote":
        words[0, 2] = torch.tensor([2, 11], dtype=torch.int32)
    workspace, states, comm, calls = cohort(words, 1)
    packet = torch.full((8 * 65,), float("nan"), dtype=torch.float32)
    monkeypatch.setattr(torch.Tensor, "numpy", lambda self: pytest.fail("inspected a failed candidate frame"))
    with pytest.raises(KVNumericError) as caught:
        workspace.read_candidates(states, packet, 1, comm=comm)
    assert len(calls) == 1 and all(not state._kv_pending for state in states)
    assert states[0]._kv_error is caught.value
    if fault == "local-and-remote":
        assert states[2]._kv_error is caught.value and "layer=11" in str(caught.value)
    else:
        assert "rank 7" in str(caught.value)


def test_invalid_candidate_metadata_still_reports_a_failed_peer_first():
    words = healthy(2, 1)
    words[1, 0, 0] = 8
    workspace, states, comm, _calls = cohort(words, 1)
    with pytest.raises(KVNumericError):
        workspace.read_candidates(states, torch.empty(0), 2, comm=comm)


def test_numeric_error_traceback_does_not_export_a_reusable_pinned_view():
    words = healthy(2, 1)
    words[1, 0, 0] = 4
    workspace, states, comm, _calls = cohort(words, 1)
    with pytest.raises(KVNumericError) as caught:
        workspace.read_candidates(states, torch.zeros(2 * 65), 1, comm=comm)
    first, last = workspace.host.data_ptr(), workspace.host.data_ptr() + workspace.host.nbytes
    traceback = caught.value.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_filename.endswith("/kv_status.py"):
            assert not any(isinstance(value, torch.Tensor) and first <= value.data_ptr() < last
                           for value in frame.f_locals.values())
        traceback = traceback.tb_next


@pytest.mark.parametrize("failure", [False, True])
def test_one_copy_and_one_positive_completion_or_original_failure(monkeypatch, failure):
    workspace, states, comm, _calls = cohort(healthy(2, 1), 1)
    workspace.receive.fill_(29)
    # Exercise the asynchronous completion branch with real CPU copies and a
    # controlled stream controller; this is not a CUDA execution claim.
    workspace.device = NS(type="cuda")
    primary, events = KeyboardInterrupt("controlled readback completion"), []
    def complete():
        assert torch.equal(workspace.host, workspace.receive)
        events.append("completed")
        if failure:
            raise primary
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: NS(synchronize=complete))
    original = torch.Tensor.copy_
    def copy(destination, source, **options):
        if destination.data_ptr() == workspace.host.data_ptr():
            events.append("copy")
            assert options == {"non_blocking": True}
        return original(destination, source, **options)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    if failure:
        with pytest.raises(KeyboardInterrupt) as caught:
            workspace._readback(workspace.host.numel())
        assert caught.value is primary
        with pytest.raises(RuntimeError, match="requires retirement") as retired:
            workspace.check(states, comm=comm)
        assert retired.value.__cause__ is primary
    else:
        workspace._readback(workspace.host.numel())
    assert events == ["copy", "completed"]


def test_failed_collective_never_reuses_an_ambiguous_owner():
    workspace, states, comm, _calls = cohort(healthy(2, 1), 1)
    primary = RuntimeError("controlled collective failure")
    def gather(send, receive):
        raise primary
    comm.all_gather = gather
    with pytest.raises(RuntimeError) as caught:
        workspace.read_candidates(states, torch.zeros(2 * 65), 1, comm=comm)
    assert caught.value is primary
    with pytest.raises(RuntimeError, match="requires retirement") as retired:
        workspace.check(states, comm=comm)
    assert retired.value.__cause__ is primary


@pytest.mark.parametrize("after_copy", [False, True])
def test_copy_failure_preserves_primary_and_prevents_storage_reuse(monkeypatch, after_copy):
    workspace, states, comm, _calls = cohort(healthy(2, 1), 1)
    primary = MemoryError("controlled host transfer failure")
    original = torch.Tensor.copy_
    def copy(destination, source, **options):
        if destination.data_ptr() == workspace.host.data_ptr():
            if after_copy:
                original(destination, source, **options)
            raise primary
        return original(destination, source, **options)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(MemoryError) as caught:
        workspace.read_candidates(states, torch.zeros(2 * 65), 1, comm=comm)
    assert caught.value is primary
    with pytest.raises(RuntimeError, match="requires retirement") as retired:
        workspace.check(states, comm=comm)
    assert retired.value.__cause__ is primary


def test_admitted_bytes_cover_real_device_pinned_and_owned_array_peak():
    workspace, states, comm, _calls = cohort(healthy(8, 26), 32)
    received = workspace.read_candidates(states, torch.zeros(8 * 32 * 65), 32, comm=comm)
    device, host = workspace_bytes(26, 8, candidate_rows=32)
    assert device == workspace.send.nbytes + workspace.receive.nbytes
    assert host == workspace.host.nbytes + received.__sizeof__()
    assert workspace.pinned_host_bytes == workspace.host.nbytes
    assert workspace.owned_candidate_bytes == received.__sizeof__()
    assert workspace_bytes(6, 8) == (432, 384)  # Status-only users keep their original resource contract.


def complete_packet(widths, rows=3, *, tied=False):
    rng = np.random.default_rng(712)
    shards, frames, ids = [], [], []
    offset = 0
    for width in widths:
        logits = torch.from_numpy(rng.normal(size=(rows, width)).astype(np.float32))
        if tied:
            logits.zero_()
        mapping = torch.arange(offset, offset + width, dtype=torch.int64).flip(0)
        offset += width
        buffer = NS(cand=torch.empty(rows, 65), cand_all=torch.empty(rows * 65), world=1)
        owner = NS(meta={"execution_provider": "sm70"}, comm=NS(
            all_gather=lambda send, receive: receive.copy_(send.reshape(-1))))
        candidates(owner, buffer, logits, rows, id_map=mapping)
        frames.append(buffer.cand_all.view(rows, 65).numpy().copy())
        shards.append(logits.numpy())
        ids.append(mapping.numpy())
    return np.stack(frames), np.concatenate(shards, axis=1), np.concatenate(ids)


@pytest.mark.parametrize("sampling", [None, Sampling(17, 0), Sampling(17, .7, 2, .9, .01)])
@pytest.mark.parametrize("tied", [False, True])
def test_received_scores_match_full_vocabulary_greedy_and_independent_keyed_oracle(sampling, tied):
    widths, positions = [7, 17, 31], [2, 77, 258]
    packet, full, ids = complete_packet(widths, tied=tied)
    chosen, probabilities = decode._choose_received(NS(meta={}), packet, positions, sampling,
                                                     with_prob=True, plan=NS(widths=widths))
    expected = [int(ids[np.lexsort((ids, -row))[0]]) if sampling is None or sampling.temperature <= 0
                else choose(row, ids, position, sampling) for row, position in zip(full, positions)]
    assert chosen == expected
    norms = packet[:, :, -1].astype(np.float64)
    top = norms.max(axis=0)
    normalization = top + np.log(np.exp(norms - top).sum(axis=0))
    expected_probability = [float(np.exp(float(full[row, np.flatnonzero(ids == token)[0]]) - normalization[row]))
                            for row, token in enumerate(expected)]
    assert probabilities == expected_probability


@pytest.mark.parametrize("fault", ["score-nan", "score-inf", "norm-nan", "norm-inf", "all-masked"])
def test_received_invalid_scores_never_enter_a_fallback_collective(monkeypatch, fault):
    packet, _full, _ids = complete_packet([7, 17], rows=1)
    if fault == "all-masked":
        packet[:, :, :CAND] = -np.inf
        packet[:, :, -1] = -np.inf
    elif fault.startswith("score"):
        packet[0, 0, 0] = np.nan if fault.endswith("nan") else np.inf
    else:
        packet[0, 0, -1] = np.nan if fault.endswith("nan") else np.inf
    monkeypatch.setattr(decode, "tp_sample_rows", lambda *a, **kw: pytest.fail("fell back from invalid scores"))
    with pytest.raises(ValueError):
        decode._choose_received(NS(meta={}), packet, [2], None, plan=NS(widths=[7, 17]))


def test_tied_truncated_packet_uses_original_source_only_after_valid_status(monkeypatch):
    packet, full, ids = complete_packet([33, 33], rows=1, tied=True)
    workspace, states, comm, _calls = cohort(healthy(2, 1), 1)
    received = workspace.read_candidates(states, torch.from_numpy(packet.reshape(-1)), 1, comm=comm)
    expected = [int(ids[np.lexsort((ids, -full[0]))[0]])]
    source, seen = object(), []
    def complete(owner, logits, positions, sampling, **options):
        assert logits is source and positions == [17] and sampling is None
        assert all(not state._kv_pending for state in states)
        seen.append("original full source")
        return expected
    monkeypatch.setattr(decode, "tp_sample_rows", complete)
    result = decode._choose_received(NS(meta={}), received, [17], None,
                                     logits=source, plan=NS(widths=[33, 33]))
    assert result == expected and seen == ["original full source"]


def test_actual_mtp_boundary_finishes_status_before_publishing_owned_candidates(monkeypatch):
    workspace, states, comm, calls = cohort(healthy(2, 1), 1)
    packet, _full, _ids = complete_packet([7, 17], rows=1)
    logits = torch.zeros((1, 7), dtype=torch.float32)
    dec = NS(w=NS(comm=comm, meta={"execution_provider": "sm70"}), kv_cohort=workspace,
             streams={0: object()}, filling=[], live=lambda: 1, _rotor_lookup=lambda *a: False,
             mbuf=NS(h=torch.empty(1, dtype=torch.float16), logits=logits,
                     cand_all=torch.from_numpy(packet.reshape(-1))), graphs=None, controller=None,
             _draft_plan=NS(graph_ids_safe=True))
    dec._candidate_readback = MethodType(MultiDecoder._candidate_readback, dec)
    monkeypatch.setattr(multi, "mtp_compute", lambda *a, **kw: logits)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self: pytest.fail("performed a second readback"))
    result, received = MultiDecoder._mtp(dec, [(states[0], 0, 1)], candidate_policies=[None])
    assert result is logits and received.flags.owndata and len(calls) == 1
    np.testing.assert_array_equal(received.view(np.int32), packet.view(np.int32))
    events = []
    words = KVWords([(states[0], 0, 1)], torch.zeros((1, 2), dtype=torch.int32), lambda: events.append("published"))
    words.settle()
    assert events == ["published"]


@pytest.mark.parametrize("invalid", [
    Sampling(17, "not-a-temperature"), Sampling(17, float("nan"), 2),
    Sampling(17, float("inf"), 2), Sampling(17, -float("inf"), 2),
    Sampling(17, .7, 2, 2.0), Sampling(17, .7, 2, 1.0, 2.0),
])
@pytest.mark.parametrize("failed_peer", [False, True])
def test_optional_transport_does_not_overtake_original_sampling_error_or_publication(
        monkeypatch, invalid, failed_peer):
    status = healthy(2, 1)
    if failed_peer:
        status[1, 0, 0] = 4
    workspace, states, comm, calls = cohort(status, 1)
    dec = NS(w=NS(comm=comm, meta={"execution_provider": "sm70"}), kv_cohort=workspace,
             _draft_map=None)
    def unexpected_copy(*args, **kwargs):
        raise MemoryError("invalid metadata reached optional candidate allocation")
    monkeypatch.setattr(workspace, "read_candidates", unexpected_copy)
    packet = torch.zeros(2 * 65)
    if failed_peer:
        with pytest.raises(KVNumericError):
            MultiDecoder._candidate_readback(dec, [(states[0], 0, 1)], packet, 1, [invalid], None)
        assert len(calls) == 1 and not states[0]._kv_pending
        return
    with pytest.raises((ValueError, TypeError, OverflowError)) as original:
        multi.validate_policy([17], invalid)
    result = MultiDecoder._candidate_readback(dec, [(states[0], 0, 1)], packet, 1, [invalid], None)
    assert result is None and len(calls) == 1 and not states[0]._kv_pending
    events = []
    words = KVWords([(states[0], 0, 1)], torch.zeros((1, 2), dtype=torch.int32), lambda: events.append("published"))
    # Keep actual policy validation and callback ordering, supplying only the
    # controlled CPU equivalent of the unchanged CUDA tensor-boundary check.
    monkeypatch.setattr(multi, "validate_rows", lambda logits, positions, sampling: None)
    with pytest.raises(type(original.value)) as caught:
        MultiDecoder._picks(dec, torch.zeros((1, 7)), [17], [invalid], words)
    assert str(caught.value) == str(original.value) and events == ["published"]


@pytest.mark.parametrize("field", ["seed", "top_k"])
def test_equal_policy_values_do_not_hide_a_later_invalid_native_type(monkeypatch, field):
    valid = Sampling(17, 0, 2)
    invalid = Sampling(np.int64(17), 0, 2) if field == "seed" else Sampling(17, 0, 2)
    if field == "top_k":
        # The constructor normalizes top_k; forge only this malformed frozen
        # state so the existing producing validator remains the refusal oracle.
        object.__setattr__(invalid, "top_k", np.int64(2))
    assert invalid == valid  # Equality of values alone does not establish the supported policy domain.
    workspace, states, comm, calls = cohort(healthy(2, 2), 2)
    def unexpected_copy(*args, **kwargs):
        raise MemoryError("equal invalid policy reached optional candidate allocation")
    monkeypatch.setattr(workspace, "read_candidates", unexpected_copy)
    dec = NS(w=NS(comm=comm, meta={"execution_provider": "sm70"}), kv_cohort=workspace,
             _draft_map=None)
    segs = [(states[0], 0, 1), (states[1], 1, 2)]
    assert MultiDecoder._candidate_readback(dec, segs, torch.zeros(2 * 2 * 65), 2,
                                           [valid, invalid], None) is None
    assert len(calls) == 1 and all(not state._kv_pending for state in states)
    events = []
    words = KVWords(segs, torch.zeros((2, 2), dtype=torch.int32), lambda: events.append("published"))
    monkeypatch.setattr(multi, "validate_rows", lambda logits, positions, sampling: None)
    with pytest.raises(ValueError, match="integer keyed seed"):
        MultiDecoder._picks(dec, torch.zeros((2, 7)), [17, 18], [valid, invalid], words)
    assert events == ["published"]
