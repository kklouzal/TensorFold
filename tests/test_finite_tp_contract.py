"""Closed finite JSON/source extent and actual ordered-peer activation controls."""
import json
from types import SimpleNamespace as NS
import threading

import pytest

pytest.importorskip('torch')

from tensorfold.cuda import rank_protocol
from tensorfold.cuda.rank_protocol import FiniteTPContract, finite_error_identity
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
from tests.test_flashnext_tp_scheduler import BatchStore, Decoder


def role():
    return FiniteTPContract(context=128, vocab=64, inputs=8, outputs=12, requests=2, depth=2)


def boundary():
    return {'epoch': 0, 'settings': {'row_budget': 6, 'proposal_limit': 2, 'prefill_tokens': 16, 'session_ceiling': 2},
            'kernel_family': 'torch-reference', 'graph_family': 'uncaptured', 'profile_id': None}


def command():
    return {'kind': 'admit', 'sequence': 0, 'wire': 1, 'points': [1],
        'control': {'forget': [], 'cancel': [], 'arrived': False, 'admission_policy': 'fifo', 'flow': [],
                    'pass_rows': 16, 'execution_boundary': boundary()},
        'stream': {'prompt': [1, 2], 'count': 12, 'draft': True, 'stop_eos': True,
            'background': False, 'sampling': [17, 0, 0, 1, 0], 'grammar': [], 'owed': [], 'carry': None,
            'token_limit': False, 'profile_epoch': None, 'committed_output_tokens': 0,
            'admission_ticket': 1, 'principal': None, 'retention': None, 'output_capacity': 12}}


def test_closed_finite_roundtrip_keeps_native_context_distinct_from_cpu_extent():
    contract, body = role(), command()
    assert contract.cpu_tokens == 26 and contract.context == 128
    assert contract.command_bytes < rank_protocol.MAX_MESSAGE_BYTES
    assert FiniteTPContract.from_record(contract.record()) == contract
    contract.validate_command(body)
    encoded = rank_protocol.bounded_json(body, separators=(',', ':'), maximum=contract.command_bytes).encode()
    assert contract.loads(encoded) == body
    error = type('X' * 300 + '\U00010000', (RuntimeError,), {})()
    label = finite_error_identity(error)
    assert len(label) == 129 and label.startswith('X' * 64 + ':')
    assert finite_error_identity(type('X' * 300 + 'Y', (RuntimeError,), {})()) != label
    ack = {'error': label, 'state': 'f' * 64}
    raw = rank_protocol.bounded_json(ack, separators=(',', ':'), maximum=contract.ack_bytes).encode()
    assert contract.loads(raw, acknowledgment=True) == ack


@pytest.mark.parametrize('mutation', [
    lambda b: b.update(kind='serving_replace'),
    lambda b: b.update(extra='unexpected'),
    lambda b: b['stream'].update(prompt=[1] * 9),
    lambda b: b['stream'].update(prompt=[True]),
    lambda b: b['stream'].update(count=13),
    lambda b: b['stream'].update(stop_eos=False),
    lambda b: b['stream'].update(background=True),
    lambda b: b['stream'].update(grammar=[1]),
    lambda b: b['stream'].update(carry={}),
    lambda b: b['stream'].update(principal='foreign'),
    lambda b: b['stream'].update(sampling=[17, float('nan'), 0, 1, 0]),
    lambda b: b['stream'].update(sampling=[17, 0, 0, 2, 0]),
    lambda b: b['control'].update(cancel=[[1, 1]]),
    lambda b: b['control'].update(flow=[[1, 'output', 13]]),
    lambda b: b['control'].update(extra=0),
    lambda b: b['control']['execution_boundary']['settings'].update(foreign=0),
])
def test_unsupported_type_schema_or_source_extent_rejects_before_encoding(mutation):
    body = command()
    mutation(body)
    with pytest.raises(ValueError):
        role().validate_command(body)


def test_raw_length_and_type_gate_precedes_json_parser(monkeypatch):
    contract = role()
    monkeypatch.setattr(rank_protocol.json, 'loads', lambda *_: pytest.fail('invalid raw reached parser'))
    for raw in (b' ' * (contract.command_bytes + 1), 'foreign str', memoryview(b'{}')):
        with pytest.raises(ValueError):
            contract.loads(raw)
    with pytest.raises(ValueError):
        contract.loads(b' ' * (contract.ack_bytes + 1), acknowledgment=True)


def test_small_closed_json_has_no_executable_reconstruction_and_rejects_bad_roots():
    contract = role()
    for raw in (b'null', b'[]', b'42', b'{"kind":"GLOBAL","reduce":[]}'):
        with pytest.raises(ValueError):
            contract.loads(raw)
    for value in ({'error': None, 'state': 'bad'}, {'error': [], 'state': 'f' * 64},
                  {'error': None, 'state': 'f' * 64, 'extra': 1}):
        with pytest.raises(ValueError):
            contract.loads(json.dumps(value).encode(), acknowledgment=True)


class FiniteDecoder(Decoder):
    def __init__(self, rank, store, world=2):
        super().__init__(rank, world, store)
        self.logical_context, self.depth, self.full_resident, self.copy = 128, 2, True, False
        self.w.cfg = NS(vocab=64)
        self.states = [NS(), NS()]
        self.kept = []
        self.controller = AdaptiveController(slots=2, row_budget=6, proposal_limit=2, prefill_tokens=16)
        from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation

        self.w.positions = object()
        self.w.meta = {'world': world, 'rank': rank, 'coordinate_profile_identity': 'a' * 64}
        self.input_preparation_context = install_input_preparation(self.w)
        self.input_preparation_context.bind_decoder(self)
    def freeze_execution(self):
        return self.controller.freeze(list(self.streams))
    def accept_execution(self, plan):
        self.controller.accept(plan)


class PreparationStore(BatchStore):
    def wait(self, keys, duration):
        # Actual TP transactions and the real input-context retirement use
        # their different documented deadlines. Preserve the base fixture.
        assert duration.total_seconds() in (60, 3600)
        with self.condition:
            assert self.condition.wait_for(lambda: all(key in self.data for key in keys), 5), keys


def pair(*, wrong_peer=False):
    store = PreparationStore()
    peers = [TensorParallelDecoder(FiniteDecoder(rank, store), NS(
        _abort_comm=False, _closing=False, _lifecycle=threading.Condition(), serving_profiles=None)) for rank in range(2)]
    for peer in peers:
        peer.configure_admission_policy('fifo')
    if wrong_peer:
        peers[1].decoder.copy = True
    errors = []
    def follow():
        try:
            while (body := peers[1].receive()) is not None:
                peers[1].apply(body)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=follow, daemon=True)
    thread.start()
    return peers, thread, errors, store


def test_real_ordered_peers_activate_same_role_before_generation_then_finish_and_stop():
    peers, thread, errors, store = pair()
    root = peers[0]
    try:
        root.configure_finite_caller(role())
        assert peers[0]._finite_contract == peers[1]._finite_contract == role()
        assert not root.requests and root.decoder.live() == 0
        stream = Stream([3, 5], 3, Sampling(17, 0, 0, 1), draft=False, cancel=threading.Event())
        stream.admission_ticket, stream.principal = 1, None
        stream.admission_status, stream.retention_intent, stream.output_capacity = NS(ticket=1, principal=None), None, 3
        root.admit(stream)
        for _ in range(3):
            ended = root.round()
        assert stream.out == [6, 7, 8] and ended == [stream]
        root.finish(ended)
        assert all(not peer.requests and not peer.decoder.live() for peer in peers)
    finally:
        root.shutdown()
        thread.join(6)
    assert not thread.is_alive() and errors == [] and not store.data


def test_partial_role_installation_poison_has_no_generic_continuation():
    peers, thread, errors, _ = pair(wrong_peer=True)
    with pytest.raises(RuntimeError):
        peers[0].configure_finite_caller(role())
    assert peers[0].failure is not None and peers[0].engine._abort_comm
    thread.join(6)
    assert not thread.is_alive() and errors
    assert peers[1].failure is not None and peers[1].engine._abort_comm
    with pytest.raises(RuntimeError, match='poisoned'):
        peers[0].round()



def test_tp8_all_peers_acknowledge_the_same_finite_role_and_retire():
    store, errors = PreparationStore(), []
    peers = [TensorParallelDecoder(FiniteDecoder(rank, store, world=8), NS(
        _abort_comm=False, _closing=False, _lifecycle=threading.Condition(), serving_profiles=None))
        for rank in range(8)]
    for peer in peers:
        peer.configure_admission_policy('fifo')
    def follow(peer):
        try:
            while (body := peer.receive()) is not None:
                peer.apply(body)
        except BaseException as error:
            errors.append(error)
    threads = [threading.Thread(target=follow, args=(peer,), daemon=True) for peer in peers[1:]]
    for thread in threads:
        thread.start()
    try:
        peers[0].configure_finite_caller(role())
        assert all(peer._finite_contract == role() for peer in peers)
        stream = Stream([3, 5], 2, Sampling(17, 0, 0, 1), draft=False, cancel=threading.Event())
        stream.admission_ticket, stream.principal = 1, None
        stream.admission_status, stream.retention_intent, stream.output_capacity = NS(ticket=1, principal=None), None, 2
        peers[0].admit(stream)
        peers[0].round()
        ended = peers[0].round()
        peers[0].finish(ended)
        assert all(not peer.requests and not peer.decoder.live() for peer in peers)
    finally:
        peers[0].shutdown()
        for thread in threads:
            thread.join(6)
    assert not errors and all(not thread.is_alive() for thread in threads) and not store.data


def test_maximum_supported_source_shapes_fit_the_derived_encoded_extent():
    contract = FiniteTPContract(262144, 248320, 4096, 320, 24, 2)
    body = command()
    maximum = (1 << 63) - 1
    body.update(sequence=maximum, wire=maximum, points=[4096] * 4096)
    body['stream'].update(prompt=[248319] * 4096, count=320, admission_ticket=maximum, output_capacity=320)
    body['control'].update(forget=[maximum] * 24, cancel=[[maximum, True] for _ in range(24)],
                           flow=[[maximum, 'tier_unavailable', 320] for _ in range(24)])
    contract.validate_command(body)
    raw = rank_protocol.bounded_json(body, separators=(',', ':'), maximum=contract.command_bytes).encode()
    assert len(raw) <= contract.command_bytes and contract.loads(raw) == body
    name = '\U00010000' * 64
    label = finite_error_identity(type(name, (RuntimeError,), {})())
    ack = {'error': label, 'state': 'f' * 64}
    encoded = rank_protocol.bounded_json(ack, separators=(',', ':'), maximum=contract.ack_bytes).encode()
    assert len(encoded) == contract.ack_bytes
    assert contract.loads(encoded, acknowledgment=True) == ack


def test_finite_ack_failure_preserves_original_local_operation_error():
    contract, sent = role(), []
    store = NS(set=lambda *args: sent.append(args), multi_set=lambda keys, values: sent.extend(zip(keys, values)),
        wait=lambda *_: None, multi_get=lambda keys: [b' ' * (contract.ack_bytes + 1) for _ in keys],
        delete_key=lambda _: None)
    root = TensorParallelDecoder(FiniteDecoder(0, store), NS(
        _abort_comm=False, _closing=False, _lifecycle=threading.Condition(), serving_profiles=None))
    root.configure_admission_policy('fifo')
    root._finite_contract = contract  # Isolate the ACK-failure seam; real activation is covered above.
    primary = ValueError('first local operation failure')
    root._execute = lambda *_: (None, primary, {'error': finite_error_identity(primary), 'state': 'f' * 64})
    with pytest.raises(ValueError) as caught:
        root._invoke({'kind': 'drop', 'error': None})
    assert caught.value is primary and root.failure is primary and root.engine._abort_comm
    assert isinstance(primary.__cause__, ValueError) and 'received frame' in str(primary.__cause__)


def test_finite_receiver_rejects_unexpected_operation_before_controller_or_model_changes():
    sent = []
    store = NS(set=lambda *args: sent.append(args))
    peer = TensorParallelDecoder(FiniteDecoder(1, store), NS(
        _abort_comm=False, _closing=False, _lifecycle=threading.Condition(), serving_profiles=None))
    peer.configure_admission_policy('fifo')
    peer._finite_contract = role()
    before = (peer.prefix_operation, peer.decoder.controller.epoch, list(peer.decoder.observed))
    with pytest.raises(RuntimeError, match='peer failed'):
        peer.apply({'kind': 'serving_replace', 'sequence': 0})
    assert (peer.prefix_operation, peer.decoder.controller.epoch, peer.decoder.observed) == before
    assert peer.failure is not None and peer.engine._abort_comm and sent


def test_finite_receive_rejects_stale_sequence_before_increment_or_store_retirement():
    body, deleted = command(), []
    body['sequence'] = 1
    store = NS(wait=lambda *_: None, get=lambda _: json.dumps(body).encode(), delete_key=deleted.append)
    peer = TensorParallelDecoder(FiniteDecoder(1, store), NS(
        _abort_comm=False, _closing=False, _lifecycle=threading.Condition(), serving_profiles=None))
    peer.configure_admission_policy('fifo')
    peer._finite_contract = role()
    with pytest.raises(ValueError, match='Store sequence'):
        peer.receive()
    assert peer.sequence == 0 and not deleted and peer.failure is not None and not peer.requests


@pytest.mark.parametrize('cut', ['encoder', 'publication', 'containment'])
def test_finite_apply_preserves_original_operation_error_across_ack_failure(monkeypatch, cut):
    primary, secondary = ValueError('first follower operation failure'), MemoryError('ACK publication failure')
    def fail(*_args, **_kwargs):
        raise secondary
    store = NS(set=fail if cut != 'encoder' else lambda *_: None)
    peer = TensorParallelDecoder(FiniteDecoder(1, store), NS(
        _abort_comm=False, _closing=False, _lifecycle=threading.Condition(), serving_profiles=None))
    peer.configure_admission_policy('fifo')
    peer._finite_contract = role()
    peer._execute = lambda *_: (None, primary, {'error': finite_error_identity(primary), 'state': 'f' * 64})
    if cut == 'encoder':
        monkeypatch.setattr(rank_protocol, 'bounded_json', fail)
    if cut == 'containment':
        def failed_containment(error):
            peer.failure, peer.engine._abort_comm = error, True
            raise OSError('containment diagnostic')
        peer._poison_control = failed_containment
    with pytest.raises(ValueError) as caught:
        peer.apply({'kind': 'round', 'sequence': 0})
    assert caught.value is primary and peer.failure is primary and peer.engine._abort_comm
    assert primary.__cause__ is not None
    if cut != 'containment':
        assert primary.__cause__ is secondary


@pytest.mark.parametrize('cut', ['error-identity', 'state-digest'])
def test_finite_status_preparation_cannot_replace_original_operation_error(monkeypatch, cut):
    primary, secondary = ValueError('first dispatch failure'), MemoryError('status preparation failure')
    peer = TensorParallelDecoder(FiniteDecoder(1, NS()), NS(
        _abort_comm=False, _closing=False, _lifecycle=threading.Condition(), serving_profiles=None))
    peer.configure_admission_policy('fifo')
    peer._finite_contract = role()
    def fail_dispatch(*_):
        raise primary
    def fail(*_):
        raise secondary
    peer._dispatch = fail_dispatch
    if cut == 'error-identity':
        monkeypatch.setattr(rank_protocol, 'finite_error_identity', fail)
    else:
        peer._state_digest = fail
    with pytest.raises(ValueError) as caught:
        peer._execute({'kind': 'round', 'sequence': 0})
    assert caught.value is primary and primary.__cause__ is secondary and peer.failure is primary
