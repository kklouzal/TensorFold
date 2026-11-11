"""Bounded Q8 diagnostic ownership, same-input ordering and failure controls."""
from types import SimpleNamespace as NS

import pytest
import torch

from tools import v100_owned_qualification as owned


def fixture(monkeypatch, tmp_path, *, fault=None):
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    monkeypatch.setattr(owned, '_history_resources', lambda *a: {
        'per_rank_payload_bytes': 65536, 'global_host_obligation_bytes': 8*65536,
        'receipt_bytes_limit': 1 << 20, 'materialized_credit_bytes': 0})
    requests = [{'index': i, 'prompt_sha256': f'{i:064x}'} for i in range(24)]
    cohort = {'id': 6, 'level': 24, 'warmup': False, 'input_tokens': 64, 'tokens': 32, 'requests': requests}
    plan = {'profile': 'q8', 'cohorts': [cohort]}
    resources = owned._q8_parity_resources(plan, tmp_path)
    resources['input_sha256'] = 'a'*64
    layers = [NS(index=i, attn=NS(q_scale=torch.ones(256), k_scale=torch.ones(256), iq_scale=torch.ones(128)))
              for i in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)]
    weights = NS(device=torch.device('cpu'), layers=layers)
    p = torch.zeros((24, 2688), dtype=torch.float16)
    p[:, 0] = torch.arange(1, 25)
    p[:, 1] = torch.arange(2, 26)
    coordinates = torch.full((24, 3), 64, dtype=torch.int64)
    buf = NS(pa=p, target_indexer_coordinates=coordinates,
             q=torch.zeros((24, 3, 256), dtype=torch.float16), iq=torch.zeros((24, 4, 128), dtype=torch.float16))
    states = []
    for index in range(24):
        cache = NS(key_format=get_pair('int8').key, value_format=get_pair('int8').value,
            placement=NS(page_tokens=128), offsets=(0,), k=torch.full((128, 1, 256), 9, dtype=torch.int8),
            v=torch.full((128, 1, 256), 9, dtype=torch.int8), ks=torch.full((128, 1, 8), 9, dtype=torch.float16),
            vs=torch.full((128, 1, 8), 9, dtype=torch.float16))
        states.append(NS(pos=64, history_slot=index, history_rank=0, slot_generation=1, history_generation=0,
                         history_generations=torch.tensor([1, 0]), kv_status=torch.tensor([0, -2], dtype=torch.int32),
                         cache=cache))
    streams = {i: NS(st=state, done=False, out=[1, 2], index=i) for i, state in enumerate(states)}
    buf.gdn_tables = NS(segs=[(state, i, i+1) for i, state in enumerate(states)])
    native = NS(__name__='synthetic-native-control', __file__=owned.__file__)
    buf.history_owner = NS(owned_q8_preparation=native)
    decoder = NS(buf=buf, states=states, streams=streams, filling=[], gdn=NS(q=torch.zeros((24, 2, 128))))
    engine = NS(rank=0, w=weights, multi=decoder, _closed=False, _abort_comm=False,
                tp_decoder=NS(wires={id(stream): i for i, stream in streams.items()}))
    close_calls = []
    def close():
        close_calls.append(True)
        engine._closed = True
        return 'closed-result'
    engine.close = close
    log, notices = [], []
    result = object()
    def status(value, layer):
        value[0].bitwise_or_(1)
        if int(value[1]) == -2:
            value[1] = layer.index
    def reference(projection, shadow, layer, weights, buffers, first, last, *, mtp):
        log.append(('reference', shadow.kv_status.clone()))
        buffers.q.copy_(projection[0, 0].expand_as(buffers.q))
        buffers.iq.copy_(projection[0, 1].expand_as(buffers.iq))
        status(shadow.kv_status, layer)
        shadow.kc[0].append(shadow.pos, torch.full((1, 1, 256), 3, dtype=torch.int8),
            torch.full((1, 1, 256), 4, dtype=torch.int8), torch.ones((1, 1, 8), dtype=torch.float16),
            torch.full((1, 1, 8), 2, dtype=torch.float16))
    def original(module, projection, coords, weights, cache, state, layer, query, iq, **kwargs):
        log.append(('candidate', state.kv_status.clone()))
        if fault == 'candidate':
            raise primary
        query.copy_(projection[0, 0].expand_as(query))
        iq.copy_(projection[0, 1].expand_as(iq))
        status(state.kv_status, layer)
        if cache.offsets[0] >= 0:
            cache.k[64].fill_(3)
            cache.v[64].fill_(4)
            cache.ks[64].fill_(1)
            cache.vs[64].fill_(2)
        if fault == 'bit':
            query.view(torch.int16).view(-1)[0].bitwise_xor_(1)
        return result
    primary = OSError('actual candidate primary')
    module = NS(owned_q8_prepare=original, __file__=owned.__file__)
    observer = NS(current=6, finished=True, failed=None,
                  _identity=lambda stream: (cohort, requests[stream.index]))
    monkeypatch.setattr(owned, '_q8_witness_region', lambda e: e.multi)
    monkeypatch.setattr(owned, '_observer_failure', lambda e, p: notices.append(p))
    def allocate(shape, **kwargs):
        return torch.empty(shape, dtype=kwargs['dtype'])
    witness = owned._Q8PreparationParity(engine, resources, tmp_path, module=module, reference=reference,
                                        allocator=allocate, fence=lambda: None)
    witness.observer = observer
    def call(state_index=0, layer_index=0):
        state, layer = states[state_index], layers[layer_index]
        return witness(None, p[state_index:state_index+1], coordinates[state_index:state_index+1],
            weights, state.cache, state, layer, buf.q[state_index:state_index+1], buf.iq[state_index:state_index+1],
            frequency=torch.ones(64), sections=(11, 11, 10), amplitude=1., start=64)
    return NS(witness=witness, engine=engine, observer=observer, resources=resources, states=states,
              layers=layers, module=module, original=original, primary=primary, result=result,
              call=call, log=log, notices=notices, close_calls=close_calls)


def test_one_actual_warmed_round_reference_first_status_shadow_return_identity_and_close(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    f.witness.install()
    f.witness.begin_round(f.observer)
    for layer in range(12):
        for state in range(24):
            assert f.call(state, layer) is f.result
    f.witness.end_round()
    assert f.witness.completed and f.witness.calls == f.witness.matches == 288
    for index in range(0, len(f.log), 2):
        assert f.log[index][0] == 'reference' and f.log[index+1][0] == 'candidate'
        assert torch.equal(f.log[index][1], f.log[index+1][1])
    f.witness.begin_round(f.observer)
    assert not f.witness.active  # Never select a second round.
    saved = f.engine.close
    assert saved() == 'closed-result'
    assert f.module.owned_q8_prepare is f.original
    assert not f.witness.views and not f.witness.slabs and f.witness.engine is None
    with pytest.raises(RuntimeError, match='retired'):
        saved()
    assert len(f.close_calls) == 1


@pytest.mark.parametrize('fault', ['filling', 'short-output', 'not24', 'foreign-cohort'])
def test_round_selection_never_substitutes_unwarmed_or_foreign_work(monkeypatch, tmp_path, fault):
    f = fixture(monkeypatch, tmp_path)
    if fault == 'filling':
        f.engine.multi.filling = [object()]
    elif fault == 'short-output':
        f.engine.multi.streams[0].out = [1]
    elif fault == 'not24':
        f.engine.multi.streams.pop(23)
    else:
        f.observer.current = 7
    f.witness.begin_round(f.observer)
    assert not f.witness.selected and not f.witness.active


@pytest.mark.parametrize('fault', ['generation', 'wire', 'projection', 'coordinates', 'duplicate'])
def test_selected_row_authority_and_bounded_no_repeat(monkeypatch, tmp_path, fault):
    f = fixture(monkeypatch, tmp_path)
    f.witness.begin_round(f.observer)
    if fault == 'generation':
        f.states[0].slot_generation += 1
    elif fault == 'wire':
        f.engine.tp_decoder.wires[id(f.engine.multi.streams[0])] += 1
    elif fault == 'projection':
        f.engine.multi.buf.pa = f.engine.multi.buf.pa.clone()
    elif fault == 'coordinates':
        f.engine.multi.buf.target_indexer_coordinates = f.engine.multi.buf.target_indexer_coordinates.clone()
    else:
        f.call()
    with pytest.raises(ValueError, match='current issued'):
        f.call()
    assert f.notices and f.witness.failed


@pytest.mark.parametrize('failure', ['candidate', 'bit'])
def test_authentic_primary_survives_evidence_notification_and_close_failure(monkeypatch, tmp_path, failure):
    f = fixture(monkeypatch, tmp_path, fault=failure)
    f.witness.install()
    f.witness.begin_round(f.observer)
    export, notification, cleanup = MemoryError('export'), MemoryError('notification'), OSError('close')
    def exporting(*args):
        raise export
    def notifying(*args):
        raise notification
    monkeypatch.setattr(f.witness, '_evidence', exporting)
    monkeypatch.setattr(owned, '_observer_failure', notifying)
    primary = f.primary if failure == 'candidate' else f.witness.mismatch
    with pytest.raises(BaseException) as caught:
        f.call()
    assert caught.value is primary
    assert f.witness.status[2] is notification
    if failure == 'bit':
        assert f.witness.status[1] is export
    f.witness.end_round(primary)
    def close():
        raise cleanup
    f.witness.original_close = close
    with pytest.raises(BaseException) as caught:
        f.engine.close()
    assert caught.value is primary
    assert f.witness.status[3] is cleanup and f.witness.views and f.witness.slabs
    assert f.module.owned_q8_prepare is f.witness


def test_mismatch_emits_only_initialized_owned_outputs_and_prestatus(monkeypatch, tmp_path):
    import json
    import numpy as np
    f = fixture(monkeypatch, tmp_path, fault='bit')
    f.states[0].cache.offsets = (-1,)
    f.states[0].kv_status.copy_(torch.tensor([8, 2], dtype=torch.int32))
    f.witness.begin_round(f.observer)
    with pytest.raises(RuntimeError) as caught:
        f.call()
    assert caught.value is f.witness.mismatch
    record = json.loads((f.witness.directory/'first-mismatch.json').read_text())
    assert record['owned'] is False and not record['P3_valid']
    assert 'actual_k' not in record['files'] and 'expected_k' in record['files']
    values = np.load(f.witness.directory/'initial_status.npy')
    assert values.tolist() == [8, 2]
    assert f.notices == [f.witness.mismatch]


def test_summary_publication_failure_retires_borrow_and_does_not_retry_close(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    f.witness.install()
    f.witness.completed = True
    primary = MemoryError('summary')
    monkeypatch.setattr(owned, 'write_json', lambda *args: (_ for _ in ()).throw(primary))
    with pytest.raises(MemoryError) as caught:
        f.engine.close()
    assert caught.value is primary
    assert not f.witness.views and not f.witness.slabs
    assert f.module.owned_q8_prepare is f.original and len(f.close_calls) == 1


def test_resource_bounds_are_one_round_no_extra_device_allocation(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    assert f.resources['maximum_calls'] == 288 and f.resources['maximum_rounds'] == 1
    assert f.resources['additional_device_bytes'] == 0
    assert f.resources['global_host_obligation_bytes'] >= 8*f.resources['pinned_tensor_payload_bytes']
    assert f.witness.device_borrow_bytes <= f.witness.workspace.numel()*4


def test_parent_flag_forwarding_default_off_and_conflict_before_launch(monkeypatch, tmp_path):
    from tools import launch_cuda_tp
    commands = []
    monkeypatch.setattr(launch_cuda_tp, 'supervise', lambda cmd, environments, **kw: commands.extend(cmd) or 0)
    base = ['--gpus', '0,1,2,3,4,5,6,7']
    assert owned.main([*base, '--out', str(tmp_path/'base'), '--', 'model']) == 0
    assert len(commands) == 8 and all('--q8-preparation-parity' not in row for row in commands)
    commands.clear()
    assert owned.main([*base, '--out', str(tmp_path/'diagnostic'), '--benchmark-plan', str(tmp_path/'plan.json'),
                       '--q8-preparation-parity', '--', 'model']) == 0
    assert len(commands) == 8 and all('--q8-preparation-parity' in row for row in commands)
    commands.clear()
    with pytest.raises(ValueError, match='excludes other captures'):
        owned.main([*base, '--out', str(tmp_path/'conflict'), '--benchmark-plan', str(tmp_path/'plan.json'),
                    '--q8-preparation-parity', '--gdn-batch-parity', '--', 'model'])
    assert not commands and not (tmp_path/'conflict').exists()


def test_absent_preconstructor_host_credit_refuses_before_observer_install(tmp_path):
    engine = NS(_owned_qualification_host=(0, 'wrong'))
    with pytest.raises(ValueError, match='original host admission'):
        owned.prepare_q8_preparation_parity(engine, {'total_global_host_obligation_bytes': 1,
                                                   'input_sha256': 'a'*64}, tmp_path)
    assert not hasattr(engine, '_q8_preparation_parity')


@pytest.mark.parametrize('fault', [False, True])
def test_actual_benchmark_observer_brackets_one_original_round_and_preserves_primary(monkeypatch, fault):
    events, marker = [], object()
    primary = RuntimeError('actual round primary')
    def round_():
        events.append('round')
        if fault:
            raise primary
        return marker
    observations = NS(target=lambda *a, **k: None, record=lambda **kwargs: {})
    decoder = NS(controller=NS(work_observations=observations),
                 admit=lambda *a: None, _prepare_pass=lambda *a: None, _joined=lambda *a: None,
                 _take=lambda *a: None, round=round_, finish=lambda *a: None)
    engine = NS(multi=decoder)
    witness = NS(begin_round=lambda observer: events.append('begin'),
                 end_round=lambda error=None: events.append(('end', error)), published={})
    observer = owned._BenchmarkObserver(engine, {'run_identity': 'a'*32, 'cohorts': []}, timed=False,
                                        preparation_witness=witness)
    monkeypatch.setattr(observer, '_observe', lambda: events.append('observe'))
    observer.install()
    if fault:
        with pytest.raises(RuntimeError) as caught:
            decoder.round()
        assert caught.value is primary
        assert events == ['begin', 'round', ('end', primary)]
    else:
        assert decoder.round() is marker
        assert events == ['begin', 'round', ('end', None), 'observe']


def test_active_shared_shadow_rejects_foreign_thread_before_oracle_or_candidate(monkeypatch, tmp_path):
    import threading
    f = fixture(monkeypatch, tmp_path)
    f.witness.begin_round(f.observer)
    failures = []
    def call():
        try:
            f.call()
        except BaseException as error:
            failures.append(error)
    thread = threading.Thread(target=call)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive() and failures == [f.witness.concurrent_error]
    assert not f.log and f.witness.failed
