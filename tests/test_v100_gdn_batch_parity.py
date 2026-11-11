"""Private diagnostic admission/order/failure/lifetime controls; no math qualification."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from tools import v100_owned_qualification as owned


def plan():
    return {'cohorts': [{'id': 6, 'level': 24, 'warmup': False, 'input_tokens': 64, 'tokens': 2}]}


def fixture(tmp_path, monkeypatch, fault=False, engine_type=NS):
    monkeypatch.setattr(owned, '_gdn_witness_region', lambda engine: engine.multi)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: NS(synchronize=lambda: None))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    failures, order = [], []
    monkeypatch.setattr(owned, '_observer_failure', lambda engine, error: failures.append(error))
    monkeypatch.setattr(owned, '_history_resources', lambda capacity, limit: {
        'per_rank_payload_bytes': 4096+1024*capacity, 'global_host_obligation_bytes': 8*(4096+1024*capacity),
        'receipt_bytes_limit': 4096+1024*capacity})
    resource = owned._gdn_batch_resources(plan(), tmp_path)
    def cpu(shape, **kw):
        return torch.empty(shape, dtype=kw['dtype'])
    def scratch():
        return NS(k=torch.zeros(1, 2, 128), v=torch.zeros(1, 6, 128, dtype=torch.float16),
                  g=torch.zeros(1, 6), b=torch.zeros(1, 6))
    states = [NS(rec=torch.full((2, 1, 6, 128, 128), float(i)), conv=torch.zeros(1, 3, 1280),
                 cur=[i%2], scratch=[scratch()], history_slot=i, slot_generation=2,
                 history_generation=3, pos=64) for i in range(24)]
    shared = NS(k=torch.empty(6*128*128), q=torch.empty(280),
                v=torch.empty(1536, dtype=torch.float16), g=torch.empty(6), beta=torch.empty(6), lin=1)
    dec = NS(gdn=shared, buf=object(), held={}, states=states)
    engine = engine_type(rank=0, w=NS(device=torch.device('cpu')), multi=dec, _closed=False, _abort_comm=False)
    def close():
        order.append('owner-close')
        engine._closed = True
        return 'closed-owner'
    engine.close = close
    def write(p, initial, sc, final, out, sums):
        final.copy_(initial + p[0, 0])
        for name in ('k', 'v', 'g', 'b'):
            getattr(sc, name).fill_(p[0, 0])
        out.fill_(p[0, 0])
        sums.fill_(p[0, 0])
    def scalar(p, conv, weight, initial, decay, bias, norm, eps, rows, sc, final, out, sums, **kw):
        order.append('scalar')
        write(p, initial, sc, final, out, sums)
    sentinel = object()
    def batch(p, weight, decay, bias, norm, eps, tables, layer, out, sums, **kw):
        order.append('batch')
        for state, a, z in tables.segs:
            write(p[a:z], state.rec[state.cur[0], 0], state.scratch[0],
                  state.rec[1-state.cur[0], 0], out[a:z], sums[a:z])
        if fault:
            out.view(torch.int16)[0, 0] ^= 1
        return sentinel
    module = NS(batch_chain=batch, chain=scalar, __file__=owned.__file__)
    witness = owned._GDNBatchParity(engine, resource, tmp_path, module=module, allocator=cpu)
    streams = [NS(st=s) for s in states]
    dec.streams = {i: stream for i, stream in enumerate(streams)}
    engine.tp_decoder = NS(wires={id(stream): i for i, stream in enumerate(streams)})
    witness.observer = NS(finished=True, _identity=lambda stream: ({'id': 6},
        {'index': stream.st.history_slot, 'prompt_sha256': 'a'*64}))
    witness.install()
    tables = NS(segs=[(s, i, i+1) for i, s in enumerate(states)], rows=24, buffers=dec.buf,
                static=NS(i64=torch.zeros(8*24, dtype=torch.int64)), starts=torch.arange(25, dtype=torch.int32))
    args = (torch.ones(24, 2060, dtype=torch.float16), None, None, None, None, 1e-6,
            tables, 0, torch.zeros(24, 768, dtype=torch.float16), torch.zeros(24, 24))
    return engine, witness, module, args, states, order, failures, sentinel, batch


def test_same_live_inputs_scalar_first_actual_return_and_retirement(tmp_path, monkeypatch):
    engine, witness, module, args, states, order, failures, sentinel, original = fixture(tmp_path, monkeypatch)
    inputs = [s.rec[s.cur[0], 0].clone() for s in states]
    assert module.batch_chain(*args) is sentinel
    assert order == ['scalar']*24+['batch'] and not failures
    for s, value in zip(states, inputs):
        assert torch.equal(s.rec[s.cur[0], 0], value)
    assert witness.calls == 1 and witness.counts[0] == {'calls': 1, 'rows': 24, 'matched': 24}
    saved = module.batch_chain
    assert engine.close() == 'closed-owner' and module.batch_chain is original
    assert witness.closed and not witness.slabs and witness.engine is None
    with pytest.raises(RuntimeError, match='retired'):
        saved(*args)
    summary = json.loads((witness.directory/'summary.json').read_text())
    assert summary['closed'] and not summary['failed'] and summary['P3_valid'] is False


def test_wrong_bit_keeps_preinputs_and_actual_results_before_abort(tmp_path, monkeypatch):
    _, witness, module, args, _, order, failures, _, _ = fixture(tmp_path, monkeypatch, fault=True)
    with pytest.raises(RuntimeError, match='raw-bit mismatch') as raised:
        module.batch_chain(*args)
    assert order == ['scalar']*24+['batch'] and failures == [raised.value]
    proof = json.loads((witness.directory/'mismatch.json').read_text())
    assert proof['rows'] == 24 and len(proof['files']) == 23
    assert (witness.directory/'pre_rec.npy').exists() and (witness.directory/'actual_out.npy').exists()
    assert witness.failed and witness.slabs


def test_mismatch_capture_failure_preserves_numeric_primary_and_always_notifies(tmp_path, monkeypatch):
    _, witness, module, args, _, _, failures, _, _ = fixture(tmp_path, monkeypatch, fault=True)
    error = OSError('evidence sink failed')
    witness.writer = lambda *a: (_ for _ in ()).throw(error)
    with pytest.raises(RuntimeError, match='raw-bit mismatch') as raised:
        module.batch_chain(*args)
    assert failures == [raised.value] and raised.value.__cause__ is error


def test_candidate_exception_and_close_failure_keep_owners(tmp_path, monkeypatch):
    engine, witness, module, args, _, _, failures, _, _ = fixture(tmp_path, monkeypatch)
    primary = OSError('candidate failure')
    witness.original = lambda *a, **k: (_ for _ in ()).throw(primary)
    with pytest.raises(OSError) as raised:
        module.batch_chain(*args)
    assert raised.value is primary and failures == [primary]
    close_error = OSError('retirement uncertain')
    witness.original_close = lambda: (_ for _ in ()).throw(close_error)
    with pytest.raises(OSError) as raised:
        engine.close()
    assert raised.value is close_error and witness.slabs and witness.engine is engine and module.batch_chain is witness


def test_work_bounds_refuse_without_calling_operator(tmp_path, monkeypatch):
    _, witness, module, args, _, order, failures, _, _ = fixture(tmp_path, monkeypatch)
    witness.resources['maximum_calls'] = 0
    with pytest.raises(ValueError, match='admitted live'):
        module.batch_chain(*args)
    assert not order and failures


def test_segment_journal_uses_ordered_benchmark_authority_and_never_drops_overflow(tmp_path, monkeypatch):
    _, witness, _, args, states, _, _, _, _ = fixture(tmp_path, monkeypatch)
    streams = [NS(st=s) for s in states]
    witness.engine.tp_decoder = NS(wires={id(s): i+100 for i, s in enumerate(streams)})
    witness.decoder.streams = {i: s for i, s in enumerate(streams)}
    witness.observer = NS(_identity=lambda stream: ({'id': 6}, {'index': stream.st.history_slot,
                                                               'prompt_sha256': 'a'*64}), finished=True)
    witness.segment('target-decode', list(reversed(args[6].segs)))
    rows = witness.segments[0]['rows']
    assert rows[0][:4] == [123, 6, 23, 23] and rows[-1][:4] == [100, 6, 0, 0]
    witness.resources['maximum_segment_rows'] = 24
    with pytest.raises(ValueError, match='exceeds'):
        witness.segment('target-decode', args[6].segs)


def test_capture_unsupported_image_and_C24_plan_refuse_early(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='C24'):
        owned._gdn_batch_resources({'cohorts': [{'level': 1, 'warmup': False}]}, tmp_path)


def test_private_flag_exclusions_before_effects(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(tmp_path/'out'), '--gdn-batch-parity'])
    assert not (tmp_path/'out').exists()


def test_numeric_primary_survives_simultaneous_capture_and_publication_exhaustion(tmp_path, monkeypatch):
    from tests.test_flashnext_t47_lifecycle_source import roots
    _, witness, module, args, _, _, failures, _, _ = fixture(tmp_path, monkeypatch, fault=True)
    capture, publish = OSError('capture'), MemoryError('publication')
    witness.writer = lambda *a: (_ for _ in ()).throw(capture)
    witness._publish = lambda: (_ for _ in ()).throw(publish)
    with pytest.raises(RuntimeError, match='raw-bit mismatch') as raised:
        module.batch_chain(*args)
    assert failures == [raised.value]
    assert id(capture) in roots(raised.value) and id(publish) in roots(raised.value)


def test_parent_forwards_only_explicit_diagnostic_option(tmp_path, monkeypatch):
    from tools import launch_cuda_tp as launcher
    commands = []
    monkeypatch.setattr(launcher, 'supervise', lambda cmd, environments, **kw: commands.extend(cmd) or 0)
    assert owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(tmp_path/'normal'), '--', 'model']) == 0
    assert len(commands) == 8 and all('--gdn-batch-parity' not in c for c in commands)
    commands.clear()
    assert owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(tmp_path/'diagnostic'),
                       '--benchmark-plan', str(tmp_path/'immutable-plan.json'), '--gdn-batch-parity', '--', 'model']) == 0
    assert len(commands) == 8 and all('--gdn-batch-parity' in c for c in commands)


def test_missing_original_host_obligation_refuses_before_install(tmp_path):
    engine = NS(_owned_qualification_host=(0, 'wrong'))
    resources = {'total_global_host_obligation_bytes': 1, 'input_sha256': 'a'*64}
    with pytest.raises(ValueError, match='original admission'):
        owned.prepare_gdn_batch_parity(engine, resources, tmp_path)
    assert not hasattr(engine, '_gdn_batch_parity')


def test_resource_payload_and_plan_work_are_explicit_and_not_materialized_credit(tmp_path, monkeypatch):
    monkeypatch.setattr(owned, '_history_resources', lambda c, n: {'per_rank_payload_bytes': 100,
        'global_host_obligation_bytes': 800, 'receipt_bytes_limit': c*1024})
    values = owned._gdn_batch_resources(plan(), tmp_path)
    expected = sum(__import__('math').prod(shape)*width for shape, width in values['slabs'].values())
    assert values['pinned_tensor_payload_bytes'] == expected
    assert values['per_rank_payload_bytes'] > expected and values['global_host_obligation_bytes'] == 8*values['per_rank_payload_bytes']
    assert values['maximum_calls'] == 48*24*2 and values['additional_device_bytes'] == 0
    assert values['P3_valid'] is False


def test_candidate_cannot_silently_modify_readonly_recurrent_input(tmp_path, monkeypatch):
    _, witness, module, args, states, _, failures, _, original = fixture(tmp_path, monkeypatch)
    def corrupt(*a, **kw):
        result = original(*a, **kw)
        state = states[0]
        state.rec[state.cur[0], 0].view(torch.int32)[0, 0, 0] ^= 1
        return result
    witness.original = corrupt
    with pytest.raises(RuntimeError, match='input_rec') as raised:
        module.batch_chain(*args)
    assert failures == [raised.value]
    proof = json.loads((witness.directory/'mismatch.json').read_text())
    assert 'pre_rec' in proof['files'] and 'actual_input_rec' in proof['files']


@pytest.mark.parametrize('method,parallel', [('mtp', '24'), ('none', 'auto'), ('none', '1')])
def test_non_none_or_non_C24_intent_refuses_before_constructor(method, parallel):
    with pytest.raises(ValueError, match='before construction'):
        owned._gdn_batch_preflight(NS(draft_method=method, parallel=parallel), NS(_parallel=int))


def test_exact_C24_none_preflight_has_no_constructor_effect():
    assert owned._gdn_batch_preflight(NS(draft_method='none', parallel='24'), NS(_parallel=int)) is None



def test_uncertain_close_preserves_primary_when_failure_notification_also_fails(tmp_path, monkeypatch):
    engine, witness, module, _, _, _, _, _, _ = fixture(tmp_path, monkeypatch)
    primary, notification = OSError('uncertain native retirement'), MemoryError('failure notifier')
    witness.original_close = lambda: (_ for _ in ()).throw(primary)
    monkeypatch.setattr(owned, '_observer_failure', lambda *a: (_ for _ in ()).throw(notification))
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is primary and primary.__cause__ is notification
    assert witness.slabs and witness.engine is engine and not witness.closed
    assert module.batch_chain is witness



def test_close_binding_restore_precedes_later_summary_failure(tmp_path, monkeypatch):
    from tests.test_flashnext_t47_lifecycle_source import roots
    restore, summary = OSError('restore close binding'), MemoryError('later summary publication')
    class Engine(NS):
        def __setattr__(self, name, value):
            if (name == 'close' and getattr(self, 'refuse_restore', False)
                    and value is getattr(self, 'restore_target', None)):
                raise restore
            return super().__setattr__(name, value)
    engine, witness, module, args, _, _, failures, _, original = fixture(tmp_path, monkeypatch, engine_type=Engine)
    module.batch_chain(*args)
    engine.restore_target, engine.refuse_restore = witness.old_close, True
    actual_write = owned.write_json
    def fail_summary(path, value):
        if path.name == 'summary.json':
            raise summary
        return actual_write(path, value)
    monkeypatch.setattr(owned, 'write_json', fail_summary)
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is restore and id(summary) in roots(restore)
    assert failures[-1] is restore and engine._closed and witness.closed
    assert not witness.slabs and witness.engine is None and module.batch_chain is original


def prefill_case(witness, states, *, count=2):
    class Stream:
        pass
    streams = []
    for i, state in enumerate(states[:count]):
        stream = Stream()
        stream.st, stream.sid, stream.prompt, stream.count = state, i, [i+10]+[3]*63, 2
        streams.append(stream)
    witness.engine.tp_decoder = NS(wires={id(s): i+300 for i, s in enumerate(streams)})
    witness.decoder.streams = {}  # Production's prefill-before-_joined interval.
    witness.observer = NS(finished=True, _identity=lambda s: ({'id': 6},
        {'index': s.sid, 'prompt_sha256': owned._prompt_digest(s.prompt)}))
    pieces = [(s, 0, 64) for s in streams]
    segments = [(s.st, i*64, (i+1)*64) for i, s in enumerate(streams)]
    return streams, pieces, segments


@pytest.mark.parametrize('incomplete', [False, True])
def test_actual_benchmark_prepare_target_before_join_uses_retained_authority(tmp_path, monkeypatch, incomplete):
    engine, witness, _, _, states, _, _, _, _ = fixture(tmp_path, monkeypatch)
    streams, pieces, segments = prefill_case(witness, states)
    events = []
    observations = NS(target=lambda segs, **kw: events.append('target-before-join') or 'target-result',
                      record=lambda **kw: {}, incomplete=lambda primary: events.append('incomplete'))
    dec = witness.decoder
    dec.controller = NS(work_observations=observations)
    dec.admit, dec._take, dec.round, dec.finish = (lambda *a: None for _ in range(4))
    def prepare(actual):
        assert actual is pieces and not dec.streams
        events.append('prepare')
        return [], segments, [], []
    def joined(actual, *args):
        assert witness.prefill_pending is None and not dec.streams
        events.append('join')
        dec.streams.update({s.sid: s for s, _, _ in actual})
        return 'joined-result'
    dec._prepare_pass, dec._joined = prepare, joined
    cohort = {'id': 6, 'level': 2, 'input_tokens': 64, 'tokens': 2, 'warmup': False,
              'requests': [{'index': s.sid, 'prefix': s.prompt[:1],
                            'prompt_sha256': owned._prompt_digest(s.prompt)} for s in streams]}
    observer = owned._BenchmarkObserver(engine, {'run_identity': 'a'*32, 'cohorts': [cohort]},
                                        timed=False, strict=True, segment_journal=witness)
    witness.observer = observer
    observer.install()
    assert dec._prepare_pass(pieces)[1] is segments
    assert witness.prefill_pending is not None and not dec.streams
    if incomplete:
        observations.incomplete(OSError('compute failed'))
        assert witness.prefill_pending is None and not dec.streams
        return
    states[0].pos = 128  # A mutable position is observation, not prepare authority.
    assert observations.target(segments, prefill=True) == 'target-result'
    assert not dec.streams and witness.prefill_pending is None
    assert witness.segments[1]['rows'][0][6] == 128 and witness.segments[1]['rows'][0][9] == 0
    assert dec._joined(pieces) == 'joined-result'
    assert events == ['prepare', 'target-before-join', 'join']


@pytest.mark.parametrize('changed', ['slot-generation', 'history-generation', 'stream-state',
                                     'wire', 'request', 'order', 'span'])
def test_prefill_reused_or_changed_authority_refuses_and_releases(tmp_path, monkeypatch, changed):
    _, witness, _, _, states, _, _, _, _ = fixture(tmp_path, monkeypatch)
    streams, pieces, segments = prefill_case(witness, states)
    witness.segment('prepare-prefill', segments, pieces)
    if changed == 'slot-generation':
        states[0].slot_generation += 1
    elif changed == 'history-generation':
        states[0].history_generation += 1
    elif changed == 'stream-state':
        streams[0].st = states[3]
    elif changed == 'wire':
        witness.engine.tp_decoder.wires[id(streams[0])] += 1
    elif changed == 'request':
        witness.observer._identity = lambda s: ({'id': 6}, {'index': 99, 'prompt_sha256': 'b'*64})
    elif changed == 'order':
        segments = list(reversed(segments))
    else:
        state, begin, end = segments[0]
        segments = [(state, begin, end-1), segments[1]]
    with pytest.raises(ValueError, match='changed|authority'):
        witness.segment('target-prefill', segments)
    assert witness.prefill_pending is None and witness.segment_count == 1
    with pytest.raises(ValueError, match='no matching'):
        witness.segment('target-prefill', segments)


def test_prefill_authority_is_strong_until_target_then_released(tmp_path, monkeypatch):
    import gc
    import weakref
    _, witness, _, _, states, _, _, _, _ = fixture(tmp_path, monkeypatch)
    streams, pieces, segments = prefill_case(witness, states)
    ref = weakref.ref(streams[0])
    witness.segment('prepare-prefill', segments, pieces)
    del streams, pieces
    gc.disable()
    try:
        assert ref() is not None
        witness.segment('target-prefill', segments)
        assert ref() is None and witness.prefill_pending is None
    finally:
        gc.enable()


def test_pending_prefill_reprepare_join_and_overflow_never_silently_drop(tmp_path, monkeypatch):
    _, witness, _, _, states, _, _, _, _ = fixture(tmp_path, monkeypatch)
    _, pieces, segments = prefill_case(witness, states)
    witness.segment('prepare-prefill', segments, pieces)
    with pytest.raises(ValueError, match='unconsumed'):
        witness.segment('prepare-prefill', segments, pieces)
    assert witness.prefill_pending is None
    witness.segment('prepare-prefill', segments, pieces)
    with pytest.raises(ValueError, match='joined before'):
        witness.prefill_joined(pieces)
    assert witness.prefill_pending is None
    witness.resources['maximum_segment_rows'] = witness.segment_rows+1
    with pytest.raises(ValueError, match='exceeds'):
        witness.segment('prepare-prefill', segments, pieces)
    assert witness.prefill_pending is None


def test_close_keeps_prefill_authority_until_callback_retirement(tmp_path, monkeypatch):
    engine, witness, module, args, states, _, _, _, _ = fixture(tmp_path, monkeypatch)
    module.batch_chain(*args)  # Satisfy the separate all-layer/C24 coverage gate.
    _, pieces, segments = prefill_case(witness, states)
    witness.segment('prepare-prefill', segments, pieces)
    def drain_then_close():
        assert witness.prefill_pending is not None
        witness.segment('target-prefill', segments)  # Accepted callback still completes during drain.
        engine._closed = True
        return 'drained'
    witness.original_close = drain_then_close
    assert engine.close() == 'drained' and witness.prefill_pending is None and witness.closed


def test_uncertain_close_retains_pending_prefill_owner_binding(tmp_path, monkeypatch):
    engine, witness, _, _, states, _, _, _, _ = fixture(tmp_path, monkeypatch)
    _, pieces, segments = prefill_case(witness, states)
    witness.segment('prepare-prefill', segments, pieces)
    pending = witness.prefill_pending
    error = OSError('drain not proved')
    witness.original_close = lambda: (_ for _ in ()).throw(error)
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is error and witness.prefill_pending is pending
    assert witness.slabs and witness.engine is engine and not witness.closed
