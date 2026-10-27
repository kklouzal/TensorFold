"""Independent timing/count oracle and bounded caller failure/retirement controls."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from tensorfold.server.request_limits import RequestLimit
from tools import finite_concurrency as finite
from tools import v100_backend_probe as probe
from tests.test_v100_backend_probe import cli


def test_exact_grid_and_default_single_contract(tmp_path):
    args = probe.arguments(cli(tmp_path))
    assert not args.concurrency_grid
    assert probe.waiting_input_credit(args) == probe.INPUT_CREDIT
    assert finite.GRID == (1, 2, 4, 8, 16, 24)
    concurrent = probe.arguments([*cli(tmp_path), '--concurrency-grid', '--tokens', '512'])
    assert concurrent.concurrency_grid and concurrent.tokens == 512
    expected = 24 * (56 + (262144 + 1) * 8 + 262144 * 28)
    assert probe.waiting_input_credit(concurrent) == expected
    assert '120 numbered entries' in finite.SUSTAINED_PROMPT
    for extra in (['--concurrency-grid'], ['--concurrency-grid', '--tokens', '255'],
                  ['--concurrency-grid', '--tokens', '512', '--expect-prefix-reuse']):
        with pytest.raises(SystemExit):
            probe.arguments([*cli(tmp_path), *extra])


def test_timed_tokens_bound_event_cells_and_retire_without_draft_counts(monkeypatch):
    stamps = iter((100, 200, 300))
    monkeypatch.setattr(finite.time, 'monotonic_ns', lambda: next(stamps))
    abort = threading.Event()
    collector = finite.TimedTokens(3, 10, abort)
    assert collector([]) is False
    collector([1, 2])
    for invalid in ([3, 4], [True], [-1], [10], (3,)):
        with pytest.raises(ValueError):
            collector(invalid)
    assert collector.events == 1 and collector.ids == [1, 2]
    abort.set()
    assert collector([3]) is True
    assert collector.record() == {'token_ids': [1, 2, 3], 'chunk_monotonic_ns': [100, 200],
                                  'chunk_cumulative_committed_tokens': [2, 3]}
    assert len(collector.times) == len(collector.counts) == 3
    collector.close()
    with pytest.raises(RuntimeError, match='after retirement'):
        collector([])


def measured_row(times, counts):
    return {'complete': True, 'token_ids': list(range(counts[-1])),
            'chunk_monotonic_ns': times, 'chunk_cumulative_committed_tokens': counts,
            'offered_monotonic_ns': 50, 'finished_monotonic_ns': 400}


def test_independent_common_window_oracle_excludes_first_chunk_and_drafts():
    rows = [measured_row([100, 200, 300], [3, 5, 6]), measured_row([150, 250, 350], [2, 6, 8])]
    result = finite.metrics(rows, 40, 420)
    assert result['common_decode_start_monotonic_ns'] == 150
    assert result['common_decode_end_monotonic_ns'] == 300
    assert result['common_window_tokens_per_request'] == [3, 4]
    assert result['common_window_aggregate_tps'] == 7e9 / 150
    assert result['whole_wall_aggregate_committed_tps'] == 14e9 / 380
    assert rows[0]['timing']['delivery_decode_tps'] == 3e9 / 200
    assert rows[1]['timing']['delivery_decode_tps'] == 6e9 / 200
    assert result['peak_overlapping_delivery_decode_intervals'] == 2
    assert result['ttft_ns']['minimum'] == 50
    assert result['ttft_ns']['maximum'] == 100


def test_nonoverlap_and_no_output_cannot_become_full_concurrency_proof():
    rows = [measured_row([100, 200], [1, 2]), measured_row([300, 350], [1, 2])]
    result = finite.metrics(rows, 40, 420)
    assert result['common_decode_ns'] == 0
    assert result['common_window_aggregate_tps'] is None
    assert result['peak_overlapping_delivery_decode_intervals'] == 1
    rows[0]['complete'] = False
    assert finite.metrics(rows, 40, 420)['measurable'] is False


class Burst:
    """Substitute only engine boundary; real threaded RequestLimit scopes remain."""
    def __init__(self, count, *, tokens=256, failure=False, overlap=True):
        self.count, self.tokens, self.failure, self.overlap = count, tokens, failure, overlap
        self.lock = threading.Lock()
        self.active = set()
        self.calls = self.admitted = self.retired = 0
        self.observed = threading.Event()
        self.emit_barrier = threading.Barrier(count)
        self.callback_refs = []

    def observe(self):
        with self.lock:
            states = tuple((index, 'active' if self.overlap else 'waiting') for index in sorted(self.active))
            if len(self.active) == self.count:
                self.observed.set()
            return {'states': states, 'decoding': tuple(index for index, phase in states if phase == 'active'), 'outcomes': {'admitted': self.admitted, 'cancelled': 0,
                    'expired': 0, 'refused': 0}, 'charged_live_input_bytes': len(states) * 100}

    def invoke(self, callback, progress):
        with self.lock:
            index = self.calls
            self.calls += 1
            self.admitted += 1
            self.active.add(index)
            self.callback_refs.append(callback)
        try:
            assert self.observed.wait(2)
            if self.failure and index == 0:
                raise RuntimeError('primary generation failure')
            for start in range(0, self.tokens, 64):
                if not self.failure:
                    self.emit_barrier.wait(2)
                if callback([1] * min(64, self.tokens - start)) or progress(0, 0, 0, 0):
                    break
                time.sleep(.005)
            return {'text': 'bounded visible text', 'stats': {'drafted': 999999}}
        finally:
            with self.lock:
                self.active.remove(index)
                self.retired += 1


def run_burst(burst, *, minimum=256):
    records = []
    limit = RequestLimit(24)
    with ThreadPoolExecutor(max_workers=burst.count) as pool:
        try:
            result = finite.run_case(pool, count=burst.count, maximum_tokens=512, vocab=3,
                limit=limit, invoke=burst.invoke, observe=burst.observe, write=records.append,
                minimum_tokens=minimum, timeout=3)
        except BaseException as error:
            error.records = records
            error.limit = limit
            raise
    assert not limit.used and burst.retired == burst.count and not burst.active
    assert all(callback.closed for callback in burst.callback_refs)
    return result, records


@pytest.mark.parametrize('count', finite.GRID)
def test_actual_burst_scopes_overlap_and_all_siblings_retire(count):
    burst = Burst(count)
    summary, records = run_burst(burst)
    record = records[0]
    assert record['offered'] == record['logical_admitted'] == record['completed'] == count
    assert record['peak_observed_backend_active'] == count
    assert record['backend_outcome_delta']['admitted'] == count
    assert record['errors'] == record['cancel_requested'] == 0
    assert summary['metrics']['common_decode_ns'] > 0
    assert all(value > 0 for value in summary['metrics']['common_window_tokens_per_request'])
    assert sum(len(row['token_ids']) for row in record['requests']) == 256 * count
    assert all(row['stats']['drafted'] == 999999 for row in record['requests'])
    assert record['passed']


def test_primary_failure_preserves_raw_outcomes_and_drains_started_siblings():
    burst = Burst(4, failure=True)
    with pytest.raises(RuntimeError, match='primary generation failure') as caught:
        run_burst(burst)
    error = caught.value
    assert error.records[0]['passed'] is False
    assert not error.limit.used and burst.retired == 4 and not burst.active
    assert any(row.get('failure', {}).get('message') == 'primary generation failure'
               for row in error.records[0]['requests'])
    assert all(callback.closed for callback in burst.callback_refs)


@pytest.mark.parametrize('kind', ['natural-eos', 'no-admitted-overlap'])
def test_insufficient_output_or_only_offered_threads_fails_with_durable_receipt(kind):
    burst = Burst(2, tokens=64 if kind == 'natural-eos' else 256, overlap=kind != 'no-admitted-overlap')
    with pytest.raises(ValueError) as caught:
        run_burst(burst)
    assert caught.value.records[0]['passed'] is False
    assert not caught.value.limit.used and burst.retired == 2
    assert all(len(row['token_ids']) == burst.tokens for row in caught.value.records[0]['requests'])


def test_diagnostic_failure_does_not_replace_primary_and_success_can_fail_persistence():
    burst = Burst(1, tokens=64)
    limit = RequestLimit(24)
    def failed_write(_):
        raise OSError('receipt failure')
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match='measurement length') as caught:
            finite.run_case(pool, count=1, maximum_tokens=512, vocab=3, limit=limit,
                invoke=burst.invoke, observe=burst.observe, write=failed_write, minimum_tokens=256, timeout=3)
    assert isinstance(caught.value.__cause__, OSError)
    assert not limit.used and burst.retired == 1


def test_bounded_timeline_overflow_is_failed_measurement_not_silent_truncation(monkeypatch):
    monkeypatch.setattr(finite, 'TIMELINE_EVENTS', 0)
    burst = Burst(2, failure=True)
    with pytest.raises(ValueError, match='timeline exceeded') as caught:
        run_burst(burst)
    assert caught.value.records[0]['passed'] is False
    assert not caught.value.limit.used and burst.retired == burst.calls and not burst.active


def test_pool_prestarts_exact_bounded_workers_restores_stack_and_retires(monkeypatch):
    monkeypatch.setattr(finite, 'MAXIMUM', 2)
    before = threading.stack_size()
    pool = finite.owned_pool(2)
    try:
        assert len(pool._threads) == 2
        assert threading.stack_size() == before
        assert pool.submit(lambda: 7).result() == 7
    finally:
        threads = tuple(pool._threads)
        pool.shutdown(wait=True, cancel_futures=True)
    assert all(not thread.is_alive() for thread in threads)


def test_control_bound_is_declared_and_scales_with_fixed_timestamp_cells():
    assert finite.control_payload_bytes(512) > finite.control_payload_bytes(256) > 0
    for tokens in (0, 513, True):
        with pytest.raises(ValueError):
            finite.control_payload_bytes(tokens)


def test_pool_preparation_failure_retires_started_tasks_and_restores_stack(monkeypatch):
    monkeypatch.setattr(finite, 'MAXIMUM', 2)
    actual = finite.ThreadPoolExecutor
    pools = []
    class SubmitFailure(actual):
        def submit(self, *args, **kwargs):
            if pools and len(self._threads) == 1:
                raise MemoryError('second worker publication')
            pools.append(self)
            return super().submit(*args, **kwargs)
    monkeypatch.setattr(finite, 'ThreadPoolExecutor', SubmitFailure)
    before = threading.stack_size()
    with pytest.raises(MemoryError, match='second worker publication'):
        finite.owned_pool(1)
    assert threading.stack_size() == before
    assert pools and all(not thread.is_alive() for thread in pools[0]._threads)


def test_progress_false_continues_and_true_requests_cancellation():
    # The actual Scheduler's callback contract is True=>cancel. This independent
    # boundary checks both normal prefill and cancellation after sibling error.
    normal = Burst(2)
    _, records = run_burst(normal)
    assert all(row['prefill_progress_callbacks'] == 4 for row in records[0]['requests'])
    assert all(row['last_prefill_progress']['processed'] == 0 for row in records[0]['requests'])
    assert records[0]['cancel_requested'] == 0


def test_leader_declares_and_admits_all_owners_before_configure_and_worker_start(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import tensorfold.families.qwen4_exp.cuda.retention_resources as resources
    import tensorfold.token_metadata as metadata

    args = probe.arguments([*cli(tmp_path), '--concurrency-grid', '--tokens', '512'])
    calls = []
    class Pool:
        def shutdown(self, **kwargs):
            calls.append(('shutdown', kwargs))
    pool = Pool()
    scheduler = SimpleNamespace(configure_requests=lambda owner: calls.append(('configure', owner.maximum)))
    engine = SimpleNamespace(scheduler=scheduler, retention_catalog=object(), sm70_profile='source-packed-trial')
    def admit(catalog, *, tokenizer, scheduler, request_limit, caller):
        assert catalog is engine.retention_catalog and scheduler is engine.scheduler
        assert type(caller) is resources.FiniteBackendCaller and type(request_limit) is RequestLimit
        assert request_limit.maximum == caller.max_concurrent_requests == 24
        assert caller.retained_results == 24 and caller.worker_stack_bytes == finite.STACK_BYTES
        assert caller.control_payload_bytes == finite.control_payload_bytes(512)
        calls.append(('admit', caller))
        return {'verified_test_boundary_only': True}
    def make_pool(timeout):
        assert calls[-1] == ('configure', 24)
        calls.append(('pool', timeout))
        return pool
    def run(pool_arg, **kwargs):
        assert pool_arg is pool and kwargs['limit'].maximum == 24
        assert kwargs['maximum_tokens'] == 512
        assert kwargs['minimum_tokens'] in (1, 256)
        calls.append(('case', kwargs['count'], kwargs['minimum_tokens']))
        return {'passed': True}
    def warmup(engine, tokenizer, tokens, *, count, draft, limit, result):
        assert count == 1 and not limit.used
        calls.append(('warmup', draft))
        result['test_only'] = True
    monkeypatch.setattr(resources, 'finalize_backend_resources', admit)
    monkeypatch.setattr(probe, 'prepared_prompt', lambda *_: ([0], {'test_only': True}))
    monkeypatch.setattr(metadata, 'tokenizer_vocab_size', lambda _: 3)
    monkeypatch.setattr(finite, 'owned_pool', make_pool)
    monkeypatch.setattr(finite, 'run_case', run)
    monkeypatch.setattr(probe, 'generate_reply', warmup)
    monkeypatch.setattr(probe, 'write_json', lambda *_: None)
    monkeypatch.setattr(probe, 'write_report', lambda *_a, **_kw: None)
    report = {}
    probe.run_concurrency_leader(engine, object(), args, report)
    assert [item[0] for item in calls[:3]] == ['admit', 'configure', 'pool']
    assert [item[1] for item in calls if item[0] == 'case'] == list(finite.GRID) * 2
    assert report['concurrency']['completed_cases'] == 12
    assert calls[-1] == ('shutdown', {'wait': True, 'cancel_futures': True})


def test_interrupted_future_wait_does_not_retire_callback_before_worker_finishes():
    burst = Burst(1)
    limit, records = RequestLimit(24), []
    class InterruptedFuture:
        def __init__(self, actual):
            self.actual = actual
            self.once = False
        def result(self):
            if not self.once:
                self.once = True
                raise KeyboardInterrupt('injected interrupted caller wait')
            return self.actual.result()
        def done(self):
            return self.actual.done()
        def cancelled(self):
            return self.actual.cancelled()
        def cancel(self):
            return self.actual.cancel()
    class WrappedPool:
        def __init__(self, actual):
            self.actual = actual
        def submit(self, *args):
            return InterruptedFuture(self.actual.submit(*args))
    # Force coordinator cancellation while the invoked engine still owns the
    # callback, then interrupt its first retirement wait before worker completion.
    actual_observe = burst.observe
    def observe():
        record = actual_observe()
        if burst.active:
            raise ValueError('coordinator primary')
        return record
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match='coordinator primary'):
            finite.run_case(WrappedPool(pool), count=1, maximum_tokens=512, vocab=3,
                limit=limit, invoke=burst.invoke, observe=observe, write=records.append,
                minimum_tokens=256, timeout=3)
    assert burst.retired == 1 and not burst.active and not limit.used
    assert burst.callback_refs[0].closed


def test_correlated_decode_requires_live_decode_inside_committed_window():
    rows = [measured_row([100, 200, 300], [1, 2, 3]), measured_row([150, 250, 350], [1, 2, 3])]
    measurement = finite.metrics(rows, 40, 420)
    assert finite.correlated_overlap(rows, measurement, [[1, 90, 5]]) is None
    assert finite.correlated_overlap(rows, measurement, [[360, 410, 5]]) is None
    assert finite.correlated_overlap(rows, measurement, [[150, 300, 1]]) is None
    overlap = finite.correlated_overlap(rows, measurement, [[140, 310, 5]])
    assert overlap['start_monotonic_ns'] == 150 and overlap['end_monotonic_ns'] == 300
    assert overlap['committed_tokens_per_request'] == [2, 1]
    assert overlap['aggregate_committed_tps'] == 3e9 / 150


def test_decoding_identity_excludes_prefill_done_waiting_and_postmodel_delivery():
    from types import SimpleNamespace
    rows = [SimpleNamespace(sid=index, done=False, waiting=False, started=1,
                            committed_output_tokens=1) for index in range(6)]
    rows[1].committed_output_tokens = 0
    rows[2].done = True
    rows[3].waiting = True
    rows[4].started = 0
    streams = {row.sid: row for row in rows[:5]}
    entries = tuple((0, 100 + row.sid, (row, None)) for row in rows)
    assert finite.decoding_tickets(entries, streams) == (100,)
    streams[0] = SimpleNamespace(sid=0)
    assert not finite.decoding_tickets(entries, streams)


def test_error_rendering_failure_latches_cancellation_and_preserves_original():
    class UnprintableError(RuntimeError):
        def __str__(self):
            raise MemoryError('injected error formatting exhaustion')
    original = UnprintableError()
    burst = Burst(2, failure=True)
    actual = burst.invoke
    def invoke(callback, progress):
        try:
            return actual(callback, progress)
        except RuntimeError:
            raise original
    burst.invoke = invoke
    with pytest.raises(UnprintableError) as caught:
        run_burst(burst)
    assert caught.value is original
    assert isinstance(caught.value.__cause__, MemoryError)
    assert burst.retired == 2 and not burst.active
    assert any(caught.value.records[0]['diagnostic_failure_types'])


def test_explicit24_ceiling_is_optional_and_never_silently_reduces_grid_floor(tmp_path):
    assert probe.arguments(cli(tmp_path)).session_ceiling is None
    selected = probe.arguments([*cli(tmp_path), '--session-ceiling', '24', '--concurrency-grid', '--tokens', '320'])
    assert selected.session_ceiling == 24 and finite.GRID[-1] == 24
    for value in ('1', '23', '25', '0', '-1', '24.5'):
        with pytest.raises(SystemExit):
            probe.arguments([*cli(tmp_path), '--session-ceiling', value])


def test_explicit24_region_requires_actual24_materialized_slots():
    from types import SimpleNamespace as NS
    plan = {'session_capacity': {'context_tokens': 262144, 'configured_ceiling': 24,
                'resident_capacity': 24, 'full_context_commitments': 24},
            'kv_full_windows': {'materialized': True}}
    engine = NS(tp=8, depth=2, confidence=.7, context_window=262144, indexer_history='gpu',
        kv_pair=NS(key_dtype='int8', value_dtype='int8'), scheduler=NS(_request_limit=None),
        retention_catalog=object(), capacity_plan=plan,
        w=NS(mtp=object(), meta={'execution_provider': 'sm70', 'execution_dtype': 'float16',
                               'full': NS(hidden=2560, layers=48, vocab=248320)}))
    probe.check_engine(engine, session_ceiling=24)
    for resident in (23, 25, 26):
        plan['session_capacity']['resident_capacity'] = resident
        with pytest.raises(ValueError):
            probe.check_engine(engine, session_ceiling=24)
    plan['session_capacity'].update(configured_ceiling=None, resident_capacity=26, full_context_commitments=26)
    probe.check_engine(engine)
    with pytest.raises(ValueError):
        probe.check_engine(engine, session_ceiling=24)


def test_common_funding_and_per_case_graph_counters_exclude_bringup_and_warmup(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from tensorfold.families.qwen4_exp.cuda import retention_resources as resources
    from tensorfold import token_metadata

    args = probe.arguments([*cli(tmp_path), '--session-ceiling', '24', '--concurrency-grid', '--tokens', '320'])
    counter = NS(hits=123)
    engine = NS(scheduler=NS(configure_requests=lambda _: None), retention_catalog=object(),
                multi=NS(graphs=NS(bound=counter, verify={}, mtp={})), sm70_profile='source-packed-trial')
    receipts, admission = [], []
    def finalize(_catalog, *, caller, **_kwargs):
        assert caller.max_concurrent_requests == caller.retained_results == 24
        assert probe.MAX_RECEIPT_BYTES == 1507328
        assert caller.max_receipt_bytes == 25 * 1507328 == 37683200
        admission.append(caller)
        return {'test_only': True}
    class Pool:
        def shutdown(self, **_kwargs):
            pass
    def warmup(*_args, result, **_kwargs):
        counter.hits += 1000
        result['test_only'] = True
    def burst(_pool, *, count, write, **_kwargs):
        counter.hits += count * 10
        write({'offered': count, 'passed': True})
        return {'passed': True}
    def record(path, value, **_kwargs):
        if '-n' in path.name:
            receipts.append((path.name, value))
    monkeypatch.setattr(resources, 'finalize_backend_resources', finalize)
    monkeypatch.setattr(probe, 'prepared_prompt', lambda *_: ([0], {'test_only': True}))
    monkeypatch.setattr(token_metadata, 'tokenizer_vocab_size', lambda _: 3)
    monkeypatch.setattr(finite, 'owned_pool', lambda _: Pool())
    monkeypatch.setattr(finite, 'run_case', burst)
    monkeypatch.setattr(probe, 'generate_reply', warmup)
    monkeypatch.setattr(probe, 'write_json', lambda *_: None)
    monkeypatch.setattr(probe, 'write_report', record)
    report = {}
    probe.run_concurrency_leader(engine, object(), args, report)
    assert len(admission) == 1 and len(receipts) == 12
    bringup, sustained = receipts[:6], receipts[6:]
    assert all('bringup' in name and 'graph_execution' not in row for name, row in bringup)
    assert [row['graph_execution']['delta']['bound.hits'] for _, row in sustained] == [10 * n for n in finite.GRID]
    assert sustained[0][1]['graph_execution']['before_requests']['counters']['bound.hits'] == 123 + 2000 + 550
    assert all('before optional quality' in row['graph_execution']['scope'] for _, row in sustained)
    assert report['concurrency']['completed_cases'] == 12
