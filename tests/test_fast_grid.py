"""Independent timer/cancellation/ownership controls for the operational screen."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import threading
import time

import pytest

from tensorfold.server.request_limits import RequestLimit
from tools import finite_concurrency as finite
from tools import v100_backend_probe as probe
from tests.test_v100_backend_probe import cli


class ScreenBackend:
    def __init__(self, count, *, chunks=512, delay=.003, overlap=True, emit=True, failure=False):
        self.count, self.chunks, self.delay = count, chunks, delay
        self.overlap, self.emit, self.failure = overlap, emit, failure
        self.lock, self.active, self.decoding = threading.Lock(), set(), set()
        self.calls = self.admitted = self.retired = 0
        self.callbacks = []
        self.started = threading.Event()
        self.last_stats = None
    def observe(self):
        with self.lock:
            if len(self.active) == self.count:
                self.started.set()
            return {'states': tuple((index, 'active') for index in sorted(self.active)),
                'decoding': tuple(sorted(self.decoding)) if self.overlap else (),
                'outcomes': {'admitted': self.admitted, 'cancelled': 0, 'expired': 0, 'refused': 0},
                'backend_live': len(self.active), 'backend_retiring': 0, 'backend_mailboxes': len(self.active)}
    def invoke(self, callback, progress):
        with self.lock:
            index = self.calls
            self.calls += 1
            self.admitted += 1
            self.active.add(index)
            self.callbacks.append(callback)
        try:
            assert self.started.wait(2)
            if self.failure and index == 0:
                raise RuntimeError('original fast generation failure')
            for _ in range(self.chunks):
                if self.emit:
                    with self.lock:
                        self.decoding.add(index)
                    if callback([1]):
                        # Simulate one already in-flight committed chunk. It is
                        # real raw evidence, not part of the frozen window TPS.
                        time.sleep(.004)
                        callback([1])
                        break
                elif progress(100, 0, 0, 1):
                    break
                time.sleep(self.delay)
            return {'stats': {'cached': 84, 'drafted': 99999, 'committed_output_tokens': len(callback.ids)}, 'text': 'partial screen text',
                    'natural_stop': 'natural_eos' if self.chunks < 512 else 'output_cap'}
        finally:
            with self.lock:
                self.active.remove(index)
                self.decoding.discard(index)
                self.retired += 1


def screen(backend, *, ready=.15, window=.04, writer=None):
    rows, limit = [], RequestLimit(24)
    write = rows.append if writer is None else writer
    with ThreadPoolExecutor(max_workers=backend.count) as pool:
        try:
            summary = finite.run_screen_case(pool, count=backend.count, maximum_tokens=512, vocab=3,
                limit=limit, invoke=backend.invoke, observe=backend.observe, write=write,
                timeout=3, readiness_seconds=ready, decode_seconds=window)
        except BaseException as error:
            error.records, error.limit = rows, limit
            raise
    assert not limit.used and not backend.active and backend.retired == backend.calls
    assert all(callback.closed for callback in backend.callbacks)
    return summary, rows[0]


@pytest.mark.parametrize('count', finite.GRID)
def test_fixed_window_counts_committed_tokens_and_drains_every_requested_worker(count):
    backend = ScreenBackend(count)
    summary, row = screen(backend)
    assert summary['passed'] and row['screen']['measurement_valid']
    assert row['screen']['stop_reason'] == 'decode_window_elapsed'
    assert row['screen']['duration_ns'] == 40000000
    assert row['completed'] == 0 and row['intentionally_cancelled'] == count
    assert all(not item['complete'] and item['backend_retired'] and not item['quality_assessed'] for item in row['requests'])
    assert row['screen']['all_workers_permits_backend_owners_drained']
    assert row['screen']['cancel_requested_monotonic_ns'] >= row['screen']['end_monotonic_ns']
    assert row['screen']['cancel_and_drain_ns'] > 0 and not row['screen']['full_qualification']
    assert all(item['committed_tokens_after_window'] > 0 for item in row['requests'])
    begin, end = row['screen']['begin_monotonic_ns'], row['screen']['end_monotonic_ns']
    # Independent exact event integration: all chunks are one actual committed ID.
    expected = [sum(begin < at <= end for at in item['chunk_monotonic_ns']) for item in row['requests']]
    assert expected == [item['measured_window_committed_tokens'] for item in row['requests']]
    assert row['screen']['window_aggregate_committed_tps'] == sum(expected) / .04
    assert row['screen']['correlated_all_decode_interval']['duration_ns'] > 0


def test_natural_eos_makes_short_valid_measurement_without_claiming_full_window():
    summary, row = screen(ScreenBackend(2, chunks=24), window=.4)
    assert summary['passed'] and row['screen']['short_window']
    assert 0 < row['screen']['duration_ns'] < 400000000
    assert row['completed'] >= 1 and row['completed'] + row['intentionally_cancelled'] == 2
    assert all(item['natural_stop'] == 'natural_eos' for item in row['requests'])
    assert not row['screen']['quality_assessed'] and not row['screen']['full_qualification']


@pytest.mark.parametrize('kind', ['zero', 'no-overlap'])
def test_readiness_or_zero_output_is_honest_invalid_screen_and_still_drained(kind):
    backend = ScreenBackend(2, emit=kind != 'zero', overlap=kind != 'no-overlap')
    summary, row = screen(backend, ready=.04)
    assert not summary['passed'] and not row['screen']['measurement_valid']
    assert row['screen']['stop_reason'] == 'readiness_timeout'
    assert row['screen']['duration_ns'] == 0 and row['screen']['window_aggregate_committed_tps'] is None
    assert row['screen']['all_workers_permits_backend_owners_drained']
    assert row['intentionally_cancelled'] == 2 and row['completed'] == 0
    if kind == 'zero':
        assert all(not item['token_ids'] for item in row['requests'])


def test_runtime_error_remains_primary_and_all_siblings_retire_before_failure():
    backend = ScreenBackend(4, failure=True)
    with pytest.raises(RuntimeError, match='original fast generation failure') as caught:
        screen(backend)
    assert backend.retired == 4 and not backend.active and not caught.value.limit.used
    assert caught.value.records[0]['errors'] >= 1 and not caught.value.records[0]['passed']


def test_report_failure_cannot_replace_existing_runtime_primary():
    backend = ScreenBackend(1, failure=True)
    def fail_write(_):
        raise OSError('fast report failed')
    with pytest.raises(RuntimeError, match='original fast generation failure') as caught:
        screen(backend, writer=fail_write)
    assert isinstance(caught.value.__cause__, OSError) and backend.retired == 1


def test_callback_close_and_finite_cells_remain_owned_until_actual_retirement():
    stop = threading.Event()
    callback = finite.ScreenTokens(320, 5, stop)
    assert not callback.first_ready.is_set()
    assert callback([1]) is False and callback.first_ready.is_set()
    stop.set()
    assert callback([2]) is True and callback.stop_observed
    callback.close()
    with pytest.raises(RuntimeError, match='after retirement'):
        callback([3])
    assert len(callback.times) == len(callback.counts) == 320
    assert finite.screen_control_payload_bytes() > 0


def test_fast_flag_is_optin_bounded_and_preserves_full_mode(tmp_path):
    assert not probe.arguments(cli(tmp_path)).fast_grid
    args = probe.arguments(cli(tmp_path) + ['--session-ceiling', '24', '--concurrency-grid', '--fast-grid', '--tokens', '512'])
    assert args.fast_grid and args.tokens == 512 and args.repetitions == 1
    assert finite.GRID == (1, 2, 4, 8, 16, 24)
    for extra in (['--fast-grid'], ['--concurrency-grid', '--fast-grid', '--tokens', '256'],
                  ['--concurrency-grid', '--fast-grid', '--tokens', '512', '--serial-reference'],
                  ['--concurrency-grid', '--fast-grid', '--tokens', '512', '--repetitions', '2']):
        with pytest.raises(SystemExit):
            probe.arguments(cli(tmp_path) + extra)


def test_all_six_screen_points_run_after_only_one_cheap_bringup_even_when_invalid(tmp_path, monkeypatch):
    args = probe.arguments(cli(tmp_path) + ['--concurrency-grid', '--fast-grid', '--tokens', '512'])
    events, receipts = [], []
    engine = NS(sm70_profile='source-packed-trial')
    def warmup(*_args, count, result, **_kwargs):
        events.append(('bringup', count))
        result.update(stats={'cached': 0}, token_ids=[1, 2], first_committed_token_elapsed_ns=5)
    def case(_pool, *, count, write, **_kwargs):
        events.append(('screen', count))
        write({'offered': count, 'passed': False})
        return {'offered': count, 'passed': False}
    monkeypatch.setattr(probe, 'generate_reply', warmup)
    monkeypatch.setattr(finite, 'run_screen_case', case)
    monkeypatch.setattr(probe, 'write_report', lambda path, value, **_: receipts.append(path.name))
    report = {'concurrency': {'completed_cases': 0}}
    prompt = {'rendered': 'fixedNotebook'}
    invalid = probe.run_fast_grid(engine, object(), args, report, RequestLimit(24), object(),
        [('bringup', [0], {}), ('sustained', [1], prompt)], finite, 3)
    assert invalid == 6 and events == [('bringup', 2)] + [('screen', n) for n in finite.GRID]
    assert report['fast_screen']['all_required_N_attempted'] and report['concurrency']['completed_cases'] == 6
    assert len([name for name in receipts if name.startswith('fast-mtp')]) == 6


def test_output_cap_before_target_window_is_valid_but_explicitly_short():
    summary, row = screen(ScreenBackend(1, chunks=512, delay=.0002), window=.8)
    assert summary['passed'] and row['screen']['short_window']
    assert row['completed'] == 1 and row['intentionally_cancelled'] == 0
    assert row['requests'][0]['natural_stop'] == 'output_cap' and len(row['requests'][0]['token_ids']) == 512
    assert not row['screen']['full_qualification'] and not row['screen']['quality_assessed']


def test_interrupted_fast_wait_still_proves_actual_worker_retirement():
    backend, limit, records = ScreenBackend(1), RequestLimit(24), []
    class Interrupted:
        def __init__(self, future):
            self.future, self.once = future, False
        def result(self):
            if not self.once:
                self.once = True
                raise KeyboardInterrupt('injected fast drain interruption')
            return self.future.result()
        def done(self):
            return self.future.done()
        def cancelled(self):
            return self.future.cancelled()
        def cancel(self):
            return self.future.cancel()
    class Pool:
        def __init__(self, executor):
            self.executor = executor
        def submit(self, *args):
            return Interrupted(self.executor.submit(*args))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(KeyboardInterrupt, match='injected fast drain interruption'):
            finite.run_screen_case(Pool(executor), count=1, maximum_tokens=512, vocab=3, limit=limit,
                invoke=backend.invoke, observe=backend.observe, write=records.append, timeout=3,
                readiness_seconds=.15, decode_seconds=.03)
    assert backend.retired == 1 and not backend.active and not limit.used
    assert backend.callbacks[0].closed and records[0]['screen']['all_workers_permits_backend_owners_drained']
    assert not records[0]['passed'] and records[0]['screen']['stop_reason'] == 'runtime_or_controller_failure'


def test_native_commits_suppressed_after_cancel_are_retained_untimed_not_in_window():
    stop = threading.Event()
    collector = finite.ScreenTokens(512, 4, stop)
    stream = NS(out=[1], committed_output_tokens=1)
    collector.bind_native_stream(stream)
    assert collector([1]) is False
    stop.set()
    stream.out.append(2)
    stream.committed_output_tokens += 1
    assert collector([2]) is True
    # Canonical Scheduler drops later callback delivery after cancellation.
    stream.out.extend([3, 1])
    stream.committed_output_tokens += 2
    collector.close()
    record = collector.native_record({'committed_output_tokens': 4})
    assert collector.ids == [1, 2] and record['native_token_ids'] == [1, 2, 3, 1]
    assert record['native_undelivered_tokens'] == 2 and record['native_undelivered_commit_timestamps'] is None
    assert collector.native_stream is None


def test_slow_observation_requires_read_start_confirmation_of_exact_end():
    backend, observed = ScreenBackend(1), []
    actual = backend.observe
    def slow_observe():
        before = time.monotonic_ns()
        value = actual()
        time.sleep(.012)
        observed.append((before, time.monotonic_ns(), len(value['decoding'])))
        return value
    limit, records = RequestLimit(24), []
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = finite.run_screen_case(pool, count=1, maximum_tokens=512, vocab=3, limit=limit,
            invoke=backend.invoke, observe=slow_observe, write=records.append, timeout=3,
            readiness_seconds=.3, decode_seconds=.025)
    row = records[0]
    assert result['passed'] and row['screen']['duration_ns'] == 25000000
    end, cancel = row['screen']['end_monotonic_ns'], row['screen']['cancel_requested_monotonic_ns']
    assert any(start >= end and stopped <= cancel and count == 1 for start, stopped, count in observed)
    assert not limit.used and backend.retired == 1


def test_controller_fault_notifies_real_pipe_before_blocked_future_drain():
    import os
    import select
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    primary, outcomes = RuntimeError('controller snapshot failed'), []
    limit, records = RequestLimit(24), []
    state = {'active': False, 'admitted': 0}
    def invoke(callback, _progress):
        state['active'], state['admitted'] = True, 1
        entered.set()
        assert release.wait(2)
        state['active'] = False
        return {'stats': {'committed_output_tokens': 0}, 'text': ''}
    def observe():
        if entered.is_set() and state['active']:
            raise primary
        return {'states': (), 'decoding': (), 'outcomes': {'admitted': state['admitted'], 'cancelled': 0,
            'expired': 0, 'refused': 0}, 'backend_live': 0, 'backend_retiring': 0, 'backend_mailboxes': 0}
    read_fd, write_fd = os.pipe()
    def notify():
        return os.write(write_fd, b'\1') == 1
    def run():
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                finite.run_screen_case(pool, count=1, maximum_tokens=512, vocab=3, limit=limit,
                    invoke=invoke, observe=observe, write=records.append, timeout=3,
                    readiness_seconds=.1, decode_seconds=.03, failure_notify=notify)
        except BaseException as error:
            outcomes.append(error)
        finally:
            retired.set()
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(1)
        assert select.select([read_fd], [], [], 1)[0]
        assert os.read(read_fd, 1) == b'\1' and not retired.is_set()
    finally:
        release.set()
        thread.join(2)
        os.close(write_fd)
        os.close(read_fd)
    assert not thread.is_alive() and outcomes == [primary] and not limit.used
    assert records[0]['screen']['all_workers_permits_backend_owners_drained']


@pytest.mark.parametrize('count', [None, True, 1.0, -1])
def test_unavailable_native_count_never_invents_native_completeness(count):
    collector = finite.ScreenTokens(512, 4, threading.Event())
    collector([1])
    collector.close()
    record = collector.native_record({} if count is None else {'committed_output_tokens': count})
    assert collector.ids == [1] and record['native_token_ids'] is None
    assert record['native_committed_tokens'] is None and record['native_undelivered_tokens'] is None
    assert 'unavailable' in record['native_trace_source']


def test_measurement_without_native_count_or_stream_proof_cannot_pass():
    backend, limit, rows = ScreenBackend(1), RequestLimit(24), []
    original = backend.invoke
    def invoke(callback, progress):
        result = original(callback, progress)
        result['stats'].pop('committed_output_tokens')
        return result
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = finite.run_screen_case(pool, count=1, maximum_tokens=512, vocab=3, limit=limit,
            invoke=invoke, observe=backend.observe, write=rows.append, timeout=3,
            readiness_seconds=.15, decode_seconds=.03)
    assert not result['passed'] and not rows[0]['screen']['native_commit_trace_complete']
    assert rows[0]['requests'][0]['token_ids'] and rows[0]['requests'][0]['native_token_ids'] is None
    assert not limit.used and backend.retired == 1


def test_notification_and_report_failure_are_both_retained_with_primary():
    from tests.test_v100_backend_probe import retained_failure
    backend, limit = ScreenBackend(1, failure=True), RequestLimit(24)
    notification, report = OSError('early notification failure'), MemoryError('later report failure')
    def notify():
        raise notification
    def write(_):
        raise report
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='original fast generation failure') as caught:
            finite.run_screen_case(pool, count=1, maximum_tokens=512, vocab=3, limit=limit,
                invoke=backend.invoke, observe=backend.observe, write=write, timeout=3,
                readiness_seconds=.15, decode_seconds=.03, failure_notify=notify)
    assert retained_failure(caught.value, notification) and retained_failure(caught.value, report)
    assert not limit.used and backend.retired == 1


def test_v2_is_explicit_2048_cap_with_unchanged_v1_and_quality_arguments(tmp_path):
    old=probe.arguments(cli(tmp_path)+["--concurrency-grid","--fast-grid","--tokens","512"])
    assert old.fast_grid_protocol==finite.FAST_V1
    args=probe.arguments(cli(tmp_path)+["--concurrency-grid","--fast-grid","--fast-grid-protocol",finite.FAST_V2,
        "--tokens","2048","--quality-spotchecks"])
    assert args.tokens==2048 and args.quality_spotchecks and args.fast_grid_protocol==finite.FAST_V2
    from tools import finite_quality
    assert finite_quality.OUTPUT_TOKENS==320 and len(finite_quality.CASES)==6
    for extra in (["--tokens","2048"],["--fast-grid-protocol",finite.FAST_V2],
                  ["--concurrency-grid","--fast-grid","--tokens","2048"],
                  ["--concurrency-grid","--fast-grid","--fast-grid-protocol",finite.FAST_V2,"--tokens","512"],
                  ["--concurrency-grid","--fast-grid","--fast-grid-protocol",finite.FAST_V2,"--tokens","2049"]):
        with pytest.raises(SystemExit):
            probe.arguments(cli(tmp_path)+extra)


@pytest.mark.parametrize("count",finite.GRID)
def test_v2_full_fixed_window_retains_all_native_ids_and_every_owner_drains(count):
    backend=ScreenBackend(count,chunks=2048)
    rows=[]
    limit=RequestLimit(24)
    with ThreadPoolExecutor(max_workers=count) as pool:
        summary=finite.run_screen_case(pool,count=count,maximum_tokens=2048,vocab=3,limit=limit,
            invoke=backend.invoke,observe=backend.observe,write=rows.append,
            timeout=3,readiness_seconds=.15,decode_seconds=.04,protocol=finite.FAST_V2)
    row=rows[0]
    assert summary["passed"] and row["screen"]["protocol"]==finite.FAST_V2
    assert row["screen"]["duration_ns"]==40000000 and not row["screen"]["short_window"]
    assert row["screen"]["all_workers_permits_backend_owners_drained"] and not limit.used
    begin,end=row["screen"]["begin_monotonic_ns"],row["screen"]["end_monotonic_ns"]
    expected=[sum(begin<at<=end for at in request["chunk_monotonic_ns"]) for request in row["requests"]]
    assert expected==[request["measured_window_committed_tokens"] for request in row["requests"]]
    assert all(request["native_token_ids"]==request["token_ids"] for request in row["requests"])
    assert backend.retired==count and all(callback.closed for callback in backend.callbacks)


def test_v2_exact_collector_boundary_and_full_native_trace_are_not_truncated():
    stop=threading.Event()
    with pytest.raises(ValueError):
        finite.ScreenTokens(2048,4096,stop)  # V1 ceiling is preserved by default.
    collector=finite.ScreenTokens(2048,4096,stop,maximum_bound=2048)
    for token in range(2048):
        assert not collector([token])
    assert collector.events==2048 and collector.record()["token_ids"]==list(range(2048))
    with pytest.raises(ValueError,match="output/token"):
        collector([1])
    native=NS(out=list(range(2048)),committed_output_tokens=2048)
    collector.bind_native_stream(native)
    collector.close()
    proof=collector.native_record({})
    assert proof["native_token_ids"]==list(range(2048)) and proof["native_committed_tokens"]==2048
    assert proof["native_undelivered_tokens"]==0 and collector.native_stream is None
    assert finite.control_payload_bytes(2048)>finite.control_payload_bytes(512)
    assert finite.screen_control_payload_bytes(2048)-finite.screen_control_payload_bytes(512)==24*2*1536*8


def test_v2_observation_refuses_wrong_protocol_cap_before_any_worker():
    with pytest.raises(ValueError,match="bounds"):
        finite.run_screen_case(None,count=1,maximum_tokens=2048,vocab=3,limit=None,
            invoke=None,observe=None,write=None,timeout=1,protocol=finite.FAST_V1)
    with pytest.raises(ValueError,match="bounds"):
        finite.run_screen_case(None,count=1,maximum_tokens=512,vocab=3,limit=None,
            invoke=None,observe=None,write=None,timeout=1,protocol=finite.FAST_V2)

@pytest.mark.parametrize("count", [1,8,16,24])
def test_profile_concurrency_is_one_instrumented_cohort_without_all6_credit(tmp_path, monkeypatch, count):
    extra=(["--tokens","2048","--fast-grid-protocol",finite.FAST_V2] if count in (1,24) else ["--tokens","512"])
    args=probe.arguments(cli(tmp_path)+["--concurrency-grid","--fast-grid",*extra,"--profile-concurrency",str(count)])
    monkeypatch.setattr(probe,"concurrent_observation_snapshot",lambda:{"test_observations":True})
    events=[]
    engine=NS(sm70_profile="source-owner-target-guard-trial")
    def warmup(*_args,count,result,**_kwargs):
        events.append(("bringup",count))
        result.update(stats={},first_committed_token_elapsed_ns=1,token_ids=[1])
    def case(_pool,*,count,write,**_kwargs):
        events.append(("screen",count))
        write({})
        return {"passed":True}
    monkeypatch.setattr(probe,"generate_reply",warmup)
    monkeypatch.setattr(finite,"run_screen_case",case)
    monkeypatch.setattr(probe,"write_report",lambda *_args,**_kwargs:None)
    report={"concurrency":{"completed_cases":0}}
    assert probe.run_fast_grid(engine,object(),args,report,RequestLimit(24),object(),
        [("sustained",[1],{"rendered":"Notebook"})],finite,3)==0
    assert events==[("bringup",2),("screen",count)]
    assert not report["fast_screen"]["all_required_N_attempted"]
    assert report["fast_screen"]["offered_counts"]==[count]
    assert report["fast_screen"]["measurement_role"]=="instrumentedONEcohortdiagnostic_notall6comparison"


@pytest.mark.parametrize("extra", [
    ["--profile-concurrency","2"],
    ["--profile-concurrency","8"],
    ["--concurrency-grid","--fast-grid","--tokens","512","--profile-concurrency","8","--quality-spotchecks"],
    ["--concurrency-grid","--fast-grid","--tokens","512","--profile-concurrency","16","--serial-reference"],
])
def test_profile_concurrency_refuses_unbounded_or_misleading_combinations(tmp_path,extra):
    with pytest.raises(SystemExit):
        probe.arguments(cli(tmp_path)+extra)


def test_profile_observer_payload_is_funded_globally_before_any_request(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retention_resources
    args=probe.arguments(cli(tmp_path)+["--concurrency-grid","--fast-grid","--tokens","512","--profile-concurrency","8"])
    monkeypatch.setattr(probe,"prepared_prompt",lambda *_args:([1],{"rendered":"Notebook"}))
    seen=[]
    error=RuntimeError("stop at real host admission, before requests")
    def finalize(_catalog,*,caller,**_kwargs):
        expected=finite.control_payload_bytes(512)+finite.screen_control_payload_bytes(512)+8*((128<<10)+(2<<30))
        assert caller.control_payload_bytes==expected
        assert caller.max_concurrent_requests==24 and caller.max_receipt_bytes==60620800
        seen.append(caller)
        raise error
    monkeypatch.setattr(retention_resources,"finalize_backend_resources",finalize)
    engine=NS(retention_catalog=object(),scheduler=object())
    with pytest.raises(RuntimeError) as caught:
        probe.run_concurrency_leader(engine,object(),args,{})
    assert caught.value is error and len(seen)==1


def test_single_profile_requires_v2_and_exposes_real_graph_hit_delta(tmp_path):
    with pytest.raises(SystemExit):
        probe.arguments(cli(tmp_path)+["--concurrency-grid","--fast-grid","--tokens","512","--profile-concurrency","1"])
    args=probe.arguments(cli(tmp_path)+["--concurrency-grid","--fast-grid","--tokens","2048",
        "--fast-grid-protocol",finite.FAST_V2,"--profile-concurrency","1"])
    assert args.profile_concurrency==1 and args.fast_grid_protocol==finite.FAST_V2
    bound=NS(hits=7,misses=0,lane_declines=0)
    engine=NS(multi=NS(graphs=NS(bound=bound,verify={},mtp={})),w=NS(meta={}))
    before=probe.graph_execution_snapshot(engine)
    bound.hits+=12
    after=probe.graph_execution_snapshot(engine)
    assert probe.graph_execution_delta(before,after)["bound.hits"]==12
    assert before["counters"]["bound.hits"]==7 and after["counters"]["bound.hits"]==19


def test_full24_profile_requires_v2_and_keeps_actual24_request_cohort(tmp_path):
    with pytest.raises(SystemExit):
        probe.arguments(cli(tmp_path)+["--concurrency-grid","--fast-grid","--tokens","512","--profile-concurrency","24"])
    args=probe.arguments(cli(tmp_path)+["--concurrency-grid","--fast-grid","--tokens","2048",
        "--fast-grid-protocol",finite.FAST_V2,"--profile-concurrency","24"])
    assert args.profile_concurrency==24 and args.tokens==2048 and not args.quality_spotchecks
