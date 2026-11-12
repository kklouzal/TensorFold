"""Diagnostic ownership and model-effect preservation without CUDA."""
from types import SimpleNamespace

import pytest

from tools.v100_commit_trace import CommitTrace, resources


class Decoder:
    def __init__(self):
        self.calls = 0

    def _take(self, stream, new):
        self.calls += 1
        stream.committed_output_tokens += len(new)
        return 'original-result'


def stream():
    return SimpleNamespace(sid=17, committed_output_tokens=0)


def test_mtp_batch_is_one_interval_and_original_effect_occurs_once():
    ticks = iter([100, 160, 300, 340])
    decoder, s = Decoder(), stream()
    trace = CommitTrace(2, clock=lambda: next(ticks)).install(decoder)
    assert decoder._take(s, [1, 2, 3, 4]) == 'original-result'
    decoder._take(s, [5])
    report = trace.close()
    assert report['complete'] and decoder.calls == 2
    assert report['events'] == [[17, 100, 160, 0, 4, 4], [17, 300, 340, 4, 5, 1]]
    assert '_take' not in vars(decoder)


def test_exhaustion_fails_diagnostic_without_suppressing_model_or_growing_records():
    decoder, s = Decoder(), stream()
    trace = CommitTrace(1).install(decoder)
    for _ in range(5):
        decoder._take(s, [1])
    report = trace.close()
    assert decoder.calls == s.committed_output_tokens == 5
    assert len(report['events']) == 1 and not report['complete']
    assert report['failure']['stage'] == 'event-capacity'


def test_original_failure_propagates_unchanged_without_retry():
    primary = ValueError('model failure')
    decoder, s = Decoder(), stream()
    def fail(*args):
        decoder.calls += 1
        raise primary
    decoder._take = fail
    trace = CommitTrace(3).install(decoder)
    with pytest.raises(ValueError) as caught:
        decoder._take(s, [1])
    assert caught.value is primary and decoder.calls == 1
    report = trace.close()
    assert decoder._take is fail and report['failure']['stage'] == 'model-take'


def test_marker_failure_does_not_change_completed_model_output():
    decoder, s = Decoder(), stream()
    def marker(*args):
        raise OSError('marker failure')
    trace = CommitTrace(2, marker=marker).install(decoder)
    assert decoder._take(s, [1, 2]) == 'original-result'
    assert s.committed_output_tokens == 2 and not trace.close()['complete']


def test_changed_binding_is_not_overwritten():
    decoder = Decoder()
    trace = CommitTrace(1).install(decoder)
    replacement = lambda *args: None
    decoder._take = replacement
    with pytest.raises(RuntimeError, match='binding changed'):
        trace.close()
    assert decoder._take is replacement


@pytest.mark.parametrize('count', [False, 0, -1, 1.5, 1 << 63])
def test_event_storage_bound_rejects_invalid_counts_before_allocation(count):
    with pytest.raises(ValueError):
        resources(count)


def test_diagnostic_input_credit_is_finite_and_separate_from_closed_operations():
    from tools import v100_owned_qualification as owned
    args = SimpleNamespace(commit_trace_events=3, commit_trace_nvtx=True)
    spec = owned._commit_trace_spec(args)
    assert spec['global_host_obligation_bytes'] > 8 * spec['retained_event_bytes']
    assert spec['nvtx'] is True
    args.teacher_plan = 'frozen-teacher'
    with pytest.raises(ValueError, match='separate finite diagnostic'):
        owned._commit_trace_spec(args)
    with pytest.raises(ValueError, match='explicit finite trace'):
        owned._commit_trace_spec(SimpleNamespace(commit_trace_nvtx=True))
    with pytest.raises(ValueError, match='global host obligation'):
        owned._commit_trace_spec(SimpleNamespace(commit_trace_events=1 << 51))


def test_positive_retirement_binds_report_and_preserves_previous_history(tmp_path):
    import json
    from tools import v100_owned_qualification as owned
    decoder, s = Decoder(), stream()
    trace = CommitTrace(2).install(decoder)
    decoder._take(s, [1, 2])
    engine = SimpleNamespace(rank=0, tp=8, _closed=True, _abort_comm=False)
    spec = owned._commit_trace_spec(SimpleNamespace(commit_trace_events=2))
    owned._finish_commit_trace(trace, engine, tmp_path, spec)
    target = tmp_path / 'commit-trace-rank-0.json'
    raw = target.read_bytes()
    result = json.loads(raw)
    assert result['complete'] and result['events'][0][-1] == 2
    with pytest.raises(ValueError, match='already exists'):
        owned._finish_commit_trace(trace, engine, tmp_path, spec)
    assert target.read_bytes() == raw


def test_aborted_model_cannot_make_positive_commit_diagnostic(tmp_path):
    import json
    from tools import v100_owned_qualification as owned
    decoder = Decoder()
    trace = CommitTrace(1).install(decoder)
    engine = SimpleNamespace(rank=1, tp=8, _closed=True, _abort_comm=True)
    spec = owned._commit_trace_spec(SimpleNamespace(commit_trace_events=1))
    with pytest.raises(RuntimeError, match='incomplete'):
        owned._finish_commit_trace(trace, engine, tmp_path, spec)
    assert json.loads((tmp_path / 'commit-trace-rank-1.json').read_text())['complete'] is False


def test_open_benchmark_keeps_diagnostic_marker_on_every_publication():
    from tools import v100_owned_qualification as owned
    # Publication must remain diagnostic even before/after cohort replacement;
    # open observers have no engine._owned_benchmark closed-protocol journal.
    engine = SimpleNamespace(multi=SimpleNamespace(controller=SimpleNamespace(work_observations=object())))
    spec = owned._commit_trace_spec(SimpleNamespace(commit_trace_events=2))
    observer = owned._BenchmarkObserver(engine, {'run_identity': 'diagnostic', 'cohorts': []},
                                        timed=False, commit_trace=spec)
    for _ in range(3):
        assert observer.published['P3_valid'] is False
        assert observer.published['diagnostic_only'] is True
        assert observer.published['commit_trace'] == spec
        observer._publish()
    assert not hasattr(engine, '_owned_benchmark')
