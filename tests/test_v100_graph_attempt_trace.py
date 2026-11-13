"""Actual graph dispatch boundaries with CPU replay doubles; no CUDA proof."""
import json
from types import SimpleNamespace as NS

import pytest

from tools.v100_commit_trace import CommitTrace, resources, graph_histogram
from tests.test_flashnext_exact_graphs import owner


class Replay:
    def __init__(self, error=None):
        self.calls = self.resets = 0
        self.error = error
    def replay(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
    def reset(self):
        self.resets += 1


def fixture(monkeypatch, *, replay_count=2):
    from tensorfold.families.qwen4_exp.cuda import exact_graphs, compact_indexer
    bound, states = owner(slots=8, rows=40)
    bound.owner.retain_failure = lambda error: setattr(bound.owner, 'failure', error)
    decoder = bound.dec
    decoder.buf.ple_late = None
    decoder.graphs = NS(bound=bound)
    segs = [(state, i * 5, (i + 1) * 5) for i, state in enumerate(states)]
    graphs = [Replay() for _ in range(replay_count)]
    pieces = NS(graphs=graphs[:], inserts=[('ple_stage', None)] * (replay_count - 1), out=object())
    bound.owner.verify[(40, 8)] = pieces
    monkeypatch.setattr(exact_graphs, '_kv_begin', lambda segs: None)
    monkeypatch.setattr(exact_graphs, 'replay', lambda graph, comm: graph.replay())
    monkeypatch.setattr(compact_indexer, 'validate_target_indexer_frame', lambda *args: None)
    monkeypatch.setattr(compact_indexer, 'validate_mtp_indexer_frame', lambda *args: None)
    return decoder, bound, segs, pieces, graphs


def test_actual_hit_miss_shapes_and_replay_api_counts_are_separate(monkeypatch):
    dec, bound, segs, pieces, originals = fixture(monkeypatch)
    trace = CommitTrace(8, graph_attempts=True).install(dec)
    assert bound.forward('verify', segs) is pieces.out
    assert bound.forward('mtp', segs) is None
    report = trace.close()
    assert report['events'] == [[0, 40, 8, 1, 0, 2, 2], [1, 40, 8, 0, 1, 0, 0]]
    assert report['histogram'] == graph_histogram(report['events'])
    assert [x.calls for x in originals] == [1, 1] and pieces.graphs == originals
    assert 'forward' not in vars(bound) and report['complete']
    assert 'GPU completion' in report['scope'] and report['P3_valid'] is False


@pytest.mark.parametrize('mode,reason', [('device-greedy', 2), ('rotor', 3), ('media', 4)])
def test_actual_excluded_branches_are_not_reported_as_replay(monkeypatch, mode, reason):
    dec, bound, segs, _, graphs = fixture(monkeypatch)
    trace = CommitTrace(4, graph_attempts=True).install(dec)
    kwargs = {}
    if mode == 'device-greedy':
        kwargs['device_tokens'] = object()
    elif mode == 'rotor':
        dec.buf.rotor_lookup = True
    else:
        segs[0][0].image_positions = object()
    assert bound.forward('verify', segs, **kwargs) is None
    report = trace.close()
    assert report['events'][0] == [0, 40, 8, 0, reason, 0, 0]
    assert sum(graph.calls for graph in graphs) == 0


def test_retirement_delegates_existing_resets_and_restores_original_graphs(monkeypatch):
    dec, bound, _, pieces, graphs = fixture(monkeypatch)
    trace = CommitTrace(4, graph_attempts=True).install(dec)
    for graph in pieces.graphs:
        graph.reset()  # Existing Engine's reset owner, no observer-initiated reset.
    report = trace.close()
    assert report['closed'] and pieces.graphs == graphs
    assert [g.resets for g in graphs] == [1, 1]
    assert trace._decoder is trace._graph_bound is None
    assert all(row is None for row in trace._graph_bindings)


def test_exhaustion_keeps_model_work_exactly_once_with_bounded_storage(monkeypatch):
    dec, bound, segs, pieces, graphs = fixture(monkeypatch)
    trace = CommitTrace(2, graph_attempts=True).install(dec)
    for _ in range(5):
        assert bound.forward('verify', segs) is pieces.out
    report = trace.close()
    assert len(report['events']) == 2 and not report['complete']
    assert report['failure']['stage'] == 'event-capacity'
    assert [g.calls for g in graphs] == [5, 5]
    assert len(trace._rows) == 2 * 7 and len(trace._graph_bindings) == 2


def test_binding_credit_refuses_installation_before_mutation(monkeypatch):
    dec, bound, _, pieces, graphs = fixture(monkeypatch)
    with pytest.raises(ValueError, match='binding'):
        CommitTrace(1, graph_attempts=True).install(dec)
    assert pieces.graphs == graphs and 'forward' not in vars(bound)


@pytest.mark.parametrize('after_replay', [False, True])
def test_original_error_propagates_without_retry_and_reports_replay_progress(monkeypatch, after_replay):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer
    dec, bound, segs, _, graphs = fixture(monkeypatch)
    primary = ValueError('controlled original failure')
    if after_replay:
        graphs[0].error = primary
    else:
        def fail(*args):
            raise primary
        monkeypatch.setattr(compact_indexer, 'validate_target_indexer_frame', fail)
    trace = CommitTrace(4, graph_attempts=True).install(dec)
    with pytest.raises(ValueError) as caught:
        bound.forward('verify', segs)
    assert caught.value is primary
    report = trace.close()
    assert report['events'][0][3:] == ([2, 6, 1, 0] if after_replay else [2, 5, 0, 0])
    assert not report['complete']
    assert sum(g.calls for g in graphs) == int(after_replay)


def test_graph_mode_reuses_explicit_host_grant_and_rejects_cuda_markers():
    from tools import v100_owned_qualification as owned, v100_onecat_campaign as campaign
    spec = owned._commit_trace_spec(NS(commit_trace_events=256, commit_trace_graph_attempts=True, benchmark_plan=object()))
    other = campaign.commit_diagnostic_spec(256, False, True)
    assert all(spec[key] == other[key] for key in spec)
    assert spec['retained_event_bytes'] > 256 * 7 * 8
    for args in (NS(commit_trace_graph_attempts=True),
                 NS(commit_trace_events=4, commit_trace_graph_attempts=True, commit_trace_nvtx=True, benchmark_plan=object())):
        with pytest.raises(ValueError):
            owned._commit_trace_spec(args)
    with pytest.raises(ValueError, match='markers'):
        CommitTrace(4, graph_attempts=True, marker=lambda *a: None)


def test_closed_rank_histogram_bound_and_source_are_required(monkeypatch, tmp_path):
    from tools import v100_owned_qualification as owned, v100_onecat_campaign as campaign
    dec, bound, segs, _, _ = fixture(monkeypatch)
    trace = CommitTrace(8, graph_attempts=True).install(dec)
    bound.forward('verify', segs)
    spec = owned._commit_trace_spec(NS(commit_trace_events=8, commit_trace_graph_attempts=True, benchmark_plan=object()))
    engine = NS(rank=0, tp=8, _closed=True, _abort_comm=False)
    owned._finish_commit_trace(trace, engine, tmp_path, spec)
    report = json.loads((tmp_path / 'commit-trace-rank-0.json').read_text())
    assert len(json.dumps(report).encode()) <= spec['serialization_bytes']
    output = tmp_path / 'host'; (output / 'qualification').mkdir(parents=True)
    source = {'files': {'tools/v100_owned_qualification.py': {'sha256': report['observer_source_sha256']}}}
    for rank in range(8):
        row = {**report, 'rank': rank}
        (output / 'qualification' / f'commit-trace-rank-{rank}.json').write_text(json.dumps(row))
    result = campaign.commit_trace_receipts(output, source, campaign.commit_diagnostic_spec(8, False, True))
    assert len(result) == 8 and result[0]['multi_participant_attempts'] == 1
    path = output / 'qualification/commit-trace-rank-7.json'
    row = json.loads(path.read_text()); row['histogram'][0]['attempts'] += 1; path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='histogram'):
        campaign.commit_trace_receipts(output, source, campaign.commit_diagnostic_spec(8, False, True))


def test_a_source_hit_counter_without_a_replay_call_cannot_be_a_complete_witness(monkeypatch):
    dec, bound, segs, pieces, _ = fixture(monkeypatch, replay_count=0)
    pieces.inserts = []
    trace = CommitTrace(4, graph_attempts=True).install(dec)
    assert bound.forward('verify', segs) is pieces.out
    report = trace.close()
    assert not report['complete'] and report['failure']['stage'] == 'graph-after-observation'


def test_campaign_forwards_graph_mode_only_with_finite_unprofiled_matrix():
    from tools import v100_onecat_campaign as campaign
    from tests.test_v100_onecat_campaign import ModelArmTests, CommitDiagnosticTests
    plan = campaign.diagnostic_c1_c8_plan(CommitDiagnosticTests().plan())
    args = campaign.model_arguments(ModelArmTests().settings(), plan,
        diagnostic_commit_events=1024, diagnostic_graph_attempts=True)
    assert '--commit-trace-graph-attempts' in args and '--commit-trace-nvtx' not in args
    assert '--benchmark-plan' in args
    with pytest.raises(ValueError, match='finite C1/C8'):
        campaign.model_arguments(ModelArmTests().settings(), plan, smoke_only=True,
            diagnostic_commit_events=1024, diagnostic_graph_attempts=True)
