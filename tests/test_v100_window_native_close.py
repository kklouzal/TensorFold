"""Real Engine/Window close transaction with CPU physical-owner stand-ins only."""
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.exact_graphs import ExactStateGraphs
from tests.test_v100_target_window_teacher import plan
from tools import v100_owned_qualification as owned
from tools import v100_target_window_teacher as window


def close_fixture(tmp_path, monkeypatch, rank=0, fault=None):
    events = []
    engine = object.__new__(FlashNextEngine)
    engine.rank = rank
    engine._lifecycle = threading.Condition()
    engine._calls, engine._closed, engine._closing = {}, False, False
    engine._close_running = engine._receiving = False
    engine._teacher_attention_provider_paths = ('/ordinary/torch/_C.so', '/ordinary/torch/lib/libtorch_cuda.so')
    engine.w = SimpleNamespace(device='cpu-physical-standin', meta={})
    resource_failure = RuntimeError('physical resource fence failed')
    native_failure = RuntimeError('native SDK observer failed')
    def fence(device):
        assert device == 'cpu-physical-standin'
        events.append('resource-fence')
        if fault == 'resource':
            raise resource_failure
    monkeypatch.setattr(torch.cuda, 'synchronize', fence)
    graphs = SimpleNamespace(bound=object.__new__(ExactStateGraphs),
        record=lambda: {'serving_hits': {'verify': 1, 'mtp': 1}, 'serving_misses': {'verify': 0, 'mtp': 0}},
        close=lambda: events.append('graphs-closed'))
    engine.multi = SimpleNamespace(graphs=graphs, recover_failed_slots=lambda:events.append('slots-retired'))
    observer = object.__new__(window.WindowTeacher)
    observer.engine, observer.decoder = engine, engine.multi
    data = plan(16)
    for case in data['sequences']:
        case['prompt_ids'] = [1]*256
    resource = window.resources(1, data)
    observer.prepared = {'data': data, 'resources': resource, 'input_sha256': '1'*64}
    observer.directory = tmp_path/f'rank-{rank}'
    observer.directory.mkdir()
    observer.failed, observer.close_attempted, observer.closed = None, False, False
    observer.finished = {row['case_id'] for row in data['sequences']}
    observer.round_index = len(data['rounds'])
    observer.frame_index = resource['maximum_total_frames']
    observer.profile, observer.runtime_sources, observer.initial_graphs = {}, {}, window.graph_counts(graphs)
    observer.streams, observer.by_state, observer.originals = {}, {}, {}
    observer.restores, observer.staged = [], []
    observer.prefill_refs, observer.mtp_stage_refs, observer.mtp_pick_refs = [], [], []
    observer.window_refs = []
    for index in range(observer.round_index):
        path = observer.directory/f'window-{index:06d}.json'
        value = {'graph_calls': [{'role': role, 'participants': 8, 'serving_hits_delta': 1}
                                 for role in ('verify', 'mtp')]}
        window.frames._owned_teacher_json(path, value, 1 << 20)
        raw = path.read_bytes()
        observer.window_refs.append({'file': path.name, 'bytes': len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
    observer.raise_failures = raise_failures
    observer.old_close = engine.close  # REAL FlashNextEngine.close and _close_resources.
    engine.close = observer.close
    engine._owned_target_window_teacher = observer
    assert not hasattr(engine, '_owned_teacher_frames')
    def record(required, *, rank, tag, evidence_dir, limits, provider_paths):
        assert events[-2:] == ['resource-fence', 'slots-retired']
        assert not engine._closed and not observer.closed  # Both publish only after the inner observer.
        assert required == owned._model_contract().REQUIRED_PREFIXES
        assert limits == owned._model_contract().MODEL_OBSERVER_LIMITS
        assert tag == 'teacher-closed' and evidence_dir == tmp_path
        assert provider_paths == engine._teacher_attention_provider_paths
        assert rank == engine.rank
        events.append('native-SDK-post-close')
        if fault == 'native':
            raise native_failure
    selected = {'module':SimpleNamespace(record_native_elfs=record), 'directory':tmp_path}
    owned._attach_teacher_native_observer(engine, selected, window=True)
    return engine, observer, events, resource_failure, native_failure


@pytest.mark.parametrize('rank', range(8))
def test_actual_window_engine_resource_close_keeps_native_SDK_observer(tmp_path, monkeypatch, rank):
    engine, observer, events, _, _ = close_fixture(tmp_path, monkeypatch, rank)
    assert observer.native_capture_complete()
    engine.close()
    assert events == ['graphs-closed', 'resource-fence', 'slots-retired', 'native-SDK-post-close']
    assert engine._closed and observer.closed
    summary = json.loads((observer.directory/'summary.json').read_text())
    assert summary['closed'] is True and summary['failed'] is False and len(summary['finished']) == 8
    assert observer.engine is None  # Only positive retirement releases the owner references.
    with pytest.raises(ValueError, match='cannot retry'):
        engine.close()


@pytest.mark.parametrize('fault', ['partial', 'wrong_ids', 'wrong_plan', 'round', 'frames', 'failed', 'bool_round', 'bad_set', 'rebound'])
def test_incomplete_or_stale_window_never_certifies_native_close(tmp_path, monkeypatch, fault):
    engine, observer, events, _, _ = close_fixture(tmp_path, monkeypatch)
    primary = RuntimeError('actual prior capture failure')
    if fault == 'partial':observer.finished.pop()
    elif fault == 'wrong_ids':observer.finished = {'other_'+str(i) for i in range(8)}
    elif fault == 'wrong_plan':
        case = observer.prepared['data']['sequences'].pop()
        observer.finished.remove(case['case_id'])
    elif fault == 'round':observer.round_index -= 1
    elif fault == 'frames':observer.frame_index -= 1
    elif fault == 'failed':observer.failed = primary
    elif fault == 'bool_round':observer.round_index = True
    elif fault == 'bad_set':observer.finished = list(observer.finished)
    else:engine._owned_target_window_teacher = object()
    with pytest.raises((ValueError, RuntimeError)) as caught:
        engine.close()
    if fault == 'failed':assert caught.value is primary
    assert 'native-SDK-post-close' not in events
    assert 'resource-fence' in events  # The real physical-owner path still retires before evidence selection.
    assert json.loads((observer.directory/'summary.json').read_text())['failed'] is True


@pytest.mark.parametrize('fault', ['resource', 'native'])
def test_actual_window_close_retains_resource_or_native_primary(tmp_path, monkeypatch, fault):
    engine, observer, events, resource_failure, native_failure = close_fixture(tmp_path, monkeypatch, fault=fault)
    notified = []
    monkeypatch.setattr(owned, '_observer_failure', lambda current, error:notified.append((current, error)))
    with pytest.raises(RuntimeError) as caught:
        engine.close()
    expected = resource_failure if fault == 'resource' else native_failure
    assert caught.value is expected and engine._closed is False and observer.closed is False
    assert observer.engine is engine  # Unproved retirement does not release captured owners.
    assert ('native-SDK-post-close' in events) == (fault == 'native')
    assert notified == ([(engine, native_failure)] if fault == 'native' else [])
    assert json.loads((observer.directory/'summary.json').read_text())['failed'] is True
