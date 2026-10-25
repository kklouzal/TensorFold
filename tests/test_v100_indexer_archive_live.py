"""Offline frozen-reference and live-procedure acceptance gates; no HTTP/GPU."""

import copy
import json
import threading

import pytest

from tools import v100_indexer_archive_live as tool
from tests.test_v100_rotorquant_live import response, stats


def archive_stats(kv_dtype='int8'):
    value = stats(parallel=2)
    value['model'].update(kv_dtype=kv_dtype, context_length=64)
    value['capacity_plan']['context_window'] = 64
    window = value['capacity_plan']['kv_full_windows']
    window.update(context_window=64, cache_slots_each=68, startup_capacities=[68] * 2)
    for row in window['ranks']:
        row['startup_capacities'] = [68] * 2
    placement = dict(node=0, cpus=[0, 32], sockets=[0], page_size=4096, gpu_uuid='GPU-local')
    allocations = []
    for _ in range(2):
        for tag in ('archive', 'prefill', 'decode', 'ring-restore', 'frame-origins'):
            shape = [3, 68, 128] if tag == 'archive' else [3, 16, 128]
            proof = dict(verified=True, pages=16, node_pages={'0': 16}, extent_bytes=65536)
            allocations.append(dict(tag='indexer-' + tag, shape=shape, dtype='torch.float16',
                                    pinned=tag != 'archive', placement=placement,
                                    mapping='MAP_PRIVATE|MAP_ANONYMOUS', bind_mode=32770,
                                    bind_flags=0, write_first_touch=True, before_pin=proof,
                                    after_pin=proof, backing_bytes=73728))
    backing, pinned = len(allocations) * 73728, 8 * 65536
    history = dict(mode='ram', policy='lossless-fp16-ram-archive-v1', materialized=True,
                   archive_limit=68, layers=3, device_bytes_per_layer=200,
                   maximum_owners=4, host_archive='full-window-pageable', pinned='bounded-transfer-staging-only',
                   placement=placement,
                   ranks=[dict(rank=i, node=i // 4, socket=i // 4, gpu_uuid_words=[i, 1, 2, 3],
                               host_grant_bytes=backing, pinned_grant_bytes=pinned) for i in range(8)],
                   startup_ranks=[dict(rank=i, owners=2, host_backing_bytes=backing,
                                       pinned_staging_bytes=pinned, device_raw_bytes=1200) for i in range(8)],
                   local_allocator=dict(policy='numa-local-private-anon-mbind-verified-v1', placement=placement,
                                        charged_bytes=backing, pinned_extent_bytes=pinned, allocations=allocations))
    value['capacity_plan']['indexer_history'] = history
    return value


@pytest.fixture
def references(tmp_path):
    directory = tmp_path / 'reference'
    directory.mkdir()
    authority = dict(image='sha256:' + 'a' * 64, history='gpu', kv_dtype='int8', results={})
    originals = [(name, dict(model='test', messages=[dict(role='user', content=name)],
                            max_tokens=32, temperature=0, draft=False, return_token_ids=True)) for name in tool.live.CASES]
    seeded = [('seeded-' + name, {**body, 'draft': False}) for name, body in tool.live.profiles('test', 128)]
    for name, body in originals + seeded:
        label = name if name.startswith('seeded-') else 'serial-' + name
        target = directory / label
        target.mkdir()
        result = response(body['max_tokens'])
        authority['results'][name] = tool.live.completion(result, body['max_tokens'])
        for file, item in {'request.json': dict(method='POST', body=body),
                           'exchange.json': dict(complete=True, cancelled_by_client=False, status=200),
                           'result.json': result}.items():
            (target / file).write_text(json.dumps(item))
    (directory / 'authority.json').write_text(json.dumps(authority))
    return directory


def test_all_saved_requests_and_frozen_results_are_loaded_without_budget_changes(references):
    rows, authority, hashes = tool.load_references(references, 'int8')
    assert len(rows) == len(authority['results']) == 21 and len(hashes) == 64
    assert [body['max_tokens'] for _, body, _ in rows] == [32] * 12 + [128] * 9
    assert all(body['draft'] is False for _, body, _ in rows)


@pytest.mark.parametrize('mutation', ['format', 'missing', 'result', 'budget', 'incomplete', 'symlink'])
def test_reference_tampering_is_rejected(references, mutation):
    target = references / 'seeded-code-s17'
    if mutation in ('format', 'missing'):
        path = references / 'authority.json'
        value = json.loads(path.read_text())
        if mutation == 'format':
            value['kv_dtype'] = tool.FORMATS[1]
        else:
            value['results'].pop('multiply')
    elif mutation == 'result':
        path = target / 'result.json'
        value = json.loads(path.read_text())
        value['tensorfold']['token_ids'][0] = 999
    elif mutation == 'budget':
        path = target / 'request.json'
        value = json.loads(path.read_text())
        value['body']['max_tokens'] = 127
    elif mutation == 'incomplete':
        path = target / 'exchange.json'
        value = json.loads(path.read_text())
        value['complete'] = False
    else:
        path = target / 'result.json'
        path.unlink()
        path.symlink_to(references / 'serial-multiply/result.json')
        with pytest.raises(OSError):
            tool.load_references(references, 'int8')
        return
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        tool.load_references(references, 'int8')


@pytest.mark.parametrize('mutation', ['rank', 'window', 'archive', 'remote', 'post-pin', 'mode', 'workspace', 'raw-bytes'])
def test_capacity_rejects_missing_or_remote_actual_allocations(mutation):
    value = archive_stats()
    window = value['capacity_plan']['kv_full_windows']
    history = value['capacity_plan']['indexer_history']
    allocation = history['local_allocator']['allocations'][1]
    if mutation == 'rank':
        window['ranks'].pop()
    elif mutation == 'window':
        window['startup_capacities'] = [68, 67]
    elif mutation == 'archive':
        history['local_allocator']['allocations'][0]['shape'][1] = 67
    elif mutation == 'remote':
        allocation['before_pin']['node_pages'] = {'1': 16}
    elif mutation == 'post-pin':
        allocation.pop('after_pin')
    elif mutation == 'mode':
        allocation['bind_mode'] = 2
    elif mutation == 'workspace':
        window['workspace_bytes'] = 0
    else:
        history['startup_ranks'][3]['device_raw_bytes'] = 1199
    with pytest.raises(ValueError):
        tool.capacity_proof(value, parallel=2, context=64, kv_dtype='int8')


def test_fixed_launch_qualification_rejects_swapped_rank_numa_assignments():
    value = archive_stats()
    ranks = value['capacity_plan']['indexer_history']['ranks']
    for rank in (1, 4):
        ranks[rank]['node'] = ranks[rank]['socket'] = 1 - ranks[rank]['node']
    with pytest.raises(ValueError, match='fixed rank/NUMA order'):
        tool.capacity_proof(value, parallel=2, context=64, kv_dtype='int8')


class Client:
    mutation = None
    kv_dtype = 'int8'
    last = None

    def __init__(self, *args, **kwargs):
        Client.last = self
        self.calls, self.cancelled, self.release = [], False, threading.Event()

    def call(self, label, body=None, **kwargs):
        self.calls.append((label, copy.deepcopy(body), kwargs))
        if label.startswith('concurrent-stats-'):
            self.release.set()
            value = archive_stats(self.kv_dtype)
            value['load']['requests_in_flight'] = 2
            return value
        if body is None:
            if label.startswith('health-'):
                return dict(ok=True)
            value = archive_stats(self.kv_dtype)
            value['totals']['cancelled'] = int(self.cancelled)
            if self.mutation == 'capacity':
                value['capacity_plan']['indexer_history']['materialized'] = False
            return value
        if label.startswith('concurrent-'):
            assert self.release.wait(2)
        if kwargs.get('cancel'):
            self.cancelled = True
            return dict(intentionally_incomplete=True)
        value = response(body['max_tokens'])
        if self.kv_dtype != 'int8':
            value['tensorfold']['cached'] = 0
        if self.mutation == 'cold-leak':
            value['tensorfold']['cached'] = 1
        if self.mutation == 'mismatch' and label == 'seeded-code-s19-serial':
            value['tensorfold']['token_ids'][0] = 999
        if self.mutation == 'recovery' and label == 'prefix-recovery-reuse':
            value['tensorfold']['cached'] = 0
        return value


def arguments(references, output):
    return ['--url', 'http://unused', '--reference-dir', str(references), '--output-dir', str(output),
            '--parallel', '2', '--context', '64', '--kv-dtype', 'int8']


@pytest.mark.parametrize('mutation,code,complete', [(None, 0, True), ('mismatch', 2, True),
                                                    ('capacity', 1, False), ('recovery', 1, False)])
def test_entire_procedure_preserves_frozen_authority_and_failed_gates(monkeypatch, references, tmp_path, mutation, code, complete):
    monkeypatch.setattr(tool.live, 'Client', Client)
    monkeypatch.setattr(Client, 'mutation', mutation)
    output = tmp_path / 'out'
    assert tool.main(arguments(references, output)) == code
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['complete'] is complete and summary['passed'] is (code == 0)
    if complete:
        assert len(summary['checks']) == 42 + 2 + 1 + 2
        bodies = [body for label, body, _ in Client.last.calls if label.startswith('multiply-')]
        assert [body['draft'] for body in bodies] == [False, True]
        assert all(body['max_tokens'] == 32 for body in bodies)
        assert summary['concurrency_peak_in_flight'] == 2
    elif mutation == 'capacity':
        assert all(body is None for _, body, _ in Client.last.calls)


def test_existing_output_refuses_before_http(monkeypatch, references, tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    (output / 'keep').write_text('old')
    monkeypatch.setattr(tool.live, 'Client', lambda *args, **kwargs: pytest.fail('network client created'))
    with pytest.raises(ValueError, match='empty output'):
        tool.main(arguments(references, output))


@pytest.mark.parametrize('kv_dtype', tool.live.Q4_FORMATS)
def test_q4_full_procedure_requires_cold_replies(monkeypatch, references, tmp_path, kv_dtype):
    authority = references / 'authority.json'
    value = json.loads(authority.read_text())
    value['kv_dtype'] = kv_dtype
    authority.write_text(json.dumps(value))
    monkeypatch.setattr(tool.live, 'Client', Client)
    monkeypatch.setattr(Client, 'kv_dtype', kv_dtype)
    args = arguments(references, tmp_path / 'cold')
    args[-1] = kv_dtype
    assert tool.main(args) == 0
    summary = json.loads((tmp_path / 'cold/summary.json').read_text())
    assert summary['passed'] and summary['prefix_reuse'] == 'cold'
    assert len(summary['checks']) == 47 and all(row['cached'] == 0 for row in summary['checks'])
    monkeypatch.setattr(Client, 'mutation', 'cold-leak')
    args[args.index('--output-dir') + 1] = str(tmp_path / 'leak')
    assert tool.main(args) == 1
    failure = json.loads((tmp_path / 'leak/summary.json').read_text())
    assert not failure['complete'] and 'cold prefix' in failure['error']['message']


def test_failed_concurrent_request_is_observed_after_all_workers_retire(monkeypatch, references, tmp_path):
    class Failed(Client):
        def call(self, label, body=None, **kwargs):
            result = super().call(label, body, **kwargs)
            if label == 'concurrent-0':
                raise RuntimeError('controlled concurrent failure')
            return result

    monkeypatch.setattr(tool.live, 'Client', Failed)
    output = tmp_path / 'failure'
    assert tool.main(arguments(references, output)) == 1
    summary = json.loads((output / 'summary.json').read_text())
    assert not summary['complete'] and 'controlled concurrent failure' in summary['error']['message']
    assert {'concurrent-0', 'concurrent-1'} <= {label for label, _, _ in Client.last.calls}
    assert not any(thread.name.startswith('archive-http-') for thread in threading.enumerate())


def test_final_summary_failure_preserves_primary_native_cause_and_publication_error(monkeypatch, references, tmp_path):
    primary, previous, publication = ValueError('controlled primary'), LookupError('prior'), OSError('summary failed')
    primary.__cause__, primary.__notes__ = previous, 123

    def invalid(*args, **kwargs):
        raise primary

    def fail(*args, **kwargs):
        raise publication

    monkeypatch.setattr(tool.live, 'Client', Client)
    monkeypatch.setattr(tool, 'capacity_proof', invalid)
    monkeypatch.setattr(tool.live, 'write_json', fail)
    with pytest.raises(ValueError) as error:
        tool.main(arguments(references, tmp_path / 'publication-failure'))
    assert error.value is primary
    cause = error.value.__cause__
    assert isinstance(cause, BaseExceptionGroup)
    assert previous in cause.exceptions and publication in cause.exceptions
    assert any(isinstance(item, TypeError) for item in cause.exceptions)


def test_abort_failure_still_drains_workers_and_preserves_primary_and_native_causes(monkeypatch, references, tmp_path):
    primary, previous, abort = OSError('poll failed'), LookupError('prior native cause'), RuntimeError('abort failed')
    primary.__cause__ = previous
    real_barrier, real_drain = threading.Barrier, tool.drain
    observed = []

    class BrokenBarrier(real_barrier):
        def abort(self):
            super().abort()
            raise abort

    class PollFailure(Client):
        def call(self, label, body=None, **kwargs):
            result = super().call(label, body, **kwargs)
            if label.startswith('concurrent-stats-'):
                raise primary
            return result

    def drain(tasks, error):
        try:
            return real_drain(tasks, error)
        finally:
            assert all(task.done.is_set() for task in tasks)
            observed.append(tuple(tasks))

    monkeypatch.setattr(tool.live, 'Client', PollFailure)
    monkeypatch.setattr(tool.threading, 'Barrier', BrokenBarrier)
    monkeypatch.setattr(tool, 'drain', drain)
    output = tmp_path / 'abort-failure'
    assert tool.main(arguments(references, output)) == 1
    assert observed and not any(thread.name.startswith('archive-http-') for thread in threading.enumerate())
    assert json.loads((output / 'summary.json').read_text())['error']['message'] == 'poll failed'
    assert isinstance(primary.__cause__, BaseExceptionGroup)
    def members(error):
        return [error, *(item for child in error.exceptions for item in members(child))] if isinstance(
            error, BaseExceptionGroup) else [error]

    retained = members(primary.__cause__)
    assert previous in retained and abort in retained
