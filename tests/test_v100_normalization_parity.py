"""CPU diagnostic-contract controls; no live-model or CUDA parity claims."""
from types import SimpleNamespace as NS

import pytest
import torch

from tools import v100_owned_qualification as owned
from tensorfold.cuda.sm70 import reference_owner_tiles as runtime
from tensorfold.families.qwen4_exp.cuda.history_owner import merge_statistics
from tests.test_sm70_global_normalization import aligned_workspace, sample

pytestmark = pytest.mark.torch


class Engine:
    def __init__(self, owner):
        self.rank, self._closed, self._abort_comm = 0, False, False
        self.device = owner.partial.device
        self.error, self.close_calls, self.token = None, 0, object()
        self.observations = NS(record=lambda **kwargs: {'old': True})
        self.multi = NS(buf=NS(history_owner=owner), pbuf=None, mbuf=None,
                        controller=NS(work_observations=self.observations))

    def close(self, **kwargs):
        self.close_calls += 1
        if self.error is not None:
            raise self.error
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)  # Test owner retirement, not an extra helper fence.
        self._closed = True
        return self.token


def fixture(tmp_path, monkeypatch, *, fault=None, shared_status=False, rows=5, maximum=None,
            device='cpu', group=0, capture_writer=None, phase=0):
    maximum = rows if maximum is None else maximum
    local = aligned_workspace(maximum, device)
    receive = aligned_workspace(8 * maximum, device).view(8, maximum, 12, 256)
    output = torch.empty((maximum, 12, 2), device=device)
    owner = NS(rows=maximum, partial=local, partials=receive, global_stats=output,
               reference_tiles=NS(normalization_workspace=local, normalization_output=output))
    engine = Engine(owner)
    data = receive.flatten()[:8 * rows * 24].view(8, rows, 12, 2)
    data.copy_(sample(rows, device, group, phase=phase))
    status = torch.tensor([0, -2], dtype=torch.int32, device=device)
    second = status if shared_status else torch.tensor([2, 9], dtype=torch.int32, device=device)
    cut = min(2, rows)
    ranges = [(status, 0, cut)] + ([(second, cut, rows)] if cut < rows else [])
    made, saved, failures = [], {}, []
    def candidate(*args):
        result = runtime.merge_global_statistics(*args)
        if fault == 'input':
            args[0].flatten()[0] = 19.
        made.append(result)
        return result
    def scalar(*args, **kwargs):
        result = merge_statistics(*args, **kwargs)
        if fault == 'value':
            result[0].view(torch.int32)[0].bitwise_xor_(1)
        elif fault == 'status':
            kwargs['status'][1].fill_(42)
        return result
    def writer(path, value):
        saved[path.stem] = value.clone()
        path.write_bytes(value.contiguous().numpy().tobytes())
        return {'file': path.name, 'bytes': path.stat().st_size}
    module = NS(merge_global_statistics=candidate, __file__=runtime.__file__)
    monkeypatch.setattr(owned, '_observer_failure', lambda engine, error: failures.append(error))
    witness = owned._NormalizationParity(engine, tmp_path, {'maximum_rows': 256, 'maximum_layer': 48},
                                           module=module, scalar=scalar, writer=writer if capture_writer is None else capture_writer)
    witness.install()
    return engine, witness, module, data, local, output, ranges, made, saved, failures


def test_same_input_shadow_status_and_return_identity_preserved(tmp_path, monkeypatch):
    engine, witness, module, data, local, output, ranges, made, saved, failures = fixture(tmp_path, monkeypatch)
    # An independent call on cloned statuses checks that the oracle never mutates
    # the candidate's actual State status words.
    expected_ranges = [(s.clone(), first, last) for s, first, last in ranges]
    runtime.merge_global_statistics(data, aligned_workspace(5, 'cpu'), torch.empty_like(output), 0, expected_ranges, 17)
    returned = module.merge_global_statistics(data, local, output, 0, ranges, 17)
    assert returned is made[0] and all(a is b for a, b in zip(returned, made[0]))
    assert all(torch.equal(a[0], b[0]) for a, b in zip(ranges, expected_ranges))
    assert not failures and not saved
    record = engine.observations.record()['normalization_parity']
    assert record['diagnostic_only'] and record['counts']['decode:17'] == {'calls': 1, 'rows': 5, 'mismatches': 0}
    old_callback = module.merge_global_statistics
    assert engine.close() is engine.token
    assert witness.closed and not witness.roles and witness.engine is None and witness.observations is None
    assert module.merge_global_statistics is not witness
    with pytest.raises(RuntimeError, match='already retired'):
        old_callback(data, local, output, 0, ranges, 17)
    assert not failures
    assert (witness.directory / 'summary.json').exists()


@pytest.mark.parametrize('fault', ['value', 'status', 'input'])
def test_wrong_original_bit_or_status_or_input_change_captures_before_fatal(tmp_path, monkeypatch, fault):
    engine, witness, module, data, local, output, ranges, made, saved, failures = fixture(tmp_path, monkeypatch, fault=fault)
    before = [s.clone() for s, _, _ in ranges]
    expected_ranges = [(s.clone(), first, last) for s, first, last in ranges]
    runtime.merge_global_statistics(data, aligned_workspace(5, 'cpu'), torch.empty_like(output), 0, expected_ranges, 17)
    with pytest.raises(RuntimeError, match='raw-bit/status parity mismatch') as caught:
        module.merge_global_statistics(data, local, output, 0, ranges, 17)
    assert len(made) == 1 and failures == [caught.value]
    assert set(saved) == {'gathered-before', 'gathered-after', 'new-max', 'new-den', 'old-max', 'old-den',
                          'status-before', 'status-new', 'status-old'}
    assert torch.equal(saved['status-before'].view(torch.int32), torch.stack(before))
    assert all(torch.equal(a[0], b[0]) for a, b in zip(ranges, expected_ranges))
    assert (witness.directory / 'mismatch.json').exists()
    assert witness.published['counts']['decode:17']['mismatches'] == 1
    engine.close()


def test_same_status_pointer_is_one_shadow_owner(tmp_path, monkeypatch):
    _, _, module, data, local, output, ranges, _, _, failures = fixture(tmp_path, monkeypatch, shared_status=True)
    module.merge_global_statistics(data, local, output, 0, ranges, 17)
    assert not failures


def test_close_failure_retains_owned_views_and_original_exception(tmp_path, monkeypatch):
    engine, witness, module, _, _, _, _, _, _, failures = fixture(tmp_path, monkeypatch)
    primary = OSError('close failed before retirement')
    engine.error = primary
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is primary and failures == [primary]
    assert engine.close_calls == 1 and witness.roles and module.merge_global_statistics is witness
    engine.error = None  # Explicit owner retry, never performed by the observer.
    assert engine.close() is engine.token
    assert engine.close_calls == 2 and not witness.roles


def test_host_and_gpu_tail_plan_and_preconstructor_capability_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(owned, '_history_resources', lambda *a: {'per_rank_payload_bytes': 100,
        'global_host_obligation_bytes': 800, 'receipt_bytes_limit': 200000, 'materialized_credit_bytes': 0})
    plan = owned._normalization_resources(tmp_path)
    assert plan['maximum_rows'] == 256 and plan['global_host_obligation_bytes'] == 8 * plan['per_rank_payload_bytes']
    assert plan['additional_device_bytes'] == plan['materialized_credit_bytes'] == 0
    monkeypatch.setattr(runtime, 'merge_global_statistics', None)
    monkeypatch.setattr(owned, '_history_resources', lambda *a: pytest.fail('capability refusal was too late'))
    with pytest.raises(ValueError, match='before construction'):
        owned._normalization_resources(tmp_path)


@pytest.mark.parametrize('extra', [[], ['--benchmark-plan', 'plan'], ['--reference-tiles'],
                                  ['--benchmark-plan', 'plan', '--reference-tiles', '--scheduler-trace', 'trace']])
def test_parity_flags_refuse_before_parent_creates_output(tmp_path, extra):
    out = tmp_path / 'new'
    with pytest.raises(SystemExit):
        owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(out), '--normalization-parity', *extra])
    assert not out.exists()


def test_parent_forwards_private_flag_and_default_does_not_install(tmp_path, monkeypatch):
    from tools import launch_cuda_tp
    seen = []
    monkeypatch.setattr(launch_cuda_tp, 'main', lambda *a, **kw: seen.append(kw['_command_prefix']) or 0)
    owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(tmp_path / 'normal'), '--', 'model'])
    assert '--normalization-parity' not in seen[-1]
    owned.main(['--gpus', '0,1,2,3,4,5,6,7', '--out', str(tmp_path / 'parity'), '--normalization-parity',
                '--reference-tiles', '--benchmark-plan', str(tmp_path / 'plan'), '--', 'model'])
    assert '--normalization-parity' in seen[-1] and '--reference-tiles' in seen[-1]


def test_alias_and_tail_budget_refuse_before_installation(tmp_path):
    local = aligned_workspace(5, 'cpu')
    output = torch.empty((5, 12, 2))
    module = NS(merge_global_statistics=runtime.merge_global_statistics, __file__=runtime.__file__)
    for name, receive, storage in [('alias', local, local), ('small', aligned_workspace(40, 'cpu'), local[:10])]:
        owner = NS(rows=5, partial=storage, partials=receive, global_stats=output,
                   reference_tiles=NS(normalization_workspace=storage, normalization_output=output))
        engine = Engine(owner)
        with pytest.raises(ValueError, match='prepared owner|unused paid local tail'):
            owned._NormalizationParity(engine, tmp_path / name, {'maximum_rows': 256, 'maximum_layer': 48}, module=module)
        assert module.merge_global_statistics is runtime.merge_global_statistics
        assert not (tmp_path / name).exists()


def test_candidate_primary_survives_publish_failure_and_notification_still_occurs(tmp_path, monkeypatch):
    engine, witness, module, data, local, output, ranges, _, _, failures = fixture(tmp_path, monkeypatch)
    primary, cleanup = OSError('candidate primary'), MemoryError('CPU publication cleanup')
    def fail(*a, **k):
        raise primary
    witness.original = fail
    monkeypatch.setattr(witness, '_publish', lambda: (_ for _ in ()).throw(cleanup))
    with pytest.raises(OSError) as caught:
        module.merge_global_statistics(data, local, output, 0, ranges, 17)
    assert caught.value is primary and failures == [primary]
    assert caught.value.__cause__ is cleanup
    assert witness.roles and not witness.active
    # Owner retirement is independent; restore publication before deliberate close.
    monkeypatch.undo()
    engine.close()


def test_retirement_publish_error_releases_views_and_notifies(tmp_path, monkeypatch):
    engine, witness, module, _, _, _, _, _, _, failures = fixture(tmp_path, monkeypatch)
    cleanup = MemoryError('retirement summary publication')
    monkeypatch.setattr(witness, '_publish', lambda: (_ for _ in ()).throw(cleanup))
    with pytest.raises(MemoryError) as caught:
        engine.close()
    assert caught.value is cleanup and failures == [cleanup]
    assert engine._closed and witness.closed and witness.roles == []
    assert witness.engine is witness.observations is None
    assert module.merge_global_statistics is not witness
    assert not (witness.directory / 'summary.json').exists()


def test_setup_failure_notifies_before_receipt_failed_suppresses_outer_notifier(tmp_path, monkeypatch):
    import os
    from tensorfold import cli
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tools import bench_concurrent
    from tests.test_v100_owned_qualification import worker_args

    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(bench_concurrent, 'load_matrix', lambda _: {'cohorts': [{'level': 1}]})
    monkeypatch.setattr(owned, '_history_resources', lambda *a: {'global_host_obligation_bytes': 800,
                                                              'per_rank_payload_bytes': 100})
    monkeypatch.setattr(owned, '_normalization_resources', lambda *a: {'global_host_obligation_bytes': 80,
                                                                    'per_rank_payload_bytes': 10})
    monkeypatch.setattr(owned, 'prepare_benchmark', lambda *a: None)
    primary = OSError('parity setup primary')
    def fail(*a):
        raise primary
    monkeypatch.setattr(owned, 'prepare_normalization_parity', fail)
    monkeypatch.setattr(owned, 'publish_receipt', lambda *a, **k: pytest.fail('failed setup published readiness'))
    engines = []
    def serve(args):
        assert args._owned_expert_qualification.host_obligation[0] == 880
        engine = NS(_abort_comm=False)
        engines.append(engine)
        args._owned_expert_qualification.before_listener(engine)
    monkeypatch.setattr(cli, 'cmd_serve', serve)
    read, write = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    monkeypatch.setenv(FAILURE_FD_ENV, str(write))
    try:
        with pytest.raises(OSError) as caught:
            owned.worker(['--normalization-parity', '--reference-tiles', '--benchmark-plan', str(plan), *worker_args(tmp_path)])
        assert caught.value is primary and engines[0]._abort_comm
        assert os.read(read, 8) == b'F'
    finally:
        os.close(write)
        os.close(read)


def test_detected_wrong_bit_remains_primary_and_capture_runs_when_publication_fails(tmp_path, monkeypatch):
    engine, witness, module, data, local, output, ranges, _, saved, failures = fixture(tmp_path, monkeypatch, fault='value')
    publication = MemoryError('publication after detected mismatch')
    original_publish = witness._publish
    monkeypatch.setattr(witness, '_publish', lambda: (_ for _ in ()).throw(publication))
    with pytest.raises(RuntimeError, match='raw-bit/status parity mismatch') as caught:
        module.merge_global_statistics(data, local, output, 0, ranges, 17)
    assert failures == [caught.value] and caught.value.__cause__ is publication
    assert len(saved) == 9 and (witness.directory / 'mismatch.json').exists()
    assert witness.failed and not witness.active
    monkeypatch.setattr(witness, '_publish', original_publish)
    engine.close()
