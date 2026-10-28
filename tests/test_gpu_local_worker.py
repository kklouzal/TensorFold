"""CPU-only boundary, exec identity/FD and thread-affinity inheritance controls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from tools import gpu_local_worker as worker


class Executed(BaseException):
    pass


def _document(cpus=('0-1', '2-3')):
    return {'schema': worker.SCHEMA, 'ranks': [
        {'rank': rank, 'gpu_uuid': f'GPU-00000000-0000-0000-0000-{rank:012x}',
         'pci_bus_id': f'0000:{rank + 1:02x}:00.0', 'numa_node': rank // 4,
         'local_cpulist': cpus[rank // 4]} for rank in range(8)]}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    document = _document()
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps(document))
    args = NS(policy=policy, policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest(), rank=0,
              mode='bind', receipt=tmp_path / 'bootstrap.json', command=['--', sys.executable, 'literal space;data'])
    devices = [{'index': str(r['rank']), 'gpu_uuid': r['gpu_uuid'], 'pci_bus_id': r['pci_bus_id']}
               for r in document['ranks']]
    topology = {'numa_node': 0, 'local_cpulist': '0-1', 'online_cpulist': '0-3'}
    mask, calls = {'current': {0, 1, 2, 3}}, []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.delenv('TENSORFOLD_TP_FAILURE_FD', raising=False)
    monkeypatch.setattr(worker, 'gpu_devices', lambda: devices)
    monkeypatch.setattr(worker, 'local_topology', lambda _bus: topology.copy())
    monkeypatch.setattr(worker, 'thread_ids', lambda: [os.getpid()])
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _tid: mask['current'].copy())
    def set_mask(tid, cpus):
        assert tid == 0
        calls.append(set(cpus))
        mask['current'] = set(cpus)
    def execvpe(file, command, env):
        calls.append((file, command, dict(env)))
        raise Executed()
    monkeypatch.setattr(os, 'sched_setaffinity', set_mask)
    monkeypatch.setattr(os, 'execvpe', execvpe)
    return args, document, devices, topology, mask, calls


@pytest.mark.parametrize('mode', ['bind', 'observe-reference'])
def test_exec_carries_exact_mode_identity_and_argv(prepared, mode, monkeypatch):
    args, _, _, _, _, calls = prepared
    args.mode = mode
    identity = worker.process_identity()
    with pytest.raises(Executed):
        worker.execute(args)
    receipt = json.loads(args.receipt.read_text())
    assert receipt['identity'] == identity and not receipt['torch_loaded_before_bind']
    assert receipt['actual_cpus'] == ('0-1' if mode == 'bind' else '0-3')
    assert calls[-1][0:2] == (sys.executable, [sys.executable, 'literal space;data'])
    assert receipt['input_cuda_visible_devices'] == '0'
    assert calls[-1][2]['CUDA_VISIBLE_DEVICES'] == receipt['exec_cuda_visible_devices'] == receipt['gpu']['gpu_uuid']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', receipt['exec_cuda_visible_devices'])
    assert len(calls) == (2 if mode == 'bind' else 1)
    assert worker.sample(args.receipt, rank=0, policy_sha=args.policy_sha, mode=mode)['mode'] == mode


@pytest.mark.parametrize('fault', ['hash', 'visible', 'GPU', 'empty', 'threads', 'torch'])
def test_prebind_rejections_do_not_mutate_or_exec(prepared, monkeypatch, fault):
    args, _, devices, _, mask, calls = prepared
    if fault == 'hash':
        args.policy_sha = '0' * 64
    elif fault == 'visible':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    elif fault == 'GPU':
        devices[0]['gpu_uuid'] = devices[1]['gpu_uuid']
    elif fault == 'empty':
        mask['current'] = {2, 3}
    elif fault == 'threads':
        monkeypatch.setattr(worker, 'thread_ids', lambda: [1, 2])
    else:
        monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises((ValueError, RuntimeError)):
        worker.execute(args)
    assert not calls and not args.receipt.exists()


def test_changed_topology_after_binding_cannot_exec(prepared, monkeypatch):
    args, _, _, topology, _, calls = prepared
    readings = 0
    def changed(_bus):
        nonlocal readings
        readings += 1
        return topology.copy() if readings == 1 else {**topology, 'online_cpulist': '0-2'}
    monkeypatch.setattr(worker, 'local_topology', changed)
    with pytest.raises(RuntimeError, match='changed during'):
        worker.execute(args)
    assert calls == [{0, 1}] and not args.receipt.exists()


def test_syscall_failure_preserves_primary_and_never_execs(prepared, monkeypatch):
    args, _, _, _, _, calls = prepared
    primary = PermissionError('controlled affinity refusal')
    def fail(*_args):
        raise primary
    monkeypatch.setattr(os, 'sched_setaffinity', fail)
    with pytest.raises(PermissionError) as caught:
        worker.execute(args)
    assert caught.value is primary and not calls and not args.receipt.exists()


def test_kernel_mask_narrowing_is_not_silently_accepted(prepared, monkeypatch):
    args, _, _, _, mask, calls = prepared
    def narrowed(*_args):
        mask['current'] = {0}
    monkeypatch.setattr(os, 'sched_setaffinity', narrowed)
    with pytest.raises(RuntimeError, match='changed during'):
        worker.execute(args)
    assert not calls and not args.receipt.exists()


def test_receipt_failure_prevents_exec(prepared, monkeypatch):
    args, _, _, _, _, calls = prepared
    primary = OSError('controlled receipt failure')
    def fail(*_args):
        raise primary
    monkeypatch.setattr(worker, 'write_receipt', fail)
    with pytest.raises(OSError) as caught:
        worker.execute(args)
    assert caught.value is primary and calls == [{0, 1}]


def test_retired_thread_census_and_escape(prepared, monkeypatch):
    monkeypatch.setattr(worker, 'thread_ids', lambda: [999999999])
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _tid: {0})
    assert worker.thread_snapshot({0})['threads'] == [{'tid': 999999999, 'retired_before_read': True}]
    monkeypatch.setattr(worker, 'thread_ids', lambda: [os.getpid()])
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _tid: {2})
    with pytest.raises(RuntimeError, match='escaped'):
        worker.thread_snapshot({0, 1})


def test_changed_mode_or_process_receipt_is_rejected(prepared, monkeypatch):
    args, _, _, _, _, _ = prepared
    with pytest.raises(Executed):
        worker.execute(args)
    with pytest.raises(RuntimeError, match='identity changed'):
        worker.sample(args.receipt, rank=0, policy_sha=args.policy_sha, mode='observe-reference')
    data = json.loads(args.receipt.read_text())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', data['exec_cuda_visible_devices'])
    data['identity']['start_ticks'] += 1
    args.receipt.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='identity changed'):
        worker.sample(args.receipt, rank=0, policy_sha=args.policy_sha, mode='bind')


def test_actual_child_exec_keeps_pid_failure_fd_exit_status_and_thread_mask(tmp_path):
    # GPU discovery/sysfs are controlled at the adapter boundary. CPU affinity,
    # exec, PID/SID, pipe inheritance and new-thread mask are actual OS behavior.
    parent_mask = os.sched_getaffinity(0)
    cpu = min(parent_mask)
    document = _document((str(cpu), str(cpu)))
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps(document))
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    read_fd, write_fd = os.pipe()
    probe = tmp_path / 'probe.json'
    result = tmp_path / 'result.json'
    target = ("import json,os,threading,pathlib; rows=[]; "
              "t=threading.Thread(target=lambda:rows.append(sorted(os.sched_getaffinity(0)))); "
              "t.start();t.join(); "
              f"pathlib.Path({str(result)!r}).write_text(json.dumps({{'pid':os.getpid(),'mask':sorted(os.sched_getaffinity(0)),'thread':rows,'session':os.getsid(0),'visible':os.environ['CUDA_VISIBLE_DEVICES']}})); "
              f"os.write({write_fd},b'F'); raise SystemExit(7)")
    bootstrap = f'''
import json,os,sys
from types import SimpleNamespace as NS
sys.path.insert(0,{str(Path(worker.__file__).parent)!r})
import gpu_local_worker as w
p=w.policy({str(policy)!r},{digest!r})
w.gpu_devices=lambda:[dict(index=str(r['rank']),gpu_uuid=r['gpu_uuid'],pci_bus_id=r['pci_bus_id']) for r in p['ranks']]
w.local_topology=lambda bus:dict(numa_node=0,local_cpulist={str(cpu)!r},online_cpulist={str(cpu)!r})
w.execute(NS(policy={str(policy)!r},policy_sha={digest!r},rank=0,mode='bind',receipt={str(probe)!r},command=[sys.executable,'-c',{target!r}]))
'''
    child = None
    try:
        child = subprocess.Popen([sys.executable, '-c', bootstrap], pass_fds=(write_fd,), start_new_session=True,
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'TENSORFOLD_TP_FAILURE_FD': str(write_fd)})
        os.close(write_fd)
        write_fd = None
        assert child.wait(timeout=30) == 7
        assert os.read(read_fd, 2) == b'F'
        actual = json.loads(result.read_text())
        assert actual == {'pid': child.pid, 'mask': [cpu], 'thread': [[cpu]], 'session': child.pid, 'visible': document['ranks'][0]['gpu_uuid']}
        assert json.loads(probe.read_text())['identity']['pid'] == child.pid
        assert os.sched_getaffinity(0) == parent_mask
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        if write_fd is not None:
            os.close(write_fd)
        os.close(read_fd)


def test_exec_failure_notifies_owned_cohort_and_preserves_primary(prepared, monkeypatch):
    args, _, _, _, _, _ = prepared
    primary = OSError('controlled exec failure')
    def fail(*_args):
        raise primary
    monkeypatch.setattr(os, 'execvpe', fail)
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK)
    try:
        os.set_inheritable(write_fd, True)
        monkeypatch.setenv('TENSORFOLD_TP_FAILURE_FD', str(write_fd))
        with pytest.raises(OSError) as caught:
            worker.main(['--policy', str(args.policy), '--policy-sha', args.policy_sha,
                         '--rank', '0', '--receipt', str(args.receipt), '--', sys.executable])
        assert caught.value is primary
        assert os.read(read_fd, 2) == b'F'
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_notification_failure_does_not_replace_exec_primary(prepared, monkeypatch):
    from tensorfold.cuda import cohort

    args, _, _, _, _, _ = prepared
    primary, secondary = OSError('controlled exec failure'), OSError('controlled notification failure')
    prior = ValueError('prior native root')
    primary.__cause__ = prior
    def fail_exec(*_args):
        raise primary
    def fail_notify():
        raise secondary
    monkeypatch.setattr(os, 'execvpe', fail_exec)
    monkeypatch.setattr(cohort, 'notify_failure', fail_notify)
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK)
    try:
        os.set_inheritable(write_fd, True)
        monkeypatch.setenv('TENSORFOLD_TP_FAILURE_FD', str(write_fd))
        with pytest.raises(OSError) as caught:
            worker.main(['--policy', str(args.policy), '--policy-sha', args.policy_sha,
                         '--rank', '0', '--receipt', str(args.receipt), '--', sys.executable])
        assert caught.value is primary
        group = primary.__cause__
        assert isinstance(group, BaseExceptionGroup) and prior in group.exceptions and secondary in group.exceptions
    finally:
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.parametrize('operation', ['read', 'write'])
def test_file_io_preserves_operation_primary_native_root_and_close_failure(monkeypatch, operation):
    primary, prior, cleanup = OSError('primary IO failure'), ValueError('prior IO root'), OSError('close failure')
    primary.__cause__ = prior
    calls = []
    class File:
        def read(self, _limit):
            calls.append('read')
            raise primary
        def write(self, _raw):
            calls.append('write')
            raise primary
        def close(self):
            calls.append('close')
            raise cleanup
    monkeypatch.setattr(worker.Path, 'open', lambda *_args: File())
    with pytest.raises(OSError) as caught:
        if operation == 'read':
            worker.file_operation('unused', 'rb', lambda f: f.read(4))
        else:
            worker.write_receipt('unused', {'evidence': True})
    assert caught.value is primary and calls == [operation, 'close']
    assert isinstance(primary.__cause__, BaseExceptionGroup)
    assert prior in primary.__cause__.exceptions and cleanup in primary.__cause__.exceptions


def test_short_receipt_write_is_failed_and_closed(monkeypatch):
    calls = []
    class File:
        def write(self, raw):
            return len(raw) - 1
        def close(self):
            calls.append('close')
    monkeypatch.setattr(worker.Path, 'open', lambda *_args: File())
    with pytest.raises(OSError, match='short worker affinity'):
        worker.write_receipt('unused', {'evidence': True})
    assert calls == ['close']
