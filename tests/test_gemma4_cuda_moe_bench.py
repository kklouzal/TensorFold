"""Component protocol, read-adapter and owning-outcome contracts (no CUDA)."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import gemma4_cuda_moe_bench as bench


def test_fixed_schedule_has_six_whole_balanced_blocks():
    blocks = bench.schedule()
    assert len(blocks) == 6
    assert blocks[0] == (bench.PUBLIC, *bench.ARMS, bench.PUBLIC, bench.PUBLIC,
                         *reversed(bench.ARMS), bench.PUBLIC)
    for order in blocks:
        assert len(order) == 16
        assert Counter(order) == Counter({bench.PUBLIC: 4, **dict.fromkeys(bench.ARMS, 2)})
        for name in (bench.PUBLIC, *bench.ARMS):
            positions = [index for index, value in enumerate(order) if value == name]
            assert sum(positions) / len(positions) == 7.5
    for name in bench.ARMS:
        assert Counter(index % 8 for order in blocks for index, value in enumerate(order) if value == name) == dict.fromkeys(range(1, 7), 2)


def test_route_contract_preserves_eight_distinct_ids_and_full_scan():
    for rows in bench.ROWS:
        scan = [bench.routes(rows, index, 'scan') for index in range(16)]
        assert {expert for frame in scan for row in frame for expert in row} == set(range(128))
        assert all(len(row) == len(set(row)) == 8 for frame in scan for row in frame)
        assert all(row == list(range(8)) for index in range(16) for row in bench.routes(rows, index, 'hot'))
    assert bench.routes(5, 3, 'scan')[:2] == [list(range(120, 128)), list(range(8))]
    assert bench.pair_members(((2, 0, 1), (0, 2, 0))) == [1, 3, 5, 2, 0, 4]
    with pytest.raises(ValueError):
        bench.routes(5, 16, 'scan')


@pytest.mark.parametrize('damage', ('boolean-schema', 'boolean-bytes', 'bad-sha', 'cumulative-bytes'))
def test_authority_rejects_wrong_types_and_cumulative_work(damage):
    value = {'schema': 1, 'checkpoint_revision': 'pinned', 'files': {'one': {'sha256': 'a' * 64, 'bytes': 6}}}
    if damage == 'boolean-schema':
        value['schema'] = True
    elif damage == 'boolean-bytes':
        value['files']['one']['bytes'] = True
    elif damage == 'bad-sha':
        value['files']['one']['sha256'] = 'g' * 64
    else:
        value['files']['two'] = {'sha256': 'b' * 64, 'bytes': 6}
    with pytest.raises(ValueError):
        bench.validate_authority(value, 10)


def test_bound_calls_keep_checked_public_private_argument_contracts():
    observed = []
    ext = SimpleNamespace(fused_up=lambda *values: observed.append(('public', values)),
                          gemma_fused_up=lambda *values: observed.append(('private', values)))
    base = (*range(11), 64, 128, 8, True)
    owner = SimpleNamespace(up_base=base)
    bindings = bench.bindings(ext, [owner])
    up, args, returned = bindings[bench.PUBLIC][0]
    up(*args)
    assert returned is owner
    assert len(args) == 19 and args[11:] == (64, 128, 1, 8, True, False, 16, 1)
    for name, config in zip(bench.ARMS, bench.CONFIGS, strict=True):
        up, args, returned = bindings[name][0]
        up(*args)
        assert returned is owner and args == (*base, *config) and len(args) == 17
    assert [kind for kind, _values in observed] == ['public', *(['private'] * 6)]


class Raw:
    def __init__(self, size, stream):
        self.data, self.stream = bytearray(size), stream

    def numpy(self):
        return self.data

    def to(self, device):
        assert self.stream.closed  # FD retirement precedes device transfer.
        self.device = device
        return self


class Stream:
    def __init__(self, data):
        self.data, self.position, self.closed = data, 0, False

    def fileno(self):
        return 17

    def seek(self, position):
        self.position = position

    def readinto(self, view):
        size = min(2, len(view), len(self.data) - self.position)
        view[:size] = self.data[self.position:self.position + size]
        self.position += size
        return size


def test_bounded_reader_handles_partial_progress_and_cumulative_work(monkeypatch):
    stream = Stream(b'abcde')
    stat = SimpleNamespace(st_dev=1, st_ino=2, st_mode=0o100600, st_size=5, st_mtime_ns=10, st_ctime_ns=11)
    opened = []

    @contextmanager
    def scope(_path):
        opened.append(True)
        stream.closed = False
        try:
            yield stream
        finally:
            stream.closed = True

    fake = SimpleNamespace(uint8='U8', empty=lambda shape, **_kw: Raw(shape[0], stream))
    monkeypatch.setattr(bench, '_stream', scope)
    monkeypatch.setattr(bench, '_descriptor_stamp', lambda _source: bench._identity(stat))
    reader = bench._CheckpointBytes(fake, {'payload': [1, 2, 0o100600, 5, 10, 11]}, 5)
    result = reader.read(Path('payload'), 1, 3, 'cuda:0')
    assert result.data == b'bcd' and result.device == 'cuda:0'
    assert reader.requested_bytes == 3 and reader.read_calls == 1
    with pytest.raises(ValueError, match='cumulative'):
        reader.read(Path('payload'), 0, 3)
    assert len(opened) == 1
    stat.st_ctime_ns = 12
    with pytest.raises(ValueError, match='identity changed'):
        reader.read(Path('payload'), 0, 1)
    assert stream.closed


def test_read_adapter_short_read_is_not_zero_filled_success(monkeypatch):
    stream = Stream(b'a')
    stat = SimpleNamespace(st_dev=1, st_ino=2, st_mode=0o100600, st_size=5, st_mtime_ns=10, st_ctime_ns=11)

    @contextmanager
    def scope(_path):
        try:
            yield stream
        finally:
            stream.closed = True

    fake = SimpleNamespace(uint8='U8', empty=lambda shape, **_kw: Raw(shape[0], stream))
    monkeypatch.setattr(bench, '_stream', scope)
    monkeypatch.setattr(bench, '_descriptor_stamp', lambda _source: bench._identity(stat))
    reader = bench._CheckpointBytes(fake, {'payload': [1, 2, 0o100600, 5, 10, 11]}, 5)
    with pytest.raises(OSError, match='invalid progress'):
        reader.read(Path('payload'), 0, 5)
    assert stream.closed and reader.requested_bytes == 5


def test_buffered_reader_integrates_real_cpu_storage_and_owned_file(tmp_path):
    import torch

    path = tmp_path / 'bytes.json'
    bench.write_json(path, {'value': 1})
    expected = b'{\n "value": 1\n}\n'
    assert path.stat().st_size == len(expected)
    reader = bench._CheckpointBytes(torch, {str(path): bench._stamp(path)}, len(expected))
    initialized = torch.cuda.is_initialized()
    actual = reader.read(path, 0, len(expected))
    assert actual.dtype == torch.uint8 and actual.device.type == 'cpu'
    assert actual.tolist() == list(expected)
    assert int(actual[:4].view(torch.int32)[0]) == int.from_bytes(expected[:4], 'little')
    assert torch.cuda.is_initialized() == initialized
    assert reader.requested_bytes == len(expected) and reader.read_calls == 1


def _roots(error):
    if isinstance(error, BaseExceptionGroup):
        return [root for nested in error.exceptions for root in _roots(nested)]
    return [error]


def test_retired_transport_preserves_primary_native_and_cleanup_roots():
    primary, prior, context, cleanup = RuntimeError('operation'), ValueError('native'), KeyError('context'), OSError('close')
    primary.__cause__, primary.__context__ = prior, context
    with pytest.raises(RuntimeError) as caught:
        bench._raise_retired(primary, prior, context, True, [None, cleanup])
    assert caught.value is primary
    assert all(any(root is expected for root in _roots(primary.__cause__)) for expected in (prior, context, cleanup))


def test_diagnostic_exhaustion_keeps_primary_and_raw_links(monkeypatch):
    primary, prior, allocation = RuntimeError('operation'), OSError('native'), MemoryError()

    def exhaust(_primary, _errors):
        raise allocation

    monkeypatch.setattr(bench, 'raise_failures', exhaust)
    with pytest.raises(RuntimeError) as caught:
        bench._raise_retired(primary, prior, None, False, [ValueError('retirement')])
    assert caught.value is primary and primary.__cause__ is allocation
    frame = primary.__traceback__
    assert any(item.tb_frame.f_locals.get('prior_cause') is prior for item in _tracebacks(frame))


def _tracebacks(frame):
    while frame is not None:
        yield frame
        frame = frame.tb_next


def test_main_publication_failure_does_not_replace_operation(tmp_path, monkeypatch):
    primary, publication = RuntimeError('operation'), OSError('publish')
    snapshots = []

    def fail_run(_args, _report):
        raise primary

    def publish(_path, value):
        snapshots.append(value['status'])
        if len(snapshots) == 2:
            raise publication

    monkeypatch.setattr(bench, 'run', fail_run)
    monkeypatch.setattr(bench, 'write_json', publish)
    argv = ['--model', 'unused', '--layers', '0,1', '--authority', 'unused', '--authority-sha256', 'a' * 64,
            '--max-input-bytes', '1', '--max-payload-read-bytes', '1', '--retained-device-tensor-budget-bytes', '1',
            '--min-device-free-bytes', '1', '--min-host-available-bytes', '1', '--threads', '1', '--inter-op-threads', '1',
            '--synthetic-inputs', '--out', str(tmp_path / 'new.json')]
    with pytest.raises(RuntimeError) as caught:
        bench.main(argv)
    assert caught.value is primary and snapshots == ['RUNNING', 'FAILED']
    assert any(root is publication for root in _roots(primary.__cause__))


class LibraryContext:
    """Observe our owning adapter's calls, not Torch's capture algorithm."""
    def __init__(self, exit_error=None, *, enter_error=None, suppress=False):
        self.exit_error, self.enter_error, self.suppress = exit_error, enter_error, suppress
        self.entries, self.exits = 0, []
        self.value = object()

    def __enter__(self):
        self.entries += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self.value

    def __exit__(self, error_type, error, traceback):
        self.exits.append((error_type, error, traceback))
        if self.exit_error is not None:
            if error is not None:
                # Library exit work may alter native correlation before failing.
                BaseException.__cause__.__set__(error, None)
                BaseException.__context__.__set__(error, None)
            raise self.exit_error
        return self.suppress


def test_owned_context_keeps_enter_value_and_library_exit_protocol():
    context = LibraryContext()
    # The with protocol resolves special methods on the type, not the instance.
    context.__enter__ = context.__exit__ = None
    with bench._OwnedContext(context) as value:
        assert value is context.value
    assert context.entries == 1 and context.exits == [(None, None, None)]
    primary = RuntimeError('explicit library suppression')
    suppressing = LibraryContext(suppress=True)
    with bench._OwnedContext(suppressing):
        raise primary
    assert suppressing.entries == 1
    assert suppressing.exits[0][:2] == (RuntimeError, primary)
    assert suppressing.exits[0][2] is not None


@pytest.mark.parametrize('primary', (MemoryError('body exhaustion'), KeyboardInterrupt()))
def test_owned_context_exit_cannot_replace_body_or_native_roots(primary):
    prior, native_context, cleanup = OSError('prior'), KeyError('prior context'), RuntimeError('capture exit')
    primary.__cause__, primary.__context__ = prior, native_context
    context = LibraryContext(cleanup)
    with pytest.raises(type(primary)) as caught, bench._OwnedContext(context):
        raise primary
    assert caught.value is primary and context.entries == 1 and len(context.exits) == 1
    assert context.exits[0][:2] == (type(primary), primary)
    assert context.exits[0][2] is not None
    assert all(any(root is expected for root in _roots(primary.__cause__))
               for expected in (prior, native_context, cleanup))


def test_nested_contexts_retain_body_and_both_independent_exit_roots():
    primary, prior = RuntimeError('bank preparation'), KeyError('prior')
    primary.__cause__ = prior
    stream_error, inference_error = OSError('stream unwind'), ValueError('inference unwind')
    stream, inference = LibraryContext(stream_error), LibraryContext(inference_error)
    with pytest.raises(RuntimeError) as caught, bench._OwnedContext(stream), bench._OwnedContext(inference):
        raise primary
    assert caught.value is primary
    assert stream.entries == inference.entries == len(stream.exits) == len(inference.exits) == 1
    assert stream.exits[0][:2] == inference.exits[0][:2] == (RuntimeError, primary)
    assert all(any(root is expected for root in _roots(primary.__cause__))
               for expected in (prior, stream_error, inference_error))


def test_context_enter_failure_never_calls_exit_and_exit_only_failure_is_exact():
    enter_error, exit_error = MemoryError('enter'), OSError('exit')
    entering = LibraryContext(enter_error=enter_error)
    with pytest.raises(MemoryError) as caught, bench._OwnedContext(entering):
        pytest.fail('body must not run after failed entry')
    assert caught.value is enter_error and entering.entries == 1 and entering.exits == []
    exiting = LibraryContext(exit_error)
    with pytest.raises(OSError) as caught, bench._OwnedContext(exiting):
        pass
    assert caught.value is exit_error and exiting.exits == [(None, None, None)]


def test_context_diagnostic_exhaustion_keeps_raw_body_exit_and_prior(monkeypatch):
    primary, prior, cleanup, allocation = RuntimeError('body'), KeyError('prior'), OSError('exit'), MemoryError()
    primary.__cause__ = prior
    context = LibraryContext(cleanup)

    def exhaust(*_args):
        raise allocation

    monkeypatch.setattr(bench, '_raise_retired', exhaust)
    with pytest.raises(RuntimeError) as caught, bench._OwnedContext(context):
        raise primary
    assert caught.value is primary and primary.__cause__ is allocation
    frames = [item.tb_frame.f_locals for item in _tracebacks(primary.__traceback__)]
    assert any(frame.get('prior_cause') is prior and isinstance(frame.get('self'), bench._OwnedContext)
               and frame['self']._statuses[0] is cleanup for frame in frames)
    assert context.entries == 1 and len(context.exits) == 1


def test_context_native_field_snapshot_ignores_hostile_exception_getters():
    class HostileError(RuntimeError):
        def __getattribute__(self, name):
            if name in ('__cause__', '__context__', '__suppress_context__'):
                raise AssertionError('exception hook must not run')
            return super().__getattribute__(name)

    primary, cleanup = HostileError('body'), OSError('exit')
    context = LibraryContext(cleanup)
    with pytest.raises(HostileError) as caught, bench._OwnedContext(context):
        raise primary
    assert caught.value is primary and len(context.exits) == 1
    assert any(root is cleanup for root in _roots(BaseException.__cause__.__get__(primary)))
