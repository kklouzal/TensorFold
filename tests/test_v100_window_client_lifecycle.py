"""Actual file-reader API and ordered eight-request client lifetime controls."""
import ast
import hashlib
import inspect
import json
import threading

import pytest

from tests.test_v100_target_window_http_contract import actual
from tests.test_v100_target_window_teacher import plan
from tools import openai_protocol as protocol
from tools import v100_owned_qualification as owner
from tools import v100_rotorquant_quality as frames
from tools import v100_target_window_teacher as window
from tools import worker_lifetime


@pytest.mark.parametrize('stop', [None, 1, 2, 3, 4])
def test_actual_reader_optional_control_and_owned_file_lifetime(tmp_path, stop):
    path = tmp_path/'input.json'
    path.write_bytes(b'{"index":0}')
    failure = RuntimeError('cooperative stop')
    class Control:
        calls = 0
        def check(self):
            self.calls += 1
            if self.calls == stop:
                raise failure
    control = Control()
    if stop is None:
        assert frames._owned_teacher_read(path, 100, control) == path.read_bytes()
        assert control.calls == 4
        assert frames._owned_teacher_read(path, 100) == path.read_bytes()
    else:
        with pytest.raises(RuntimeError) as caught:
            frames._owned_teacher_read(path, 100, control)
        assert caught.value is failure
    # A failed controlled read neither modifies nor consumes the held input.
    assert frames._owned_teacher_read(path, 100) == b'{"index":0}'
    with pytest.raises(ValueError, match='bounded immutable'):
        frames._owned_teacher_read(path, 1)
    link = tmp_path/'symlink'
    link.symlink_to(path)
    with pytest.raises(OSError):
        frames._owned_teacher_read(link, 100)


def test_all_static_window_frame_api_calls_bind_actual_functions():
    tree = ast.parse(inspect.getsource(window))
    checked = []
    modules = {'frames': frames, 'owner': owner, 'protocol': protocol,
               'numa': window.numa, 'legacy': window.legacy, 'numerical': window.numerical}
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in modules):
            # owner is also a local retention-record name in other functions;
            # audit it only under the actual module-importing function scope.
            if node.func.value.id in ('owner', 'protocol'):
                scope, imports = node, []
                while scope in parents:
                    scope = parents[scope]
                    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        imports.extend(item for item in ast.walk(scope)
                                       if isinstance(item, (ast.Import, ast.ImportFrom)))
                if not any((alias.asname or alias.name) == node.func.value.id
                           for item in imports for alias in item.names):
                    continue
            assert not any(isinstance(arg, ast.Starred) for arg in node.args)
            assert all(key.arg is not None for key in node.keywords)
            function = getattr(modules[node.func.value.id], node.func.attr)
            inspect.signature(function).bind(*[None for _ in node.args],
                                            **{key.arg: None for key in node.keywords})
            checked.append((node.func.value.id, node.func.attr, node.lineno))
    assert sum(module == 'frames' and name == '_owned_teacher_read'
               for module, name, _ in checked) == 8
    assert {'frames', 'owner', 'protocol', 'numa', 'legacy', 'numerical'} == {row[0] for row in checked}


def client_fixture(monkeypatch, tmp_path, *, malformed=False):
    import resource
    monkeypatch.setattr(resource, 'getrlimit', lambda kind: (8 << 20, 64 << 20))
    data = plan(4)
    retained = actual()
    selected = {'data': retained['client'], 'prepared_operation': {},
                'prepared_manifest': retained['prepared'], 'sha256': '2'*64}
    teacher = {'data': data, 'input_sha256': '1'*64}
    client = window.WindowClient(teacher, selected, tmp_path)
    directory = tmp_path/'window-teacher-rank0'
    directory.mkdir()
    release, entered = threading.Event(), []
    lock = threading.Lock()
    control = frames._teacher_read_control(10, client.cancelled)
    class HTTP:
        def call(self, name, body=None, *, path, object_response=False, read_control=None):
            assert read_control is control
            if path == '/stats':
                assert object_response and len(entered) == 8
                return retained['stats']
            index = int(name.rsplit('-', 1)[1])
            case = data['sequences'][index]
            with lock:
                entered.append(index)
                assert entered == list(range(index+1))
                record = {'schema': 'tensorfold.target-window-teacher-admission-v1',
                          'index': -1 if malformed else index, 'rank': 0,
                          'case_id': case['case_id'], 'input_sha256': teacher['input_sha256']}
                frames._owned_teacher_publish(directory/f'admission-{index}.json', record, 1 << 20)
                if len(entered) == 8:
                    release.set()
            assert release.wait(5), 'test releases every accepted HTTP callback'
            ids = case['forced_ids']
            lanes = [row['lanes'][index] for row in data['rounds']]
            return {'choices': [{'text': 'forced', 'index': 0, 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': len(case['prompt_ids']), 'completion_tokens': len(ids),
                              'total_tokens': len(case['prompt_ids'])+len(ids),
                              'prompt_tokens_details': {'cached_tokens': 0}},
                    'tensorfold': {'token_ids': ids, 'drafts': True, 'cached': 0,
                                  'drafted': sum(row['rows']-1 for row in lanes),
                                  'accepted': sum(row['keep']-1 for row in lanes),
                                  'token_sha': hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()[:12]}}
    def open_client(actual, role):
        assert actual is client and role == 'teacher'
        before = window.validate_service(retained['stats'], selected['data'], retained['instance'],
                                         prepared_manifest=selected['prepared_manifest'])
        return HTTP(), control, before, retained['instance']
    monkeypatch.setattr(owner, '_owned_client_open', open_client)
    monkeypatch.setattr(window, 'collect', lambda operation, output: {'fixture': True})
    return client, release, entered


def test_full_client_ordered_eight_admissions_use_actual_reader(monkeypatch, tmp_path):
    client, _, entered = client_fixture(monkeypatch, tmp_path)
    client.run()
    assert entered == list(range(8)) and client.post_attempts == 8
    assert client.failure() is None
    report = json.loads((tmp_path/'teacher-client-report.json').read_text())
    assert report['complete'] is True and len(report['cases']) == 8
    assert not report['quality_acceptance'] and not report['P3_valid']


def test_main_failure_visible_before_blocked_callback_drain(monkeypatch, tmp_path):
    client, release, entered = client_fixture(monkeypatch, tmp_path, malformed=True)
    draining = threading.Event()
    actual_drain = worker_lifetime.drain
    def drain(tasks, *, primary=None):
        draining.set()
        return actual_drain(tasks, primary=primary)
    monkeypatch.setattr(worker_lifetime, 'drain', drain)
    task = worker_lifetime.Task(client.run)
    task.start()
    try:
        assert draining.wait(3)
        primary = client.failure()
        assert isinstance(primary, ValueError)
        assert 'ordered admission' in str(primary)
        assert client.cancelled.is_set() and not task.done.is_set()
        assert entered == [0] and client.post_attempts == 1
    finally:
        release.set()
        assert task.done.wait(3)
        task.thread.join()
    assert task.error is primary and client.result is None


def test_atomic_admission_absent_until_actual_writer_finishes(monkeypatch, tmp_path):
    target = tmp_path/'admission-0.json'
    partial, release = threading.Event(), threading.Event()
    original = json.JSONEncoder.iterencode
    def paused(encoder, value):
        pieces = iter(original(encoder, value))
        yield next(pieces)
        partial.set()
        assert release.wait(3)
        yield from pieces
    monkeypatch.setattr(json.JSONEncoder, 'iterencode', paused)
    task = worker_lifetime.Task(lambda: frames._owned_teacher_publish(target, {'index': 0}, 1024))
    task.start()
    try:
        assert partial.wait(3)
        assert not target.is_file()  # The production client uses this predicate.
        assert (tmp_path/'.admission-0.json.pending/payload.json').is_file()
    finally:
        release.set()
        assert task.done.wait(3)
        task.thread.join()
    assert task.error is None
    assert frames._owned_teacher_object(frames._owned_teacher_read(target, 1024)) == {'index': 0}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('fault', ['existing', 'size', 'control'])
def test_atomic_admission_no_clobber_and_owned_failure_cleanup(tmp_path, fault):
    target = tmp_path/'admission-0.json'
    failure = RuntimeError('cancel before publication')
    class Control:
        def check(self):
            raise failure
    if fault == 'existing':
        target.write_bytes(b'original')
        with pytest.raises(FileExistsError):
            frames._owned_teacher_publish(target, {'index': 0}, 1024)
        assert target.read_bytes() == b'original'
    elif fault == 'size':
        with pytest.raises(ValueError, match='prepaid schema bound'):
            frames._owned_teacher_publish(target, {'index': 0}, 1)
    else:
        with pytest.raises(RuntimeError) as caught:
            frames._owned_teacher_publish(target, {'index': 0}, 1024, Control())
        assert caught.value is failure
    assert not (tmp_path/'.admission-0.json.pending').exists()
    if fault != 'existing':
        assert not target.exists()


def test_atomic_admission_primary_retains_cleanup_failures(monkeypatch, tmp_path):
    from pathlib import Path
    target = tmp_path/'admission-0.json'
    original = Path.unlink
    def fail_cleanup(path, *, missing_ok=False):
        if path.name == 'payload.json':
            raise OSError('retained cleanup failure')
        return original(path, missing_ok=missing_ok)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, 'unlink', fail_cleanup)
        with pytest.raises(ValueError, match='prepaid schema bound') as caught:
            frames._owned_teacher_publish(target, {'index': 0}, 1)
    assert any('cleanup' in note for note in caught.value.__notes__)
    assert not target.exists()
    staging = tmp_path/'.admission-0.json.pending'
    (staging/'payload.json').unlink()
    staging.rmdir()
