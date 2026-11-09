"""Stdlib-only orchestration tests; no installed runtime or GPU qualification."""
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from tools import v100_native_contract_driver as driver
from tools.worker_lifetime import raise_failures


DRIVER_PATH = str(Path(driver.__file__).resolve())


def _spawn_import_probe(path, queue):
    # A real spawn must safely re-execute the driver as __mp_main__ without
    # reading an operator spec or loading any runtime qualification dependency.
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'pytest', 'tensorfold'):
            raise AssertionError('runtime import during spawn module import: ' + name)
        return original(name, *args, **kwargs)
    with mock.patch('builtins.__import__', side_effect=guarded), mock.patch.object(Path, 'open', side_effect=AssertionError('spec IO')):
        namespace = runpy.run_path(path, run_name='__mp_main__')
    queue.put(('Guard' in namespace, 'main' in namespace))


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.evidence, self.app = [self.root / name for name in ('source', 'evidence', 'app')]
        for path in (self.source, self.evidence, self.app):
            path.mkdir()
        self.test = self.source / 'tests/cuda/test_contract.py'
        self.test.parent.mkdir(parents=True)
        self.test.write_text('def test_contract(): pass\n')
        (self.app / '__init__.py').write_text('# installed fixture\n')
        self.fd = self.app / '_fd_owner.so'
        self.fd.write_bytes(b'owned FD fixture')
        self.sdk = self.root / 'sdk.json'
        self.sdk.write_text('{}\n')
        self.spec = {'schema': driver.SCHEMA, 'mode': 'cuda', 'image_id': 'sha256:' + 'a' * 64,
            'source_commit': 'b' * 40, 'app_root': str(self.app), 'expected_app': driver.app_inventory(self.app),
            'sdk_manifest': {'path': str(self.sdk), 'sha256': driver.file_identity(self.sdk)['sha256']},
            'fd': {'sha256': driver.file_identity(self.fd)['sha256'], 'source_sha256': 'c' * 64},
            'torch': {'git_version': 'd' * 40, 'cuda': '12.9'}, 'UUIDs': ['GPU-' + '1' * 32, 'GPU-' + '2' * 32],
            'SM': [7, 0], 'modules': ['tests/cuda/test_contract.py'],
            'test_files': {'tests/cuda/test_contract.py': driver.file_identity(self.test)},
            'diagnostics': {'sample_limit': 4096, 'read_limit_bytes': 65536, 'failure_trace_limit_bytes': 65536},
            'required_native_elf_prefixes': []}
        self.events = []
        self.pytest_argv = None
        self.code = 0
        self.skips = 0
        self.failure = None
        self.primary = None
        self.mutate = None
        self.sync_error = None
        self.root_override = None
        self.module_id = self.spec['modules'][0]
        self.session = types.SimpleNamespace(shouldstop=None)
        cuda = types.SimpleNamespace(device_count=lambda: 2,
            get_device_properties=lambda i: types.SimpleNamespace(uuid=self.spec['UUIDs'][i]),
            get_device_capability=lambda i: (7, 0), synchronize=self.synchronize,
            current_device=lambda: 0, memory_allocated=lambda: 1, memory_reserved=lambda: 2,
            max_memory_allocated=lambda: 3, max_memory_reserved=lambda: 4)
        self.runtime = types.SimpleNamespace(app=types.SimpleNamespace(__file__=str(self.app / '__init__.py')),
            fd=types.SimpleNamespace(__file__=str(self.fd), _source_sha256='c' * 64),
            torch=types.SimpleNamespace(__version__='fixture', version=types.SimpleNamespace(git_version='d' * 40, cuda='12.9'), cuda=cuda),
            pytest=types.SimpleNamespace(Module=object(), main=self.pytest_main))

    def synchronize(self):
        self.events.append('synchronize')
        if self.sync_error is not None:
            raise self.sync_error

    def pytest_main(self, argv, *, plugins):
        self.pytest_argv = list(argv)
        self.events.append('pytest')
        guard = plugins[0]
        module = types.SimpleNamespace(path=self.test, nodeid=self.module_id)
        item = types.SimpleNamespace(getparent=lambda kind: module)
        guard.pytest_collection_modifyitems(self.session, types.SimpleNamespace(rootpath=self.root_override or self.source), [item])
        if self.primary is not None:
            raise self.primary
        if self.failure is not None:
            guard.pytest_runtest_logreport(types.SimpleNamespace(failed=True, nodeid=self.module_id + '::test_contract',
                                                                when='call', longrepr=self.failure))
        if self.mutate is not None:
            self.mutate()
        suite = ET.Element('testsuite', tests='1', errors='0', failures=str(int(self.code != 0)), skipped=str(self.skips))
        ET.ElementTree(suite).write(self.evidence / 'named-suite.xml')
        return self.code

    def publish(self, path, value):
        Path(path).write_text(json.dumps(value))

    def execute(self, publisher=None):
        return driver.run(self.spec, self.source, self.evidence, runtime_loader=lambda: self.runtime,
                          helpers_loader=lambda: (publisher or self.publish, raise_failures))

    def outcome(self):
        return json.loads((self.evidence / 'named-suite-result.json').read_text())

    def test_schema_uses_declared_inventory_not_a_run_specific_count(self):
        self.assertIs(driver.validate_spec(self.spec), self.spec)
        self.assertEqual(len(self.spec['expected_app']), 2)

    def test_bad_schema_and_ranges_fail_without_runtime_effects(self):
        changes = [{'schema': 'wrong'}, {'SM': [7, False]}, {'modules': [{}]},
                   {'UUIDs': ['GPU-' + '1' * 32] * 2}, {'source_commit': 'x' * 40},
                   {'diagnostics': {**self.spec['diagnostics'], 'sample_limit': 4097}}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                driver.validate_spec({**self.spec, **change})
        self.assertEqual(list(self.evidence.iterdir()), [])

    def test_spec_identity_and_duplicate_fields_fail_before_execution(self):
        path = self.root / 'spec.json'
        raw = b'{"schema":1,"schema":2}'
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            driver.load_spec(path, expected_sha=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw))
        with self.assertRaisesRegex(ValueError, 'identity'):
            driver.load_spec(path, expected_sha='0' * 64, expected_bytes=len(raw))
        with self.assertRaisesRegex(ValueError, 'identity'):
            driver.load_spec(path, expected_sha=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw) - 1)

    def test_spawn_main_import_is_inert_and_definitions_remain_available(self):
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        process = context.Process(target=_spawn_import_probe, args=(DRIVER_PATH, queue))
        process.start()
        try:
            process.join(20)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(queue.get(timeout=5), (True, True))
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            queue.close()
            queue.join_thread()
        self.assertEqual(list(self.evidence.iterdir()), [])

    def test_ordinary_import_and_help_do_not_import_runtime_or_read_spec(self):
        script = """import builtins,runpy,sys
from pathlib import Path
from unittest import mock
original=builtins.__import__
def guarded(name,*a,**k):
 if name.split('.')[0] in ('torch','pytest','tensorfold'): raise AssertionError(name)
 return original(name,*a,**k)
with mock.patch('builtins.__import__',side_effect=guarded), mock.patch.object(Path,'open',side_effect=AssertionError('IO')):
 n=runpy.run_path(sys.argv[1],run_name='arbitrary_import_name')
 assert callable(n['main']) and n['Guard']
 try:n['main'](['--help'])
 except SystemExit as e:assert e.code==0
"""
        result = subprocess.run([sys.executable, '-B', '-c', script, DRIVER_PATH], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--spec-sha256', result.stdout)
        self.assertEqual(list(self.evidence.iterdir()), [])

    def test_cli_bad_spec_does_not_create_evidence_or_import_runtime(self):
        path = self.root / 'spec.json'
        raw = b'{}'
        path.write_bytes(raw)
        script = """import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','pytest','tensorfold'):raise AssertionError('runtime import before schema')
 return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=sys.argv[1:]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except ValueError as error:
 print(str(error),file=sys.stderr)
 raise SystemExit(3)
"""
        result = subprocess.run([sys.executable, '-B', '-c', script, DRIVER_PATH, '--spec', str(path),
            '--spec-sha256', hashlib.sha256(raw).hexdigest(), '--spec-bytes', str(len(raw)),
            '--source', str(self.source), '--evidence', str(self.evidence)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn('spec schema', result.stderr)
        self.assertEqual(list(self.evidence.iterdir()), [])

    def test_cli_forwards_verified_spec_and_declared_paths(self):
        path = self.root / 'spec.json'
        raw = json.dumps(self.spec).encode()
        path.write_bytes(raw)
        with mock.patch.object(driver, 'run', return_value=17) as execute:
            code = driver.main(['--spec', str(path), '--spec-sha256', hashlib.sha256(raw).hexdigest(),
                '--spec-bytes', str(len(raw)), '--source', str(self.source), '--evidence', str(self.evidence)])
        self.assertEqual(code, 17)
        execute.assert_called_once_with(self.spec, self.source, self.evidence)
        self.assertEqual(list(self.evidence.iterdir()), [])

    def test_runtime_identity_mismatches_refuse_before_pytest(self):
        mutations = ('SDK', 'FD_C', 'Torch_git', 'CUDA', 'device_count', 'UUID_order', 'SM', 'app_origin')
        for index, name in enumerate(mutations):
            with self.subTest(name=name):
                self.evidence = self.root / ('identity-' + str(index))
                self.evidence.mkdir()
                # Each case starts from the actual fixture identities.
                self.sdk.write_text('{}\n')
                self.runtime.fd._source_sha256 = 'c' * 64
                self.runtime.torch.version.git_version, self.runtime.torch.version.cuda = 'd' * 40, '12.9'
                self.runtime.torch.cuda.device_count = lambda: 2
                self.runtime.torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(uuid=self.spec['UUIDs'][i])
                self.runtime.torch.cuda.get_device_capability = lambda i: (7, 0)
                self.runtime.app.__file__ = str(self.app / '__init__.py')
                if name == 'SDK':
                    self.sdk.write_text('changed')
                elif name == 'FD_C':
                    self.runtime.fd._source_sha256 = 'e' * 64
                elif name == 'Torch_git':
                    self.runtime.torch.version.git_version = 'e' * 40
                elif name == 'CUDA':
                    self.runtime.torch.version.cuda = '13.0'
                elif name == 'device_count':
                    self.runtime.torch.cuda.device_count = lambda: 1
                elif name == 'UUID_order':
                    self.runtime.torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(uuid=self.spec['UUIDs'][1-i])
                elif name == 'SM':
                    self.runtime.torch.cuda.get_device_capability = lambda i: (8, 0)
                else:
                    self.runtime.app.__file__ = str(self.source / '__init__.py')
                self.events.clear()
                with self.assertRaises(RuntimeError):
                    self.execute()
                self.assertNotIn('pytest', self.events)
                self.assertIsNotNone(self.outcome()['primary_error'])

    def test_success_checks_origins_collection_junit_source_and_preserves_path(self):
        old = list(sys.path)
        self.assertEqual(self.execute(), 0)
        self.assertEqual(sys.path, old)
        result = self.outcome()
        self.assertEqual(result['schema'], 'tensorfold.native-contract-result-v1')
        self.assertTrue(result['source_after_match'])
        self.assertEqual(result['post_errors'], [])
        self.assertEqual(result['JUnit']['tests'], '1')
        self.assertEqual(self.events, ['pytest', 'synchronize'])
        self.assertEqual([value for value in self.pytest_argv if value.startswith('--capture')], ['--capture=no'])

    def test_explicit_cpu_frontend_route_never_queries_cuda(self):
        class ForbiddenCuda:
            def __getattribute__(self, name):
                raise AssertionError('CPU route queried CUDA: ' + name)
        self.spec.update(mode='cpu', UUIDs=[], SM=[], required_native_elf_prefixes=[])
        self.runtime.torch.cuda = ForbiddenCuda()
        self.assertEqual(self.execute(), 0)
        self.assertEqual(self.events, ['pytest'])
        result = self.outcome()
        self.assertEqual(result['mode'], 'cpu')
        self.assertFalse(result['context_fatal'])
        origin = json.loads((self.evidence / 'installed-SDK-source-origin.json').read_text())
        self.assertEqual(origin['devices_visible'], [])
        self.assertEqual(origin['FD_C'], self.spec['fd']['source_sha256'])
        self.assertEqual(origin['Torch_git'], self.spec['torch']['git_version'])

    def test_cpu_does_not_silently_satisfy_a_required_native_route(self):
        self.runtime.torch.cuda.device_count = lambda: 0
        with self.assertRaisesRegex(RuntimeError, 'device count'):
            self.execute()
        self.assertNotIn('pytest', self.events)
        self.assertEqual(self.outcome()['mode'], 'cuda')
        with self.assertRaisesRegex(ValueError, 'CPU-only'):
            driver.validate_spec({**self.spec, 'mode': 'cpu'})

    def test_external_pytest_options_refuse_before_runtime_tests_or_evidence(self):
        runtime = mock.Mock(side_effect=AssertionError('runtime must not load'))
        helpers = mock.Mock(side_effect=AssertionError('publisher must not load'))
        with mock.patch.object(driver, 'check_tests', side_effect=AssertionError('tests must not be read')) as tests:
            for value in ('-k selected', '--maxfail=1', ' '):
                with self.subTest(value=value), mock.patch.dict('os.environ', {'PYTEST_ADDOPTS': value}):
                    with self.assertRaisesRegex(ValueError, 'PYTEST_ADDOPTS'):
                        driver.run(self.spec, self.source, self.evidence,
                                   runtime_loader=runtime, helpers_loader=helpers)
        runtime.assert_not_called()
        helpers.assert_not_called()
        tests.assert_not_called()
        self.assertEqual(list(self.evidence.iterdir()), [])
        self.assertEqual(self.events, [])

    def test_wrong_input_hash_refuses_before_loading_runtime(self):
        self.test.write_text('changed')
        loader = mock.Mock(side_effect=AssertionError('runtime must not load'))
        with self.assertRaisesRegex(RuntimeError, 'test input'):
            driver.run(self.spec, self.source, self.evidence, runtime_loader=loader,
                       helpers_loader=lambda: (self.publish, raise_failures))
        loader.assert_not_called()
        self.assertIsNotNone(self.outcome()['primary_error'])

    def test_wrong_collection_root_or_module_id_never_runs_test_body(self):
        for root, node in ((self.root, self.module_id), (self.source, 'wrong.py')):
            with self.subTest(root=root, node=node):
                evidence = self.root / ('ev-' + str(len(list(self.root.iterdir()))))
                evidence.mkdir()
                self.evidence = evidence
                self.root_override, self.module_id = root, node
                with self.assertRaises(RuntimeError):
                    self.execute()
                self.assertIsNotNone(self.outcome()['primary_error'])

    def test_context_fatal_stops_session_without_further_cuda_sampling_or_sync(self):
        self.code, self.failure = 1, 'CUDA error: illegal memory access'
        self.assertEqual(self.execute(), 1)
        self.assertIn('remaining affected work unrun', self.session.shouldstop)
        self.assertNotIn('synchronize', self.events)
        self.assertTrue(self.outcome()['context_fatal'])
        first = json.loads((self.evidence / 'first-failure.json').read_text())
        self.assertEqual(first['trace'], self.failure)

    def test_observed_failure_cannot_be_overruled_by_zero_exit_and_junit_counts(self):
        self.failure = 'authentic failed report, contradictory zero aggregate'
        self.assertEqual(self.execute(), 1)
        result = self.outcome()
        self.assertEqual(result['pytest_exit'], 0)
        self.assertEqual(result['JUnit']['failures'], '0')
        self.assertTrue(result['first_failure_observed'])

    def test_skips_are_not_success_and_sampling_errors_are_separate(self):
        self.skips = 1
        read_text = driver._read_text
        def fail_sample(path, limit):
            if str(path) == '/proc/self/maps':
                return read_text(path, limit)
            raise OSError('sample unavailable')
        with mock.patch.object(driver, '_read_text', side_effect=fail_sample):
            self.assertEqual(self.execute(), 1)
        result = self.outcome()
        self.assertEqual(result['pytest_exit'], 0)
        self.assertEqual(result['JUnit']['skipped'], '1')
        self.assertEqual(result['post_errors'], [])
        self.assertEqual(len(result['diagnostic_errors']), 1)

    def test_primary_and_source_after_error_are_both_preserved(self):
        primary = RuntimeError('pytest owning exception')
        self.primary = primary
        # Admission succeeds, then the selected test mutates a declared input
        # before raising its authentic primary.
        def failing(argv, *, plugins):
            self.test.write_text('mutation')
            raise primary
        self.runtime.pytest.main = failing
        with self.assertRaises(RuntimeError) as caught:
            self.execute()
        self.assertIs(caught.exception, primary)
        result = self.outcome()
        self.assertIn('pytest owning exception', result['primary_error']['error'])
        self.assertTrue(any('test input differs' in row['error'] for row in result['post_errors']))
        self.assertFalse(result['source_after_match'])

    def test_test_failure_survives_sync_and_receipt_publication_errors(self):
        self.code, self.failure = 1, 'independent test failure'
        sync = OSError('synchronize failed')
        self.sync_error = sync
        def writer(path, value):
            if Path(path).name == 'named-suite-result.json':
                raise PermissionError('result write failed')
            self.publish(path, value)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(OSError) as caught:
            self.execute(writer)
        self.assertIs(caught.exception, sync)
        self.assertIn('pytest_exit= 1', stderr.getvalue())
        self.assertIn('PermissionError', stderr.getvalue())
        first = json.loads((self.evidence / 'first-failure.json').read_text())
        self.assertEqual(first['trace'], self.failure)

    def test_primary_is_preserved_when_result_diagnostic_formatting_also_fails(self):
        class Unrenderable(RuntimeError):
            def __repr__(self):
                raise MemoryError('cannot format error')
        primary = Unrenderable('authentic owning exception')
        self.primary = primary
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(Unrenderable) as caught:
            self.execute()
        self.assertIs(caught.exception, primary)
        self.assertIsNotNone(primary.__cause__)
        self.assertFalse((self.evidence / 'named-suite-result.json').exists())

    def native_library(self, name='tf_fixture.so'):
        path = self.root / name
        path.write_bytes(b'\x7fELFfixture mapped bytes')
        return path

    def mapping(self, path, *, inode=None, permissions='r-xp', device='00:23', extent='1000-2000'):
        info = path.stat()
        encoded = str(path).replace(' ', r'\040')
        return (extent + ' ' + permissions + ' 00000000 ' + device + ' '
                + str(info.st_ino if inode is None else inode) + ' ' + encoded + '\n')

    @contextlib.contextmanager
    def mapped_model(self, library, *, text=None, backing_paths=None, path_paths=None, denied=False):
        # Independent OS-boundary model: real ordinary FDs/read/hash/close and
        # real namespace/mount metadata; only privileged VMA lookup is modeled.
        real_open = os.open
        state = {'vma_fds': [], 'path_fds': [], 'map_reads': 0, 'vma_calls': 0, 'path_calls': 0}
        def open_vma(path, flags, *args, **kwargs):
            path = os.fspath(path)
            if path.startswith('/proc/self/map_files/'):
                if denied:
                    raise PermissionError(1, 'actual model permission refusal', path)
                choices = backing_paths or [library]
                target = choices[min(state['vma_calls'], len(choices) - 1)]
                state['vma_calls'] += 1
                descriptor = real_open(target, flags, *args, **kwargs)
                state['vma_fds'].append(descriptor)
                return descriptor
            if path == str(library):
                choices = path_paths or [library]
                target = choices[min(state['path_calls'], len(choices) - 1)]
                state['path_calls'] += 1
                descriptor = real_open(target, flags, *args, **kwargs)
                state['path_fds'].append(descriptor)
                return descriptor
            return real_open(path, flags, *args, **kwargs)
        def maps():
            state['map_reads'] += 1
            if state['map_reads'] > 1:
                for descriptor in state['vma_fds'] + state['path_fds']:
                    os.fstat(descriptor)  # Held through second global snapshot.
            return self.mapping(library) if text is None else text
        access = {'uids': [1000] * 4, 'gids': [1000] * 4,
                  'caps': {name: '0000010000000000' for name in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')},
                  'NoNewPrivs': '0'}
        with mock.patch.object(driver, '_read_native_maps', side_effect=maps), \
             mock.patch.object(driver, '_mapfile_access_precondition', return_value=access), \
             mock.patch.object(driver.os, 'open', side_effect=open_vma):
            yield state

    def test_mapped_unregistered_elf_uses_held_vma_fd_instead_of_raw_device(self):
        library = self.native_library()
        self.spec['required_native_elf_prefixes'] = ['tf_fixture']
        with self.mapped_model(library) as state:
            self.assertEqual(self.execute(), 0)
        report = json.loads((self.evidence / 'actual-native-ELF.json').read_text())
        row = report['mapped_elfs'][0]
        self.assertEqual(report['identity_method'], 'kernel-vma-backed-fd-v1')
        self.assertEqual(row['sha256'], hashlib.sha256(library.read_bytes()).hexdigest())
        self.assertEqual(row['inode'], library.stat().st_ino)
        self.assertEqual(row['raw_map_device'], [0, 35])
        self.assertGreaterEqual(state['vma_calls'], 2)
        self.assertEqual(report['missing_required_parent_prefixes'], [])
        for descriptor in state['vma_fds'] + state['path_fds']:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_matching_raw_device_still_requires_privileged_vma_fd(self):
        library = self.native_library()
        info = library.stat()
        text = self.mapping(library, device=format(os.major(info.st_dev), 'x') + ':' + format(os.minor(info.st_dev), 'x'))
        with self.mapped_model(library, text=text, denied=True) as state:
            with self.assertRaises(PermissionError):
                driver.observe_native_elfs(['tf_fixture'])
        self.assertEqual(state['path_calls'], 0)

    def test_python_registry_alone_cannot_satisfy_required_mapping(self):
        library = self.native_library()
        module = types.ModuleType('tf_fixture')
        module.__file__ = str(library)
        self.spec['required_native_elf_prefixes'] = ['tf_fixture']
        with mock.patch.dict(sys.modules, {'tf_fixture': module}), self.assertRaisesRegex(RuntimeError, 'actually mapped'):
            self.execute()
        report = json.loads((self.evidence / 'actual-native-ELF.json').read_text())
        self.assertEqual(report['python_registry']['tf_fixture'], str(library))
        self.assertEqual(report['missing_required_parent_prefixes'], ['tf_fixture'])

    def test_segments_deduplicate_but_every_distinct_vma_is_bound(self):
        library = self.native_library('tf_space name.so')
        text = self.mapping(library) * 2 + self.mapping(library, extent='3000-4000')
        with self.mapped_model(library, text=text) as state:
            report = driver.observe_native_elfs(['tf_space'])
        self.assertEqual(len(report['mapped_elfs']), 1)
        self.assertEqual(len(report['mapped_elfs'][0]['vmas']), 2)
        self.assertEqual(state['vma_calls'], 4)

    def test_cross_subvolume_same_raw_inode_and_equal_content_is_not_object_identity(self):
        library, foreign = self.native_library(), self.native_library('foreign.so')
        descriptor = os.open(library, os.O_RDONLY)
        try:
            template = driver._file_stat(descriptor)
        finally:
            os.close(descriptor)
        original_stat = driver._file_stat
        foreign_inode = foreign.stat().st_ino
        def domains(descriptor):
            current = original_stat(descriptor)
            return {**template, 'inode': 123, 'device': 38 if current['inode'] == foreign_inode else 37,
                    'device_major': 0, 'device_minor': 38 if current['inode'] == foreign_inode else 37}
        with self.mapped_model(library, text=self.mapping(library, inode=123), path_paths=[foreign]):
            with mock.patch.object(driver, '_file_stat', side_effect=domains), \
                 mock.patch.object(driver, '_hash_held_elf', side_effect=AssertionError('must not hash mismatched objects')):
                with self.assertRaisesRegex(RuntimeError, 'pathname versus VMA FD'):
                    driver.observe_native_elfs(['tf_fixture'])

    def test_mapping_aba_same_text_and_address_rejects_replaced_backing_object(self):
        library, replacement = self.native_library(), self.native_library('replacement.so')
        with self.mapped_model(library, backing_paths=[library, replacement]):
            with self.assertRaisesRegex(RuntimeError, 'reopened VMA FD'):
                driver.observe_native_elfs(['tf_fixture'])

    def test_path_rename_to_equal_content_other_object_is_rejected(self):
        library, replacement = self.native_library(), self.native_library('replacement.so')
        with self.mapped_model(library, path_paths=[library, replacement]):
            with self.assertRaisesRegex(RuntimeError, 'reopened pathname FD'):
                driver.observe_native_elfs(['tf_fixture'])

    def test_file_mutation_and_mount_snapshot_or_alias_changes_are_rejected(self):
        library = self.native_library()
        hashing = driver._hash_held_elf
        def mutate(descriptor):
            result = hashing(descriptor)
            library.write_bytes(b'\x7fELFchanged')
            return result
        with self.mapped_model(library), mock.patch.object(driver, '_hash_held_elf', side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, 'retained pathname metadata'):
                driver.observe_native_elfs(['tf_fixture'])
        table = driver._mount_table()
        changed = {key: {**value, 'root': value['root'] + '-changed'} for key, value in table.items()}
        with self.mapped_model(library), mock.patch.object(driver, '_mount_table', side_effect=[table, changed]):
            with self.assertRaisesRegex(RuntimeError, 'mount namespace changed'):
                driver.observe_native_elfs(['tf_fixture'])
        mount = driver._fd_mount
        calls = []
        def alias(descriptor, rows):
            result = mount(descriptor, rows)
            calls.append(descriptor)
            return result if len(calls) == 1 else {**result, 'mount_id': result['mount_id'] + 1}
        with self.mapped_model(library), mock.patch.object(driver, '_fd_mount', side_effect=alias):
            with self.assertRaisesRegex(RuntimeError, 'pathname versus VMA mount'):
                driver.observe_native_elfs(['tf_fixture'])

    def test_deleted_symlink_nonelf_and_changed_map_fail_without_fallback(self):
        library = self.native_library()
        with mock.patch.object(driver, '_read_native_maps', return_value=self.mapping(library).rstrip() + ' (deleted)\n'):
            with self.assertRaisesRegex(ValueError, 'stable ordinary'):
                driver.observe_native_elfs([])
        library.write_bytes(b'not ELF')
        with self.mapped_model(library):
            with self.assertRaisesRegex(ValueError, 'not ELF'):
                driver.observe_native_elfs([])
        library.write_bytes(b'\x7fELFvalid')
        with self.mapped_model(library), mock.patch.object(driver, '_read_native_maps', side_effect=[self.mapping(library), '']):
            with self.assertRaisesRegex(RuntimeError, 'mappings changed'):
                driver.observe_native_elfs([])
        link = self.root / 'tf_link.so'
        link.symlink_to(library)
        with self.mapped_model(link, backing_paths=[library]):
            with self.assertRaises(OSError):
                driver.observe_native_elfs([])

    def test_raw_reads_and_cleanup_preserve_authentic_primary_prior_causes(self):
        library = self.native_library()
        primary, cause, cleanup = OSError('read failure'), ValueError('prior cause'), OSError('close failure')
        primary.__cause__ = cause
        primary.add_note('authentic note')
        close = os.close
        def close_fault(descriptor):
            close(descriptor)
            raise cleanup
        with self.mapped_model(library), mock.patch.object(driver.os, 'read', side_effect=primary), \
             mock.patch.object(driver.os, 'close', side_effect=close_fault):
            with self.assertRaises(OSError) as caught:
                driver.observe_native_elfs([])
        self.assertIs(caught.exception, primary)
        self.assertIn(cause, primary.__cause__.exceptions)
        self.assertIn(cleanup, primary.__cause__.exceptions)
        self.assertEqual(primary.__notes__[0], 'authentic note')

    def test_partial_reads_and_required_declared_provider_selection(self):
        library = self.native_library('provider.so')
        read = os.read
        with self.mapped_model(library), mock.patch.object(driver.os, 'read', side_effect=lambda fd, n: read(fd, min(n, 2))):
            report = driver.observe_native_elfs([], provider_paths=[str(library)])
        self.assertEqual(report['mapped_elfs'][0]['sha256'], hashlib.sha256(library.read_bytes()).hexdigest())
        self.assertEqual(report['missing_required_provider_paths'], [])
        with mock.patch.object(driver, '_read_native_maps', return_value=''):
            self.assertEqual(driver.observe_native_elfs([], provider_paths=[str(library)])['missing_required_provider_paths'], [str(library)])

    def test_namespace_identity_and_descriptor_exhaustion_remain_fail_closed(self):
        library = self.native_library()
        actual_stat = os.stat
        def switched(path, *args, **kwargs):
            result = actual_stat(path, *args, **kwargs)
            if str(path) == '/proc/thread-self/ns/mnt':
                return types.SimpleNamespace(st_dev=result.st_dev, st_ino=result.st_ino + 1)
            return result
        with self.mapped_model(library), mock.patch.object(driver.os, 'stat', side_effect=switched):
            with self.assertRaisesRegex(RuntimeError, 'mount namespace'):
                driver.observe_native_elfs([])
        descriptor = None
        with self.assertRaisesRegex(RuntimeError, 'capacity was not prepaid'):
            with driver._mapping_fds(1) as journal:
                descriptor = journal.open(library, os.O_RDONLY)
                journal.open(library, os.O_RDONLY)
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_identity_note_failure_keeps_mismatch_primary_and_diagnostic(self):
        library, foreign = self.native_library(), self.native_library('foreign.so')
        diagnostic = MemoryError('identity-note allocation')
        with self.mapped_model(library, path_paths=[foreign]), mock.patch.object(driver.json, 'dumps', side_effect=diagnostic):
            with self.assertRaisesRegex(RuntimeError, 'pathname versus VMA FD') as caught:
                driver.observe_native_elfs([])
        self.assertIs(caught.exception.__cause__, diagnostic)

    def test_permission_precondition_refuses_before_any_fd(self):
        library = self.native_library()
        with mock.patch.object(driver, '_read_native_maps', return_value=self.mapping(library)), \
             mock.patch.object(driver, '_mapfile_access_precondition', side_effect=RuntimeError('ONLY E/P/I/A/B invalid')), \
             mock.patch.object(driver, '_MappingFDs', side_effect=AssertionError('no FD after invalid caps')):
            with self.assertRaisesRegex(RuntimeError, 'ONLY E/P/I/A/B'):
                driver.observe_native_elfs([])

    def test_child_record_is_durable_scoped_exclusive_and_source_identified(self):
        library = self.native_library()
        with self.mapped_model(library):
            report = driver.record_native_elfs(['tf_fixture'], rank=7, tag='owner-head4-1', evidence_dir=self.evidence)
        with self.mapped_model(library):
            with self.assertRaises(FileExistsError):
                driver.record_native_elfs(['tf_fixture'], rank=7, tag='owner-head4-1', evidence_dir=self.evidence)
        path = self.evidence / ('owner-head4-1-rank-7-pid-' + str(os.getpid()) + '.json')
        self.assertEqual(json.loads(path.read_text())['mapped_elfs'], json.loads(json.dumps(report['mapped_elfs'])))
        self.assertEqual(report['observer']['sha256'], driver.file_identity(driver.__file__)['sha256'])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_missing_child_mapping_is_published_before_owning_failure(self):
        with mock.patch.object(driver, '_read_native_maps', return_value=''):
            with self.assertRaisesRegex(RuntimeError, 'required child'):
                driver.record_native_elfs(['tf_fixture'], rank=0, tag='missing', evidence_dir=self.evidence)
        self.assertEqual(len(list(self.evidence.glob('missing-*.json'))), 1)

    def test_child_write_failure_and_environment_restore_failure_remain_owning(self):
        library = self.native_library()
        primary = OSError('durability failed')
        with self.mapped_model(library), mock.patch.object(driver.os, 'fsync', side_effect=primary):
            with self.assertRaises(OSError) as caught:
                driver.record_native_elfs(['tf_fixture'], rank=0, tag='fault', evidence_dir=self.evidence)
        self.assertIs(caught.exception, primary)
        # Use a fresh evidence root for driver-owned files after child record.
        for path in self.evidence.iterdir():
            path.unlink()
        primary, cleanup = OSError('authentic pytest primary'), RuntimeError('unsetenv failure')
        self.primary = primary
        try:
            with mock.patch.object(driver.os.environ, 'pop', side_effect=cleanup):
                with self.assertRaises(OSError) as caught:
                    self.execute()
            self.assertIs(caught.exception, primary)
            self.assertTrue(any('unsetenv failure' in error['error'] for error in self.outcome()['post_errors']))
        finally:
            os.environ.pop(driver.CHILD_EVIDENCE_ENV, None)

    def test_external_child_evidence_cannot_redirect_worker_records(self):
        with mock.patch.dict(os.environ, {driver.CHILD_EVIDENCE_ENV: '/unowned'}):
            with self.assertRaisesRegex(ValueError, 'external child-native'):
                self.execute()
        self.assertEqual(self.events, [])
        self.assertEqual(list(self.evidence.iterdir()), [])

    def qualified_spec(self):
        core = self.native_library('torch_C.so')
        self.runtime.torch._C = types.SimpleNamespace(__file__=str(core))
        providers = {str(path): {key: driver.file_identity(path)[key] for key in ('bytes', 'sha256')}
                     for path in (self.fd, core)}
        contract = {'mode': 'map-files-cap40-v1', 'base_image_id': 'sha256:' + 'e' * 64,
            'launcher': {'path': '/opt/qualification/bin/native-mapfile-cap-launcher', 'bytes': 123,
                'sha256': 'f' * 64, 'source_bytes': 45, 'source_sha256': '0' * 64,
                'file_capability_hex': '0100000200000000000000000001000000000000'},
            'python': {'path': '/opt/venv/bin/python3', 'bytes': 321, 'sha256': '1' * 64}, 'providers': providers}
        self.spec['verification_launcher'] = contract
        access = {'uids': [1000] * 4, 'gids': [1000] * 4, 'NoNewPrivs': '0',
                  'caps': {name: '0000010000000000' for name in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')}}
        proof = {'schema': 'tensorfold.native-elf-observation-v3', 'identity_method': 'kernel-vma-backed-fd-v1',
                 'mapped_elfs': [{'path': path, **record} for path, record in providers.items()],
                 'missing_required_provider_paths': [], 'missing_required_prefixes': []}
        return access, ({'path': contract['launcher']['path']}, {'path': contract['python']['path']}), proof

    def test_optional_qualification_schema_is_closed_and_default_spec_unchanged(self):
        self.assertNotIn('verification_launcher', self.spec)
        self.assertIs(driver.validate_spec(self.spec), self.spec)
        self.qualified_spec()
        self.assertIs(driver.validate_spec(self.spec), self.spec)
        bad = {**self.spec['verification_launcher'], 'unreviewed': True}
        with self.assertRaises(ValueError):
            driver.validate_spec({**self.spec, 'verification_launcher': bad})
        bad = {**self.spec['verification_launcher'], 'launcher': {**self.spec['verification_launcher']['launcher'], 'file_capability_hex': '00' * 20}}
        with self.assertRaises(ValueError):
            driver.validate_spec({**self.spec, 'verification_launcher': bad})

    def test_qualification_receipt_precedes_cuda_queries_and_test_body(self):
        access, metadata, proof = self.qualified_spec()
        events = []
        def initial(spec):
            events.append('cap-python-launcher')
            return access, *metadata
        def load():
            events.append('runtime-import')
            return self.runtime
        def mapped(*args, **kwargs):
            events.append('mapped-provider-proof')
            return {**proof}
        def count():
            self.assertTrue((self.evidence / 'verification-launcher-preflight.json').exists())
            events.append('cuda-query')
            return 2
        self.runtime.torch.cuda.device_count = count
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(driver.os, 'getpid', return_value=1), \
             mock.patch.object(driver, '_verification_process', side_effect=initial), \
             mock.patch.object(driver, '_mapfile_access_precondition', return_value=access), \
             mock.patch.object(driver, 'observe_native_elfs', side_effect=mapped):
            # Parent final origin observation is empty in this model, but its
            # mapping call must retain the normal required-parent guard.
            with mock.patch.object(driver, '_native_elfs', return_value={'missing_required_parent_prefixes': []}):
                self.assertEqual(driver.run(self.spec, self.source, self.evidence, runtime_loader=load,
                    helpers_loader=lambda: (self.publish, raise_failures)), 0)
        self.assertEqual(events[:4], ['cap-python-launcher', 'runtime-import', 'mapped-provider-proof', 'cuda-query'])
        receipt = json.loads((self.evidence / 'verification-launcher-preflight.json').read_text())
        self.assertEqual(receipt['verification_launcher'], self.spec['verification_launcher'])
        self.assertEqual(receipt['providers']['required_providers'], self.spec['verification_launcher']['providers'])
        self.assertEqual(receipt['loader_environment'], {'LD_LIBRARY_PATH': None})
        self.assertEqual(receipt['PID'], 1)

    def test_qualification_access_failure_happens_before_runtime_import(self):
        self.qualified_spec()
        load = mock.Mock(side_effect=AssertionError('must not import runtime'))
        with mock.patch.object(driver, '_mapfile_access_precondition', side_effect=RuntimeError('ONLY E/P/I/A/B invalid')):
            with self.assertRaisesRegex(RuntimeError, 'ONLY E/P/I/A/B'):
                driver.run(self.spec, self.source, self.evidence, runtime_loader=load,
                           helpers_loader=lambda: (self.publish, raise_failures))
        load.assert_not_called()
        self.assertEqual(self.events, [])

    def test_missing_or_wrong_provider_blocks_device_query_and_pytest(self):
        access, metadata, proof = self.qualified_spec()
        self.runtime.torch.cuda.device_count = mock.Mock(side_effect=AssertionError('no device query'))
        bad = {**proof, 'missing_required_provider_paths': [str(self.fd)]}
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(driver, '_verification_process', return_value=(access, *metadata)), \
             mock.patch.object(driver, '_mapfile_access_precondition', return_value=access), \
             mock.patch.object(driver, 'observe_native_elfs', return_value=bad):
            with self.assertRaisesRegex(RuntimeError, 'not actually mapped'):
                self.execute()
        self.runtime.torch.cuda.device_count.assert_not_called()
        self.assertNotIn('pytest', self.events)
        self.assertFalse((self.evidence / 'verification-launcher-preflight.json').exists())

    def test_wrong_provider_digest_and_restored_loader_environment_block_tests(self):
        access, metadata, proof = self.qualified_spec()
        wrong = {**proof, 'mapped_elfs': [{**row, 'sha256': '0' * 64} for row in proof['mapped_elfs']]}
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(driver, '_verification_process', return_value=(access, *metadata)), \
             mock.patch.object(driver, '_mapfile_access_precondition', return_value=access), \
             mock.patch.object(driver, 'observe_native_elfs', return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, 'provider identity differs'):
                self.execute()
        self.assertNotIn('pytest', self.events)
        # Fresh output directory for the independent environment-retirement case.
        self.evidence = self.root / 'second-evidence'
        self.evidence.mkdir()
        def restored(*args, **kwargs):
            os.environ['LD_LIBRARY_PATH'] = '/must-not-restore'
            return proof
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(driver, '_verification_process', return_value=(access, *metadata)), \
             mock.patch.object(driver, '_mapfile_access_precondition', return_value=access), \
             mock.patch.object(driver, 'observe_native_elfs', side_effect=restored):
            with self.assertRaisesRegex(RuntimeError, 'access/environment changed'):
                self.execute()
        self.assertNotIn('pytest', self.events)

    def test_prequalification_write_failure_prevents_test_and_preserves_primary(self):
        access, metadata, proof = self.qualified_spec()
        primary = OSError('prequalification receipt failure')
        def publish(path, value):
            if Path(path).name == 'verification-launcher-preflight.json':
                raise primary
            self.publish(path, value)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(driver, '_verification_process', return_value=(access, *metadata)), \
             mock.patch.object(driver, '_mapfile_access_precondition', return_value=access), \
             mock.patch.object(driver, 'observe_native_elfs', return_value=proof):
            with self.assertRaises(OSError) as caught:
                self.execute(publish)
        self.assertIs(caught.exception, primary)
        self.assertNotIn('pytest', self.events)
        self.assertEqual(self.outcome()['primary_error']['type'], 'OSError')

    def test_access_status_rejects_any_extra_capability_and_uid_change(self):
        status = ('Uid:\t1000 1000 1000 1000\nGid:\t1000 1000 1000 1000\nNoNewPrivs:\t0\n'
                  + ''.join(name + ':\t0000010000000000\n' for name in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')))
        with mock.patch.object(driver, '_read_text', return_value=status):
            self.assertEqual(driver._mapfile_access_precondition()['uids'], [1000] * 4)
        for bad in (status.replace('1000 1000 1000 1000', '0 0 0 0', 1),
                    status.replace('0000010000000000', '0000010000200000', 1), status.replace('NoNewPrivs:\t0', 'NoNewPrivs:\t1')):
            with self.subTest(status=bad), mock.patch.object(driver, '_read_text', return_value=bad):
                with self.assertRaisesRegex(RuntimeError, 'ONLY E/P/I/A/B'):
                    driver._mapfile_access_precondition()

    def test_app_mutation_cannot_be_hidden_by_pytest_zero_exit(self):
        self.mutate = lambda: (self.app / '__init__.py').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'post-suite source identity'):
            self.execute()
        self.assertEqual(self.outcome()['pytest_exit'], 0)
        self.assertFalse(self.outcome()['source_after_match'])


if __name__ == '__main__':
    unittest.main()
