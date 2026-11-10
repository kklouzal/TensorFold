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
        self.subreports = []
        self.extra_xml_tests = 0
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
        item = types.SimpleNamespace(getparent=lambda kind: module, nodeid=self.module_id + '::test_contract')
        guard.pytest_collection_modifyitems(self.session, types.SimpleNamespace(rootpath=self.root_override or self.source), [item])
        if self.primary is not None:
            raise self.primary
        for report in self.subreports:
            guard.pytest_runtest_logreport(report)
        if self.failure is not None:
            guard.pytest_runtest_logreport(types.SimpleNamespace(failed=True, nodeid=self.module_id + '::test_contract',
                                                                when='call', longrepr=self.failure))
        if self.mutate is not None:
            self.mutate()
        suite = ET.Element('testsuite', tests=str(1 + len(self.subreports) + self.extra_xml_tests),
                           errors='0', failures=str(int(self.code != 0)), skipped=str(self.skips))
        ET.SubElement(suite, 'testcase', name='test_contract')
        ET.ElementTree(suite).write(self.evidence / 'named-suite.xml')
        return self.code

    def test_typed_subtests_reconcile_reports_collection_and_junit_without_extra_items(self):
        class SubtestReport(types.SimpleNamespace):
            pass
        self.subreports = [SubtestReport(failed=False, nodeid=self.module_id + '::test_contract', when='call')
                           for _ in range(2)]
        with mock.patch.dict(sys.modules, {'_pytest.subtests': types.SimpleNamespace(SubtestReport=SubtestReport)}):
            self.assertEqual(self.execute(), 0)
        result = self.outcome()
        self.assertEqual(result['collected_items'], 1)
        self.assertEqual(result['reported_subtests'], 2)
        self.assertEqual(result['JUnit']['tests'], '3')

    def test_unobserved_extra_xml_count_is_still_rejected(self):
        self.extra_xml_tests = 1
        with self.assertRaisesRegex(RuntimeError, 'JUnit test count'):
            self.execute()

    def test_failed_subtest_cannot_be_overruled_by_zero_exit_or_junit(self):
        class SubtestReport(types.SimpleNamespace):
            pass
        self.subreports = [SubtestReport(failed=True, nodeid=self.module_id + '::test_contract',
                                        when='call', longrepr='owning subtest failure')]
        with mock.patch.dict(sys.modules, {'_pytest.subtests': types.SimpleNamespace(SubtestReport=SubtestReport)}):
            self.assertEqual(self.execute(), 1)
        self.assertTrue(self.outcome()['first_failure_observed'])

    def test_subtest_must_belong_to_a_collected_call(self):
        class SubtestReport(types.SimpleNamespace):
            pass
        for nodeid, when in [('uncollected::test', 'call'), (self.module_id + '::test_contract', 'setup')]:
            with self.subTest(nodeid=nodeid, when=when):
                guard = driver.Guard(self.spec, self.source, types.SimpleNamespace(), self.runtime.torch,
                                     self.runtime.pytest)
                guard.collected_nodeids = {self.module_id + '::test_contract'}
                with mock.patch.dict(sys.modules, {'_pytest.subtests': types.SimpleNamespace(SubtestReport=SubtestReport)}):
                    with self.assertRaisesRegex(RuntimeError, 'owning collected call'):
                        guard.pytest_runtest_logreport(SubtestReport(nodeid=nodeid, when=when, failed=False))
                self.assertEqual(guard.reported_subtests, 0)

    def test_junit_actual_case_membership_count_must_match_collection(self):
        path = self.evidence / 'counts.xml'
        for count in (0, 2):
            suite = ET.Element('testsuite', tests='2', errors='0', failures='0', skipped='0')
            for _ in range(count):
                ET.SubElement(suite, 'testcase')
            ET.ElementTree(suite).write(path)
            with self.assertRaisesRegex(RuntimeError, 'JUnit test count'):
                driver._junit(path, 1, 1)

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


    def model_operation(self):
        c = driver.model_contract
        root = '/models/hf/hub/models--fixture/snapshots/' + 'a'*40
        def item(path):
            return {'path': path, 'bytes': 1, 'sha256': 'a'*64}
        frontend = '/frontend/' + 'b'*64
        model = [item(root + '/fixture-0000' + str(i) + '-of-00002.gguf') for i in (1, 2)]
        return {'teacher_plan': item('/operator/teacher-plan.json'), 'teacher_client': item('/operator/teacher-client.json'),
                'model_entrypoint': model[0]['path'], 'model_assets': model, 'frontend_root': frontend,
                'frontend_assets': [item(frontend + '/' + name) for name in sorted(c.FRONTEND_NAMES)],
                'execution_manifest': None}

    def teacher_client(self):
        c = driver.model_contract
        return {'schema': 'tensorfold.owned-teacher-client-v1', 'teacher_plan_sha256': 'a'*64,
                'base': 'http://127.0.0.1:8000', 'model': 'diagnostic-fixture', 'expected_service': dict(c.SERVICE),
                'readiness_timeout_s': 3600, 'whole_timeout_s': 3900, 'socket_timeout_s': 300, 'response_bytes': 1048576}

    def model_spec(self):
        self.qualified_spec()
        self.spec.update(schema=driver.model_contract.SCHEMA, mode=driver.model_contract.MODE,
                         UUIDs=['GPU-' + format(i, '032x') for i in range(8)],
                         required_native_elf_prefixes=['tf_gguf_'], operation=self.model_operation())
        del self.spec['modules'], self.spec['test_files']
        source_files = {}
        for name in driver.model_contract.REQUIRED_TOOLS:
            data = (Path(driver.__file__).parent / Path(name).name).read_bytes()
            path = self.source / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            source_files[name] = {key: driver.file_identity(path)[key] for key in ('bytes', 'sha256')}
        self.spec['source_files'] = source_files
        self.runtime.torch.cuda = types.SimpleNamespace(_initialized=False)
        return self.spec

    def test_model_schema_is_distinct_closed_capability_and_eight_rank_only(self):
        spec = self.model_spec()
        self.assertIs(driver.validate_spec(spec), spec)
        self.assertNotIn('modules', spec)
        self.assertNotIn('test_files', spec)
        for patch in ({'modules': ['tests/example.py']}, {'mode': 'cuda'}, {'UUIDs': spec['UUIDs'][:4]},
                      {'required_native_elf_prefixes': []}, {'verification_launcher': None}):
            with self.subTest(patch=patch), self.assertRaises((ValueError, TypeError)):
                driver.validate_spec({**spec, **patch})

    def test_teacher_client_and_fixed_argv_reject_profile_timeout_or_command_escape(self):
        c = driver.model_contract
        client = self.teacher_client()
        argv = c.qualifier_argv(self.model_operation(), client,
                               ['GPU-00000000-0000-0000-0000-' + format(i, '012x') for i in range(8)], '/evidence/teacher')
        self.assertEqual(argv.count('--teacher-client'), 1)
        for flag, value in (('--context', '262144'), ('--parallel', '24'), ('--draft-method', 'none'),
                            ('--expert-tensor-parallel-size', '4'), ('--kv-full-windows', '24')):
            self.assertEqual(argv[argv.index(flag)+1], value)
        self.assertIn('--no-graphs', argv)
        self.assertNotIn('--reference-tiles', argv)
        for change in ({'base': 'http://example.com:8000'}, {'base': 'http://127.0.0.1:8000/?x=1'},
                       {'base': 'http://user@127.0.0.1:8000'}, {'socket_timeout_s': True},
                       {'whole_timeout_s': float('inf')}, {'readiness_timeout_s': 3601},
                       {'response_bytes': 2**20+1}, {'argv': ['python', '-c', 'unsafe']},
                       {'expected_service': {**client['expected_service'], 'resident_capacity': 23}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_teacher_client({**client, **change})

    def test_model_operation_rejects_extra_assets_paths_frontend_and_manifest(self):
        c = driver.model_contract
        operation = self.model_operation()
        self.assertIs(c.validate_operation(operation), operation)
        for change in ({'execution_manifest': '/arbitrary.json'}, {'model_assets': operation['model_assets'][:1]},
                       {'frontend_assets': operation['frontend_assets'][:-1]}, {'frontend_root': '/cache/frontend'},
                       {'argv': []}, {'model_entrypoint': '/models/../arbitrary.gguf'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_operation({**operation, **change})

    def test_held_assets_hash_once_retain_fds_and_refuse_post_use_change(self):
        asset = self.root / 'payload.gguf'
        asset.write_bytes(b'actual independent file bytes')
        record = {'path': str(asset), **{k: driver.file_identity(asset)[k] for k in ('bytes', 'sha256')}}
        content = {'assets': []}
        import time
        with driver._held_model_assets([record], time.monotonic()+10, content) as held:
            fd = held[0]['descriptor']
            self.assertEqual(os.fstat(fd).st_size, record['bytes'])
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertEqual(len(content['assets']), 1)
        with self.assertRaisesRegex(RuntimeError, 'input changed'):
            with driver._held_model_assets([record], time.monotonic()+10, {'assets': []}):
                asset.write_bytes(b'changed input')

    def test_content_binding_failure_and_primary_plus_postcheck_failure_propagate(self):
        import time
        asset = self.root / 'payload.gguf'
        asset.write_bytes(b'payload')
        record = {'path': str(asset), 'bytes': 7, 'sha256': '0'*64}
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            with driver._held_model_assets([record], time.monotonic()+10, {'assets': []}):
                self.fail('bad content reached operation')
        record['sha256'] = hashlib.sha256(b'payload').hexdigest()
        primary = ValueError('authentic model failure')
        try:
            with driver._held_model_assets([record], time.monotonic()+10, {'assets': []}):
                asset.write_bytes(b'changed')
                raise primary
        except BaseException as caught:
            self.assertIs(caught, primary)
            self.assertIsNotNone(caught.__cause__)
        else:
            self.fail('combined failures suppressed')
        with self.assertRaises(TimeoutError):
            with driver._held_model_assets([record], 0, {'assets': []}):
                self.fail('expired operation entered')

    def test_model_asset_fd_is_journalled_before_metadata_and_close_failure_preserves_primary(self):
        import time
        asset = self.root/'owned-payload'
        asset.write_bytes(b'payload')
        record = {'path': str(asset), 'bytes': 7, 'sha256': hashlib.sha256(b'payload').hexdigest()}
        actual_close, closed = os.close, []
        primary = MemoryError('metadata allocation failed after owned open')
        def close(descriptor):
            closed.append(descriptor)
            actual_close(descriptor)
            raise OSError('close completion report failed')
        with mock.patch.object(driver.os, 'fstat', side_effect=primary), mock.patch.object(driver.os, 'close', side_effect=close):
            try:
                with driver._held_model_assets([record], time.monotonic()+10, {'assets': []}):
                    self.fail('metadata failure reached model')
            except BaseException as caught:
                self.assertIs(caught, primary)
                self.assertIsNotNone(caught.__cause__)
            else:
                self.fail('primary and cleanup failure suppressed')
        self.assertEqual(len(closed), 1)
        with self.assertRaises(OSError):
            os.fstat(closed[0])

    def model_run_controls(self):
        self.model_spec()
        @contextlib.contextmanager
        def fake_held(*args):
            yield []
        calls = []
        plan = {'mode': 'capture', 'context': 262144, 'max_frames': 64}
        client = self.teacher_client()
        def data(path, *args, **kwargs):
            return client if str(path).endswith('teacher-client.json') else plan
        patches = contextlib.ExitStack()
        patches.enter_context(mock.patch.object(driver, '_verification_process', return_value='prequal'))
        patches.enter_context(mock.patch.object(driver, '_verification_receipt', return_value={'verified': True}))
        patches.enter_context(mock.patch.object(driver, '_held_model_assets', fake_held))
        patches.enter_context(mock.patch.object(driver, '_bound_model_json', side_effect=data))
        patches.enter_context(mock.patch.object(driver, '_prepare_model_frontend', return_value=Path(self.spec['operation']['frontend_root'])))
        patches.enter_context(mock.patch.object(driver, '_accept_teacher_result', return_value={'typed': True}))
        patches.enter_context(mock.patch.object(driver.model_contract, 'qualifier_argv', return_value=['fixed-argv']))
        patches.enter_context(mock.patch.object(driver, '_visible_devices', side_effect=AssertionError('CUDA device query')))
        patches.enter_context(mock.patch.object(driver, 'Guard', side_effect=AssertionError('pytest/Guard entered')))
        patches.enter_context(mock.patch.object(driver, '_call_owned_model', side_effect=lambda argv: calls.append(argv) or 0))
        patches.enter_context(mock.patch.dict(os.environ, {'TF_GGUF_ASSETS_DIR': '/frontend'}))
        self.addCleanup(patches.close)
        return calls

    def test_model_branch_calls_canonical_parent_once_no_cuda_pytest_or_guard(self):
        calls = self.model_run_controls()
        self.assertEqual(self.execute(), 0)
        self.assertEqual(calls, [['fixed-argv']])
        result = json.loads((self.evidence / 'owned-teacher-operation-result.json').read_text())
        self.assertTrue(result['source_after_match'])
        self.assertTrue(result['parent_cuda_uninitialized'])
        self.assertFalse((self.evidence / 'named-suite-result.json').exists())
        self.assertNotIn(driver.CHILD_EVIDENCE_ENV, os.environ)

    def test_model_parent_flag_and_frontend_cache_mismatch_fail_before_launch(self):
        calls = self.model_run_controls()
        self.runtime.torch.cuda._initialized = True
        with self.assertRaisesRegex(RuntimeError, 'uninitialized'):
            self.execute()
        self.assertEqual(calls, [])

    def test_model_failure_and_result_persistence_failure_preserve_primary(self):
        self.model_run_controls()
        primary = ValueError('actual owned model failure')
        def publish(path, value):
            if Path(path).name == 'owned-teacher-operation-result.json':
                raise OSError('result storage failed')
            self.publish(path, value)
        with mock.patch.object(driver, '_call_owned_model', side_effect=primary):
            try:
                self.execute(publisher=publish)
            except BaseException as caught:
                self.assertIs(caught, primary)
                self.assertIsNotNone(caught.__cause__)
            else:
                self.fail('model and persistence failures suppressed')
        self.assertNotIn(driver.CHILD_EVIDENCE_ENV, os.environ)

    def test_model_observer_limits_are_explicit_optional_and_enforced(self):
        limits = driver.model_contract.MODEL_OBSERVER_LIMITS
        self.assertEqual(driver._native_map_rows('', limits=limits), {})
        for text in ('\n' * limits['map_lines'], 'x' * (limits['line_characters']+1)):
            with self.assertRaisesRegex(ValueError, 'budget exceeded'):
                driver._native_map_rows(text, limits=limits)
        with self.assertRaisesRegex(ValueError, 'unknown model observer'):
            driver.observe_native_elfs([], limits={**limits, 'elfs': 65})
        resources = driver.model_contract.child_observer_resources()
        self.assertEqual(resources['global_host_bytes'], 8 * resources['per_rank_host_bytes'])
        self.assertEqual(resources['limits'], limits)


    def test_model_observer_budgets_mount_fields_and_module_snapshot_before_materializing(self):
        limits = driver.model_contract.MODEL_OBSERVER_LIMITS
        text = '1 0 0:1 / /' + 'x' * limits['path_characters'] + ' rw - tmpfs tmpfs rw'
        with mock.patch.object(driver, '_read_text', return_value=text):
            with self.assertRaisesRegex(ValueError, 'mount field budget'):
                driver._mount_table(limits=limits)
        modules = dict.fromkeys(range(limits['modules']+1))
        with mock.patch.object(driver, '_read_native_maps', return_value=''), \
                mock.patch.object(driver, 'sys', types.SimpleNamespace(modules=modules)):
            with self.assertRaisesRegex(ValueError, 'module snapshot budget'):
                driver.observe_native_elfs([], limits=limits)
        projection = driver.model_contract.child_observer_resources()
        self.assertLessEqual(projection['projection']['simultaneous_controlled_bytes'], projection['global_host_bytes'])

    def test_model_cleanup_attempts_all_restorations_checks_and_reports_despite_exhaustion(self):
        self.model_run_controls()
        primary = ValueError('authentic model primary')
        attempted, paths = [], []
        class Environment(dict):
            def pop(env, key, default=None):
                if key in (driver.model_contract.CHILD_ENV, driver.model_contract.DRIVER_ENV,
                           driver.model_contract.DRIVER_SHA_ENV, driver.model_contract.PREFIX_ENV):
                    attempted.append(key)
                    raise MemoryError('independent restoration failure')
                return super().pop(key, default)
        class Paths(list):
            def __setitem__(items, key, value):
                if isinstance(key, slice) and key.start is None and key.stop is None:
                    paths.append('restore')
                    raise MemoryError('path restoration failed')
                return super().__setitem__(key, value)
        env = Environment(os.environ)
        check_counts = {}
        def fail_post(name, function):
            def call(*args):
                check_counts[name] = check_counts.get(name, 0)+1
                if check_counts[name] > 1:
                    raise OSError('independent postcheck failure '+name)
                return function(*args)
            return call
        def publish(path, value):
            if Path(path).name == 'owned-teacher-operation-result.json':
                raise OSError('independent report publication failure')
            self.publish(path, value)
        with contextlib.ExitStack() as patches:
            patches.enter_context(mock.patch.object(driver.os, 'environ', env))
            patches.enter_context(mock.patch.object(driver.sys, 'path', Paths(sys.path)))
            for name in ('check_tests', '_admit_runtime', '_installed_origins', '_parent_cuda_flag'):
                patches.enter_context(mock.patch.object(driver, name, side_effect=fail_post(name, getattr(driver, name))))
            patches.enter_context(mock.patch.object(driver, '_call_owned_model', side_effect=primary))
            patches.enter_context(mock.patch('tools.worker_lifetime._raise_failures', side_effect=MemoryError('neutral grouping exhausted')))
            try:
                self.execute(publisher=publish)
            except BaseException as caught:
                self.assertIs(caught, primary)
                self.assertIsInstance(caught.__cause__, MemoryError)
                statuses = []
                trace = caught.__traceback__
                while trace is not None:
                    value = trace.tb_frame.f_locals.get('errors')
                    if isinstance(value, driver._FailureSlots):
                        statuses.extend(error for error in value.values if error is not None)
                    trace = trace.tb_next
                self.assertGreaterEqual(len(statuses), 10)
            else:
                self.fail('cleanup/report/transport exhaustion replaced or suppressed primary')
        self.assertEqual(len(attempted), 4)
        self.assertEqual(paths, ['restore'])
        self.assertEqual(set(check_counts.values()), {2})
        self.assertEqual(json.loads((self.evidence/'phase.json').read_text())['phase'], 'owned-teacher-terminal')

    def assert_model_context_failure_incomplete(self, message):
        self.model_run_controls()
        @contextlib.contextmanager
        def fail_exit(*args):
            yield []
            raise OSError(message)
        with mock.patch.object(driver, '_held_model_assets', fail_exit):
            with self.assertRaisesRegex(OSError, message):
                self.execute()
        content = json.loads((self.evidence/'model-content-binding.json').read_text())
        result = json.loads((self.evidence/'owned-teacher-operation-result.json').read_text())
        self.assertFalse(content['completed'])
        self.assertFalse(result['content_binding_complete'])

    def test_model_postcheck_failure_cannot_label_content_completed(self):
        self.assert_model_context_failure_incomplete('asset postcheck failed')

    def test_model_close_failure_cannot_label_content_completed(self):
        self.assert_model_context_failure_incomplete('asset close failed')

    def test_model_error_rendering_exhaustion_preserves_primary_and_terminal_phase(self):
        self.model_run_controls()
        primary = ValueError('authentic client failure')
        with mock.patch.object(driver, '_call_owned_model', side_effect=primary), \
                mock.patch.object(driver, '_error', side_effect=MemoryError('error rendering exhausted')):
            try:
                self.execute()
            except BaseException as caught:
                self.assertIs(caught, primary)
            else:
                self.fail('error rendering replaced primary')
        self.assertEqual(json.loads((self.evidence/'phase.json').read_text())['phase'], 'owned-teacher-terminal')

    def test_held_asset_all_postchecks_and_neutral_transport_keep_original_failure(self):
        import time
        records = []
        for index in range(2):
            path = self.root/('held-'+str(index))
            path.write_bytes(b'payload')
            records.append({'path': str(path), 'bytes': 7, 'sha256': hashlib.sha256(b'payload').hexdigest()})
        seen, postchecks = {}, []
        actual = driver._check_opened_asset
        def check(held):
            path = held['identity']['path']
            seen[path] = seen.get(path, 0)+1
            if seen[path] > 1:
                postchecks.append(path)
                raise OSError('independent retained-asset postcheck')
            return actual(held)
        primary = ValueError('authentic body failure')
        with mock.patch.object(driver, '_check_opened_asset', side_effect=check), \
                mock.patch('tools.worker_lifetime._raise_failures', side_effect=MemoryError('neutral transport exhaustion')):
            try:
                with driver._held_model_assets(records, time.monotonic()+10, {'assets': []}):
                    raise primary
            except BaseException as caught:
                self.assertIs(caught, primary)
            else:
                self.fail('asset cleanup/transport lost primary')
        self.assertEqual(postchecks, [record['path'] for record in records])

    def teacher_result_fixture(self):
        spec = self.model_spec()
        teacher, children = self.evidence/'teacher', self.evidence/'child-native'
        teacher.mkdir()
        children.mkdir()
        c = driver.model_contract
        def publish(name, data, rank=None):
            path = self.evidence/name
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(data))
            identity = {key: driver.file_identity(path)[key] for key in ('bytes', 'sha256')}
            return {'file': name, **identity, **({} if rank is None else {'rank': rank})}
        ranks, startups, native = [], [], []
        observer = {'path': str(self.source/'tools/v100_native_contract_driver.py'),
                    **spec['source_files']['tools/v100_native_contract_driver.py']}
        access = {'uids': [1000]*4, 'gids': [1000]*4, 'NoNewPrivs': '0',
                  'caps': {key: '0000010000000000' for key in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')}}
        for rank in range(8):
            ranks.append(publish(f'teacher/teacher-rank{rank}/summary.json', {'rank': rank}, rank))
            startups.append(publish(f'teacher/rank-{rank}.json', {'rank': rank, 'world': 8,
                'teacher_worker_PID': rank+10, 'teacher_server_instance_id': 'fresh-server-'+str(rank),
                'teacher_observer': observer}, rank))
            native.append(publish(f'child-native/teacher-closed-rank-{rank}-pid-{rank+10}.json', {
                'schema': 'tensorfold.native-elf-observation-v3', 'identity_method': 'kernel-vma-backed-fd-v1',
                'rank': rank, 'PID': rank+10, 'tag': 'teacher-closed', 'operation_limits': dict(c.MODEL_OBSERVER_LIMITS),
                'observer': spec['source_files']['tools/v100_native_contract_driver.py'], 'access_precondition': access,
                'missing_required_prefixes': [], 'mapped_elfs': [{'name': 'tf_gguf_fixture'}]}, rank))
        authority = publish('teacher/teacher-authority.json', {'schema': 'tensorfold.owned-teacher-authority-v1',
            'launch_exit': 0, 'cohort_complete': True, 'mode': 'capture', 'reference_sha256': None, 'teacher_ids': [7, 8],
            'rank_receipts': [{**row, 'file': row['file'].removeprefix('teacher/')} for row in ranks],
            'startup_receipts': [{**row, 'file': row['file'].removeprefix('teacher/')} for row in startups]})
        report = publish('teacher/teacher-client-report.json', {'schema': 'tensorfold.owned-teacher-client-report-v1',
            'complete': True, 'post_attempts': 1, 'mode': 'capture', 'teacher_plan_sha256': 'a'*64,
            'teacher_client_sha256': 'a'*64, 'server_instance_id': 'fresh-server-0', 'token_ids': [7, 8],
            'completion_tokens': 2, 'prompt_tokens': 52, 'cached_tokens': 0, 'drafts': False, 'drafted': 0, 'accepted': 0,
            'client_protocol_passed': True, 'quality_acceptance': False, 'P3_valid': False})
        result = {'schema': c.RESULT_SCHEMA, 'teacher_plan_sha256': 'a'*64, 'teacher_client_sha256': 'a'*64,
                  'launch_exit': 0, 'client_report': report, 'teacher_authority': authority,
                  'rank_receipts': ranks, 'startup_receipts': startups, 'child_native_records': native}
        (teacher/'teacher-model-result.json').write_text(json.dumps(result))
        return result

    def test_actual_bound_evidence_accepts_exact_eight_children_and_matching_cold_client(self):
        expected = self.teacher_result_fixture()
        result = driver._accept_teacher_result(self.evidence, self.spec, self.source)
        self.assertEqual(result, expected)
        record = self.evidence/expected['child_native_records'][4]['file']
        record.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'referenced file differs'):
            driver._accept_teacher_result(self.evidence, self.spec, self.source)

    def test_model_result_refuses_missing_rank_or_changed_input_before_authority_use(self):
        expected = self.teacher_result_fixture()
        target = self.evidence/'teacher/teacher-model-result.json'
        for change in ({'child_native_records': expected['child_native_records'][:-1]},
                       {'teacher_plan_sha256': 'b'*64}, {'launch_exit': 1},
                       {'client_report': {**expected['client_report'], 'file': '../escape'}}):
            with self.subTest(change=change):
                target.write_text(json.dumps({**expected, **change}))
                with self.assertRaises((ValueError, RuntimeError)):
                    driver._accept_teacher_result(self.evidence, self.spec, self.source)


    def replay_fixture(self):
        operation = self.model_operation()
        def row(name, rank=None):
            return {'file': name, 'bytes': 1, 'sha256': 'a'*64, **({} if rank is None else {'rank': rank})}
        authority = {'teacher_ids': [18, 22, 17, 248046], 'declared_identity': {'profile_sha256': 'b'*64},
                     'rank_receipts': [row(f'teacher-rank{rank}/summary.json', rank) for rank in range(8)],
                     'startup_receipts': [row(f'rank-{rank}.json', rank) for rank in range(8)],
                     'frames': [{'vector': row(f'teacher-rank0/frame-{index:06d}.npy')} for index in range(4)]}
        files = {'teacher/teacher-authority.json': {'bytes': 1, 'sha256': 'a'*64}}
        for item in (*authority['rank_receipts'], *authority['startup_receipts'],
                     *(frame['vector'] for frame in authority['frames'])):
            files['teacher/'+item['file']] = {key: item[key] for key in ('bytes', 'sha256')}
        operation['reference'] = {'root': '/reference', 'authority': {'path': '/reference/teacher/teacher-authority.json',
            'bytes': 1, 'sha256': 'a'*64}, 'files': files,
            'host_result': {'path': '/reference/host-result.json', 'bytes': 1, 'sha256': 'c'*64},
            'completion': {'path': '/operator/reference-capture-completion.json', 'bytes': 1, 'sha256': 'd'*64}}
        plan = {'mode': 'replay', 'context': 262144, 'max_frames': 4, 'input_ids': list(range(52)),
                'reference': {'path': operation['reference']['authority']['path'], 'sha256': 'a'*64},
                'identity': {'profile_sha256': 'b'*64}}
        prepared = {'data': plan, 'input_sha256': operation['teacher_plan']['sha256'],
                    'reference_sha256': 'a'*64, 'teacher_ids': authority['teacher_ids']}
        completion = {'schema': 'tensorfold.owned-teacher-completion-v1', 'host_result_sha256': 'c'*64,
                      'authority_sha256': 'a'*64, 'host_cli_exit': 0, 'operation_gate_sha256': 'e'*64,
                      'review_sha256': 'f'*64, 'terminal_observation_sha256': '0'*64}
        return operation, plan, prepared, authority, completion

    def test_capture_guard_64_and_no_reference_are_unchanged(self):
        c = driver.model_contract
        operation = self.model_operation()
        plan = {'mode': 'capture', 'context': 262144, 'max_frames': 64}
        self.assertEqual(c.validate_teacher_operation_plan(operation, plan, self.teacher_client())['mode'], 'capture')
        with self.assertRaises(ValueError):
            c.validate_teacher_operation_plan(operation, {**plan, 'max_frames': 4}, self.teacher_client())
        replay = self.replay_fixture()[0]
        with self.assertRaises(ValueError):
            c.validate_teacher_operation_plan(replay, plan, self.teacher_client())
        with self.assertRaises(ValueError):
            c.validate_operation({**operation, 'reference': None})

    def test_replay_requires_exact_verified_reference_profile_prefix_and_closure(self):
        c = driver.model_contract
        operation, plan, prepared, authority, _ = self.replay_fixture()
        c.validate_operation(operation)
        result = c.validate_teacher_operation_plan(operation, plan, self.teacher_client(), prepared=prepared, reference=authority)
        self.assertEqual(result, {'mode': 'replay', 'reference_sha256': 'a'*64, 'teacher_ids': (18, 22, 17, 248046)})
        for candidate in (None, {**prepared, 'reference_sha256': 'e'*64},
                          {**prepared, 'input_sha256': 'e'*64}, {**prepared, 'teacher_ids': []},
                          {**prepared, 'teacher_ids': [18, 22, 17]}):
            with self.subTest(prepared=candidate), self.assertRaises(ValueError):
                c.validate_teacher_operation_plan(operation, plan, self.teacher_client(), prepared=candidate, reference=authority)
        for change in ({'max_frames': 64}, {'input_ids': list(range(51))},
                       {'reference': {'path': '/reference/other.json', 'sha256': 'a'*64}},
                       {'identity': {'profile_sha256': 'e'*64}}):
            changed = {**plan, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_teacher_operation_plan(operation, changed, self.teacher_client(),
                    prepared={**prepared, 'data': changed}, reference=authority)
        with self.assertRaises(ValueError):
            c.validate_teacher_operation_plan(self.model_operation(), plan, self.teacher_client(), prepared=prepared, reference=authority)

    def test_replay_asset_closure_is_32_distinct_and_authority_not_hashed_twice(self):
        c = driver.model_contract
        operation = self.replay_fixture()[0]
        assets = c.operation_assets(operation)
        self.assertEqual(len(assets), 32)
        self.assertEqual(len({item['path'] for item in assets}), 32)
        self.assertEqual(assets[:9], c.operation_assets(self.model_operation()))
        reference = operation['reference']
        for change in ({'files': {name: row for name, row in reference['files'].items() if 'frame-000003' not in name}},
                       {'files': {**reference['files'], '../escape': {'bytes': 1, 'sha256': 'a'*64}}},
                       {'host_result': {**reference['host_result'], 'path': '/cache/host-result.json'}},
                       {'completion': {**reference['completion'], 'path': '/reference/completion.json'}}):
            if 'files' in change and len(change['files']) == 20:
                # A smaller complete closure is valid DATA, but must disagree
                # with the actual four-frame verified capture at plan binding.
                small = {**operation, 'reference': {**reference, **change}}
                _, plan, prepared, authority, _ = self.replay_fixture()
                with self.assertRaises(ValueError):
                    c.validate_teacher_operation_plan(small, plan, self.teacher_client(), prepared=prepared, reference=authority)
            else:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.validate_operation({**operation, 'reference': {**reference, **change}})

    def test_external_cli_completion_is_exact_bound_and_not_inferred_from_attach(self):
        c = driver.model_contract
        completion = self.replay_fixture()[4]
        self.assertIs(c.validate_teacher_completion(completion, host_result_sha256='c'*64, authority_sha256='a'*64), completion)
        for change in ({'host_cli_exit': 1}, {'host_cli_exit': False}, {'host_result_sha256': 'e'*64},
                       {'authority_sha256': 'e'*64}, {'terminal_observation_sha256': None}, {'attach_exit': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_teacher_completion({**completion, **change}, host_result_sha256='c'*64, authority_sha256='a'*64)

    def test_replay_result_mode_reference_and_frozen_ids_are_independently_bound(self):
        expected = self.teacher_result_fixture()
        result_path = self.evidence/'teacher/teacher-model-result.json'
        def change(item, updates):
            path = self.evidence/item['file']
            data = json.loads(path.read_text())
            data.update(updates)
            path.write_text(json.dumps(data))
            item.update({key: driver.file_identity(path)[key] for key in ('bytes', 'sha256')})
        change(expected['teacher_authority'], {'mode': 'replay', 'reference_sha256': 'e'*64})
        change(expected['client_report'], {'mode': 'replay'})
        result_path.write_text(json.dumps(expected))
        binding = {'mode': 'replay', 'reference_sha256': 'e'*64, 'teacher_ids': (7, 8)}
        self.assertEqual(driver._accept_teacher_result(self.evidence, self.spec, self.source, teacher_binding=binding), expected)
        for changed in ({**binding, 'reference_sha256': 'f'*64}, {**binding, 'teacher_ids': (7, 9)}):
            with self.assertRaisesRegex(RuntimeError, 'all-rank authority'):
                driver._accept_teacher_result(self.evidence, self.spec, self.source, teacher_binding=changed)
        change(expected['client_report'], {'mode': 'capture'})
        result_path.write_text(json.dumps(expected))
        with self.assertRaisesRegex(RuntimeError, 'cold teacher client'):
            driver._accept_teacher_result(self.evidence, self.spec, self.source, teacher_binding=binding)

    def test_replay_canonical_preparation_refusal_stops_before_any_model_launch(self):
        calls = self.model_run_controls()
        operation, plan, _, _, _ = self.replay_fixture()
        self.spec['operation'] = operation
        client = self.teacher_client()
        def data(path, *args, **kwargs):
            return client if str(path).endswith('teacher-client.json') else plan
        with mock.patch.object(driver, '_bound_model_json', side_effect=data), \
                mock.patch.object(driver, '_prepare_replay_plan', side_effect=ValueError('capture is not positively closed')):
            with self.assertRaisesRegex(ValueError, 'not positively closed'):
                self.execute()
        self.assertEqual(calls, [])


    def test_replay_positive_driver_path_verifies_before_launch_and_passes_mode_binding(self):
        calls = self.model_run_controls()
        operation, plan, prepared, authority, completion = self.replay_fixture()
        self.spec['operation'] = operation
        client = self.teacher_client()
        def data(path, *args, **kwargs):
            name = str(path)
            if name.endswith('teacher-client.json'):
                return client
            if name == operation['reference']['authority']['path']:
                return authority
            if name == operation['reference']['completion']['path']:
                return completion
            if name == operation['reference']['host_result']['path']:
                return {'originalHost': 'separately validated boundary fixture'}
            return plan
        events = []
        def verify(_operation):
            self.assertEqual(calls, [])
            events.append('verified-original-capture')
            return prepared
        with mock.patch.object(driver, '_bound_model_json', side_effect=data), \
                mock.patch.object(driver, '_prepare_replay_plan', side_effect=verify), \
                mock.patch.object(driver.model_contract, 'validate_completed_teacher_host', return_value={'operation': self.model_operation()}) as positive_host, \
                mock.patch.object(driver, '_accept_teacher_result', return_value={'typed': True}) as accept:
            self.assertEqual(self.execute(), 0)
        self.assertEqual(events, ['verified-original-capture'])
        self.assertEqual(calls, [['fixed-argv']])
        positive_host.assert_called_once()
        self.assertEqual(accept.call_args.kwargs['teacher_binding'], {
            'mode': 'replay', 'reference_sha256': 'a'*64, 'teacher_ids': (18, 22, 17, 248046)})


    def test_long_reference_socket_budget_is_finite_positive_and_within_whole(self):
        c = driver.model_contract
        client = {**self.teacher_client(), 'readiness_timeout_s': 600, 'whole_timeout_s': 3900, 'socket_timeout_s': 3000}
        self.assertIs(c.validate_teacher_client(client), client)
        self.assertEqual(self.teacher_client()['whole_timeout_s'] + self.teacher_client()['socket_timeout_s'], 4200)
        for value in (True, False, float('nan'), float('inf'), -float('inf'), -1, 0, 3901):
            with self.subTest(socket=value), self.assertRaises(ValueError):
                c.validate_teacher_client({**client, 'socket_timeout_s': value})
        for change in ({'readiness_timeout_s': 3601}, {'whole_timeout_s': 7501},
                       {'readiness_timeout_s': 3901}, {'whole_timeout_s': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_teacher_client({**client, **change})

    def timeout_run_controls(self, *, whole=3900, socket=3000, elapsed=810):
        calls = self.model_run_controls()
        client = {**self.teacher_client(), 'readiness_timeout_s': 600,
                  'whole_timeout_s': whole, 'socket_timeout_s': socket}
        plan = {'mode': 'capture', 'context': 262144, 'max_frames': 64}
        clock = [1000.0]
        def data(path, *args, **kwargs):
            return client if str(path).endswith('teacher-client.json') else plan
        def prepare(operation):
            clock[0] += elapsed  # Actual time spent before worker admission.
            return Path(operation['frontend_root'])
        patches = contextlib.ExitStack()
        patches.enter_context(mock.patch.object(driver, '_bound_model_json', side_effect=data))
        patches.enter_context(mock.patch.object(driver, '_prepare_model_frontend', side_effect=prepare))
        patches.enter_context(mock.patch.object(driver.time, 'monotonic', side_effect=lambda: clock[0]))
        self.addCleanup(patches.close)
        return calls

    def test_whole_budget_that_cannot_fund_socket_tail_refuses_before_workers(self):
        # Whole alone fits even after parent work; whole+socket exceeds the
        # remaining7710 work budget by one second and must never start ranks.
        calls = self.timeout_run_controls(elapsed=811)
        with self.assertRaisesRegex(TimeoutError, 'whole deadline and socket tail'):
            self.execute()
        self.assertEqual(calls, [])
        result = json.loads((self.evidence/'owned-teacher-operation-result.json').read_text())
        self.assertFalse(result['content_binding_complete'])

    def test_exact_whole_plus_socket_boundary_includes_parent_elapsed_time(self):
        # 7800outer-90retire-810actual parent =6900 =3900whole+3000tail.
        calls = self.timeout_run_controls(elapsed=810)
        self.assertEqual(self.execute(), 0)
        self.assertEqual(calls, [['fixed-argv']])
        result = json.loads((self.evidence/'owned-teacher-operation-result.json').read_text())
        self.assertTrue(result['content_binding_complete'])

    def test_old_max_whole_cannot_hide_an_unfunded_short_socket_tail(self):
        # Valid DATA independently:7500+300 must still refuse under7710 even
        # with zero parent work. Actual short3900/300 remains funded elsewhere.
        calls = self.timeout_run_controls(whole=7500, socket=300, elapsed=0)
        with self.assertRaisesRegex(TimeoutError, 'whole deadline and socket tail'):
            self.execute()
        self.assertEqual(calls, [])


    def test_phase_publication_time_is_charged_before_final_worker_funding(self):
        calls = self.model_run_controls()
        client = {**self.teacher_client(), 'readiness_timeout_s': 600,
                  'whole_timeout_s': 3900, 'socket_timeout_s': 3000}
        plan = {'mode': 'capture', 'context': 262144, 'max_frames': 64}
        clock, phases = [1000.0], []
        def data(path, *args, **kwargs):
            return client if str(path).endswith('teacher-client.json') else plan
        def prepare(operation):
            clock[0] += 809  #6901 remains: enough before phase publication.
            return Path(operation['frontend_root'])
        def publish(path, value):
            if Path(path).name == 'phase.json' and value['phase'] == 'owned-teacher-model':
                self.assertTrue(driver.model_contract.qualifier_argv.called)
                phases.append(value['phase'])
                clock[0] += 2  #6899 remains: synchronous fsync consumed funding.
            self.publish(path, value)
        with mock.patch.object(driver, '_bound_model_json', side_effect=data), \
                mock.patch.object(driver, '_prepare_model_frontend', side_effect=prepare), \
                mock.patch.object(driver.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'whole deadline and socket tail'):
                self.execute(publisher=publish)
        self.assertEqual(phases, ['owned-teacher-model'])
        self.assertEqual(calls, [])
        result = json.loads((self.evidence/'owned-teacher-operation-result.json').read_text())
        self.assertFalse(result['content_binding_complete'])


    def benchmark_setup(self):
        c = driver.model_contract
        if 'modules' in self.spec:
            self.model_spec()
        operation = self.spec['operation']
        operation['benchmark_plan'] = {**operation.pop('teacher_plan'), 'path': '/operator/benchmark-plan.json'}
        operation['benchmark_client'] = {**operation.pop('teacher_client'), 'path': '/operator/benchmark-client.json'}
        operation['trace'] = {'prefill_start': 1, 'host_obligation_bytes': 4096,
                              'device_headroom_bytes': 8192, 'disk_bytes': 1 << 20}
        self.spec.update(schema=c.BENCHMARK_SCHEMA, mode=c.BENCHMARK_MODE)
        for name in ('bench_concurrent.py', 'v100_adaptive_qualification.py', 'openai_protocol.py',
                     'worker_lifetime.py', 'benchmark_output.py'):
            path = self.source/'tools'/name
            path.write_bytes((Path(driver.__file__).parent/name).read_bytes())
            self.spec['source_files']['tools/'+name] = {k: driver.file_identity(path)[k] for k in ('bytes', 'sha256')}
        for name in ('cli.py', 'families/qwen4_exp/cuda/engine.py', 'families/qwen4_exp/cuda/multi.py',
                     'families/qwen4_exp/cuda/decode.py', 'cuda/sampling.py'):
            path = self.app/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# installed source '+name.encode())
        self.spec['expected_app'] = driver.app_inventory(self.app)
        client = self.teacher_client()
        client.update(schema=c.BENCHMARK_CLIENT_SCHEMA, benchmark_plan_sha256=client.pop('teacher_plan_sha256'), sse_line_bytes=4096)
        digest = hashlib.sha256(b'[1,2,2,2]').hexdigest()
        cohort = {'id': 7, 'warmup': False, 'level': 1, 'tokens': 2, 'input_tokens': 4,
                  'requests': [{'index': 0, 'prefix': [1], 'input_tokens': 4, 'prompt_sha256': digest}]}
        plan = {'schema': 'tensorfold.phase-matrix-plan-v1', 'model': client['model'], 'profile': 'q8',
                'admin_cap': 24, 'context': 262144, 'levels': [1], 'reps': 1, 'excluded_levels': [],
                'tokens': 2, 'inputs': [4], 'cohorts': [cohort], 'filler_ids': [2], 'run_identity': '1'*32}
        binding = c.validate_benchmark_operation_plan(operation, plan, client)
        return plan, client, binding

    def benchmark_result_fixture(self):
        plan, client, binding = self.benchmark_setup()
        c, op = driver.model_contract, self.spec['operation']
        def publish(name, data, rank=None):
            path = self.evidence/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            return {'file': name, **{k: driver.file_identity(path)[k] for k in ('bytes', 'sha256')},
                    **({} if rank is None else {'rank': rank})}
        observer = {'path': str(self.source/'tools/v100_native_contract_driver.py'),
                    **self.spec['source_files']['tools/v100_native_contract_driver.py']}
        access = {'uids': [1000]*4, 'gids': [1000]*4, 'NoNewPrivs': '0',
                  'caps': {k: '0000010000000000' for k in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')}}
        resources = {'trace_grants': op['trace'], 'global_host_obligation_bytes': 10000}
        headroom = {'free_bytes': 16384, 'total_bytes': 32768, 'required_free_bytes': 8192, 'scope': 'measured admission only'}
        startups, ranks, native = [], [], []
        for rank in range(8):
            semantics = {'method': 'none', 'context': 262144, 'resident_slots': 24, 'full_resident': True,
                'state_kv_identity': ['bound-state'], 'capacity_min': 262145, 'capacity_max': 262145,
                'limit_min': 262145, 'limit_max': 262145, 'logical_min': 262144, 'logical_max': 262144,
                'target_rows': 160, 'prefill_rows': 160, 'target_family': {'kernel_family': 'torch-reference', 'graph_family': 'uncaptured'},
                'prefill_family': {'kernel_family': 'torch-reference', 'graph_family': 'uncaptured'}}
            profile = {'rank': rank, 'benchmark_semantics': semantics}
            windows = {**{key: semantics[key] for key in ('full_resident', 'capacity_min', 'capacity_max',
                'limit_min', 'limit_max', 'logical_min', 'logical_max')}, 'slots': 24}
            startups.append(publish(f'benchmark/rank-{rank}.json', {'rank': rank, 'world': 8,
                'benchmark_worker_PID': rank+10, 'benchmark_server_instance_id': 'server-'+str(rank),
                'benchmark_observer': observer, 'benchmark_profile': profile, 'benchmark_windows': windows,
                'benchmark_resources': resources, 'benchmark_headroom': headroom,
                'tool_sha256': self.spec['source_files']['tools/v100_owned_qualification.py']['sha256'],
                'runtime_cli_sha256': self.spec['expected_app']['cli.py']['sha256']}, rank))
            ranks.append(publish(f'benchmark/benchmark-rank{rank}/summary.json', {
                'schema': 'tensorfold.owned-benchmark-rank-result-v1', 'rank': rank, 'world': 8,
                'closed': True, 'finished': True, 'failed': False, 'benchmark_plan_sha256': 'a'*64,
                'benchmark_client_sha256': 'a'*64, 'run_identity': plan['run_identity'],
                'server_instance_id': 'server-'+str(rank), 'actual_owned_profile': profile,
                'resources': resources, 'headroom': headroom, 'qualification': False, 'P3_valid': False,
                'runtime_sources': {**{key: self.spec['expected_app'][name]['sha256'] for key, name in (
                    ('engine', 'families/qwen4_exp/cuda/engine.py'), ('multi', 'families/qwen4_exp/cuda/multi.py'),
                    ('decode', 'families/qwen4_exp/cuda/decode.py'), ('sampling', 'cuda/sampling.py'))},
                    'observer_tool': self.spec['source_files']['tools/v100_owned_qualification.py']['sha256']},
                'cohorts': [{'id': 7, 'prompt_hashes': [binding['prompt_sha256']], 'completed': 1, 'level': 1}]}, rank))
            native.append(publish(f'child-native/benchmark-closed-rank-{rank}-pid-{rank+10}.json', {
                'schema': 'tensorfold.native-elf-observation-v3', 'identity_method': 'kernel-vma-backed-fd-v1',
                'rank': rank, 'PID': rank+10, 'tag': 'benchmark-closed', 'operation_limits': dict(c.MODEL_OBSERVER_LIMITS),
                'observer': self.spec['source_files']['tools/v100_native_contract_driver.py'],
                'access_precondition': access, 'missing_required_prefixes': [], 'mapped_elfs': [{'name': 'tf_gguf_fixture'}]}, rank))
        report = publish('benchmark/benchmark-client-report.json', {
            'schema': 'tensorfold.owned-benchmark-client-report-v1', 'complete': True,
            'client_protocol_passed': True, 'matrix_passed': True, 'observed_trace': True,
            'benchmark_plan_sha256': 'a'*64, 'benchmark_client_sha256': 'a'*64,
            'run_identity': plan['run_identity'], 'server_instance_id': 'server-0', 'request_count': 1,
            'matrix_tool_sha256': self.spec['source_files']['tools/bench_concurrent.py']['sha256'],
            'quality_acceptance': False, 'P3_valid': False})
        matrix = publish('benchmark/matrix-results.json', {'schema': 'tensorfold.phase-matrix-result-v1',
            'passed': True, 'unrun_cohorts': [], 'plan': plan, 'instance_id': 'server-0',
            'cohorts': [{'cohort': 7, 'passed': True, 'status': 'measured', 'warmup': False}]})
        selection = {'status': 'OBSERVED', 'reason': None, 'run_identity': plan['run_identity'],
            'cohort_id': 7, 'prompt_sha256': binding['prompt_sha256'], 'input_tokens': 4,
            'requested_start': 1, 'prepared_calls': 1, 'prepared_returned': True, 'P3_valid': False,
            'boundary_start': 1, 'actual_start': 1, 'actual_count': 3, 'slot': 0, 'slot_generation': 0,
            'state_pos': 0, 'actual_state_pos': 0, 'wire': 0, 'profiler_entry_headroom': headroom}
        trace = {'inventory': publish('benchmark/trace/inventory.json', {'rank': 0, 'sm70_profile': 'reference',
                                                                       'actual_matrix_samples': [{'role': 'head'}]}),
                 'selection': publish('benchmark/trace/prefill-selection.json', selection),
                 'chrome': publish('benchmark/trace/c1-prefill-start-ge-1-round.json', {'traceEvents': []})}
        trace['receipt'] = publish('benchmark/trace/c1-prefill-start-ge-1-round-receipt.json', {
            'rank': 0, 'primary': None, 'post_errors': [], 'prefill_selection': selection, 'P3_valid': False,
            'exported_trace_bytes': trace['chrome']['bytes']})
        result = {'schema': c.BENCHMARK_RESULT_SCHEMA, 'benchmark_plan_sha256': 'a'*64,
            'benchmark_client_sha256': 'a'*64, 'launch_exit': 0, 'client_report': report,
            'matrix_report': matrix, 'startup_receipts': startups, 'rank_receipts': ranks,
            'child_native_records': native, 'trace': trace, 'P3_valid': False}
        (self.evidence/'benchmark/benchmark-model-result.json').write_text(json.dumps(result))
        return plan, client, binding, result

    def test_closed_benchmark_schema_cannot_be_teacher_or_pytest(self):
        self.benchmark_setup()
        self.assertIs(driver.validate_spec(self.spec), self.spec)
        for change in ({'mode': driver.model_contract.MODE}, {'modules': ['tests/x.py']},
                       {'schema': driver.model_contract.SCHEMA}, {'source_files': {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                driver.validate_spec({**self.spec, **change})

    def accept_benchmark_fixture(self, fixture):
        plan, _, binding, _ = fixture
        return driver._accept_benchmark_result(self.evidence, self.spec, self.source, binding=binding, plan=plan)

    def test_closed_benchmark_accepts_actual_eight_native_close_and_one_observed_trace(self):
        fixture = self.benchmark_result_fixture()
        self.assertEqual(self.accept_benchmark_fixture(fixture), fixture[3])
        self.assertFalse((self.evidence/'teacher').exists())

    def test_closed_benchmark_rejects_rebound_failed_client_cohort_close_source_and_trace(self):
        fixture = self.benchmark_result_fixture()
        original_result = fixture[3]
        original_files = {p: p.read_bytes() for p in self.evidence.rglob('*') if p.is_file()}
        mutations = [('client_report', {'complete': False}), ('client_report', {'P3_valid': True}),
            ('client_report', {'request_count': True}), ('client_report', {'matrix_tool_sha256': '0'*64}),
            ('matrix_report', {'unrun_cohorts': [7]}), ('matrix_report', {'instance_id': 'foreign'}),
            ('rank_receipts', {'closed': False}), ('rank_receipts', {'runtime_sources': {}}),
            ('rank_receipts', {'cohorts': []}), ('rank_receipts', {'headroom': {}}),
            ('startup_receipts', {'benchmark_worker_PID': 9999}),
            ('child_native_records', {'tag': 'teacher-closed'}),
            ('child_native_records', {'access_precondition': {}}),
            ('selection', {'status': 'SELECTED_UNVERIFIED'}), ('selection', {'actual_count': 0}),
            ('selection', {'slot_generation': True}), ('receipt', {'post_errors': ['export-failed']}),
            ('receipt', {'exported_trace_bytes': 1})]
        import copy
        for key, change in mutations:
            with self.subTest(key=key, change=change):
                for path, raw in original_files.items():
                    path.write_bytes(raw)
                result = copy.deepcopy(original_result)
                row = (result['trace'][key] if key in result['trace'] else
                       result[key][3] if isinstance(result[key], list) else result[key])
                path = self.evidence/row['file']
                data = json.loads(path.read_text())
                data.update(change)
                path.write_text(json.dumps(data))
                row.update({k: driver.file_identity(path)[k] for k in ('bytes', 'sha256')})
                (self.evidence/'benchmark/benchmark-model-result.json').write_text(json.dumps(result))
                with self.assertRaises((ValueError, RuntimeError)):
                    self.accept_benchmark_fixture(fixture)

    def test_closed_benchmark_rejects_extra_capture_and_wrong_mode_result(self):
        fixture = self.benchmark_result_fixture()
        extra = self.evidence/'benchmark/trace/c1-warmed-decode-round-1.json'
        extra.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'another capture'):
            self.accept_benchmark_fixture(fixture)
        extra.unlink()
        path = self.evidence/'benchmark/benchmark-model-result.json'
        raw = json.loads(path.read_text())
        raw['schema'] = driver.model_contract.RESULT_SCHEMA
        path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            self.accept_benchmark_fixture(fixture)

    def benchmark_run_controls(self):
        self.model_run_controls()
        plan, client, _ = self.benchmark_setup()
        calls, patches = [], contextlib.ExitStack()
        def call(argv, **kwargs):
            calls.append((argv, kwargs))
            return 0
        patches.enter_context(mock.patch.object(driver, '_load_benchmark_plan', return_value=plan))
        patches.enter_context(mock.patch.object(driver, '_bound_model_json', return_value=client))
        patches.enter_context(mock.patch.object(driver.model_contract, 'benchmark_qualifier_argv', return_value=['fixed-benchmark']))
        patches.enter_context(mock.patch.object(driver, '_accept_benchmark_result', return_value={'typed': 'benchmark'}))
        patches.enter_context(mock.patch.object(driver, '_call_owned_model', side_effect=call))
        self.addCleanup(patches.close)
        return calls, client

    def test_closed_benchmark_dispatches_private_operation_once_and_preserves_budget_tail(self):
        calls, _ = self.benchmark_run_controls()
        self.assertEqual(self.execute(), 0)
        self.assertEqual(calls, [(['fixed-benchmark'], {'benchmark_operation': self.spec['operation']})])
        result = json.loads((self.evidence/'owned-benchmark-operation-result.json').read_text())
        self.assertEqual(result['schema'], driver.model_contract.BENCHMARK_HOST_RESULT_SCHEMA)
        self.assertTrue(result['content_binding_complete'])
        self.assertFalse((self.evidence/'owned-teacher-operation-result.json').exists())

    def test_closed_benchmark_phase_elapsed_time_cannot_spend_socket_tail(self):
        calls, client = self.benchmark_run_controls()
        client.update(whole_timeout_s=3900, socket_timeout_s=3000)
        clock = [1000.0]
        def publish(path, value):
            if Path(path).name == 'phase.json' and value['phase'] == 'owned-benchmark-model':
                clock[0] += 811
            self.publish(path, value)
        with mock.patch.object(driver.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'whole deadline and socket tail'):
                self.execute(publisher=publish)
        self.assertEqual(calls, [])
        result = json.loads((self.evidence/'owned-benchmark-operation-result.json').read_text())
        self.assertFalse(result['content_binding_complete'])

    def test_closed_benchmark_failed_launch_and_publisher_keep_primary_and_terminal(self):
        self.benchmark_run_controls()
        primary = ValueError('authentic closed benchmark failure')
        def publish(path, value):
            if Path(path).name == 'owned-benchmark-operation-result.json':
                raise OSError('benchmark result publisher failed')
            self.publish(path, value)
        with mock.patch.object(driver, '_call_owned_model', side_effect=primary):
            try:
                self.execute(publisher=publish)
            except BaseException as caught:
                self.assertIs(caught, primary)
                self.assertIsNotNone(caught.__cause__)
            else:
                self.fail('closed benchmark failure was replaced or hidden')
        self.assertEqual(json.loads((self.evidence/'phase.json').read_text())['phase'], 'owned-benchmark-terminal')
        for name in (driver.model_contract.CHILD_ENV, driver.model_contract.DRIVER_ENV,
                     driver.model_contract.DRIVER_SHA_ENV, driver.model_contract.PREFIX_ENV):
            self.assertNotIn(name, os.environ)

    def test_closed_benchmark_loader_origin_and_real_prompt_hash_are_required(self):
        plan, _, _ = self.benchmark_setup()
        path = self.root/'plan.json'
        path.write_text(json.dumps(plan))
        code = """
import json,sys,types
from pathlib import Path
from tools import bench_concurrent, v100_native_contract_driver as driver
path,source=Path(sys.argv[1]),Path(sys.argv[2])
plan=json.loads(path.read_text())
fake=types.SimpleNamespace(__file__=str(source/'tools/bench_concurrent.py'),load_matrix=bench_concurrent.load_matrix)
sys.modules['bench_concurrent']=fake
operation={'benchmark_plan':{'path':str(path)}}
assert driver._load_benchmark_plan(operation,source)==plan
plan['cohorts'][0]['requests'][0]['prompt_sha256']='0'*64
path.write_text(json.dumps(plan))
try:driver._load_benchmark_plan(operation,source)
except ValueError as error:assert 'hash mismatch' in str(error)
else:raise AssertionError('invalid full prompt hash accepted')
fake.__file__=str(path)
try:driver._load_benchmark_plan(operation,source)
except ValueError as error:assert 'origin' in str(error)
else:raise AssertionError('foreign loader accepted')
"""
        env = {**os.environ, 'PYTHONPATH': str(Path(driver.__file__).resolve().parents[1]/'src')}
        result = subprocess.run([sys.executable, '-c', code, str(path), str(self.source)],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
