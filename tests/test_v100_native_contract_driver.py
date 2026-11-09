"""Stdlib-only orchestration tests; no installed runtime or GPU qualification."""
import contextlib
import hashlib
import io
import json
import multiprocessing
import mmap
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

    def mapping(self, path, *, inode=None, permissions='r-xp'):
        info = path.stat()
        encoded = str(path).replace(' ', r'\040')
        return ('1000-2000 ' + permissions + ' 00000000 '
                + format(os.major(info.st_dev), 'x') + ':' + format(os.minor(info.st_dev), 'x')
                + ' ' + str(info.st_ino if inode is None else inode) + ' ' + encoded + '\n')

    def test_observed_native_elf_is_hashed_without_extension_loading(self):
        library = self.native_library()
        self.spec['required_native_elf_prefixes'] = ['tf_fixture']
        # An actual executable file mapping, with NO Python module registration
        # or extension import; no instruction from this synthetic ELF executes.
        with library.open('rb') as stream, mmap.mmap(stream.fileno(), 0,
                flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_EXEC):
            self.assertNotIn('tf_fixture', sys.modules)
            self.assertEqual(self.execute(), 0)
        result = json.loads((self.evidence / 'actual-native-ELF.json').read_text())
        row = next(row for row in result['mapped_elfs'] if row['name'] == 'tf_fixture')
        self.assertEqual(row['sha256'], hashlib.sha256(library.read_bytes()).hexdigest())
        self.assertEqual(row['inode'], library.stat().st_ino)
        self.assertEqual(result['missing_required_parent_prefixes'], [])

    def test_python_registry_alone_cannot_satisfy_required_mapping(self):
        library = self.native_library()
        module = types.ModuleType('tf_fixture')
        module.__file__ = str(library)
        self.spec['required_native_elf_prefixes'] = ['tf_fixture']
        with mock.patch.dict(sys.modules, {'tf_fixture': module}), self.assertRaisesRegex(RuntimeError, 'actually mapped'):
            self.execute()
        result = json.loads((self.evidence / 'actual-native-ELF.json').read_text())
        self.assertEqual(result['python_registry']['tf_fixture'], str(library))
        self.assertEqual(result['missing_required_parent_prefixes'], ['tf_fixture'])

    def test_mapping_segments_deduplicate_and_escaped_path_is_verified(self):
        library = self.native_library('tf_space name.so')
        text = self.mapping(library) * 3 + self.mapping(library, permissions='r--p')
        with mock.patch.object(driver, '_read_native_maps', return_value=text):
            result = driver.observe_native_elfs(['tf_space'])
        self.assertEqual(len(result['mapped_elfs']), 1)
        self.assertEqual(result['mapped_elfs'][0]['path'], str(library))

    def test_mapping_identity_disagreement_and_deleted_files_are_errors(self):
        library = self.native_library()
        for text, expected in ((self.mapping(library, inode=library.stat().st_ino + 1), 'device/inode'),
                               (self.mapping(library).rstrip() + ' (deleted)\n', 'stable ordinary')):
            with self.subTest(text=text), mock.patch.object(driver, '_read_native_maps', return_value=text):
                with self.assertRaisesRegex((ValueError, RuntimeError), expected):
                    driver.observe_native_elfs(['tf_fixture'])

    def test_nonelf_symlink_and_mapping_change_are_not_absence(self):
        library = self.native_library()
        text = self.mapping(library)
        with mock.patch.object(driver, '_read_native_maps', side_effect=[text, '']):
            with self.assertRaisesRegex(RuntimeError, 'mappings changed'):
                driver.observe_native_elfs([])
        library.write_bytes(b'not an ELF')
        with mock.patch.object(driver, '_read_native_maps', return_value=text):
            with self.assertRaisesRegex(ValueError, 'not ELF'):
                driver.observe_native_elfs([])
        link = self.root / 'tf_symlink.so'
        link.symlink_to(library)
        with mock.patch.object(driver, '_read_native_maps', return_value=self.mapping(link)):
            with self.assertRaises(OSError):
                driver.observe_native_elfs([])

    def test_identity_mutation_and_descriptor_cleanup_preserve_authentic_failure(self):
        library = self.native_library()
        info = library.stat()
        changed = types.SimpleNamespace(**{field: getattr(info, field) for field in
            ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_mtime_ns', 'st_ctime_ns')})
        changed.st_size += 1
        with mock.patch.object(driver, '_read_native_maps', return_value=self.mapping(library)):
            with mock.patch.object(driver.os, 'fstat', side_effect=[info, changed]):
                with self.assertRaisesRegex(RuntimeError, 'changed while hashed'):
                    driver.observe_native_elfs([])
        primary, cleanup = OSError('body'), OSError('close')
        with mock.patch.object(driver.os, 'open', return_value=123), mock.patch.object(driver.os, 'close', side_effect=cleanup):
            with self.assertRaises(OSError) as caught:
                with driver._owned_descriptor('/unused', os.O_RDONLY):
                    raise primary
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cleanup)

    def test_descriptor_close_failure_retains_prior_cause_context_notes_and_policy(self):
        for suppression in (False, True):
            with self.subTest(suppression=suppression):
                primary = OSError('authentic body')
                cause, context, cleanup = ValueError('prior cause'), RuntimeError('prior context'), OSError('close')
                primary.__cause__, primary.__context__ = cause, context
                primary.__suppress_context__ = suppression
                primary.add_note('authentic note')
                with mock.patch.object(driver.os, 'open', return_value=123), mock.patch.object(driver.os, 'close', side_effect=cleanup):
                    with self.assertRaises(OSError) as caught:
                        with driver._owned_descriptor('/unused', os.O_RDONLY):
                            raise primary
                self.assertIs(caught.exception, primary)
                self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
                self.assertEqual(primary.__cause__.exceptions, (cause, context, cleanup))
                self.assertEqual(primary.__notes__[0], 'authentic note')
                # Existing neutral helper makes all owned cleanup causes explicit;
                # the prior context remains present even if formerly suppressed.
                self.assertTrue(primary.__suppress_context__)

    def test_short_raw_reads_hash_complete_mapped_file(self):
        library = self.native_library()
        read = driver.os.read
        def short_read(descriptor, count):
            return read(descriptor, min(count, 2))
        with mock.patch.object(driver, '_read_native_maps', return_value=self.mapping(library)):
            with mock.patch.object(driver.os, 'read', side_effect=short_read):
                report = driver.observe_native_elfs(['tf_fixture'])
        self.assertEqual(report['mapped_elfs'][0]['sha256'], hashlib.sha256(library.read_bytes()).hexdigest())

    def test_raw_elf_read_failure_and_close_failure_preserve_both(self):
        library = self.native_library()
        text = self.mapping(library)
        primary, cleanup = OSError('authentic ELF read'), OSError('FD close')
        close = driver.os.close
        def close_then_fail(descriptor):
            close(descriptor)
            raise cleanup
        with mock.patch.object(driver, '_read_native_maps', return_value=text):
            with mock.patch.object(driver.os, 'read', side_effect=primary), mock.patch.object(driver.os, 'close', side_effect=close_then_fail):
                with self.assertRaises(OSError) as caught:
                    driver.observe_native_elfs([])
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cleanup)

    def test_map_read_failure_and_missing_prefix_remain_distinct(self):
        primary = PermissionError('authentic maps read failure')
        with mock.patch.object(driver, '_read_native_maps', side_effect=primary):
            with self.assertRaises(PermissionError) as caught:
                driver.observe_native_elfs(['tf_fixture'])
        self.assertIs(caught.exception, primary)
        with mock.patch.object(driver, '_read_native_maps', return_value=''):
            self.assertEqual(driver.observe_native_elfs(['tf_fixture'])['missing_required_prefixes'], ['tf_fixture'])

    def test_child_record_is_scoped_durable_exclusive_and_source_identified(self):
        library = self.native_library()
        with mock.patch.object(driver, '_read_native_maps', return_value=self.mapping(library)):
            report = driver.record_native_elfs(['tf_fixture'], rank=7, tag='owner-head4-1', evidence_dir=self.evidence)
            with self.assertRaises(FileExistsError):
                driver.record_native_elfs(['tf_fixture'], rank=7, tag='owner-head4-1', evidence_dir=self.evidence)
        path = self.evidence / ('owner-head4-1-rank-7-pid-' + str(os.getpid()) + '.json')
        self.assertEqual(json.loads(path.read_text()), report)
        self.assertEqual(report['rank'], 7)
        self.assertEqual(report['PID'], os.getpid())
        self.assertEqual(report['observer']['sha256'], driver.file_identity(driver.__file__)['sha256'])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_missing_child_mapping_is_published_before_owning_failure(self):
        with mock.patch.object(driver, '_read_native_maps', return_value=''):
            with self.assertRaisesRegex(RuntimeError, 'required child'):
                driver.record_native_elfs(['tf_fixture'], rank=0, tag='missing', evidence_dir=self.evidence)
        records = list(self.evidence.glob('missing-*.json'))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_text())['missing_required_prefixes'], ['tf_fixture'])

    def test_child_publication_failure_is_owning_and_preserves_error(self):
        library = self.native_library()
        primary = OSError('durability failed')
        with mock.patch.object(driver, '_read_native_maps', return_value=self.mapping(library)):
            with mock.patch.object(driver.os, 'fsync', side_effect=primary):
                with self.assertRaises(OSError) as caught:
                    driver.record_native_elfs(['tf_fixture'], rank=0, tag='fault', evidence_dir=self.evidence)
        self.assertIs(caught.exception, primary)

    def test_child_context_is_scoped_and_restored_after_test_failure(self):
        primary = OSError('authentic test failure')
        def fail(argv, *, plugins):
            self.assertEqual(os.environ[driver.CHILD_EVIDENCE_ENV], str(self.evidence / 'child-native-ELF'))
            raise primary
        self.runtime.pytest.main = fail
        with self.assertRaises(OSError) as caught:
            self.execute()
        self.assertIs(caught.exception, primary)
        self.assertNotIn(driver.CHILD_EVIDENCE_ENV, os.environ)

    def test_environment_restore_failure_is_retained_without_losing_test_primary(self):
        primary, cleanup = OSError('authentic pytest primary'), RuntimeError('unsetenv failure')
        self.primary = primary
        try:
            with mock.patch.object(driver.os.environ, 'pop', side_effect=cleanup):
                with self.assertRaises(OSError) as caught:
                    self.execute()
            self.assertIs(caught.exception, primary)
            report = self.outcome()
            self.assertEqual(report['primary_error']['type'], 'OSError')
            self.assertIn('authentic pytest primary', report['primary_error']['error'])
            self.assertTrue(any('unsetenv failure' in error['error'] for error in report['post_errors']))
            self.assertIsNotNone(primary.__cause__)
        finally:
            os.environ.pop(driver.CHILD_EVIDENCE_ENV, None)

    def test_external_child_evidence_cannot_redirect_worker_records(self):
        with mock.patch.dict(os.environ, {driver.CHILD_EVIDENCE_ENV: '/unowned'}):
            with self.assertRaisesRegex(ValueError, 'external child-native'):
                self.execute()
        self.assertEqual(self.events, [])
        self.assertEqual(list(self.evidence.iterdir()), [])

    def test_app_mutation_cannot_be_hidden_by_pytest_zero_exit(self):
        self.mutate = lambda: (self.app / '__init__.py').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'post-suite source identity'):
            self.execute()
        self.assertEqual(self.outcome()['pytest_exit'], 0)
        self.assertFalse(self.outcome()['source_after_match'])


if __name__ == '__main__':
    unittest.main()
