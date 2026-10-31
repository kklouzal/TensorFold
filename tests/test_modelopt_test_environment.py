"""TEST-prefix boundaries only; no installs, venv creation, pytest9 or GPU jobs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile
import xml.etree.ElementTree as ET

import pytest
from packaging.markers import default_environment

spec = importlib.util.spec_from_file_location('test_app_contract',
    Path(__file__).resolve().parents[1]/'deploy/gb10/calibration/test_environment.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def basic_role():
    return SimpleNamespace(sha=checksum, read=lambda path: json.loads(Path(path).read_text()))


def test_selected_extra_dependency_graph_excludes_unused_optional_stacks():
    packages = {'pytest': {'requires': ['pluggy>=1']}, 'pluggy': {'requires': []},
        'coverage': {'requires': ['tomli; extra=="toml"']}, 'tomli': {'requires': []},
        'diffusers': {'requires': []}}
    assert T.required_names(packages, ['pytest>=9.2', 'coverage[toml]'], default_environment()) == {
        'pytest', 'pluggy', 'coverage', 'tomli'}


def test_new_baseline_declares_framework_replacement_without_mutating_original():
    old = {'packages': {'pytest': {'version': '8.4.2'}, 'transformers': {'version': '5.17.0'}},
           'producer_packages': {'nvidia-modelopt': {'version': 'fork'}}}
    result = T.effective_packages(old, {'pytest': {'version': '9.2.0'}})
    assert result['pytest']['version'] == '9.2.0'
    assert old['packages']['pytest']['version'] == '8.4.2'
    assert result['transformers']['version'] == '5.17.0'


@pytest.mark.parametrize('hook,python_requirement,admitted', [
    ('a1_coverage.pth', '>=3.11', True), ('unreviewed.pth', '>=3.11', False),
    ('a1_coverage.pth', '>=99', False)])
def test_test_only_startup_hook_is_exact_and_recorded(tmp_path, monkeypatch, hook, python_requirement, admitted):
    wheel = tmp_path/'coverage-7.17.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('coverage-7.17.0.dist-info/METADATA',
            f'Metadata-Version: 2.1\nName: coverage\nVersion: 7.17.0\nRequires-Python: {python_requirement}\n')
        archive.writestr(hook, '# published startup fixture\n')
    monkeypatch.setattr(T, 'role', basic_role)
    if admitted:
        record = T.test_wheel(wheel)
        assert set(record['startup_hooks']) == {'a1_coverage.pth'}
        assert record['sha256'] == checksum(wheel)
    else:
        with pytest.raises(ValueError):
            T.test_wheel(wheel)


def test_inventory_detects_payload_and_link_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(T, 'role', basic_role)
    (tmp_path/'file').write_text('first')
    (tmp_path/'link').symlink_to('file')
    before = T.inventory(tmp_path)
    (tmp_path/'file').write_text('second')
    assert T.inventory(tmp_path) != before
    assert before['link'] == {'link': 'file'}


def test_coverage_hook_exception_does_not_admit_dormant_cuda_provider(tmp_path, monkeypatch):
    wheel = tmp_path/'coverage-7.17.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('coverage-7.17.0.dist-info/METADATA',
            'Metadata-Version: 2.1\nName: coverage\nVersion: 7.17.0\n')
        archive.writestr('a1_coverage.pth', '# published hook fixture')
        archive.writestr('coverage.libs/libcudart.so.13', 'dormant replacement fixture')
    monkeypatch.setattr(T, 'role', basic_role)
    with pytest.raises(ValueError, match='protected native provider'):
        T.test_wheel(wheel)


def runtime_fixture(tmp_path, monkeypatch):
    app, state, producer = tmp_path/'app', tmp_path/'state', tmp_path/'producer'
    app.mkdir()
    state.mkdir()
    producer.mkdir()
    for name in ('manifest.json', 'seal.json'):
        (producer/name).write_text('{}')
    monkeypatch.setattr(T, 'TEST_APP', app)
    monkeypatch.setattr(T, 'TEST_STATE', state)
    monkeypatch.setattr(T, 'PRODUCER', producer)
    monkeypatch.setattr(T.sys, 'prefix', str(app))
    monkeypatch.setattr(T.sys, 'path', [str(tmp_path/'original-script')])
    r = basic_role()
    expected = {'pytest': {'version': '9.2.0', 'requires': []}}
    r.installed = lambda paths=None: expected
    r.build_import_contract = lambda paths: {}
    r.check_import_context = lambda paths, contract: None
    r.provider_proof = lambda: None
    monkeypatch.setattr(T, 'role', lambda: r)
    manifest = {'baseline': {'old_application': str(tmp_path/'old'), 'foundation': str(tmp_path/'foundation'),
        'producer_manifest_sha256': checksum(producer/'manifest.json'),
        'producer_seal_sha256': checksum(producer/'seal.json')},
        'effective_packages': expected, 'tools_sha256': checksum(T.__file__)}
    (state/'manifest.json').write_text(json.dumps(manifest))
    (state/'seal.json').write_text(json.dumps({'test_app': str(app), 'tools_sha256': checksum(T.__file__),
        'manifest_sha256': checksum(state/'manifest.json'), 'files': T.inventory(app)}))
    return manifest, r


def test_runtime_has_one_explicit_new_namespace_and_rejects_old_site(tmp_path, monkeypatch):
    manifest, _ = runtime_fixture(tmp_path, monkeypatch)
    assert T.verify_runtime() == manifest
    T.sys.path.append(str(T.site(manifest['baseline']['old_application'])))
    with pytest.raises(ValueError, match='original application site'):
        T.verify_runtime()


def test_runtime_rejects_framework_or_sealed_file_drift(tmp_path, monkeypatch):
    _, r = runtime_fixture(tmp_path, monkeypatch)
    r.installed = lambda paths=None: {'pytest': {'version': '8.4.2', 'requires': []}}
    with pytest.raises(ValueError, match='distribution namespace'):
        T.verify_runtime()
    (T.TEST_APP/'unexpected').write_text('new')
    with pytest.raises(ValueError, match='seal differs'):
        T.verify_runtime()


def test_failed_preparation_independently_audits_originals_and_never_seals(tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setattr(T, 'role', basic_role)
    monkeypatch.setattr(T, 'TEST_STATE', tmp_path/'unpublished')
    calls = []
    monkeypatch.setattr(T, 'audit_original', lambda: calls.append('audit'))
    def populate(*args):
        raise RuntimeError('install failure')
    def unchanged(*args):
        calls.append('bytes')
        raise ValueError('original bytes changed')
    monkeypatch.setattr(T, 'populate_test_app', populate)
    monkeypatch.setattr(T, 'original_app_unchanged', unchanged)
    with pytest.raises(ExceptionGroup) as error:
        T.materialize(manifest, tmp_path)
    assert str(error.value.exceptions[0]) == 'install failure'
    assert calls == ['audit', 'audit', 'bytes']
    assert not T.TEST_STATE.exists()


@pytest.mark.parametrize('fraction,live,admitted', [(.5, True, True), (1.0, True, False), (.5, False, False)])
def test_gpu_collection_requires_live_guard_and_existing_quota_without_install(monkeypatch, fraction, live, admitted):
    calls = []
    r = SimpleNamespace(sha=lambda path: T.ALLOCATOR_SHA, provider_proof=lambda: calls.append('native-proof'))
    monkeypatch.setattr(T, 'role', lambda: r)
    monkeypatch.setenv('TF_FOUNDATION_RUNTIME_PHASE', 'cuda')
    monkeypatch.setenv('TF_DIAG_GUARD_NONCE', 'a'*32)
    monkeypatch.setenv('TF_DIAG_GUARD_RECEIPT', '/receipt/guard/test.json')
    def check(*args):
        calls.append('live-check')
        if not live:
            raise RuntimeError('stale guard')
        return {'cuda': 2**31}
    monkeypatch.setattr(T.runpy, 'run_path', lambda path: {
        'read_receipt': lambda path: {'nonce': 'a'*32}, 'check_live': check})
    torch = SimpleNamespace(cuda=SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(total_memory=2**32),
        memory=SimpleNamespace(get_per_process_memory_fraction=lambda device: fraction),
        memory_reserved=lambda device: 0))
    if admitted:
        result = T.prove_guarded_collection(torch)
        assert result['cuda_budget'] == 2**31
    else:
        with pytest.raises(RuntimeError):
            T.prove_guarded_collection(torch)
    assert calls == ['native-proof', 'live-check']


def test_test_process_preserves_primary_and_runs_both_final_audits(monkeypatch):
    calls = []
    manifest = {'baseline': {'old_application': '/old'}}
    monkeypatch.setattr(T.sys, 'path', T.sys.path.copy())
    monkeypatch.setattr(T.sys, 'argv', T.sys.argv.copy())
    original_path = T.sys.path.copy()
    def verify():
        calls.append('test-audit')
        if calls.count('test-audit') > 1:
            raise ValueError('test seal failed')
        return manifest
    def old_audit(value):
        calls.append('original-audit')
        if calls.count('original-audit') > 1:
            raise ValueError('original seal failed')
    def workload(*args, **kwargs):
        T.sys.path.append('/temporary-pytest-path')
        raise RuntimeError('workload failed')
    monkeypatch.setattr(T, 'verify_runtime', verify)
    monkeypatch.setattr(T, 'clean_original_audit', old_audit)
    monkeypatch.setattr(T.runpy, 'run_path', workload)
    with pytest.raises(ExceptionGroup) as error:
        T.run('owned-test.py', [])
    assert str(error.value.exceptions[0]) == 'workload failed'
    assert calls == ['test-audit', 'original-audit', 'test-audit', 'original-audit']
    assert T.sys.path == original_path


def native_results(tmp_path):
    collected, reports = [], []
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite', tests='39', failures='0', skipped='0', errors='0')
    for family, count in T.EXPECTED_CASES.items():
        for index in range(count):
            name = f'{family}[{index}]'
            nodeid = f'{T.TEST_FILE}::Testfp4::{name}'
            collected.append(nodeid)
            for phase in ('setup', 'call', 'teardown'):
                reports.append({'nodeid': nodeid, 'when': phase, 'outcome': 'passed', 'wasxfail': None})
            ET.SubElement(suite, 'testcase', classname='source.Testfp4', name=name)
    path = tmp_path/'junit.xml'
    ET.ElementTree(root).write(path)
    return collected, reports, path


def test_native_39_pass_gate_reconciles_all_three_families_and_junit(tmp_path):
    collected, reports, path = native_results(tmp_path)
    assert T.qualify_native_tests(collected, reports, [], path, 0)['status'] == 'passed'


@pytest.mark.parametrize('fault', ['empty', 'missing', 'duplicate', 'skip', 'failure', 'rerun',
                                   'setup_skip', 'wrong_family', 'missing_xml', 'invalid_xml',
                                   'xml_failure', 'xml_error', 'xml_skip', 'xpass_empty_reason'])
def test_native_gate_rejects_incomplete_or_unsuccessful_execution(tmp_path, fault):
    collected, reports, path = native_results(tmp_path)
    if fault == 'empty':
        collected, reports = [], []
    elif fault == 'missing':
        collected.pop()
    elif fault == 'duplicate':
        collected[-1] = collected[0]
    elif fault in ('skip', 'failure'):
        reports[1]['outcome'] = 'skipped' if fault == 'skip' else 'failed'
    elif fault == 'rerun':
        reports.insert(0, {**reports[1], 'outcome': 'rerun', 'longrepr': 'initial failure retained'})
    elif fault == 'setup_skip':
        reports[0]['outcome'] = 'skipped'
        reports.pop(1)
    elif fault == 'wrong_family':
        collected[0] = collected[0].replace(T.NODES[0], 'unselected_method')
    elif fault == 'xpass_empty_reason':
        reports[1]['wasxfail'] = ''
    elif fault == 'missing_xml':
        path.unlink()
    elif fault.startswith('xml_'):
        tree = ET.parse(path)
        ET.SubElement(next(tree.iter('testcase')), {'xml_failure': 'failure', 'xml_error': 'error', 'xml_skip': 'skipped'}[fault])
        tree.write(path)
    else:
        path.write_text('not XML')
    assert T.qualify_native_tests(collected, reports, [], path, 0)['status'] == 'failed'


def test_native_observer_keeps_initial_rerun_failure_and_collection_errors():
    observer = T.NativeOutcomes()
    observer.pytest_collection_finish(SimpleNamespace(items=[SimpleNamespace(nodeid='case')]))
    observer.pytest_runtest_logreport(SimpleNamespace(nodeid='case', when='call', outcome='rerun',
        longreprtext='initial native failure'))
    observer.pytest_runtest_logreport(SimpleNamespace(nodeid='case', when='call', outcome='passed'))
    observer.pytest_collectreport(SimpleNamespace(failed=True, nodeid='module', longreprtext='collection error'))
    assert observer.reports[0]['longrepr'] == 'initial native failure'
    assert observer.reports[1]['outcome'] == 'passed'
    assert observer.collection_errors[0]['longrepr'] == 'collection error'


@pytest.mark.parametrize('changes,admitted', [({}, True), ({'COVERAGE_PROCESS_CONFIG': '{}'}, False),
    ({'COVERAGE_FILE': '/old/.coverage'}, False), ({'COVERAGE_PROCESS_START': '/source/pyproject.toml'}, False)])
def test_startup_coverage_contract_is_checked_before_new_interpreter(tmp_path, changes, admitted):
    output = tmp_path/'reports'
    environment = changes
    if admitted:
        result = T.startup_environment(environment, output, [tmp_path/'old'])
        assert result['COVERAGE_FILE'] == str(output/'.coverage')
        assert 'COVERAGE_PROCESS_CONFIG' not in result
    else:
        with pytest.raises(ValueError):
            T.startup_environment(environment, output, [tmp_path/'old'])


def test_startup_accepts_declared_start_route_but_not_protected_prefix(tmp_path):
    output = tmp_path/'reports'
    data = {'COVERAGE_PROCESS_START': '/source/pyproject.toml', 'COVERAGE_FILE': str(output/'.coverage')}
    assert T.startup_environment(data, output, [tmp_path/'old'])['COVERAGE_PROCESS_START'] == '/source/pyproject.toml'
    with pytest.raises(ValueError, match='sealed prefix'):
        T.startup_environment({}, tmp_path/'old/reports', [tmp_path/'old'])


def test_start_refuses_initial_config_before_fixed_new_python_exec(tmp_path, monkeypatch):
    _, r = runtime_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(T, 'audit_original', lambda: None)
    monkeypatch.setattr(T, 'old_state', lambda: (r, {'application': str(tmp_path/'old'), 'foundation': str(tmp_path/'foundation')}))
    monkeypatch.setenv('COVERAGE_PROCESS_CONFIG', '{}')
    monkeypatch.setattr(T.os, 'execve', lambda *args: pytest.fail('new interpreter started before startup validation'))
    with pytest.raises(ValueError, match='initial COVERAGE_PROCESS_CONFIG'):
        T.start(tmp_path/'output', str(Path(T.__file__).resolve()), ['cpu-smoke'])


def test_start_execs_only_fixed_test_python_after_seal_and_environment_checks(tmp_path, monkeypatch):
    _, r = runtime_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(T, 'audit_original', lambda: None)
    monkeypatch.setattr(T, 'old_state', lambda: (r, {'application': str(tmp_path/'old'), 'foundation': str(tmp_path/'foundation')}))
    for name in ('COVERAGE_PROCESS_START', 'COVERAGE_PROCESS_CONFIG', 'COVERAGE_FILE'):
        monkeypatch.delenv(name, raising=False)
    r.write = lambda path, value: Path(path).write_text(json.dumps(value))
    called = []
    class Replaced(Exception):
        pass
    def execute(executable, argv, environment):
        called.append((executable, argv, environment))
        raise Replaced()
    monkeypatch.setattr(T.os, 'execve', execute)
    script = str(Path(T.__file__).resolve())
    output = tmp_path/'output'
    with pytest.raises(Replaced):
        T.start(output, script, ['cpu-smoke'])
    executable, argv, environment = called[0]
    assert executable == str(T.TEST_APP/'bin/python')
    assert argv == [executable, script, 'run', script, 'cpu-smoke']
    assert environment['COVERAGE_FILE'] == str(output/'.coverage')
    assert 'COVERAGE_PROCESS_CONFIG' not in environment
    assert (tmp_path/'output.startup.json').is_file()


def reporting_fixture(tmp_path):
    collected, reports, _ = native_results(tmp_path)
    outcomes = T.NativeOutcomes()
    outcomes.collected, outcomes.reports = collected, reports
    def write(path, value):
        with path.open('x') as stream:
            json.dump(value, stream)
    return outcomes, SimpleNamespace(write=write)


def test_plugin_exception_still_publishes_both_owned_native_records(tmp_path):
    outcomes, r = reporting_fixture(tmp_path)
    primary = RuntimeError('unexpected plugin exception')
    with pytest.raises(RuntimeError) as error:
        T.publish_native_results(r, tmp_path, {}, outcomes, None, primary, [])
    assert error.value is primary
    raw = json.loads((tmp_path/'pytest-native-outcomes.json').read_text())
    assert len(raw['reports']) == 117
    assert raw['primary_exception']['type'] == 'RuntimeError'
    assert json.loads((tmp_path/'pytest-qualification.json').read_text())['status'] == 'failed'


def test_nonzero_pytest_status_survives_publication_error_and_other_write_attempt(tmp_path):
    outcomes, ordinary = reporting_fixture(tmp_path)
    calls = []
    def write(path, value):
        calls.append(path.name)
        if path.name == 'pytest-native-outcomes.json':
            raise OSError('raw publication failed')
        ordinary.write(path, value)
    with pytest.raises(BaseExceptionGroup) as error:
        T.publish_native_results(SimpleNamespace(write=write), tmp_path, {}, outcomes, 1, None, [])
    assert isinstance(error.value.exceptions[0], SystemExit)
    assert error.value.exceptions[0].code == 1
    assert calls == ['pytest-native-outcomes.json', 'pytest-qualification.json']
    assert json.loads((tmp_path/'pytest-qualification.json').read_text())['status'] == 'failed'


def test_primary_and_both_publication_failures_are_retained(tmp_path):
    outcomes, _ = reporting_fixture(tmp_path)
    calls = []
    def write(path, value):
        calls.append(path.name)
        raise OSError('write failed: '+path.name)
    primary = RuntimeError('primary')
    with pytest.raises(ExceptionGroup) as error:
        T.publish_native_results(SimpleNamespace(write=write), tmp_path, {}, outcomes, None, primary, [])
    assert error.value.exceptions[0] is primary
    assert len(error.value.exceptions) == 3
    assert len(calls) == 2


def test_real_setuptools_fallback_is_private_and_repeated_tool_loads_need_no_global_handoff(tmp_path):
    source = Path(T.__file__).resolve()
    role_source = source.with_name('role.py')
    control = tmp_path/'namespace.py'
    control.write_text(f'''import importlib.util,json,sys
from pathlib import Path
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
role=load('actual_role',{str(role_source)!r})
first=load('first_tool',{str(source)!r});second=load('second_tool',{str(source)!r})
prefix=Path({str(tmp_path / 'producer')!r});(prefix/'prefix').mkdir(parents=True)
for tool in (first,second):
 tool.TEST_APP=Path(sys.prefix);tool.PRODUCER=prefix
baseline={{'foundation':'/opt/tensorfold-foundation'}}
sys.path.append(str(prefix/'prefix'))
before=first.declared_namespace_catalogue(role,baseline)
import setuptools.build_meta
import wheel
path_before=sys.path.copy()
assert second.declared_namespace_catalogue(role,baseline)==before
assert sys.path==path_before  # Validation never drops actual search paths.
assert wheel.__version__=='0.47.0'
try:role.installed()
except ValueError as error:assert 'duplicate active distribution: wheel' in str(error)
else:raise AssertionError('old strict role was weakened')
original=wheel.__file__
wheel.__file__=str(Path(setuptools.__file__).parent/'_vendor/wheel/__init__.py')
try:
 try:second.declared_namespace_catalogue(role,baseline)
 except ValueError as error:assert 'selected origin' in str(error)
 else:raise AssertionError('public wheel vendor shadow was accepted')
finally:wheel.__file__=original
vendor=sys.path.pop();sys.path.insert(0,vendor)
try:
 try:second.declared_namespace_catalogue(role,baseline)
 except ValueError as error:assert 'search order' in str(error)
 else:raise AssertionError('vendor prepend was accepted')
finally:sys.path.pop(0);sys.path.append(vendor)
print(json.dumps({{'scope':'real SDK metadata only','wheel':wheel.__version__,
 'wheel_file':wheel.__file__,'private_vendor_path':vendor,'declared_catalogue_count':len(before)}}))
''')
    result = subprocess.run([T.sys.executable, str(control)], check=True, capture_output=True, text=True)
    assert 'real SDK metadata only' in result.stdout
    print(result.stdout.strip())
