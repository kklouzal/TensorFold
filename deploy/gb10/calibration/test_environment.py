"""Complete, separately sealed ModelOpt TEST application; original SDK unchanged.

Preparation runs in the old clean interpreter. The new venv has its own full
application site, with the old site absent, and exactly one effective pytest.
Runtime uses the unchanged native activation/guard chain, not a producer-role
version exception. This module never launches a GPU job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from email.parser import BytesParser
import importlib
import importlib.util
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from urllib.parse import unquote, urlsplit
import zipfile
import xml.etree.ElementTree as ET

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

ROLE_FILE = Path('/opt/tensorfold-calibration/tools/role.py')
PRODUCER = Path('/opt/tensorfold-calibration')
TEST_APP = Path('/opt/tensorfold-modelopt-test')
TEST_STATE = Path('/opt/tensorfold-modelopt-test-state')
ROOTS = frozenset(('pytest', 'coverage', 'pytest-cov', 'pytest-instafail', 'pytest-rerunfailures', 'pytest-timeout'))
REPLACE = frozenset(('pytest',))
TEST_FILE = 'tests/gpu/torch/quantization/test_tensor_quant_cuda.py'
KERNEL_FILE = 'modelopt/torch/kernels/quantization/gemm/fp4_kernel_hopper.py'
NODES = ('test_static_vs_dynamic_fp4_kernels', 'test_fp4_kernel_ragged_noncontiguous',
         'test_fp4_kernel_per_slice_global_amax')
EXPECTED_CASES = dict(zip(NODES, (27, 6, 6)))
ALLOCATOR_SOURCE = Path('/receipt/guarded-diagnostics/protected_allocator_v1.py')
ALLOCATOR_SHA = 'a354d3c1886cc2036136851c0e17d4017d2526ac4af3dd674b197858cd1ec945'
BOOTSTRAP = '/receipt/qualified_gpu_bootstrap_v3_guarded.py'
BOOTSTRAP_SHA = 'a5d5ac379f79f555a27d8fccb913b5a94263e959b09cf231d8ebfc399bf547d7'


def role():
    spec = importlib.util.spec_from_file_location('test_producer_contract', ROLE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def site(prefix):
    if sys.version_info[:2] != (3, 12):
        raise ValueError('test role requires the recorded CPython3.12 target')
    return Path(prefix)/'lib/python3.12/site-packages'


def inventory(directory):
    r = role()
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        relative = str(path.relative_to(directory))
        if path.is_symlink():
            result[relative] = {'link': os.readlink(path)}
        elif path.is_file():
            result[relative] = {'sha256': r.sha(path), 'mode': stat.S_IMODE(path.stat().st_mode)}
    return result


def old_state():
    r = role()
    state = r.read(Path(os.environ['TF_FOUNDATION_STATE']))
    return r, state


def audit_original():
    r, state = old_state()
    if Path(sys.prefix).resolve() != Path(state['application']).resolve():
        raise ValueError('original audit must run in the original clean interpreter')
    r.audit(PRODUCER/'prefix', r.read(PRODUCER/'manifest.json'))
    expected = r.read(PRODUCER/'seal.json')
    if expected != r.seal_inputs(PRODUCER/'prefix', PRODUCER/'manifest.json'):
        raise ValueError('original producer seal changed')


def snapshot(pyproject, producer_image, output):
    audit_original()
    r, state = old_state()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', producer_image):
        raise ValueError('immutable actual producer image required')
    config = tomllib.loads(pyproject.read_text())
    requirements = [raw for raw in config['project']['optional-dependencies']['dev-test']
                    if canonicalize_name(Requirement(raw).name) in ROOTS]
    if {canonicalize_name(Requirement(raw).name) for raw in requirements} != ROOTS:
        raise ValueError('focused upstream test requirements changed')
    old_app = Path(state['application'])
    baseline = {'schema_version': 1, 'producer_image': producer_image, 'environment': default_environment(),
                'packages': r.installed(), 'foundation_packages': r.installed([str(site(state['foundation']))]),
                'app_packages': r.installed([str(site(old_app))]), 'old_application': str(old_app),
                'old_app_tree': inventory(site(old_app)), 'foundation': state['foundation'],
                'producer_prefix': str(PRODUCER/'prefix'), 'producer_packages': r.installed([str(PRODUCER/'prefix')]),
                'producer_manifest_sha256': r.sha(PRODUCER/'manifest.json'),
                'producer_seal_sha256': r.sha(PRODUCER/'seal.json'), 'role_sha256': r.sha(ROLE_FILE),
                'requirements': requirements, 'source_files': {name: r.sha(pyproject.parent/name)
                    for name in ('pyproject.toml', 'tests/conftest.py', 'tests/gpu/conftest.py',
                                 'tests/gpu/torch/quantization/conftest.py', TEST_FILE, KERNEL_FILE)}}
    output.mkdir(exist_ok=False)
    r.write(output/'baseline.json', baseline)
    (output/'constraints.txt').write_text(''.join(f'{name}=={row["version"]}\n'
        for name, row in sorted(baseline['packages'].items()) if name not in REPLACE))
    (output/'requirements.in').write_text('\n'.join(requirements)+'\n')


def test_wheel(path):
    r = role()
    # All ordinary wheels retain the production admission policy. Only the
    # explicitly requested coverage startup hook has a distinct TEST contract.
    if canonicalize_name(parse_wheel_filename(path.name)[0]) != 'coverage':
        return {**r.wheel_record(path), 'startup_hooks': {}}
    _, _, _, tags = parse_wheel_filename(path.name)
    if not tags.intersection(sys_tags()):
        raise ValueError('coverage wheel does not match the actual native target')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        records = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(records) != 1:
            raise ValueError('coverage wheel metadata is ambiguous')
        metadata = BytesParser().parsebytes(archive.read(records[0]))
        if canonicalize_name(metadata['Name']) != 'coverage':
            raise ValueError('coverage artifact name differs')
        if metadata.get('Requires-Python') and not SpecifierSet(metadata['Requires-Python']).contains(default_environment()['python_full_version']):
            raise ValueError('coverage wheel does not support this Python version')
        hooks = {}
        for name in names:
            member = Path(name)
            if member.is_absolute() or '..' in member.parts or member.name in {'sitecustomize.py', 'usercustomize.py'}:
                raise ValueError('unsafe coverage wheel member')
            if re.match(r'lib(?:cuda|nvidia|cudart|cublas|cudnn|nvrtc|nvJitLink|nccl|cupti|stdc\+\+|gcc_s).*\.so', member.name):
                raise ValueError('coverage test wheel contains a protected native provider')
            if member.suffix == '.pth':
                if name != 'a1_coverage.pth':
                    raise ValueError('unreviewed test startup hook')
                data = archive.read(name)
                import hashlib
                hooks[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'text': data.decode('utf-8')}
    return {'name': 'coverage', 'version': metadata['Version'], 'requires': metadata.get_all('Requires-Dist', []),
            'artifact': path.name, 'sha256': r.sha(path), 'tags': sorted(map(str, tags)), 'startup_hooks': hooks}


def effective_packages(baseline, additions):
    return {name: row for name, row in baseline['packages'].items() if name not in REPLACE} | baseline['producer_packages'] | additions


def required_names(packages, requirements, environment):
    pending = [Requirement(raw) for raw in requirements]
    seen, names = set(), set()
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        for extra in ('', *sorted(requirement.extras)):
            if (name, extra) in seen:
                continue
            seen.add((name, extra))
            names.add(name)
            for raw in packages[name]['requires']:
                child = Requirement(raw)
                if child.marker is None or child.marker.evaluate({**environment, 'extra': extra}):
                    pending.append(child)
    return names


def prepare(baseline_path, report_path, wheels, output):
    r = role()
    baseline, report = r.read(baseline_path), r.read(report_path)
    if report['environment'] != baseline['environment']:
        raise ValueError('test dependency resolution must use the actual native baseline')
    additions = {}
    for item in report['install']:
        name = canonicalize_name(item['metadata']['name'])
        if (name in additions or name in baseline['foundation_packages']
                or name in baseline['producer_packages']
                or (name in baseline['packages'] and name not in REPLACE)):
            raise ValueError('test resolver changes an undeclared/protected dependency: '+name)
        info = item['download_info']
        url = urlsplit(info['url'])
        if url.scheme != 'https' or url.username or url.password:
            raise ValueError('test dependencies require genuine HTTPS wheel origins')
        path = wheels/Path(unquote(url.path)).name
        record = test_wheel(path)
        if (record['name'] != name or record['version'] != item['metadata']['version']
                or record['sha256'] != info['archive_info']['hashes']['sha256']):
            raise ValueError('test artifact differs from full native resolver report')
        additions[name] = record
    if not ROOTS <= additions.keys() or Version(additions['pytest']['version']) < Version('9.2'):
        raise ValueError('focused pytest9.2/plugin solution is incomplete')
    expected = effective_packages(baseline, additions)
    r.closure(expected | {'test-root-requirements': {'version': '1', 'requires': baseline['requirements']}}, baseline['environment'])
    if additions.keys() - required_names(expected, baseline['requirements'], baseline['environment']):
        raise ValueError('test report includes unrelated optional dependencies')
    output.mkdir(exist_ok=False)
    r.write(output/'manifest.json', {'schema_version': 1, 'scope': 'test-only-complete-app', 'baseline': baseline,
        'additions': additions, 'effective_packages': {name: {'version': row['version'], 'requires': row['requires']}
                                                    for name, row in expected.items()},
        'resolver_sha256': r.sha(report_path), 'tools_sha256': r.sha(Path(__file__))})
    (output/'requirements.lock').write_text(''.join(f'{name}=={row["version"]} --hash=sha256:{row["sha256"]}\n'
        for name, row in sorted(additions.items())))


def populate_test_app(manifest_path, wheels, manifest):
    r = role()
    baseline = manifest['baseline']
    old_site = site(baseline['old_application'])
    if inventory(old_site) != baseline['old_app_tree'] or r.installed() != baseline['packages']:
        raise ValueError('original application changed before test preparation')
    if TEST_APP.exists():
        raise FileExistsError('test prefix must be new; preserve previous outcomes')
    for record in manifest['additions'].values():
        if test_wheel(wheels/record['artifact']) != record:
            raise ValueError('test wheel/metadata/startup hook changed')
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(TEST_APP)], check=True)
    new_site = site(TEST_APP)
    shutil.copytree(old_site, new_site, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for distribution in r.unique_distributions([str(old_site)]):
        if canonicalize_name(distribution.metadata['Name']) in REPLACE:
            removed_roots = set()
            for member in distribution.files or []:
                if member.is_absolute() or '..' in member.parts:
                    continue  # Old bin scripts were never copied into the new venv.
                removed_roots.add(member.parts[0])
                destination = new_site/member
                if destination.is_file() or destination.is_symlink():
                    destination.unlink()
            for name in removed_roots:
                directory = new_site/name
                if directory.is_dir():
                    for child in sorted(directory.rglob('*'), reverse=True):
                        if not child.is_dir() or child.is_symlink():
                            raise ValueError('old framework contains unrecorded files')
                        child.rmdir()
                    directory.rmdir()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
        '--require-hashes', '--find-links', str(wheels), '--target', str(new_site),
        '--requirement', str(manifest_path.with_name('requirements.lock'))], check=True)
    catalog = r.installed([str(new_site)])
    expected_app = {name: row for name, row in baseline['app_packages'].items() if name not in REPLACE}
    expected_app |= {name: {'version': row['version'], 'requires': row['requires']} for name, row in manifest['additions'].items()}
    if catalog != expected_app:
        raise ValueError('test prefix is not the complete declared application closure')


def materialize(manifest_path, wheels):
    audit_original()
    r = role()
    manifest = r.read(manifest_path)
    primary, failures = None, []
    try:
        populate_test_app(manifest_path, wheels, manifest)
    except BaseException as error:
        primary = error
    for check in (audit_original, lambda: original_app_unchanged(manifest)):
        try:
            check()
        except BaseException as error:
            failures.append(error)
    if primary is not None:
        if failures:
            raise BaseExceptionGroup('test preparation and original audits failed', [primary, *failures]) from None
        raise primary
    if failures:
        raise BaseExceptionGroup('original audits after test preparation failed', failures)
    TEST_STATE.mkdir(exist_ok=False)
    shutil.copyfile(manifest_path, TEST_STATE/'manifest.json')
    r.write(TEST_STATE/'seal.json', {'schema_version': 1, 'manifest_sha256': r.sha(TEST_STATE/'manifest.json'),
        'test_app': str(TEST_APP), 'files': inventory(TEST_APP), 'tools_sha256': r.sha(Path(__file__))})


def original_app_unchanged(manifest):
    baseline = manifest['baseline']
    if inventory(site(baseline['old_application'])) != baseline['old_app_tree']:
        raise ValueError('original application bytes changed during test preparation')


def declared_namespace_catalogue(r, baseline):
    """Validate the public namespace before excluding its private vendor scan.

    Roots, not a runpy module-global snapshot, are the process authority. The
    SDK-owned Setuptools fallback is allowed only as its exact final append;
    public specs/loaded origins and all other metadata remain strictly checked.
    No actual sys.path entry is removed or changed here.
    """
    roots = [str(site(baseline['foundation'])), str(site(TEST_APP)), str(PRODUCER/'prefix')]
    contract = r.build_import_contract(roots)
    scan = sys.path.copy()
    setuptools = contract.get('setuptools')
    if setuptools is not None:
        vendor = str(setuptools['origin'].parent/'_vendor')
        positions = [index for index, value in enumerate(scan)
                     if Path(value or '.').resolve() == Path(vendor)]
        if positions:
            if positions != [len(scan)-1] or scan[-1] != vendor:
                raise ValueError('private Setuptools fallback changed public search order')
            scan.pop()
    if r.build_import_contract(scan) != contract:
        raise ValueError('public build-module resolution differs from declared test roots')
    r.check_import_context(scan, contract)
    declared = r.installed(roots)
    if r.installed(scan) != declared:
        raise ValueError('undeclared or shadowing distributions in the test namespace')
    return declared


def verify_runtime():
    r = role()
    manifest, seal = r.read(TEST_STATE/'manifest.json'), r.read(TEST_STATE/'seal.json')
    baseline = manifest['baseline']
    if (Path(sys.prefix).resolve() != TEST_APP.resolve() or seal['test_app'] != str(TEST_APP)
            or seal['tools_sha256'] != r.sha(Path(__file__)) or manifest['tools_sha256'] != r.sha(Path(__file__))
            or seal['manifest_sha256'] != r.sha(TEST_STATE/'manifest.json') or seal['files'] != inventory(TEST_APP)):
        raise ValueError('test interpreter/prefix/tool seal differs')
    if any(Path(path or '.').resolve() == site(baseline['old_application']).resolve() for path in sys.path):
        raise ValueError('original application site must not be active in the test process')
    original_modelopt = str(PRODUCER/'prefix')
    if original_modelopt not in sys.path:
        sys.path.append(original_modelopt)
    if declared_namespace_catalogue(r, baseline) != manifest['effective_packages']:
        raise ValueError('test-effective distribution namespace differs (including duplicate pytest)')
    r.provider_proof()
    if r.sha(PRODUCER/'manifest.json') != baseline['producer_manifest_sha256'] or r.sha(PRODUCER/'seal.json') != baseline['producer_seal_sha256']:
        raise ValueError('original producer identity changed')
    return manifest


def clean_original_audit(manifest):
    old_python = Path(manifest['baseline']['old_application'])/'bin/python'
    subprocess.run([str(old_python), str(Path(__file__).resolve()), 'audit-original'],
                   env={**os.environ, 'PYTHONPATH': ''}, check=True)


def startup_environment(environment, output, protected_roots):
    """Validate initial settings BEFORE coverage's new-interpreter .pth hook."""
    if environment.get('COVERAGE_PROCESS_CONFIG'):
        raise ValueError('initial COVERAGE_PROCESS_CONFIG is unsupported; use declared upstream START')
    output = output.resolve()
    if any(output.is_relative_to(Path(path).resolve()) for path in protected_roots):
        raise ValueError('test output cannot be inside an original/sealed prefix')
    expected = str(output/'.coverage')
    if environment.get('COVERAGE_FILE') not in (None, '', expected):
        raise ValueError('initial coverage data path differs from the declared test output')
    if environment.get('COVERAGE_PROCESS_START') and environment.get('COVERAGE_FILE') != expected:
        raise ValueError('coverage START requires its declared output data path before startup')
    result = dict(environment)
    result.pop('COVERAGE_PROCESS_CONFIG', None)
    result['COVERAGE_FILE'] = expected
    return result


def start(output, script, arguments):
    # This old interpreter has no newly installed coverage startup hook.
    audit_original()
    r, original = old_state()
    manifest, seal = r.read(TEST_STATE/'manifest.json'), r.read(TEST_STATE/'seal.json')
    if (seal['manifest_sha256'] != r.sha(TEST_STATE/'manifest.json')
            or seal['tools_sha256'] != r.sha(Path(__file__)) or manifest['tools_sha256'] != r.sha(Path(__file__))
            or seal['files'] != inventory(TEST_APP)):
        raise ValueError('test startup prefix/configuration seal changed')
    self_script = str(Path(__file__).resolve())
    if script == BOOTSTRAP:
        if r.sha(Path(script)) != BOOTSTRAP_SHA:
            raise ValueError('unchanged guarded bootstrap identity changed')
    elif script != self_script or arguments != ['cpu-smoke']:
        raise ValueError('test startup permits only the guarded bootstrap or CPU admission smoke')
    output = output.resolve()
    environment = startup_environment(os.environ, output,
        (original['application'], original['foundation'], PRODUCER, TEST_APP, TEST_STATE, Path(__file__).resolve().parent))
    config = environment.get('COVERAGE_PROCESS_START')
    if config:
        config_path = Path(config).resolve(strict=True)
        if r.sha(config_path) != manifest['baseline']['source_files']['pyproject.toml']:
            raise ValueError('coverage START config differs from frozen upstream pyproject')
        environment['COVERAGE_PROCESS_START'] = str(config_path)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('test output must be new with an existing parent')
    r.write(output.with_name(output.name+'.startup.json'), {
        'schema_version': 1, 'initial_coverage_process_config': None,
        'initial_coverage_process_start': os.environ.get('COVERAGE_PROCESS_START'),
        'effective_coverage_process_start': environment.get('COVERAGE_PROCESS_START'),
        'effective_coverage_file': environment['COVERAGE_FILE'],
        'test_manifest_sha256': r.sha(TEST_STATE/'manifest.json'), 'tools_sha256': r.sha(Path(__file__))})
    executable = str(TEST_APP/'bin/python')
    os.execve(executable, [executable, self_script, 'run', script, *arguments], environment)


def run(script, arguments):
    manifest = verify_runtime()
    clean_original_audit(manifest)
    selected_path = sys.path.copy()
    path_owner = sys.path
    from importlib.machinery import PathFinder
    origins = {name: Path(spec.origin).resolve() for name in
        ('torch', 'triton', 'transformers', 'pytest', '_pytest', 'coverage', 'pytest_cov',
         'pytest_instafail', 'pytest_rerunfailures', 'pytest_timeout', 'modelopt')
        if (spec := PathFinder.find_spec(name, selected_path)) is not None and spec.origin}
    primary, failures = None, []
    try:
        sys.argv = [script, *arguments]
        runpy.run_path(script, run_name='__main__')
    except BaseException as error:
        primary = error
    for name, expected in origins.items():
        try:
            module = sys.modules.get(name)
            if module is not None and Path(module.__file__).resolve() != expected:
                raise ValueError('test module origin changed: '+name)
        except BaseException as error:
            failures.append(error)
    try:
        path_owner[:] = selected_path
        sys.path = path_owner
    except BaseException as error:
        failures.append(error)
    for check in (lambda: verify_runtime(), lambda: clean_original_audit(manifest)):
        try:
            check()
        except BaseException as error:
            failures.append(error)
    if primary is not None:
        if failures:
            raise BaseExceptionGroup('test workload and final audits failed', [primary, *failures]) from None
        raise primary
    if failures:
        raise BaseExceptionGroup('test final audits failed', failures)


def prove_guarded_collection(torch):
    if os.environ.get('TF_FOUNDATION_RUNTIME_PHASE') != 'cuda' or not os.environ.get('TF_DIAG_GUARD_NONCE'):
        raise RuntimeError('GPU test collection requires the existing guarded CUDA bootstrap')
    r = role()
    r.provider_proof()
    if r.sha(ALLOCATOR_SOURCE) != ALLOCATOR_SHA:
        raise ValueError('frozen allocator admission helper changed')
    helper = runpy.run_path(str(ALLOCATOR_SOURCE))
    receipt = helper['read_receipt'](os.environ['TF_DIAG_GUARD_RECEIPT'])
    budgets = helper['check_live'](os.environ, receipt, time.monotonic_ns())
    total = int(torch.cuda.get_device_properties(0).total_memory)
    fraction = float(torch.cuda.memory.get_per_process_memory_fraction(0))
    reserved = int(torch.cuda.memory_reserved(0))
    if (not math.isfinite(fraction) or total <= 0 or not 0 < fraction <= budgets['cuda']/total
            or not 0 <= reserved <= budgets['cuda']):
        raise RuntimeError('existing Torch allocator quota is absent or exceeds the live guard budget')
    return {'guard_nonce': receipt['nonce'], 'cuda_budget': budgets['cuda'],
            'fraction': fraction, 'reserved_bytes': reserved, 'allocator_source_sha256': ALLOCATOR_SHA}


class NativeOutcomes:
    """Supported pytest reporting hooks only; no test/numerical behavior changes."""
    def __init__(self):
        self.collected = []
        self.reports = []
        self.collection_errors = []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append({'nodeid': report.nodeid, 'longrepr': report.longreprtext})

    def pytest_runtest_logreport(self, report):
        self.reports.append({'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome,
            'wasxfail': getattr(report, 'wasxfail', None),
            'longrepr': report.longreprtext if report.outcome != 'passed' else None})


def qualify_native_tests(collected, reports, collection_errors, junit_path, exit_code):
    errors = []
    prefix = TEST_FILE+'::Testfp4::'
    families = Counter()
    for nodeid in collected:
        if not nodeid.startswith(prefix):
            errors.append('collected a node outside the exact source-bound selectors')
            continue
        families[nodeid[len(prefix):].split('[', 1)[0]] += 1
    if (len(collected) != 39 or len(set(collected)) != 39 or dict(families) != EXPECTED_CASES):
        errors.append('required source-bound 27/6/6 case inventory is empty, missing, duplicated or changed')
    if exit_code != 0 or collection_errors:
        errors.append('native pytest exit/collection failed')
    phases = Counter()
    for report in reports:
        if report['nodeid'] not in collected:
            errors.append('runtime report does not belong to the collected inventory')
        phases[(report['nodeid'], report['when'])] += 1
        if report['when'] not in ('setup', 'call', 'teardown'):
            errors.append('unknown native runtime report phase')
        if report['outcome'] != 'passed' or report.get('wasxfail') is not None:
            errors.append('native skip/failure/rerun/xfail observed; retained in raw outcomes')
    if any(phases[(nodeid, phase)] != 1 for nodeid in collected for phase in ('setup', 'call', 'teardown')):
        errors.append('required cases did not each execute exactly one successful setup/call/teardown')
    try:
        if junit_path.stat().st_size > 64 << 20:
            raise ValueError('native JUnit report exceeds the foundation document bound')
        tree = ET.parse(junit_path)
        cases = list(tree.iter('testcase'))
        expected_names = Counter(nodeid.rsplit('::', 1)[-1] for nodeid in collected)
        if (len(cases) != 39 or Counter(case.get('name') for case in cases) != expected_names
                or any(not case.get('classname', '').endswith('.Testfp4') for case in cases)):
            errors.append('native JUnit case inventory differs from collection')
        if any(list(case.iter('failure')) or list(case.iter('error')) or list(case.iter('skipped')) for case in cases):
            errors.append('native JUnit contains failure/error/skip')
    except (OSError, ValueError, ET.ParseError) as error:
        errors.append('native JUnit unavailable/invalid: '+type(error).__name__+': '+str(error))
    return {'schema_version': 1, 'status': 'failed' if errors else 'passed', 'errors': errors,
            'pytest_exit_code': int(exit_code), 'collected_cases': len(collected),
            'expected_families': EXPECTED_CASES, 'collected_families': dict(families)}


def publish_native_results(r, output, source_hashes, outcomes, code, primary, failures):
    if primary is None and code is None:
        primary = RuntimeError('pytest returned no exit status')
    if primary is None and code:
        primary = SystemExit(int(code))
    status = int(code) if code is not None else None
    try:
        r.write(output/'pytest-native-outcomes.json', {'schema_version': 1,
            'source_sha256': source_hashes, 'selectors': list(NODES),
            'collected': outcomes.collected, 'reports': outcomes.reports,
            'collection_errors': outcomes.collection_errors, 'pytest_exit_code': status,
            'primary_exception': None if primary is None else {'type': type(primary).__name__, 'message': str(primary)}})
    except BaseException as error:
        failures.append(error)
    try:
        qualification = qualify_native_tests(outcomes.collected, outcomes.reports, outcomes.collection_errors,
                                            output/'junit.xml', status if status is not None else 3)
        if primary is not None or failures:
            qualification['status'] = 'failed'
            qualification['errors'].append('pytest exception/status or cleanup/publication failure occurred')
        r.write(output/'pytest-qualification.json', qualification)
    except BaseException as error:
        failures.append(error)
    if primary is not None:
        if failures:
            raise BaseExceptionGroup('pytest and final reporting failed', [primary, *failures]) from None
        raise primary
    if failures:
        raise BaseExceptionGroup('pytest final reporting failed', failures)
    if qualification['status'] != 'passed':
        raise RuntimeError('required Testfp4 cases did not all execute/pass; inspect native outcomes/JUnit')


def run_pytest(source, output):
    source, output = source.resolve(strict=True), output.resolve()
    if output.is_relative_to(source):
        raise ValueError('test reports cannot be written into the frozen source tree')
    manifest = verify_runtime()
    # The bootstrap already owns installation and heartbeat. This only reuses
    # its live admission checker and reads back the existing quota, never resets.
    torch = importlib.import_module('torch')
    admission = prove_guarded_collection(torch)
    if os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'):
        raise ValueError('upstream pytest plugins cannot be disabled')
    if os.environ.get('COVERAGE_PROCESS_START') and os.environ.get('COVERAGE_FILE') != str(output/'.coverage'):
        raise ValueError('coverage data path must be declared in the output directory before interpreter startup')
    r = role()
    for name, expected in manifest['baseline']['source_files'].items():
        if r.sha(source/name) != expected:
            raise ValueError('frozen upstream test/kernel/config source changed')
    # Import the installed fork before pytest's normal source-path handling.
    module = importlib.import_module('modelopt')
    if not Path(module.__file__).resolve().is_relative_to(PRODUCER/'prefix'):
        raise ValueError('GPU tests must exercise the installed producer fork')
    if r.sha(PRODUCER/'prefix'/KERNEL_FILE) != manifest['baseline']['source_files'][KERNEL_FILE]:
        raise ValueError('installed kernel differs from the frozen source under test')
    import pytest
    output.mkdir(exist_ok=False)
    outcomes = NativeOutcomes()
    previous = Path.cwd()
    code, primary, failures = None, None, []
    try:
        os.chdir(source)
        args = [f'{TEST_FILE}::Testfp4::{node}' for node in NODES]
        args += ['-o', 'cache_dir='+str(output/'.pytest_cache')]
        args += ['--junitxml='+str(output/'junit.xml')]
        if os.environ.get('COVERAGE_PROCESS_START'):
            args += ['--cov', '--cov-report=html:'+str(output/'htmlcov'), '--cov-report=xml:'+str(output/'coverage.xml')]
        code = pytest.main(args, plugins=[outcomes])
    except BaseException as error:
        primary = error
    try:
        os.chdir(previous)
    except BaseException as error:
        failures.append(error)
    publish_native_results(r, output, manifest['baseline']['source_files'], outcomes, code, primary, failures)
    print({'scope': 'three explicit Testfp4 nodes', 'guard_admission': admission})


def cpu_smoke():
    manifest = verify_runtime()
    if os.environ.get('TF_FOUNDATION_RUNTIME_PHASE') != 'cpu':
        raise RuntimeError('CPU test-environment smoke requires the managed no-device CPU phase')
    modules = {name: importlib.import_module(name) for name in
        ('torch', 'triton', 'transformers', 'pytest', 'coverage', 'pytest_cov',
         'pytest_instafail', 'pytest_rerunfailures', 'pytest_timeout')}
    if modules['torch'].cuda.is_initialized():
        raise RuntimeError('test-environment imports unexpectedly initialized CUDA')
    foundation = Path(manifest['baseline']['foundation']).resolve()
    for name in ('torch', 'triton'):
        if not Path(modules[name].__file__).resolve().is_relative_to(foundation):
            raise ValueError('test module shadows the protected foundation: '+name)
    role().provider_proof()
    print({'scope': 'CPU imports/provenance only; no GPU collection',
           'pytest': modules['pytest'].__version__, 'cuda_initialized': False,
           'origins': {name: str(module.__file__) for name, module in modules.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    snap = sub.add_parser('snapshot')
    snap.add_argument('--pyproject', type=Path, required=True)
    snap.add_argument('--producer-image', required=True)
    snap.add_argument('--output', type=Path, required=True)
    prepare_parser = sub.add_parser('prepare')
    for name in ('baseline', 'report', 'wheels', 'output'):
        prepare_parser.add_argument('--'+name, type=Path, required=True)
    material = sub.add_parser('materialize')
    material.add_argument('--manifest', type=Path, required=True)
    material.add_argument('--wheels', type=Path, required=True)
    sub.add_parser('audit-original')
    sub.add_parser('verify')
    sub.add_parser('cpu-smoke')
    execution = sub.add_parser('run')
    execution.add_argument('script')
    execution.add_argument('arguments', nargs=argparse.REMAINDER)
    startup = sub.add_parser('start')
    startup.add_argument('--output', type=Path, required=True)
    startup.add_argument('script')
    startup.add_argument('arguments', nargs=argparse.REMAINDER)
    test = sub.add_parser('pytest')
    test.add_argument('--source', type=Path, required=True)
    test.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'snapshot':
        snapshot(args.pyproject, args.producer_image, args.output)
    elif args.command == 'prepare':
        prepare(args.baseline, args.report, args.wheels, args.output)
    elif args.command == 'materialize':
        materialize(args.manifest, args.wheels)
    elif args.command == 'audit-original':
        audit_original()
    elif args.command == 'verify':
        verify_runtime()
    elif args.command == 'cpu-smoke':
        cpu_smoke()
    elif args.command == 'run':
        run(args.script, args.arguments)
    elif args.command == 'start':
        start(args.output, args.script, args.arguments)
    else:
        run_pytest(args.source, args.output)


if __name__ == '__main__':
    main()
