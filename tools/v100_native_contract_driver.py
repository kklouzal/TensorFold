"""Installed-package V100 contract checks; import is inert under spawn.

The host owns the container/image/cgroup/health and descendant retirement. This
inner driver owns declared identities, collection, pytest outcomes and evidence.
No module import reads a spec, creates evidence, imports Torch/pytest or queries
CUDA. Stage this file and its unchanged benchmark_output/worker_lifetime helpers
read-only; generated operator copies must retain the canonical bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import types
import xml.etree.ElementTree as ET

SCHEMA = 'tensorfold.v100-native-contract.v1'
CONTEXT_FATAL = ('illegal memory access', 'device-side assert triggered', 'unspecified launch failure')
OUTPUTS = ('installed-SDK-source-origin.json', 'collection-identities.json', 'first-failure.json',
           'named-suite.xml', 'sampled-allocation-process-cgroup.json', 'actual-native-ELF.json',
           'source-after.json', 'named-suite-result.json', 'phase.json')


def _sha(value, length=64):
    return type(value) is str and len(value) == length and all(c in '0123456789abcdef' for c in value)


def _relative(value):
    return (type(value) is str and bool(value) and '\x00' not in value
            and not PurePosixPath(value).is_absolute() and '..' not in PurePosixPath(value).parts
            and PurePosixPath(value).as_posix() == value and value != '.')


def _absolute(value):
    return type(value) is str and '\x00' not in value and Path(value).is_absolute() and '..' not in Path(value).parts


def _keys(value, required):
    return type(value) is dict and set(value) == set(required)


def _positive(value):
    return type(value) is int and 0 < value <= sys.maxsize


def normal_uuid(value):
    if type(value) is not str:
        raise ValueError('device UUID must be a string')
    normalized = value.lower().removeprefix('gpu-').replace('-', '')
    if not _sha(normalized, 32):
        raise ValueError('device UUID must name its exact 128-bit device identity')
    return normalized


def validate_spec(spec):
    fields = ('schema', 'mode', 'image_id', 'source_commit', 'app_root', 'expected_app', 'sdk_manifest',
              'fd', 'torch', 'UUIDs', 'SM', 'modules', 'test_files', 'diagnostics', 'required_native_elf_prefixes')
    if not _keys(spec, fields) or spec['schema'] != SCHEMA:
        raise ValueError('unsupported or incomplete native-contract spec schema')
    if spec['mode'] not in ('cpu', 'cuda'):
        raise ValueError('an explicit CPU or CUDA qualification route is required')
    if (not _sha(spec['source_commit'], 40) or type(spec['image_id']) is not str
            or not spec['image_id'].startswith('sha256:') or not _sha(spec['image_id'][7:])
            or not _absolute(spec['app_root'])):
        raise ValueError('exact image, source commit and installed app root are required')
    if (not _keys(spec['sdk_manifest'], ('path', 'sha256')) or not _absolute(spec['sdk_manifest']['path'])
            or not _sha(spec['sdk_manifest']['sha256'])
            or not _keys(spec['fd'], ('sha256', 'source_sha256')) or not all(map(_sha, spec['fd'].values()))
            or not _keys(spec['torch'], ('git_version', 'cuda')) or not _sha(spec['torch']['git_version'], 40)
            or type(spec['torch']['cuda']) is not str or not spec['torch']['cuda']):
        raise ValueError('complete SDK, native FD and Torch identities are required')
    if spec['mode'] == 'cpu':
        if spec['UUIDs'] != [] or spec['SM'] != [] or spec['required_native_elf_prefixes'] != []:
            raise ValueError('CPU-only qualification must not declare native GPU requirements')
    elif (type(spec['UUIDs']) is not list or not spec['UUIDs'] or type(spec['SM']) is not list
            or any(type(value) is not int for value in spec['SM']) or spec['SM'] != [7, 0]):
        raise ValueError('V100 qualification requires ordered UUIDs and SM70')
    normalized = [normal_uuid(value) for value in spec['UUIDs']]
    if len(set(normalized)) != len(normalized):
        raise ValueError('visible device UUIDs must be distinct')
    for label in ('expected_app', 'test_files'):
        mapping = spec[label]
        if type(mapping) is not dict or not mapping:
            raise ValueError('a complete nonempty file inventory is required: ' + label)
        for name, record in mapping.items():
            required = {'bytes', 'sha256', 'mode'} if label == 'expected_app' else {'bytes', 'sha256'}
            if (not _relative(name) or type(record) is not dict or not required <= record.keys()
                    or set(record).difference({'bytes', 'sha256', 'mode'})
                    or type(record['bytes']) is not int or not 0 <= record['bytes'] <= sys.maxsize
                    or not _sha(record['sha256']) or 'mode' in record and (
                        type(record['mode']) is not int or not 0 <= record['mode'] <= 0o777)):
                raise ValueError('invalid file identity: ' + str(name))
    modules = spec['modules']
    if (type(modules) is not list or not modules or any(type(name) is not str for name in modules)
            or len(set(modules)) != len(modules)
            or any(type(name) is not str or not _relative(name) or not name.startswith('tests/')
                   or not name.endswith('.py') or '::' in name or name not in spec['test_files'] for name in modules)):
        raise ValueError('named test modules must be distinct declared relative Python files')
    diagnostics = spec['diagnostics']
    if (not _keys(diagnostics, ('sample_limit', 'read_limit_bytes', 'failure_trace_limit_bytes'))
            or not all(_positive(value) for value in diagnostics.values()) or diagnostics['sample_limit'] > 4096):
        raise ValueError('explicit positive diagnostic bounds and at most 4096 samples are required')
    prefixes = spec['required_native_elf_prefixes']
    if (type(prefixes) is not list or any(type(p) is not str or not p.startswith('tf_')
            or not all(c.isascii() and (c.isalnum() or c == '_') for c in p) for p in prefixes)
            or len(set(prefixes)) != len(prefixes)):
        raise ValueError('required parent ELF prefixes must name distinct owned tf_ modules')
    return spec


def load_spec(path, *, expected_sha, expected_bytes):
    if not _sha(expected_sha) or not _positive(expected_bytes):
        raise ValueError('explicit spec SHA256 and positive byte extent are required')
    with Path(path).open('rb') as stream:
        raw = stream.read(expected_bytes + 1)
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('spec bytes do not match their declared immutable identity')
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate JSON spec field: ' + key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('nonfinite JSON spec constant: ' + value)
    return validate_spec(json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant))


def file_identity(path):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('identity requires an ordinary file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('file changed while its identity was read: ' + str(path))
    return {'bytes': after.st_size, 'sha256': digest.hexdigest(), 'mode': stat.S_IMODE(after.st_mode)}


def app_inventory(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        if path.is_symlink():
            raise ValueError('installed package inventory contains a symbolic link: ' + str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_identity(path)
    return result


def check_tests(source, expected):
    result = {}
    for name, identity in expected.items():
        (source / name).resolve().relative_to(source)
        actual = file_identity(source / name)
        if any(actual[key] != value for key, value in identity.items()):
            raise RuntimeError('test input differs from its declared identity: ' + name)
        result[name] = actual
    return result


def _error(stage, error):
    return {'stage': stage, 'type': type(error).__name__, 'error': repr(error)}


def _helpers():
    if __package__:
        from .benchmark_output import write_json
        from .worker_lifetime import raise_failures
    else:
        from benchmark_output import write_json
        from worker_lifetime import raise_failures
    return write_json, raise_failures


class Evidence:
    def __init__(self, root, writer):
        self.root, self.writer = Path(root), writer
        if (not self.root.is_dir() or self.root.is_symlink()
                or self.root.stat().st_uid != os.getuid() or self.root.stat().st_mode & 0o022):
            raise ValueError('evidence must be an existing ordinary owned directory')
        for name in OUTPUTS:
            if (self.root / name).exists() or (self.root / name).is_symlink():
                raise FileExistsError('evidence artifact already exists: ' + name)
        self.saved = set()

    def save(self, name, value):
        if name in self.saved or (self.root / name).exists() or name not in OUTPUTS:
            raise FileExistsError('evidence publication is not a fresh owned artifact: ' + name)
        self.writer(self.root / name, value)
        self.saved.add(name)

    def phase(self, name):
        self.writer(self.root / 'phase.json', {'phase': name, 'monotonic_seconds': time.monotonic()})


def _read_text(path, limit):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('diagnostic input exceeds its declared byte limit: ' + str(path))
    return raw.decode('utf-8')


class Guard:
    """Explicit parent-only context. Merely defining this class has no effects."""
    def __init__(self, spec, source, evidence, torch, pytest):
        self.spec, self.source, self.evidence, self.torch, self.pytest = spec, source, evidence, torch, pytest
        self.samples, self.diagnostic_errors, self.required_errors = [], [], []
        self.failed = self.sampling_failed = self.context_fatal = False
        self.session = None
        self.first_report = None
        self.collected = 0

    def sample(self, label):
        if self.sampling_failed or self.context_fatal:
            return
        try:
            bound = self.spec['diagnostics']
            if len(label.encode('utf-8')) > bound['read_limit_bytes']:
                raise ValueError('diagnostic label exceeds its declared byte limit')
            if len(self.samples) >= bound['sample_limit']:
                raise RuntimeError('diagnostic sample capacity exhausted')
            row = {'label': label, 'monotonic_seconds': time.monotonic(),
                   'process_status': _read_text('/proc/self/status', bound['read_limit_bytes'])}
            if self.spec['mode'] == 'cuda':
                cuda = self.torch.cuda
                row.update(device=cuda.current_device(), cuda_allocated=cuda.memory_allocated(),
                           cuda_reserved=cuda.memory_reserved(), cuda_peak_allocated=cuda.max_memory_allocated(),
                           cuda_peak_reserved=cuda.max_memory_reserved())
            else:
                row['cuda_observation'] = 'Not queried: explicit CPU-only route.'
            row['cgroup'] = {name: _read_text(Path('/sys/fs/cgroup') / name, bound['read_limit_bytes'])
                             for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.max', 'memory.stat')
                             if (Path('/sys/fs/cgroup') / name).is_file()}
            self.samples.append(row)
        except Exception as error:
            self.sampling_failed = True
            try:
                self.diagnostic_errors.append(_error(label, error))
            except BaseException as rendering:
                self.required_errors.extend((error, rendering))

    def pytest_collection_modifyitems(self, session, config, items):
        self.session = session
        if Path(config.rootpath).resolve() != self.source:
            raise RuntimeError('pytest root differs from the declared source root')
        modules = {}
        for item in items:
            module = item.getparent(self.pytest.Module)
            if module is None:
                raise RuntimeError('collected item has no owning pytest Module')
            name = Path(module.path).resolve().relative_to(self.source).as_posix()
            if module.nodeid != name:
                raise RuntimeError('pytest Module ID differs from its owning path: ' + name)
            modules[name] = str(module.path)
        if set(modules) != set(self.spec['modules']) or not items:
            raise RuntimeError('pytest collected a different named module set or no tests')
        self.collected = len(items)
        self.evidence.save('collection-identities.json', {'root': str(config.rootpath), 'count': len(items), 'modules': modules})

    def pytest_runtest_logreport(self, report):
        if report.failed:
            first = not self.failed
            if first:
                self.failed, self.first_report = True, report
            try:
                trace = str(report.longrepr)
            except BaseException as error:
                self.required_errors.append(error)
                if self.session is not None:
                    self.session.shouldstop = 'Failure report could not be preserved; remaining work unrun'
                return
            if first:
                try:
                    if len(trace.encode('utf-8')) > self.spec['diagnostics']['failure_trace_limit_bytes']:
                        raise ValueError('first failure trace exceeds its declared evidence limit; full pytest console remains owning evidence')
                    self.evidence.save('first-failure.json', {'nodeid': report.nodeid, 'when': report.when, 'trace': trace})
                except BaseException as error:
                    self.required_errors.append(error)
                try:
                    print('FIRST_OWNING_FAILURE', report.nodeid, report.when, flush=True)
                except BaseException as error:
                    self.required_errors.append(error)
            if self.spec['mode'] == 'cuda' and any(marker in trace.lower() for marker in CONTEXT_FATAL):
                self.context_fatal = True
                if self.session is None:
                    self.required_errors.append(RuntimeError('context failure has no owning pytest session'))
                else:
                    self.session.shouldstop = 'GPU context failure; remaining affected work unrun'
        self.sample(report.nodeid + ':' + report.when)


def _load_runtime():
    import tensorfold
    import tensorfold._fd_owner as fd
    import torch
    import pytest
    return types.SimpleNamespace(app=tensorfold, fd=fd, torch=torch, pytest=pytest)


def _installed_origins(root):
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'tensorfold' or name.startswith('tensorfold.'):
            path = module.__dict__.get('__file__') if isinstance(module, types.ModuleType) else None
            if path is not None:
                resolved = Path(path).resolve()
                resolved.relative_to(root)
                origins[name] = str(resolved)
    return origins


def _admit_runtime(spec, runtime):
    root = Path(runtime.app.__file__).resolve().parent
    if root != Path(spec['app_root']).resolve() or app_inventory(root) != spec['expected_app']:
        raise RuntimeError('installed application origin or complete file inventory differs')
    if file_identity(spec['sdk_manifest']['path'])['sha256'] != spec['sdk_manifest']['sha256']:
        raise RuntimeError('protected SDK manifest differs')
    if (file_identity(runtime.fd.__file__)['sha256'] != spec['fd']['sha256']
            or runtime.fd._source_sha256 != spec['fd']['source_sha256']):
        raise RuntimeError('native FD binary or C source identity differs')
    torch = runtime.torch
    if torch.version.git_version != spec['torch']['git_version'] or torch.version.cuda != spec['torch']['cuda']:
        raise RuntimeError('installed Torch revision or CUDA version differs')
    if spec['mode'] == 'cpu':
        return root, []
    if torch.cuda.device_count() != len(spec['UUIDs']):
        raise RuntimeError('visible device count differs')
    devices = [{'index': i, 'UUID': str(torch.cuda.get_device_properties(i).uuid),
                'SM': list(torch.cuda.get_device_capability(i))} for i in range(len(spec['UUIDs']))]
    if ([normal_uuid(device['UUID']) for device in devices] != [normal_uuid(value) for value in spec['UUIDs']]
            or any(device['SM'] != spec['SM'] for device in devices)):
        raise RuntimeError('visible device UUID order or SM differs')
    return root, devices


def _native_elfs(required):
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith('tf_') and isinstance(module, types.ModuleType):
            value = module.__dict__.get('__file__')
            if type(value) is str and value.endswith('.so'):
                path = Path(value).resolve()
                modules[name] = {'path': str(path), **file_identity(path)}
    missing = [prefix for prefix in required if not any(name.startswith(prefix) for name in modules)]
    return {'modules': modules, 'missing_required_parent_prefixes': missing,
            'scope': 'Already loaded parent-process modules only; no forced load or child execution inference.'}


def _junit(path, collected):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    if len(suites) != 1:
        raise RuntimeError('exactly one owning pytest suite is required')
    suite = suites[0].attrib
    for key in ('tests', 'errors', 'failures', 'skipped'):
        if not suite.get(key, '').isdecimal():
            raise ValueError('JUnit counter is absent or invalid: ' + key)
    if int(suite['tests']) != collected or collected <= 0:
        raise RuntimeError('JUnit test count differs from actual collection')
    return suite


def run(spec, source, evidence, *, runtime_loader=_load_runtime, helpers_loader=_helpers):
    """Execute once. Tests may inject dependencies; production has no bypass CLI."""
    validate_spec(spec)
    if os.environ.get('PYTEST_ADDOPTS'):
        raise ValueError('external PYTEST_ADDOPTS is forbidden for full-module qualification')
    source, evidence = Path(source).resolve(strict=True), Path(evidence).resolve(strict=True)
    if (not source.is_dir() or evidence.is_relative_to(source)
            or Path(spec['app_root']).resolve().is_relative_to(source)):
        raise ValueError('source must be a test tree disjoint from the installed app')
    write_json, raise_failures = helpers_loader()
    output = Evidence(evidence, write_json)
    primary, errors, diagnostic_errors = None, [], []
    runtime = guard = None
    code = suite = None
    source_after_match = False
    old_path = list(sys.path)
    try:
        if any(Path(value or os.getcwd()).resolve().is_relative_to(source / 'src') for value in sys.path):
            raise ValueError('application source overlay is forbidden')
        check_tests(source, spec['test_files'])
        sys.path[:0] = [str(source), str(source / 'tests'), str(source / 'tests/cuda')]
        runtime = runtime_loader()
        root, devices = _admit_runtime(spec, runtime)
        origins = _installed_origins(root)
        output.save('installed-SDK-source-origin.json', {'app_origin': str(root), 'runtime_members': spec['expected_app'],
            'source_commit': spec['source_commit'], 'image_id_declared': spec['image_id'],
            'Torch': runtime.torch.__version__, 'Torch_git': runtime.torch.version.git_version,
            'CUDA': runtime.torch.version.cuda, 'FD_SHA': spec['fd']['sha256'], 'FD_C': spec['fd']['source_sha256'],
            'devices_visible': devices, 'module_origins': origins,
            'mode': spec['mode'],
            'scope': ('CPU-only frontend/components; CUDA is not queried by this driver.' if spec['mode'] == 'cpu' else
                      'Image identity is host-bound; device visibility is not multirank execution proof.')})
        guard = Guard(spec, source, output, runtime.torch, runtime.pytest)
        output.phase('named-' + spec['mode'].upper() + '-tests')
        guard.sample('before-suite')
        # The host owns full raw logs. Keep worker stderr/faulthandler output
        # live instead of hiding it in pytest's global FD capture until return.
        code = int(runtime.pytest.main(['-c', '/dev/null', '--capture=no', '--rootdir=' + str(source), '--import-mode=importlib',
            '-o', 'pythonpath=' + ' '.join((str(source), str(source / 'tests'), str(source / 'tests/cuda'))),
            '-p', 'no:cacheprovider', '-vv', '-ra', '-o', 'junit_family=legacy',
            '--junitxml=' + str(evidence / 'named-suite.xml'), *[str(source / name) for name in spec['modules']]], plugins=[guard]))
    except BaseException as error:
        primary = error
    finally:
        if guard is not None:
            errors.extend(guard.required_errors)
            if spec['mode'] == 'cuda' and not guard.context_fatal:
                try:
                    runtime.torch.cuda.synchronize()
                except BaseException as error:
                    errors.append(error)
                    guard.context_fatal = True
            try:
                guard.sample('after-suite')
            except BaseException as error:
                errors.append(error)
            diagnostic_errors = list(guard.diagnostic_errors)
            try:
                output.save('sampled-allocation-process-cgroup.json', {'samples': guard.samples,
                    'diagnostic_errors': diagnostic_errors,
                    'scope': 'Bounded outer report envelopes/cached allocator peaks, not native allocation or physical bounds.'})
            except BaseException as error:
                errors.append(error)
        after = {}
        before_source_errors = len(errors)
        checks = {'tests': lambda: check_tests(source, spec['test_files']),
                  'app': lambda: app_inventory(Path(spec['app_root'])),
                  'SDK': lambda: file_identity(spec['sdk_manifest']['path']),
                  'nativeFD': lambda: file_identity(runtime.fd.__file__) if runtime is not None else None,
                  'module_origins': lambda: _installed_origins(Path(spec['app_root']).resolve())}
        for name, check in checks.items():
            try:
                value = check()
                if name == 'app' and value != spec['expected_app'] or name == 'SDK' and value['sha256'] != spec['sdk_manifest']['sha256']:
                    raise RuntimeError('post-suite source identity changed: ' + name)
                if name == 'nativeFD' and value is not None and value['sha256'] != spec['fd']['sha256']:
                    raise RuntimeError('post-suite native FD changed')
                after[name] = value
            except BaseException as error:
                errors.append(error)
        source_after_match = runtime is not None and len(errors) == before_source_errors
        try:
            output.save('source-after.json', after)
        except BaseException as error:
            errors.append(error)
        if guard is not None:
            try:
                native = _native_elfs(spec['required_native_elf_prefixes'])
                output.save('actual-native-ELF.json', native)
                if native['missing_required_parent_prefixes']:
                    raise RuntimeError('required parent native ELF was not actually loaded')
            except BaseException as error:
                errors.append(error)
            try:
                suite = _junit(evidence / 'named-suite.xml', guard.collected)
            except BaseException as error:
                errors.append(error)
        sys.path[:] = old_path
    result = None
    try:
        result = {'schema': 'tensorfold.native-contract-result-v1', 'mode': spec['mode'], 'pytest_exit': code, 'JUnit': suite,
                  'primary_error': None if primary is None else _error('execution', primary),
                  'post_errors': [_error('post-check', error) for error in errors], 'diagnostic_errors': diagnostic_errors,
                  'context_fatal': bool(guard is not None and guard.context_fatal),
                  'first_failure_observed': bool(guard is not None and guard.failed),
                  'source_after_match': source_after_match,
                  'source_commit': spec['source_commit'], 'image_id_declared': spec['image_id'], 'modules': spec['modules'],
                  'scope': 'Installed named components only; no full admission/model/quality/P3 qualification.'}
        output.save('named-suite-result.json', result)
        output.phase('named-suite-terminal')
    except BaseException as error:
        errors.append(error)
        # Bounded combined fallback: do not stringify the failed result or call
        # exception repr while handling a diagnostic/allocation failure.
        try:
            print('NATIVE_CONTRACT_EVIDENCE_FAILURE', 'pytest_exit=', code,
                  'primary_type=', None if primary is None else type(primary).__name__,
                  'post_error_types=', [type(value).__name__ for value in errors], file=sys.stderr, flush=True)
        except BaseException as fallback:
            errors.append(fallback)
    raise_failures(primary, errors)
    return int(code or bool(diagnostic_errors) or guard is not None and (guard.failed or guard.context_fatal)
               or suite is None or any(int(suite[name]) for name in ('errors', 'failures', 'skipped')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--spec-bytes', type=int, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = load_spec(args.spec, expected_sha=args.spec_sha256, expected_bytes=args.spec_bytes)
    return run(spec, args.source, args.evidence)


if __name__ == '__main__':
    raise SystemExit(main())
