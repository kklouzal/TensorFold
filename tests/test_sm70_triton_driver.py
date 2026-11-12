"""Owned startup locking and mapped-helper identity, without a GPU substitute."""
import fcntl
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from tensorfold.cuda.sm70 import triton_driver


def _install(monkeypatch, root, *, arch=70, during=None, version='3.9.0'):
    # This fixture models the legacy default profile. Explicit3.2 cases select
    # their own profile below; the container's real SDK must not choose it.
    monkeypatch.delenv(triton_driver.PROFILE_ENV, raising=False)
    module = SimpleNamespace(__file__=str(root / 'key/cuda_utils.test.so'))
    utils = SimpleNamespace(load_binary=SimpleNamespace(__self__=module))
    target = SimpleNamespace(backend='cuda', arch=arch, warp_size=32)
    def get_target():
        if during:
            during()
        return target
    active = SimpleNamespace(utils=utils, get_current_target=get_target)
    knobs = SimpleNamespace(cache=SimpleNamespace(dir=str(root)))
    monkeypatch.setenv('TRITON_CACHE_DIR', str(root))
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(knobs=knobs, __version__=version))
    original_version = triton_driver.importlib.metadata.version
    monkeypatch.setattr(triton_driver.importlib.metadata, 'version',
                        lambda name: version if name == 'triton' else original_version(name))
    monkeypatch.setitem(sys.modules, 'triton.runtime', SimpleNamespace(driver=SimpleNamespace(active=active)))
    return active, knobs


def _install32(monkeypatch, root, **kwargs):
    active, _ = _install(monkeypatch, root, version='3.2.0', **kwargs)
    # The actual3.2 distribution has no Knobs API. A missing field must never
    # accidentally choose the legacy branch or an unverified cache fallback.
    del sys.modules['triton'].knobs
    class FileCacheManager:
        def __init__(self, key):
            self.cache_dir = str(root / key)
    def get_manager(key):
        assert len(key) == 64
        bytes.fromhex(key)  # The real3.2 API accepts a digest, not an arbitrary cache label.
        return FileCacheManager(key)
    cache = SimpleNamespace(FileCacheManager=FileCacheManager, __cache_cls=FileCacheManager,
                            get_cache_manager=get_manager)
    sys.modules['triton.runtime'].cache = cache
    monkeypatch.setenv(triton_driver.PROFILE_ENV, triton_driver.PROFILE32)
    manifest = root.parent / 'runtime.json'
    manifest.write_text(json.dumps({'schema': 'tensorfold.v100-source-runtime', 'version': 3,
        'runtime_profile': {'schema': 'tensorfold.v100-runtime-profile', 'version': 1,
            'name': triton_driver.PROFILE32, 'changed_distributions': ['triton']},
        'packages': {'triton': {'version': '3.2.0'}}}))
    monkeypatch.setattr(triton_driver, '_RUNTIME_MANIFEST', manifest)
    return active, cache, manifest


@pytest.mark.parametrize('timeout', [None, True, 0, -1, float('nan'), float('inf')])
def test_invalid_timeout_is_refused_before_cache_or_runtime_work(timeout, monkeypatch):
    monkeypatch.delenv('TRITON_CACHE_DIR', raising=False)
    with pytest.raises(ValueError, match='startup timeout'):
        triton_driver.prepare_driver(timeout_s=timeout)


def test_actual_driver_preparation_holds_shared_lock_and_preserves_cache(monkeypatch, tmp_path):
    root = tmp_path / 'cache'
    observed = []
    def during():
        fd = os.open(root / '.tensorfold-driver.lock', os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            observed.append('locked')
        finally:
            os.close(fd)
    active, knobs = _install(monkeypatch, root, during=during)
    monkeypatch.setattr(triton_driver, '_helper_identity', lambda utils, path: {
        'path': str(path / 'key/cuda_utils.test.so'), 'bytes': 1, 'sha256': 'a'*64})
    result = triton_driver.prepare_driver(timeout_s=1)
    assert observed == ['locked']
    assert result['backend'] == 'cuda' and result['arch'] == 70 and result['warp_size'] == 32
    assert result['cache'] == knobs.cache.dir == os.environ['TRITON_CACHE_DIR'] == str(root)
    assert result['scope'] == 'startup-only-serialized-driver-preparation'
    with triton_driver._cache_lock(root, .1):
        pass  # The driver's preparation has positively released the lock.


def test_existing_primary_survives_preparation_and_lock_is_released(monkeypatch, tmp_path):
    primary = RuntimeError('helper refused')
    root = tmp_path / 'cache'
    _install(monkeypatch, root)
    def fail(*args):
        raise primary
    monkeypatch.setattr(triton_driver, '_helper_identity', fail)
    with pytest.raises(RuntimeError) as found:
        triton_driver.prepare_driver(timeout_s=1)
    assert found.value is primary
    with triton_driver._cache_lock(root, .1):
        pass


def test_lock_timeout_preserves_other_owner(tmp_path):
    root = tmp_path / 'cache'
    root.mkdir()
    lock = root / '.tensorfold-driver.lock'
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TimeoutError, match='lock timed out'):
            with triton_driver._cache_lock(root, .01):
                pytest.fail('entered another owner lock')
        other = os.open(lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(other)
    finally:
        os.close(fd)


@pytest.mark.parametrize('change', ['target', 'cache-before', 'cache-after'])
def test_target_and_cache_changes_fail_explicitly(monkeypatch, tmp_path, change):
    root = tmp_path / 'cache'
    def during():
        if change == 'cache-after':
            os.environ['TRITON_CACHE_DIR'] = str(tmp_path / 'other')
    _, knobs = _install(monkeypatch, root, arch=80 if change == 'target' else 70, during=during)
    if change == 'cache-before':
        knobs.cache.dir = str(tmp_path / 'other')
    monkeypatch.setattr(triton_driver, '_helper_identity', lambda *a: {})
    with pytest.raises(ValueError, match='cache|SM70'):
        triton_driver.prepare_driver(timeout_s=1)


def _maps(path, info, *, inode=None, deleted=False, executable=True):
    device = f'{os.major(info.st_dev):x}:{os.minor(info.st_dev):x}'
    return f'1000-2000 {"r-xp" if executable else "r--p"} 00000000 {device} {info.st_ino if inode is None else inode} {path}{" (deleted)" if deleted else ""}\n'


@pytest.mark.parametrize('fault', [None, 'deleted', 'inode', 'nonexec', 'outside'])
def test_helper_checks_actual_persistent_inode_and_keeps_deleted_guard(monkeypatch, tmp_path, fault):
    root = tmp_path / 'cache'
    path = (tmp_path if fault == 'outside' else root) / 'key/cuda_utils.test.so'
    path.parent.mkdir(parents=True)
    data = b'independent mapped-file identity fixture'
    path.write_bytes(data)
    info = path.stat()
    utils = SimpleNamespace(load_binary=SimpleNamespace(__self__=SimpleNamespace(__file__=str(path))))
    line = _maps(path, info, inode=info.st_ino+1 if fault == 'inode' else None,
                 deleted=fault == 'deleted', executable=fault != 'nonexec')
    monkeypatch.setattr(triton_driver, '_backing_identity', lambda fd, path: (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino))
    original_open = open
    from io import StringIO
    def opened(name, *args, **kwargs):
        return StringIO(line) if name == '/proc/self/maps' else original_open(name, *args, **kwargs)
    monkeypatch.setattr('builtins.open', opened)
    if fault:
        with pytest.raises(ValueError):
            triton_driver._helper_identity(utils, root)
    else:
        assert triton_driver._helper_identity(utils, root) == {
            'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def test_held_fd_backing_probe_releases_its_private_mapping(tmp_path):
    path = tmp_path / 'cuda_utils.test.so'
    original = b'held-file backing identity'
    path.write_bytes(original)
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        first = triton_driver._backing_identity(fd, path)
        second = triton_driver._backing_identity(fd, path)
        assert first == second and len(first) == 3 and first[2] > 0
        assert os.read(fd, len(original)) == original  # mmap did not consume the FD cursor.
    finally:
        os.close(fd)
    assert path.read_bytes() == original  # The private writable view was never written.


def test_profile32_compilation_and_launcher_preparation_share_driver_lock(monkeypatch, tmp_path):
    root = tmp_path / 'cache'
    _install32(monkeypatch, root)
    calls = []
    monkeypatch.setattr(triton_driver, '_helper_identity', lambda *args: {'driver': 'mapped'})
    def prepare(path, geometry):
        fd = os.open(root / '.tensorfold-driver.lock', os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        calls.append((path, geometry))
        return {'launcher': 'mapped'}
    monkeypatch.setattr(triton_driver, '_prepare_fma', prepare)
    result = triton_driver.prepare_driver(timeout_s=1, fma_geometry=(2560, 10, 11))
    assert calls == [(root, (2560, 10, 11))]
    assert result['profile'] == triton_driver.PROFILE32
    assert result['fma'] == {'launcher': 'mapped'}
    with triton_driver._cache_lock(root, .1):
        pass


@pytest.mark.parametrize('geometry', [None, [2560, 10, 11], (2560, True, 11), (2560, 10, 10), (1280, 10, 11)])
def test_profile32_requires_actual_sc117_signature_before_driver(monkeypatch, tmp_path, geometry):
    root = tmp_path / 'cache'
    _install32(monkeypatch, root, during=lambda: pytest.fail('driver used before geometry validation'))
    with pytest.raises(ValueError, match='validated SC117'):
        triton_driver.prepare_driver(timeout_s=1, fma_geometry=geometry)
    assert not root.exists()


@pytest.mark.parametrize('fault', ['profile', 'version', 'schema', 'changed', 'cache-manager', 'cache-root'])
def test_profile32_manifest_and_vendor_cache_contracts_fail_before_driver(monkeypatch, tmp_path, fault):
    root = tmp_path / 'cache'
    _, cache, manifest = _install32(monkeypatch, root, during=lambda: pytest.fail('driver used after invalid profile'))
    data = json.loads(manifest.read_text())
    if fault == 'profile':
        data['runtime_profile']['name'] = 'unselected'
    elif fault == 'version':
        data['packages']['triton']['version'] = '3.9.0'
    elif fault == 'schema':
        data['version'] = 2
    elif fault == 'changed':
        data['runtime_profile']['changed_distributions'].append('torch')
    elif fault == 'cache-manager':
        monkeypatch.setenv('TRITON_CACHE_MANAGER', 'unselected:Cache')
    else:
        cache.get_cache_manager = lambda key: SimpleNamespace(cache_dir=str(tmp_path / 'different' / key))
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='manifest|cache manager'):
        triton_driver.prepare_driver(timeout_s=1, fma_geometry=(2560, 10, 11))


@pytest.mark.parametrize('fault', ['unknown-profile', 'wrong-module', 'wrong-distribution'])
def test_version_profile_selection_is_explicit_and_not_a_missing_attribute_fallback(monkeypatch, tmp_path, fault):
    _install(monkeypatch, tmp_path / 'cache')
    if fault == 'unknown-profile':
        monkeypatch.setenv(triton_driver.PROFILE_ENV, 'sm70-triton-other')
    elif fault == 'wrong-module':
        sys.modules['triton'].__version__ = '3.2.0'
    else:
        monkeypatch.setattr(triton_driver.importlib.metadata, 'version', lambda name: '3.2.0')
    with pytest.raises(ValueError, match='profile'):
        triton_driver.prepare_driver(timeout_s=1)


@pytest.mark.parametrize('compiler_options', [{}, {'ptx_version': 83}])
def test_production_fma_warmup_realizes_vendor_launcher_without_invoking_it(monkeypatch, tmp_path, compiler_options):
    from tensorfold.families.qwen4_exp.cuda import glue
    import torch

    calls = []
    class MockTensor:
        def __init__(self, dtype):
            self.dtype = dtype
    class CudaLauncher:
        def __init__(self):
            self.launch = object()
        def __call__(self, *args):
            pytest.fail('startup must not invoke the launcher')
    class CompiledKernel:
        def __init__(self):
            self.src = SimpleNamespace(fn=None,
                signature={'Y': '*fp32', 'WTS': '*fp32', 'OUT': '*fp32'},
                constants={'D': 2560, 'TOPK': 10, 'SLOTS': 11, 'BLOCK': 256})
        @property
        def run(self):
            calls.append('launcher-realized')
            return CudaLauncher()
    compiled = CompiledKernel()
    def warmup(*operands, **options):
        assert all(type(value) is MockTensor and value.dtype == torch.float32 for value in operands)
        assert options == {'D': 2560, 'TOPK': 10, 'SLOTS': 11, 'BLOCK': 256,
                           'num_warps': 2, 'grid': (1, 10), **compiler_options}
        calls.append('actual-signature-warmup')
        return compiled
    kernel = SimpleNamespace(warmup=warmup, __name__='_moe_partial')
    compiled.src.fn = kernel
    monkeypatch.setattr(glue, '_moe_partial', kernel)
    monkeypatch.setattr(glue, '_MOE_COMPILER_OPTIONS', compiler_options)
    monkeypatch.setitem(sys.modules, 'triton.backends.nvidia.driver', SimpleNamespace(CudaLauncher=CudaLauncher))
    monkeypatch.setitem(sys.modules, 'triton.compiler.compiler', SimpleNamespace(CompiledKernel=CompiledKernel))
    monkeypatch.setitem(sys.modules, 'triton.runtime.jit', SimpleNamespace(MockTensor=MockTensor))
    def identity(function, root, prefix):
        assert root == tmp_path and prefix == '__triton_launcher'
        calls.append('actual-launcher-mapping')
        return {'sha256': 'b'*64}
    monkeypatch.setattr(triton_driver, '_mapped_code_identity', identity)
    result = triton_driver._prepare_fma(tmp_path, (2560, 10, 11))
    assert calls == ['actual-signature-warmup', 'launcher-realized', 'actual-launcher-mapping']
    assert result['geometry']['num_warps'] == 2 and result['launcher']['sha256'] == 'b'*64
    assert result['compiler_options'] == compiler_options
