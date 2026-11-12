"""Sealed artifact boundaries and explicit FD/role ownership; no native math proof."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest
from tensorfold.cuda.sm70 import flash_v100_artifact as artifact


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    raw = b'// foreign fixture\n'
    inputs = MappingProxyType({'kernel/forward.cu': (len(raw), hashlib.sha256(raw).hexdigest())})
    monkeypatch.setattr(artifact, 'INPUTS', inputs)
    (tmp_path / 'foreign/kernel').mkdir(parents=True)
    (tmp_path / 'foreign/kernel/forward.cu').write_bytes(raw)
    (tmp_path / 'owned').mkdir()
    owned = {}
    for name in artifact.OWNED:
        (tmp_path / 'owned' / name).write_bytes(artifact.owned_path(name).read_bytes())
        owned[name] = artifact.file_record(tmp_path / 'owned' / name)
    (tmp_path / 'native').mkdir()
    native = tmp_path / 'native' / ('tf_fa_v100_forward_' + '0' * 20 + '.so')
    native.write_bytes(b'not executable: host fixture only')
    abi = {'torch': '2.16.fixture', 'torch_git': '0' * 40, 'cuda': '12.9', 'cxx11_abi': True,
           'torch_cxx_flags_sha256': '1' * 64, 'torch_python_binding_sha256': '3' * 64,
           'headers': {'include_roots': ['/sdk/cccl', '/sdk/cutlass']}}
    options = {'extra_cflags': ['-O3'], 'extra_cuda_cflags': list(artifact.CUDA_FLAGS),
               'extra_include_paths': ['/sdk/cccl', '/sdk/cutlass', '/foreign/include']}
    receipt = {'status': 'compiled-and-imported', 'upstream_commit': artifact.PIN,
        'foreign_inputs': {k: list(v) for k, v in inputs.items()}, 'module_sha256': artifact.file_record(native)['sha256'],
        'requested_options': options, 'SDK_headers': abi['headers'], 'torch': abi['torch'], 'torch_cuda': abi['cuda'],
        'cuda_initialized_before': False, 'cuda_initialized_after': False,
        'owned_source_sha256': {'/source/' + n: owned[n]['sha256'] for n in artifact.NATIVE_INPUTS}}
    receipt['owned_source_sha256']['/foreign/kernel/forward.cu'] = inputs['kernel/forward.cu'][1]
    receipt['owned_source_sha256']['/source/tools/build_flash_v100_forward.py'] = '2' * 64
    (tmp_path / 'build.json').write_bytes(artifact.canonical(receipt))
    (tmp_path / 'producer').mkdir(); (tmp_path / 'producer/build_flash_v100_forward.py').write_bytes(b'# producer fixture\n')
    data = {'schema': 1, 'family': artifact.FAMILY, 'foreign_pin': artifact.PIN,
        'foreign_inputs': receipt['foreign_inputs'], 'owned_sources': owned,
        'native': {'name': native.name, **artifact.file_record(native)}, 'abi': abi, 'options': options,
        'abi_origin': artifact.ABI_ORIGIN, 'compile_producer_sha256': '2' * 64,
        'build': artifact.file_record(tmp_path / 'build.json'),
        'producer': artifact.file_record(tmp_path / 'producer/build_flash_v100_forward.py')}
    def write():
        data.pop('profile_sha256', None)
        data['profile_sha256'] = hashlib.sha256(artifact.canonical(data)).hexdigest()
        path = tmp_path / 'manifest.json'; path.write_bytes(artifact.canonical(data))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    return tmp_path, data, write


def test_manifest_is_complete_immutable_identity_and_import_does_not_need_torch(sealed):
    root, data, write = sealed
    path, digest = write(); checked = artifact.validate_manifest(path, digest)
    assert checked.profile_sha256 == data['profile_sha256']
    with pytest.raises(AttributeError): checked.profile_sha256 = '0' * 64
    source = Path(__file__).resolve().parents[1] / 'src'
    result = subprocess.run([sys.executable, '-S', '-c',
        'import sys;sys.path.insert(0,sys.argv[1]);from tensorfold.cuda.sm70 import flash_v100_artifact;'
        'assert "torch" not in sys.modules', str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('target', ['foreign/kernel/forward.cu', 'native/tf_fa_v100_forward_' + '0'*20 + '.so',
                                  'owned/flash_v100_forward.py', 'build.json', 'producer/build_flash_v100_forward.py'])
def test_changed_payload_refused_before_native_acquisition(sealed, target):
    root, data, write = sealed; path, digest = write()
    victim = root / target; victim.write_bytes(victim.read_bytes() + b'!')
    with pytest.raises(ValueError): artifact.validate_manifest(path, digest)


def test_symlink_profile_digest_flags_and_compiler_source_receipt_refused(sealed):
    root, data, write = sealed; path, digest = write()
    native = root / 'native' / data['native']['name']; original = native.read_bytes()
    native.unlink(); native.symlink_to(root / 'missing')
    with pytest.raises((ValueError, FileNotFoundError)): artifact.validate_manifest(path, digest)
    native.unlink(); native.write_bytes(original)
    with pytest.raises(ValueError): artifact.validate_manifest(path, 'f' * 64)
    data['options']['extra_cuda_cflags'] = ['--use_fast_math']
    path, digest = write()
    with pytest.raises(ValueError, match='options'): artifact.validate_manifest(path, digest)


def test_loading_rejects_writable_operator_mount_before_torch_or_dlopen(sealed, monkeypatch):
    root, data, write = sealed; path, digest = write(); checked = artifact.validate_manifest(path, digest)
    monkeypatch.setattr(artifact.os, 'statvfs', lambda path: type('Stat', (), {'f_flag': 0})())
    with pytest.raises(ValueError, match='read-only'): artifact.load_native(checked)


def test_role_loans_bound_fd_retirement_without_unload_claim(sealed):
    root, data, write = sealed; path, digest = write(); checked = artifact.validate_manifest(path, digest)
    import os
    fd = os.open(path, os.O_RDONLY)
    module = object(); owner = artifact.NativeArtifact(checked, module, fd)
    loans = [owner.loan() for _ in range(3)]
    with pytest.raises(MemoryError): owner.loan()
    with pytest.raises(RuntimeError, match='roles'): owner.close()
    for loan in loans: owner.release(loan)
    owner.close(); owner.close()
    assert owner.closed and owner.record()['extension_unload_claimed'] is False
    with pytest.raises(RuntimeError): owner.module


def test_cold_abi_uses_static_flags_and_binding_bytes_not_device_summary(monkeypatch):
    import torch
    from tensorfold.cuda.sm70 import foundation_headers
    monkeypatch.setattr(foundation_headers, 'selected_headers', lambda: type('Headers', (), {'record': lambda s: {}})())
    def forbidden(): pytest.fail('device-dependent show_config is not a build ABI identity')
    monkeypatch.setattr(torch.__config__, 'show', forbidden)
    result = artifact.runtime_abi()
    assert result['torch_cxx_flags_sha256'] == hashlib.sha256(torch.__config__._cxx_flags().encode()).hexdigest()
    assert result['torch_python_binding_sha256'] == artifact.file_record(torch._C.__file__)['sha256']


@pytest.mark.parametrize('target', ['manifest.json', 'build.json', 'record'])
def test_growing_input_is_read_only_to_captured_extent_plus_one(sealed, monkeypatch, target):
    import io
    root, data, write = sealed
    manifest, digest = write()
    path = root / target
    if target == 'record': path.write_bytes(b'abcd')
    extent, requested, opened = path.stat().st_size, [], []
    original = io.FileIO
    class Growing:
        def __init__(self, name, mode, **kwargs):
            self.stream = original(name, mode, **kwargs)
            self.target = Path(name) == path
            if self.target: opened.append(name)
        def fileno(self): return self.stream.fileno()
        def close(self): self.stream.close()
        def read(self, size):
            if self.target:
                requested.append(size)
                if len(requested) == 1:
                    with path.open('ab') as writer: writer.write(b'x' * (1 << 20))
            return self.stream.read(size)
    monkeypatch.setattr(io, 'FileIO', Growing)
    with pytest.raises(ValueError, match='grew'):
        if target == 'record': artifact.file_record(path)
        else: artifact.validate_manifest(manifest, digest)
    assert len(opened) == 1 and requested == [extent, 1]
