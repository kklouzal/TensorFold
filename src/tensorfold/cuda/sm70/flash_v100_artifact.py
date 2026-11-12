"""Torch-absent sealed Flash26.10 artifact validation; explicit cold native loading.

Operator trees are administrator-staged read-only inputs. Manifest SHA is an
external selection identity, not a signature or evidence of model quality.
Native extension/code mappings may remain process resident after close; only
owned descriptors/loans retire here. CUDA work/graphs retire in their role owner.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType

PIN = '0ababf82e3836c1786d7937c6002670ca4ecda62'
INPUTS = MappingProxyType({
    "AUTHORS": (33, "f12de99664c47d52a464bd7b7df54f7c07d2056a33fef22ea3ab191de43e0d6c"),
    "LICENSE": (1511, "4eec967a15191488e23d9955fa472ba8074f89cfc3e4afacbf8e378180c28449"),
    "include/api.h": (16706, "9cc837ae93563de7b91a77dcbff71aa22905243bb4f4cc3b5ce2e9c02ae55845"),
    "include/backward.h": (4209, "70eb9f9613523b53b33faae673724e111b583efa20546384998b9b43f4b2e557"),
    "include/debug.h": (24628, "aafe08e7fc5b594227048a56e37613d840c9b22cca62f38952cb157e45d7c527"),
    "include/entropy.h": (1751, "87b991e89c2a8741ab4db346d61b5cb9d1335ba81543aaf95b1ce121daf09a9c"),
    "include/forward.h": (2234, "0ec3ef3849b386176b4ff725a39cbe9abdd28fc24b1e0a26f0122ec6df134560"),
    "include/gemm_smem.h": (10728, "20bd31e42c45610376821723625dc760d7f33ea36faecebd2c35581701a3c60d"),
    "include/kernel.h": (6167, "5838088500e25aa3f16abd6ebf1e89557095b27188bfbb7e163afc20b3e67d03"),
    "include/mat_mul.h": (12106, "db0466d1319e3051f10ed28889edaeeffb4f3fac5eb0fdca7a9a209d9408ff4a"),
    "include/mma_m16n16k16.h": (29663, "704c0a7ac60749947c0d88b8bd000fa5f9805db70e1916b322a3a39c977e4262"),
    "include/mma_m8n8k4.h": (22923, "84dc251cdd3c563bc3a3a9006e3128c174f143880a64853c9a3de8fc735a430c"),
    "include/philox.h": (2350, "34b867fd565e064949d3a0963c82b7f28cf00023ac1ea246efbb30da9957c0c1"),
    "include/product.h": (4399, "22061c0fad00921aa023e3aa7125f0219eafe8bc785eecc8b7a9513b5214a89d"),
    "include/rotary.h": (13165, "740f0fd2ce99df6fdc3eb49cf27b4fbbe7cb31235f1303eb05757cd5de5971f0"),
    "include/softmax.h": (16462, "11ca1feaafef3592263e61783cbdb3907eb389f13808f8c320a59e8d1b198b0d"),
    "include/swizzle.h": (8284, "a8c07ce63bb92abf1f888d735f7956ca59bd66d3007b8fe7fc2bc6c81079d056"),
    "include/template.h": (12486, "21d4369bd93466d5f1250768f6320ebd2d19f9ab9e901c3a5dc9ba80420bd09d"),
    "kernel/forward_varlen.cu": (27154, "876bdcc25c72104cb6ec71c044606ab81ccd861d7d4ded9ae3211258a7960874"),
    "kernel/forward.cu": (19802, "add843cc3c95ff9a3d74accf51c57ec9c00965f4a9a5945f71ec0fde83e7a991"),
})

CUDA_FLAGS = ('-O3', '-gencode=arch=compute_70,code=sm_70', '--use_fast_math',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr', '--expt-extended-lambda', '--threads=1')
ABI_FIELDS = frozenset(('torch', 'torch_git', 'cuda', 'cxx11_abi',
                        'torch_cxx_flags_sha256', 'torch_python_binding_sha256', 'headers'))
ABI_ORIGIN = ('seal metadata derived from unchanged protected SDK; '
              'compile receipt observes torch/cuda/SDK_headers, seal observes git/cxx11/static-cxx-flags/Python-binding')

FAMILY = 'flash-v100-q8-normalized-lse-owner-v1'
NATIVE_NAME_PATTERN = r'tf_fa_v100_forward_([0-9a-f]{20})\.so'
OWNED = ('flash_v100_forward.py', 'flash_v100_forward.cpp', 'flash_v100_forward.cu',
         'flash_v100_forward.cuh', 'prefill_dense_contracts.cuh', 'flash_v100_artifact.py',
         'flash_v100_NOTICE.txt', 'flash_v100_LICENSE.txt', 'attention_backends.py', 'history_owner.py')
NATIVE_INPUTS = ('flash_v100_forward.cpp', 'flash_v100_forward.cu',
                 'flash_v100_forward.cuh', 'prefill_dense_contracts.cuh')
FIELDS = frozenset(('schema', 'family', 'foreign_pin', 'foreign_inputs', 'owned_sources',
                   'native', 'abi', 'abi_origin', 'build', 'options', 'producer',
                   'compile_producer_sha256', 'profile_sha256'))


def owned_path(name):
    if name not in OWNED: raise ValueError('unknown sealed Flash owned source')
    package = Path(__file__).parents[2]
    return (package / 'families/qwen4_exp/cuda/history_owner.py' if name == 'history_owner.py'
            else package / 'cuda/sm70' / name)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _reject_constant(value):
    raise ValueError('nonfinite sealed Flash JSON constant: ' + value)


def _identity(st):
    return st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _acquire(path, *, limit=None, expected_bytes=None, capture=False, sink=None):
    """One captured-size acquisition; growth never extends the read envelope."""
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('sealed Flash inputs cannot traverse symlinks')
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('sealed Flash input must be an ordinary file')
    if ((limit is not None and before.st_size > limit)
            or (expected_bytes is not None and before.st_size != expected_bytes)):
        raise ValueError('sealed Flash input extent differs')
    digest = hashlib.sha256()
    chunks, cleanup = [], [None]
    stream, primary, result = None, None, None
    try:
        # The standard opener contract transfers the descriptor directly to
        # FileIO, including construction failure; no separate open/fdopen loan.
        import io
        stream = io.FileIO(path, 'rb', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW))
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError('sealed Flash input changed during acquisition')
        remaining = before.st_size
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError('sealed Flash input truncated during reading')
            digest.update(block)
            if capture: chunks.append(block)
            if sink is not None and sink.write(block) != len(block):
                raise OSError('sealed Flash output write was incomplete')
            remaining -= len(block)
        if stream.read(1):
            raise ValueError('sealed Flash input grew during reading')
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError('sealed Flash input changed during reading')
        if _identity(path.stat()) != _identity(before):
            raise ValueError('sealed Flash input changed during validation')
        result = ({'bytes': before.st_size, 'sha256': digest.hexdigest()}, b''.join(chunks) if capture else None)
    except BaseException as error:
        primary = error
    if stream is not None:
        try: stream.close()
        except BaseException as error: cleanup[0] = error
    # Keep this host-staging module independently importable with stdlib only.
    # There is exactly one completed FD retirement; retain both errors while
    # preserving the acquisition/read failure as the primary outcome.
    if primary is not None:
        if cleanup[0] is not None: raise primary from cleanup[0]
        raise primary
    if cleanup[0] is not None: raise cleanup[0]
    return result


def file_record(path):
    """Stream exactly the captured ordinary-file extent, plus one growth byte."""
    return _acquire(path)[0]


def _check_file(path, expected, *, capture=False, limit=None):
    if (type(expected) is not dict or set(expected) != {'bytes', 'sha256'}
            or type(expected['bytes']) is not int or not 0 <= expected['bytes'] <= sys.maxsize
            or not _sha(expected['sha256'])):
        raise ValueError('sealed Flash file identity schema differs: ' + str(path))
    identity, raw = _acquire(path, expected_bytes=expected['bytes'], capture=capture, limit=limit)
    if identity != expected:
        raise ValueError('sealed Flash file identity differs: ' + str(path))
    return raw


@dataclass(frozen=True, init=False)
class ValidatedArtifact:
    path: Path
    manifest_sha256: str
    _raw: bytes
    profile_sha256: str

    def __init__(self, path, digest, data):
        object.__setattr__(self, 'path', path)
        object.__setattr__(self, 'manifest_sha256', digest)
        object.__setattr__(self, '_raw', canonical(data))
        object.__setattr__(self, 'profile_sha256', data['profile_sha256'])

    def record(self):
        data = json.loads(self._raw)
        return {'native_manifest': str(self.path), 'native_manifest_sha256': self.manifest_sha256,
                'profile_sha256': self.profile_sha256, 'family': FAMILY,
                'native_bytes': data['native']['bytes'],
                'sealed_payload_bytes': sum(r['bytes'] for r in data['owned_sources'].values())
                    + sum(r[0] for r in data['foreign_inputs'].values()) + data['native']['bytes'],
                'opaque_residency': 'bounded process-resident extension/driver maps; measure cold admission',
                'seal_producer_sha256': data['producer']['sha256'],
                'compile_producer_sha256': data['compile_producer_sha256'],
                'abi': data['abi'], 'abi_origin': data['abi_origin']}

    def staging_files(self):
        """Complete producer closure for bounded host copying, never native import.

        The manifest SHA remains the external authority. The checksum sidecar
        is checked as transport data, not accepted as its own authority.
        """
        if self.path.name != 'manifest.json':
            raise ValueError('Flash operator staging requires the producer manifest.json')
        data = json.loads(self._raw)
        manifest, _ = _acquire(self.path, limit=1 << 20)
        if manifest['sha256'] != self.manifest_sha256:
            raise ValueError('Flash manifest changed before staging inventory')
        checksum = (self.manifest_sha256 + '\n').encode('ascii')
        checksum_identity = {'bytes': len(checksum), 'sha256': hashlib.sha256(checksum).hexdigest()}
        _check_file(self.path.parent / 'manifest.sha256', checksum_identity)
        files = {'manifest.json': manifest, 'manifest.sha256': checksum_identity,
                 'build.json': data['build'], 'producer/build_flash_v100_forward.py': data['producer'],
                 'native/' + data['native']['name']: {k: data['native'][k] for k in ('bytes', 'sha256')}}
        files.update({'foreign/' + name: {'bytes': size, 'sha256': sha} for name, (size, sha) in INPUTS.items()})
        files.update({'owned/' + name: value for name, value in data['owned_sources'].items()})
        return files


def validate_manifest(path, digest):
    """Host staging API: stdlib only, no Torch, native import or CUDA query."""
    path = Path(path).absolute()
    if not _sha(digest):
        raise ValueError('sealed Flash manifest identity/extent')
    identity, raw = _acquire(path, limit=1 << 20, capture=True)
    if identity['sha256'] != digest:
        raise ValueError('sealed Flash manifest identity differs before parse')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate sealed Flash manifest key')
            result[key] = value
        return result
    data = json.loads(raw, object_pairs_hook=pairs, parse_constant=_reject_constant)
    if (type(data) is not dict or set(data) != FIELDS or type(data['schema']) is not int or data['schema'] != 1
            or data['family'] != FAMILY or data['foreign_pin'] != PIN
            or data['foreign_inputs'] != {k: list(v) for k, v in INPUTS.items()}
            or type(data['owned_sources']) is not dict or set(data['owned_sources']) != set(OWNED)
            or type(data['abi']) is not dict or type(data['options']) is not dict
            or data['abi_origin'] != ABI_ORIGIN or not _sha(data['compile_producer_sha256'])
            or not _sha(data['profile_sha256'])):
        raise ValueError('sealed Flash schema/profile/source closure differs')
    unsigned = {k: v for k, v in data.items() if k != 'profile_sha256'}
    if hashlib.sha256(canonical(unsigned)).hexdigest() != data['profile_sha256']:
        raise ValueError('sealed Flash numerical profile identity differs')
    abi, options = data['abi'], data['options']
    if (set(abi) != ABI_FIELDS or type(abi['torch']) is not str or not abi['torch'].startswith('2.16.')
            or type(abi['torch_git']) is not str or re.fullmatch('[0-9a-f]{40}', abi['torch_git']) is None
            or abi['cuda'] != '12.9' or type(abi['cxx11_abi']) is not bool
            or not _sha(abi['torch_cxx_flags_sha256']) or not _sha(abi['torch_python_binding_sha256'])
            or type(abi['headers']) is not dict
            or set(options) != {'extra_cflags', 'extra_cuda_cflags', 'extra_include_paths'}
            or options['extra_cflags'] != ['-O3'] or options['extra_cuda_cflags'] != list(CUDA_FLAGS)):
        raise ValueError('sealed Flash ABI/compiler numerical options differ')
    native = data['native']
    if (type(native) is not dict or set(native) != {'name', 'bytes', 'sha256'}
            or type(native['name']) is not str
            or re.fullmatch(NATIVE_NAME_PATTERN, native['name']) is None):
        raise ValueError('sealed Flash native module name/schema differs')
    root = path.parent
    for name, (size, sha) in INPUTS.items():
        _check_file(root / 'foreign' / name, {'bytes': size, 'sha256': sha})
    for name in OWNED:
        # Reject an advertised source extent against the actual runtime source
        # before streaming a potentially hostile staged copy of that extent.
        _check_file(owned_path(name), data['owned_sources'][name])
        _check_file(root / 'owned' / name, data['owned_sources'][name])
    _check_file(root / 'native' / native['name'], {k: native[k] for k in ('bytes', 'sha256')})
    build_raw = _check_file(root / 'build.json', data['build'], capture=True, limit=1 << 20)
    _check_file(root / 'producer' / 'build_flash_v100_forward.py', data['producer'])
    receipt = json.loads(build_raw, object_pairs_hook=pairs, parse_constant=_reject_constant)
    if (receipt.get('status') != 'compiled-and-imported' or receipt.get('cuda_initialized_before') is not False
            or receipt.get('cuda_initialized_after') is not False or receipt.get('upstream_commit') != PIN
            or receipt.get('foreign_inputs') != data['foreign_inputs'] or receipt.get('module_sha256') != native['sha256']
            or receipt.get('requested_options') != data['options']
            or receipt.get('SDK_headers') != data['abi'].get('headers')
            or receipt.get('torch') != data['abi'].get('torch') or receipt.get('torch_cuda') != data['abi'].get('cuda')):
        raise ValueError('sealed Flash build receipt differs from selected profile')
    producer = [sha for file, sha in receipt['owned_source_sha256'].items()
                if Path(file).name == 'build_flash_v100_forward.py']
    if producer != [data['compile_producer_sha256']]:
        raise ValueError('sealed Flash original compile producer differs')
    includes = data['options']['extra_include_paths']
    foreign_cu = [file for file in receipt['owned_source_sha256'] if file.endswith('/kernel/forward.cu')]
    if (len(foreign_cu) != 1 or includes != [*abi['headers'].get('include_roots', []),
            str(Path(foreign_cu[0]).parent.parent / 'include')]):
        raise ValueError('sealed Flash compiler include closure differs')
    for name in NATIVE_INPUTS:
        matches = [sha for file, sha in receipt['owned_source_sha256'].items() if Path(file).name == name]
        if matches != [data['owned_sources'][name]['sha256']]:
            raise ValueError('sealed Flash compiled source differs from runtime source')
    return ValidatedArtifact(path, digest, data)


def runtime_abi():
    """Cold SDK-only identity; import metadata does not initialize CUDA."""
    import torch
    from .foundation_headers import selected_headers
    headers = selected_headers()
    if headers is None: raise ValueError('sealed Flash requires selected protected SDK headers')
    # show() is intentionally excluded: owning Torch05adb Version.cpp includes
    # a CUDAHooks section only when devices are visible. CXX flags come from
    # compile-time CXX_FLAGS, and the actual CPython binding bytes are static.
    return {'torch': torch.__version__, 'torch_git': torch.version.git_version, 'cuda': torch.version.cuda,
            'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            'torch_cxx_flags_sha256': hashlib.sha256(torch.__config__._cxx_flags().encode()).hexdigest(),
            'torch_python_binding_sha256': file_record(torch._C.__file__)['sha256'],
            'headers': headers.record()}


class NativeArtifact:
    """Explicit bounded role loans. Close retires FD/journal, not extension maps."""
    def __init__(self, validated, module, fd):
        self.validated, self._module, self._fd = validated, module, fd
        self._loans, self.closed, self.failed = set(), False, False

    @property
    def module(self):
        if self.closed or self.failed: raise RuntimeError('Flash native artifact retired or failed')
        return self._module

    def loan(self):
        self.module
        if len(self._loans) >= 3: raise MemoryError('Flash artifact exceeds three prepared role owners')
        token = object(); self._loans.add(token); return token

    def release(self, token):
        if token not in self._loans: raise RuntimeError('Flash artifact loan was not active')
        self._loans.remove(token)

    def record(self):
        return {**self.validated.record(), 'active_role_loans': len(self._loans),
                'host_journal_bytes': sys.getsizeof(self) + sys.getsizeof(self._loans) + len(self.validated._raw),
                'closed': self.closed, 'retirement_failed': self.failed, 'extension_unload_claimed': False}

    def close(self):
        if self.closed: return
        if self.failed: raise RuntimeError('Flash artifact FD retirement uncertain; retry forbidden')
        if self._loans: raise RuntimeError('Flash artifact roles/graphs must retire before FD close')
        try: os.close(self._fd)
        except BaseException:
            self.failed = True; raise
        self._fd = self._module = None
        self.closed = True


def load_native(validated):
    """Cold only; caller owns read-only operator staging and serialized startup."""
    validated = validate_manifest(validated.path, validated.manifest_sha256)
    data = json.loads(validated._raw)
    path = validated.path.parent / 'native' / data['native']['name']
    if not os.statvfs(path).f_flag & os.ST_RDONLY:
        raise ValueError('Flash native artifact requires an operator read-only mount')
    import torch
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Flash native loading is forbidden inside CUDA capture')
    observed_abi = runtime_abi()
    if observed_abi != data['abi']:
        different = sorted(key for key in ABI_FIELDS if observed_abi[key] != data['abi'][key])
        raise ValueError('Flash native Torch/CUDA/header ABI differs from seal: ' + ','.join(different))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _check_file(path, {k: data['native'][k] for k in ('bytes', 'sha256')})
        spec = importlib.util.spec_from_file_location(Path(data['native']['name']).stem, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return NativeArtifact(validated, module, fd)
    except BaseException as primary:
        try: os.close(fd)
        except BaseException as cleanup: raise primary from cleanup
        raise
