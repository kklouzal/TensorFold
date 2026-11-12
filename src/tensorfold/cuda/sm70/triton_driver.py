"""Serialize cold Triton driver publication before SM70 rank warmup.

Pinned Triton's cache publisher atomically replaces an existing helper path.
Concurrent cold callers can therefore unlink a helper already mapped by a peer.
This startup-only lock covers driver preparation, without changing vendor code,
cache identity or inference scheduling. All ranks using this cache participate.
"""
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import math
import mmap
import os
from pathlib import Path
import stat
import time

from tensorfold.cleanup import raise_failures

PROFILE_ENV = 'TENSORFOLD_SM70_TRITON_PROFILE'
PROFILE32 = 'sm70-triton32-v1'
_RUNTIME_MANIFEST = Path('/opt/tensorfold-pins/source-runtime-amd64.json')


@contextmanager
def _cache_lock(root, timeout_s):
    """Bound lock acquisition; the existing startup owner bounds compilation."""
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / '.tensorfold-driver.lock', os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    primary, errors = None, []
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('Triton driver lock must be a regular file')
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Triton driver preparation lock timed out') from None
                time.sleep(min(.05, remaining))
        yield
    except BaseException as error:
        primary = error
    finally:
        try:
            os.close(fd)  # Closing the sole owned descriptor releases flock.
        except BaseException as error:
            errors.append(error)
    raise_failures(primary, errors)


def _backing_identity(fd, path):
    """Observe the held file's actual VMA backing, including OverlayFS.

    OverlayFS can expose a synthetic stat device while mmap uses its real
    backing file. A private one-byte writable view supplies an address without
    writing the file or requiring privileged map_files descriptor access.
    """
    probe = borrowed = None
    primary, errors, result = None, [], None
    try:
        probe = mmap.mmap(fd, 1, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        borrowed = ctypes.c_char.from_buffer(probe)
        address = ctypes.addressof(borrowed)
        borrowed = None
        with open('/proc/self/maps', encoding='utf-8') as stream:
            for line in stream:
                fields = line.rstrip('\n').split(None, 5)
                start, end = (int(value, 16) for value in fields[0].split('-'))
                if start <= address < end:
                    if len(fields) != 6 or fields[5] != str(path):
                        raise ValueError('Triton held helper backing path changed or was deleted')
                    major, minor = (int(value, 16) for value in fields[3].split(':'))
                    result = major, minor, int(fields[4])
                    break
        if result is None:
            raise ValueError('Triton held helper backing mapping is unavailable')
    except BaseException as error:
        primary = error
    finally:
        borrowed = None  # Retire the exported buffer before its mapping.
        if probe is not None:
            try:
                probe.close()
            except BaseException as error:
                errors.append(error)
    raise_failures(primary, errors)
    return result


def _mapped_code_identity(function, root, prefix):
    # Triton loads this extension through module_from_spec without promising a
    # sys.modules entry. The actual builtin retains its defining module.
    module = getattr(function, '__self__', None)
    filename = getattr(module, '__file__', None)
    if type(filename) is not str:
        raise ValueError('Triton driver helper lacks its actual module path')
    path = Path(filename)
    resolved = path.resolve(strict=True)
    if path != resolved or not resolved.is_relative_to(root) or not resolved.name.startswith(prefix + '.'):
        raise ValueError('Triton driver helper is outside its persistent cache')
    fd = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    primary, errors, result = None, [], None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError('Triton driver helper must be a nonempty regular file')
        backing = _backing_identity(fd, resolved)
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            data = os.read(fd, min(remaining, 1 << 20))
            if not data:
                raise ValueError('Triton driver helper was truncated while hashing')
            digest.update(data)
            remaining -= len(data)
        if os.read(fd, 1):
            raise ValueError('Triton driver helper grew while hashing')
        mappings = []
        with open('/proc/self/maps', encoding='utf-8') as stream:
            for line in stream:
                fields = line.rstrip('\n').split(None, 5)
                if len(fields) == 6 and fields[5] in (str(resolved), str(resolved) + ' (deleted)'):
                    if fields[5].endswith(' (deleted)'):
                        raise ValueError('Triton driver helper has a deleted loaded mapping: ' + str(resolved))
                    major, minor = (int(value, 16) for value in fields[3].split(':'))
                    if (major, minor, int(fields[4])) != backing:
                        raise ValueError('Triton driver helper mapping differs from its persistent inode')
                    mappings.append(fields[1])
        after, current = os.fstat(fd), resolved.stat()
        def identity(info):
            return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if (identity(before) != identity(after) or identity(before) != identity(current)
                or not any('x' in permissions for permissions in mappings)):
            raise ValueError('Triton driver helper changed or lacks its executable mapping')
        result = {'path': str(resolved), 'bytes': before.st_size, 'sha256': digest.hexdigest()}
    except BaseException as error:
        primary = error
    finally:
        try:
            os.close(fd)
        except BaseException as error:
            errors.append(error)
    raise_failures(primary, errors)
    return result


def _helper_identity(utils, root):
    return _mapped_code_identity(utils.load_binary, root, 'cuda_utils')


def _profile(triton):
    selected = os.environ.get(PROFILE_ENV)
    expected = '3.9.0' if selected is None else '3.2.0'
    if selected is not None and selected != PROFILE32:
        raise ValueError('unsupported explicit SM70 Triton startup profile')
    if triton.__version__ != expected or importlib.metadata.version('triton') != expected:
        raise ValueError('Triton module/distribution differs from its explicit SM70 profile')
    if selected is None:
        return None
    # The SDK guard owns provider contents, toolchain and cache-key hashes.
    # This bounded read binds the explicit branch to its sealed runtime schema;
    # an environment label alone cannot admit another compiler distribution.
    with _RUNTIME_MANIFEST.open('rb') as stream:
        raw = stream.read((1 << 20) + 1)
    if len(raw) > 1 << 20:
        raise ValueError('SM70 Triton runtime manifest exceeds its startup extent')
    runtime = json.loads(raw)
    if type(runtime) is not dict:
        raise ValueError('SM70 Triton startup requires the selected SDK object manifest')
    profile = runtime.get('runtime_profile')
    packages = runtime.get('packages')
    if (runtime.get('schema') != 'tensorfold.v100-source-runtime' or type(runtime.get('version')) is not int
            or runtime['version'] != 3 or type(profile) is not dict
            or profile.get('schema') != 'tensorfold.v100-runtime-profile'
            or type(profile.get('version')) is not int or profile['version'] != 1
            or profile.get('name') != PROFILE32 or profile.get('changed_distributions') != ['triton']
            or type(packages) is not dict or type(packages.get('triton')) is not dict
            or packages['triton'].get('version') != expected):
        raise ValueError('SM70 Triton startup profile differs from the selected SDK manifest')
    return selected


def _cache_root(triton, selected, configured):
    if type(configured) is not str or not configured or configured != configured.strip():
        raise ValueError('Triton driver cache differs from the configured SDK/runtime root')
    root = Path(configured)
    if not root.is_absolute():
        raise ValueError('Triton driver cache must be an absolute persistent path')
    if selected is None:
        if triton.knobs.cache.dir != configured:
            raise ValueError('Triton driver cache differs from the configured SDK/runtime root')
    else:
        from triton.runtime import cache

        if os.environ.get('TRITON_CACHE_MANAGER') is not None or cache.__cache_cls is not cache.FileCacheManager:
            raise ValueError('Triton3.2 startup requires the selected standard file cache manager')
        # Instantiate the actual vendor manager, whose CUDA helper publishers
        # use this same implementation. No private copy of its key encoding.
        manager = cache.get_cache_manager(hashlib.sha256(b'tensorfold-startup-cache-root-v1').hexdigest())
        if type(manager) is not cache.FileCacheManager or Path(manager.cache_dir).parent != root:
            raise ValueError('Triton3.2 actual cache manager selected a different persistent root')
    return root.resolve()


def _launcher_factory_source(factory):
    """Bind the active vendor factory to the selected SDK source file.

    Triton3.2 loads its backend under a dynamic ``nvidia`` module name. Importing
    the canonical module again creates a distinct nominal class from the same
    source, so that second class is not the launcher's ownership authority.
    """
    import triton

    expected = (Path(triton.__file__).resolve(strict=True).parent / 'backends/nvidia/driver.py').resolve(strict=True)
    initializer = getattr(factory, '__init__', None)
    code, namespace = getattr(initializer, '__code__', None), getattr(initializer, '__globals__', None)
    if (code is None or type(namespace) is not dict or type(namespace.get('__file__')) is not str
            or Path(code.co_filename).resolve(strict=True) != expected
            or Path(namespace['__file__']).resolve(strict=True) != expected):
        raise ValueError('Triton3.2 active launcher factory escaped the selected SDK source')
    data = expected.read_bytes()
    return {'path': str(expected), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def _prepare_fma(root, geometry):
    import torch
    from triton.runtime.driver import driver
    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.jit import MockTensor
    from tensorfold.families.qwen4_exp.cuda import glue

    hidden, top_k, slots = geometry
    operands = tuple(MockTensor(torch.float32) for _ in range(3))
    compiled = glue._moe_partial.warmup(*operands, D=hidden, TOPK=top_k, SLOTS=slots,
                                      BLOCK=256, num_warps=2, grid=(1, hidden // 256),
                                      **glue._MOE_COMPILER_OPTIONS)
    if (type(compiled) is not CompiledKernel
            or compiled.src.fn is not glue._moe_partial
            or compiled.src.signature != {'Y': '*fp32', 'WTS': '*fp32', 'OUT': '*fp32'}
            or compiled.src.constants != {'D': hidden, 'TOPK': top_k, 'SLOTS': slots, 'BLOCK': 256}):
        raise ValueError('Triton3.2 startup did not compile the actual FP32 MoE signature')
    # CompiledKernel.run initializes its vendor CudaLauncher and loads the
    # device module. Merely warming compilation leaves the launcher publisher
    # for the first inference, where peers could invalidate a loaded mapping.
    # Do not invoke this callable: startup prepares no model tensor/collective.
    launcher = compiled.run
    factory = driver.active.launcher_cls
    if type(factory) is not type or type(launcher) is not factory:
        raise ValueError('Triton3.2 FP32 kernel did not realize its actual CUDA launcher')
    factory_source = _launcher_factory_source(factory)
    identity = _mapped_code_identity(launcher.launch, root, '__triton_launcher')
    return {'kernel': glue._moe_partial.__name__, 'signature': dict(compiled.src.signature),
            'compiler_options': dict(glue._MOE_COMPILER_OPTIONS),
            'geometry': {'hidden': hidden, 'top_k': top_k, 'slots': slots, 'block': 256, 'num_warps': 2},
            'launcher': identity, 'launcher_factory_source': factory_source,
            'scope': 'actual-production-signature-compiled-and-loaded; no-kernel-launch'}


def prepare_driver(*, timeout_s, fma_geometry=None):
    """Prepare the actual SM70 CUDA driver under its shared cache lock.

    ``timeout_s`` is the caller's existing startup allowance, not an inference
    budget. It bounds lock acquisition; the outer startup owner bounds compiler
    subprocess lifetime. No lock or file hashing occurs during inference.
    """
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('positive finite existing startup timeout required')
    import triton
    selected = _profile(triton)
    if selected is not None and (type(fma_geometry) is not tuple or len(fma_geometry) != 3
            or any(type(value) is not int for value in fma_geometry) or fma_geometry != (2560, 10, 11)):
        raise ValueError('Triton3.2 startup requires the validated SC117 FP32 FMA geometry')
    configured = os.environ.get('TRITON_CACHE_DIR')
    root = _cache_root(triton, selected, configured)
    with _cache_lock(root, timeout_s):
        from triton.runtime import driver

        active = driver.active
        target = active.get_current_target()
        if target.backend != 'cuda' or target.arch != 70 or target.warp_size != 32:
            raise ValueError('Triton driver preparation requires the actual SM70 CUDA target')
        helper = _helper_identity(active.utils, root)
        fma = None if selected is None else _prepare_fma(root, fma_geometry)
        if (os.environ.get('TRITON_CACHE_DIR') != configured or os.environ.get(PROFILE_ENV) != selected
                or _cache_root(triton, selected, configured) != root
                or Path(configured).resolve(strict=True) != root):
            raise ValueError('Triton SDK/runtime cache changed during driver preparation')
        return {'backend': target.backend, 'arch': target.arch, 'warp_size': target.warp_size,
                'cache': configured, 'helper': helper,
                'scope': 'startup-only-serialized-driver-preparation' if selected is None else
                         'startup-only-serialized-driver-and-fp32-launcher-preparation',
                **({} if selected is None else {'profile': selected, 'fma': fma})}
