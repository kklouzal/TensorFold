#!/usr/bin/env python3
"""Build the optional exact26.10 forward component without package installation.

Download-only uses standard Python. Compile/import requires the GPU-hidden
CUDA12.9/Torch2.16 SDK, explicit SM70 and caller-bounded jobs. This standalone
recipe does not register or qualify a production attention provider.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
import urllib.request

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
    "kernel/forward.cu": (19802, "add843cc3c95ff9a3d74accf51c57ec9c00965f4a9a5945f71ec0fde83e7a991"),
})


def read_input(path, size, digest):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError('FA source type/size differs from pin: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('FA source digest differs from pin: ' + str(path))
    return raw


def verify(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError('FA source root must be an ordinary directory')
    for name in ('include', 'kernel'):
        if (root / name).is_symlink() or not (root / name).is_dir():
            raise ValueError('FA source directories must be ordinary directories')
    headers = {Path(name).name for name in INPUTS if name.startswith('include/')}
    observed = set()
    for path in (root / 'include').iterdir():
        if path.name not in headers:
            raise ValueError('FA include directory contains an undeclared input')
        observed.add(path.name)
    if observed != headers:
        raise ValueError('FA include inventory differs from its complete pinned closure')
    if any((root / 'kernel' / name).exists() or (root / 'kernel' / name).is_symlink() for name in headers):
        raise ValueError('FA kernel directory shadows a pinned quoted include')
    for name, (size, digest) in INPUTS.items():
        read_input(root / name, size, digest)


def download(root):
    if root.is_symlink():
        raise ValueError('FA source destination must not be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    for name in ('include', 'kernel'):
        path = root / name
        if path.is_symlink():
            raise ValueError('FA source directory must not be a symlink')
        path.mkdir(exist_ok=True)
    for name, (size, digest) in INPUTS.items():
        target = root / name
        if target.exists() or target.is_symlink():
            read_input(target, size, digest)  # Resume valid files; never overwrite a changed input.
            continue
        url = 'https://raw.githubusercontent.com/ai-bond/flash-attention-v100/' + PIN + '/' + name
        with urllib.request.urlopen(url, timeout=30) as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('FA download differs from immutable pin: ' + name)
        with target.open('xb') as stream:
            stream.write(raw)
    verify(root)


def compile_component(root, output, jobs):
    import torch
    from torch.utils import cpp_extension
    from tensorfold.cuda.sm70.foundation_headers import selected_headers
    from tensorfold.cleanup import raise_failures

    if os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void' or torch.cuda.is_initialized():
        raise ValueError('compile-only requires a GPU-hidden SDK process without a CUDA context')
    if torch.version.cuda != '12.9' or not torch.__version__.startswith('2.16.'):
        raise ValueError('component recipe targets the actual CUDA12.9/Torch2.16 SDK')
    headers = selected_headers()
    if headers is None:
        raise ValueError('component compilation requires protected selected SDK headers')
    owned = Path(__file__).resolve().parents[1] / 'src/tensorfold/cuda/sm70'
    sources = [owned / 'flash_v100_forward.cpp', owned / 'flash_v100_forward.cu', root / 'kernel/forward.cu']
    files = [*sources, owned / 'flash_v100_forward.cuh', owned / 'prefill_dense_contracts.cuh']
    digests = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    options = dict(extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '-gencode=arch=compute_70,code=sm_70', '--use_fast_math',
            '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
            '--expt-relaxed-constexpr', '--expt-extended-lambda', '--threads=1'],
        extra_include_paths=[*map(str, headers.roots), str(root / 'include')])
    record = {'scope': 'standalone controlled component compile/import; no JIT/provider/model qualification',
        'upstream_commit': PIN, 'foreign_inputs': dict(INPUTS), 'owned_source_sha256': digests,
        'SDK_headers': headers.record(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'requested_options': json.loads(json.dumps(options)), 'jobs': jobs, 'cuda_initialized_before': False}
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    directory = output / ('build-' + identity[:20])
    directory.mkdir(exist_ok=True)
    (output / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    os.environ['MAX_JOBS'] = str(jobs)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'
    primary = None
    cleanup = [None, None, None]
    unexpected_cuda = RuntimeError('compile-only process unexpectedly initialized CUDA')
    try:
        module = cpp_extension.load(name='tf_fa_v100_forward_' + identity[:20], sources=list(map(str, sources)),
            build_directory=str(directory), verbose=True, with_cuda=True, **options)
        module.validate_geometry(1, 1, 12, 2051, 1)
        module.validate_geometry(1, 1, 32768, 256, 32768)
        try:
            module.validate_geometry(1, 1, 32768, 257, 32768)
        except RuntimeError as error:
            if 'KV tile offset overflow' not in str(error):
                raise
        else:
            raise AssertionError('native FA geometry failed to reject the actual signed-int overflow boundary')
        record['native_no_allocation_geometry_probes'] = 'SC117 and last-safe accepted; first-overflow refused'
        verify(root)
        headers.verify()
        if digests != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}:
            raise ValueError('owned component sources changed during compilation')
        record.update(status='compiled-and-imported', module_path=module.__file__,
            module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        for name in ('LICENSE', 'AUTHORS'):
            (output / name).write_bytes((root / name).read_bytes())
    except BaseException as error:
        primary = error
        try:
            record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        except BaseException as diagnostic:
            cleanup[0] = diagnostic
    active = torch.cuda.is_initialized()
    try:
        record['cuda_initialized_after'] = active
        (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    except BaseException as diagnostic:
        cleanup[1] = diagnostic
    if active:
        cleanup[2] = unexpected_cuda
    try:
        failures = [error for error in cleanup if error is not None]
    except BaseException as transport:
        if primary is not None:
            raise primary from transport
        for error in cleanup:
            if error is not None:
                raise error from transport
        raise
    raise_failures(primary, failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='fetch exact inputs if absent; resume verified files')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error('--jobs must be positive')
    root = args.source_root.absolute()
    if args.download:
        download(root)
    verify(root)
    if not args.download_only:
        if args.output is None:
            parser.error('--output is required for compilation')
        compile_component(root, args.output.resolve(), args.jobs)


if __name__ == '__main__':
    main()

