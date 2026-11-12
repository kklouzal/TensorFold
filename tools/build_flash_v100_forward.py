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
import urllib.request

# One stdlib authority shared with host validation and the cold runtime loader.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tensorfold.cuda.sm70.flash_v100_artifact import PIN, INPUTS, CUDA_FLAGS



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
    sources = [owned / 'flash_v100_forward.cpp', owned / 'flash_v100_forward.cu', root / 'kernel/forward.cu', root / 'kernel/forward_varlen.cu']
    files = [*sources, owned / 'flash_v100_forward.cuh', owned / 'prefill_dense_contracts.cuh', Path(__file__).resolve(),
        owned / 'flash_v100_NOTICE.txt', owned / 'flash_v100_LICENSE.txt']
    digests = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    options = dict(extra_cflags=['-O3'],
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=[*map(str, headers.roots), str(root / 'include')])
    record = {'scope': 'standalone dense+compacted-varlen controlled component compile/import; no JIT/provider/model qualification',
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
        for batch in (1, 256):
            assert module.compacted_workspace_bytes(batch) == 2 * batch * 2051 * 256 * 2 + 4 * (3 * batch + 2)
        for batch in (0, 257):
            try:
                module.compacted_workspace_bytes(batch)
            except RuntimeError:
                pass
            else:
                raise AssertionError('native compacted geometry accepted unsupported row envelope')
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


def seal_component(root, result, output):
    """Publish a new sealed directory from an actual compile receipt; never rebuild or install."""
    from tensorfold.cuda.sm70 import flash_v100_artifact as artifact
    import re

    verify(root)
    result = result.absolute()
    result_identity, raw = artifact._acquire(result, limit=1 << 20, capture=True)
    record = json.loads(raw)
    if (record.get('status') != 'compiled-and-imported' or record.get('upstream_commit') != PIN
            or record.get('foreign_inputs') != {k: list(v) for k, v in INPUTS.items()}
            or record.get('cuda_initialized_before') is not False or record.get('cuda_initialized_after') is not False):
        raise ValueError('sealing requires the genuine GPU-hidden exact-pin compile receipt')
    module_path = Path(record['module_path'])
    name = re.fullmatch(artifact.NATIVE_NAME_PATTERN, module_path.name)
    if name is None or module_path.parent.name != 'build-' + name.group(1):
        raise ValueError('sealing native path differs from the owned compile layout')
    native = result.parent / module_path.parent.name / module_path.name
    native_identity = artifact.file_record(native)
    if native_identity['sha256'] != record['module_sha256']:
        raise ValueError('sealing native ELF differs from compile receipt')
    owned = Path(__file__).resolve().parents[1] / 'src/tensorfold/cuda/sm70'
    for name in artifact.NATIVE_INPUTS:
        compiled = [sha for path, sha in record['owned_source_sha256'].items() if Path(path).name == name]
        if compiled != [artifact.file_record(owned / name)['sha256']]:
            raise ValueError('sealing compiled native input differs from current owned source: ' + name)
    abi = artifact.runtime_abi()
    if (abi['torch'] != record['torch'] or abi['cuda'] != record['torch_cuda']
            or abi['headers'] != record['SDK_headers']):
        raise ValueError('sealing SDK differs from actual compiled native ABI')
    producers = [sha for path, sha in record['owned_source_sha256'].items()
                 if Path(path).name == 'build_flash_v100_forward.py']
    if len(producers) != 1:
        raise ValueError('sealing requires the original compile producer identity')
    if output.exists():
        raise FileExistsError('seal destination must be a new immutable artifact directory')
    output.mkdir(parents=True)
    def copy(source, target, expected):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            observed, _ = artifact._acquire(source, expected_bytes=expected['bytes'], sink=stream)
            if observed != expected:
                raise ValueError('seal source changed during bounded copy: ' + str(source))
        target.chmod(0o444)
    for name, (size, digest) in INPUTS.items():
        copy(root / name, output / 'foreign' / name, {'bytes': size, 'sha256': digest})
    sources = {}
    for name in artifact.OWNED:
        sources[name] = artifact.file_record(artifact.owned_path(name))
        copy(artifact.owned_path(name), output / 'owned' / name, sources[name])
    copy(native, output / 'native' / native.name, native_identity)
    copy(result, output / 'build.json', result_identity)
    producer_identity = artifact.file_record(Path(__file__).resolve())
    copy(Path(__file__).resolve(), output / 'producer' / 'build_flash_v100_forward.py', producer_identity)
    data = {'schema': 1, 'family': artifact.FAMILY, 'foreign_pin': PIN,
        'foreign_inputs': {k: list(v) for k, v in INPUTS.items()}, 'owned_sources': sources,
        'native': {'name': native.name, **native_identity}, 'abi': abi,
        'abi_origin': artifact.ABI_ORIGIN, 'compile_producer_sha256': producers[0],
        'build': result_identity, 'options': record['requested_options'],
        'producer': producer_identity}
    data['profile_sha256'] = hashlib.sha256(artifact.canonical(data)).hexdigest()
    manifest = output / 'manifest.json'
    manifest_raw = artifact.canonical(data) + b'\n'
    manifest.write_bytes(manifest_raw); manifest.chmod(0o444)
    digest = hashlib.sha256(manifest_raw).hexdigest()
    (output / 'manifest.sha256').write_text(digest + '\n')
    (output / 'manifest.sha256').chmod(0o444)
    verified = artifact.validate_manifest(manifest, digest)
    for path in sorted(output.rglob('*'), reverse=True):
        if path.is_dir(): path.chmod(0o555)
    output.chmod(0o555)
    return verified.record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='fetch exact inputs if absent; resume verified files')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seal-from', type=Path, help='seal an existing genuine compile result without rebuilding')
    parser.add_argument('--seal-output', type=Path, help='new immutable operator artifact directory')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error('--jobs must be positive')
    root = args.source_root.absolute()
    if args.download:
        download(root)
    verify(root)
    if args.seal_from is not None:
        if args.seal_output is None or args.download_only or args.output is not None:
            parser.error('--seal-from requires --seal-output and excludes compile/download-only')
        print(json.dumps(seal_component(root, args.seal_from, args.seal_output.absolute()), sort_keys=True))
        return
    if args.seal_output is not None:
        parser.error('--seal-output requires --seal-from')
    if not args.download_only:
        if args.output is None:
            parser.error('--output is required for compilation')
        compile_component(root, args.output.resolve(), args.jobs)


if __name__ == '__main__':
    main()

