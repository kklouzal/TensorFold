"""Bounded BitBLAS FP16/FP32-accumulation preparation and component diagnostics.

The plan command is CPU-only. Generation and diagnostics explicitly require a
parent-owned GPU lease. Exported libraries retain their source/config/toolchain
identity at persistent paths; diagnostics do not import the compiler packages.
No operation edits the protected runtime, installs packages, or selects serving.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

VERSION = '0.1.0.post1'
WHEEL_SHA256 = '5beaf38899671dc35da63ff31b8d7bea4d98a2c929a102fe40d2f3004cec391b'
PYTHON_COMMIT = '9c9620678d2c292843a8d91506b573ff3d12eebc'
ROWS = (1, 5, 32, 120)
SHAPES = (('gu-reference', 160, 2560), ('down-reference', 2560, 160),
          ('hc-down', 320, 10240), ('hc-inject', 4, 10240), ('hc-up', 10240, 320))
SCHEMA = 'tensorfold.bitblas-fp16-pilot-v1'


def file_identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            data = stream.read(min(remaining, 1 << 20))
            if not data:
                raise ValueError('artifact truncated during hashing')
            digest.update(data)
            remaining -= len(data)
        if stream.read(1):
            raise ValueError('artifact grew during hashing')
        after = os.fstat(stream.fileno())
    def stamp(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if stamp(before) != stamp(after) or stamp(before) != stamp(path.stat()):
        raise ValueError('artifact identity changed during hashing')
    return {'bytes': before.st_size, 'sha256': digest.hexdigest()}


def write_record(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def cases():
    return [{'id': f'{role}-m{rows}', 'role': role, 'M': rows, 'N': n, 'K': k,
             'A_dtype': 'float16', 'W_dtype': 'float16', 'out_dtype': 'float32',
             'accum_dtype': 'float32', 'layout': 'nt', 'propagate_a': False,
             'propagate_b': False, 'with_bias': False}
            for role, n, k in SHAPES for rows in ROWS]


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(right) is dict:
        return left.keys() == right.keys() and all(_same(left[key], value) for key, value in right.items())
    if type(right) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def validate_case(case):
    if not any(_same(case, expected) for expected in cases()):
        raise ValueError('case differs from the finite FP16 pilot matrix')
    return case


def check_abi(source):
    """Require the exact static three-buffer/current-stream generated ABI."""
    match = re.search(r'extern\s+"C"\s+void\s+call\s*\(([^)]*)\)', source)
    if match is None:
        raise ValueError('generated library lacks its explicit void call ABI')
    parameters = [re.sub(r'\s+', ' ', item.strip()) for item in match.group(1).split(',')]
    if len(parameters) != 4:
        raise ValueError('generated call must have three buffers and one stream')
    expected = (r'half_t\s*\*\s*__restrict__\s+\w+', r'half_t\s*\*\s*__restrict__\s+\w+',
                r'float\s*\*\s*__restrict__\s+\w+', r'cudaStream_t\s+stream\s*=\s*cudaStreamDefault')
    if not all(re.fullmatch(pattern, parameter) for pattern, parameter in zip(expected, parameters)):
        raise ValueError('generated call buffer dtype/order or stream ABI differs')
    if re.search(r'extern\s+"C"\s+void\s+init\s*\(\s*\)', source) is None:
        raise ValueError('generated library lacks its explicit initialization ABI')
    return {'init': {'return': 'void', 'arguments': []},
            'call': {'return': 'void', 'arguments': ['A:fp16[M,K]', 'W:fp16[N,K]', 'C:fp32[M,N]', 'cudaStream_t']},
            'layout': 'A[M,K] times transpose(W[N,K])', 'allocation': 'caller-owned buffers',
            'weights': 'original FP16 projections or separately qualified decoded reference; no GGUF reinterpretation'}


def plan(wheel, output):
    wheel = Path(wheel).resolve(strict=True)
    identity = file_identity(wheel)
    if identity['sha256'] != WHEEL_SHA256:
        raise ValueError('official BitBLAS whole-wheel SHA256 differs')
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read('bitblas-0.1.0.post1.dist-info/METADATA').decode()
    if '\nVersion: ' + VERSION + '\n' not in metadata:
        raise ValueError('BitBLAS wheel version metadata differs')
    result = {'schema': SCHEMA, 'phase': 'plan', 'wheel': {'path': str(wheel), **identity},
              'project_python_commit': PYTHON_COMMIT, 'cases': cases(),
              'target': 'single visible physical V100, logical cuda:0 / sm_70',
              'required_toolchain': 'protected CUDA12.9 and CUTLASS4.8.1; no runtime dependency mutation',
              'generation_policy': {'backend': 'tl', 'enable_tuning': False, 'weight_propagation': False,
                                    'compile_during_serving': False},
              'GPU_status': 'UNRUN', 'quality_acceptance': False, 'performance_acceptance': False,
              'scope': 'finite dense/reference components; no grouped expert port or model promotion'}
    write_record(output, result)
    return result


def read_plan(path):
    with Path(path).open(encoding='utf-8') as stream:
        result = json.load(stream)
    if (type(result) is not dict or result.get('schema') != SCHEMA or result.get('phase') != 'plan'
            or result.get('project_python_commit') != PYTHON_COMMIT or not _same(result.get('cases'), cases())
            or not _same(result.get('generation_policy'), {'backend': 'tl', 'enable_tuning': False,
                                                 'weight_propagation': False, 'compile_during_serving': False})
            or result.get('wheel', {}).get('sha256') != WHEEL_SHA256):
        raise ValueError('frozen BitBLAS pilot plan differs')
    return result


def compiler_identity():
    selected = shutil.which('nvcc')
    if selected is None:
        raise ValueError('protected CUDA compiler is unavailable')
    compiler = Path(selected).resolve(strict=True)
    version = subprocess.check_output([str(compiler), '--version'], text=True)
    if 'release 12.9,' not in version:
        raise ValueError('pilot requires the protected CUDA12.9 compiler')
    include = Path(os.environ['TL_CUTLASS_PATH']).resolve(strict=True)
    version_header = include / 'cutlass/version.h'
    text = version_header.read_text()
    expected = {'CUTLASS_MAJOR': 4, 'CUTLASS_MINOR': 8, 'CUTLASS_PATCH': 1}
    if any(re.search(r'#define\s+' + name + r'\s+' + str(value) + r'\b', text) is None
           for name, value in expected.items()):
        raise ValueError('pilot requires explicit protected CUTLASS4.8.1 headers')
    from tensorfold.cuda.sm70.foundation_headers import selected_headers

    foundation = selected_headers()
    if foundation is None or include != foundation.roots[1].resolve(strict=True):
        raise ValueError('BitBLAS include root differs from the protected foundation authority')
    return {'nvcc': {'path': str(compiler), 'version': version, **file_identity(compiler)},
            'cutlass': {'include': str(include), 'version': '4.8.1', 'header': file_identity(version_header)},
            'protected_include_authority': foundation.record()}


def require_device(torch):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (7, 0):
        raise ValueError('pilot requires exactly one visible V100 bound by the parent GPU lease')
    torch.cuda.set_device(0)


def verify_installed_payload(wheel):
    if file_identity(wheel)['sha256'] != WHEEL_SHA256:
        raise ValueError('active-generation wheel differs from the official artifact')
    specification = importlib.util.find_spec('bitblas')
    if specification is None or specification.origin is None:
        raise ValueError('isolated BitBLAS tool prefix is unavailable')
    package = Path(specification.origin).resolve(strict=True).parent
    count, python_files, shared_libraries = 0, 0, 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith('bitblas/') or name.endswith('/'):
                continue
            relative = Path(name).relative_to('bitblas')
            path = package / relative
            data = archive.read(name)
            if path.is_symlink() or file_identity(path) != {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}:
                raise ValueError('installed BitBLAS payload differs: ' + str(relative))
            count += 1
            python_files += int(name.endswith('.py') and '/3rdparty/' not in name)
            shared_libraries += int(name.endswith('.so'))
    if python_files != 133 or shared_libraries != 4:
        raise ValueError('exact BitBLAS source/native payload census differs')
    return {'root': str(package), 'verified_files': count, 'owned_python_files': python_files,
            'bundled_libraries': shared_libraries, 'whole_wheel_sha256': WHEEL_SHA256,
            'owned_python_commit': PYTHON_COMMIT}


def check_dependencies(dynamic_section, *, require_shared=False):
    libraries = re.findall(r'\(NEEDED\).*?\[([^\]]+)\]', dynamic_section)
    if not libraries or any(any(name in library.lower() for name in ('tvm', 'tilelang', 'llvm', 'nvrtc'))
                            for library in libraries):
        raise ValueError('standalone generated runtime must not depend on compiler libraries')
    if require_shared and 'libcudart.so.12' not in libraries:
        raise ValueError('checked generated runtime requires shared CUDA12 cudart')
    return libraries


def compile_shared(source, library, toolchain, package, log, *, timeout_s):
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('parent-owned finite compiler deadline required')
    include = Path(package) / '3rdparty/tilelang/src'
    if not (include / 'tl_templates/cuda/gemm_sm70.h').is_file():
        raise ValueError('verified TileLang SM70 template closure is unavailable')
    command = [toolchain['nvcc']['path'], '-std=c++17', '-Xcudafe', '--diag_suppress=177',
               '-Xcompiler=-fPIC', '-lineinfo', '--shared', '--cudart=shared',
               '-gencode=arch=compute_70,code=sm_70',
               *[flag for root in toolchain['protected_include_authority']['include_roots']
                 for flag in ('-I', root)], '-I', str(include), str(source), '-lcuda',
               '-o', str(library)]
    with Path(log).open('x') as stream:
        subprocess.run(command, check=True, timeout=timeout_s, stdout=stream, stderr=subprocess.STDOUT)
    os.chmod(library, 0o444)
    return command


def runtime_provider():
    path = Path('/usr/local/cuda/lib64/libcudart.so').resolve(strict=True)
    return {'path': str(path), **file_identity(path)}


def generate(plan_path, output, wheel=None, case_id=None, compile_timeout_s=None):
    specification = read_plan(plan_path)
    if type(compile_timeout_s) not in (int, float) or not math.isfinite(compile_timeout_s) or compile_timeout_s <= 0:
        raise ValueError('parent-owned finite compiler deadline required')
    selected = specification['cases'] if case_id is None else [case for case in specification['cases'] if case['id'] == case_id]
    if not selected:
        raise ValueError('requested case is outside the finite pilot matrix')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = verify_installed_payload(wheel or specification['wheel']['path'])
    # The caller uses an isolated tool prefix/image; imports do not install or
    # alter any protected package. No compiler library is needed by diagnostics.
    import torch

    require_device(torch)
    import bitblas

    if bitblas.__version__ != VERSION:
        raise ValueError('active tool prefix has a different BitBLAS version')
    toolchain = compiler_identity()
    manifest = {'schema': SCHEMA, 'phase': 'generated', 'plan': file_identity(plan_path),
                'source_tool': file_identity(__file__), 'installed_payload': payload, 'BitBLAS': VERSION, 'project_python_commit': PYTHON_COMMIT,
                'toolchain': toolchain, 'torch': str(torch.__version__), 'cases': [],
                'requested_case_ids': [case['id'] for case in selected], 'complete_matrix': len(selected) == len(cases()),
                'quality_acceptance': False, 'performance_acceptance': False,
                'scope': 'persistent source/config/ABI-bound generated dense libraries; native diagnostics pending'}
    for case in selected:
        directory = output / case['id']
        directory.mkdir()
        write_record(directory / 'started.json', {'case': case, 'state': 'generating'})
        options = {key: value for key, value in case.items() if key not in ('id', 'role')}
        operation = bitblas.Matmul(bitblas.MatmulConfig(**options), target='cuda -arch=sm_70',
                                   enable_tuning=False, backend='tl')
        if (operation.arch.sm_version != 70 or str(operation.target.arch) != 'sm_70'
                or operation.lib is None or operation.workspace is not None
                or operation.propagate_a or operation.propagate_b):
            raise ValueError('prepared BitBLAS native library or no-workspace contract differs')
        source = operation.get_source()
        abi = check_abi(source)
        (directory / 'generated.cu').write_text(source)
        shutil.copyfile(operation.libpath, directory / 'upstream-static-reference.so')
        os.chmod(directory / 'upstream-static-reference.so', 0o444)
        if compile_timeout_s is None:
            raise ValueError('shared export requires the parent compiler deadline')
        command = compile_shared(directory / 'generated.cu', directory / 'library.so',
                                 toolchain, payload['root'], directory / 'compile-shared.log',
                                 timeout_s=compile_timeout_s)
        dependencies = subprocess.check_output(['readelf', '-d', str(directory / 'library.so')], text=True)
        (directory / 'dynamic-section.txt').write_text(dependencies)
        needed = check_dependencies(dependencies, require_shared=True)
        config_sha256 = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        record = {'case': case, 'config_sha256': config_sha256, 'abi': abi, 'needed_libraries': needed,
                  'runtime_policy': 'shared-cuda12.9-checked', 'runtime_provider': runtime_provider(),
                  'reference_library': file_identity(directory / 'upstream-static-reference.so'),
                  'compile_command': command, 'source': file_identity(directory / 'generated.cu'),
                  'library': file_identity(directory / 'library.so'),
                  'dynamic_section': file_identity(directory / 'dynamic-section.txt'),
                  'export_policy': 'persistent read-only artifact; runtime does not import BitBLAS/TVM',
                  'native_execution': 'UNRUN', 'graph_execution': 'UNRUN'}
        write_record(directory / 'artifact.json', record)
        manifest['cases'].append(record)
    if not _same(compiler_identity(), toolchain):
        raise ValueError('protected compiler/header authority changed during generation')
    write_record(output / 'manifest.json', manifest)
    return manifest


def validate_subset(records):
    if type(records) is not list or not records or any(type(record) is not dict for record in records):
        raise ValueError('nonempty generated case records required')
    actual = [record.get('case') for record in records]
    expected = [case for case in cases() if any(_same(case, value) for value in actual)]
    if not _same(actual, expected):
        raise ValueError('generated case subset is reordered, duplicated or outside the finite matrix')
    return actual


def relink_shared(manifest_path, output, *, wheel, compile_timeout_s):
    """CPU-compiler-only final export from an authenticated generation record."""
    manifest_path = Path(manifest_path)
    original = json.loads(manifest_path.read_text())
    if (original.get('schema') != SCHEMA or original.get('phase') != 'generated'
            or original.get('BitBLAS') != VERSION or original.get('project_python_commit') != PYTHON_COMMIT):
        raise ValueError('exact generated BitBLAS reference manifest required')
    validate_subset(original.get('cases'))
    payload, toolchain = verify_installed_payload(wheel), compiler_identity()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {**original, 'cases': [], 'source_tool': file_identity(__file__),
              'relinked_from': file_identity(manifest_path), 'toolchain': toolchain,
              'installed_payload': payload, 'scope': 'shared-runtime final export; native execution and graphs UNRUN'}
    for old in original['cases']:
        validate_case(old['case'])
        source_dir = manifest_path.parent / old['case']['id']
        if file_identity(source_dir / 'generated.cu') != old['source'] or file_identity(source_dir / 'library.so') != old['library']:
            raise ValueError('reference generated source or library content changed')
        directory = output / old['case']['id']
        directory.mkdir()
        shutil.copyfile(source_dir / 'generated.cu', directory / 'generated.cu')
        shutil.copyfile(source_dir / 'library.so', directory / 'upstream-static-reference.so')
        os.chmod(directory / 'upstream-static-reference.so', 0o444)
        abi = check_abi((directory / 'generated.cu').read_text())
        command = compile_shared(directory / 'generated.cu', directory / 'library.so', toolchain,
                                 payload['root'], directory / 'compile-shared.log', timeout_s=compile_timeout_s)
        section = subprocess.check_output(['readelf', '-d', str(directory / 'library.so')], text=True)
        (directory / 'dynamic-section.txt').write_text(section)
        record = {**old, 'abi': abi, 'library': file_identity(directory / 'library.so'),
                  'reference_library': old['library'], 'needed_libraries': check_dependencies(section, require_shared=True),
                  'runtime_policy': 'shared-cuda12.9-checked', 'runtime_provider': runtime_provider(),
                  'dynamic_section': file_identity(directory / 'dynamic-section.txt'), 'compile_command': command,
                  'native_execution': 'UNRUN', 'graph_execution': 'UNRUN'}
        write_record(directory / 'artifact.json', record)
        result['cases'].append(record)
    if not _same(compiler_identity(), toolchain):
        raise ValueError('protected compiler/header authority changed during relink')
    write_record(output / 'manifest.json', result)
    return result


@dataclass(frozen=True)
class PreparedLibrary:
    native: object
    cudart: object
    dimensions: tuple[int, int, int]
    runtime_provider: dict


def _cuda_status(cudart):
    status = cudart.cudaGetLastError()
    if status:
        message = cudart.cudaGetErrorString(status)
        raise RuntimeError('prebuilt CUDA launch failed: ' + (message.decode() if message else str(status)))


def checked_runtime(library, expected):
    path = Path(expected['path']).resolve(strict=True)
    if str(path) != runtime_provider()['path']:
        raise ValueError('runtime provider differs from the protected CUDA SDK path')
    if file_identity(path) != {key: expected[key] for key in ('bytes', 'sha256')}:
        raise ValueError('loaded CUDA runtime provider content differs')
    functions = (library.cudaRuntimeGetVersion, library.cudaGetLastError, library.cudaGetErrorString)
    with open('/proc/self/maps', encoding='utf-8') as stream:
        mappings = [line.rstrip('\n').split(None, 5) for line in stream]
    for function in functions:
        address = ctypes.cast(function, ctypes.c_void_p).value
        found = False
        for fields in mappings:
            begin, end = (int(value, 16) for value in fields[0].split('-'))
            if begin <= address < end:
                if (len(fields) != 6 or 'x' not in fields[1] or fields[5].endswith(' (deleted)')
                        or Path(fields[5]).resolve(strict=True) != path):
                    raise ValueError('CUDA error function does not belong to the declared live SDK runtime')
                found = True
                break
        if not found:
            raise ValueError('CUDA runtime function mapping is unavailable')
    library.cudaRuntimeGetVersion.argtypes, library.cudaRuntimeGetVersion.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    version = ctypes.c_int()
    if library.cudaRuntimeGetVersion(ctypes.byref(version)) != 0 or version.value != 12090:
        raise ValueError('prebuilt pilot requires the protected CUDA12.9 runtime')
    library.cudaGetLastError.argtypes, library.cudaGetLastError.restype = [], ctypes.c_int
    library.cudaGetErrorString.argtypes, library.cudaGetErrorString.restype = [ctypes.c_int], ctypes.c_char_p
    return {'path': str(path), **file_identity(path), 'binding': 'actual-error-function-executable-VMA'}



def load_artifact(directory, record):
    directory = Path(directory)
    validate_case(record['case'])
    expected_config = hashlib.sha256(json.dumps(record['case'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if record.get('config_sha256') != expected_config:
        raise ValueError('prepared artifact config identity differs')
    for filename, key in [('generated.cu', 'source'), ('library.so', 'library')]:
        path = directory / filename
        if path.is_symlink() or file_identity(path) != record[key]:
            raise ValueError('prepared artifact content differs')
    if check_abi((directory / 'generated.cu').read_text()) != record['abi']:
        raise ValueError('prepared artifact ABI differs')
    if record.get('runtime_policy') != 'shared-cuda12.9-checked' or 'libcudart.so.12' not in record.get('needed_libraries', []):
        raise ValueError('static unchecked reference library cannot run through the checked pilot ABI')
    library = ctypes.CDLL(str((directory / 'library.so').resolve(strict=True)))
    library.init.argtypes, library.init.restype = [], None
    library.call.argtypes, library.call.restype = [ctypes.c_void_p] * 4, None
    provider = checked_runtime(library, record['runtime_provider'])
    library.init()
    _cuda_status(library)
    return PreparedLibrary(library, library, tuple(record['case'][key] for key in ('M', 'N', 'K')), provider)



def call_library(library, a, w, output, torch):
    """Borrow prepared code and operands through positive stream/graph retirement.

    The caller retains every object, including this library, through completion
    and any capture/replay lifetime. The helper neither allocates nor retires it.
    """
    if type(library) is not PreparedLibrary:
        raise TypeError('prepared static library binding required')
    rows, columns, reduction = library.dimensions
    if (type(a) is not torch.Tensor or type(w) is not torch.Tensor or type(output) is not torch.Tensor
            or a.dtype != torch.float16 or w.dtype != torch.float16 or output.dtype != torch.float32
            or a.shape != (rows, reduction) or w.shape != (columns, reduction)
            or output.shape != (rows, columns) or a.device.type != 'cuda' or a.device != w.device
            or a.device != output.device or any(not value.is_contiguous() or value.requires_grad or value.is_neg() or value.is_conj()
                                              for value in (a, w, output))):
        raise ValueError('prebuilt call requires exact contiguous Half/Half/Float CUDA matrices')
    if output.is_inference() and not torch.is_inference_mode_enabled():
        raise RuntimeError('prebuilt inference output mutated outside inference mode')
    begin, end = output.data_ptr(), output.data_ptr() + output.numel() * output.element_size()
    if any(begin < value.data_ptr() + value.numel() * value.element_size() and value.data_ptr() < end for value in (a, w)):
        raise ValueError('prebuilt output aliases a read operand')
    with torch.cuda.device(a.device):
        stream = torch.cuda.current_stream(device=a.device).cuda_stream
        library.native.call(a.data_ptr(), w.data_ptr(), output.data_ptr(), stream)
        _cuda_status(library.cudart)
    torch.autograd.graph.increment_version(output)
    return output


def numerical_observation(actual, expected, torch):
    unequal = actual.view(torch.int32) != expected.view(torch.int32)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    maximum = torch.max(torch.where(finite, torch.abs(actual - expected), 0)).item()
    return {'elements': actual.numel(), 'unequal_bits': int(unequal.sum().item()),
            'actual_nonfinite': int((~torch.isfinite(actual)).sum().item()),
            'expected_nonfinite': int((~torch.isfinite(expected)).sum().item()),
            'max_absolute_finite': float(maximum), 'quality_acceptance': False}


def diagnose(manifest_path, output):
    # Explicit GPU component diagnostics, never a benchmark or serving choice.
    import torch
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense

    require_device(torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('phase') != 'generated':
        raise ValueError('generated BitBLAS manifest required')
    available = validate_subset(manifest.get('cases'))
    if manifest.get('requested_case_ids') != [case['id'] for case in available]:
        raise ValueError('generated case subset identity differs')
    reports, regroup = [], {}
    for index, (role, n, k) in enumerate(SHAPES):
        selected_rows = [rows for rows in reversed(ROWS) if any(record['case']['role'] == role and record['case']['M'] == rows for record in manifest['cases'])]
        if not selected_rows:
            continue
        generator = torch.Generator(device='cuda').manual_seed(81721 + index)
        inputs = torch.randn((max(selected_rows), k), device='cuda', dtype=torch.float16, generator=generator) / 32
        weights = torch.randn((n, k), device='cuda', dtype=torch.float16, generator=generator) / 32
        longest = None
        for rows in selected_rows:
            record = next(item for item in manifest['cases'] if item['case']['role'] == role and item['case']['M'] == rows)
            case = record['case']
            library = load_artifact(manifest_path.parent / case['id'], record)
            a, w = inputs[:rows].clone(), weights.clone()
            reference = GGUFDense(w)
            result = torch.empty((rows, n), dtype=torch.float32, device='cuda')
            call_library(library, a, w, result, torch)
            expected = reference.matmul_reference(a, f32=True)
            observed = numerical_observation(result, expected, torch)
            if longest is None:
                longest = result.clone()
                regroup[case['id']] = {'status': 'baseline-only-no-regroup-comparison', 'reference_rows': rows}
            else:
                regroup[case['id']] = {'reference_rows': longest.shape[0],
                                       **numerical_observation(result, longest[:rows], torch)}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call_library(library, a, w, result, torch)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call_library(library, a, w, result, torch)
            a.add_(torch.tensor(2**-8, device='cuda', dtype=torch.float16))
            w.mul_(2.)
            graph.replay()
            captured = result.clone()
            call_library(library, a, w, result, torch)
            torch.cuda.synchronize()
            graph_equal = torch.equal(captured.view(torch.int32), result.view(torch.int32))
            if not graph_equal:
                raise ValueError('changed-input graph replay differs from same-library eager execution')
            report = {'case': case, 'against_tensorfold_reference': observed,
                      'changed_input_and_weight_graph_bitwise': graph_equal, 'actual_output_dtype': str(result.dtype),
                      'loaded_runtime_provider': library.runtime_provider}
            write_record(output / (case['id'] + '.json'), report)
            reports.append(report)
            del graph
        torch.cuda.synchronize()
    result = {'schema': SCHEMA, 'phase': 'diagnostic', 'manifest': file_identity(manifest_path),
              'source_tool': file_identity(__file__), 'reports': reports, 'row_regrouping': regroup,
              'complete_matrix': len(available) == len(cases()),
              'positive_device_retirement': True, 'quality_acceptance': False, 'performance_acceptance': False,
              'scope': 'synthetic native/current-stream/graph and numerical observations; no timing or model acceptance'}
    write_record(output / 'diagnostic.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    selected = subparsers.add_parser('plan')
    selected.add_argument('--wheel', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    selected = subparsers.add_parser('generate')
    selected.add_argument('--plan', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    selected.add_argument('--wheel', type=Path, help='relocated identical official wheel for installed-payload verification')
    selected.add_argument('--compile-timeout-s', type=float, required=True, help='parent-owned compiler deadline')
    selected.add_argument('--case', choices=[case['id'] for case in cases()], help='explicit bounded generation subset; omitted selects the finite full matrix')
    selected = subparsers.add_parser('relink-shared')
    selected.add_argument('--manifest', type=Path, required=True)
    selected.add_argument('--wheel', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    selected.add_argument('--compile-timeout-s', type=float, required=True)
    selected = subparsers.add_parser('diagnose')
    selected.add_argument('--manifest', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        result = plan(args.wheel, args.output)
    elif args.command == 'generate':
        result = generate(args.plan, args.output, wheel=args.wheel, case_id=args.case, compile_timeout_s=args.compile_timeout_s)
    elif args.command == 'relink-shared':
        result = relink_shared(args.manifest, args.output, wheel=args.wheel, compile_timeout_s=args.compile_timeout_s)
    else:
        result = diagnose(args.manifest, args.output)
    print(json.dumps({'schema': result['schema'], 'phase': result['phase'],
                      'cases': len(result.get('cases', result.get('reports', []))),
                      'quality_acceptance': False, 'performance_acceptance': False}, sort_keys=True))


if __name__ == '__main__':
    main()
