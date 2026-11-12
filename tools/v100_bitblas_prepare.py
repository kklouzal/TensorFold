"""Bounded BitBLAS FP16/FP32-accumulation preparation and component diagnostics.

The plan command is CPU-only. Generation and diagnostics explicitly require a
parent-owned GPU lease. Exported libraries retain their source/config/toolchain
identity at persistent paths; diagnostics do not import the compiler packages.
No operation edits the protected runtime, installs packages, or selects serving.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

VERSION = '0.1.0.post1'
WHEEL_SHA256 = '5beaf38899671dc35da63ff31b8d7bea4d98a2c929a102fe40d2f3004cec391b'
PYTHON_COMMIT = '9c9620678d2c292843a8d91506b573ff3d12eebc'
ROWS = (1, 5, 32, 120)
SHAPES = (('gu-reference', 160, 2560), ('down-reference', 2560, 160),
          ('hc-down', 320, 10240), ('hc-inject', 4, 10240), ('hc-up', 10240, 320))
SCHEMA = 'tensorfold.bitblas-fp16-padded-pilot-v2'
PADDING_POLICY = 'default-volta-simt-n8-k128-v2'


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


def physical_case(case):
    """Use explicit zero tails where the default GEMV emitter has no masks."""
    validate_case(case)
    physical = dict(case)
    simt = case['M'] < 8 or case['N'] < 16 or case['K'] < 16
    if simt:
        physical['N'] = (case['N'] + 7) // 8 * 8
        physical['K'] = (case['K'] + 127) // 128 * 128
    return physical


def padding_storage(case):
    physical = physical_case(case)
    logical_a, logical_w, logical_c = case['M'] * case['K'] * 2, case['N'] * case['K'] * 2, case['M'] * case['N'] * 4
    a, w, c = physical['M'] * physical['K'] * 2, physical['N'] * physical['K'] * 2, physical['M'] * physical['N'] * 4
    return {'logical_bytes': {'A': logical_a, 'W': logical_w, 'C': logical_c},
            'physical_bytes': {'A': a, 'W': w, 'C': c},
            'zero_tail_bytes': a - logical_a + w - logical_w + c - logical_c,
            'additional_payload_bytes': (a if a != logical_a else 0) + (w if w != logical_w else 0) + (c if c != logical_c else 0),
            'input_copy_per_call': (physical['M'], physical['K']) != (case['M'], case['K']),
            'logical_output_copy_per_call': (physical['M'], physical['N']) != (case['M'], case['N']),
            'weight_padding': 'once during preparation; zero tails retained unchanged',
            'input_zero_tail': 'initialized during preparation; only logical values copied during calls'}


def geometry_record(case, physical, source):
    """Finite default-emitter geometry record, not a general CUDA proof system.

    The actual source lines and predicates remain attached for independent
    review. Native memcheck/value/capture gates are separate requirements.
    """
    launch = re.findall(r'<<<dim3\((\d+), (\d+), (\d+)\), dim3\((\d+), (\d+), (\d+)\), (\d+), stream>>>', source)
    if len(launch) != 1:
        raise ValueError('unrecognized default-emitter launch requires a new bounds review')
    values = tuple(map(int, launch[0]))
    m, n, k = (physical[key] for key in ('M', 'N', 'K'))
    simt = m < 8 or n < 16 or k < 16
    if simt:
        expected = (n // 8, m, 1, 16, 8, 1, 0)
        if n % 8 or k % 128 or values != expected or 'tl::gemm_ss<' in source:
            raise ValueError('unsafe or unrecognized default GEMV physical geometry')
        reasoning = {'family': 'default-GEMV',
            'domains': {'bx': [0, n//8-1], 'by': [0, m-1], 'tx': [0, 15], 'ty': [0, 7], 'ko': [0, k//128-1], 'lane': [0, 7]},
            'A': {'formula': 'by*K + ko*128 + tx*8 + lane', 'range': [0, m*k-1]},
            'B': {'formula': '(bx*8+ty)*K + ko*128 + tx*8 + lane', 'range': [0, n*k-1]},
            'C': {'formula': 'by*N + bx*8 + ty; tx==0 writes', 'range': [0, m*n-1]},
            'write_ownership': '(by,bx,ty) uniquely owns each physical output element',
            'alignment_bytes': {'A': 16, 'W': 16, 'C': 4},
            'local_arrays': 'A_local/B_local[8] use vector8 at0 and ki0..7; accum/reduction arrays[1] use index0'}
    else:
        expected = ((n+63)//64, (m+63)//64, 1, 128, 1, 1, 16384)
        if k % 32 or m % 8 or n % 32 or values != expected or 'tl::gemm_ss<64, 64, 32, 2, 2, 0, 1>' not in source:
            raise ValueError('unrecognized Volta tile geometry requires a new bounds review')
        outputs = set()
        for tx in range(128):
            for i in range(16):
                for vec in range(2):
                    row = (tx>>6)*32+((tx&31)>>4)*16+((tx&7)>>2)*8+((i&7)>>2)*4+(i&1)*2+(tx&1)
                    column = ((tx&63)>>5)*32+((i&3)>>1)*16+((tx&15)>>3)*8+(i>>3)*4+((tx&3)>>1)*2+vec
                    outputs.add((row, column))
        if outputs != {(row, column) for row in range(64) for column in range(64)}:
            raise ValueError('default Volta output permutation does not cover its tile uniquely')
        reasoning = {'family': 'Volta64x64x32',
            'A': {'row': 'by*64 + i*16 + tx//8', 'column': 'ko*32 + (tx%8)*4 + vector_lane0..3',
                  'row_guard_required': m % 64 != 0, 'required_guard_meaning': 'floor(row/8) < M/8', 'range_after_guard': [0, m*k-1]},
            'B': {'row': 'bx*64 + i*16 + tx//8', 'column': 'ko*32 + (tx%8)*4 + vector_lane0..3',
                  'row_guard_required': n % 64 != 0, 'required_guard_meaning': 'floor(row/32) < N/32', 'range_after_guard': [0, n*k-1]},
            'C': {'tile_permutation': '4096 thread/i/vector mappings form a bijection over64x64',
                  'required_tail_predicates': 'row-group8 < M/8 and column-group32 < N/32 where tails exist', 'range_after_guard': [0, m*n-1]},
            'shared': 'two A tiles occupy Half0..4095, two B tiles Half4096..8191; total16384bytes',
            'local_arrays': 'C_local[32] uses i0..15/vector0..1; index=i*2+vector',
            'alignment_bytes': {'A': 8, 'W': 8, 'C': 4},
            'review_requirement': 'Check the attached actual M/N predicates against these finite formulas; no automatic mask assumption.'}
    if f'matmul_m{m}n{n}k{k}_f16xf16' not in source:
        raise ValueError('generated kernel configuration name differs from its physical geometry')
    accesses = [line.strip() for line in source.splitlines() if re.search(r'\b[ABC]\s*(?:\+|\[)', line)]
    predicates = [line.strip() for line in source.splitlines() if line.strip().startswith('if (')]
    if not accesses:
        raise ValueError('default emitter global access evidence is unavailable')
    return {'logical_case': case, 'physical_case': physical, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'launch': {'grid': list(values[:3]), 'block': list(values[3:6]), 'shared_bytes': values[6]},
            'finite_reasoning': reasoning, 'actual_global_access_lines': accesses, 'actual_predicate_lines': predicates,
            'status': 'finite_geometry_record_requires_independent_source_review_and_native_memcheck',
            'native_memory_acceptance': False}


def config_identity(case):
    value = {'logical': case, 'physical': physical_case(case), 'padding_policy': PADDING_POLICY}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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
              'physical_cases': [physical_case(case) for case in cases()], 'padding_policy': PADDING_POLICY,
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
            or result.get('padding_policy') != PADDING_POLICY
            or not _same(result.get('physical_cases'), [physical_case(case) for case in cases()])
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
    return _runtime_library().runtime_provider()


def generate(plan_path, output, wheel=None, case_ids=None, compile_timeout_s=None):
    specification = read_plan(plan_path)
    if type(compile_timeout_s) not in (int, float) or not math.isfinite(compile_timeout_s) or compile_timeout_s <= 0:
        raise ValueError('parent-owned finite compiler deadline required')
    if case_ids is not None and (type(case_ids) is not list or not case_ids
                                 or any(type(name) is not str or name not in {case['id'] for case in cases()} for name in case_ids)
                                 or len(set(case_ids)) != len(case_ids)):
        raise ValueError('requested cases must be a finite unique declared subset')
    selected = specification['cases'] if case_ids is None else [case for case in specification['cases'] if case['id'] in case_ids]
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
        physical = physical_case(case)
        options = {key: value for key, value in physical.items() if key not in ('id', 'role')}
        operation = bitblas.Matmul(bitblas.MatmulConfig(**options), target='cuda -arch=sm_70',
                                   enable_tuning=False, backend='tl')
        if (operation.arch.sm_version != 70 or str(operation.target.arch) != 'sm_70'
                or operation.lib is None or operation.workspace is not None
                or operation.propagate_a or operation.propagate_b):
            raise ValueError('prepared BitBLAS native library or no-workspace contract differs')
        source = operation.get_source()
        abi = check_abi(source)
        bounds = geometry_record(case, physical, source)
        write_record(directory / 'bounds.json', bounds)
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
        config_sha256 = config_identity(case)
        record = {'case': case, 'physical_case': physical, 'padding_policy': PADDING_POLICY,
                  'padding_storage': padding_storage(case), 'bounds': file_identity(directory / 'bounds.json'),
                  'config_sha256': config_sha256, 'abi': abi, 'needed_libraries': needed,
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
        source = (directory / 'generated.cu').read_text()
        abi = check_abi(source)
        bounds = geometry_record(old['case'], physical_case(old['case']), source)
        write_record(directory / 'bounds.json', bounds)
        command = compile_shared(directory / 'generated.cu', directory / 'library.so', toolchain,
                                 payload['root'], directory / 'compile-shared.log', timeout_s=compile_timeout_s)
        section = subprocess.check_output(['readelf', '-d', str(directory / 'library.so')], text=True)
        (directory / 'dynamic-section.txt').write_text(section)
        record = {**old, 'abi': abi, 'bounds': file_identity(directory / 'bounds.json'),
                  'library': file_identity(directory / 'library.so'),
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


def assemble_reviewed(old_manifest, corrected_manifest, output, *, wheel, compile_timeout_s):
    """CPU-only recompilation/copy assembly; never relabel old data tests."""
    old_path, corrected_path = Path(old_manifest), Path(corrected_manifest)
    old, corrected = json.loads(old_path.read_text()), json.loads(corrected_path.read_text())
    if (old.get('schema') != 'tensorfold.bitblas-fp16-pilot-v1'
            or corrected.get('schema') != SCHEMA or old.get('BitBLAS') != VERSION
            or corrected.get('BitBLAS') != VERSION
            or old.get('project_python_commit') != PYTHON_COMMIT
            or corrected.get('project_python_commit') != PYTHON_COMMIT):
        raise ValueError('exact historical/corrected source manifests required')
    if not _same(validate_subset(old['cases']), cases()):
        raise ValueError('historical twenty-case source census differs')
    validate_subset(corrected['cases'])
    corrected_by_id = {record['case']['id']: record for record in corrected['cases']}
    changed = {case['id'] for case in cases() if not _same(case, physical_case(case))}
    if set(corrected_by_id) != changed:
        raise ValueError('corrected export set must contain exactly the eight changed geometries')
    payload, toolchain = verify_installed_payload(wheel), compiler_identity()
    if not _same(old['toolchain'], toolchain) or not _same(corrected['toolchain'], toolchain):
        raise ValueError('source SDK/compiler/header provenance differs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {**corrected, 'cases': [], 'complete_matrix': True,
              'requested_case_ids': [case['id'] for case in cases()], 'source_tool': file_identity(__file__),
              'assembly_inputs': {'old': file_identity(old_path), 'corrected': file_identity(corrected_path)},
              'scope': 'CPU recompiled12reviewed unchanged sources plus8exact corrected exports; all20native memory/value/graph gates pending',
              'quality_acceptance': False, 'performance_acceptance': False}
    for previous in old['cases']:
        case = previous['case']
        copied = case['id'] in changed
        original = corrected_by_id[case['id']] if copied else previous
        manifest_path = corrected_path if copied else old_path
        source_dir = manifest_path.parent / case['id']
        for filename, key in [('generated.cu', 'source'), ('library.so', 'library')]:
            if file_identity(source_dir / filename) != original[key]:
                raise ValueError('reviewed original source/binary identity differs')
        directory = output / case['id']
        directory.mkdir()
        shutil.copyfile(source_dir / 'generated.cu', directory / 'generated.cu')
        source = (directory / 'generated.cu').read_text()
        physical = physical_case(case)
        bounds = geometry_record(case, physical, source)
        write_record(directory / 'bounds.json', bounds)
        if copied:
            if (original.get('padding_policy') != PADDING_POLICY
                    or not _same(original.get('physical_case'), physical)
                    or original.get('runtime_policy') != 'shared-cuda12.9-checked'):
                raise ValueError('corrected source physical/runtime contract differs')
            shutil.copyfile(source_dir / 'library.so', directory / 'library.so')
            os.chmod(directory / 'library.so', 0o444)
            command = original['compile_command']
        else:
            command = compile_shared(directory / 'generated.cu', directory / 'library.so', toolchain,
                                     payload['root'], directory / 'compile-shared.log', timeout_s=compile_timeout_s)
        section = subprocess.check_output(['readelf', '-d', str(directory / 'library.so')], text=True)
        (directory / 'dynamic-section.txt').write_text(section)
        record = {'case': case, 'physical_case': physical, 'padding_policy': PADDING_POLICY,
                  'padding_storage': padding_storage(case), 'config_sha256': config_identity(case),
                  'abi': check_abi(source), 'bounds': file_identity(directory / 'bounds.json'),
                  'needed_libraries': check_dependencies(section, require_shared=True),
                  'runtime_policy': 'shared-cuda12.9-checked', 'runtime_provider': runtime_provider(),
                  'source': file_identity(directory / 'generated.cu'), 'library': file_identity(directory / 'library.so'),
                  'dynamic_section': file_identity(directory / 'dynamic-section.txt'), 'compile_command': command,
                  'provenance': {'action': 'copied exact corrected shared export' if copied else 'CPU recompiled exact reviewed historical CUDA source',
                                 'original_manifest': file_identity(manifest_path), 'original_directory': str(source_dir.resolve()),
                                 'original_source': original['source'], 'original_library': original['library'],
                                 'old_native_results_reused': False},
                  'export_policy': 'persistent read-only source/config-bound artifact; no TVM runtime dependency',
                  'native_execution': 'UNRUN', 'graph_execution': 'UNRUN', 'native_memory_acceptance': False}
        write_record(directory / 'artifact.json', record)
        result['cases'].append(record)
    if not _same(compiler_identity(), toolchain):
        raise ValueError('protected compiler/header authority changed during CPU assembly')
    write_record(output / 'manifest.json', result)
    return result


def _runtime_library():
    # Installed SDK execution uses its owned module. A standalone checkout's
    # CPU tools may not have TensorFold installed; load the same stdlib source
    # directly, without importing Torch or changing global DLL search paths.
    import importlib
    import sys
    source=Path(__file__).resolve().parents[1]/'src/tensorfold/cuda/sm70/bitblas_library.py'
    if not source.is_file():
        return importlib.import_module('tensorfold.cuda.sm70.bitblas_library')
    name='_tensorfold_bitblas_library_host'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]


def __getattr__(name):
    if name in ('PreparedLibrary','_cuda_status','ctypes'):
        return getattr(_runtime_library(),name)
    raise AttributeError(name)


def load_artifact(directory, record):
    directory = Path(directory)
    validate_case(record['case'])
    expected_config = config_identity(record['case'])
    if record.get('padding_policy') != PADDING_POLICY or not _same(record.get('physical_case'), physical_case(record['case'])):
        raise ValueError('historical or unpadded geometry cannot run through the corrected pilot')
    if record.get('config_sha256') != expected_config:
        raise ValueError('prepared artifact config identity differs')
    if record.get('runtime_policy') != 'shared-cuda12.9-checked' or 'libcudart.so.12' not in record.get('needed_libraries', []):
        raise ValueError('static unchecked reference library cannot run through the checked pilot ABI')
    for filename, key in [('generated.cu', 'source'), ('library.so', 'library')]:
        path = directory / filename
        if path.is_symlink() or file_identity(path) != record[key]:
            raise ValueError('prepared artifact content differs')
    source = (directory / 'generated.cu').read_text()
    if check_abi(source) != record['abi']:
        raise ValueError('prepared artifact ABI differs')
    bounds_path = directory / 'bounds.json'
    if file_identity(bounds_path) != record.get('bounds'):
        raise ValueError('finite geometry evidence content differs')
    bounds = json.loads(bounds_path.read_text())
    if not _same(bounds, geometry_record(record['case'], record['physical_case'], source)):
        raise ValueError('finite geometry evidence differs from the actual generated source')
    return _runtime_library().load_verified_library(directory,record,
        tuple(record['physical_case'][key] for key in ('M','N','K')),
        tuple(bounds['finite_reasoning']['alignment_bytes'][key] for key in ('A','W','C')))


def call_library(library,a,w,output,torch):
    return _runtime_library().call_library(library,a,w,output,torch)


def numerical_observation(actual, expected, torch):
    unequal = actual.view(torch.int32) != expected.view(torch.int32)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    finite_count = int(finite.sum().item())
    maximum = torch.max(torch.where(finite, torch.abs(actual - expected), 0)).item() if finite_count else None
    return {'elements': actual.numel(), 'unequal_bits': int(unequal.sum().item()),
            'actual_nonfinite': int((~torch.isfinite(actual)).sum().item()),
            'expected_nonfinite': int((~torch.isfinite(expected)).sum().item()),
            'max_absolute_finite': None if maximum is None else float(maximum), 'finite_comparisons': finite_count,
            'quality_acceptance': False}


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
            physical = physical_case(case)
            storage = padding_storage(case)
            staged_a = torch.zeros((physical['M'], physical['K']), dtype=torch.float16, device='cuda') if storage['input_copy_per_call'] else a
            staged_w = torch.zeros((physical['N'], physical['K']), dtype=torch.float16, device='cuda') if physical['N'] != n or physical['K'] != k else w
            if staged_w is not w:
                staged_w[:n, :k].copy_(w)
            result = torch.empty((rows, n), dtype=torch.float32, device='cuda')
            staged_c = torch.empty((physical['M'], physical['N']), dtype=torch.float32, device='cuda') if storage['logical_output_copy_per_call'] else result
            def execute(library=library, a=a, staged_a=staged_a, staged_w=staged_w,
                        staged_c=staged_c, result=result, rows=rows, k=k, n=n):
                if staged_a is not a:
                    staged_a[:rows, :k].copy_(a)
                call_library(library, staged_a, staged_w, staged_c, torch)
                if staged_c is not result:
                    result.copy_(staged_c[:rows, :n])
            execute()
            expected = reference.matmul_reference(a, f32=True)
            observed = numerical_observation(result, expected, torch)
            if observed['actual_nonfinite'] and not observed['expected_nonfinite']:
                write_record(output / (case['id'] + '-failure.json'), {
                    'case': case, 'physical_case': physical, 'kind': 'unexpected-nonfinite-output',
                    'against_tensorfold_reference': observed, 'quality_acceptance': False})
                raise ValueError('finite pilot inputs produced unexpected nonfinite output')
            if observed['actual_nonfinite'] or observed['expected_nonfinite']:
                write_record(output/(case['id']+'-nonfinite-failure.json'), observed)
                raise ValueError('unexpected nonfinite fixed32 diagnostic output')
            if longest is None:
                longest = result.clone()
                regroup[case['id']] = {'status': 'baseline-only-no-regroup-comparison', 'reference_rows': rows}
            else:
                regroup[case['id']] = {'reference_rows': longest.shape[0],
                                       **numerical_observation(result, longest[:rows], torch)}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                execute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                execute()
            a.add_(torch.tensor(2**-8, device='cuda', dtype=torch.float16))
            staged_w[:n, :k].mul_(2.)
            graph.replay()
            captured = result.clone()
            execute()
            torch.cuda.synchronize()
            graph_equal = torch.equal(captured.view(torch.int32), result.view(torch.int32))
            if not graph_equal:
                write_record(output / (case['id'] + '-failure.json'), {
                    'case': case, 'physical_case': physical, 'kind': 'changed-input-graph-eager-mismatch',
                    'against_tensorfold_reference': observed,
                    'graph_vs_same_library_eager': numerical_observation(captured, result, torch),
                    'artifact_library': record['library'], 'quality_acceptance': False})
                raise ValueError('changed-input graph replay differs from same-library eager execution')
            report = {'case': case, 'physical_case': physical, 'padding_storage': storage,
                      'against_tensorfold_reference': observed,
                      'changed_input_and_weight_graph_bitwise': graph_equal, 'actual_output_dtype': str(result.dtype),
                      'loaded_runtime_provider': library.runtime_provider, 'loaded_native_provider': library.native_provider}
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


FIXED_SCHEMA = 'tensorfold.bitblas-fixed-m32-pilot-v3'


def fixed32_budget(case):
    validate_case(case)
    kernel = next(item for item in cases() if item['role'] == case['role'] and item['M'] == 32)
    physical = physical_case(kernel)
    padded_weight = physical['N'] != case['N']
    frame_a = 32 * case['K'] * 2
    frame_c = 32 * physical['N'] * 4
    derived_w = physical['N'] * case['K'] * 2 if padded_weight else 0
    valid_rows = [min(32, case['M']-start) for start in range(0, case['M'], 32)]
    return {'physical_kernel': physical, 'panels': valid_rows,
            'logical_caller_bytes': {'A': case['M']*case['K']*2, 'W': case['N']*case['K']*2,
                                     'C': case['M']*case['N']*4},
            'reusable_frame_bytes': {'A32': frame_a, 'C32': frame_c, 'derived_W': derived_w},
            'additional_payload_bytes': frame_a + frame_c + derived_w,
            'input_copy_bytes_per_call': sum(count*case['K']*2 for count in valid_rows if count != 32),
            'input_tail_zero_bytes_per_call': sum((32-count)*case['K']*2 for count in valid_rows),
            'output_copy_bytes_per_call': sum(count*case['N']*4 for count in valid_rows
                                             if count != 32 or physical['N'] != case['N']),
            'weight_policy': 'immutable padded N8 HC-inject prepared once; other weights borrowed unchanged',
            'zero_policy': 'partial panels copy valid rows and zero unused rows; full panels borrow the aligned input directly',
            'output_frame_policy': 'reserved across all logical groupings; full N-matching panels write caller output directly',
            'scope': 'exact tensor payload/copy counts, not allocator/code/graph-pool capacity acceptance'}


def prepare_fixed32(source_manifest, output):
    source_manifest = Path(source_manifest)
    original = json.loads(source_manifest.read_text())
    if original.get('schema') != SCHEMA or original.get('phase') != 'generated':
        raise ValueError('clean schema-v2 kernels required for fixed32 preparation')
    validate_subset(original['cases'])
    selected = [item for item in original['cases'] if item['case']['M'] == 32]
    if {item['case']['role'] for item in selected} != {shape[0] for shape in SHAPES}:
        raise ValueError('all five reviewed physical M32 kernels required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': FIXED_SCHEMA, 'phase': 'prepared', 'source_tool': file_identity(__file__),
                'original_manifest': file_identity(source_manifest), 'kernels': {},
                'cases': [{'case': case, 'kernel_id': case['role'], 'storage': fixed32_budget(case)} for case in cases()],
                'quality_acceptance': False, 'performance_acceptance': False, 'MTP_adoption': False,
                'scope': 'CPU reuse of five physical M32 kernels; new pipeline and row-position law require native checks'}
    for item in selected:
        role = item['case']['role']
        source = source_manifest.parent / item['case']['id']
        destination = output / 'kernels' / role
        destination.mkdir(parents=True)
        for filename, key in [('generated.cu', 'source'), ('library.so', 'library'),
                              ('bounds.json', 'bounds'), ('dynamic-section.txt', 'dynamic_section')]:
            if file_identity(source / filename) != item[key]:
                raise ValueError('reviewed M32 source/artifact identity differs')
            shutil.copyfile(source / filename, destination / filename)
            os.chmod(destination / filename, 0o444)
        if not _same(json.loads((destination/'bounds.json').read_text()),
                     geometry_record(item['case'], item['physical_case'], (destination/'generated.cu').read_text())):
            raise ValueError('reviewed fixed32 finite geometry differs')
        if (not _same(item.get('physical_case'), physical_case(item['case']))
                or item.get('config_sha256') != config_identity(item['case'])
                or item.get('padding_policy') != PADDING_POLICY):
            raise ValueError('reviewed fixed32 configuration differs')
        record = {**item, 'provenance': {'action': 'copied exact previously memchecked physical M32 kernel',
                                       'original_manifest': file_identity(source_manifest),
                                       'original_directory': str(source.resolve()),
                                       'new_pipeline_native_results_reused': False}}
        write_record(destination / 'artifact.json', record)
        manifest['kernels'][role] = record
    write_record(output / 'manifest.json', manifest)
    return manifest


def diagnose_fixed32(manifest_path, output):
    import torch

    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = [{'case': case, 'kernel_id': case['role'], 'storage': fixed32_budget(case)} for case in cases()]
    if (manifest.get('schema') != FIXED_SCHEMA or manifest.get('phase') != 'prepared'
            or not _same(manifest.get('cases'), expected)
            or set(manifest.get('kernels', {})) != {shape[0] for shape in SHAPES}):
        raise ValueError('finite fixed32 plan differs')
    require_device(torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports, regroup, placement = [], {}, {}
    for index, (role, n, k) in enumerate(SHAPES):
        record = manifest['kernels'][role]
        if record['case']['role'] != role or record['case']['M'] != 32:
            raise ValueError('fixed kernel role or physical row count differs')
        library = load_artifact(manifest_path.parent/'kernels'/role, record)
        generator = torch.Generator(device='cuda').manual_seed(81721+index)
        inputs = torch.randn((max(ROWS), k), dtype=torch.float16, device='cuda', generator=generator)/32
        weights = torch.randn((n, k), dtype=torch.float16, device='cuda', generator=generator)/32
        np = library.dimensions[1]
        frame_a = torch.empty((32, k), dtype=torch.float16, device='cuda')
        frame_w = torch.zeros((np, k), dtype=torch.float16, device='cuda') if np != n else weights
        frame_c = torch.empty((32, np), dtype=torch.float32, device='cuda')
        if frame_w is not weights:
            frame_w[:n].copy_(weights)
        # Each distinct input is also tested in several physical row positions,
        # across the16-row/warp boundaries, without claiming exhaustive floats.
        probe = inputs[:32].clone()
        frame_a.copy_(probe)
        call_library(library, frame_a, frame_w, frame_c, torch)
        canonical = frame_c[:, :n].clone()
        role_placement = {}
        orders = {'reverse': torch.arange(31, -1, -1, device='cuda'),
                  'adjacent': torch.roll(torch.arange(32, device='cuda'), 1),
                  'half-tile': torch.roll(torch.arange(32, device='cuda'), 16)}
        for name, order in orders.items():
            frame_a.copy_(probe.index_select(0, order))
            call_library(library, frame_a, frame_w, frame_c, torch)
            comparison = numerical_observation(frame_c[:, :n], canonical.index_select(0, order), torch)
            role_placement[name] = comparison
            if comparison['unequal_bits']:
                write_record(output/(role+'-row-position-failure.json'), role_placement)
                raise ValueError('fixed physical M32 kernel changes bits with row placement')
        placement[role] = role_placement
        longest = None
        for rows in reversed(ROWS):
            case = next(item for item in cases() if item['role'] == role and item['M'] == rows)
            # Restore the immutable baseline after the preceding graph's deliberate
            # changed-weight diagnostic; production weights do not change per call.
            if frame_w is weights:
                baseline_w = weights.clone()
            else:
                baseline_w = weights
                frame_w[:n].copy_(weights)
            a = inputs[:rows].clone()
            result = torch.empty((rows, n), dtype=torch.float32, device='cuda')
            reference = GGUFDense(baseline_w)
            def execute(a=a, result=result, library=library, frame_a=frame_a, frame_w=frame_w,
                        frame_c=frame_c, rows=rows, n=n):
                for start in range(0, rows, 32):
                    count = min(32, rows-start)
                    if count < 32:
                        frame_a[:count].copy_(a[start:start+count])
                        frame_a[count:].zero_()
                        source = frame_a
                    else:
                        source = a[start:start+count]
                    direct = count == 32 and n == library.dimensions[1]
                    target = result[start:start+count] if direct else frame_c
                    call_library(library, source, frame_w, target, torch)
                    if not direct:
                        result[start:start+count].copy_(frame_c[:count, :n])
            execute()
            observed = numerical_observation(result, reference.matmul_reference(a, f32=True), torch)
            if observed['actual_nonfinite'] or observed['expected_nonfinite']:
                write_record(output/(case['id']+'-nonfinite-failure.json'), observed)
                raise ValueError('unexpected nonfinite fixed32 diagnostic output')
            if longest is None:
                longest = result.clone()
                regroup[case['id']] = {'status': 'baseline-only-no-regroup-comparison', 'reference_rows': rows}
            else:
                comparison = numerical_observation(result, longest[:rows], torch)
                regroup[case['id']] = {'reference_rows': longest.shape[0], **comparison}
                if comparison['unequal_bits']:
                    write_record(output/(case['id']+'-regroup-failure.json'), regroup[case['id']])
                    raise ValueError('fixed32 logical grouping changes output bits')
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                execute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                execute()
            a.add_(torch.tensor(2**-8, dtype=torch.float16, device='cuda'))
            frame_w[:n].mul_(2.)
            graph.replay()
            captured = result.clone()
            execute()
            torch.cuda.synchronize()
            if not torch.equal(captured.view(torch.int32), result.view(torch.int32)):
                write_record(output/(case['id']+'-graph-failure.json'), numerical_observation(captured, result, torch))
                raise ValueError('fixed32 staged graph differs from identical eager pipeline')
            report = {'case': case, 'storage': fixed32_budget(case), 'against_tensorfold_reference': observed,
                      'changed_input_weight_graph_bitwise': True, 'loaded_native_provider': library.native_provider,
                      'loaded_runtime_provider': library.runtime_provider}
            write_record(output/(case['id']+'.json'), report)
            reports.append(report)
            del graph
            if frame_w is weights:
                weights.copy_(baseline_w)
        torch.cuda.synchronize()
    result = {'schema': FIXED_SCHEMA, 'phase': 'diagnostic', 'source_tool': file_identity(__file__),
              'manifest': file_identity(manifest_path), 'reports': reports, 'row_regrouping': regroup,
              'row_placement': placement, 'positive_device_retirement': True,
              'quality_acceptance': False, 'performance_acceptance': False, 'MTP_adoption': False,
              'scope': 'fixed32 component row-law/value/pipeline capture observations, not model/PPL/performance acceptance'}
    write_record(output/'diagnostic.json', result)
    return result


def prepare_hc(manifest_path, output):
    """Seal only three existing HC libraries; no vendor import/compiler/GPU work."""
    manifest_path=Path(manifest_path)
    original=json.loads(manifest_path.read_text())
    if (original.get('schema') != FIXED_SCHEMA or original.get('phase') != 'prepared'
            or set(original.get('kernels',{})) != {v[0] for v in SHAPES}):
        raise ValueError('complete fixed32 producer manifest required')
    runtime=_runtime_library()
    selected={}
    for role,(m,n,k,alignment) in runtime.HC_KERNELS.items():
        record=original['kernels'][role]
        case=next(v for v in cases() if v['role']==role and v['M']==32)
        directory=manifest_path.parent/'kernels'/role
        if (not _same(record.get('case'),case) or not _same(record.get('physical_case'),physical_case(case))
                or record.get('config_sha256') != config_identity(case)
                or record.get('padding_policy') != PADDING_POLICY):
            raise ValueError('HC producer configuration differs')
        for filename,key in [('generated.cu','source'),('library.so','library'),('bounds.json','bounds')]:
            if (directory/filename).is_symlink() or file_identity(directory/filename) != record[key]:
                raise ValueError('HC producer artifact bytes differ')
        source=(directory/'generated.cu').read_text()
        bounds=json.loads((directory/'bounds.json').read_text())
        if (check_abi(source) != record['abi'] or not _same(bounds,geometry_record(case,physical_case(case),source))
                or [bounds['finite_reasoning']['alignment_bytes'][v] for v in ('A','W','C')] != [alignment,alignment,4]
                or record.get('runtime_policy') != 'shared-cuda12.9-checked'
                or 'libcudart.so.12' not in record.get('needed_libraries',[])):
            raise ValueError('HC producer ABI/bounds/runtime differs')
        selected[role]={'directory':role,'dimensions':[m,n,k],'alignment_bytes':[alignment,alignment,4],
                        'input_dtype':'float16','output_dtype':'float32','accumulation_dtype':'float32','record':record}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for role in selected:
        directory=output/role;directory.mkdir()
        for filename in ('generated.cu','library.so','bounds.json'):
            shutil.copyfile(manifest_path.parent/'kernels'/role/filename,directory/filename)
            os.chmod(directory/filename,0o444)
    result={'schema':runtime.HC_SCHEMA,'phase':'prepared','policy':'fixed-m32-hc-only-f32-accumulation-v1',
            'source_tool':file_identity(__file__),'library_helper':file_identity(runtime.__file__),
            'producer_manifest':file_identity(manifest_path),'project_python_commit':PYTHON_COMMIT,
            'whole_wheel_sha256':WHEEL_SHA256,'kernels':selected,
            'quality_acceptance':False,'performance_acceptance':False,
            'scope':'three unchanged fixed32 HC exports; explicit Half publication remains adapter responsibility'}
    write_record(output/'manifest.json',result)
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
    selected.add_argument('--case', action='append', choices=[case['id'] for case in cases()], help='explicit bounded generation subset; omitted selects the finite full matrix')
    selected = subparsers.add_parser('relink-shared')
    selected.add_argument('--manifest', type=Path, required=True)
    selected.add_argument('--wheel', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    selected.add_argument('--compile-timeout-s', type=float, required=True)
    selected = subparsers.add_parser('assemble-reviewed')
    selected.add_argument('--old-manifest', type=Path, required=True)
    selected.add_argument('--corrected-manifest', type=Path, required=True)
    selected.add_argument('--wheel', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    selected.add_argument('--compile-timeout-s', type=float, required=True)
    selected = subparsers.add_parser('prepare-fixed32')
    selected.add_argument('--manifest', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    selected = subparsers.add_parser('prepare-hc')
    selected.add_argument('--manifest', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    selected = subparsers.add_parser('diagnose-fixed32')
    selected.add_argument('--manifest', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    selected = subparsers.add_parser('diagnose')
    selected.add_argument('--manifest', type=Path, required=True)
    selected.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        result = plan(args.wheel, args.output)
    elif args.command == 'generate':
        result = generate(args.plan, args.output, wheel=args.wheel, case_ids=args.case, compile_timeout_s=args.compile_timeout_s)
    elif args.command == 'relink-shared':
        result = relink_shared(args.manifest, args.output, wheel=args.wheel, compile_timeout_s=args.compile_timeout_s)
    elif args.command == 'assemble-reviewed':
        result = assemble_reviewed(args.old_manifest, args.corrected_manifest, args.output,
                                   wheel=args.wheel, compile_timeout_s=args.compile_timeout_s)
    elif args.command == 'prepare-fixed32':
        result = prepare_fixed32(args.manifest, args.output)
    elif args.command == 'prepare-hc':
        result = prepare_hc(args.manifest, args.output)
    elif args.command == 'diagnose-fixed32':
        result = diagnose_fixed32(args.manifest, args.output)
    else:
        result = diagnose(args.manifest, args.output)
    print(json.dumps({'schema': result['schema'], 'phase': result['phase'],
                      'cases': len(result.get('cases', result.get('reports', result.get('kernels', [])))),
                      'quality_acceptance': False, 'performance_acceptance': False}, sort_keys=True))


if __name__ == '__main__':
    main()
