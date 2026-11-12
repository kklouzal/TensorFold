"""Finite BitBLAS artifact/ABI contracts; no GPU or compiler imports required."""
import hashlib
import json

import pytest

from tools import v100_bitblas_prepare as pilot

SOURCE = '''extern "C" void init() {}
extern "C" void call(half_t* __restrict__ A, half_t* __restrict__ B,
float* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {}
'''


def test_closed_matrix_preserves_fp16_fp32_and_real_hc_shapes():
    cases = pilot.cases()
    assert len(cases) == 20 and len({case['id'] for case in cases}) == 20
    assert {(case['N'], case['K']) for case in cases if case['role'].startswith('hc')} == {
        (320, 10240), (4, 10240), (10240, 320)}
    assert {case['M'] for case in cases} == {1, 5, 32, 120}
    assert all(case['A_dtype'] == case['W_dtype'] == 'float16'
               and case['accum_dtype'] == case['out_dtype'] == 'float32'
               and case['propagate_a'] is case['propagate_b'] is False for case in cases)


@pytest.mark.parametrize('key,value', [('M', True), ('M', 2), ('N', 161), ('A_dtype', 'bfloat16'),
                                     ('accum_dtype', 'float16'), ('propagate_a', 0), ('id', '../escape')])
def test_case_mutation_is_refused_before_native_work(key, value):
    case = pilot.cases()[0]
    case[key] = value
    with pytest.raises(ValueError, match='finite FP16'):
        pilot.validate_case(case)


def test_generated_static_c_abi_has_no_dynamic_dimensions_or_hidden_parameters():
    abi = pilot.check_abi(SOURCE)
    assert abi['init'] == {'return': 'void', 'arguments': []}
    assert abi['call']['return'] == 'void'
    assert abi['call']['arguments'] == ['A:fp16[M,K]', 'W:fp16[N,K]', 'C:fp32[M,N]', 'cudaStream_t']


@pytest.mark.parametrize('source', [SOURCE.replace('float*', 'half_t*'),
                                   SOURCE.replace('cudaStream_t stream=cudaStreamDefault', 'int M'),
                                   SOURCE.replace('cudaStream_t stream=cudaStreamDefault', 'int M, cudaStream_t stream=cudaStreamDefault'),
                                   SOURCE.replace('void init()', 'int init()'),
                                   SOURCE.replace('void call(', 'int call(')])
def test_incompatible_generated_abi_is_refused(source):
    with pytest.raises(ValueError):
        pilot.check_abi(source)


def test_artifact_hash_and_exclusive_record_do_not_overwrite_history(tmp_path):
    artifact = tmp_path / 'source.cu'
    artifact.write_bytes(b'original source')
    assert pilot.file_identity(artifact) == {'bytes': 15, 'sha256': hashlib.sha256(b'original source').hexdigest()}
    record = tmp_path / 'record.json'
    pilot.write_record(record, {'state': 'original-failure'})
    with pytest.raises(FileExistsError):
        pilot.write_record(record, {'state': 'replacement'})
    assert json.loads(record.read_text()) == {'state': 'original-failure'}


def test_wrong_whole_wheel_is_refused_before_zip_or_import(tmp_path):
    wheel = tmp_path / 'wheel.whl'
    wheel.write_bytes(b'not the official package')
    with pytest.raises(ValueError, match='whole-wheel SHA256'):
        pilot.plan(wheel, tmp_path / 'plan.json')
    assert not (tmp_path / 'plan.json').exists()


@pytest.mark.parametrize('mutation', ['matrix', 'tuning', 'compiler', 'version'])
def test_frozen_plan_refuses_incompatible_selection(tmp_path, mutation):
    record = {'schema': pilot.SCHEMA, 'phase': 'plan', 'project_python_commit': pilot.PYTHON_COMMIT,
              'cases': pilot.cases(), 'generation_policy': {'backend': 'tl', 'enable_tuning': False,
                  'weight_propagation': False, 'compile_during_serving': False},
              'wheel': {'sha256': pilot.WHEEL_SHA256}}
    if mutation == 'matrix':
        record['cases'][0]['M'] = True
    elif mutation == 'tuning':
        record['generation_policy']['enable_tuning'] = True
    elif mutation == 'compiler':
        record['generation_policy']['backend'] = 'tir'
    else:
        record['project_python_commit'] = 'unverified'
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='frozen'):
        pilot.read_plan(path)


@pytest.mark.parametrize('count,capability', [(0, None), (2, None), (1, (8, 0))])
def test_generation_refuses_unleased_or_wrong_device_before_vendor_import(monkeypatch, tmp_path, count, capability):
    import builtins
    import sys
    from types import SimpleNamespace

    monkeypatch.setattr(pilot, 'read_plan', lambda path: {'wheel': {'path': 'authenticated elsewhere'}, 'cases': pilot.cases()})
    monkeypatch.setattr(pilot, 'verify_installed_payload', lambda path: {})
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == 'bitblas':
            pytest.fail('imported device-querying vendor before checking the GPU lease')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', importing)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: count, get_device_capability=lambda device: capability,
        set_device=lambda device: pytest.fail('selected unsupported device'))))
    with pytest.raises(ValueError, match='one visible V100'):
        pilot.generate('unused', tmp_path / 'attempt', compile_timeout_s=600)


def test_missing_protected_compiler_is_reported_before_any_command(monkeypatch):
    monkeypatch.setattr(pilot.shutil, 'which', lambda name: None)
    monkeypatch.setattr(pilot.subprocess, 'check_output', lambda *a, **k: pytest.fail('executed missing compiler'))
    with pytest.raises(ValueError, match='compiler is unavailable'):
        pilot.compiler_identity()


def test_prebuilt_cuda_failure_is_never_silently_reported_as_success():
    from types import SimpleNamespace

    runtime = SimpleNamespace(cudaGetLastError=lambda: 719, cudaGetErrorString=lambda error: b'launch failed')
    with pytest.raises(RuntimeError, match='launch failed'):
        pilot._cuda_status(runtime)


def test_standalone_export_refuses_runtime_compiler_dependencies():
    assert pilot.check_dependencies('0x1 (NEEDED) Shared library: [libcudart.so.12]\n0x2 (NEEDED) Shared library: [libcuda.so.1]') == ['libcudart.so.12', 'libcuda.so.1']
    for dependency in ('libtvm.so', 'libtilelang_module.so', 'libLLVM.so.18', 'libnvrtc.so.12'):
        with pytest.raises(ValueError, match='compiler libraries'):
            pilot.check_dependencies(f'0x1 (NEEDED) Shared library: [{dependency}]')


def test_checked_export_requires_shared_cudart_in_dynamic_dependencies():
    section = '0x1 (NEEDED) Shared library: [libstdc++.so.6]'
    assert pilot.check_dependencies(section) == ['libstdc++.so.6']
    with pytest.raises(ValueError, match='shared CUDA12'):
        pilot.check_dependencies(section, require_shared=True)
    assert pilot.check_dependencies(section + '\n0x2 (NEEDED) Shared library: [libcudart.so.12]', require_shared=True) == ['libstdc++.so.6', 'libcudart.so.12']


def test_generated_subset_is_ordered_unique_and_type_exact():
    first, second = pilot.cases()[:2]
    assert pilot.validate_subset([{'case': first}]) == [first]
    for records in ([{'case': first}, {'case': first}], [{'case': second}, {'case': first}],
                    [{'case': {**first, 'M': True}}], []):
        with pytest.raises(ValueError, match='case'):
            pilot.validate_subset(records)


@pytest.mark.parametrize('timeout', [None, True, 0, -1, float('inf'), float('nan')])
def test_compiler_deadline_is_validated_before_source_or_subprocess(timeout, monkeypatch):
    monkeypatch.setattr(pilot.subprocess, 'run', lambda *a, **k: pytest.fail('unbounded compile'))
    with pytest.raises(ValueError, match='finite compiler deadline'):
        pilot.compile_shared('source.cu', 'library.so', {}, 'missing', 'log', timeout_s=timeout)


def test_static_reference_is_refused_before_native_library_loading(tmp_path, monkeypatch):
    case = pilot.cases()[0]
    (tmp_path / 'generated.cu').write_text(SOURCE)
    (tmp_path / 'library.so').write_bytes(b'not loaded')
    record = {'case': case, 'abi': pilot.check_abi(SOURCE),
              'config_sha256': hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              'source': pilot.file_identity(tmp_path / 'generated.cu'),
              'library': pilot.file_identity(tmp_path / 'library.so'), 'needed_libraries': ['libstdc++.so.6']}
    monkeypatch.setattr(pilot.ctypes, 'CDLL', lambda *a, **k: pytest.fail('loaded unchecked static reference'))
    with pytest.raises(ValueError, match='static unchecked'):
        pilot.load_artifact(tmp_path, record)
