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
              'wheel': {'sha256': pilot.WHEEL_SHA256}, 'padding_policy': pilot.PADDING_POLICY,
              'physical_cases': [pilot.physical_case(case) for case in pilot.cases()]}
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
              'config_sha256': pilot.config_identity(case), 'physical_case': pilot.physical_case(case),
              'padding_policy': pilot.PADDING_POLICY,
              'source': pilot.file_identity(tmp_path / 'generated.cu'),
              'library': pilot.file_identity(tmp_path / 'library.so'), 'needed_libraries': ['libstdc++.so.6']}
    monkeypatch.setattr(pilot.ctypes, 'CDLL', lambda *a, **k: pytest.fail('loaded unchecked static reference'))
    with pytest.raises(ValueError, match='static unchecked'):
        pilot.load_artifact(tmp_path, record)


def test_padding_uses_supported_physical_geometry_and_pays_retained_staging():
    for case in pilot.cases():
        physical = pilot.physical_case(case)
        simt = case['M'] < 8 or case['N'] < 16
        if simt:
            assert physical['N'] % 8 == 0 and physical['K'] % 128 == 0
        else:
            assert physical == case
        assert physical['M'] == case['M']
    down = next(case for case in pilot.cases() if case['id'] == 'down-reference-m1')
    assert pilot.physical_case(down)['K'] == 256
    storage = pilot.padding_storage(down)
    assert storage['additional_payload_bytes'] == 1*256*2 + 2560*256*2
    assert storage['input_copy_per_call'] is True and storage['logical_output_copy_per_call'] is False
    inject = next(case for case in pilot.cases() if case['id'] == 'hc-inject-m120')
    assert pilot.physical_case(inject)['N'] == 8
    assert pilot.padding_storage(inject)['logical_output_copy_per_call'] is True
    up = next(case for case in pilot.cases() if case['id'] == 'hc-up-m5')
    assert pilot.physical_case(up)['K'] == 384


def test_historical_unpadded_library_is_refused_before_loading(tmp_path, monkeypatch):
    case = next(case for case in pilot.cases() if case['id'] == 'hc-inject-m120')
    monkeypatch.setattr(pilot.ctypes, 'CDLL', lambda *a, **k: pytest.fail('loaded unsafe historical code'))
    with pytest.raises(ValueError, match='historical or unpadded'):
        pilot.load_artifact(tmp_path, {'case': case})


def test_finite_geometry_checks_actual_launch_and_rejects_unsafe_default_dimensions():
    case = next(case for case in pilot.cases() if case['id'] == 'down-reference-m5')
    unsafe = 'A + ((ko*128)+(tx*8)); C[by*2560+ty]=0; f<<<dim3(320, 5, 1), dim3(16, 8, 1), 0, stream>>>(A,B,C);'
    with pytest.raises(ValueError, match='unsafe'):
        pilot.geometry_record(case, case, unsafe)
    physical = pilot.physical_case(case)
    report = pilot.geometry_record(case, physical, 'matmul_m5n2560k256_f16xf16 ' + unsafe)
    assert report['finite_reasoning']['A']['range'] == [0, 5*256-1]
    assert report['finite_reasoning']['B']['range'] == [0, 2560*256-1]
    assert report['native_memory_acceptance'] is False
    assert report['status'] == 'finite_geometry_record_requires_independent_source_review_and_native_memcheck'


@pytest.mark.parametrize('selection', [['gu-reference-m1', 'gu-reference-m1'], ['unknown'], [{}], 'gu-reference-m1'])
def test_generation_subset_is_checked_before_payload_or_hardware(selection, monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, 'read_plan', lambda path: {'cases': pilot.cases()})
    monkeypatch.setattr(pilot, 'verify_installed_payload', lambda path: pytest.fail('read payload for invalid subset'))
    with pytest.raises(ValueError, match='finite unique'):
        pilot.generate('unused', tmp_path / 'attempt', case_ids=selection, compile_timeout_s=240)
    assert not (tmp_path / 'attempt').exists()


@pytest.mark.parametrize('input_alignment', [8, 16])
@pytest.mark.parametrize('operand', [0, 1, 2])
def test_reviewed_vector_alignment_is_enforced_before_native_calls(input_alignment, operand):
    from types import SimpleNamespace

    device = SimpleNamespace(type='cuda', index=0)
    class MetadataTensor:
        requires_grad = False
        def __init__(self, shape, dtype, pointer):
            self.shape, self.dtype, self.pointer, self.device = shape, dtype, pointer, device
        def is_contiguous(self):
            return True
        def is_neg(self):
            return False
        def is_conj(self):
            return False
        def data_ptr(self):
            return self.pointer
    fake = SimpleNamespace(Tensor=MetadataTensor, float16='Half', float32='Float')
    values = [MetadataTensor((1, 2560), 'Half', 0x10000000),
              MetadataTensor((160, 2560), 'Half', 0x20000000),
              MetadataTensor((1, 160), 'Float', 0x30000000)]
    values[operand].pointer += 2
    native = SimpleNamespace(call=lambda *args: pytest.fail('launched a misaligned vector access'))
    prepared = pilot.PreparedLibrary(native, None, (1, 160, 2560), {}, (input_alignment, input_alignment, 4))
    with pytest.raises(ValueError, match='emitter alignment'):
        pilot.call_library(prepared, *values, fake)


def test_fixed32_preserves_logical_shapes_and_reuses_exact_m32_geometry():
    for case in pilot.cases():
        budget = pilot.fixed32_budget(case)
        physical = budget['physical_kernel']
        assert physical['M'] == 32
        assert physical['N'] == (8 if case['role'] == 'hc-inject' else case['N'])
        assert physical['K'] == case['K']
        assert sum(budget['panels']) == case['M']
        assert budget['panels'] == ([32, 32, 32, 24] if case['M'] == 120 else [case['M']])
        frame = budget['reusable_frame_bytes']
        assert frame['A32'] == 32*case['K']*2
        assert frame['C32'] == 32*physical['N']*4
        assert frame['derived_W'] == (8*case['K']*2 if case['role'] == 'hc-inject' else 0)
        assert budget['additional_payload_bytes'] == sum(frame.values())


def test_fixed32_counts_partial_staging_zero_crop_and_full_panel_borrowing():
    by_id = {case['id']: case for case in pilot.cases()}
    down = pilot.fixed32_budget(by_id['down-reference-m120'])
    assert down['input_copy_bytes_per_call'] == 24*160*2
    assert down['input_tail_zero_bytes_per_call'] == 8*160*2
    assert down['output_copy_bytes_per_call'] == 24*2560*4
    full = pilot.fixed32_budget(by_id['down-reference-m32'])
    assert full['input_copy_bytes_per_call'] == full['input_tail_zero_bytes_per_call'] == full['output_copy_bytes_per_call'] == 0
    inject = pilot.fixed32_budget(by_id['hc-inject-m120'])
    assert inject['output_copy_bytes_per_call'] == 120*4*4
    single = pilot.fixed32_budget(by_id['hc-up-m1'])
    assert single['input_copy_bytes_per_call'] == 320*2
    assert single['input_tail_zero_bytes_per_call'] == 31*320*2
    assert single['output_copy_bytes_per_call'] == 10240*4


def test_fixed32_refuses_historical_schema_before_creating_outputs(tmp_path):
    source = tmp_path/'manifest.json'
    source.write_text(json.dumps({'schema': 'historical', 'phase': 'generated'}))
    with pytest.raises(ValueError, match='clean schema-v2'):
        pilot.prepare_fixed32(source, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_fixed32_requires_all_five_roles_before_creating_outputs(tmp_path):
    source = tmp_path/'manifest.json'
    source.write_text(json.dumps({'schema': pilot.SCHEMA, 'phase': 'generated',
                                  'cases': [{'case': pilot.cases()[0]}]}))
    with pytest.raises(ValueError, match='all five'):
        pilot.prepare_fixed32(source, tmp_path/'output')
    assert not (tmp_path/'output').exists()
