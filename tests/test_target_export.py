"""Staging coordinator boundaries only: tiny files, no ModelOpt/model/GPU run."""
import hashlib
import json
import os
import struct
import sys
from types import ModuleType, SimpleNamespace
import warnings

import pytest

from tensorfold.offline_quantization import target_export as target


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_file(path, tensors):
    header, payload = {}, b''
    for name, raw in tensors.items():
        header[name] = {'dtype': 'U8', 'shape': [len(raw)], 'data_offsets': [len(payload), len(payload)+len(raw)]}
        payload += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded))+encoded+payload)


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    config = {'model_type': 'tiny-synthetic-fixture', 'eos_token_id': 2}
    (source/'config.json').write_text(json.dumps(config))
    (source/'tokenizer.json').write_text('{"fixture":true}')
    (source/'merges.txt').write_bytes(b'')
    tensor_file(source/'model.safetensors', {'synthetic': b'xyz'})
    records = {}
    for path in source.iterdir():
        st = path.stat()
        records[path.name] = {'sha256': sha(path.read_bytes()),
                              'identity': (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)}
    parameter = object()
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=parameter),
                            get_input_embeddings=lambda: SimpleNamespace(weight=parameter), named_modules=lambda: ())
    owner = target.CalibratedTarget()
    owner.model = model
    owner.source_root = source.resolve()
    owner._runtime_namespace = {'fixture': True}
    runtime = {'producer_commit': 'a'*40, 'wheel_sha256': 'b'*64}
    owner.receipt = {'status': 'calibrated-native-target-not-exported-or-quality-qualified',
                     'source_file_hashes': records, 'runtime': runtime, 'recipe_sha256': 'c'*64}
    phases, rechecks, exports = [], [], []
    monkeypatch.setattr(target, '_phase', lambda name, guard: phases.append(name) or guard(name))
    monkeypatch.setattr(target, '_recheck_source', lambda root, actual: rechecks.append((root, actual)))
    monkeypatch.setattr(target, '_runtime_proof', lambda namespace: runtime if namespace is owner._runtime_namespace else None)
    monkeypatch.setattr(target, 'E2BConfig', SimpleNamespace(from_dict=lambda raw: 'synthetic-cfg'))
    monkeypatch.setattr(target, '_inventory', lambda *args, **kwargs: ({}, set()))
    monkeypatch.setattr(target, 'text_tensor_contract', lambda _: ({}, set(), set()))
    module = ModuleType('modelopt.torch.export')

    def export(model, **kwargs):
        exports.append((model, kwargs))
        stage = kwargs['export_dir']
        (stage/'config.json').write_text(json.dumps(config | {
            'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}}))
        (stage/'hf_quant_config.json').write_text(json.dumps({
            'producer': {'name': 'modelopt'}, 'quantization': {'quant_algo': 'NVFP4'}}))
        (stage/'generation_config.json').write_text('{"do_sample":true}')
        (stage/'tokenizer.json').write_text('{"WRONG":true}')
        tensor_file(stage/'model.safetensors', {'synthetic': b'packed'})
    module.export_hf_checkpoint = export
    monkeypatch.setitem(sys.modules, 'modelopt.torch.export', module)
    stage = tmp_path/'new-stage'
    return SimpleNamespace(owner=owner, source=source, stage=stage, module=module,
                           records=records, phases=phases, rechecks=rechecks, exports=exports,
                           run=lambda **kwargs: target.export_target(owner, source, stage,
                               admit_phase=lambda name: {'actual-test-guard': name},
                               source_revision='d'*40, **kwargs))


def test_stages_unqualified_preserves_assets_absence_and_actual_manifest(scenario):
    s = scenario
    result = s.run()
    assert result['status'] == 'exported-staging-UNQUALIFIED'
    assert result['consumer_reload_pending'] and result['quality_pending'] and not result['output_published']
    assert not (s.stage/'generation_config.json').exists()
    for name in ('tokenizer.json', 'merges.txt'):
        assert (s.stage/name).read_bytes() == (s.source/name).read_bytes()
    assert json.loads((s.stage/'config.json').read_text())['quantization_config']['quant_method'] == 'modelopt'
    assert result['manifest']['tensor_sha256'] == {'synthetic': sha(b'packed')}
    assert result['manifest']['provenance']['producer_git_revision'] == 'a'*40
    assert 'WHOLE' in result['manifest']['provenance']['source_bf16_payloads_scope']
    assert s.exports[0][1]['max_shard_size'] == 1_000_000_000
    assert s.exports[0][1]['save_modelopt_state'] is False
    assert os.stat(s.stage).st_mode & 0o777 == 0o700
    assert len(s.rechecks) == 4 and s.phases[-1] == 'after-export-manifest'
    assert s.owner.model is s.exports[0][0]  # caller still owns retirement
    with pytest.raises(ValueError, match='single-use'):
        s.run()


def test_source_generation_bytes_restored_when_original_present(scenario):
    s = scenario
    raw = b'{"do_sample":false,"eos_token_id":2}\n'
    path = s.source/'generation_config.json'
    path.write_bytes(raw)
    st = path.stat()
    s.records[path.name] = {'sha256': sha(raw), 'identity': (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)}
    assert s.run()['manifest']['metadata_sha256'][path.name] == sha(raw)
    assert (s.stage/path.name).read_bytes() == raw


@pytest.mark.parametrize('hint', [0, -1, True, '1GB'])
def test_invalid_shard_hint_refused_before_provider_or_stage(scenario, hint):
    with pytest.raises(ValueError, match='shard'):
        scenario.run(max_shard_size=hint)
    assert not scenario.exports and not scenario.stage.exists()


def test_existing_source_overlap_and_aliases_refused_before_mutation(scenario):
    s = scenario
    s.stage.symlink_to(s.source, target_is_directory=True)
    with pytest.raises(ValueError, match='NEW'):
        s.run()
    assert not s.exports and not hasattr(s.owner, 'export_transition')
    s.stage.unlink()
    with pytest.raises(ValueError, match='disjoint'):
        target.export_target(s.owner, s.source, s.source/'nested', admit_phase=lambda _: {}, source_revision='d'*40)


def test_runtime_change_refused_before_model_mutation(scenario, monkeypatch):
    monkeypatch.setattr(target, '_runtime_proof', lambda _: {'different': True})
    with pytest.raises(ValueError, match='runtime differs'):
        scenario.run()
    assert not scenario.exports and not scenario.stage.exists()


def test_provider_failure_retains_original_and_mutated_owner_without_retry(scenario):
    primary = RuntimeError('public exporter failed after mutation')
    def fail(*args, **kwargs):
        warnings.warn('partial producer warning', UserWarning)
        raise primary
    scenario.module.export_hf_checkpoint = fail
    with pytest.raises(RuntimeError) as caught:
        scenario.run()
    assert caught.value is primary
    assert primary._tensorfold_calibrated_target is scenario.owner
    assert scenario.owner.export_transition['warnings'] == ['partial producer warning']
    assert scenario.owner.export_transition['status'].startswith('export-failed')
    assert not (scenario.stage/'tensorfold_quantization_manifest.json').exists()
    with pytest.raises(ValueError, match='single-use'):
        scenario.run()


@pytest.mark.parametrize('native_cause', [False, True])
def test_foreign_exception_refusing_failure_annotations_does_not_replace_primary(scenario, native_cause):
    class Primary(RuntimeError):
        def __setattr__(self, name, value):
            if name.startswith('_tensorfold'):
                raise MemoryError('foreign annotation failure')
            super().__setattr__(name, value)
    primary = Primary('real public export failure')
    original_cause = ValueError('original native export cause')
    if native_cause:
        primary.__cause__ = original_cause
    def fail(*args, **kwargs):
        raise primary
    scenario.module.export_hf_checkpoint = fail
    with pytest.raises(Primary) as caught:
        scenario.run()
    assert caught.value is primary
    causes = primary.__cause__.exceptions if isinstance(primary.__cause__, BaseExceptionGroup) else (primary.__cause__,)
    assert any(isinstance(error, MemoryError) for error in causes)
    if native_cause:
        assert original_cause in causes
    assert scenario.owner.model is not None and scenario.stage.is_dir()
    assert scenario.owner.export_transition['status'].startswith('export-failed')


def test_warning_is_not_silent_layout_fallback(scenario):
    original = scenario.module.export_hf_checkpoint
    def warn(*args, **kwargs):
        original(*args, **kwargs)
        warnings.warn('reverse conversion fallback', UserWarning)
    scenario.module.export_hf_checkpoint = warn
    with pytest.raises(ValueError, match='warnings'):
        scenario.run()
    assert not (scenario.stage/'tensorfold_quantization_manifest.json').exists()


@pytest.mark.parametrize('change', ['symlink', 'source-hardlink', 'unknown', 'config', 'no-sidecar'])
def test_invalid_provider_output_remains_private_unqualified(scenario, change):
    original = scenario.module.export_hf_checkpoint
    def malformed(*args, **kwargs):
        original(*args, **kwargs)
        stage = kwargs['export_dir']
        if change == 'symlink':
            (stage/'escape').symlink_to(scenario.source)
        elif change == 'source-hardlink':
            (stage/'tokenizer.json').unlink()
            os.link(scenario.source/'tokenizer.json', stage/'tokenizer.json')
        elif change == 'unknown':
            (stage/'unbound.py').write_text('fixture')
        elif change == 'config':
            (stage/'config.json').write_text('{"eos_token_id":99}')
        else:
            (stage/'hf_quant_config.json').unlink()
    scenario.module.export_hf_checkpoint = malformed
    with pytest.raises(ValueError):
        scenario.run()
    assert scenario.owner.model is not None
    assert not (scenario.stage/'tensorfold_quantization_manifest.json').exists()


def test_after_manifest_guard_failure_retains_complete_unpublished_stage(scenario, monkeypatch):
    def phase(name, guard):
        if name == 'after-export-manifest':
            raise MemoryError('actual guard revoked admission')
        return guard(name)
    monkeypatch.setattr(target, '_phase', phase)
    with pytest.raises(MemoryError):
        scenario.run()
    assert (scenario.stage/'tensorfold_quantization_manifest.json').is_file()
    assert scenario.owner.export_transition['status'].startswith('export-failed')


@pytest.mark.parametrize('drain_raises,native_cause', [(False, False), (True, False), (True, True)])
def test_asset_write_and_retirement_failures_preserve_primary_and_io_owner(scenario, monkeypatch, drain_raises, native_cause):
    primary, secondary = OSError('asset flush failed'), OSError('owned retirement failed')
    original_cause = ValueError('original native IO cause')
    if native_cause:
        primary.__cause__ = original_cause
    def fail():
        raise primary
    class Scope:
        retired = False
        def __init__(self, **kwargs):
            pass
        def open(self, *args, **kwargs):
            return SimpleNamespace(stream=SimpleNamespace(write=lambda raw: len(raw), flush=fail)), None
        def drain(self):
            if drain_raises:
                raise secondary
            return [secondary]
    monkeypatch.setattr(target, 'FileStreams', Scope)
    with pytest.raises(OSError) as caught:
        scenario.run()
    assert caught.value is primary
    causes = primary.__cause__.exceptions if isinstance(primary.__cause__, BaseExceptionGroup) else (primary.__cause__,)
    assert secondary in causes
    if native_cause:
        assert original_cause in causes
    assert isinstance(primary._tensorfold_export_io_scope, Scope)
    assert primary._tensorfold_calibrated_target is scenario.owner


def test_optional_unused_hashes_read_only_selected_raw_payloads(tmp_path):
    tensor_file(tmp_path/'one.safetensors', {'unused': b'abc', 'other': b'not-selected'})
    assert target._unused_hashes(tmp_path, {'one.safetensors': None}, {'unused'}) == {'unused': sha(b'abc')}
    with pytest.raises(ValueError, match='complete54'):
        target._unused_hashes(tmp_path, {'one.safetensors': None}, {'missing'})


def test_config_source_semantics_cannot_be_hidden_by_added_quantization():
    source = {'text_config': {'eos_token_id': [1, 2]}, 'torch_dtype': 'bfloat16'}
    target._config_semantics(source, {'text_config': {'eos_token_id': [1, 2]}, 'dtype': 'bfloat16', 'new': 5})
    with pytest.raises(ValueError, match='semantics'):
        target._config_semantics(source, {'text_config': {'eos_token_id': [2]}, 'dtype': 'bfloat16'})
    with pytest.raises(ValueError, match='semantics'):
        target._config_semantics({'native_flags': [False, None]}, {'native_flags': [0, None]})
    with pytest.raises(ValueError, match='removed'):
        target._config_semantics({'native_nullable': None}, {})


@pytest.fixture
def inventory_case(monkeypatch):
    from tensorfold.families.gemma4.cuda import dense_loader
    # Header-only synthetic258 contract, not an actual E4B checkpoint/model.
    modules = [f'model.language_model.layers.{i}.mlp.gate_proj' for i in range(258)]
    active = {name+'.weight' for name in modules}
    unused = {'model.language_model.layers.999.self_attn.k_norm.weight'}
    source = {name: {'dtype': 'BF16', 'shape': [128, 128]} for name in active}
    source.update({'model.language_model.embed_tokens.weight': {'dtype': 'BF16', 'shape': [4, 128]},
                   'model.language_model.norm.weight': {'dtype': 'BF16', 'shape': [128]},
                   next(iter(unused)): {'dtype': 'BF16', 'shape': [128]}})
    shapes = {name: tuple(value['shape']) for name, value in source.items()}
    def contract(_):
        return shapes, active, unused
    monkeypatch.setattr(target, 'text_tensor_contract', contract)
    monkeypatch.setattr(dense_loader, 'text_tensor_contract', contract)
    config = {'quantization_config': {'quant_method': 'modelopt', 'config_groups': {'g': {
        'targets': modules, 'weights': {'num_bits': 4, 'type': 'float', 'group_size': 16},
        'input_activations': {'num_bits': 4, 'type': 'float', 'group_size': 16, 'dynamic': False}}}}}
    output = {name: value.copy() for name, value in source.items() if name not in active | unused}
    for name in modules:
        for suffix, dtype, shape in [('weight_packed', 'U8', [128, 64]), ('weight_scale', 'F8_E4M3', [128, 8]),
                                     ('weight_scale_2', 'F32', []), ('input_scale', 'F32', [1])]:
            output[name+'.'+suffix] = {'dtype': dtype, 'shape': shape}
    monkeypatch.setattr(target, 'read_config', lambda _: config)
    monkeypatch.setattr(target, 'E2BConfig', SimpleNamespace(from_architecture=lambda _: 'cfg'))
    monkeypatch.setattr(target, 'headers', lambda _: output)
    return source, output, unused


def test_actual_packed_descriptor_all258_and_bf16_exception_inventory(inventory_case):
    source, output, unused = inventory_case
    actual, discarded = target._inventory(source, 'cfg', None, 'omitted')
    assert actual is output and discarded == unused
    output.update({name: source[name] for name in unused})
    assert target._inventory(source, 'cfg', None, 'retained-original-bf16')[0] is output


def test_native_linear_bf16_exception_must_agree_with_exported_policy(inventory_case):
    source, output, _ = inventory_case
    name = 'protected_linear.weight'
    source[name] = output[name] = {'dtype': 'BF16', 'shape': [128, 128]}
    target._inventory(source, 'cfg', None, 'omitted', bf16_linears=('protected_linear',))
    target.read_config(None)['quantization_config']['config_groups']['g']['targets'].append('protected_linear')
    with pytest.raises(ValueError, match='disagree'):
        target._inventory(source, 'cfg', None, 'omitted', bf16_linears=('protected_linear',))


@pytest.mark.parametrize('change', ['missing-scale', 'partial-active', 'bf16-exception', 'tie-alias', 'unused'])
def test_real_descriptor_and_complete_inventory_reject_malformed_export(inventory_case, change):
    source, output, unused = inventory_case
    if change == 'missing-scale':
        output.pop(next(name for name in output if name.endswith('.weight_scale')))
    elif change == 'partial-active':
        prefix = 'model.language_model.layers.0.mlp.gate_proj.'
        for name in list(output):
            if name.startswith(prefix):
                del output[name]
        output[prefix+'weight'] = {'dtype': 'BF16', 'shape': [128, 128]}
    elif change == 'bf16-exception':
        output['model.language_model.embed_tokens.weight']['dtype'] = 'F16'
    elif change == 'tie-alias':
        output['lm_head.weight'] = {'dtype': 'BF16', 'shape': [4, 128]}
    else:
        output.update({name: source[name] for name in unused})
    with pytest.raises(ValueError):
        target._inventory(source, 'cfg', None, 'omitted')
