"""Packed serving boundary/ownership controls; no GPU or full model allocation."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig  # noqa: E402
from tensorfold.families.gemma4.cuda import dense_engine_storage as storage  # noqa: E402
from tensorfold.families.gemma4.cuda import dense_loader  # noqa: E402
from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget  # noqa: E402
from tensorfold.families.gemma4.dense_tensor_contract import text_tensor_contract  # noqa: E402

FIXTURES = Path(__file__).parent / 'fixtures'


def packed_fixture():
    raw = json.loads((FIXTURES/'gemma4_e4b_quant_plan_config.json').read_text())
    cfg = E2BConfig.from_dict(raw)
    contract, active, _ = text_tensor_contract(cfg)
    raw['quantization_config'] = {
        'quant_method': 'modelopt', 'quant_algo': 'NVFP4', 'group_size': 16,
        'config_groups': {'core': {'targets': sorted(name.removesuffix('.weight') for name in active),
                                  'weights': {'num_bits': 4, 'type': 'float', 'group_size': 16},
                                  'input_activations': {'num_bits': 4, 'type': 'float', 'dynamic': False}}},
    }
    inventory = json.loads((FIXTURES/'gemma4_e4b_quant_plan_headers.json').read_text())
    for name in active:
        n, k = contract[name]
        module = name.removesuffix('.weight')
        inventory[name] = {'dtype': 'U8', 'shape': [n, k//2]}
        inventory[module+'.weight_scale'] = {'dtype': 'F8_E4M3', 'shape': [n, k//16]}
        for suffix in ('weight_scale_2', 'input_scale'):
            inventory[module+'.'+suffix] = {'dtype': 'F32', 'shape': []}
    return raw, inventory


def setup_headers(monkeypatch, raw, inventory):
    monkeypatch.setattr(storage, 'read_config', lambda _: raw)
    monkeypatch.setattr(storage, 'headers', lambda _: inventory)
    monkeypatch.setattr(dense_loader, '_manifest', lambda *_: {'unused_shared_kv_policy': 'retained-original-bf16'})


def test_storage_aware_metadata_keeps_original_bf16_boundary():
    raw, _ = packed_fixture()
    before = copy.deepcopy(raw)
    assert E2BConfig.from_checkpoint(raw).profile == 'E4B'
    assert raw == before
    with pytest.raises(ValueError, match='unquantized'):
        E2BConfig.from_dict(raw)
    del raw['quantization_config']
    assert E2BConfig.from_checkpoint(raw) == E2BConfig.from_dict(raw)


@pytest.mark.parametrize('patch', [
    {'quant_method': 'compressed-tensors'}, {'quant_algo': 'FP8'},
    {'kv_cache_scheme': {'num_bits': 8}}, {'transform_config': {'x': 1}},
])
def test_metadata_refuses_other_storage_before_runtime(patch):
    raw, _ = packed_fixture()
    raw['quantization_config'].update(patch)
    with pytest.raises(ValueError):
        E2BConfig.from_checkpoint(raw)


def test_header_estimate_covers_prepared_core_without_cpu_model_placeholders(monkeypatch):
    raw, inventory = packed_fixture()
    setup_headers(monkeypatch, raw, inventory)
    import transformers
    original = transformers.Gemma4ForConditionalGeneration
    observed_devices = set()

    def construct(config):
        model = original(config)
        observed_devices.update(value.device.type for value in (*model.parameters(), *model.buffers()))
        return model

    monkeypatch.setattr(transformers, 'Gemma4ForConditionalGeneration', construct)
    plan = storage.inspect_packed_storage(Path('unused'))
    assert observed_devices == {'meta'}
    assert len(plan.projection_shapes) == 258
    cfg = E2BConfig.from_dict({k: v for k, v in raw.items() if k != 'quantization_config'})
    contract, active, unused = text_tensor_contract(cfg)
    packed = {name.removesuffix('.weight') for name in active}
    noncore = sum(2*math.prod(v['shape']) for n, v in inventory.items()
                  if n not in unused and n.rsplit('.', 1)[0] not in packed)
    prepared = sum(((n+63)//64)*64*k*9//16 for n, k in plan.projection_shapes)
    assert plan.resident == noncore + prepared + plan.native_buffer_bound
    # Native parameter skeleton is META, not a full CPU BF16 model mirror.
    assert 2*(64 << 20) <= plan.host_staging < noncore
    assert plan.device_staging > 0
    assert plan.workspace(512) >= plan.workspace(8) > 0


def test_header_refuses_partial_core_and_quantized_exclusion(monkeypatch):
    raw, inventory = packed_fixture()
    setup_headers(monkeypatch, raw, inventory)
    inventory['model.language_model.embed_tokens.weight']['dtype'] = 'F8_E4M3'
    with pytest.raises(ValueError, match='non-core'):
        storage.inspect_packed_storage(Path('unused'))


def setup_target(monkeypatch, raw, failure=None, install_failure=False, unsafe_load=False):
    import tensorfold.checkpoint_metadata as metadata
    from tensorfold.families.gemma4.cuda import e2b_runtime, e2b_policy, e2b_attention, e2b_norm
    calls = []
    weight = torch.nn.Parameter(torch.ones(8, 4, dtype=torch.bfloat16), requires_grad=False)
    embedding = SimpleNamespace(weight=weight)
    model = SimpleNamespace(
        get_input_embeddings=lambda: embedding, lm_head=SimpleNamespace(weight=weight),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=[])),
        parameters=lambda: iter([weight]),
    )
    monkeypatch.setattr(metadata, 'read_config', lambda _: raw)
    monkeypatch.setattr(e2b_runtime, 'require_runtime', lambda *_: None)
    monkeypatch.setattr(storage, 'inspect_packed_storage', lambda _: calls.append('inspect'))
    monkeypatch.setattr(e2b_policy, 'E2BBlasPolicy', lambda _: SimpleNamespace(close=lambda: calls.append('close')))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *_: calls.append('sync'))

    def load(*_, **kwargs):
        calls.append('load')
        kwargs['status'].safe_to_release = not unsafe_load
        assert kwargs['device'] == 'cuda:0' and kwargs['prompt_policy'].prompt
        if failure is not None:
            raise failure
        return SimpleNamespace(model=model, receipt={'fully_consumed': True})

    def attention(*_):
        calls.append('attention')
        if install_failure:
            raise RuntimeError('install failed')

    monkeypatch.setattr(dense_loader, 'load_dense_quantized', load)
    monkeypatch.setattr(e2b_attention, 'install_attention', attention)
    monkeypatch.setattr(e2b_norm, 'install_row_norms', lambda *_: calls.append('norms'))
    return model, calls


def test_target_adopts_packed_owner_without_bf16_to_or_mirror(monkeypatch, tmp_path):
    raw, _ = packed_fixture()
    model, calls = setup_target(monkeypatch, raw)
    target = E2BTarget(tmp_path)
    assert target.model is model and target.embedding_weight is target.head_weight
    assert target.consumption_receipt == {'fully_consumed': True}
    assert calls == ['inspect', 'load', 'attention', 'norms']
    target.close()
    assert calls[-1] == 'close'


@pytest.mark.parametrize('terminal,marker', [(False, False), (True, True), (True, False)])
def test_packed_loader_failure_preserves_primary_and_terminal_lease(monkeypatch, tmp_path, terminal, marker):
    raw, _ = packed_fixture()
    primary = RuntimeError('loader failed')
    if marker:
        primary._tensorfold_retained_owners = [object()]
    _, calls = setup_target(monkeypatch, raw, primary, unsafe_load=terminal)
    with pytest.raises(RuntimeError) as caught:
        E2BTarget(tmp_path)
    assert caught.value is primary
    assert calls == (['inspect', 'load'] if terminal else ['inspect', 'load', 'close'])
    if terminal:
        assert any(isinstance(owner, E2BTarget) for owner in primary._tensorfold_retained_owners)


def test_packed_late_install_failure_drains_before_release(monkeypatch, tmp_path):
    raw, _ = packed_fixture()
    _, calls = setup_target(monkeypatch, raw, install_failure=True)
    with pytest.raises(RuntimeError, match='install failed'):
        E2BTarget(tmp_path)
    assert calls[-2:] == ['sync', 'close']


@pytest.mark.parametrize('projection', ['route', 'padded'])
def test_incompatible_projection_refused_before_lease_or_load(monkeypatch, tmp_path, projection):
    raw, _ = packed_fixture()
    _, calls = setup_target(monkeypatch, raw)
    with pytest.raises(ValueError, match='BF16 projection overrides'):
        E2BTarget(tmp_path, projection=projection)
    assert calls == []


def test_family_and_vision_metadata_use_same_dense_storage_gate():
    from tensorfold.families.gemma4 import check_metadata
    from tensorfold.vision.config import validate_vision_config
    raw, _ = packed_fixture()
    check_metadata(raw, 'cuda')
    assert validate_vision_config(raw, 'gemma4') is raw['vision_config']
    raw['quantization_config']['quant_algo'] = 'FP8'
    with pytest.raises(ValueError):
        check_metadata(raw, 'cuda')
    with pytest.raises(ValueError):
        validate_vision_config(raw, 'gemma4')


@pytest.mark.parametrize('uma', [False, True])
def test_capacity_charges_packed_staging_and_workspace_without_reuse_credit(monkeypatch, uma):
    from tensorfold.cuda import capacity
    from tensorfold.families.gemma4.cuda import e2b_engine
    raw, inventory = packed_fixture()
    cfg = E2BConfig.from_checkpoint(raw)
    plan = storage.PackedStorage(1000, 200, 300, ((64, 64),), 16)
    captured = {}
    monkeypatch.setattr(capacity, 'headers', lambda _: inventory)
    monkeypatch.setattr(capacity, 'unified', lambda _: uma)
    host_checks = []
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda value: host_checks.append(value))

    def admit(*args, **kwargs):
        weights, host = kwargs['weight_estimator'](Path('unused'))
        captured.update(weights=weights, host=host, geometry=args[4].bytes_at(32))
        return captured

    monkeypatch.setattr(capacity, 'admit', admit)
    e2b_engine.admit(Path('unused'), cfg, 32, True, torch, None, 1, False, packed_storage=plan)
    assert captured['weights'].resident == 1000
    assert captured['weights'].staging == 200 + (300 if uma else 0)
    assert captured['host'] == 300
    assert host_checks == [300]
    packed_geometry = captured['geometry']
    e2b_engine.admit(Path('unused'), cfg, 32, True, torch, None, 1, False)
    assert packed_geometry - captured['geometry'] == plan.workspace(512)
    original = sum(capacity.itemsize(info, name)*math.prod(info['shape']) for name, info in inventory.items())
    assert captured['weights'].resident == captured['weights'].staging == captured['host'] == original


def test_cli_family_modelopt_gate_and_full_precision_refusal(monkeypatch, tmp_path):
    from tensorfold import families
    from tensorfold.cuda import precision
    from tensorfold.families.gemma4.cuda.e2b_engine import E2BEngine
    import tensorfold.checkpoint_metadata as metadata
    raw, _ = packed_fixture()
    family = families.Family('gemma4', 'Gemma 4', 'tensorfold.families.gemma4', True)
    families.require_readable(family, raw, 'cuda')
    monkeypatch.setattr(metadata, 'read_config', lambda _: raw)
    with precision.using(precision.FULL), pytest.raises(ValueError, match='precision checkpoint'):
        E2BEngine(tmp_path)


@pytest.mark.parametrize('section,key', [('vision_config', 'intermediate_size'),
                                        ('audio_config', 'num_hidden_layers')])
def test_untrusted_media_growth_refused_before_any_native_constructor(monkeypatch, section, key):
    import transformers
    raw, inventory = packed_fixture()
    raw[section][key] *= 1000000
    setup_headers(monkeypatch, raw, inventory)
    monkeypatch.setattr(transformers, 'Gemma4ForConditionalGeneration',
                        lambda *_: pytest.fail('unbounded config reached constructor'))
    with pytest.raises(ValueError, match='media allocation profile'):
        storage.inspect_packed_storage(Path('unused'))


def test_missing_media_tensor_refused_by_allocation_free_native_schema(monkeypatch):
    raw, inventory = packed_fixture()
    setup_headers(monkeypatch, raw, inventory)
    del inventory[next(name for name in inventory if name.startswith('model.audio_tower.') and name.endswith('.weight'))]
    with pytest.raises(ValueError, match='complete native'):
        storage.inspect_packed_storage(Path('unused'))


def test_packed_host_cgroup_refusal_precedes_device_plan_and_target(monkeypatch):
    from tensorfold.cuda import capacity
    from tensorfold.families.gemma4.cuda import e2b_engine
    raw, _ = packed_fixture()
    cfg = E2BConfig.from_checkpoint(raw)
    plan = storage.PackedStorage(1000, 200, 300, ((64, 64),), 16)
    monkeypatch.setattr(capacity, 'headers', lambda _: pytest.fail('packed repeated dead header scan'))
    monkeypatch.setattr(capacity, 'unified', lambda _: True)
    monkeypatch.setattr(capacity, '_meminfo', lambda: {'MemTotal': 1024**4, 'MemAvailable': 1024**4})
    monkeypatch.setattr(capacity, 'host_cgroup_bytes', lambda **_: 299)
    monkeypatch.setattr(capacity, 'admit', lambda *_, **__: pytest.fail('host refusal reached device plan'))
    with pytest.raises(ValueError, match='cgroup allowance'):
        e2b_engine.admit(Path('unused'), cfg, 32, True, torch, None, 1, False, packed_storage=plan)


@pytest.mark.parametrize('retirement_fails,transport_fails', [(False, False), (True, False), (False, True)])
def test_actual_loader_status_records_retirement_without_exception_annotations(monkeypatch, tmp_path, retirement_fails, transport_fails):
    import transformers
    from tensorfold.families.gemma4.cuda import e2b_runtime
    raw, _ = packed_fixture()
    # This status test models the owned-manifest protocol, not publisher input.
    (tmp_path/dense_loader.MANIFEST).write_text('{}')
    cfg = E2BConfig.from_checkpoint(raw)
    status = dense_loader.DenseLoadStatus()
    primary = RuntimeError('bind failed')
    native_cause, native_context = ValueError('native cause'), KeyError('native context')
    primary.__cause__, primary.__context__ = native_cause, native_context
    calls = []
    monkeypatch.setattr(dense_loader, 'read_config', lambda _: raw)
    monkeypatch.setattr(dense_loader, 'headers', lambda _: {})
    monkeypatch.setattr(dense_loader, '_file_hash', lambda _: 'a'*64)
    monkeypatch.setattr(dense_loader, '_manifest', lambda *_: {'metadata_sha256': {}, 'tensor_sha256': {},
                                                            'unused_shared_kv_policy': 'omitted'})
    monkeypatch.setattr(dense_loader, '_projection_specs', lambda *_: {})
    monkeypatch.setattr(e2b_runtime, 'require_runtime', lambda *_: None)
    monkeypatch.setattr(transformers, 'Gemma4ForConditionalGeneration', lambda *_: torch.nn.Identity())

    class Source:
        device_touched = 'cuda:0'
        consumed = {}

        def __init__(self, *_):
            pass

        def close(self):
            calls.append('reader-close')

    monkeypatch.setattr(dense_loader, '_VerifiedTensors', Source)

    def bind(*_, **__):
        assert not status.safe_to_release and not status.transferred
        assert status.model is not None and isinstance(status.source, Source)
        raise primary

    def sync(*_):
        calls.append('sync')
        if retirement_fails:
            raise RuntimeError('drain failed')

    def rollback(_owner, error, retire):
        # Model the documented annotation-exhaustion branch: NO exception
        # marker is published. The separate status must remain authoritative.
        if transport_fails:
            raise MemoryError('rollback preparation exhausted')
        try:
            retire()
        finally:
            raise error

    monkeypatch.setattr(dense_loader, '_bind_native', bind)
    monkeypatch.setattr(dense_loader, 'rollback', rollback)
    monkeypatch.setattr(torch.cuda, 'synchronize', sync)
    with pytest.raises(RuntimeError) as caught:
        dense_loader.load_dense_quantized(tmp_path, device='cuda:0',
                                          prompt_policy=SimpleNamespace(prompt=True), status=status)
    assert caught.value is primary
    if transport_fails:
        frames = []
        tb = caught.value.__traceback__
        while tb:
            if tb.tb_frame.f_code is dense_loader.load_dense_quantized.__code__:
                frames.append(tb.tb_frame.f_locals)
            tb = tb.tb_next
        assert frames and frames[0]['_native_cause'] is native_cause
        assert frames[0]['_native_context'] is native_context
        assert isinstance(caught.value.__cause__, MemoryError)
    assert not hasattr(primary, '_tensorfold_retained_owners')
    assert status.safe_to_release is not (retirement_fails or transport_fails)
    assert not status.transferred
    assert calls == ([] if transport_fails else ['sync', 'reader-close'])
    if retirement_fails or transport_fails:
        assert status.model is not None and isinstance(status.source, Source)
    else:
        assert status.model is status.source is None
    assert cfg.profile == 'E4B'  # Real architecture parser used, no model weights.
