"""Packed boundary metadata/ownership and fake-operator CPU controls, no CUDA."""
import gc
from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip('torch')
from tensorfold.cuda.nvfp4.format import LinearMetadata  # noqa: E402
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear, Fp8BlockLinear  # noqa: E402
from tensorfold.families.gemma4.cuda.dense_packed import PackedProjection, PackedProjectionSpec  # noqa: E402


def spec(kind='nvfp4', activation=True, reciprocal=False):
    if kind == 'nvfp4':
        headers = {'weight_packed': ('U8', [128, 64]), 'weight_scale': ('F8_E4M3', [128, 8]),
                   'weight_global_scale' if reciprocal else 'weight_scale_2': ('F32', [])}
        weights = {'num_bits': 4, 'type': 'float', 'group_size': 16}
        acts = {'num_bits': 4, 'type': 'float', 'group_size': 16, 'dynamic': False} if activation else None
    else:
        headers = {'weight': ('F8_E4M3', [128, 128])}
        weights = {'num_bits': 8, 'type': 'float'}
        acts = None
        if kind == 'fp8block':
            headers['weight_scale_inv'] = 'F32', [1, 1]
            weights['block_structure'] = [128, 128]
        elif kind == 'fp8channel':
            headers['weight_scale'] = 'F32', [128]
            weights['strategy'] = 'channel'
            acts = {'num_bits': 8, 'type': 'float', 'strategy': 'token', 'dynamic': True} if activation else None
        else:
            headers['weight_scale'] = 'F32', []
            acts = {'num_bits': 8, 'type': 'float', 'dynamic': False} if activation else None
    if acts is not None and kind != 'fp8channel':
        headers['input_global_scale' if reciprocal else 'input_scale'] = 'F32', [1]
    metadata = LinearMetadata({'quant_method': 'compressed-tensors' if reciprocal else 'modelopt',
                               'config_groups': {'g': {'targets': ['Linear'], 'weights': weights, 'input_activations': acts}}})
    return PackedProjectionSpec.from_header('model.language_model.layers.0.mlp.gate_proj', headers, metadata,
                                           expected_shape=(128, 128)), headers, metadata


@pytest.mark.parametrize('kind,activation', [('nvfp4', False), ('nvfp4', True), ('fp8', False), ('fp8', True),
                                           ('fp8channel', False), ('fp8channel', True), ('fp8block', False)])
def test_explicit_storage_and_activation_descriptors(kind, activation):
    descriptor, _, _ = spec(kind, activation)
    assert descriptor.kind == kind and descriptor.shape == (128, 128)
    assert (descriptor.activation_bits is not None) is activation


@pytest.mark.parametrize('change', ['extra', 'shape', 'missing-scale', 'activation-scale-missing'])
def test_storage_contract_refused_before_any_packing(change):
    _, headers, metadata = spec()
    expected = (128, 128)
    if change == 'extra':
        headers['bias'] = 'BF16', [128]
    elif change == 'shape':
        expected = (128, 256)
    elif change == 'missing-scale':
        del headers['weight_scale']
    else:
        del headers['input_scale']
    with pytest.raises(ValueError):
        PackedProjectionSpec.from_header('mlp', headers, metadata, expected_shape=expected)


def test_fp8_row64_cannot_masquerade_as_block128():
    _, headers, metadata = spec('fp8block', False)
    headers['weight_scale_inv'] = 'F32', [128, 2]
    with pytest.raises(ValueError, match='128x128'):
        PackedProjectionSpec.from_header('mlp', headers, metadata, expected_shape=(128, 128))


def op(descriptor):
    if descriptor.kind == 'nvfp4':
        return Fp4Linear(torch.zeros(2, 2, 8, 32, 2, dtype=torch.int32), torch.zeros(2, 2, 64, 4, dtype=torch.uint8),
                         .5, 128, 128, act=.25 if descriptor.activation_bits else None)
    if descriptor.kind == 'fp8block':
        return Fp8BlockLinear(torch.zeros(16384, dtype=torch.uint8), torch.zeros(1024, dtype=torch.uint8),128,128,128)
    return Fp8Linear(torch.zeros(16384, dtype=torch.uint8), .5,128,128,128,
                     act=.25 if descriptor.activation_bits and descriptor.kind == 'fp8' else None,
                     cols=torch.ones(128) if descriptor.kind == 'fp8channel' else None,
                     token=descriptor.activation_bits is not None and descriptor.kind == 'fp8channel')


@pytest.fixture
def fake_cuda(monkeypatch):
    # Metadata/control fixture only: underlying tensors and dispatcher remain
    # CPU. No CUDA initialization, kernels or numerical performance are tested.
    monkeypatch.setattr(torch.Tensor, 'device', property(lambda _: torch.device('cuda:0')))


@pytest.mark.parametrize('kind', ['nvfp4', 'fp8', 'fp8channel', 'fp8block'])
def test_owned_buffers_are_exact_operator_aliases_and_no_floating_weight_facade(fake_cuda, kind):
    descriptor, _, _ = spec(kind, kind != 'fp8block')
    operation = op(descriptor)
    module = PackedProjection(descriptor, operation, SimpleNamespace(prompt=False))
    assert not hasattr(module, 'weight') and list(module.parameters()) == []
    assert module.bias is None and module.compute_dtype is torch.bfloat16
    assert all(getattr(operation, name) is value for name, value in module.named_buffers())
    assert not any(value.dtype == torch.bfloat16 for value in module.buffers())
    with pytest.raises(RuntimeError, match='immutable'):
        module.to('cpu')
    with pytest.raises(RuntimeError, match='immutable'):
        module.load_state_dict(module.state_dict())


@pytest.mark.parametrize('shape', [(128,), (3, 128), (2, 3, 128), (0, 2, 128)])
@pytest.mark.parametrize('prompt', [False, True])
@torch.inference_mode()
def test_arbitrary_leading_dims_noncontiguous_empty_and_prompt_routing(fake_cuda, monkeypatch, shape, prompt):
    descriptor, _, _ = spec()
    policy = SimpleNamespace(prompt=prompt)
    calls = []
    weight = torch.arange(128*128, dtype=torch.float32).reshape(128, 128).to(torch.bfloat16)/1024

    def run(_operation, rows, label):
        assert rows.ndim == 2 and rows.is_contiguous()
        calls.append(label)
        return torch.nn.functional.linear(rows, weight)

    monkeypatch.setattr(Fp4Linear, '__call__', lambda self, rows: run(self, rows, 'decode'))
    monkeypatch.setattr(Fp4Linear, 'prefill', lambda self, rows: run(self, rows, 'prompt'))
    module = PackedProjection(descriptor, op(descriptor), policy)
    value = torch.arange(2*max(1, int(torch.tensor(shape).prod())), dtype=torch.float32).to(torch.bfloat16)[::2]
    value = value[:int(torch.tensor(shape).prod())].reshape(shape)
    actual = module(value)
    assert actual.shape == (*shape[:-1], 128)
    assert torch.equal(actual, torch.nn.functional.linear(value, weight))
    assert calls == ([] if value.numel() == 0 else ['prompt' if prompt else 'decode'])


@torch.inference_mode()
def test_fp8block_never_follows_process_prompt_fp8_requantization(fake_cuda, monkeypatch):
    descriptor, _, _ = spec('fp8block', False)
    module = PackedProjection(descriptor, op(descriptor), SimpleNamespace(prompt=True))
    monkeypatch.setattr(Fp8BlockLinear, '__call__', lambda self, rows: rows.clone())
    monkeypatch.setattr(Fp8BlockLinear, 'prefill', lambda *_: pytest.fail('undeclared activation quantization'))
    assert torch.equal(module(torch.ones(2, 128, dtype=torch.bfloat16)), torch.ones(2, 128, dtype=torch.bfloat16))


def test_actual_cpu_storage_rejected_without_attempting_gpu_pack():
    descriptor, _, _ = spec()
    with pytest.raises(ValueError, match='CUDA'):
        PackedProjection(descriptor, op(descriptor), SimpleNamespace(prompt=False))


@torch.inference_mode()
def test_invalid_activation_or_compute_metadata_cannot_silently_cast(fake_cuda):
    descriptor, _, _ = spec()
    operation = op(descriptor)
    operation.act = None
    with pytest.raises(ValueError, match='activation math'):
        PackedProjection(descriptor, operation, SimpleNamespace(prompt=False))
    module = PackedProjection(descriptor, op(descriptor), SimpleNamespace(prompt=False))
    for value in (torch.ones(2,128), torch.ones(2,64,dtype=torch.bfloat16), torch.tensor(1,dtype=torch.bfloat16)):
        with pytest.raises(ValueError):
            module(value)


def test_failed_registration_releases_owned_partial_buffer_aliases(fake_cuda, monkeypatch):
    descriptor, _, _ = spec()
    operation = op(descriptor)
    refs = [weakref.ref(operation.words), weakref.ref(operation.bs)]
    original = PackedProjection.register_buffer
    failure = RuntimeError('injected registration failure')

    def register(self, name, tensor, **kwargs):
        if name == 'bs':
            raise failure
        return original(self, name, tensor, **kwargs)

    monkeypatch.setattr(PackedProjection, 'register_buffer', register)
    with pytest.raises(RuntimeError) as caught:
        PackedProjection(descriptor, operation, SimpleNamespace(prompt=False))
    assert caught.value is failure
    operation = None
    gc.collect()
    # Test's injected callback itself retains its tensor argument through its
    # traceback; inspect only the successfully registered first owner.
    assert refs[0]() is None


@pytest.mark.parametrize('reciprocal', [False, True])
@pytest.mark.parametrize('activation', [False, True])
def test_loading_preserves_stored_codes_and_global_conventions(monkeypatch, reciprocal, activation):
    descriptor, headers, _ = spec(activation=activation, reciprocal=reciprocal)
    payload = {}
    from tensorfold.families.gemma4.cuda.dense_packed import DTYPES
    for suffix, (dtype, shape) in headers.items():
        payload[suffix] = torch.ones(shape, dtype=torch.float32).to(DTYPES[dtype])
    payload['weight_global_scale' if reciprocal else 'weight_scale_2'].fill_(2)
    if activation:
        payload['input_global_scale' if reciprocal else 'input_scale'].fill_(4)
    # CPU scalar calibrations, fake CUDA matrix metadata only. No real GPU work.
    monkeypatch.setattr(torch.Tensor, 'device', property(lambda value: torch.device('cpu' if value.numel() == 1 else 'cuda:0')))
    calls = []

    def prepare(cls, weight, scales, scale, *, act, act_inverse):
        calls.append((weight is payload['weight_packed'], scales is payload['weight_scale'], scale, act, act_inverse))
        operation = op(descriptor)
        operation.scale, operation.act, operation.act_inverse = scale, act, act_inverse
        return operation

    monkeypatch.setattr(Fp4Linear, 'from_checkpoint', classmethod(prepare))
    module = PackedProjection.from_checkpoint(descriptor, payload, SimpleNamespace(prompt=False))
    assert calls == [(True, True, .5 if reciprocal else 2,
                      (.25 if reciprocal else 4) if activation else None,
                      4 if reciprocal and activation else None)]
    assert module._operation.words is module.words
    assert all(value is not payload['weight_packed'] for value in module.buffers())


def test_loading_invalid_payload_refuses_before_kernel_or_pack(monkeypatch):
    descriptor, headers, _ = spec()
    from tensorfold.families.gemma4.cuda.dense_packed import DTYPES
    payload = {suffix: torch.ones(shape, dtype=torch.float32).to(DTYPES[dtype])
               for suffix, (dtype, shape) in headers.items()}
    monkeypatch.setattr(Fp4Linear, 'from_checkpoint', lambda *_: pytest.fail('invalid payload reached kernel pack'))
    with pytest.raises(ValueError, match='CUDA'):
        PackedProjection.from_checkpoint(descriptor, payload, SimpleNamespace(prompt=False))
    payload['weight_packed'] = payload['weight_packed'].float()
    with pytest.raises(ValueError, match='dtype/shape'):
        PackedProjection.from_checkpoint(descriptor, payload, SimpleNamespace(prompt=False))


@pytest.mark.parametrize('field,value', [('scale', float('nan')), ('scale', 0), ('act', float('inf')), ('act', -1)])
def test_prepared_fp8_globals_are_validated_at_public_constructor(fake_cuda, field, value):
    descriptor, _, _ = spec('fp8', True)
    operation = op(descriptor)
    setattr(operation, field, value)
    with pytest.raises(ValueError, match='finite|positive'):
        PackedProjection(descriptor, operation, SimpleNamespace(prompt=False))


def test_native_architecture_validation_does_not_strip_or_admit_quantized_storage():
    import copy
    from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig
    from test_gemma4_e4b import e4b_config
    raw = e4b_config()
    raw['quantization_config'] = {'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}
    original = copy.deepcopy(raw)
    cfg = E2BConfig.from_architecture(raw)
    assert cfg.profile == 'E4B' and cfg.shared_layers == 18 and raw == original
    with pytest.raises(ValueError, match='unquantized'):
        E2BConfig.from_dict(raw)
    raw['text_config']['num_key_value_heads'] = 1
    with pytest.raises(ValueError):
        E2BConfig.from_architecture(raw)
