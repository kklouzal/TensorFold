"""CPU preparation/dispatch checks; CUDA NVFP4 numerical proof remains separate."""
from importlib import import_module
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
PackedProjection = import_module('tensorfold.families.gemma4.cuda.dense_packed').PackedProjection
PackedProjectionSpec = import_module('tensorfold.families.gemma4.cuda.dense_packed').PackedProjectionSpec
DTYPES = import_module('tensorfold.families.gemma4.cuda.dense_packed').DTYPES
Fp4Linear = import_module('tensorfold.cuda.nvfp4.linear').Fp4Linear
LinearMetadata = import_module('tensorfold.cuda.nvfp4.format').LinearMetadata
effective_config = import_module('tensorfold.checkpoint_quantization').effective_config


def descriptor(prescale=True, awq=True):
    config = {'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4_AWQ' if awq else 'NVFP4'}}
    block = effective_config(config, None, where='AWQ-control')['quantization_config']
    headers = {'weight': ('U8', [128, 64]), 'weight_scale': ('F8_E4M3', [128, 8]),
               'input_scale': ('F32', [1]), 'weight_scale_2': ('F32', [1])}
    if prescale:
        headers['pre_quant_scale'] = 'BF16', [128]
    spec = PackedProjectionSpec.from_header('mlp.down_proj', headers, LinearMetadata(block), expected_shape=(128, 128))
    return spec, headers


def operation():
    return Fp4Linear(torch.zeros(2, 2, 8, 32, 2, dtype=torch.int32),
                     torch.zeros(2, 2, 64, 4, dtype=torch.uint8), .5, 128, 128, act=.25)


@pytest.fixture
def fake_cuda(monkeypatch):
    # CPU data + metadata-only CUDA label; no compiler, model or GPU work.
    monkeypatch.setattr(torch.Tensor, 'device', property(lambda value: torch.device('cpu' if value.numel() == 1 else 'cuda:0')))


@pytest.mark.parametrize('present', [False, True])
def test_spec_optional_parts_and_owned_scale_buffer_footprint(fake_cuda, present):
    spec, _ = descriptor(present)
    scale = torch.ones(128, dtype=torch.bfloat16) if present else None
    op = operation()
    module = PackedProjection(spec, op, SimpleNamespace(prompt=False), pre_quant_scale=scale)
    assert spec.pre_quant_scale is present and module.pre_quant_scale is scale
    assert not hasattr(module, 'weight') and list(module.parameters()) == []
    buffers = dict(module.named_buffers())
    assert set(buffers) == ({'words', 'bs', 'pre_quant_scale'} if present else {'words', 'bs'})
    assert sum(value.numel()*value.element_size() for value in buffers.values()) == op.nbytes()+(256 if present else 0)


@pytest.mark.parametrize('prompt', [False, True])
@torch.inference_mode()
def test_bf16_rounding_precedes_quantization_for_prompt_and_decode_without_input_mutation(fake_cuda, monkeypatch, prompt):
    spec, _ = descriptor()
    scale = torch.ones(128, dtype=torch.bfloat16)
    scale[:2] = 1.0078125
    value = torch.zeros(1, 128, dtype=torch.bfloat16)
    value[0, :3] = torch.tensor([1.0078125, -1.0078125, -0.0], dtype=torch.bfloat16)
    original = value.clone()
    captured = []

    def consume(op, rows):
        captured.append(rows.clone())
        return rows.clone()

    monkeypatch.setattr(Fp4Linear, '__call__', consume)
    monkeypatch.setattr(Fp4Linear, 'prefill', consume)
    module = PackedProjection(spec, operation(), SimpleNamespace(prompt=prompt), pre_quant_scale=scale)
    output = module(value)
    assert captured[0].dtype == torch.bfloat16
    # 1.0078125**2 is1.01568603515625; BF16 must round BEFORE FP4's quantizer.
    # Expected raw BF16 words are independent of a repeated tensor multiply.
    assert captured[0].view(torch.int16)[0, :3].tolist() == [0x3f82, -0x407e, -0x8000]
    assert torch.equal(output.view(torch.int16), captured[0].view(torch.int16))
    assert torch.equal(value.view(torch.int16), original.view(torch.int16))


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_invalid_scale_values_are_rejected_without_repair(fake_cuda, bad):
    spec, _ = descriptor()
    scale = torch.ones(128, dtype=torch.bfloat16)
    scale[0] = bad
    with pytest.raises(ValueError, match='finite positive') as caught:
        PackedProjection(spec, operation(), SimpleNamespace(prompt=False), pre_quant_scale=scale)
    assert scale[0].isnan() if bad != bad else scale[0].item() == bad
    frame = caught.value.__traceback__
    while frame is not None and frame.tb_frame.f_code.co_name != '__init__':
        frame = frame.tb_next
    assert frame is not None
    assert frame.tb_frame.f_locals['pre_quant_scale'] is None
    assert frame.tb_frame.f_locals['operation'] is None


@pytest.mark.parametrize('scale', [torch.ones(128), torch.ones(1, 128, dtype=torch.bfloat16),
                                   torch.ones(256, dtype=torch.bfloat16)[::2], None])
def test_scale_shape_dtype_contiguity_and_presence_are_not_coerced(fake_cuda, scale):
    spec, _ = descriptor()
    with pytest.raises(ValueError, match='AWQ|input scale'):
        PackedProjection(spec, operation(), SimpleNamespace(prompt=False), pre_quant_scale=scale)


@torch.inference_mode()
def test_legacy_plain_nvfp4_keeps_original_input_bytes(fake_cuda, monkeypatch):
    spec, _ = descriptor(False, awq=False)
    value = torch.randn(2, 128, dtype=torch.bfloat16)
    seen = []
    monkeypatch.setattr(Fp4Linear, '__call__', lambda op, rows: seen.append(rows) or rows)
    module = PackedProjection(spec, operation(), SimpleNamespace(prompt=False))
    assert module.pre_quant_scale is None
    assert torch.equal(module(value), value) and seen[0].data_ptr() == value.data_ptr()


def test_from_checkpoint_binds_actual_prescale_without_changing_global_calibrations(fake_cuda, monkeypatch):
    spec, headers = descriptor()
    payload = {suffix: torch.ones(shape, dtype=torch.float32).to(DTYPES[dtype])
               for suffix, (dtype, shape) in headers.items()}
    payload['input_scale'].fill_(4)
    payload['weight_scale_2'].fill_(2)
    seen = []

    def prepare(cls, weight, scales, global_scale, *, act, act_inverse):
        seen.append((global_scale, act, act_inverse))
        op = operation()
        op.scale, op.act, op.act_inverse = global_scale, act, act_inverse
        return op

    monkeypatch.setattr(Fp4Linear, 'from_checkpoint', classmethod(prepare))
    module = PackedProjection.from_checkpoint(spec, payload, SimpleNamespace(prompt=False))
    assert seen == [(2., 4., None)]
    assert module.pre_quant_scale is payload['pre_quant_scale']
