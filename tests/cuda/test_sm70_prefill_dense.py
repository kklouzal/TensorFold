"""Unselected larger-M component: original M32 observations and exact scalar gold."""
from fractions import Fraction
import json

import pytest
import torch

from tensorfold.cuda.sm70 import prefill_dense as dense

pytestmark = pytest.mark.skipif(not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0), reason='requires actual SM70')


def original_m32(x, weight):
    result = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float16, device=x.device)
    for start in range(0, x.shape[0], 32):
        count = min(32, x.shape[0] - start)
        padded = torch.zeros((32, x.shape[1]), device=x.device, dtype=torch.float16)
        padded[:count].copy_(x[start:start + count])
        result[start:start + count].copy_(torch.mm(padded, weight.T)[:count])
    return result


def exact_sample_errors(x, weight, accumulation):
    # Half and Float32 values convert exactly to Python binary64. Only Fraction
    # performs products and sums: this is NOT a claim of exact Float64 dot sums.
    errors = []
    for row, column in [(0, 0), (x.shape[0] - 1, weight.shape[0] - 1)]:
        a = x[row].cpu().tolist()
        b = weight[column].cpu().tolist()
        gold = sum((Fraction(v) * Fraction(w) for v, w in zip(a, b)), Fraction())
        actual = Fraction(accumulation[row, column].item())
        errors.append({'row': row, 'column': column, 'absolute_error': str(abs(actual - gold)),
                       'exact_dot': str(gold)})
    return errors


@pytest.mark.parametrize('rows', [64, 256, 1024, 4096])
@pytest.mark.parametrize('columns,outputs', [(10240, 320), (10240, 4), (320, 10240)])
def test_actual_hc_shapes_report_errors_without_fitted_acceptance(rows, columns, outputs, record_property):
    generator = torch.Generator(device='cuda').manual_seed(3917 + rows + outputs)
    x = torch.randn((rows, columns), device='cuda', dtype=torch.float16, generator=generator) * .1
    weight = torch.randn((outputs, columns), device='cuda', dtype=torch.float16, generator=generator) * .02
    # Preserve actual HC320+4 Half column views; up uses its contiguous width.
    accumulation_store = torch.full((rows, outputs + 4), -711., device='cuda')
    accumulation = accumulation_store[:, 2:-2]
    output_width = 324 if columns == 10240 else outputs
    output_offset = 320 if outputs == 4 else 0
    output_store = torch.full((rows + 2, output_width), -713., device='cuda', dtype=torch.float16)
    out = output_store[1:-1, output_offset:output_offset + outputs]
    baseline = original_m32(x, weight)
    policy = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.allow_fp16_accumulation,
              torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    versions = accumulation._version, out._version
    workspace = torch.empty(4 * 1024**2, dtype=torch.uint8, device='cuda')  # Component fixture, not runtime funding.
    with dense.prepare(workspace) as plan:
        settings = plan.settings()
        module = dense.extension()
        assert settings['math_mode'] == module.math_mode
        assert settings['pointer_mode'] == module.pointer_mode and settings['atomics_mode'] == module.atomics_mode
        plan.run(x, weight, accumulation, out)
        plan.finish()
    assert (accumulation._version, out._version) == (versions[0] + 1, versions[1] + 1)
    assert policy == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.allow_fp16_accumulation,
                      torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    assert torch.isfinite(accumulation).all() and torch.isfinite(out).all()
    assert torch.equal(out.contiguous().view(torch.int16), accumulation.half().view(torch.int16))
    assert (accumulation_store[:, :2] == -711).all() and (accumulation_store[:, -2:] == -711).all()
    assert (output_store[:1] == -713).all() and (output_store[-1:] == -713).all()
    assert (output_store[1:-1, :output_offset] == -713).all()
    assert (output_store[1:-1, output_offset + outputs:] == -713).all()
    metrics = {'rows': rows, 'K': columns, 'N': outputs, 'mode': settings,
               'out_stride': list(out.stride()), 'accumulation_stride': list(accumulation.stride()),
               'old_m32_half_mismatches': int((out != baseline).sum().item()),
               'old_m32_max_absolute': float((out.float() - baseline.float()).abs().max().item()),
               'independent_exact_samples': exact_sample_errors(x, weight, accumulation),
               'quality_acceptance': False, 'P3_valid': False, 'vendor_unconditional_bound': False}
    record_property('prefill_dense_diagnostic', json.dumps(metrics, sort_keys=True))


@pytest.mark.parametrize('columns', [320, 10240])
def test_uniform_dyadic_dot_is_exact_and_current_stream_owned(columns):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.full((64, columns), .125, device='cuda', dtype=torch.float16)
        weight = torch.full((4, columns), .5, device='cuda', dtype=torch.float16)
        accumulation, out = torch.empty((64, 4), device='cuda'), torch.empty((64, 4), device='cuda').half()
        with dense.prepare(torch.empty(4 * 1024**2, dtype=torch.uint8, device='cuda')) as plan:
            plan.run(x, weight, accumulation, out)
            plan.finish()
    torch.cuda.current_stream().wait_stream(stream)
    assert (accumulation == Fraction(columns, 16).numerator / Fraction(columns, 16).denominator).all()
    assert torch.equal(out.view(torch.int16), accumulation.half().view(torch.int16))


@pytest.mark.parametrize('fault', ['dtype', 'shape', 'column-stride', 'weight-stride', 'alias-out',
                                  'alias-workspace', 'inference-output', 'stream'])
def test_native_abi_refuses_unsupported_region_before_success(fault):
    x, weight = torch.ones((64, 320), device='cuda').half(), torch.ones((4, 320), device='cuda').half()
    accumulation, out = torch.empty((64, 4), device='cuda'), torch.empty((64, 4), device='cuda').half()
    workspace = torch.empty(4 * 1024**2, dtype=torch.uint8, device='cuda')
    if fault == 'dtype':
        x = x.float()
    elif fault == 'shape':
        weight = weight[:, :319].contiguous()
    elif fault == 'column-stride':
        x = torch.ones((64, 640), device='cuda').half()[:, ::2]
    elif fault == 'weight-stride':
        weight = torch.ones((4, 640), device='cuda').half()[:, ::2]
    elif fault == 'alias-out':
        out = accumulation.view(torch.float16)[:, :4]
    elif fault == 'alias-workspace':
        accumulation = workspace[:64*4*4].view(torch.float32).view(64, 4)
    elif fault == 'inference-output':
        with torch.inference_mode():
            out = out.clone()
    with dense.prepare(workspace) as plan:
        if fault == 'stream':
            with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match='stream'):
                plan.run(x, weight, accumulation, out)
        else:
            with pytest.raises(RuntimeError):
                plan.run(x, weight, accumulation, out)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_actual_byte_row_helper_keeps_320_plus4_disjoint_and_detects_crossing(device, dtype):
    owner = torch.empty((65, 324), dtype=dtype, device=device)
    left, right = owner[:, :320], owner[:, 320:]
    overlap = dense.extension()._overlaps
    assert not overlap(left, right) and not overlap(right, left)
    assert overlap(left, owner[:, 319:321])
    assert overlap(right, owner[:, 319:321])
    assert overlap(left, left)
    assert not overlap(left[:0], left)
    with pytest.raises(RuntimeError, match='contiguous rows'):
        overlap(owner[:, ::2], owner)
