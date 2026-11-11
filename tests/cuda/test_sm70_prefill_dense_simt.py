"""Direct-Half SIMT component; bound remains conditional on emitted RN32 proof."""
from fractions import Fraction
import json

import pytest
import torch

from tensorfold.cuda.sm70 import prefill_dense_simt as simt, prefill_dense as pedantic
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
from test_sm70_gguf_dense32 import _frozen_m32
from tests.sm70_native_rn_oracle import bits_to_fraction, gamma, round_fraction

pytestmark = pytest.mark.skipif(not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0), reason='requires actual pinned SM70')


def exact_samples(x, weight, out):
    records = []
    for row, column in [(0, 0), (x.shape[0] - 1, weight.shape[0] - 1)]:
        a = [bits_to_fraction(int(v) & 65535, 16) for v in x[row].view(torch.int16).cpu().tolist()]
        b = [bits_to_fraction(int(v) & 65535, 16) for v in weight[column].view(torch.int16).cpu().tolist()]
        products = [u * v for u, v in zip(a, b)]
        gold, sumabs = sum(products, Fraction()), sum(map(abs, products), Fraction())
        e32 = gamma(x.shape[1]) * sumabs
        assert abs(gold) + e32 < 65520  # Fixture must certify finite Half; no threshold fitted to observations.
        bound = e32 + Fraction(1, 2048) * (abs(gold) + e32) + Fraction(1, 2**25)
        value = bits_to_fraction(int(out[row, column].view(torch.int16).item()) & 65535, 16)
        assert abs(value - gold) <= bound
        records.append({'row': row, 'column': column, 'exact_dot': str(gold), 'sumabs': str(sumabs),
                        'E32': str(e32), 'combined_absolute_bound': str(bound), 'observed_error': str(abs(value-gold)),
                        'condition': 'requires independently verified emitted RN32 classical DAG'})
    return records


@pytest.mark.parametrize('rows', [64, 256, 1024, 4096])
@pytest.mark.parametrize('columns,outputs', [(10240, 320), (10240, 4), (320, 10240)])
def test_actual_hc_shapes_direct_half_conditional_enclosures_and_controls(rows, columns, outputs, record_property):
    generator = torch.Generator(device='cuda').manual_seed(3917 + rows + outputs)
    x = torch.randn((rows, columns), device='cuda', dtype=torch.float16, generator=generator) * .1
    weight = torch.randn((outputs, columns), device='cuda', dtype=torch.float16, generator=generator) * .02
    width, offset = (324, 320 if outputs == 4 else 0) if columns == 10240 else (outputs, 0)
    backing = torch.full((rows + 2, width), -713., device='cuda', dtype=torch.float16)
    out = backing[1:-1, offset:offset + outputs]
    old = _frozen_m32(GGUFDense(weight), x, torch.float16)
    accumulator = torch.empty((rows, outputs), device='cuda')
    control = torch.empty_like(out)
    with pedantic.prepare(torch.empty(4 * 1024**2, device='cuda', dtype=torch.uint8)) as p:
        p.run(x, weight, accumulator, control)
        p.finish()
    del accumulator  # Candidate owns no corresponding Float32 allocation.
    before = out._version
    tracked_before = torch.cuda.memory_allocated()
    with simt.prepare(x, weight, out) as plan:
        settings = plan.settings()
        assert settings['workspace_bytes'] == settings['additional_device_tensor_bytes'] == 0
        assert settings['split_k_slices'] == settings['warp_k_partitions'] == 1
        assert settings['instruction_shape'] == (1, 1, 1)
        tracked_prepared = torch.cuda.memory_allocated()
        assert tracked_prepared == tracked_before  # Tensor storage only; module/driver/shared resources are separate.
        plan.run(x, weight, out)
        plan.finish()
    assert out._version == before + 1
    assert torch.isfinite(out).all()
    assert (backing[:1] == -713).all() and (backing[-1:] == -713).all()
    assert (backing[1:-1, :offset] == -713).all() and (backing[1:-1, offset + outputs:] == -713).all()
    record_property('prefill_simt_diagnostic', json.dumps({
        'R': rows, 'K': columns, 'N': outputs, 'settings': settings,
        'tracked_tensor_bytes_before_after_prepare': [tracked_before, tracked_prepared],
        'old_M32_unequal_half_bits': int((out != old).sum().item()),
        'pedantic_unequal_half_bits': int((out != control).sum().item()),
        'old_M32_max_absolute': float((out.float()-old.float()).abs().max().item()),
        'pedantic_max_absolute': float((out.float()-control.float()).abs().max().item()),
        'exact_samples': exact_samples(x, weight, out),
        'binary_DAG_verified_by_this_test': False, 'quality_acceptance': False, 'P3_valid': False}, sort_keys=True))


@pytest.mark.parametrize('rows,columns,outputs', [(1, 1, 1), (3, 7, 4), (31, 9, 7), (63, 31, 17), (65, 33, 5)])
def test_dyadic_uniform_tails_and_no_beta_read(rows, columns, outputs):
    x = torch.full((rows, columns), .125, device='cuda', dtype=torch.float16)
    weight = torch.full((outputs, columns), .5, device='cuda', dtype=torch.float16)
    out = torch.full((rows, outputs), float('nan'), device='cuda', dtype=torch.float16)
    with simt.prepare(x, weight, out) as plan:
        plan.run(x, weight, out)
        plan.finish()
    expected = round_fraction(Fraction(columns, 16), 16)
    assert (out.view(torch.int16).to(torch.int32).bitwise_and(65535) == expected).all()


@pytest.mark.parametrize('values', [[1., 2**-11], [1., 2**-11, 2**-23], [65504., 8.], [65504., 16.],
                                    [-2**-24], [-0., -0.]])
def test_declared_rn16_midpoint_overflow_and_signedzero(values):
    x = torch.tensor([values], device='cuda', dtype=torch.float16)
    weight = torch.ones((4, len(values)), device='cuda', dtype=torch.float16)
    if values == [-2**-24]:
        weight.fill_(2**-24)
    out = torch.empty((1, 4), device='cuda', dtype=torch.float16)
    # Independent RN32 scalar oracle; these tiny explicit vectors have a known ordered sum.
    acc = 0
    for a, b in zip(x[0].cpu().tolist(), weight[0].cpu().tolist()):
        acc = round_fraction(bits_to_fraction(acc) + Fraction(a) * Fraction(b))
    expected = round_fraction(bits_to_fraction(acc), 16)
    with simt.prepare(x, weight, out) as plan:
        plan.run(x, weight, out)
        plan.finish()
    assert (out.view(torch.int16).to(torch.int32).bitwise_and(65535) == expected).all()


def test_nondefault_stream_changed_live_values_and_binding_refusal():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.ones((65, 9), device='cuda').half()
        weight = torch.ones((4, 9), device='cuda').half()
        out = torch.empty((65, 4), device='cuda').half()
        with simt.prepare(x, weight, out) as plan:
            for value in [0., -.125, .5]:
                x.fill_(value)
                plan.run(x, weight, out)
                plan.finish()
                assert (out == value * 9).all()
            with pytest.raises(RuntimeError, match='binding'):
                plan.run(x.clone(), weight, out)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize('fault', ['shape', 'dtype', 'input-stride', 'out-alias', 'inference-output', 'K-bound'])
def test_actual_native_preparation_refuses_unsupported_region(fault):
    x, weight = torch.ones((64, 320), device='cuda').half(), torch.ones((4, 320), device='cuda').half()
    out = torch.empty((64, 4), device='cuda').half()
    if fault == 'shape':
        out = out[:, :3]
    elif fault == 'dtype':
        out = out.float()
    elif fault == 'input-stride':
        x = torch.ones((64, 640), device='cuda').half()[:, ::2]
    elif fault == 'out-alias':
        out = x[:, :4]
    elif fault == 'inference-output':
        with torch.inference_mode():
            out = out.clone()
    else:
        x, weight = torch.ones((1, 10241), device='cuda').half(), torch.ones((4, 10241), device='cuda').half()
        out = out[:1]
    with pytest.raises(RuntimeError):
        simt.prepare(x, weight, out)


@pytest.mark.parametrize('value,weight_value,kind', [(float('inf'), 1., 'positive-inf'),
    (-float('inf'), 1., 'negative-inf'), (float('nan'), 1., 'nan'), (0., float('inf'), 'nan')])
def test_nonfinite_classification_keeps_payload_differences_diagnostic(value, weight_value, kind, record_property):
    x = torch.full((3, 1), value, device='cuda', dtype=torch.float16)
    weight = torch.full((4, 1), weight_value, device='cuda', dtype=torch.float16)
    out = torch.empty((3, 4), device='cuda', dtype=torch.float16)
    old = _frozen_m32(GGUFDense(weight), x, torch.float16)
    with simt.prepare(x, weight, out) as plan:
        plan.run(x, weight, out)
        plan.finish()
    if kind == 'nan':
        assert torch.isnan(out).all()
    else:
        assert torch.isinf(out).all()
        assert (torch.signbit(out) == (kind == 'negative-inf')).all()
    record_property('nonfinite_payload_observation', json.dumps({
        'kind': kind, 'SIMT_half_bits': out.view(torch.int16).cpu().tolist(),
        'old_M32_half_bits': old.view(torch.int16).cpu().tolist(),
        'payload_identity_contract': False, 'finite_bound_applies': False}))


def test_every_half_input_encoding_unit_dot_and_rn_zero_sign():
    values = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(torch.float16)
    weight = torch.ones((1, 1), device='cuda', dtype=torch.float16)
    for first in range(0, 65536, 4096):
        x = values[first:first+4096].view(4096, 1)
        out = torch.empty_like(x)
        with simt.prepare(x, weight, out) as plan:
            plan.run(x, weight, out)
            plan.finish()
        finite = torch.isfinite(x)
        expected_bits = x.view(torch.int16).clone()
        expected_bits[x == 0] = 0  # RN32 starts+0; +0 + (-0 product) gives+0.
        assert torch.equal(out.view(torch.int16)[finite], expected_bits[finite])
        assert torch.isnan(out[torch.isnan(x)]).all()
        assert torch.equal(torch.signbit(out[torch.isinf(x)]), torch.signbit(x[torch.isinf(x)]))
        assert torch.isinf(out[torch.isinf(x)]).all()


@pytest.mark.parametrize('mode', ['product', 'accum320', 'accum10240'])
def test_adverse_vectors_expose_any_hidden_half_product_or_accumulator(mode):
    columns = 2 if mode == 'product' else int(mode.removeprefix('accum'))
    x = torch.ones((3, columns), device='cuda', dtype=torch.float16)
    weight = torch.ones((4, columns), device='cuda', dtype=torch.float16)
    if mode == 'product':
        x[:, 0] = 1 + 2**-9
        weight[:, 0], weight[:, 1] = 1 + 2**-9, -1
        gold = Fraction(1, 256) + Fraction(1, 2**18)
    else:
        x[:, 0] = 2048
        gold = Fraction(2048 + columns - 1)
    out = torch.empty((3, 4), device='cuda', dtype=torch.float16)
    with simt.prepare(x, weight, out) as plan:
        plan.run(x, weight, out)
        plan.finish()
    expected = round_fraction(gold, 16)
    assert (out.view(torch.int16).to(torch.int32).bitwise_and(65535) == expected).all()


def test_uncaptured_component_refuses_capture_without_launch_or_fallback():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x, weight = torch.ones((3, 7), device='cuda').half(), torch.ones((4, 7), device='cuda').half()
        out = torch.full((3, 4), -713., device='cuda', dtype=torch.float16)
        with simt.prepare(x, weight, out) as plan:
            graph = torch.cuda.CUDAGraph()
            with pytest.raises(RuntimeError, match='uncaptured'):
                with torch.cuda.graph(graph, stream=stream):
                    plan.run(x, weight, out)
            assert (out == -713).all()
    torch.cuda.current_stream().wait_stream(stream)
