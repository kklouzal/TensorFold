"""Private logical-M HC operator/capture oracle; no model-quality tolerance."""
import json
import os
import gc
from pathlib import Path

import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.families.qwen4_exp.cuda import hc_smallrow_vendor as module
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat
from tensorfold.families.qwen4_exp.cuda.capture_lifetime import on_stream
from tensorfold.cleanup import raise_failures

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                               reason='actual SM70 HC vendor candidate')


@pytest.mark.parametrize('rows', (1, 2, 3))
@pytest.mark.parametrize('role', ('down', 'down-inject', 'up'))
@torch.inference_mode()
def test_direct_vendor_bits_original_weight_alias_and_capture_replay(monkeypatch, rows, role):
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False
    assert str(torch.backends.cuda.preferred_blas_library()) == '_BlasBackend.Cublas'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    assert gguf_ops.extension().prepare_hc_smallrow() == 3
    monkeypatch.setattr(GGUFDense, 'implementation', 'native32')
    monkeypatch.setattr(GGUFConcat, 'implementation', 'native32')
    torch.manual_seed(1051 + rows)
    k, widths = (320, (10240,)) if role == 'up' else (10240, (320, 4) if role == 'down-inject' else (320,))
    parts = tuple(GGUFDense(torch.randn((n, k), dtype=torch.float16, device='cuda') * .01) for n in widths)
    original = GGUFConcat(parts) if len(parts) == 2 else parts[0]
    candidate = module.Concat(original) if len(parts) == 2 else module.Dense(original, down=role != 'up')
    x = torch.randn((rows, k), dtype=torch.float16, device='cuda') * .03
    out = torch.empty((rows, sum(widths)), dtype=candidate.output_dtype, device='cuda')
    weights = tuple(part.weight for part in original.parts) if len(parts) == 2 else (original.weight,)
    assert all(actual is expected for actual, expected in zip(candidate.weights, weights))
    diagnostics = []
    cleanup_failures = [None]  # Reserve before capture can accept work or a primary failure.
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    primary = None
    try:
        candidate.matmul(x, out=out)  # Exact shape/provider warmed before capture.
        side.wait_stream(torch.cuda.current_stream())
        def capture():
            secondary, body_failure = [None], None
            graph.capture_begin(capture_error_mode='thread_local')
            try:
                candidate.matmul(x, out=out)
            except BaseException as error:
                body_failure = error
            try:
                graph.capture_end()
            except BaseException as error:
                secondary[0] = error
            if secondary[0] is not None:
                raise_failures(body_failure, secondary)
            if body_failure is not None:
                raise body_failure
        enabled = gc.isenabled()
        gc.disable()
        try:
            on_stream(torch.cuda, side, capture)
        finally:
            if enabled:
                gc.enable()
        torch.cuda.current_stream().wait_stream(side)
        for scale in (1., -.5, .125):
            x.mul_(scale)
            expected = torch.cat([(torch.mm(x, weight.T, out_dtype=torch.float32)
                if role != 'up' else torch.mm(x, weight.T)) for weight in weights], dim=1)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
            reference = original.matmul(x, f32=role != 'up')
            error = (out.float() - reference.float()).abs()
            diagnostics.append({'scale': scale, 'bit_exact_vs_original_M32': torch.equal(out.view(torch.uint8), reference.view(torch.uint8)),
                                'max_abs_vs_original_M32': error.max().item(),
                                'rms_vs_original_M32': error.square().mean().sqrt().item()})
        for generic_rows in (4, 32):
            higher = torch.randn((generic_rows, k), dtype=torch.float16, device='cuda') * .03
            assert torch.equal(candidate.matmul(higher, f32=role != 'up').view(torch.uint8),
                               original.matmul(higher, f32=role != 'up').view(torch.uint8))
        destination = os.environ.get('TENSORFOLD_HC_SMALLROW_EVIDENCE')
        if destination:
            Path(destination).mkdir(parents=True, exist_ok=True)
            Path(destination, f'{role}-r{rows}.json').write_text(json.dumps({'role': role, 'rows': rows,
                'strict_oracle': 'own direct ATen logical-M bits', 'M32_comparison': diagnostics,
                'scope': 'numerical characterization only; no invented model acceptance threshold'}) + '\n')
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            torch.cuda.synchronize()
            graph.reset()
        except BaseException as cleanup:
            cleanup_failures[0] = cleanup
            raise_failures(primary, cleanup_failures)


@torch.inference_mode()
def test_native_rejects_invalid_geometry_and_alias_before_output_mutation():
    operation = gguf_ops.extension().hc_smallrow
    x = torch.ones((2, 320), dtype=torch.float16, device='cuda')
    weight = torch.ones((10240, 320), dtype=torch.float16, device='cuda')
    output = torch.full((2, 10240), 7, dtype=torch.float16, device='cuda')
    with pytest.raises(RuntimeError, match='1..3'):
        operation(torch.ones((4, 320), dtype=torch.float16, device='cuda'), (weight,), output)
    assert bool((output == 7).all())
    with pytest.raises(RuntimeError, match='accepts only original down/inject/up geometries'):
        operation(x, (weight[:10],), output)
    assert bool((output == 7).all())
    before = weight.clone()
    alias = weight.view(-1)[:2 * 10240].view(2, 10240)
    with pytest.raises(RuntimeError, match='overlaps weight'):
        operation(x, (weight,), alias)
    assert torch.equal(weight, before)
