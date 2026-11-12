"""Actual selected Triton3.2 startup launcher and original FP32 FMA graph law."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import triton_driver
from tensorfold.families.qwen4_exp.cuda import glue
from tensorfold.families.qwen4_exp.cuda.engine import STARTUP_SECONDS
from test_flashnext_moe_partial import fused_partial, partial_case

pytestmark = pytest.mark.skipif(
    os.environ.get(triton_driver.PROFILE_ENV) != triton_driver.PROFILE32
    or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires the explicit installed Triton3.2 SM70 SDK profile")


def test_startup_realizes_production_launcher_before_live_fma_and_graph_replay():
    prepared = triton_driver.prepare_driver(timeout_s=STARTUP_SECONDS, fma_geometry=(2560, 10, 11))
    assert prepared['profile'] == triton_driver.PROFILE32
    assert prepared['fma']['signature'] == {'Y': '*fp32', 'WTS': '*fp32', 'OUT': '*fp32'}
    assert prepared['fma']['compiler_options'] == {'ptx_version': 83}
    before = prepared['fma']['launcher']
    rows = 7
    values, weights = partial_case(torch.float32, rows, width=2560)
    y, wts = torch.from_numpy(values).cuda(), torch.from_numpy(weights).cuda()
    out = torch.full((rows + 2, 2560), 913.25, device='cuda', dtype=torch.float32)
    glue.moe_partial(y, wts, out, rows)
    np.testing.assert_array_equal(out[:rows].cpu().numpy().view(np.uint32),
                                  fused_partial(values, weights, rows).view(np.uint32))

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=side):
        glue.moe_partial(y, wts, out, rows)
    torch.cuda.current_stream().wait_stream(side)
    # A captured launch reads changed caller storage, with the same ordered
    # FP32 fused oracle and untouched inactive rows as ordinary dispatch.
    weights[:, :10] *= np.float32(-1)
    wts.copy_(torch.from_numpy(weights))
    graph.replay()
    result = out.cpu().numpy()
    np.testing.assert_array_equal(result[:rows].view(np.uint32),
                                  fused_partial(values, weights, rows).view(np.uint32))
    np.testing.assert_array_equal(result[rows:], np.full((2, 2560), 913.25, dtype=np.float32))
    from triton.runtime.jit import MockTensor

    compiled = glue._moe_partial.warmup(*(MockTensor(torch.float32) for _ in range(3)),
        D=2560, TOPK=10, SLOTS=11, BLOCK=256, num_warps=2, grid=(1, 10), **glue._MOE_COMPILER_OPTIONS)
    assert compiled.metadata.ptx_version == 83
    assert triton_driver._mapped_code_identity(compiled.run.launch,
        Path(prepared['cache']).resolve(), '__triton_launcher') == before
    torch.cuda.synchronize()
    graph.reset()
