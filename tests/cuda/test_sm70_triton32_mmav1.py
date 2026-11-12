"""One actual-SM70 MMAv1 capability control, separate from scalar FMA semantics."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from tensorfold.cuda.sm70 import triton_driver
from tensorfold.cuda.sm70.triton_options import COMPILER_OPTIONS
from tensorfold.families.qwen4_exp.cuda.engine import STARTUP_SECONDS

pytestmark = pytest.mark.skipif(
    os.environ.get(triton_driver.PROFILE_ENV) != triton_driver.PROFILE32
    or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires the explicit installed Triton3.2 SDK and actual SM70")


@triton.jit
def _dot64(left, right, result):
    rows = tl.arange(0, 64)
    columns = tl.arange(0, 64)
    inner = tl.arange(0, 64)
    a = tl.load(left + rows[:, None] * 64 + inner[None, :])
    b = tl.load(right + inner[:, None] * 64 + columns[None, :])
    value = tl.dot(a, b)
    tl.store(result + rows[:, None] * 64 + columns[None, :], value)


def test_actual_sm70_triton32_dot_uses_volta_mma_and_preserves_integer_known_answer(record_property):
    # Use the actual installed SDK auditor: it binds selected compiler/libdevice
    # bytes and cache namespace, not a guessed compatible CUDA version string.
    path = Path('/opt/tensorfold-pins/source_foundation.py')
    spec = importlib.util.spec_from_file_location('selected_sm70_mmav1_guard', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    pins = guard.read_json(Path('/opt/tensorfold-pins/source-pins.json'))[0]
    provider = guard.triton_provider_identity(pins)
    assert provider['version'] == triton.__version__ == '3.2.0'
    prepared = triton_driver.prepare_driver(timeout_s=STARTUP_SECONDS, fma_geometry=(2560, 10, 11))
    from triton.runtime.jit import MockTensor

    with triton_driver._cache_lock(Path(prepared['cache']), STARTUP_SECONDS):
        compiled = _dot64.warmup(MockTensor(torch.float16), MockTensor(torch.float16),
                                 MockTensor(torch.float32), grid=(1,), num_warps=4, **COMPILER_OPTIONS)
        launcher = compiled.run  # Realize the vendor launcher under the same startup publication lock.
        launcher_identity = triton_driver._mapped_code_identity(
            launcher.launch, Path(prepared['cache']), '__triton_launcher')
    ptx, cubin = compiled.asm['ptx'], compiled.asm['cubin']
    assert compiled.metadata.target.backend == 'cuda' and compiled.metadata.target.arch == 70
    assert compiled.metadata.ptx_version == 83 and re.search(r'\.version\s+8\.3\b', ptx)
    assert re.search(r'\bmma\.sync(?:\.aligned)?\.m8n8k4\b', ptx), ptx
    i, k, j = np.arange(64)[:, None], np.arange(64)[None, :], np.arange(64)[None, :]
    left = ((i + 2 * k) % 7 - 3).astype(np.int64)
    right = ((2 * np.arange(64)[:, None] + j) % 5 - 2).astype(np.int64)
    # Every input and every possible partial integer sum is exactly representable
    # in Half/Float32. The independent Int64 result needs no numerical tolerance.
    expected = (left @ right).astype(np.float32)
    a, b = torch.from_numpy(left.astype(np.float16)).cuda(), torch.from_numpy(right.astype(np.float16)).cuda()
    result = torch.empty((64, 64), dtype=torch.float32, device='cuda')
    _dot64[(1,)](a, b, result, num_warps=4, **COMPILER_OPTIONS)
    np.testing.assert_array_equal(result.cpu().numpy(), expected)
    receipt = {'schema': 'tensorfold.sm70-triton32-mmav1-conformance-v1', 'shape': [64, 64, 64],
        'input': 'fp16 exact small integers', 'output': 'fp32', 'oracle': 'independent Int64 CPU matrix product',
        'actual_target': {'backend': compiled.metadata.target.backend, 'arch': compiled.metadata.target.arch},
        'compiler_options': dict(COMPILER_OPTIONS),
        'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(), 'ptx_bytes': len(ptx.encode()),
        'cubin_sha256': hashlib.sha256(cubin).hexdigest(), 'cubin_bytes': len(cubin),
        'volta_mma_m8n8k4_count': len(re.findall(r'\bmma\.sync(?:\.aligned)?\.m8n8k4\b', ptx)),
        'provider': provider, 'launcher': launcher_identity,
        'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'single native capability/known-answer case; scalar FMA and model/P3 acceptance separate',
        'P3_valid': False}
    record_property('mmav1_conformance', json.dumps(receipt, sort_keys=True))
