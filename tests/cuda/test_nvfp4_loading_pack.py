"""Actual GPU byte packing, admitted allocation peaks and transient ownership."""

import gc
import json
import struct
import weakref

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("GPU byte packing requires CUDA", allow_module_level=True)

from test_nvfp4_a4_host_control import oracle_blob

from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4 import experts_a4
from tensorfold.cuda.nvfp4.host_experts import _pack_authority_projection_a4
from tensorfold.families.qwen3_5.cuda.nvfp4_memory import projection_loading_bytes


def payload(n, k, *, seed=1701):
    generator = torch.Generator().manual_seed(seed + n * 31 + k * 17)
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
    # General bit permutation deliberately covers all encodings, independently
    # of the native-A4 consumer's separately enforced unsigned finite contract.
    scales = torch.randint(0, 256, (n, k // 16), dtype=torch.uint8, generator=generator)
    # Keep every encoding represented without periodic identical rows. Gate,
    # up and down also receive distinct local seeds in their combined oracle.
    scales.reshape(-1)[:256] = torch.randperm(256, generator=generator).to(torch.uint8)
    assert torch.unique(codes, dim=0).shape[0] == n
    assert torch.equal(torch.unique(scales), torch.arange(256, dtype=torch.uint8))
    return codes, scales.view(torch.float8_e4m3fn)


def peak_begin():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("n,k", [(64, 64), (512, 2048), (2048, 512)])
def test_ct_gpu_permutation_equals_independent_cpu_bytes_and_releases_device_storage(n, k):
    codes, scales = payload(n, k)
    expected = oracle_blob(codes, scales)
    baseline = peak_begin()
    got = _pack_authority_projection_a4(codes, scales, torch.device("cuda:0"))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    stem = "model.layers.0.mlp.experts.0.gate_proj"
    fields = {stem + ".weight_packed": {"dtype": "U8", "shape": [n, k // 2]},
              stem + ".weight_scale": {"dtype": "F8_E4M3", "shape": [n, k // 16]}}
    admitted = projection_loading_bytes(fields)[stem]
    assert got.device.type == "cpu" and torch.equal(got, expected)
    assert 0 < peak <= admitted
    assert torch.cuda.memory_allocated() == baseline
    print(json.dumps({"format": "ct-native-a4", "n": n, "k": k, "peak_allocated_bytes": peak,
                      "admitted_loading_bytes": admitted, "post_call_device_bytes": 0}))


@pytest.mark.parametrize("width,dims", [(128, 512), (512, 2048), (512, 2560), (1024, 2560)])
def test_modelopt_gpu_pack_matches_cpu_original_bytes_and_admitted_peak(width, dims):
    original = [payload(width, dims, seed=1701), payload(width, dims, seed=2802), payload(dims, width, seed=3903)]
    globals_ = [torch.tensor([struct.unpack("f", struct.pack("I", bits))[0]], dtype=torch.float32)
                for bits in (0x3fb7f13d, 0x3e63ffff, 0x3d013579)]
    cpu = nvx.make(*(tuple(t.unsqueeze(0) for t in values) + (scale,)
                     for values, scale in zip(original, globals_)))
    baseline = peak_begin()

    def packed():
        triples = [(values[0].unsqueeze(0).cuda(), values[1].unsqueeze(0).cuda(), scale)
                   for values, scale in zip(original, globals_)]
        bank = nvx.make(*triples)
        assert bank.up_scale.device.type == bank.down_scale.device.type == "cpu"
        result = tuple(getattr(bank, name).cpu() for name in ("up", "down", "up_scale", "down_scale"))
        return result

    got = packed()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    entry = 3 * (width // 32) * (dims // 32) * 144 * 4 + 32
    admitted = max(64 << 20, 128 * width * dims + 8 * entry)
    for result, name in zip(got, ("up", "down", "up_scale", "down_scale")):
        assert torch.equal(result, getattr(cpu, name))
    assert got[2].view(torch.int32)[0].tolist() == [value.view(torch.int32)[0].item() for value in globals_[:2]]
    assert got[3].view(torch.int32)[0, 0] == globals_[2].view(torch.int32)[0]
    assert 0 < peak <= admitted
    assert torch.cuda.memory_allocated() == baseline
    print(json.dumps({"format": "modelopt-w4a16", "width": width, "dims": dims,
                      "peak_allocated_bytes": peak, "admitted_loading_bytes": admitted,
                      "post_call_device_bytes": 0, "shape_scope": "synthetic component coverage"}))


def test_failed_ct_gpu_pack_retains_exception_and_releases_transients_after_outcome(monkeypatch):
    codes, scales = payload(512, 2048)
    baseline = peak_begin()
    observed = []

    def fail(weight, block_scale):
        observed.extend((weakref.ref(weight), weakref.ref(block_scale)))
        raise RuntimeError("injected GPU packing failure")

    with monkeypatch.context() as patch:
        patch.setattr(experts_a4, "pack_projection_a4", fail)
        with pytest.raises(RuntimeError, match="injected GPU packing failure") as outcome:
            _pack_authority_projection_a4(codes, scales, "cuda:0")
        # Python exception tracebacks own their arguments until the outcome is
        # released. They may not be silently detached from the primary failure.
        assert all(ref() is not None for ref in observed)
        del outcome
    gc.collect()
    torch.cuda.synchronize()
    assert all(ref() is None for ref in observed)
    assert torch.cuda.memory_allocated() == baseline
    got = _pack_authority_projection_a4(codes, scales, "cuda:0")
    assert torch.equal(got, oracle_blob(codes, scales))
    assert torch.cuda.memory_allocated() == baseline
