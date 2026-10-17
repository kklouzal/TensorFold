"""Selected ModelOpt GPU packing: exact CPU authority, peaks and failure lifetime."""

import gc
import json
import struct
import weakref
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("GPU byte packing requires CUDA", allow_module_level=True)

from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4.host_experts import load_nvfp4_host_experts

PREFIX = "model.layers.0.mlp"


def payload(n, k, *, seed):
    generator = torch.Generator().manual_seed(seed + n * 31 + k * 17)
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
    scales = torch.randint(0, 256, (n, k // 16), dtype=torch.uint8, generator=generator)
    # General packing preserves every encoding without interpreting F8 values.
    # Nonperiodic rows and independent projection seeds expose row/field swaps.
    scales.reshape(-1)[:256] = torch.randperm(256, generator=generator).to(torch.uint8)
    assert torch.unique(codes, dim=0).shape[0] == n
    assert torch.equal(torch.unique(scales), torch.arange(256, dtype=torch.uint8))
    return codes, scales.view(torch.float8_e4m3fn)


class Reader:
    """Owned CPU fixture with real selected-device transfers, not filesystem IO."""
    def __init__(self, width, dims):
        self.fields, self.reads = {}, []
        self.cfg = SimpleNamespace(experts=1, moe_width=width, hidden=dims)
        self.original = [payload(width, dims, seed=1701), payload(width, dims, seed=2802),
                         payload(dims, width, seed=3903)]
        self.globals = [torch.tensor([struct.unpack("f", struct.pack("I", bits))[0]], dtype=torch.float32)
                        for bits in (0x3fb7f13d, 0x3e63ffff, 0x3d013579)]
        for projection, values, scale in zip(("gate_proj", "up_proj", "down_proj"), self.original, self.globals):
            base = PREFIX + ".experts.0." + projection
            for suffix, value in (("weight", values[0]), ("weight_scale", values[1]), ("weight_scale_2", scale)):
                self.fields[base + "." + suffix] = value

    def has(self, name):
        return name in self.fields

    def info(self, name):
        value = self.fields[name]
        return {"shape": tuple(value.shape),
                "dtype": {torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}[value.dtype]}

    def get_rows(self, name, lo, hi, device="cpu"):
        self.reads.append((name, lo, hi, str(device)))
        return self.fields[name][lo:hi].to(device)


def peak_begin():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("width,dims", [(128, 512), (512, 2048), (512, 2560), (1024, 2560)])
def test_modelopt_gpu_authority_matches_cpu_original_bytes_and_admitted_peak(width, dims):
    reader = Reader(width, dims)
    cpu = nvx.make(*(tuple(t.unsqueeze(0) for t in values) + (scale,)
                     for values, scale in zip(reader.original, reader.globals)))
    baseline = peak_begin()
    bank = load_nvfp4_host_experts(reader, PREFIX, reader.cfg, "cuda:0")
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    entry = 3 * (width // 32) * (dims // 32) * 144 * 4 + 32
    admitted = max(64 << 20, 128 * width * dims + 8 * entry)
    for name in ("up", "down", "up_scale", "down_scale"):
        value = getattr(bank, name)
        assert value.device.type == "cpu" and torch.equal(value, getattr(cpu, name))
    assert bank.up_scale.view(torch.int32)[0].tolist() == [value.view(torch.int32)[0].item()
                                                        for value in reader.globals[:2]]
    assert bank.down_scale.view(torch.int32)[0, 0] == reader.globals[2].view(torch.int32)[0]
    assert len(reader.reads) == 9
    assert all(device == ("cpu" if name.endswith("weight_scale_2") else "cuda:0")
               for name, _, _, device in reader.reads)
    assert 0 < peak <= admitted
    assert torch.cuda.memory_allocated() == baseline
    print(json.dumps({"format": "modelopt-w4a16", "width": width, "dims": dims,
                      "peak_allocated_bytes": peak, "admitted_device_loading_bytes": admitted,
                      "post_call_device_bytes": 0, "shape_scope": "synthetic component coverage"}))


def test_modelopt_packing_failure_retains_primary_then_releases_all_device_transients(monkeypatch):
    reader = Reader(512, 2048)
    baseline = peak_begin()
    observed = []
    actual = nvx.make

    def fail(*triples):
        observed.extend(weakref.ref(value) for triple in triples for value in triple[:2])
        raise RuntimeError("injected ModelOpt GPU packing failure")

    with monkeypatch.context() as patch:
        patch.setattr(nvx, "make", fail)
        with pytest.raises(RuntimeError, match="injected ModelOpt GPU packing failure") as outcome:
            load_nvfp4_host_experts(reader, PREFIX, reader.cfg, "cuda:0")
        # The original exception tracebacks keep arguments and ownership until
        # the outcome is released; failure transport must not detach those links.
        assert all(ref() is not None for ref in observed)
        del outcome
    gc.collect()
    torch.cuda.synchronize()
    assert all(ref() is None for ref in observed)
    assert torch.cuda.memory_allocated() == baseline
    bank = load_nvfp4_host_experts(reader, PREFIX, reader.cfg, "cuda:0")
    expected = actual(*(tuple(t.unsqueeze(0) for t in values) + (scale,)
                        for values, scale in zip(reader.original, reader.globals)))
    assert torch.equal(bank.up, expected.up) and torch.equal(bank.down, expected.down)
    assert torch.cuda.memory_allocated() == baseline
