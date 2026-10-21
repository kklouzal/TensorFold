"""Prepared engine scalar factors keep native eager and captured projection bits."""

import struct

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native FP4 calibration execution requires SM12.x CUDA", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint as ck
from tensorfold.cuda.nvfp4.format import prepare_native_calibration
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear


def pair(kind, n, k, seed):
    gen = torch.Generator().manual_seed(seed)
    raw_inverse = struct.unpack("<f", struct.pack("<I", 0x3FB7F13D))[0] if kind == "ct" else None
    act = 1. / raw_inverse if raw_inverse is not None else .037
    scale = .013
    prepared = prepare_native_calibration(act, scale, name=kind, act_inverse=raw_inverse)
    if kind == "fp8":
        codes = torch.randint(0x20, 0x60, (n, k), dtype=torch.uint8, generator=gen).cuda().view(torch.float8_e4m3fn)
        return (Fp8Linear.from_checkpoint(codes, scale, act=act),
                Fp8Linear.from_checkpoint(codes, scale, act=act, prepared=prepared))
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen).cuda()
    blocks = torch.randint(0x20, 0x40, (n, k // 16), dtype=torch.uint8, generator=gen).cuda()
    return (Fp4Linear.from_checkpoint(codes, blocks, scale, act=act, act_inverse=raw_inverse),
            Fp4Linear.from_checkpoint(codes, blocks, scale, act=act, act_inverse=prepared.act_inverse,
                                      prepared=prepared))


@pytest.mark.parametrize("kind", ["modelopt", "ct", "fp8"])
def test_prepared_native_projection_and_graph_equal_uncached_oracle(kind, monkeypatch):
    raw, prepared = pair(kind, 128, 256, 1701)
    x = torch.randn((17, 256), generator=torch.Generator().manual_seed(3903)).bfloat16().cuda()
    expected_lane, expected_prompt = raw(x), raw.prefill(x)
    torch.cuda.synchronize()
    monkeypatch.setattr(ck, "alpha", lambda *args: pytest.fail("prepared path repeated scalar alpha"))
    monkeypatch.setattr(ck, "_inv", lambda *args: pytest.fail("prepared path repeated scalar inverse"))
    assert torch.equal(prepared(x), expected_lane)
    assert torch.equal(prepared.prefill(x), expected_prompt)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prepared(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, expected_lane)


@pytest.mark.parametrize("kind", ["modelopt", "ct"])
def test_fused_mlp_consumes_prepared_gate_up_and_down_factors(kind, monkeypatch):
    raw, prepared = [], []
    for seed in (1701, 2802, 3903):
        left, right = pair(kind, 128, 128, seed)
        raw.append(left)
        prepared.append(right)
    x = torch.randn((17, 128), generator=torch.Generator().manual_seed(4004)).bfloat16().cuda()
    expected = ck.mlp_prompt(x, *raw)
    assert expected is not None
    torch.cuda.synchronize()
    monkeypatch.setattr(ck, "alpha", lambda *args: pytest.fail("prepared MLP repeated scalar alpha"))
    monkeypatch.setattr(ck, "_inv", lambda *args: pytest.fail("prepared MLP repeated scalar inverse"))
    actual = ck.mlp_prompt(x, *prepared)
    assert torch.equal(actual, expected)
