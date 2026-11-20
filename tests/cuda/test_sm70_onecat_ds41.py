"""DeepSeek-V4.1 1Cat dp4a experts and GGML dequantization on real checkpoint bytes.

The dequantization oracle is NumPy, written from llama.cpp ggml-quants.c
dequantize_row_* (FP32, one rounding per operation, no contraction) with the
grids parsed from ggml-common.h, independent of the vendored device codebooks.
The expert pipeline (Q8_1 activations, an explicit approximation) is compared
with an FP32 product of those FP16 weights; its error is reported, and only a
gross-defect screen (relative L2 < 0.1) is asserted, not an accuracy contract.
"""

from functools import lru_cache
from pathlib import Path
import os
import re

import numpy as np
import pytest
import torch

from gguf_fixture import blocks, scalar_reference
from tensorfold.gguf_file import GGUFCheckpoint

MODEL = Path(os.environ.get(
    "TF_DS41_GGUF",
    "/home/kklouzal/models/ds41-uncensored-gsq-rco/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"))
GGML_COMMON = Path(os.environ.get(
    "TF_GGML_COMMON_H",
    "/home/kklouzal/build-cache/friend-1cat-20261120/build/1cat-vllm-main/"
    "csrc/quantization/gguf_upstream/llama.cpp/ggml/src/ggml-common.h"))
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0)
    or not MODEL.is_file() or not GGML_COMMON.is_file(),
    reason="SM70 with the DeepSeek-V4.1 GGUF and llama.cpp ggml-common.h")

LOCAL = 8          # local experts per test bank (EP2 ranks hold 192; any E works)
BYTES = {16: 66, 17: 74, 29: 56}
F32 = np.float32


@lru_cache(maxsize=1)
def tables():
    text = GGML_COMMON.read_text()
    found = {}
    for kind, name, body in re.findall(r"GGML_TABLE_BEGIN\((u?int\d+_t), (\w+), \w+\)(.*?)GGML_TABLE_END", text, re.S):
        if name in found:
            continue
        values = [int(v, 0) for v in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", re.sub(r"//.*", "", body))]
        found[name] = np.array(values, dtype={"uint64": np.uint64, "uint32": np.uint32, "uint8": np.uint8,
                                              "int8": np.int8}[kind.removesuffix("_t")])
    return found


def grid(name):
    return tables()[name].view(np.uint8).reshape(-1, 8)


def f16(b, at):
    return np.ascontiguousarray(b[..., at:at + 2]).view("<f2")[..., 0].astype(F32)


def u16(b, at):
    return np.ascontiguousarray(b[..., at:at + 2]).view("<u2")[..., 0].astype(np.uint32)


def u32(b, at):
    return np.ascontiguousarray(b[..., at:at + 4]).view("<u4")[..., 0]


def reference(raw, kind):
    """FP32 values of original GGML rows uint8 [n, row_bytes] (ggml-quants.c order)."""
    block, size = {8: (32, 34), 13: (256, 176), 16: (256, 66), 17: (256, 74), 20: (32, 18), 29: (256, 56)}[kind]
    b = raw.reshape(raw.shape[0], -1, size)
    y = np.empty(b.shape[:2] + (block,), F32)
    ksigns, kmask = tables()["ksigns_iq2xs"], (1 << np.arange(8)).astype(np.uint8)
    if kind == 8:
        y[:] = b[..., 2:].view(np.int8).astype(F32) * f16(b, 0)[..., None]
    elif kind == 20:
        d, qs = f16(b, 0)[..., None], b[..., 2:18]
        kv = tables()["kvalues_iq4nl"].astype(F32)
        y[..., :16], y[..., 16:] = d * kv[qs & 15], d * kv[qs >> 4]
    elif kind == 13:
        d, dmin, sc, qh, qs = f16(b, 0)[..., None], f16(b, 2)[..., None], b[..., 4:16], b[..., 16:48], b[..., 48:]
        for j64 in range(4):
            for high in range(2):
                j = 2 * j64 + high
                if j < 4:
                    s, m = sc[..., j] & 63, sc[..., j + 4] & 63
                else:
                    s = (sc[..., j + 4] & 0xF) | ((sc[..., j - 4] >> 6) << 4)
                    m = (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4)
                ql = qs[..., 32 * j64:32 * j64 + 32]
                q = (ql >> 4 if high else ql & 15).astype(F32) + np.where(qh & (1 << j), F32(16), F32(0))
                y[..., 64 * j64 + 32 * high:64 * j64 + 32 * high + 32] = (
                    (d * s[..., None].astype(F32)) * q - dmin * m[..., None].astype(F32))
    elif kind in (16, 17):
        d = f16(b, 0)
        for ib in range(8):
            for l in range(4):
                if kind == 16:
                    aux1 = u32(b, 2 + 8 * ib + 4)
                    db = (d * (F32(0.5) + (aux1 >> 28).astype(F32))) * F32(0.25)
                    g = grid("iq2xxs_grid")[b[..., 2 + 8 * ib + l]]
                    signs = ksigns[(aux1 >> (7 * l)) & 127]
                else:
                    scale = (b[..., 66 + ib] >> (4 * (l // 2))) & 15
                    db = (d * (F32(0.5) + scale.astype(F32))) * F32(0.25)
                    code = u16(b, 2 + 2 * (4 * ib + l))
                    g = grid("iq2xs_grid")[code & 511]
                    signs = ksigns[code >> 9]
                sign = np.where(signs[..., None] & kmask, F32(-1), F32(1))
                y[..., 32 * ib + 8 * l:32 * ib + 8 * l + 8] = (db[..., None] * g.astype(F32)) * sign
    else:
        sc = [u16(b, 48 + 2 * i) for i in range(4)]
        bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) | ((sc[2] >> 4) & 0xF00) | (sc[3] & 0xF000)
        d = bits.astype(np.uint16).view(np.float16).astype(F32)
        iq1 = grid("iq1s_grid").view(np.int8)
        for ib in range(8):
            for l in range(4):
                mult = 2 * ((sc[ib // 2] >> (6 * (ib % 2) + (0 if l < 2 else 3))) & 7) + 1
                dl = d * mult.astype(F32)
                qh = b[..., 32 + 2 * ib + l // 2].astype(np.uint32)
                index = b[..., 4 * ib + l] | (((qh << 4) if l % 2 else (qh << 8)) & 0x700)
                delta = np.where(qh & (0x80 if l % 2 else 0x08), F32(-0.125), F32(0.125))
                y[..., 32 * ib + 8 * l:32 * ib + 8 * l + 8] = dl[..., None] * (
                    iq1[index].astype(F32) + delta[..., None])
    return y.reshape(raw.shape[0], -1)


@pytest.fixture(scope="module")
def ops():
    from tensorfold.cuda.sm70 import onecat_ds41

    onecat_ds41.prepare()
    return onecat_ds41


@lru_cache(maxsize=1)
def checkpoint():
    return GGUFCheckpoint.open(MODEL)


def tensor_of(kind, part):
    for name, tensor in checkpoint().tensors.items():
        if name.endswith(f"ffn_{part}_exps.weight") and tensor.ggml_type == kind:
            return tensor
    pytest.skip(f"no ffn_{part}_exps of type {kind}")


def bank(tensor):
    return np.memmap(tensor.path, np.uint8, "r", offset=tensor.offset,
                     shape=(tensor.shape[0], tensor.rows // tensor.shape[0], tensor.row_bytes))


def experts():
    # Global experts of the second EP2 half, renumbered 0..LOCAL-1.
    return np.random.default_rng(41).choice(np.arange(192, 384), LOCAL, replace=False)


def bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def dense_rows(kind):
    for name, tensor in checkpoint().tensors.items():
        if tensor.ggml_type == kind and len(tensor.shape) == 2:
            offset, n = tensor.row_span(0, min(64, tensor.rows))
            raw = np.fromfile(tensor.path, np.uint8, n, offset=offset).reshape(-1, tensor.row_bytes)
            return name, raw, tensor.shape[-1]
    pytest.skip(f"no dense type {kind}")


@pytest.mark.parametrize("kind", [8, 13, 16, 17, 20, 29])
def test_dequantize_bit_exact(ops, kind):
    if kind in (16, 17, 29):
        source = bank(tensor_of(kind, "gate"))
        raw = np.ascontiguousarray(source[experts()[:2]].reshape(-1, source.shape[-1]))
        cols, name = raw.shape[1] // BYTES[kind] * 256, f"gate_exps type {kind}"
    elif kind == 20:  # No IQ4_NL tensor in this checkpoint: independent random blocks.
        raw, cols, name = blocks(20, 64, 5120, seed=4120), 5120, "synthetic IQ4_NL"
        np.testing.assert_array_equal(reference(raw, 20), scalar_reference(raw, 20, cols))
    else:
        name, raw, cols = dense_rows(kind)
        np.testing.assert_array_equal(reference(raw, kind), scalar_reference(raw, kind, cols))
    expected = torch.from_numpy(reference(raw, kind).astype(np.float16)).cuda()
    out = torch.full((raw.shape[0], cols), float("nan"), dtype=torch.float16, device="cuda")
    ops.dequantize(torch.from_numpy(raw).cuda(), kind, cols, out=out)
    torch.cuda.synchronize()
    assert bits_equal(out, expected), name
    print(f"dequantize {name}: {raw.shape[0]}x{cols} bit-exact")


def test_quantize_q8_1_bit_exact(ops):
    torch.manual_seed(7)
    x = (torch.randn(6, 5120) * 3).half()
    x[0, :32] = 0
    q8 = ops.quantize_q8_1(x.cuda()).cpu().numpy()
    v = x.float().numpy().reshape(6, 160, 32)
    maximum = np.abs(v).max(-1)
    d = maximum / F32(127)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = v / d[..., None]
    rounded = np.trunc(scaled)
    rounded += np.where(np.abs(scaled - rounded) >= F32(0.5), np.sign(scaled), F32(0))  # roundf: ties away
    qs = np.where(maximum[..., None] == 0, 0, rounded).astype(np.int8)
    np.testing.assert_array_equal(q8[..., 4:].view(np.int8), qs)
    np.testing.assert_array_equal(q8[..., :2].view("<f2")[..., 0], d.astype(np.float16))


class Carriers:
    """Original rank-r bytes and their FP16 reference weights for LOCAL experts."""

    def __init__(self, kind):
        chosen = experts()
        self.gate_t, self.up_t, self.down_t = (tensor_of(kind, p) for p in ("gate", "up", "down"))
        g, u, dn = (np.ascontiguousarray(bank(t)[chosen]) for t in (self.gate_t, self.up_t, self.down_t))
        self.raw = g, u, dn
        self.ref = [torch.from_numpy(reference(a.reshape(-1, a.shape[-1]), kind).astype(np.float16))
                    .cuda().view(LOCAL, a.shape[1], -1) for a in (g, u, dn)]
        self.kind = kind

    def rank(self, r):
        size, first = BYTES[self.kind], r * 576 // 256
        g, u, dn = self.raw
        cuda = [torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in (
            g[:, r * 576:(r + 1) * 576], u[:, r * 576:(r + 1) * 576], dn[:, :, first * size:(first + 3) * size])]
        weights = [self.ref[0][:, r * 576:(r + 1) * 576], self.ref[1][:, r * 576:(r + 1) * 576],
                   self.ref[2][:, :, r * 576:(r + 1) * 576]]
        return cuda, weights, r * 576 - 256 * first


def fp32_reference(x, ids, p, weights):
    gate, up, down = weights
    out = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    for m in range(x.shape[0]):
        for j in range(6):
            e = int(ids[m, j])
            g = (gate[e].float() @ x[m].float()).half().float().clamp(max=10)
            u = (up[e].float() @ x[m].float()).half().float().clamp(-10, 10)
            h = (g / (1 + torch.exp(-g)) * u * p[m, j]).half().float()
            out[m] += down[e].float() @ h
    return out


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_v41_experts_against_fp32(ops, kind):
    carriers = Carriers(kind)
    generator = torch.Generator().manual_seed(kind)
    worst = 0.0
    for r in range(4):
        (gate, up, down), weights, left = carriers.rank(r)
        assert left == (0, 64, 128, 192)[r]
        for m in range(1, 7):
            x = torch.randn(m, 5120, generator=generator).half().cuda()
            ids = torch.randint(0, LOCAL, (m, 6), generator=generator, dtype=torch.int32).cuda()
            p = torch.softmax(torch.randn(m, 6, generator=generator), -1).float().cuda()
            out = ops.v41_experts(x, ids, p, gate, up, down, kind, kind, left)
            again = ops.v41_experts(x, ids.long(), p, gate, up, down, kind, kind, left)
            expected = fp32_reference(x, ids, p, weights)
            assert torch.isfinite(out).all() and bits_equal(out, again)  # int32/int64 ids agree
            diff = out.float() - expected
            rel = (diff.norm() / expected.norm()).item()
            worst = max(worst, rel)
            print(f"type {kind} rank {r} left {left} M{m}: max_abs {diff.abs().max().item():.4g} "
                  f"ref_max {expected.abs().max().item():.4g} rel_l2 {rel:.4g}")
            assert rel < 0.1  # gross-defect screen only
    print(f"type {kind} worst rel_l2 {worst:.4g}")


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_probability_zero_route_contributes_zero(ops, kind):
    (gate, up, down), _, left = Carriers(kind).rank(1)
    generator = torch.Generator().manual_seed(100 + kind)
    for m in (1, 6):
        x = torch.randn(m, 5120, generator=generator).half().cuda()
        ids = torch.randint(0, LOCAL, (m, 6), generator=generator, dtype=torch.int32).cuda()
        p = torch.softmax(torch.randn(m, 6, generator=generator), -1).float().cuda()
        p[:, 2] = 0
        base = ops.v41_experts(x, ids, p, gate, up, down, kind, kind, left)
        moved = ids.clone()
        moved[:, 2] = (moved[:, 2] + 3) % LOCAL
        assert torch.equal(base, ops.v41_experts(x, moved, p, gate, up, down, kind, kind, left))
        # The zero-probability route's Q8_1 down input is exactly zero.
        q8 = ops.v41_swiglu_q8(ops.v41_gate_up(ops.quantize_q8_1(x), ids, gate, up, kind), p)
        assert not q8[:, 2, :, 4:].any() and not q8[:, 2, :, :2].view(torch.int16).bitwise_and(0x7FFF).any()


def test_adaptive_iq1m_matches_per_route(ops):
    tensor = tensor_of(29, "gate")
    chosen = np.random.default_rng(29).choice(384, 40, replace=False)
    gate, up = (torch.from_numpy(np.ascontiguousarray(bank(t)[chosen][:, :576])).cuda()
                for t in (tensor, tensor_of(29, "up")))
    generator = torch.Generator().manual_seed(29)
    q8 = ops.quantize_q8_1(torch.randn(6, 5120, generator=generator).half().cuda())
    for ids in (torch.randint(0, 6, (6, 6), generator=generator),     # reuse branch (<=32 distinct)
                torch.randperm(40, generator=generator)[:36].view(6, 6)):  # per-route branch
        for dtype in (torch.int32, torch.int64):
            routed = ids.to(dtype).cuda()
            adaptive = ops.v41_gate_up(q8, routed, gate, up, 29)
            plain = torch.empty_like(adaptive)
            ops.extension().gate_up(plain, q8, routed, gate, up, 29)
            assert bits_equal(adaptive, plain)


def test_shared_swiglu_n288(ops):
    generator = torch.Generator().manual_seed(288)
    gate, up = (torch.randn(6, 288, generator=generator) * 6).half().cuda(), \
        (torch.randn(6, 288, generator=generator) * 6).half().cuda()
    g, u = gate.float().clamp(max=10), up.float().clamp(-10, 10)
    expected = (g / (1 + torch.exp(-g)) * u).half()
    out = ops.shared_swiglu(gate, up)
    diff = (out.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
    print(f"shared_swiglu N288: {(diff == 0).float().mean().item():.4f} bit-equal, max {diff.max().item()} ulp")
    assert diff.max().item() <= 1
