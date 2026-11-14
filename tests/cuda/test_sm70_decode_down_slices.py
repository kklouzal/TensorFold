"""SM70 decode-size grouped MoE down on rank-sliced ETP4 banks vs an FP64 reference.

TP8/ETP4 keeps 160 of each expert's 640 down columns, so resident rows are 5
IQ4_NL (90 bytes), 3 Q2_0 (54 bytes, column offset 0 or 32) or 5 Q8_0 (170 bytes)
blocks: row strides are 2 mod 4 and spans may start mid-word. The down product is
checked on the kernel's own FP16 activation against FP64 sums of the exactly
decoded coefficients. Bound per output (FP32 output): the integer path quantizes
each 32-element activation block to int16 (|error| <= amax / 65534 per element)
and both paths accumulate in FP32 (gamma_160 with u = 2^-23, plus one product
rounding per eight-coefficient group).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="needs an SM70 GPU")

E, W, D = 256, 160, 2560
LAYOUT = {8: (32, 34), 20: (32, 18), 21: (256, 110), 42: (64, 18)}


def bank(kind, rows, cols, offset, scale, generator):
    block, size = LAYOUT[kind]
    blocks = cols // block
    data = torch.randint(0, 256, (rows, blocks, size), dtype=torch.uint8, device="cuda", generator=generator)
    d = (torch.rand(rows, blocks, device="cuda", generator=generator) * scale + scale).half()
    data[:, :, :2] = d.view(torch.uint8).view(rows, blocks, 2)
    if kind == 8:
        data[:, :, 2:] = torch.randint(-127, 128, (rows, blocks, 32), device="cuda", dtype=torch.int8,
                                       generator=generator).view(torch.uint8)
    return SimpleNamespace(data=data.view(rows, blocks * size).contiguous(), ggml_type=kind, column_offset=offset)


@pytest.mark.parametrize("down_kind,columns,offset", [(20, 160, 0), (42, 192, 0), (42, 192, 32), (8, 160, 0)],
                         ids=["iq4nl", "q2_0-o0", "q2_0-o32", "q8_0"])
@pytest.mark.parametrize("rows", [1, 2, 5, 20, 32])
def test_sliced_down_matches_fp64_within_the_quantization_bound(down_kind, columns, offset, rows):
    generator = torch.Generator(device="cuda").manual_seed(1000 * down_kind + rows + offset)
    gate = bank(21, E * W, D, 0, 0.004, generator)
    up = bank(21, E * W, D, 0, 0.004, generator)
    down = bank(down_kind, E * D, columns, offset, 0.004 if down_kind == 8 else 0.02, generator)
    assert down.data.stride(0) % 4 == 2
    x = (torch.randn(rows, D, device="cuda", generator=generator) * 0.5).half()
    weights = torch.exp(torch.randn(512, device="cuda", generator=generator))
    picks = torch.multinomial(weights.expand(rows, 512), 10, replacement=False, generator=generator)
    picks = torch.cat([picks, torch.full((rows, 1), 512, device="cuda")], 1)
    activation = torch.empty(rows * 11, W, dtype=torch.half, device="cuda")
    out = torch.empty(rows, 11, D, dtype=torch.float32, device="cuda")
    gguf_ops.experts_owned(x, picks, gate, up, down, first=0, experts=512, width=W,
                           shared=torch.zeros(rows, D, device="cuda"),
                           local=torch.empty(rows, 11, dtype=torch.long, device="cuda"),
                           activation=activation, out=out,
                           grouping=torch.empty(gguf_ops.grouped_workspace_words(rows * 11, E), dtype=torch.int32,
                                                device="cuda"),
                           _gate_up=gguf_ops._GroupedGateUp(4))
    owned = picks[:, :10] < E
    r, s = owned.nonzero(as_tuple=True)
    experts = picks[r, s]
    ids = (experts[:, None] * D + torch.arange(D, device="cuda")).reshape(-1)
    coefficients = gguf_ops.decode(down.data, down_kind, W, column_offset=offset, dtype=torch.float32,
                                   ids=ids).double().view(-1, D, W)
    act = activation.view(rows, 11, W)[r, s].double()
    reference = torch.einsum("pdw,pw->pd", coefficients, act)
    magnitude = torch.einsum("pdw,pw->pd", coefficients.abs(), act.abs())
    amax = act.view(-1, 5, 32).abs().amax(-1).repeat_interleave(32, -1)
    quantization = torch.einsum("pdw,pw->pd", coefficients.abs(), amax / 65534)
    bound = 1.01 * quantization + 200 * 2.0 ** -23 * magnitude + 2.0 ** -126
    error = (out[r, s].double() - reference).abs()
    assert torch.isfinite(out[r, s]).all() and torch.all(error <= bound), (error - bound).max().item()
    if down_kind == 42:  # FP16-exact coefficients take the unquantized FP32 path
        assert torch.all(error <= 200 * 2.0 ** -23 * magnitude + 2.0 ** -126)
