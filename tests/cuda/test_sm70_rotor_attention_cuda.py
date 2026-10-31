"""Packed CODEC4/5 attention against an independent stored-basis FP16 oracle."""

import pytest
import torch

from tensorfold.cuda.sm70 import attention_native as native, family_ops as ops
from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as rq
from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 attention")
SIDES = ((4, 4, 4, 4), (4, 4, 0, 0), (4, 4, 8, 0), (0, 0, 4, 4), (8, 0, 4, 4),
         (4, 5, 4, 5), (4, 5, 4, 4), (4, 4, 4, 5), (4, 5, 0, 0), (0, 0, 4, 5),
         (4, 5, 8, 0), (8, 0, 4, 5))
VARIANTS = {4: "signed-iso128", 5: "signed-iso64-norm"}


def stored_side(capacity, bits, codec):
    if codec:
        # All nibble pairs occur, including both ends and the central bins.
        data = torch.randint(256, (capacity, 1, 128), device="cuda", dtype=torch.uint8)
        data[:2, 0].copy_(torch.arange(256, device="cuda").reshape(2, 128).to(torch.uint8))
        scales = torch.rand((capacity, 1, 2), device="cuda", dtype=torch.float32)*0.4+0.025
        scales[0, 0, 0] = 0.10000000894069672
        scales[0, 0, 1] = 0.3333333432674408
        return data, scales
    raw = torch.randn((capacity, 1, 256), device="cuda", dtype=torch.float16)*0.2
    return ops.quantize_int8(raw) if bits else (raw, None)


def decoded_side(data, scales, bits, codec):
    """Independent unpack/table arithmetic, with the FP16 operand rounding explicit."""
    if codec:
        indices = torch.stack((data & 15, data >> 4), -1).reshape(data.shape[0], 1, 256).long()
        table = torch.tensor(rq.CENTROIDS_4, device="cuda", dtype=torch.float32)
        return (table[indices] * scales.repeat_interleave(128, -1)).half()
    if bits:
        grouped = data.float().reshape(data.shape[0], 1, 8, 32)
        return ((grouped+0.5)*scales.float()[..., None]*(1.0/128)).reshape(data.shape).half()
    return data


def inputs(capacity, rows, side, sparse):
    torch.manual_seed(927)
    kb, kc, vb, vc = side
    q = torch.randn((rows, 3, 256), device="cuda", dtype=torch.float16)
    if kc:
        q = rq.rotate_ref(q.float(), VARIANTS[kc]).half()
    elif kb:
        q = ops.h32(q).half()
    key, ks = stored_side(capacity, kb, kc)
    value, vs = stored_side(capacity, vb, vc)
    owner = AttnScratch(rows, 3, 256, capacity, "cuda", dtype=torch.float16)
    owner.qsa = sparse
    owner.sparse.fill_(int(sparse))
    owner.nk.fill_(64)
    owner.ids[:, :64] = torch.arange(64, device="cuda", dtype=torch.int32)*2
    position = torch.tensor([capacity-rows], device="cuda", dtype=torch.int32)
    return q, key, value, position, owner, ks, vs


def oracle(q, key, value, start, ks, vs, side, sparse):
    kb, kc, vb, vc = side
    k = decoded_side(key, ks, kb, kc)
    v = decoded_side(value, vs, vb, vc)
    outputs = []
    for row in range(q.shape[0]):
        end = start+row+1
        visible = torch.ones(end, device="cuda", dtype=torch.bool)
        if sparse:
            visible.zero_()
            selected = torch.arange(64, device="cuda")*2
            visible[selected[selected < end]] = True
        scores = torch.mm(q[row], k[:end, 0].T, out_dtype=torch.float32)*0.0625
        scores.masked_fill_(~visible[None], -torch.inf)
        probabilities = scores.softmax(-1).half()
        values = torch.where(visible[:, None], v[:end, 0], 0)
        merged = torch.mm(probabilities, values, out_dtype=torch.float32)
        if vc:
            merged = rq.rotate_ref(merged, VARIANTS[vc], inverse=True)
        elif vb:
            merged = ops.h32(merged)
        outputs.append(merged.half())
    return torch.stack(outputs)


def run(values, side, context):
    q, key, value, pos, owner, ks, vs = values
    kb, kc, vb, vc = side
    return native.attention(q, key, value, pos, owner, q.shape[0], 0.0625, owner.out,
                            context, ks, vs, kb, vb, kcodec=kc, vcodec=vc)


def same_bits(left, right):
    return torch.equal(left.view(torch.int16), right.view(torch.int16))


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("capacity", [128, 640, 8192])
@pytest.mark.parametrize("rows", [1, 3, 4, 24])
@pytest.mark.parametrize("sparse", [False, True])
def test_packed_and_mixed_attention_exactly_matches_original_query_arithmetic(side, capacity, rows, sparse):
    values = inputs(capacity, rows, side, sparse)
    q, key, value, _, _, ks, vs = values
    authority = oracle(q, key, value, capacity-rows, ks, vs, side, sparse)
    assert same_bits(run(values, side, capacity), authority)


@pytest.mark.parametrize("side", SIDES)
def test_rotor_capture_replays_changing_positions_codes_and_scales_without_new_owners(side):
    values = inputs(8192, 3, side, True)
    q, key, value, pos, owner, ks, vs = values
    pos.fill_(125)
    run(values, side, 8192)
    workspace = owner.native_workspace
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(values, side, 8192)
    try:
        for start in (64, 125, 639, 2047, 2730, 5103, 8189):
            pos.fill_(start)
            if side[1]:
                key[0].bitwise_xor_(1)
                ks[0].mul_(0.999)
            if side[3]:
                value[0].bitwise_xor_(16)
                vs[0].mul_(1.001)
            authority = oracle(q, key, value, start, ks, vs, side, True)
            graph.replay()
            assert same_bits(owner.out, authority)
            assert owner.native_workspace is workspace
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("side", SIDES)
def test_unselected_or_future_rotor_metadata_nan_does_not_poison_live_outputs(side):
    values = inputs(512, 3, side, True)
    q, key, value, pos, _, ks, vs = values
    pos.fill_(125)
    for codec, scales in ((side[1], ks), (side[3], vs)):
        if codec:
            scales[13].fill_(float("nan"))  # Odd key is not selected.
            scales[128:].fill_(float("nan"))
    authority = oracle(q, key, value, 125, ks, vs, side, True)
    actual = run(values, side, 512)
    assert actual.isfinite().all()
    assert same_bits(actual, authority)


@pytest.mark.parametrize("side", [(4, 4, 4, 4), (4, 5, 4, 5)])
def test_zero_rms_groups_preserve_canonical_inverse_and_signed_zero_bits(side):
    values = inputs(128, 3, side, False)
    q, key, value, _, _, ks, vs = values
    key.fill_(0x77)
    value.fill_(0x77)
    ks.zero_()
    vs.zero_()
    authority = oracle(q, key, value, 125, ks, vs, side, False)
    assert same_bits(run(values, side, 128), authority)


def test_rotor_entry_validates_metadata_and_preserves_legacy_entry_refusal():
    side = SIDES[0]
    values = inputs(128, 3, side, False)
    q, key, value, pos, owner, ks, vs = values
    run(values, side, 128)
    workspace = owner.native_workspace
    packet = [q, key, value, pos, owner.sparse, owner.nk, owner.ids,
              workspace.scores, workspace.probabilities, workspace.partials, owner.out,
              ks, vs, 3, 128, 4, 4, False, 0.0625]
    with pytest.raises(RuntimeError, match="formats"):
        native.extension().run(*packet)
    with pytest.raises(RuntimeError, match="formats"):
        native.extension().run_rotor(*packet, 3, 4)
    packet[11] = ks.half()
    with pytest.raises(RuntimeError, match="key scales"):
        native.extension().run_rotor(*packet, 4, 4)
    packet[11] = torch.zeros((128, 1, 8), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="key scale geometry"):
        native.extension().run_rotor(*packet, 4, 4)
    packet[11] = ks
    legacy = packet.copy()
    legacy[1], legacy[2] = decoded_side(key, ks, 4, 4), decoded_side(value, vs, 4, 4)
    legacy[11:13], legacy[15:17] = [None, None], [0, 0]
    for invalid in (float("inf"), float("nan"), 1e300, 1e-300, 0.0, -1.0):
        packet[-1] = invalid
        with pytest.raises(RuntimeError, match="positive finite attention scale"):
            native.extension().run_rotor(*packet, 4, 4)
        legacy[-1] = invalid
        with pytest.raises(RuntimeError, match="positive finite attention scale"):
            native.extension().run(*legacy)
