"""FP32-scale RotorQuant (rotorquant8/-norm, signed128-4, signed64-norm4) through the NativeRN paged owner attention.

Random K/V are encoded per owner by the native codec, the four-peer RN phases
run as in serving, and the merged output is compared with an FP64 attention
over the same decoded FP16 working operands (inverse-rotated at the end). The
RN family rounds probabilities to Half, so this is a bounded-error check, not
a bitwise oracle; the Q8/compact bitwise certificates stay in their own tests.
"""
import pytest
import torch

from tensorfold.cuda.sm70 import paged_attention as operator
from tensorfold.cuda.sm70 import rotorquant_q4_native as codec_native
from tensorfold.families.qwen4_exp.cuda.rotorquant_ref import rotate_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='NativeRN requires actual SM70 component execution')

PAGE, LOGICAL, ROWS, PEERS = 128, 512, 2, 4


def owner_shard(rank, keys, values, codec):
    width = 256 if codec in (8, 12) else 128
    """Rank-owned pages (page % 4 == rank) of the logical [512, 256] K/V, encoded in place by the native codec."""
    offsets, physical = [], 0
    for page in range(LOGICAL // PAGE):
        offsets.append(physical if page % PEERS == rank else -1)
        physical += PAGE if page % PEERS == rank else 0
    k = torch.zeros((physical, 1, width), dtype=torch.uint8, device='cuda')
    v = torch.zeros_like(k)
    ks = torch.zeros((physical, 1, 2), dtype=torch.float32, device='cuda')
    vs = torch.zeros_like(ks)
    for page, offset in enumerate(offsets):
        if offset < 0:
            continue
        rows = slice(page * PAGE, (page + 1) * PAGE)
        position = torch.tensor([offset], dtype=torch.int64, device='cuda')
        codec_native.encode(keys[rows].view(PAGE, 1, 256), k, ks, codec=codec, position=position)
        codec_native.encode(values[rows].view(PAGE, 1, 256), v, vs, codec=codec, position=position)
    page_offsets = torch.tensor(offsets, dtype=torch.int64, device='cuda')
    generations = torch.zeros(2, dtype=torch.int64, device='cuda')
    status = torch.tensor([0, -2], dtype=torch.int32, device='cuda')
    position = torch.tensor([LOGICAL - ROWS], dtype=torch.int32, device='cuda')
    live = torch.tensor([ROWS], dtype=torch.int32, device='cuda')
    records = [[k.data_ptr(), v.data_ptr(), ks.data_ptr(), vs.data_ptr(), page_offsets.data_ptr(),
                generations.data_ptr(), status.data_ptr(), position.data_ptr(), live.data_ptr(),
                0, 0, LOGICAL, physical, LOGICAL // PAGE, row, 0, 0] for row in range(ROWS)]
    table = torch.tensor(records, dtype=torch.int64, device='cuda')
    return dict(k=k, v=v, ks=ks, vs=vs, offsets=page_offsets, generations=generations, status=status,
                position=position, live=live, table=table)


def decoded(codes, scales, codec):
    out = torch.empty((codes.shape[0], 1, 256), dtype=torch.float16, device='cuda')
    return codec_native.decode(codes, scales, out, codec=codec)


# Kernel codecs: 12 rotorquant8, 8 rotorquant8-norm, 4 rotorquant-signed128-4, 5 rotorquant-signed64-norm4.
@pytest.mark.parametrize('codec', [12, 8, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('heads', [1, 4, 12])
@torch.no_grad()
def test_rn_fp32_scale_rotor_matches_fp64_attention_over_decoded_operands(codec, heads):
    bits = 8 if codec in (8, 12) else 4
    gen = torch.Generator(device='cuda').manual_seed(4108 + codec)
    keys = torch.randn((LOGICAL, 256), device='cuda', generator=gen).half()
    values = torch.randn((LOGICAL, 256), device='cuda', generator=gen).half()
    shards = [owner_shard(rank, keys, values, codec) for rank in range(PEERS)]
    layer = torch.tensor([3], dtype=torch.int32, device='cuda')
    count = 300
    query = (torch.randn((8, ROWS, 3, 256), device='cuda', generator=gen) * 4).half()
    selection = torch.zeros((8, ROWS, count + 2), dtype=torch.int32, device='cuda')
    chosen = torch.stack([torch.randperm(LOGICAL - ROWS, device='cuda', generator=gen)[:count].sort().values
                          for _ in range(ROWS)])
    selection[0, :, :count] = chosen.int()
    selection[0, :, -2:] = torch.tensor([count, 1], dtype=torch.int32, device='cuda')
    receive = torch.zeros((8, ROWS, 12, 256), dtype=torch.float32, device='cuda')
    all_stats = receive.reshape(-1)[:8 * ROWS * 12 * 2].view(8, ROWS, 12, 2)
    all_stats[..., 0] = -torch.inf
    all_stats[..., 1] = 0
    common = dict(group_start=0, page_tokens=PAGE, layer=layer)
    stats = [torch.empty((ROWS, 12, 2), dtype=torch.float32, device='cuda') for _ in range(PEERS)]
    globals_ = [torch.empty_like(stats[0]) for _ in range(PEERS)]
    local = [torch.empty((ROWS, 12, 256), dtype=torch.float32, device='cuda') for _ in range(PEERS)]
    out = [torch.empty((ROWS, 3, 256), dtype=torch.float16, device='cuda') for _ in range(PEERS)]
    for rank, shard in enumerate(shards):
        operator.rn_local_statistics(query, selection, shard['table'], stats[rank], key_bits=bits, key_codec=codec,
                                     heads_per_block=heads, **common)
        all_stats[rank].copy_(stats[rank])
    for rank, shard in enumerate(shards):
        operator.rn_merge_statistics(all_stats, shard['table'], globals_[rank], **common)
    probability = receive.view(torch.float16).reshape(-1)[:ROWS * 12 * 2051].view(ROWS, 12, 2051)
    for rank, shard in enumerate(shards):
        operator.rn_probabilities(query, selection, shard['table'], globals_[rank], probability, key_bits=bits,
                                  key_codec=codec, heads_per_block=heads, **common)
        operator.rn_partial_values(probability, selection, shard['table'], local[rank], page_tokens=PAGE,
                                   layer=layer, value_bits=bits, value_codec=codec)
    for rank in range(PEERS):
        receive[rank].copy_(local[rank])
    for rank, shard in enumerate(shards):
        operator.merge_values(receive, shard['table'], out[rank], rank=rank, page_tokens=PAGE, layer=layer,
                              value_bits=bits, value_codec=codec)
    torch.cuda.synchronize()
    for shard in shards:
        assert shard['status'].cpu().tolist() == [0, -2]
    # FP64 reference over the same decoded FP16 working K/V (rotated basis), then inverse rotation.
    dk = torch.empty((LOGICAL, 256), dtype=torch.float64, device='cuda')
    dv = torch.empty_like(dk)
    for rank, shard in enumerate(shards):
        k, v = decoded(shard['k'], shard['ks'], codec), decoded(shard['v'], shard['vs'], codec)
        for page in range(LOGICAL // PAGE):
            offset = int(shard['offsets'][page])
            if offset >= 0:
                dk[page * PAGE:(page + 1) * PAGE] = k[offset:offset + PAGE, 0].double()
                dv[page * PAGE:(page + 1) * PAGE] = v[offset:offset + PAGE, 0].double()
    variant = {12: 'signed-iso128', 8: 'signed-iso128-norm8', 1: 'planar', 2: 'isofast', 3: 'iso64-norm',
               4: 'signed-iso128', 5: 'signed-iso64-norm', 6: 'signed-iso128-norm'}[codec]
    for row in range(ROWS):
        ids = chosen[row].long()
        q = query[:PEERS, row].reshape(12, 256).double()
        scores = q @ dk[ids].T / 16
        p = torch.softmax(scores, dim=-1)
        rotated = p @ dv[ids]
        expected = rotate_ref(rotated.float(), variant, inverse=True).double()
        actual = torch.cat([out[rank][row].double() for rank in range(PEERS)])
        error = (actual - expected).abs().max().item()
        assert error <= 2e-3 * expected.abs().max().item() + 1e-3, (row, error)
