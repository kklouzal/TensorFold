"""SM70 tensor-core prompt attention (fused local + weighting) vs the RN family and an FP64 oracle.

Layout-exact synthetic owner state for one KV-head group (ranks 0..3):
unique-history Q8 pages ([phys,1,256] int8 codes, [phys,1,8] FP16 group
scales, -1 page offsets for foreign pages, owner = 1 ^ (page % degree)),
canonical selections [8,rows,idw+2], 16-word descriptors and gathered FP16
queries. Both pipelines share the RN statistics merge and value merge:

  RN : rn_local_statistics -> rn_merge -> rn_probabilities -> rn_partial_values
  TC : fused_local         -> rn_merge -> weight_values

The group sum of the four local partials is compared with FP64 attention over
the unrounded Q8 dequantization; the tensor-core family must be no less
accurate than the RN family it replaces for prefill, latch no status, agree on
empty local sets and publish zero for inactive (padded) rows.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="needs an SM70 GPU")

from tensorfold.cuda.sm70 import paged_attention as rn  # noqa: E402
from tensorfold.cuda.sm70 import prompt_attention as tc  # noqa: E402

DIM, HEADS, WORLD, BUDGET, RATIO = 256, 12, 8, 2048, 4
WIDTH = BUDGET + RATIO + 2
OWNER = 5  # indexer owner rank whose selection slice every descriptor names
SLOT_GENERATION, HISTORY_GENERATION, LAYER = 7, 9, 3
ROWS, PADDING, PAGE_TOKENS = 48, 8, 128


def quantize(x):
    """Per-32 FP16 scale = amax, code = floor(x / scale * 128) clamped (kvquant groups)."""
    groups = x.view(x.shape[0], DIM // 32, 32)
    scale = groups.abs().amax(-1).clamp_min(1e-4).half()
    code = torch.floor(groups / scale.float()[..., None] * 128).clamp(-128, 127).to(torch.int8)
    return code.view(-1, DIM), scale


def build(first, degree, generator):
    device = "cuda"
    capacity = ((first + ROWS + PAGE_TOKENS - 1) // PAGE_TOKENS) * PAGE_TOKENS + PAGE_TOKENS
    pages = capacity // PAGE_TOKENS
    kc, ks = quantize(torch.randn(capacity, DIM, device=device, generator=generator))
    vc, vs = quantize(torch.randn(capacity, DIM, device=device, generator=generator))
    caches = []
    for rank in range(4):
        local = [page for page in range(pages) if 1 ^ (page % degree) == rank]
        offsets = torch.full((pages,), -1, dtype=torch.int64)
        physical = max(1, len(local) * PAGE_TOKENS)
        k = torch.zeros(physical, 1, DIM, dtype=torch.int8, device=device)
        v = torch.zeros_like(k)
        k_scale = torch.zeros(physical, 1, DIM // 32, dtype=torch.float16, device=device)
        v_scale = torch.zeros_like(k_scale)
        for index, page in enumerate(local):
            offsets[page] = index * PAGE_TOKENS
            rows, source = slice(index * PAGE_TOKENS, (index + 1) * PAGE_TOKENS), slice(
                page * PAGE_TOKENS, (page + 1) * PAGE_TOKENS)
            k[rows, 0], v[rows, 0], k_scale[rows, 0], v_scale[rows, 0] = kc[source], vc[source], ks[source], vs[source]
        caches.append(dict(k=k, v=v, ks=k_scale, vs=v_scale, offsets=offsets.to(device), physical=physical,
                           status=torch.tensor([0, -2], dtype=torch.int32, device=device)))
    temperature = torch.linspace(0.5, 4.0, HEADS, device=device)
    q = torch.randn(WORLD, ROWS, 3, DIM, device=device, generator=generator)
    q[:4] *= temperature.view(4, 1, 3, 1)
    q = q.half().contiguous()
    selections = torch.zeros(WORLD, ROWS, WIDTH, dtype=torch.int32, device=device)
    sequence = torch.randn(max((first + ROWS) // RATIO, 1), device=device, generator=generator) * 2.0
    for row in range(ROWS):
        visible = first + row + 1
        complete = visible // RATIO
        if complete > BUDGET // RATIO:
            blocks = torch.arange(complete, device=device)
            score = sequence[:complete] + torch.randn(complete, device=device, generator=generator)
            score = score + 3.0 * (blocks >= complete - 32).float()
            top = torch.topk(score, BUDGET // RATIO).indices.sort().values
            ids = torch.cat([(top[:, None] * RATIO + torch.arange(RATIO, device=device)).reshape(-1),
                             torch.arange(RATIO * complete, visible, device=device)]).int()
            selections[OWNER, row, :ids.numel()] = ids
            selections[OWNER, row, WIDTH - 2], selections[OWNER, row, WIDTH - 1] = ids.numel(), 1
        else:
            selections[OWNER, row, WIDTH - 2] = visible
    position = torch.tensor([first], dtype=torch.int32, device=device)
    live = torch.tensor([ROWS - PADDING], dtype=torch.int32, device=device)
    generations = torch.tensor([SLOT_GENERATION, HISTORY_GENERATION], dtype=torch.int64, device=device)
    tables = [torch.tensor([[c["k"].data_ptr(), c["v"].data_ptr(), c["ks"].data_ptr(), c["vs"].data_ptr(),
                             c["offsets"].data_ptr(), generations.data_ptr(), c["status"].data_ptr(),
                             position.data_ptr(), live.data_ptr(), SLOT_GENERATION, HISTORY_GENERATION, capacity,
                             c["physical"], pages, row, OWNER] for row in range(ROWS)],
                           dtype=torch.int64, device=device) for c in caches]
    return dict(q=q, selections=selections, caches=caches, tables=tables, keep=(position, live, generations),
                logical=(kc, ks, vc, vs))


def exact(data):
    """FP64 group attention [rows,12,256] over the unrounded Q8 values (c + 0.5) * scale / 128."""
    kc, ks, vc, vs = data["logical"]
    q = data["q"][:4].permute(1, 0, 2, 3).reshape(ROWS, HEADS, DIM).double()
    selection = data["selections"][OWNER]
    out = torch.zeros(ROWS, HEADS, DIM, dtype=torch.float64, device="cuda")
    for row in range(ROWS - PADDING):
        n = int(selection[row, WIDTH - 2])
        ids = selection[row, :n].long() if int(selection[row, WIDTH - 1]) else torch.arange(n, device="cuda")
        k = (kc[ids].double().view(n, 8, 32) + 0.5) * ks[ids].double()[..., None] / 128
        v = (vc[ids].double().view(n, 8, 32) + 0.5) * vs[ids].double()[..., None] / 128
        out[row] = torch.softmax((q[row] @ k.view(n, DIM).T) / 16, -1) @ v.view(n, DIM)
    return out


def run(data, family):
    layer = torch.tensor([LAYER], dtype=torch.int32, device="cuda")
    common = dict(group_start=0, page_tokens=PAGE_TOKENS, layer=layer)
    stats = torch.zeros(WORLD, ROWS, HEADS, 2, device="cuda")
    values = torch.full((WORLD, ROWS, HEADS, DIM), math.nan, device="cuda")
    unnormalized = [torch.full((ROWS, HEADS, DIM), math.nan, device="cuda") for _ in range(4)]
    for rank in range(4):
        if family == "rn":
            rn.rn_local_statistics(data["q"], data["selections"], data["tables"][rank], stats[rank], key_bits=8,
                                   key_codec=0, heads_per_block=12, **common)
        else:
            tc.fused_local(data["q"], data["selections"], data["tables"][rank], stats[rank], unnormalized[rank],
                           **common)
    for rank in range(4):
        merged = torch.empty(ROWS, HEADS, 2, device="cuda")
        rn.rn_merge_statistics(stats, data["tables"][rank], merged, **common)
        if family == "rn":
            probability = torch.empty(ROWS, HEADS, rn.RN_MAX_SELECTED, dtype=torch.float16, device="cuda")
            rn.rn_probabilities(data["q"], data["selections"], data["tables"][rank], merged, probability, key_bits=8,
                                key_codec=0, heads_per_block=12, **common)
            rn.rn_partial_values(probability, data["selections"], data["tables"][rank], values[rank],
                                 page_tokens=PAGE_TOKENS, layer=layer, value_bits=8, value_codec=0)
        else:
            tc.weight_values(unnormalized[rank], stats[rank], merged, data["tables"][rank], values[rank], layer=layer)
    torch.cuda.synchronize()
    status = [cache["status"].tolist() for cache in data["caches"]]
    for cache in data["caches"]:
        cache["status"].copy_(torch.tensor([0, -2], dtype=torch.int32))
    return stats[:4].clone(), values[:4].clone(), status


def relative_l2(value, reference):
    return ((value.double() - reference).norm() / reference.norm()).item()


@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("first", [0, 1024, 8192], ids=["dense-short", "dense-1k", "sparse-8k"])
def test_fused_prompt_attention_is_no_less_accurate_than_rn(first, degree):
    generator = torch.Generator(device="cuda").manual_seed(1234 + first + degree)
    data = build(first, degree, generator)
    oracle = exact(data)
    live = slice(0, ROWS - PADDING)
    rn_stats, rn_values, rn_status = run(data, "rn")
    tc_stats, tc_values, tc_status = run(data, "tc")
    assert rn_status == tc_status == [[0, -2]] * 4
    assert torch.equal(torch.isfinite(rn_stats[..., 0]), torch.isfinite(tc_stats[..., 0])), "empty local sets differ"
    assert torch.all(tc_values[:, ROWS - PADDING:] == 0), "inactive rows must publish zero"
    assert torch.isfinite(tc_values).all()
    tc_error = relative_l2(tc_values.sum(0)[live], oracle[live])
    rn_error = relative_l2(rn_values.sum(0)[live], oracle[live])
    assert tc_error <= rn_error, (tc_error, rn_error)
