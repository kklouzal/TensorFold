"""Independent Torch byte oracles for the optional fused SM70 routing weights."""
import pytest
import torch

from tensorfold.cuda.sm70.routing_native import route

pytestmark = pytest.mark.skipif(not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0), reason="requires native SM70")


def reference(logits, count):
    indices = logits[:, :512].argsort(dim=-1, descending=True, stable=True)[:, :count]
    selected = logits[:, :512].gather(1, indices)
    weights = selected.softmax(-1).half().float()
    shared = logits[:, 512].half().float().sigmoid().half().float()
    picks = torch.cat((indices.int(), torch.full((logits.shape[0], 1), 512, device=logits.device, dtype=torch.int32)), dim=1)
    return picks, torch.cat((weights, shared[:, None]), dim=1)


def check(logits, count):
    expected_picks, expected_weights = reference(logits, count)
    picks, weights = torch.empty_like(expected_picks), torch.empty_like(expected_weights)
    route(logits, picks, weights)
    torch.testing.assert_close(picks, expected_picks, atol=0, rtol=0)
    torch.testing.assert_close(weights.view(torch.int32), expected_weights.view(torch.int32), atol=0, rtol=0)
    return picks, weights


@pytest.mark.parametrize("count", range(1, 33))
def test_fused_selected_weights_match_independent_torch_all_supported_widths(count):
    generator = torch.Generator(device="cuda").manual_seed(1789 + count)
    logits = torch.randn((256, 513), generator=generator, device="cuda")
    logits[:64] *= .01
    logits[64:128] *= 4
    logits[128:192] *= 20
    check(logits, count)


@pytest.mark.parametrize("rows", [0, 1, 3, 18, 129])
def test_fused_weights_handle_actual_windows_and_empty_rows(rows):
    logits = torch.randn((rows, 513), device="cuda")
    check(logits, 10)


@pytest.mark.parametrize("count", [1, 2, 3, 10, 16, 31, 32])
def test_fused_weights_preserve_special_float_payloads_ties_and_signed_zero(count):
    logits = torch.zeros((14, 513), device="cuda")
    logits[0].fill_(-torch.inf)
    logits[1].fill_(torch.inf)
    logits[2, :8] = torch.inf
    logits[3, 0] = torch.nan
    logits[4, 500] = torch.nan
    logits[5, 1::2] = -0.
    logits[6].fill_(torch.finfo(torch.float32).max)
    logits[7].fill_(-torch.finfo(torch.float32).max)
    logits[8, 2:8] = 1.
    logits[9, :32] = torch.arange(32, device="cuda") % 3
    logits[10, 0] = torch.finfo(torch.float32).tiny
    # Quiet/signaling and positive/negative FP32 NaN payloads; the Half->Float
    # conversion boundary is checked as integer bits, including all NaN outputs.
    for row, bits in zip((11, 12, 13), (0x7fc00001, 0x7f800001, -4194303)):
        logits[row, 0].view(torch.int32).fill_(bits)
        logits[row, 512].view(torch.int32).fill_(bits)
    check(logits, count)


@pytest.mark.parametrize("count", [1, 10, 32])
def test_fused_weights_respect_padded_logit_strides_and_ignore_trailing_columns(count):
    storage = torch.randn((18, 529), device="cuda")
    storage[:, 513:] = torch.nan
    check(storage[:, :513], count)
    check(storage, count)


def test_fused_shared_sigmoid_preserves_all_half_payload_bytes():
    half = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.float16)
    uniform = torch.zeros((1, 10), device="cuda").softmax(-1).half().float()
    for start in range(0, 65536, 2048):
        shared = half[start:start + 2048].float()
        logits = torch.zeros((shared.numel(), 513), device="cuda")
        logits[:, 512] = shared
        picks = torch.empty((shared.numel(), 11), device="cuda", dtype=torch.int32)
        weights = torch.empty_like(picks, dtype=torch.float32)
        route(logits, picks, weights)
        expected = shared.sigmoid().half().float()
        torch.testing.assert_close(weights[:, 10].view(torch.int32), expected.view(torch.int32), atol=0, rtol=0)
        torch.testing.assert_close(weights[:, :10].contiguous().view(torch.int32),
                                   uniform.expand(shared.numel(), -1).contiguous().view(torch.int32), atol=0, rtol=0)
        expected_picks = torch.tensor([*range(10), 512], device="cuda", dtype=torch.int32).expand_as(picks)
        torch.testing.assert_close(picks, expected_picks, atol=0, rtol=0)


@pytest.mark.parametrize("count", [1, 10, 32])
def test_fused_weights_replay_dynamic_logits_with_exact_float_storage(count):
    logits = torch.randn((3, 513), device="cuda")
    picks, weights = check(logits, count)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        route(logits, picks, weights)
    for seed in (3, 17, 113):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        logits.copy_(torch.randn(logits.shape, device="cuda", generator=generator) * 3)
        if seed == 17:
            logits[1].fill_(-torch.inf)
            logits[2, 0] = torch.nan
        expected_picks, expected_weights = reference(logits, count)
        graph.replay()
        torch.testing.assert_close(picks, expected_picks, atol=0, rtol=0)
        torch.testing.assert_close(weights.view(torch.int32), expected_weights.view(torch.int32), atol=0, rtol=0)


def test_fused_route_refuses_overlapping_outputs_and_payloads():
    logits = torch.randn((3, 513), device="cuda")
    picks = torch.empty((3, 11), device="cuda", dtype=torch.int32)
    overlap = logits.reshape(-1)[:33].view(3, 11)
    with pytest.raises(RuntimeError, match="overlap"):
        route(logits, picks, overlap)
    weights = torch.empty((3, 11), device="cuda")
    with pytest.raises(RuntimeError, match="overlap"):
        route(logits, weights.view(torch.int32), weights)


@pytest.mark.parametrize("count", [1, 10, 32])
def test_indices_and_fused_weights_preserve_signed_nan_payload_radix_order(count):
    from tensorfold.cuda.sm70.routing_native import indices

    raw = torch.zeros((4, 513), device="cuda", dtype=torch.int32)
    positive = torch.tensor([0x7f800001, 0x7fc00000, 0x7fc00001, 0x7fffffff], device="cuda", dtype=torch.int32)
    negative = torch.tensor([-8388607, -4194304, -4194303, -1], device="cuda", dtype=torch.int32)
    raw[0, :4] = positive
    raw[0, 4:8] = negative
    raw[1, :512] = positive.repeat(128)
    raw[2, :512] = negative.repeat(128)
    raw[3, :512:2] = 0
    raw[3, 1:512:2] = -2147483648  # Stable signed-zero ties preserve ascending IDs.
    logits = raw.view(torch.float32)
    expected = logits[:, :512].argsort(dim=-1, descending=True, stable=True)[:, :count]
    torch.testing.assert_close(indices(logits, count), expected, atol=0, rtol=0)
    check(logits, count)
