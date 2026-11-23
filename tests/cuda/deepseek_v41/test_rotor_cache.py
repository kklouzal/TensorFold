"""The DeepSeek row adapter preserves the registered codec bytes across graph replay."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import rotorquant_q4_native as native
from tensorfold.families.deepseek_v4.cuda.rotor_cache import RotorCache, live_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="SM70 CUDA required")


def test_rotorquant_row_adapter_keeps_masked_codec_and_basis_replay():
    cache = RotorCache("cuda")
    count, heads, dim = 4, 8, 512
    value = torch.randn(count, dim, device="cuda", dtype=torch.float16)
    queries = torch.randn(count, heads, dim, device="cuda", dtype=torch.float16)
    scratch = 6
    slots = torch.tensor([1, scratch, 2, scratch], device="cuda", dtype=torch.int32)
    step = SimpleNamespace(single_token_slots=True, req_slot=slots, valid=torch.ones(count, device="cuda", dtype=torch.bool))
    valid = slots != scratch
    value[~valid] = float("nan")
    queries[~valid] = float("nan")

    def forward():
        live = live_rows(step, scratch)
        packed = cache.pack(value, live, 7)
        turned = cache.rotate(queries, live, 7)
        return packed, turned, cache.rotate(turned, live, 7, inverse=True)

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed, turned, original = forward()
    for epoch in range(2):
        if epoch:
            slots.copy_(torch.tensor([scratch, 4, scratch, 3], device="cuda", dtype=torch.int32))
            valid = slots != scratch
            value.normal_()
            queries.normal_()
            value[~valid] = float("nan")
            queries[~valid] = float("nan")
        graph.replay()
        source = value.clone()
        source[~valid] = 0
        source = source.unsqueeze(1)
        codes = torch.empty(count, 1, 256, dtype=torch.uint8, device="cuda")
        scale = torch.empty(count, 1, 4, dtype=torch.float32, device="cuda")
        native.encode(source, codes, scale, codec=11)
        assert packed.shape == (count, 272)
        assert torch.equal(packed[:, :256], codes[:, 0])
        assert torch.equal(packed[:, 256:], scale[:, 0].view(torch.uint8))
        query = queries.clone()
        query[~valid] = 0
        want_turned, want_original = torch.empty_like(query), torch.empty_like(query)
        native.rotate(query, want_turned, codec=11)
        native.rotate(want_turned, want_original, codec=11, inverse=True)
        assert torch.equal(turned.view(torch.int16), want_turned.view(torch.int16))
        assert torch.equal(original.view(torch.int16), want_original.view(torch.int16))
        assert cache.status.tolist() == [0, -2]
