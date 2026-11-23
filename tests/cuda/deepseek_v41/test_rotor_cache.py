"""The DeepSeek row adapter preserves the registered codec bytes across graph replay."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import rotorquant_q4_native as native
from tensorfold.families.deepseek_v4.cuda.rotor_cache import RotorCache, live_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="SM70 CUDA required")


def _split_pack(value, valid):
    rows = value.shape[0]
    codes = torch.empty((rows, 1, 256), dtype=torch.uint8, device=value.device)
    scales = torch.empty((rows, 1, 4), dtype=torch.float32, device=value.device)
    native.encode(torch.where(valid[:, None], value, 0).unsqueeze(1), codes, scales, codec=11)
    return torch.cat((codes[:, 0], scales[:, 0].view(torch.uint8)), dim=1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rotorquant_row_adapter_keeps_masked_codec_and_basis_replay(dtype):
    cache = RotorCache("cuda")
    count, heads, dim = 4, 8, 512
    value = torch.empty(count, dim * 2, device="cuda", dtype=dtype)[:, ::2].normal_()
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
        assert packed.shape == (count, 272)
        assert torch.equal(packed, _split_pack(value, valid))
        query = queries.clone()
        query[~valid] = 0
        want_turned, want_original = torch.empty_like(query), torch.empty_like(query)
        native.rotate(query, want_turned, codec=11)
        native.rotate(want_turned, want_original, codec=11, inverse=True)
        assert torch.equal(turned.view(torch.int16), want_turned.view(torch.int16))
        assert torch.equal(original.view(torch.int16), want_original.view(torch.int16))
        assert cache.status.tolist() == [0, -2]


def test_h64_packed_writer_bytes_survive_prefix_copy_and_restore():
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
    from tensorfold.families.deepseek_v4.cuda.prefix import SlotState
    from tensorfold.families.deepseek_v4.cuda.target import BLOCK, Pool

    device = "cuda"
    count, source = 132, 2
    value = torch.randn((count, 512), dtype=torch.float16, device=device)
    value[:, 64:128] *= 7
    value[:, 128:192] = -0.0
    ages = torch.arange(count, dtype=torch.int64, device=device)
    valid = ((torch.arange(2 * count, device=device) // 2) % 7 != 0)[::2]
    value[~valid] = float("inf")
    writer = RotorCache(device)
    packed = writer.pack(value, valid, source)
    expected = _split_pack(value, valid)
    assert torch.equal(packed, expected)
    valid = valid.contiguous()

    layout = pk.PackedLayout({source: 2}, BLOCK, rotorquant=True)
    pool = Pool(SimpleNamespace(layers=1), pages=16, slots=3, device=device, layout=layout)
    table = torch.tensor([pool.attach(slot, count) for slot in (0, 1)], dtype=torch.int32, device=device)
    requests = torch.zeros(count, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, count], dtype=torch.int32, device=device)
    ring = pool.rings[0]
    pk.insert_ring(ring, packed, ages, valid, requests, qsl, requests[:1])
    pk.insert_rows(pool.cache, packed, ages, valid, requests, table, layout=layout, layer=source, compressed=True)

    hash_state = SimpleNamespace(history=torch.arange(12, dtype=torch.int64, device=device).view(3, 4))
    state = SlotState(SimpleNamespace(pool=pool, c2={}, hash_state=hash_state))
    state.snap(0, count, 19)
    state.copy(0, 1, count)
    pool.reset_ring(0)
    assert state.restore(1, 19) == count

    expected_ring = torch.zeros_like(ring[0])
    kept = (ages >= count - pk.RING) & valid
    expected_ring[ages[kept] % pk.RING] = expected[kept]
    assert torch.equal(ring[1], expected_ring)
    got_pages = pk.gather_request_rows(pool.cache, table[1], count // 2, layout=layout, layer=source, compressed=True)
    keep_compact = valid[1::2, None]
    assert torch.equal(got_pages, torch.where(keep_compact, expected[1::2], 0))
    assert torch.equal(hash_state.history[0], hash_state.history[1])
    assert writer.status.tolist() == [0, -2]


def _packed_writer_fault():
    writer = RotorCache("cuda")
    values = torch.ones((2, 512), dtype=torch.float16, device="cuda")
    values[1, 7] = float("nan")
    valid = torch.tensor([True, False], device="cuda")
    writer.pack(values, valid, 6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = writer.pack(values, valid, 6)
    graph.replay()
    assert torch.equal(packed, _split_pack(values, valid))
    assert writer.status.tolist() == [0, -2]
    valid[1] = True
    try:
        graph.replay()
        torch.cuda.synchronize()
    except RuntimeError as error:
        assert any(reason in str(error).lower() for reason in
                   ("illegal instruction", "launch failure", "device-side assert")), error
        print("H64 packed writer rejected a replayed live NaN", flush=True)
    else:
        raise AssertionError("H64 packed writer published a replayed live NaN")


def test_h64_packed_writer_fault_on_replay_with_changed_validity():
    invoke = "import runpy, sys; runpy.run_path(sys.argv[1])['_packed_writer_fault']()"
    child = subprocess.run((sys.executable, "-B", "-c", invoke, __file__), capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "H64 packed writer rejected a replayed live NaN" in child.stdout


def _cold_fault(boundary):
    from tensorfold.families.deepseek_v4.cuda.kernels.mu_packed import RING
    from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import rope_table
    from tensorfold.families.deepseek_v4.cuda.target import Step, Target

    tokens = 9
    rows = torch.zeros((tokens, 512), device="cuda", dtype=torch.float16)
    query = torch.zeros((tokens, 8 * 512), device="cuda", dtype=torch.float16)
    sink = torch.zeros(8, device="cuda", dtype=torch.float32)
    live = torch.ones(tokens, device="cuda", dtype=torch.bool)
    request = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    page_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    step = Step(ids=request.long(), positions=torch.arange(tokens, device="cuda"), valid=live, token_req=request,
                qsl=torch.tensor([0, tokens], device="cuda", dtype=torch.int32), qsl_cpu=[0, tokens], num_reqs=1,
                req_slot=request[:1], block_table=page_table, prefill=[(0, 0, tokens, tokens - 1)], decode_index=None)
    target = Target.__new__(Target)
    target.rotor_cache = RotorCache("cuda")
    target.pool = SimpleNamespace(rings=[torch.zeros((2, RING, 272), device="cuda", dtype=torch.uint8)])
    target.c = SimpleNamespace(sum=lambda value: value)
    target._norm = lambda value, _weight: value
    target._linear = lambda value, weight: weight if weight is query else value
    layer = SimpleNamespace(index=0, qa_kv=lambda value: (value, rows), q_norm=None, kv_norm=None,
                            wq_b=query, sink=sink, ratio=0, wo_a=None, wo_b=None)
    shared = {"rotor_valid": live, "rotor_cold_legacy": {
        "ring": torch.empty((1, RING, 512), device="cuda", dtype=torch.float16), "table": page_table}}
    rope = rope_table(tokens, device="cuda")
    poisoned = False

    def read_attention(q, *_args, **_kwargs):
        if poisoned and boundary == "query":
            torch.cuda.synchronize()
            raise AssertionError("nonfinite query reached the attention reader")
        if poisoned and boundary == "output":
            return torch.full_like(q, float("inf"))
        return torch.zeros_like(q)

    target._attend = read_attention
    result = target.attention(layer, rows, step, rope, shared)
    torch.cuda.synchronize()
    assert torch.count_nonzero(result) == 0
    poisoned = True
    if boundary == "query":
        query[0, 0] = float("nan")
    try:
        target.attention(layer, rows, step, rope, shared)
        torch.cuda.synchronize()
    except RuntimeError as error:
        assert "device-side assert" in str(error).lower(), error
        print(f"cold {boundary} assertion trapped")
    else:
        raise AssertionError(f"nonfinite cold attention {boundary} completed")


@pytest.mark.parametrize("boundary", ["query", "output"])
def test_rotorquant_cold_attention_traps_nonfinite_operands(boundary):
    invoke = "import runpy, sys; runpy.run_path(sys.argv[1])['_cold_fault'](sys.argv[2])"
    child = subprocess.run((sys.executable, "-B", "-c", invoke, __file__, boundary),
                           capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    assert f"cold {boundary} assertion trapped" in child.stdout
