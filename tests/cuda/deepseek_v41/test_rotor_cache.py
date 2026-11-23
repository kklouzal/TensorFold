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


def _split_rotate(value, valid, *, inverse=False):
    source = torch.where(valid[:, None, None], value, 0).contiguous()
    return native.rotate(source, torch.empty_like(source), codec=11, inverse=inverse, trap_on_error=True)


def _split_short_window_rope(value, positions, table, weight, eps):
    from tensorfold.families.deepseek_v4.cuda.kernels.norm import _short_rms
    from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import apply_rope

    normalized = torch.empty_like(value)
    _short_rms[(value.shape[0],)](value, weight, normalized, H=512, EPS=eps, BLOCK=512,
                                num_warps=4, enable_fp_fusion=False)
    return apply_rope(normalized, positions, table)


@pytest.mark.parametrize("rows", (1, 8, 24, 32, 64))
def test_h64_short_window_rope_matches_separate_fp16_boundaries(rows):
    from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import rope_table
    from tensorfold.families.deepseek_v4.cuda.kernels.rope_qdq import decode_window_rope

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(4190 + rows)
    base = torch.arange(rows, dtype=torch.int64, device=device)
    magnitudes = torch.logspace(-5, 3, 512, device=device)
    values = (torch.randn((rows, 512), generator=generator, device=device) * magnitudes).half()
    weight = torch.rand(512, generator=generator, device=device) + 0.5
    for compressed in (False, True):
        table = rope_table(16384, compressed=compressed, device=device)
        for start in (0, 127, 16_383, 65_535 - rows + 1):
            positions = base + start
            split = _split_short_window_rope(values, positions, table, weight, 1e-20)
            fused = decode_window_rope(values, positions, table, weight, 1e-20)
            assert torch.equal(fused.view(torch.int16), split.view(torch.int16)), (rows, start, compressed)
        values[0, 0:4] = torch.tensor([65504, -65504, 2.0**-24, -0.0], dtype=torch.float16, device=device)
        split = _split_short_window_rope(values, positions, table, weight, 1e-20)
        fused = decode_window_rope(values, positions, table, weight, 1e-20)
        assert torch.equal(fused.view(torch.int16), split.view(torch.int16)), (rows, compressed, "extreme")


def test_h64_short_window_rope_writer_replay_keeps_wrap_tail_and_padding_exact():
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
    from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import rope_table
    from tensorfold.families.deepseek_v4.cuda.kernels.rope_qdq import decode_window_rope

    device = "cuda"
    count, scratch = 8, 8
    generator = torch.Generator(device=device).manual_seed(4191)
    table = rope_table(16384, compressed=True, device=device)
    weight = torch.rand(512, generator=generator, device=device) + 0.5
    values = torch.zeros((count, 512), dtype=torch.float16, device=device)
    positions = torch.zeros(count, dtype=torch.int64, device=device)
    slots = torch.full((count,), scratch, dtype=torch.int32, device=device)
    live_flags = torch.ones(count, dtype=torch.bool, device=device)
    requests = torch.arange(count, dtype=torch.int32, device=device)
    qsl = torch.arange(count + 1, dtype=torch.int32, device=device)
    step = SimpleNamespace(single_token_slots=True, req_slot=slots, valid=live_flags)
    writer = RotorCache(device)
    ring = torch.full((scratch + 1, pk.RING, 272), 0xC4, dtype=torch.uint8, device=device)
    expected_ring = ring.clone()

    def forward():
        live = live_rows(step, scratch)
        physical = decode_window_rope(values, positions, table, weight, 1e-20)
        packed = writer.pack(physical, live, 7)
        pk.insert_ring(ring, packed, positions, live, requests, qsl, slots)
        return physical, packed

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        physical, packed = forward()
    for addresses, places in (
        ([0, 1, 2, scratch, 3, 4, scratch, 5], [127, 128, 16_383, 0, 16_384, 32_768, 0, 65_535]),
        ([scratch, 0, 1, 2, scratch, 3, 4, 5], [0, 128, 256, 16_384, 0, 16_385, 32_769, 65_535]),
    ):
        slots.copy_(torch.tensor(addresses, dtype=slots.dtype, device=device))
        positions.copy_(torch.tensor(places, dtype=positions.dtype, device=device))
        values.normal_(generator=generator)
        live = slots < scratch
        values[~live] = float("nan")
        graph.replay()
        expected = _split_short_window_rope(values, positions, table, weight, 1e-20)
        expected_packed = _split_pack(expected, live)
        assert torch.equal(physical[live].view(torch.int16), expected[live].view(torch.int16))
        assert torch.equal(packed, expected_packed)
        for row, (slot, pos) in enumerate(zip(addresses, places, strict=True)):
            if slot < scratch:
                expected_ring[slot, pos % pk.RING] = expected_packed[row]
        assert torch.equal(ring, expected_ring)
        assert writer.status.tolist() == [0, -2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rotorquant_row_adapter_keeps_masked_codec_and_basis_replay(dtype):
    cache = RotorCache("cuda")
    count, heads, dim = 4, 8, 512
    value = torch.empty(count, dim * 2, device="cuda", dtype=dtype)[:, ::2].normal_()
    queries = torch.empty(count, 2 * heads, 2 * dim, device="cuda", dtype=dtype)[:, ::2, ::2].normal_()
    scratch = 6
    slots = torch.tensor([1, scratch, 2, scratch], device="cuda", dtype=torch.int32)
    step = SimpleNamespace(single_token_slots=True, req_slot=slots, valid=torch.ones(count, device="cuda", dtype=torch.bool))
    valid = slots != scratch
    value[~valid] = float("nan")

    def adversarial_query():
        queries[:, :, 128:192] = -0.0
        queries[~valid] = float("nan")

    adversarial_query()

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
            adversarial_query()
        graph.replay()
        assert packed.shape == (count, 272)
        assert torch.equal(packed, _split_pack(value, valid))
        want_turned = _split_rotate(queries, valid)
        want_original = _split_rotate(want_turned, valid, inverse=True)
        bits = torch.int16 if dtype == torch.float16 else torch.int32
        assert torch.equal(turned.view(bits), want_turned.view(bits))
        assert torch.equal(original.view(bits), want_original.view(bits))
        assert cache.status.tolist() == [0, -2]


def test_h64_masked_rotation_strided_validity_and_disjoint_output():
    owner = RotorCache("cuda")
    values = torch.zeros((3, 1, 128), dtype=torch.float16, device="cuda")
    values[0, 0, ::2] = -0.0
    values[1].fill_(float("inf"))
    values[2, 0, 3] = 1.0
    valid = torch.tensor([True, False, False, True, True, False], device="cuda")[::2]
    for inverse in (False, True):
        actual = owner.rotate(values, valid, 0, inverse=inverse)
        expected = _split_rotate(values, valid, inverse=inverse)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert owner.status.tolist() == [0, -2]
    with pytest.raises(RuntimeError, match="disjoint"):
        native.rotate_h64(values, valid, values, status=owner.status, layer=0)
    with pytest.raises(RuntimeError, match="validity must match"):
        owner.rotate(values, valid[:1], 0)


def _masked_rotation_fault(fault):
    owner = RotorCache("cuda")
    values = torch.ones((2, 1, 128), dtype=torch.float16, device="cuda")
    valid = torch.tensor([True, False], device="cuda")
    inverse = fault == "overflow"
    if inverse:
        # Align with a column of the forward transform so its inverse
        # produces a finite FP32 value above the FP16 representable range.
        unit = torch.zeros((1, 1, 128), dtype=torch.float32, device="cuda")
        unit[0, 0, 0] = 1
        column = native.rotate(unit, torch.empty_like(unit), codec=11)
        values[1:].copy_(torch.where(column < 0, -65504.0, 65504.0))
        source = values[1:].float()
        large = native.rotate(source, torch.empty_like(source), codec=11, inverse=True)
        assert torch.isfinite(large).all() and large.abs().amax().item() > 65520
    else:
        values[1, 0, 7] = float("nan")

    owner.rotate(values, valid, 6, inverse=inverse)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rotated = owner.rotate(values, valid, 6, inverse=inverse)
    graph.replay()
    expected = _split_rotate(values, valid, inverse=inverse)
    assert torch.equal(rotated.view(torch.int16), expected.view(torch.int16))
    assert owner.status.tolist() == [0, -2]

    valid[1] = True
    try:
        graph.replay()
        torch.cuda.synchronize()
    except RuntimeError as error:
        assert any(reason in str(error).lower() for reason in
                   ("illegal instruction", "launch failure", "device-side assert")), error
        print(f"H64 masked rotation rejected replayed {fault}", flush=True)
    else:
        raise AssertionError(f"H64 masked rotation completed after {fault}")


@pytest.mark.parametrize("fault", ["nan", "overflow"])
def test_h64_masked_rotate_fault_on_replay_with_changed_validity(fault):
    invoke = "import runpy, sys; runpy.run_path(sys.argv[1])['_masked_rotation_fault'](sys.argv[2])"
    child = subprocess.run((sys.executable, "-B", "-c", invoke, __file__, fault),
                           capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    assert f"H64 masked rotation rejected replayed {fault}" in child.stdout


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


@pytest.mark.parametrize("count", [1, 5])
def test_h64_decode_ring_write_matches_post_attention_across_wrap_and_scratch(count):
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
    from tensorfold.families.deepseek_v4.cuda.target import Target

    device = "cuda"
    scratch = 3
    generator = torch.Generator(device=device).manual_seed(214)
    owner = RotorCache(device)
    target = Target.__new__(Target)
    target.rotor_cache = owner
    values = torch.empty((count, 512), dtype=torch.float16, device=device)
    query = torch.empty((count, 8, 512), dtype=torch.float16, device=device)
    sink = torch.linspace(-0.75, 1.25, 8, dtype=torch.float32, device=device)
    positions = torch.empty(count, dtype=torch.int64, device=device)
    slots = torch.empty(count, dtype=torch.int32, device=device)
    valid = torch.empty(count, dtype=torch.bool, device=device)
    token_req = torch.arange(count, dtype=torch.int32, device=device)
    qsl = torch.arange(count + 1, dtype=torch.int32, device=device)
    step = SimpleNamespace(positions=positions, req_slot=slots, valid=torch.ones_like(valid), qsl=qsl,
                           token_req=token_req, single_token_slots=True)
    history = torch.randn((scratch + 1) * pk.RING, 512, dtype=torch.float16, device=device, generator=generator)
    initial = _split_pack(history, torch.ones(len(history), dtype=torch.bool, device=device)).view(scratch + 1, pk.RING, 272)
    early, late = initial.clone(), initial.clone()
    live_slots = ([0], [0], [0]) if count == 1 else ([0, 2, 1, scratch, scratch], [2, 1, 0, scratch, scratch])
    live_positions = ([0], [127], [128]) if count == 1 else ([127, 128, -1, 0, 0], [128, 129, 255, 0, 0])

    def prepare(epoch):
        slots.copy_(torch.tensor(live_slots[epoch], device=device))
        positions.copy_(torch.tensor(live_positions[epoch], device=device))
        valid.copy_(live_rows(step, scratch))
        values.normal_(generator=generator)
        values[~valid] = float("nan")
        query.normal_(generator=generator)

    def fused():
        packed = owner.pack(values, valid, 2, decode_ring=(early, positions, slots))
        return packed, target._attend(query, step, early, packed, sink)

    prepare(0)
    fused()
    early.copy_(initial)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed, actual = fused()

    for epoch in range(len(live_positions)):
        if epoch:
            prepare(epoch)
        before = late.clone()
        want = _split_pack(values, valid)
        expected = target._attend(query, step, late, want, sink)
        pk.insert_ring(late, want, positions, valid, token_req, qsl, slots)
        graph.replay()
        assert torch.equal(packed, want)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(early, late)
        for row, (slot, position) in enumerate(zip(slots.tolist(), positions.tolist(), strict=True)):
            if slot < scratch and position >= 0:
                assert torch.equal(early[slot, position % pk.RING], packed[row])
            else:
                assert torch.equal(early[slot], before[slot])
        assert owner.status.tolist() == [0, -2]


def _decode_ring_fault():
    owner = RotorCache("cuda")
    values = torch.ones((2, 512), dtype=torch.float16, device="cuda")
    ring = torch.zeros((3, 128, 272), dtype=torch.uint8, device="cuda")
    slots = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    positions = torch.tensor([127, 128], dtype=torch.int64, device="cuda")
    valid = torch.ones(2, dtype=torch.bool, device="cuda")
    observed = torch.empty(ring.shape, dtype=torch.uint8, pin_memory=True).fill_(0xA5)
    sentinel = observed.clone()
    owner.pack(values, valid, 6, decode_ring=(ring, positions, slots))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = owner.pack(values, valid, 6, decode_ring=(ring, positions, slots))
    graph.replay()
    observed.copy_(ring, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(observed[0, 127], out[0].cpu()) and torch.equal(observed[1, 0], out[1].cpu())
    observed.copy_(sentinel)
    positions.add_(1)
    values[1, 257] = float("nan")
    try:
        graph.replay()
        observed.copy_(ring, non_blocking=True)
        torch.cuda.synchronize()
    except RuntimeError as error:
        assert any(reason in str(error).lower() for reason in
                   ("illegal instruction", "launch failure", "device-side assert")), error
        assert torch.equal(observed, sentinel), "failed ring reached the ordered host observer"
        print("H64 decode ring fault blocked the ordered host observer", flush=True)
    else:
        raise AssertionError("H64 decode ring fault reached successful publication")


def test_h64_decode_ring_fault_blocks_ordered_host_observation():
    invoke = "import runpy, sys; runpy.run_path(sys.argv[1])['_decode_ring_fault']()"
    child = subprocess.run((sys.executable, "-B", "-c", invoke, __file__), capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "H64 decode ring fault blocked the ordered host observer" in child.stdout


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
