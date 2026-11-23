"""Native DSpark verify snapshots versus the independently committed short forward."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0), reason="SM70 required")
DEVICE = "cuda"


def _step(requests, slots_total, *, masked=()):
    """requests [(slot, start, count)] → one request-ordered native step."""
    from tensorfold.families.deepseek_v4.cuda.target import Step

    positions, token_req, qsl, ids = [], [], [0], []
    for request, (slot, start, count) in enumerate(requests):
        positions.extend(range(start, start + count))
        token_req.extend([request] * count)
        ids.extend((p * 7 + slot * 13) % 500 + 3 for p in range(start, start + count))
        qsl.append(qsl[-1] + count)

    def tensor(values, dtype):
        return torch.tensor(values, dtype=dtype, device=DEVICE)

    live = torch.ones(len(ids), dtype=torch.bool, device=DEVICE)
    if masked:
        live[list(masked)] = False
    req_slot = tensor([slot for slot, _, _ in requests] + [slots_total] * 8, torch.int32)
    return Step(ids=tensor(ids, torch.long), positions=tensor(positions, torch.long), valid=live,
                token_req=tensor(token_req, torch.int32), qsl=tensor(qsl, torch.int32), qsl_cpu=qsl,
                num_reqs=len(requests), req_slot=req_slot,
                block_table=torch.full((len(requests), 1), -1, dtype=torch.int32, device=DEVICE),
                prefill=[], decode_index=None)


def _target(slots, layers, rotorquant):
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_state
    from tensorfold.families.deepseek_v4.cuda.kernels.mu_packed import PackedLayout
    from tensorfold.families.deepseek_v4.cuda.rotor_cache import RotorCache
    from tensorfold.families.deepseek_v4.cuda.target import Pool

    generator = torch.Generator().manual_seed(5)
    token_map = torch.randint(0, 400, (512,), generator=generator).to(DEVICE)
    primes = torch.randint(1000, 5000, (2, 24), generator=generator).to(DEVICE)
    offsets = torch.randint(0, 100, (2, 24), generator=generator).to(DEVICE)
    multipliers = torch.randint(1, 50, (2, 4), generator=generator).to(DEVICE)
    hs = mu_state.MuEngramHashState(token_map, primes, offsets, multipliers, slots=slots, pad_token_id=2,
                                   max_tokens=512, max_context=16_384)
    c2 = {3: mu_state.MuC2State(torch.rand(512, generator=generator).to(DEVICE) + 0.5, slots=slots)}
    layout = PackedLayout({3: 2}, 64, rotorquant=rotorquant)
    pool = Pool(SimpleNamespace(layers=layers), 1, slots + 1, DEVICE, layout=layout)
    rotor = RotorCache(DEVICE) if rotorquant else None
    return SimpleNamespace(hash_state=hs, c2=c2, pool=pool, rotor=rotor)


def _forward(target, step, values, scores, spec=None):
    """Use the production Engram, C2, H64 encoder and window insertion kernels."""
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed

    hs = target.hash_state
    hs(step.ids, step.positions, step.valid, step.token_req, step.qsl, step.req_slot)
    if spec is not None:
        spec.snapshot_hash(step)
    else:
        hs.commit_all(step.ids, step.positions, step.valid, step.qsl, step.req_slot, step.num_reqs, step.token_req)
    for layer, state in target.c2.items():
        state(values, scores, step.positions, step.token_req, step.qsl, step.req_slot)
        if spec is not None:
            spec.snapshot_c2(layer, values, scores, step)
        else:
            state.commit_all(values, scores, step.positions, step.qsl, step.req_slot, step.num_reqs, step.token_req)
    column = torch.arange(512, device=DEVICE) / 512 - 0.5
    amplitude = 0.25 + step.positions.float()[:, None] / 64
    for layer, ring in enumerate(target.pool.rings):
        raw = ((amplitude + layer / 8) * column[None, :]).half()
        window = target.rotor.pack(raw, step.valid, layer) if target.rotor is not None else raw
        if spec is not None:
            spec.backup_ring(layer, step)
        mu_packed.insert_ring(ring, window, step.positions, step.valid, step.token_req, step.qsl, step.req_slot)


def _committed(target):
    history = target.hash_state.history.clone()
    c2 = tuple(t.clone() for state in target.c2.values() for t in (state.partial, state.committed))
    rings = tuple(ring.view(torch.uint8).clone() for ring in target.pool.rings)
    return history, *c2, *rings


def _assert_exact(first, second):
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("rotorquant", [False, True], ids=["fp16", "h64"])
def test_verify_commit_equals_short_forward_with_wrapping_and_cancel(rotorquant):
    from tensorfold.families.deepseek_v4.cuda.speculation import Speculation

    slots, layers = 3, 2
    speculative = _target(slots, layers, rotorquant)
    reference = _target(slots, layers, rotorquant)
    spec = Speculation(speculative)
    rings = speculative.pool.rings
    payload = (spec.cand_n, spec.history, spec._addresses, spec._backup, spec._saved, *spec.partials, *spec.positions)
    row_bytes = 272 if rotorquant else 1024
    assert spec.nbytes == Speculation.state_bytes(slots, layers, 1, row_bytes)
    assert spec.nbytes == sum(tensor.nbytes for tensor in payload)
    assert tuple(view.data_ptr() for view in spec._rings) == tuple(ring.data_ptr() for ring in rings)

    generator = torch.Generator().manual_seed(9)
    prompt = _step([(0, 0, 127), (1, 0, 128), (2, 0, 255)], slots)
    values = torch.randn(len(prompt.ids), 512, generator=generator).to(DEVICE)
    for target in (speculative, reference):
        _forward(target, prompt, values, values * 0.5)
    verify = _step([(0, 127, 6), (1, 128, 6), (2, 255, 6), (slots, 0, 6)], slots, masked=(5,))
    values = torch.randn(len(verify.ids), 512, generator=generator).to(DEVICE)
    _forward(speculative, verify, values, values * 0.5, spec)
    short = _step([(0, 127, 3), (1, 128, 6)], slots)
    keep = torch.tensor([0, 1, 2, 6, 7, 8, 9, 10, 11], device=DEVICE)
    accepted = values[keep]
    _forward(reference, short, accepted, accepted * 0.5)

    # Compile the exact commit shape with no live commits, then replay with accepted, rejected and padded slots.
    req_slots = torch.arange(slots + 1, device=DEVICE, dtype=torch.int32)
    counts = torch.full((slots + 1,), -1, device=DEVICE, dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        spec.commit(req_slots, counts)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        spec.commit(req_slots, counts)
    counts.copy_(torch.tensor([3, 6, 0, 0], device=DEVICE, dtype=torch.int64))
    graph.replay()
    _assert_exact(_committed(speculative), _committed(reference))
    assert int(speculative.hash_state.err) == 0 and int(speculative.c2[3].err) == 0
    if rotorquant:
        assert torch.count_nonzero(rings[0][0, 127, 256:]).item() > 0
    # This continuation consumes the accepted C2 partial, including the cancelled slot's earlier partial.
    resumed = _step([(0, 130, 2), (1, 134, 2), (2, 255, 2)], slots)
    values = torch.randn(len(resumed.ids), 512, generator=generator).to(DEVICE)
    for target in (speculative, reference):
        _forward(target, resumed, values, values * 0.5)
    _assert_exact(_committed(speculative), _committed(reference))
    assert int(speculative.hash_state.err) == 0 and int(speculative.c2[3].err) == 0


def test_commit_rejects_missing_and_out_of_range_counts_without_mutation():
    from tensorfold.families.deepseek_v4.cuda.kernels.mu_state import ERR_CONTINUE
    from tensorfold.families.deepseek_v4.cuda.speculation import Speculation

    target = _target(2, 2, True)
    spec = Speculation(target)
    counts = torch.full((2,), -1, dtype=torch.int64, device=DEVICE)
    slots = torch.arange(2, device=DEVICE, dtype=torch.int32)
    before = _committed(target)
    spec.commit(slots, counts)
    _assert_exact(_committed(target), before)
    assert int(target.hash_state.err) == 0 and int(target.c2[3].err) == 0
    verify = _step([(0, 0, 4)], 2)
    values = torch.full((4, 512), 0.125, device=DEVICE)
    _forward(target, verify, values, values, spec)

    # Slot zero has four candidates. Slot one has never run a verify forward.
    before = _committed(target)
    counts[0] = torch.iinfo(torch.int64).max
    counts[1] = 0
    spec.commit(slots, counts)
    _assert_exact(_committed(target), before)
    assert int(target.hash_state.err) == ERR_CONTINUE and int(target.c2[3].err) == ERR_CONTINUE
