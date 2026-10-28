"""Strict original FP16 chain/replay/commit oracles for State scheduling only."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from tensorfold.cuda.sm70.gdn_native import extension  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import gdn  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gdn_state_batch import GDNStateBatch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="requires actual SM70 FP16 execution")


class ChainState:
    """Own the original State recurrence/scratch ABI without unrelated KV allocation."""
    def __init__(self, layers, parity):
        self.rec = torch.randn(2, layers, 6, 128, 128, device="cuda") * .01
        self.conv = torch.randn(layers, 3, 1280, dtype=torch.float16, device="cuda") * .1
        self.scratch = [gdn.GDNScratch(3, "cuda", 2, 6, dtype=torch.float16) for _ in range(layers)]
        self.cur = [parity] * layers
        self.slot_generation = self.history_generation = self.pos = 0
        self.ple_last = self.dflash2 = None

    def clone(self):
        other = ChainState(len(self.cur), 0)
        other.rec.copy_(self.rec)
        other.conv.copy_(self.conv)
        other.cur[:] = self.cur
        return other

    def kv_check(self):
        pass

    def set_pos(self, value):
        self.pos = value


def same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def fixture(states, direct=True, pattern="mixed", layers=2):
    torch.manual_seed(5031 + states + layers)
    originals = [ChainState(layers, index % 2) for index in range(states)]
    candidates = [state.clone() for state in originals]
    counts = [1 if pattern == "ones" else 3 if pattern == "three" else index % 3 + 1 for index in range(states)]
    first = 0
    old, new = [], []
    for reference, candidate, count in zip(originals, candidates, counts, strict=True):
        old.append((reference, first, first + count))
        new.append((candidate, first, first + count))
        first += count
    projection = torch.randn(layers, first, 2060, device="cuda", dtype=torch.float16) * .1
    weight = torch.randn(1280, 4, device="cuda", dtype=torch.float16) * .1
    parameter = -torch.rand(6, device="cuda") if direct else torch.randn(6, device="cuda") * .1
    bias = torch.randn(6, device="cuda") * .1
    norm = torch.ones(128, device="cuda", dtype=torch.float16)
    outputs = [torch.full((layers, first, 768), -99., device="cuda", dtype=torch.float16) for _ in range(2)]
    sums = [torch.full((layers, first, 24), -99., device="cuda") for _ in range(2)]
    owner = GDNStateBatch(layers, torch.device("cuda", torch.cuda.current_device()))
    return SimpleNamespace(old=old, new=new, p=projection, weight=weight, parameter=parameter,
                           bias=bias, norm=norm, outputs=outputs, sums=sums, owner=owner,
                           layers=layers, direct=direct)


def original(f):
    for layer in range(f.layers):
        for state, first, last in f.old:
            cur = state.cur[layer]
            gdn.chain(f.p[layer, first:last], state.conv[layer], f.weight, state.rec[cur, layer],
                      None if f.direct else f.parameter, f.bias, f.norm, 1e-6, last - first,
                      state.scratch[layer], state.rec[1 - cur, layer], f.outputs[0][layer, first:last],
                      f.sums[0][layer, first:last], a_neg=f.parameter if f.direct else None)


def candidate(f):
    enabled = f.owner.bind(f.new)
    if not enabled:
        assert len(f.new) == 1
        for layer in range(f.layers):
            state, first, last = f.new[0]
            cur = state.cur[layer]
            gdn.chain(f.p[layer, first:last], state.conv[layer], f.weight, state.rec[cur, layer],
                      None if f.direct else f.parameter, f.bias, f.norm, 1e-6, last - first,
                      state.scratch[layer], state.rec[1 - cur, layer], f.outputs[1][layer, first:last],
                      f.sums[1][layer, first:last], a_neg=f.parameter if f.direct else None)
        return
    try:
        for layer in range(f.layers):
            assert f.owner.submit(layer, f.p[layer], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                                  f.outputs[1][layer], f.sums[1][layer], negative_decay=f.direct)
    finally:
        f.owner.seal()
    f.owner.retire()


def compare(f, *, do_commit=True):
    same(f.outputs[0], f.outputs[1])
    same(f.sums[0], f.sums[1])
    for (old, first, last), (new, _, _) in zip(f.old, f.new, strict=True):
        same(old.rec, new.rec)  # Includes original read-only bank and tentative final bank.
        same(old.conv, new.conv)  # Chain has no convolution-state mutation.
        for a, b in zip(old.scratch, new.scratch, strict=True):
            for name in ("k", "v", "g", "b"):
                same(getattr(a, name)[:last - first], getattr(b, name)[:last - first])
        if do_commit:
            kept = first % (last - first) + 1
            weights = SimpleNamespace(cfg=SimpleNamespace(conv_dim=1280), dflash2=None, dspark=None)
            buffers = SimpleNamespace(h=f.outputs[0][0], prefill=False, proj=f.p)
            commit(weights, old, buffers, last - first, kept, at=first)
            commit(weights, new, buffers, last - first, kept, at=first)
            same(old.rec, new.rec)
            same(old.conv, new.conv)
            assert old.cur == new.cur and old.pos == new.pos


@pytest.mark.parametrize("states", [1, 8, 16, 24])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("pattern", ["ones", "mixed", "three"])
def test_original_chain_scratch_tentative_replay_commit_two_rounds(states, direct, pattern):
    f = fixture(states, direct, pattern)
    for _round in range(2):
        original(f)
        candidate(f)
        compare(f)
    if states == 1:
        assert f.owner.submitted == 0
    else:
        assert f.owner.submitted == 2 * f.layers
        assert all(int(f.owner.mapping[index, 3]) == 0 for index in range(states))


def test_actual36_linear_layer_catalogue_offsets():
    f = fixture(24, layers=36)
    original(f)
    candidate(f)
    compare(f)


@pytest.mark.parametrize("states", [8, 24])
@pytest.mark.parametrize("direct", [False, True])
def test_native_fixed_pointee_graph_capture(states, direct):
    f = fixture(states, direct)
    original(f)
    assert f.owner.bind(f.new)
    module = extension()
    # Warm and capture only native launches. Retain all actual State/tensor
    # pointees through graph destruction; model graph dispatch stays original.
    for layer in range(f.layers):
        module.batch(f.p[layer], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                     f.outputs[1][layer], f.sums[1][layer], f.owner.host, f.owner.frame, layer, states, direct)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(f.layers):
            module.batch(f.p[layer], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                         f.outputs[1][layer], f.sums[1][layer], f.owner.host, f.owner.frame, layer, states, direct)
    graph.replay()
    torch.cuda.synchronize()
    compare(f)
    del graph
    f.owner.retire()


@pytest.mark.parametrize("kept", [0, 1, 2, 3])
def test_live_keep_control_and_original_replay_exact(kept):
    f = fixture(8, pattern="three")
    for old_segments, new_segments in zip(f.old, f.new, strict=True):
        for state in (old_segments[0], new_segments[0]):
            for scratch in state.scratch:
                scratch.keep_rows = torch.tensor([kept], dtype=torch.int32, device="cuda")
    original(f)
    candidate(f)
    compare(f, do_commit=False)
    for old, new in zip(f.old, f.new, strict=True):
        for state in (old[0], new[0]):
            for layer in range(f.layers):
                current = state.cur[layer]
                replayed = torch.empty_like(state.rec[current, layer])
                gdn.replay(state.rec[current, layer], state.scratch[layer], kept, replayed)
                same(replayed, state.rec[1 - current, layer])


@pytest.mark.parametrize("bad", ["shared-scratch", "shared-recurrence", "rec-dtype", "scratch-dtype", "conv-shape", "duplicate", "gap"])
def test_invalid_binding_or_alias_never_launches(bad):
    f = fixture(8)
    a, b = f.new[0][0], f.new[1][0]
    if bad == "shared-scratch":
        b.scratch[0] = a.scratch[0]
    elif bad == "shared-recurrence":
        b.rec = a.rec
    elif bad == "rec-dtype":
        b.rec = b.rec.half()
    elif bad == "scratch-dtype":
        b.scratch[0].v = b.scratch[0].v.float()
    elif bad == "conv-shape":
        b.conv = b.conv[:, :2]
    elif bad == "duplicate":
        f.new[1] = (a, 1, 3)
    else:
        f.new[1] = (b, 2, 4)
    with pytest.raises((ValueError, RuntimeError)):
        f.owner.bind(f.new)
    assert f.owner.submitted == 0


@pytest.mark.parametrize("after", [0, 1, 2])
def test_cancel_before_or_after_submitted_chain_retains_then_retires_states(after):
    f = fixture(8)
    assert f.owner.bind(f.new)
    with pytest.raises(KeyboardInterrupt):
        try:
            for layer in range(after):
                f.owner.submit(layer, f.p[layer], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                               f.outputs[1][layer], f.sums[1][layer], negative_decay=True)
            raise KeyboardInterrupt("bounded component cancellation control")
        finally:
            f.owner.seal()
    assert f.owner.owners[0] is f.new[0][0]
    f.owner.retire()
    assert all(owner is None for owner in f.owner.owners)


@pytest.mark.parametrize("after", [0, 1])
def test_generation_change_between_binding_and_submission_is_rejected(after):
    f = fixture(8)
    f.owner.bind(f.new)
    if after:
        f.owner.submit(0, f.p[0], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                       f.outputs[1][0], f.sums[1][0], negative_decay=True)
    f.new[0][0].slot_generation += 1
    try:
        with pytest.raises(ValueError, match="storage changed"):
            f.owner.submit(after, f.p[after], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                           f.outputs[1][after], f.sums[1][after], negative_decay=True)
    finally:
        f.owner.seal()
        f.owner.retire()


def test_cross_stream_submission_rejected():
    f = fixture(8)
    f.owner.bind(f.new)
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(ValueError, match="cross"):
            f.owner.submit(0, f.p[0], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                           f.outputs[1][0], f.sums[1][0], negative_decay=True)
    f.owner.retire()


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("prefix", [0, 1])
def test_failure_before_or_after_native_acceptance_keeps_frontier(phase, prefix, monkeypatch):
    from tensorfold.cuda.sm70 import gdn_native

    f = fixture(8)
    original(f)
    f.owner.bind(f.new)
    module = extension()
    for layer in range(prefix):
        f.owner.submit(layer, f.p[layer], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                       f.outputs[1][layer], f.sums[1][layer], negative_decay=True)

    def failing(*args):
        if phase == "after":
            module.batch(*args)
        raise RuntimeError("injected host failure around accepted native call")

    monkeypatch.setattr(gdn_native, "extension", lambda: SimpleNamespace(batch=failing))
    try:
        with pytest.raises(RuntimeError, match="injected host failure"):
            f.owner.submit(prefix, f.p[prefix], f.weight, f.parameter, f.bias, f.norm, 1e-6,
                           f.outputs[1][prefix], f.sums[1][prefix], negative_decay=True)
    finally:
        f.owner.seal()
    assert f.owner.pending and f.owner.owners[0] is f.new[0][0]
    f.owner.retire()
    for layer in range(prefix + int(phase == "after")):
        same(f.outputs[0][layer], f.outputs[1][layer])
        same(f.sums[0][layer], f.sums[1][layer])
