"""Native whole-family tap/cache composition with synthetic draft weights."""

from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda import forward
from tensorfold.families.qwen4_exp.cuda.decode import Engine
from tensorfold.families.qwen4_exp.cuda.dspark_config import DSparkConfig
from tensorfold.families.qwen4_exp.cuda.dspark_cuda import DSparkCUDA, DeviceWeights
from tensorfold.families.qwen4_exp.cuda.dspark_runtime import DSparkRuntime
from tensorfold.families.qwen4_exp.cuda.dspark_weights import tensor_shapes
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State, KVNumericError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native family composition")


@pytest.fixture(autouse=True)
def math_policy(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 DSpark runtime scope")
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)


def weights():
    from test_flashnext_sm70_family import _small_weights

    w = _small_weights(True, hidden=2560, lowrank=320, head_dim=256)
    w.layers[1].ple, w.layers[0].ple = w.layers[0].ple, None
    w.cfg.ple_layers = [1]
    return w


def attach(w):
    c = DSparkConfig(
        w.cfg.hidden, w.cfg.vocab, 1, 4, 2, 128, 64, w.cfg.layers, (0,), w.cfg.vocab - 1, 4, 3, 64, 1e-6, 10000
    )
    g = torch.Generator(device="cuda").manual_seed(3001)
    tensors = {
        name: (torch.randn(shape, device="cuda", generator=g) * 0.003 + (1 if "norm" in name else 0)).to(
            torch.float32 if "norm" in name or name.startswith("confidence_head.") else torch.float16
        )
        for name, shape in tensor_shapes(c).items()
    }
    draft = DeviceWeights(c, tensors, "EXPLICIT-TINY-SYNTHETIC-DRAFT")
    owner = DSparkRuntime()
    owner.plan = NS(config=c, identity=c.identity + "|native-composition", cache_row_bytes=1 * 2 * 128 * 4)
    owner.weights = draft
    owner.owned = dict(tensors)

    def embed(ids):
        out = torch.empty((ids.numel(), c.hidden), dtype=torch.float16, device=w.device)
        return forward._embed(w, ids, 1, out)

    owner.provider = DSparkCUDA(
        draft,
        target=w,
        embed=embed,
        head=lambda x, s: w.head.matmul(x, f32=True),
        target_identity="shared-tiny-target",
        prefill_rows=8,
    )
    w.dspark = owner
    return owner


def exact(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def target_equal(a, b, physical=True):
    assert a.pos == b.pos
    if physical:
        assert a.cur == b.cur
    for i, (ap, bp) in enumerate(zip(a.cur, b.cur, strict=True)):
        exact(a.rec[ap, i], b.rec[bp, i])
    for name in ("conv", "ple_tail"):
        exact(getattr(a, name), getattr(b, name))
    for ka, kb in zip(a.kc, b.kc, strict=True):
        for n in ("k", "v", "ks", "vs"):
            torch.testing.assert_close(getattr(ka, n)[: a.pos], getattr(kb, n)[: b.pos], rtol=0, atol=0)


def compute(w, st, b, ids, *, cut=None, check=True):
    b.ids[: len(ids)].copy_(torch.tensor(ids, dtype=torch.int32, device=w.device))
    b.ple_v.fill_(0.125)
    return forward.compute(w, [(st, 0, len(ids))], b, cuts=() if cut is None else (cut,), check=check).clone()


@pytest.mark.parametrize("cache", ["fp16", "int8"])
@pytest.mark.parametrize("prefill", [False, True])
def test_native_target_bytes_unchanged_auxiliary_only_accepts_committed_rows(cache, prefill):
    base, w = weights(), weights()
    owner = attach(w)
    try:
        a, b = State(base, 32, 8, cache), State(w, 32, 8, cache)
        ab, bb = Buffers(base, 8, 32, prefill=prefill), Buffers(w, 8, 32, prefill=prefill)
        expected, actual = compute(base, a, ab, [4, 7, 19]), compute(w, b, bb, [4, 7, 19])
        exact(actual, expected)
        assert b.dspark.length == 0 and b.dspark.pending.rows == 3
        keep = 3 if prefill else 2
        forward.commit(base, a, ab, 3, keep)
        forward.commit(w, b, bb, 3, keep)
        target_equal(a, b)
        assert b.dspark.length == keep and b.dspark.pending is None
    finally:
        owner.close()


def test_native_prefix_cut_copy_restore_reset_and_growth_use_auxiliary_marker():
    base, w = weights(), weights()
    owner = attach(w)
    try:
        groups = []
        for current in (base, w):
            state = State(current, 32, 8, "int8", limit=64)
            buf = Buffers(current, 8, 32, prefill=True)
            cut = forward.Cut(2)
            compute(current, state, buf, [4, 7, 19], cut=cut)
            snap = forward.cut_snapshot(current, state, buf, cut, 0)
            forward.commit(current, state, buf, 3, 3)
            copied = State(current, 32, 8, "int8", limit=64)
            copied.copy_prefix(state, 2, 0)
            copied.restore(snap)
            result = compute(current, copied, buf, [19])
            forward.commit(current, copied, buf, 1, 1)
            groups.append((state, copied, result))
        target_equal(groups[0][1], groups[1][1])
        exact(groups[0][2], groups[1][2])
        current = groups[1][1]
        assert current.dspark.length == current.pos == 3
        saved = current.snapshot()
        before = current.dspark.keys[:, :3].clone()
        current.reset(w)
        current.restore(saved)
        current.ensure(40, step=8)
        assert current.capacity == current.dspark.capacity == 40
        exact(current.dspark.keys[:, :3], before)
        cloned = current.clone()
        exact(cloned.dspark.keys[:, :3], before)
        assert cloned.dspark.keys.data_ptr() != current.dspark.keys.data_ptr()
    finally:
        owner.close()


def test_native_checkfalse_numeric_failure_cannot_publish_auxiliary_prefix_then_reset_recovers():
    w = weights()
    owner = attach(w)
    try:
        st = State(w, 32, 8, "int8")
        b = Buffers(w, 8, 32)
        compute(w, st, b, [4, 7, 19], check=False)
        assert st._kv_pending and st.dspark.pending is not None and st.dspark.length == 0
        st.kv_status[0].fill_(4)
        with pytest.raises(KVNumericError):
            forward._kv_check([(st, 0, 3)])
        assert st.dspark.length == 0 and st.pos == 0
        st.reset(w)
        assert st.dspark.pending is None and st.dspark.length == 0
        compute(w, st, b, [4, 7, 19], check=False)
        forward._kv_check([(st, 0, 3)])
        forward.commit(w, st, b, 3, 2)
        assert st.pos == st.dspark.length == 2
    finally:
        owner.close()


def test_native_existing_engine_twin_keyed_sampler_and_honest_graph_fallback():
    w = weights()
    owner = attach(w)
    try:
        w.meta["capture_ready"] = True
        e = Engine(w, capacity=32, max_rows=8, prefill_rows=8, graphs=True, kv_dtype="int8")
        assert e.graphs is None
        e.forward([4, 7, 19])
        forward.commit(w, e.st, e.buf, 3, 3)
        before = e.st.dspark.keys[:, :3].clone()
        for sampling in (None, Sampling(19, temperature=0.7, top_k=20, top_p=0.9)):
            proposal = owner.draft(e, 42, 3, sampling, 0.0)
            assert len(proposal) == 3 and all(0 <= i < w.cfg.vocab for i in proposal)
        exact(e.st.dspark.keys[:, :3], before)
        twin = e.twin()
        twin.st.copy_prefix(e.st, 3, 0)
        twin.st.restore(e.st.snapshot())
        assert twin.buf is e.buf and twin.st.dspark is not e.st.dspark
        exact(twin.st.dspark.keys[:, :3], before)
    finally:
        owner.close()


def test_native_close_retains_active_tap_owner_for_retry():
    w = weights()
    owner = attach(w)
    st = State(w, 32, 8, "fp16")
    buf = Buffers(w, 8, 32)
    capture = buf.dspark_taps.begin(1, [(st, 0, 1)])
    with pytest.raises(BaseExceptionGroup):
        owner.close()
    assert not owner.closed and owner.provider is not None and not st.dspark.closed
    capture.abort()
    owner.close()
    assert owner.closed and owner.nbytes == 0
