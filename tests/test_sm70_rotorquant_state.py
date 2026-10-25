"""Exact registered SM70 Q4 state, ownership and admission boundaries, without CUDA."""

from itertools import product
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.geometry import indexed_stream_geometry
from tensorfold.families.qwen4_exp.kv_formats import FORMATS as REGISTRY, get_pair
from tensorfold.families.qwen4_exp.cuda.sm70_kv import (
    FORMATS,
    RAW_POLICY,
    ROTOR_POLICY,
    ROTOR_Q4,
    ROTOR_Q4_NORM,
    COMPACT_Q4,
    check_kernel_fields,
    check_pair,
)
from tests.test_cuda_geometry import allocations  # noqa: F401
from tests.test_cuda_capacity import fake_runtime, small_config  # noqa: F401
from tests.test_cuda_growing_caches import weights

PAIRS = tuple(product(FORMATS, repeat=2))


def model(world=1, *, mtp=True, dtype="float16"):
    w = weights()
    w.meta = {"world": world, "execution_dtype": dtype}
    w.cfg.kv_heads, w.cfg.head_dim = 1, 256
    return w if mtp else NS(**{**vars(w), "mtp": None})


def caches(state):
    return [*state.kc, *([state.mtp_kc] if hasattr(state, "mtp_kc") else [])]


def tensors(state):
    return [t for cache in caches(state) for t in (cache.k, cache.v, cache.ks, cache.vs)]


@pytest.mark.parametrize("key,value", PAIRS)
def test_exact_sm70_registry_and_operator_plan_agree(key, value):
    pair = get_pair("bf16", key, value)
    policy = check_pair(pair, head_dim=256)
    assert policy == (ROTOR_POLICY if pair.key.codec or pair.value.codec else RAW_POLICY)
    fields = [(0 if side.bits == 16 else side.bits, side.codec) for side in (pair.key, pair.value)]
    assert check_kernel_fields(*fields[0], *fields[1]) == (fields[0][0], fields[1][0])


@pytest.mark.parametrize("side", ["key", "value"])
@pytest.mark.parametrize("fmt", [f.name for f in REGISTRY if f.name not in FORMATS])
def test_other_rotor_variants_and_equal_bit_codecs_remain_refused(side, fmt):
    pair = get_pair("int8", fmt if side == "key" else None, fmt if side == "value" else None)
    with pytest.raises(ValueError, match="SM70"):
        check_pair(pair)


@pytest.mark.parametrize(
    "fields",
    [
        (4, 0, 8, 0),
        (6, 4, 8, 0),
        (8, 4, 4, 4),
        (4, 6, 4, 4),
        (0, 4, 0, 0),
        (4, 4, 4, 3),
        (True, 0, 4, 4),
        (4.0, 4, 4, 4),
        (4, None, 4, 4),
        (4, 4, "4", 4),
    ],
)
def test_native_flags_cannot_widen_registered_codec_support(fields):
    with pytest.raises(ValueError, match="SM70"):
        check_kernel_fields(*fields)


@pytest.mark.parametrize("head_dim", [0, True, 64, 192, 384])
@pytest.mark.parametrize("format_name", [ROTOR_Q4, ROTOR_Q4_NORM, COMPACT_Q4])
def test_q4_geometry_fails_before_allocation(head_dim, format_name):
    with pytest.raises(ValueError, match="head dim|head_dim"):
        check_pair(get_pair(format_name), head_dim=head_dim)


@pytest.mark.parametrize("key,value", PAIRS)
def test_engine_accepts_only_the_exact_candidate_before_model_or_cuda(monkeypatch, tmp_path, key, value):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    class NextStage(Exception):
        pass

    def next_stage(*args, **kwargs):
        raise NextStage

    monkeypatch.setattr(gguf, "resolve_gguf", lambda path: tmp_path / "model.gguf")
    monkeypatch.setattr(chat_template, "prepare_chat_template", next_stage)
    with pytest.raises(NextStage):
        FlashNextEngine(tmp_path, kv_key_dtype=key, kv_value_dtype=value)


@pytest.mark.usefixtures("allocations")
@pytest.mark.parametrize("key,value", PAIRS)
@pytest.mark.parametrize("mtp", [False, True])
def test_half_state_uses_real_packed_and_scale_bytes_through_clone_resize_copy(key, value, mtp):
    import torch
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tensorfold.families.qwen4_exp.cuda.dspark_cuda import _Stamp

    pair = get_pair("bf16", key, value)
    w = model(8, mtp=mtp)
    source = State(w, 37, 4, kv_pair=pair, limit=128)
    expected_row = sum({"fp16": 512, "int8": 272, ROTOR_Q4: 136, ROTOR_Q4_NORM: 136,
                        COMPACT_Q4: 128 + 2 * (2 + 2)}[side] for side in (key, value))
    layers = 2 + int(mtp)
    expected = layers * (37 * (expected_row + 256) + 10 * 256 + pair.dummy_bytes)
    assert source.cache_bytes() == source.owned_cache_bytes() == expected
    assert source.kv_identity == (pair.identity, check_pair(pair))
    assert _Stamp.read(source).codec == source.kv_identity
    for cache in caches(source):
        for data, scale, fmt in ((cache.k, cache.ks, pair.key), (cache.v, cache.vs, pair.value)):
            assert data.dtype == getattr(torch, fmt.payload_dtype)
            assert scale.dtype == getattr(torch, fmt.scale_dtype)
            assert scale.shape == ((1,) if fmt.bits == 16 else (37, 1, 256 // fmt.group))
            data.fill_(17)
            scale.fill_(0.5)
    source.set_pos(9)
    source.set_mtp_len(9 if mtp else 0)
    cloned = source.clone()
    for before, after in zip(tensors(source), tensors(cloned), strict=True):
        assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
        assert before.data_ptr() != after.data_ptr()
    destination = State(w, 37, 4, kv_pair=pair, limit=128)
    destination.copy_prefix(source, 9, 9 if mtp else 0)
    for before, after in zip(tensors(source), tensors(destination), strict=True):
        if before.shape == (1,):
            # Raw dummy scales are not cache data: the existing prefix path
            # leaves the destination's unused dummy unchanged.
            assert torch.all(before == 0.5) and torch.all(after == 0)
        else:
            assert torch.equal(before[:9].view(torch.uint8), after[:9].view(torch.uint8))
    snapshot = source.snapshot()
    assert snapshot["kv_identity"] == source.kv_identity and snapshot["kv_status"] == (0, -2)
    original = [t.clone() for t in tensors(source)]
    source.resize(73)
    assert source.capacity == 73 and source.kv_identity == snapshot["kv_identity"]
    for before, after in zip(original, tensors(source), strict=True):
        stop = 1 if before.shape == (1,) else 9
        assert torch.equal(before[:stop].view(torch.uint8), after[:stop].view(torch.uint8))
    source.restore(snapshot)
    assert (source.pos, source.mtp_len) == (9, 9 if mtp else 0)


@pytest.mark.usefixtures("allocations")
@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("key,value", PAIRS)
def test_geometry_status_words_match_actual_state_and_peer_packet(world, key, value):
    from tensorfold.families.qwen4_exp.cuda.state import State

    state = State(model(world), 37, 4, kv_pair=get_pair("bf16", key, value))
    actual = sum(t.nbytes for t in (state.kv_status, state._kv_peer_status) if t is not None)
    text = dict(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=world,
        head_dim=256,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=1024,
        hc_count=4,
    )
    kwargs = dict(streams=3, each=4, keep=8, mtp=True, world=world, prefill_rows=64)
    base = indexed_stream_geometry(text, **kwargs)
    checked = indexed_stream_geometry(text, **kwargs, kv_status=state.kv_status is not None)
    assert checked.bytes_at(8196) - base.bytes_at(8196) == 3 * actual


@pytest.mark.parametrize("slots", [2, 3, 6, 8])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("key,value", PAIRS)
def test_full_window_forecast_counts_q4_packing_fp32_scales_and_replace_overlap(slots, mtp, key, value):
    text = dict(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=256,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=1024,
        hc_count=4,
    )
    pair = get_pair("bf16", key, value)
    specs = tuple((side.bits, side.group, side.scale_bytes) for side in (pair.key, pair.value))
    kwargs = dict(
        streams=slots, each=4, keep=8, mtp=mtp, world=8, prefill_rows=64, full_windows=slots, workspace_bytes=2 << 30
    )
    base = indexed_stream_geometry(text, **kwargs, kv_pair=((8, 32, 2),) * 2)
    candidate = indexed_stream_geometry(text, **kwargs, kv_pair=specs)
    row = sum({"fp16": 512, "int8": 272, ROTOR_Q4: 128 + 2 * 4, ROTOR_Q4_NORM: 128 + 2 * 4,
               COMPACT_Q4: 128 + 2 * (2 + 2)}[name] for name in (key, value))
    # Every full slot owns all target/MTP caches. A replacement retains one
    # additional layer; indexer/pool, graph/status and workspace bytes cancel.
    copies = slots * (2 + int(mtp)) + 1
    for capacity in (37, 8196, 262148):
        assert base.bytes_at(capacity) - candidate.bytes_at(capacity) == copies * (
            capacity * (544 - row) - pair.dummy_bytes
        )


@pytest.mark.usefixtures("allocations")
def test_rotor_policy_numeric_failure_is_sticky_and_foreign_snapshot_cannot_mutate():
    import torch
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError, State

    w = model()
    state = State(w, 37, 4, kv_pair=get_pair(ROTOR_Q4))
    foreign = State(model(), 37, 4, kv_pair=get_pair(ROTOR_Q4_NORM))
    assert foreign.owned_cache_bytes() == state.owned_cache_bytes()
    assert foreign.kv_working_policy == state.kv_working_policy == ROTOR_POLICY
    assert foreign.kv_identity != state.kv_identity
    snapshot = foreign.snapshot()
    snapshot["rec"].fill_(17)
    before = state.rec.clone()
    with pytest.raises(ValueError, match="matching ordered formats"):
        state.restore(snapshot)
    assert torch.equal(state.rec, before)
    state.kv_begin()
    state.kv_status.copy_(torch.tensor([2, -1], dtype=torch.int32))
    with pytest.raises(KVNumericError, match="MTP"):
        state.kv_check()
    with pytest.raises(KVNumericError):
        state.snapshot()
    state.reset(w)
    assert state.kv_identity == (get_pair(ROTOR_Q4).identity, ROTOR_POLICY)
    assert state.kv_snapshot_metadata()["kv_status"] == (0, -2)


def test_tp_handshake_rejects_ordered_format_and_working_policy_disagreement(fake_runtime):  # noqa: F811
    import torch
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    class Comm:
        def __init__(self, peer=None):
            self.peer, self.sent = peer, None

        def all_gather(self, send, out):
            self.sent = send.clone()
            out.copy_(torch.cat([send, send if self.peer is None else self.peer]))

    def rank(pair, comm, policy=ROTOR_POLICY):
        engine = FlashNextEngine.__new__(FlashNextEngine)
        engine.rope = RopeParameters.from_config(small_config())
        engine.depth, engine.confidence, engine.max_len, engine.comm = 3, 0.0, 8192, comm
        engine.kv_pair, engine.prefill_rows = pair, 2048
        engine.streams, engine.graphs_enabled, engine.tp = 3, True, 2
        engine._sm70_kv_policy = policy
        return engine

    key_rotor = get_pair("int8", ROTOR_Q4, None)
    peer = Comm()
    rank(key_rotor, peer)._same_settings(torch, None)
    rank(key_rotor, Comm(peer.sent))._same_settings(torch, None)
    for pair, policy in (
        (get_pair("int8", None, ROTOR_Q4), ROTOR_POLICY),
        (get_pair("int8", ROTOR_Q4_NORM, None), ROTOR_POLICY),
        (key_rotor, RAW_POLICY),
    ):
        with pytest.raises(RuntimeError, match="different settings"):
            rank(pair, Comm(peer.sent), policy)._same_settings(torch, None)
