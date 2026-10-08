"""Exhaustive ordered-registry storage/state tests, independent of GPU kernels.

Every retained descriptor participates in both sides. These verify bytes and
ownership, not trained quantization quality or CUDA/NCCL execution.
"""
from __future__ import annotations

import dataclasses
import itertools

import pytest

from tensorfold.families.qwen4_exp.kv_formats import FORMATS, KVPairFormat, get, get_pair
from tests.test_cuda_geometry import allocations  # noqa: F401
from tests.test_cuda_growing_caches import weights

PAIRS = tuple(KVPairFormat(key, value) for key, value in itertools.product(FORMATS, repeat=2))


def pair_id(pair):
    return f"K-{pair.key_dtype}__V-{pair.value_dtype}"


def side_bytes(fmt, width=256):
    return width * fmt.bits // 8 + (0 if fmt.bits == 16 else width // fmt.group * fmt.scale_bytes)


def tensors(cache):
    return cache.k, cache.v, cache.ks, cache.vs


def context_tensors(state):
    caches = [*state.kc, state.mtp_kc]
    for cache in caches:
        yield from tensors(cache)
    yield from state.ikc
    yield from state.pooled
    yield state.mtp_ikc
    yield state.mtp_pooled


def make_state(pair, capacity=37):
    from tensorfold.families.qwen4_exp.cuda.state import State

    model = weights()
    model.meta = {"world": 1}
    model.cfg.head_dim = 256
    return State(model, capacity, 4, kv_pair=pair, limit=128)


def test_registry_cartesian_product_has_ordered_unique_protocol_ids():
    assert len(PAIRS) == len(FORMATS) ** 2
    assert len({pair.identity for pair in PAIRS}) == len(PAIRS)
    assert len({pair.handshake for pair in PAIRS}) == len(PAIRS)
    for pair in PAIRS:
        assert pair.key is get(pair.key_dtype) and pair.value is get(pair.value_dtype)
        if pair.symmetric:
            assert pair.handshake == pair.key.handshake
        else:
            assert pair.identity != KVPairFormat(pair.value, pair.key).identity
    assert {get_pair(name).handshake for name in ("bf16", "int8", "int4")} == {16, 8, 4}


@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
def test_ordered_side_traits_and_independent_byte_formula(pair):
    expected = 2 * (side_bytes(pair.key) + side_bytes(pair.value))
    dummy = 2 * (int(pair.key.bits == 16) + int(pair.value.bits == 16))
    assert pair.row_bytes(2, 256) == expected
    assert pair.dummy_bytes == dummy
    for rows in (0, 1, 37):
        assert pair.nbytes(rows, 2, 256) == rows * expected + dummy
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.key = pair.value


def test_explicit_overrides_and_invalid_inputs_are_not_truthy_defaults():
    assert get_pair("int4", "bf16", "rotorquant8-norm") == KVPairFormat(get("bf16"), get("rotorquant8-norm"))
    assert get_pair("int8", None, "int4").key_dtype == "int8"
    assert get_pair("int8", "int4", None).value_dtype == "int8"
    for bad in ("", False, True, None, "missing"):
        with pytest.raises(ValueError):
            get_pair(bad, "bf16", "bf16")
    for bad in ("", False, True, "missing"):
        with pytest.raises(ValueError):
            get_pair("bf16", bad, None)
        with pytest.raises(ValueError):
            get_pair("bf16", None, bad)


@pytest.mark.torch
@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
def test_real_cache_has_independent_payload_metadata_and_fresh_clone_resize(pair):
    import torch
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache

    cache = KVCache(37, 2, 256, "cpu", pair=pair)
    assert cache.pair == pair and cache.key_format is pair.key and cache.value_format is pair.value
    for fmt, data, scale in ((pair.key, cache.k, cache.ks), (pair.value, cache.v, cache.vs)):
        assert data.dtype == getattr(torch, fmt.payload_dtype)
        assert scale.dtype == getattr(torch, fmt.scale_dtype)
        assert data.shape == (37, 2, 256 if fmt.bits == 16 else 256 * fmt.bits // 8)
        assert scale.shape == ((1,) if fmt.bits == 16 else (37, 2, 256 // fmt.group))
    measured = sum(tensor.untyped_storage().nbytes() for tensor in tensors(cache))
    assert measured == cache.nbytes == pair.nbytes(37, 2, 256)
    for ordinal, tensor in enumerate(tensors(cache)):
        tensor.view(torch.uint8).fill_(17 + ordinal)
    children = [cache.clone(), cache.resized(64, 31), cache.resized(17, 31)]
    for child in children:
        assert child.pair == pair
        assert child.nbytes == pair.nbytes(child.capacity, 2, 256)
        for source, copied in zip(tensors(cache), tensors(child), strict=True):
            keep = min(31, child.capacity) if child.capacity != 37 and source.shape != (1,) else source.shape[0]
            assert torch.equal(source[:keep].view(torch.uint8), copied[:keep].view(torch.uint8))
            assert source.data_ptr() != copied.data_ptr()
    for tensor in tensors(children[0]):
        tensor.zero_()
    for ordinal, tensor in enumerate(tensors(cache)):
        assert torch.all(tensor.view(torch.uint8) == 17 + ordinal)
    if not pair.symmetric:
        for legacy in ("format", "bits", "codec", "dtype"):
            with pytest.raises(ValueError, match="mixed"):
                getattr(cache, legacy)


@pytest.mark.torch
@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
def test_real_state_bytes_dummy_once_identity_status_and_growth(allocations, pair):  # noqa: F811
    import torch

    state = make_state(pair)
    caches = [*state.kc, state.mtp_kc]
    measured = sum(tensor.nbytes for tensor in context_tensors(state))
    assert state.cache_bytes() == measured
    assert state.layer_bytes() == (37 * (pair.row_bytes(2, 256) + state.index_dim * 2)
                                   + pair.dummy_bytes + ((37 + state.ratio - 1) // state.ratio) * state.index_dim * 2)
    assert state.kv_identity == (pair.identity, "stored-basis-native64-v1")
    assert state.kv_snapshot_metadata() == {"kv_identity": state.kv_identity, "kv_status": (0, -2)}
    guarded = not pair.symmetric or pair.key.codec != 0 or pair.value.codec != 0
    assert (state.kv_status is not None) == guarded
    if guarded:
        assert state.kv_status.tolist() == [0, -2]
    assert all(cache.pair == pair for cache in caches)
    state.set_pos(31)
    state.set_mtp_len(30)
    # A fixed byte pattern exercises shape/stride preservation without claiming
    # that these arbitrary payload/scale bits form valid model inputs.
    for ordinal, tensor in enumerate(context_tensors(state)):
        tensor.view(torch.uint8).fill_(ordinal + 1)
    before = state.clone()
    delta = state.ensure(40, step=64)
    assert delta == sum(tensor.nbytes for tensor in context_tensors(state)) - measured
    assert state.capacity == 64 and state.kv_identity == before.kv_identity
    for index, (current, previous) in enumerate(zip([*state.kc, state.mtp_kc], [*before.kc, before.mtp_kc], strict=True)):
        keep = 30 if index == len(state.kc) else 31
        for src, dst in zip(tensors(previous), tensors(current), strict=True):
            rows = 1 if src.shape == (1,) else keep
            assert torch.equal(src[:rows].view(torch.uint8), dst[:rows].view(torch.uint8))
    assert state.cache_bytes() == sum(tensor.nbytes for tensor in context_tensors(state))


@pytest.mark.torch
def test_constructor_rejects_conflicting_pair_selectors_and_bad_counts_before_allocation(monkeypatch):
    import torch
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache

    pair = get_pair("bf16", "int8", "int4")
    def unexpected(*args, **kwargs):
        raise AssertionError("allocation occurred before invalid configuration was rejected")
    monkeypatch.setattr(torch, "zeros", unexpected)
    invalid = [dict(pair=pair, value_dtype="bf16"), dict(pair=pair, dtype="int8"),
               dict(pair=pair, dtype="missing"), dict(pair=False), dict(key_dtype=""), dict(value_dtype=True)]
    for kwargs in invalid:
        with pytest.raises(ValueError):
            KVCache(37, 2, 256, "cpu", **kwargs)
    for capacity in (True, -1, 1.5):
        with pytest.raises(ValueError):
            KVCache(capacity, 2, 256, "cpu", pair=pair)
