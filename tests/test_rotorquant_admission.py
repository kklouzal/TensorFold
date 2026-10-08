"""Independent metadata, real CPU cache lifetime and admission boundary checks."""

import dataclasses
import importlib
import itertools
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tensorfold import cli, serve_options
from tensorfold.families import qwen3_5, qwen4_exp
from tensorfold.families.qwen4_exp import kv_formats
from tests.test_cuda_capacity import HEAD, Loaded, checkpoint, fake_runtime, small_config  # noqa: F401
from tests.test_cuda_geometry import allocations  # noqa: F401
from tests.test_cuda_growing_caches import weights

ROTOR = tuple(item for item in kv_formats.FORMATS if item.rotor is not None)
NAMES = tuple(item.name for item in ROTOR)
INCOMPATIBLE = tuple((name, peer) for name, peer in itertools.combinations(("int4", *NAMES), 2)
                     if kv_formats.get(name).bits == kv_formats.get(peer).bits)


def _state(dtype, capacity=37, limit=128, *, world=1):
    module = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    model = weights()
    model.meta = {"world": world}
    model.cfg.head_dim = 256
    return module.State(model, capacity, 4, dtype, limit=limit)


def _caches(state):
    return [*state.kc, state.mtp_kc]


def _context_tensors(state):
    for cache in _caches(state):
        yield from (cache.k, cache.v, cache.ks, cache.vs)
    yield from state.ikc
    yield from state.pooled
    yield from (state.mtp_ikc, state.mtp_pooled)


def _fill_bytes(tensors, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    for tensor in tensors:
        tensor.view(torch.uint8).random_(0, 256, generator=generator)


def _real_context_bytes(state):
    return sum(tensor.nbytes for tensor in _context_tensors(state))


def test_configuration_lists_share_one_accelerator_free_format_authority():
    assert qwen4_exp.CUDA_KV_DTYPES is kv_formats.DTYPES
    parser = cli.build_parser()
    for dtype in kv_formats.DTYPES:
        args = parser.parse_args(["serve", "model", "--kv-dtype", dtype])
        assert args.kv_dtype == dtype
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    code = (
        "import sys; from tensorfold import cli; "
        "from tensorfold.families import qwen4_exp; "
        "from tensorfold.families.qwen4_exp import kv_formats; "
        "assert qwen4_exp.CUDA_KV_DTYPES is kv_formats.DTYPES; "
        "cli.build_parser(); assert 'torch' not in sys.modules; "
        "assert 'triton' not in sys.modules; assert 'scipy' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert result.returncode == 0, result.stderr


def test_descriptors_are_frozen_and_same_bits_do_not_mean_same_protocol():
    assert ROTOR, "a RotorQuant format must be selected before release"
    identifiers = set()
    for item in ROTOR:
        assert item.bits in (3, 4, 6, 7, 8) and item.group == 128 and item.scale_bytes == 4
        assert item.rotor.bits == item.bits and item.rotor.group == item.group
        assert item.rotor.scale_dtype == "float32"
        from tensorfold.families.qwen4_exp.cuda.rotorquant_ref import variant_id
        assert item.codec == variant_id(item.rotor.variant)
        assert item.handshake != kv_formats.get("int4").handshake
        assert item.handshake not in identifiers
        assert 0 <= item.handshake < 2**63
        identifiers.add(item.handshake)
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.codec = 0
        changed_table = dataclasses.replace(item.rotor, codec_id=item.rotor.codec_id + "-different-table")
        changed_format = dataclasses.replace(item, rotor=changed_table)
        assert changed_format.bits == item.bits and changed_format.name == item.name
        assert changed_format.handshake != item.handshake
    assert {kv_formats.get(name).handshake for name in ("bf16", "int8", "int4")} == {16, 8, 4}


@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
def test_only_flash_next_cuda_accepts_the_new_cache_at_cli_validation(fmt):
    args = cli.build_parser().parse_args(["serve", "model", "--kv-dtype", fmt.name])
    flash = SimpleNamespace(title=qwen4_exp.TITLE, package=qwen4_exp, model_type="qwen4_exp")
    other = SimpleNamespace(title=qwen3_5.TITLE, package=qwen3_5, model_type="qwen3_5")
    assert serve_options.check(args, flash, "cuda") is None
    with pytest.raises(ValueError, match="CUDA engine option"):
        serve_options.check(args, flash, "mlx")
    with pytest.raises(ValueError, match="KV cache"):
        serve_options.check(args, other, "cuda")


@pytest.mark.torch
@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
@pytest.mark.parametrize("capacity", [1, 37, 65])
def test_exact_cache_storage_and_group_metadata_match_the_shared_geometry(fmt, capacity):
    import torch
    from tensorfold.cuda.geometry import kv_bytes
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache

    cache = KVCache(capacity, 2, 256, "cpu", fmt.name)
    assert cache.format is fmt and cache.codec == fmt.codec and cache.bits == fmt.bits
    assert cache.k.dtype == cache.v.dtype == torch.uint8
    packed_width = 256 * fmt.bits // 8
    assert cache.k.shape == cache.v.shape == (capacity, 2, packed_width)
    assert cache.ks.dtype == cache.vs.dtype == torch.float32
    assert cache.ks.shape == cache.vs.shape == (capacity, 2, 2)
    # Independently count actual tensor storage, then compare the startup layout.
    measured = sum(tensor.nbytes for tensor in (cache.k, cache.v, cache.ks, cache.vs))
    assert measured == cache.nbytes == capacity * 2 * 2 * (packed_width + 2 * 4)
    assert measured == capacity * 2 * 2 * kv_bytes(256, fmt.bits, group=fmt.group, scale_bytes=fmt.scale_bytes)


@pytest.mark.torch
def test_cache_and_family_configuration_names_stay_identical():
    from tensorfold.families.qwen4_exp.cuda import kvcache

    assert kvcache.DTYPES is qwen4_exp.CUDA_KV_DTYPES
    assert kvcache.BITS_OF is kv_formats.BITS_OF
    assert tuple(kvcache.BITS_OF) == qwen4_exp.CUDA_KV_DTYPES


@pytest.mark.torch
@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
def test_real_state_growth_and_clone_preserve_all_committed_bytes_and_format(allocations, fmt):  # noqa: F811
    import torch

    state = _state(fmt.name)
    state.set_pos(31)
    state.set_mtp_len(33)
    _fill_bytes(_context_tensors(state), 41)
    previous = state.clone()
    before = _real_context_bytes(state)
    assert state.cache_bytes() == before
    assert state.ensure(37) == 0
    assert state.ensure(40, step=64) == _real_context_bytes(state) - before
    assert state.capacity == 64 and state.version == 1
    assert state.cache_bytes() == _real_context_bytes(state)
    for index, (now, old) in enumerate(zip(_caches(state), _caches(previous))):
        keep = 33 if index == len(state.kc) else 31
        assert now.format is old.format is fmt
        for key in ("k", "v", "ks", "vs"):
            actual, expected = getattr(now, key), getattr(old, key)
            assert actual.data_ptr() != expected.data_ptr()
            assert torch.equal(actual[:keep].view(torch.uint8), expected[:keep].view(torch.uint8))
    source_before_clone_writes = [tensor.clone() for tensor in _context_tensors(state)]
    cloned = state.clone()
    assert cloned.cache_bytes() == _real_context_bytes(cloned) == state.cache_bytes()
    for actual, original in zip(_context_tensors(cloned), _context_tensors(state)):
        assert actual.data_ptr() != original.data_ptr()
        assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
        actual.zero_()
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
               for a, b in zip(_context_tensors(state), source_before_clone_writes))
    with pytest.raises(ValueError, match="window"):
        state.ensure(129)
    assert state.capacity == 64


@pytest.mark.torch
@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
def test_real_prefix_copy_keeps_payload_scale_and_complete_index_pools_only(allocations, fmt):  # noqa: F811
    import torch

    source, destination = _state(fmt.name, 64), _state(fmt.name, 37)
    source.set_pos(40)
    source.set_mtp_len(39)
    _fill_bytes(_context_tensors(source), 67)
    saved = [tensor.clone() for tensor in _context_tensors(source)]
    for tensor in _context_tensors(destination):
        tensor.view(torch.uint8).fill_(165)
    destination.copy_prefix(source, 31, 30)
    for index, (actual, expected) in enumerate(zip(_caches(destination), _caches(source))):
        rows = 30 if index == len(source.kc) else 31
        assert actual.format is expected.format is fmt
        for key in ("k", "v", "ks", "vs"):
            a, b = getattr(actual, key), getattr(expected, key)
            assert torch.equal(a[:rows].view(torch.uint8), b[:rows].view(torch.uint8))
            assert torch.all(a[rows:].view(torch.uint8) == 165)
    for actual, expected in zip(destination.pooled, source.pooled):
        assert torch.equal(actual[:7].view(torch.uint8), expected[:7].view(torch.uint8))
        assert torch.all(actual[7:].view(torch.uint8) == 165)
    for actual, expected in zip(destination.ikc, source.ikc):
        assert torch.equal(actual[:31].view(torch.uint8), expected[:31].view(torch.uint8))
        assert torch.all(actual[31:].view(torch.uint8) == 165)
    assert torch.equal(destination.mtp_ikc[:30].view(torch.uint8), source.mtp_ikc[:30].view(torch.uint8))
    assert torch.all(destination.mtp_ikc[30:].view(torch.uint8) == 165)
    assert torch.equal(destination.mtp_pooled[:7].view(torch.uint8), source.mtp_pooled[:7].view(torch.uint8))
    assert torch.all(destination.mtp_pooled[7:].view(torch.uint8) == 165)
    assert destination.pos == destination.mtp_len == 0
    assert all(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
               for actual, expected in zip(_context_tensors(source), saved))


@pytest.mark.torch
@pytest.mark.parametrize("dtype,peer", INCOMPATIBLE)
def test_same_bit_width_prefix_copy_rejects_different_codec_or_rotation(allocations, dtype, peer):  # noqa: F811
    assert kv_formats.get(dtype).bits == kv_formats.get(peer).bits
    source, destination = _state(dtype), _state(peer)
    source.set_pos(20)
    source.set_mtp_len(19)
    before = [tensor.clone() for tensor in _context_tensors(destination)]
    with pytest.raises(ValueError, match="matching cache formats"):
        destination.copy_prefix(source, 17, 16)
    import torch
    assert all(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
               for actual, expected in zip(_context_tensors(destination), before))


@pytest.mark.torch
@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
@pytest.mark.parametrize("streams", [1, 4])
def test_geometry_format_difference_equals_measured_live_cache_difference(
        allocations, monkeypatch, fmt, streams):  # noqa: F811
    import torch
    from tensorfold.cuda import geometry

    text = dict(small_config(), head_dim=256)
    count = 2 if streams == 1 else 1
    windows = [513] if streams == 1 else [513, 256, 256, 256]
    rotor_states = [_state(fmt.name, rows, rows) for rows in windows]
    # The declared legacy comparison is INT4; six-bit candidates differ in
    # payload size as well as norm metadata and numeric status ownership.
    legacy_states = [_state("int4", rows, rows) for rows in windows]
    rotor_bytes = sum(_real_context_bytes(st) for st in rotor_states)
    legacy_bytes = sum(_real_context_bytes(st) for st in legacy_states)
    params = dict(mtp=True, kv_pair=((fmt.bits, fmt.group, fmt.scale_bytes),) * 2, kv_status=True)
    assert all(st.kv_status is None and st._kv_peer_status is None for st in legacy_states)
    status_bytes = count * sum(st.kv_status.nbytes for st in rotor_states)
    if streams == 1:
        candidate = geometry.gdn_geometry(text, 1, 4, indexed=True, **params)
        legacy = geometry.gdn_geometry(text, 1, 4, indexed=True, mtp=True, kv_bits=4)
    else:
        candidate = geometry.indexed_stream_geometry(text, 4, 4, 8, **params)
        legacy = geometry.indexed_stream_geometry(text, 4, 4, 8, mtp=True, kv_bits=4)
        # Exercise actual main/MTP Step serialization. Only the transport is
        # replaced with a CPU copy; the pointer schema/allocation remains native.
        module = importlib.import_module("tensorfold.families.qwen4_exp.cuda.attn_multi")
        monkeypatch.setattr(module.shared, "to_device", lambda values, dtype, device:
                            torch.tensor(values, dtype=dtype, device=device))
        model = weights()
        model.meta = {"world": 1}
        model.mtp.layer = SimpleNamespace(index=len(model.layers), linear=False)
        for states in (rotor_states, legacy_states):
            segments = [(state, i, i + 1) for i, state in enumerate(states)]
            for mtp in (False, True):
                step = module.Step(model, segments, mtp)
                if states is rotor_states:
                    assert step.status_ptrs.dtype == torch.int64 and step.status_ptrs.shape == (streams,)
                    assert step.status_ptrs.tolist() == [st.kv_status.data_ptr() for st in states]
                    status_bytes += step.status_ptrs.nbytes
                else:
                    assert step.status_ptrs is None
    # Actual payload/scales plus the fixed two-int32 status wire and measured
    # concurrent pointer tables account for the full admission delta.
    assert candidate.bytes_at(513) - legacy.bytes_at(513) == count * (rotor_bytes - legacy_bytes) + status_bytes


@pytest.mark.torch
@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
@pytest.mark.parametrize("head_dim", [64, 192, 384])
def test_unsupported_head_dimension_is_refused_before_any_weights(tmp_path, fake_runtime, fmt, head_dim):  # noqa: F811
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    checkpoint(tmp_path, dict(small_config(), head_dim=head_dim), HEAD)
    calls, _ = fake_runtime
    with pytest.raises(ValueError, match="multiple of 128|power.of.two"):
        FlashNextEngine(tmp_path, depth=3, kv_dtype=fmt.name)
    assert not calls


@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
def test_production_format_requires_power_of_two_heads_without_restricting_legacy_formats(fmt):
    for width in (128, 256, 512):
        assert fmt.value_bytes(width) == width * fmt.bits // 8 + width // 128 * 4
    for width in (64, 192, 384):
        with pytest.raises(ValueError, match="multiple of 128|power.of.two"):
            fmt.value_bytes(width)
    for name in ("bf16", "int8", "int4"):
        assert kv_formats.get(name).value_bytes(384) > 0


@pytest.mark.torch
@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
@pytest.mark.parametrize("streams", [1, 4])
def test_actual_engine_uses_format_layout_for_window_admission(tmp_path, monkeypatch, fake_runtime, fmt, streams):  # noqa: F811
    from tensorfold.cuda import geometry
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine, KEEP, KEEP_SERIAL

    text = dict(small_config(), head_dim=256)
    checkpoint(tmp_path, text, HEAD)
    calls, capacity = fake_runtime
    params = dict(mtp=True, kv_pair=((fmt.bits, fmt.group, fmt.scale_bytes),) * 2, kv_status=True)
    plan = (geometry.gdn_geometry(text, 1, 4, indexed=True, kept=KEEP_SERIAL + 1, **params) if streams == 1 else
            geometry.indexed_stream_geometry(text, streams, 4, KEEP, **params))
    budget = plan.needed(12000) + 32768
    monkeypatch.setattr(capacity, "available_bytes", lambda _: budget)
    engine = FlashNextEngine.__new__(FlashNextEngine)
    with pytest.raises(Loaded):
        engine.__init__(tmp_path, depth=3, kv_dtype=fmt.name, streams=streams)
    assert len(calls) == 1
    receipt = engine.capacity_plan
    assert 12000 <= receipt["context_window"] < 65536
    assert receipt["total_bytes_estimate"] <= budget
    calls.clear()
    with pytest.raises(ValueError, match="largest fitting"):
        FlashNextEngine(tmp_path, depth=3, kv_dtype=fmt.name, streams=streams, max_len=65536, context_explicit=True)
    assert not calls


@pytest.mark.torch
@pytest.mark.parametrize("fmt", ROTOR, ids=lambda item: item.name)
@pytest.mark.parametrize("world", [1, 2])
def test_numeric_status_wire_and_tp_peer_rows_match_fixed_admission(allocations, fmt, world):  # noqa: F811
    import torch
    from tensorfold.cuda import geometry

    state = _state(fmt.name, world=world)
    assert state.kv_status.dtype == torch.int32 and state.kv_status.shape == (2,)
    assert state.kv_status.tolist() == [0, -2] and state.kv_status.nbytes == 8
    if world == 1:
        assert state._kv_peer_status is None
        measured = state.kv_status.nbytes
    else:
        assert state._kv_peer_status.dtype == torch.int32 and state._kv_peer_status.shape == (world, 2)
        measured = state.kv_status.nbytes + state._kv_peer_status.nbytes
        assert state._kv_peer_status.nbytes == world * 8
    text = dict(small_config(), head_dim=256)
    params = dict(indexed=True, mtp=True, kv_pair=((fmt.bits, fmt.group, fmt.scale_bytes),) * 2)
    with_status = geometry.gdn_geometry(text, world, 4, kv_status=True, **params)
    without_status = geometry.gdn_geometry(text, world, 4, kv_status=False, **params)
    # The serial execution region admits one current state and its independent
    # reference twin; snapshots retain the validated status identity as values.
    for capacity in (1, 513, 16384):
        assert with_status.bytes_at(capacity) - without_status.bytes_at(capacity) == 2 * measured


@pytest.mark.torch
@pytest.mark.parametrize("dtype,peer", INCOMPATIBLE)
def test_two_ranks_reject_same_bits_with_different_codec_identity(fake_runtime, dtype, peer):  # noqa: F811
    import torch
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    class Exchange:
        def __init__(self, other=None):
            self.other, self.sent = other, None

        def all_gather(self, send, recv):
            self.sent = send.clone()
            recv.copy_(torch.cat((send, send if self.other is None else self.other)))

    def rank(cache, exchange):
        engine = FlashNextEngine.__new__(FlashNextEngine)
        engine.rope = RopeParameters.from_config(small_config())
        engine.depth, engine.confidence, engine.max_len = 3, 0.5, 8192
        engine.comm = exchange
        engine.kv_pair = kv_formats.get_pair(cache)
        return engine

    first = Exchange()
    rank(peer, first)._same_settings(torch, None)
    second = Exchange(first.sent)
    with pytest.raises(RuntimeError, match="different settings"):
        rank(dtype, second)._same_settings(torch, None)
    assert second.sent[5] == kv_formats.get(dtype).handshake
    assert first.sent[5] == kv_formats.get(peer).handshake
    assert second.sent[5] != first.sent[5]
