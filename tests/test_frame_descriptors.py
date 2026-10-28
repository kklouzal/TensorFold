"""Original descriptor-word oracle and bounded whole-frame ownership controls."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.frame_descriptors import FrameDescriptors, envelope
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_native_owner_adapter import _weights
from tests.frame_descriptor_controls import original_words


def prepared(count, rows_per_state=3, owner_rows=8, *, mtp=False):
    weights = _weights()
    if mtp:
        weights.mtp.layer = NS(index=12, linear=False)
    states = [State(weights, 128, 4, "int8", slot=index) for index in range(count)]
    owner = OwnerBuffers(weights, owner_rows, NS(idw=2052))
    owner.prepare_native(states)
    rows = count * rows_per_state
    buffers = NS(rows=rows, prefill=False, history_owner=owner, target_indexer_frame=None, mtp_indexer_frame=None)
    packet = FrameDescriptors(owner, 1 if mtp else 12, rows, mtp=mtp)
    owner.descriptor_frames = packet
    packet.bind(weights, buffers)
    segments = [(state, index * rows_per_state, (index + 1) * rows_per_state) for index, state in enumerate(states)]
    return weights, states, owner, packet, buffers, segments




@pytest.mark.parametrize("count,owner_rows", [(1, 4), (8, 8), (24, 32)])
@pytest.mark.parametrize("mtp", [False, True])
def test_one_packet_matches_original_descriptor_bytes_for_all_layers_states_and_chunks(count, owner_rows, mtp):
    weights, states, owner, packet, buffers, segments = prepared(count, owner_rows=owner_rows, mtp=mtp)
    layers = (weights.mtp.layer,) if mtp else weights.layers
    pointers = owner.descriptors.data_ptr(), owner.lane_rows.data_ptr(), owner.native_layer.data_ptr()
    for repeat in range(2):
        with packet.frame(segments, buffers):
            for layer in layers:
                for base in range(0, buffers.rows, owner.rows):
                    count_rows = min(owner.rows, buffers.rows - base)
                    bucket = owner.native_bucket(count_rows)
                    expected, lanes = original_words(owner, segments, layer, base, count_rows, bucket, mtp=mtp)
                    owner.stage_native_descriptors(segments, layer, base, count_rows, bucket_rows=bucket, mtp=mtp)
                    assert torch.equal(owner.descriptors[:bucket], expected)
                    assert torch.equal(owner.lane_rows[:bucket], lanes)
                    assert owner.native_layer.tolist() == [layer.index]
                    assert pointers == (owner.descriptors.data_ptr(), owner.lane_rows.data_ptr(), owner.native_layer.data_ptr())
        assert not packet.active and packet.submitted == repeat + 1
    assert packet.copied_layers == 2 * len(layers) * packet.chunks
    assert len(packet.registered) <= 2 * packet.layout["frame_rows"]
    owner.close()
    assert packet.closed and packet.device is packet.host is None
    assert all(packet not in state.history_native_readers for state in states)


def test_active_frame_blocks_reset_position_and_coordinate_restage_before_mutation():
    weights, states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    state = states[0]
    from tensorfold.families.qwen4_exp.cuda import compact_indexer
    with packet.frame(segments, buffers):
        before = state.slot_generation, state.pos, state.kv_status.clone()
        for operation in (lambda: state.reset(weights), lambda: state.reset(weights, new_sequence=False),
                          lambda: state.set_pos(1), lambda: state.set_rope_delta(1),
                          lambda: compact_indexer.stage_target_indexer_frame(buffers, segments, None)):
            with pytest.raises(RuntimeError, match="descriptor frame"):
                operation()
        assert (state.slot_generation, state.pos) == before[:2]
        assert torch.equal(state.kv_status, before[2])
        for layer in weights.layers:
            owner.stage_native_descriptors(segments, layer, 0, 3, bucket_rows=4)
    owner.close()


def test_original_storage_mismatch_fails_before_packet_or_device_table_publication():
    _weights0, states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    owner.descriptors.fill_(99)
    cache = next(cache for cache in states[0].kc if cache.k.numel())
    cache.k = cache.k.clone()
    with pytest.raises(RuntimeError, match="storage catalogue"):
        with packet.frame(segments, buffers):
            pytest.fail("invalid original storage acquired frame")
    assert bool((owner.descriptors == 99).all()) and packet.error is not None
    with pytest.raises(RuntimeError, match="failed"):
        with packet.frame(segments, buffers):
            pytest.fail("poisoned owner reused")
    owner.close()


def test_incomplete_frame_preserves_primary_and_poison_until_retirement():
    _weights0, _states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    primary = RuntimeError("operation")
    with pytest.raises(RuntimeError) as caught:
        with packet.frame(segments, buffers):
            raise primary
    assert caught.value is primary and packet.error is primary and not packet.active
    with pytest.raises(RuntimeError, match="failed"):
        with packet.frame(segments, buffers):
            pass
    owner.close()
    assert packet.error is primary


def test_metadata_envelope_scales_with_actual_accepted_bucket_catalogue():
    small = envelope(12, 72, 32, bucket_count=6)
    many = envelope(12, 72, 32, bucket_count=32)
    assert small["chunks"] == many["chunks"] == 3
    assert small["payload_bytes"] == many["payload_bytes"] == 72 * 32 + 12 * 3 * 4232
    assert many["host_metadata_bytes_bound"] > small["host_metadata_bytes_bound"]
    assert many["host_commitment_bytes"] == many["payload_bytes"] + many["host_metadata_bytes_bound"]


class Event:
    def __init__(self, *, wait_error=None, record_error=None):
        self.wait_error, self.record_error = wait_error, record_error
        self.waits = self.records = 0
    def synchronize(self):
        self.waits += 1
        if self.wait_error is not None:
            raise self.wait_error
    def record(self, stream):
        self.records += 1
        if self.record_error is not None:
            raise self.record_error


def test_failed_upload_wait_preserves_packet_and_blocks_all_future_submissions():
    _w, _states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    error = RuntimeError("host DMA uncertain")
    packet.upload, packet.pending_upload = Event(wait_error=error), True
    packet.device.fill_(17)
    before = packet.device.clone()
    with pytest.raises(RuntimeError) as caught:
        with packet.frame(segments, buffers):
            pytest.fail("uncertain DMA reused host bytes")
    assert caught.value is error and packet.error is error and torch.equal(packet.device, before)
    owner.close()  # CPU oracle has no asynchronous CUDA consumer.


def test_operation_and_completion_failure_preserve_primary_and_require_positive_fence(monkeypatch):
    _w, states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    primary, completion = RuntimeError("operation"), RuntimeError("completion unknown")
    packet.upload, packet.done = Event(), Event(record_error=completion)
    stream = NS(cuda_stream=7, wait_event=lambda event: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    with pytest.raises(RuntimeError) as caught:
        with packet.frame(segments, buffers):
            raise primary
    assert caught.value is primary and primary.__cause__ is completion and packet.error is primary
    held = packet.device
    with pytest.raises(RuntimeError, match="global fence"):
        packet.retire(fenced=False)
    assert packet.device is held and packet in states[0].history_native_readers
    packet.retire(fenced=True)
    assert packet.error is primary and packet not in states[0].history_native_readers
    packet.done = None
    owner.close()


def test_changed_frame_or_rows_refuses_before_canonical_descriptor_overwrite():
    _w, _states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    with pytest.raises(RuntimeError, match="coordinate frame changed"):
        with packet.frame(segments, buffers):
            owner.descriptors.fill_(91)
            buffers.target_indexer_frame = object()
            owner.stage_native_descriptors(segments, packet.layers[0], 0, 3, bucket_rows=4)
    assert bool((owner.descriptors == 91).all())
    owner.close()


def test_actual_preparation_funds_complete_global_packets_before_any_role_allocation(monkeypatch):
    from tensorfold.cuda import capacity
    from tensorfold.cuda.sm70 import foundation_headers
    from tensorfold.families.qwen4_exp.cuda import execution_identity
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    from tensorfold.families.qwen4_exp.cuda.owner_execution import KERNEL_FAMILY, UNCAPTURED
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    events = []
    def fund(total, **kwargs):
        events.append(("fund", total))
        return {"total_commitment_bytes": total}
    monkeypatch.setattr(capacity, "admit_host_commitment", fund)
    monkeypatch.setattr(capacity, "cuda_limit_bytes", lambda: None)
    monkeypatch.setattr(foundation_headers, "selected_identity", lambda: {"test": "prepared-source"})
    monkeypatch.setattr(execution_identity, "_retire_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (1 << 30, 1 << 30))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args: 0)
    monkeypatch.setattr(OwnerPreparation, "_phase", lambda self, engine, name, operation, **kwargs: [operation()] * 8)
    original = OwnerBuffers.from_text
    def allocate(*args, **kwargs):
        assert events and events[0][0] == "fund"
        events.append(("allocate", kwargs["rows"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(OwnerBuffers, "from_text", allocate)
    monkeypatch.setattr(OwnerBuffers, "native_run", lambda self, rows: self.native_views(rows)["out"])
    def gather(source, target):
        target.copy_(source.clone().expand_as(target))
    comm = NS(world=8, rank=0, all_gather=gather)
    engine = NS(comm=comm, tp=8, rank=0, multi=None, _calls=0)
    text = {"num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
            "hidden_size": 2560, "layer_types": ["full_attention", "full_attention"],
            "indexer_budget": 8, "indexer_compress_ratio": 4}
    spec = {"kernel_family": KERNEL_FAMILY, "graph_family": UNCAPTURED, "graph_budget_bytes": 0, "trial": True}
    prepared_owner = OwnerPreparation(spec, role_rows={"decode": 4, "mtp": 4, "prefill": 8},
                                      manifest_sha256="a" * 64, frame_rows=8)
    prepared_owner.prepare(engine, text, get_pair("int8"), page_tokens=128, device="cpu")
    expected = 8 * sum(envelope(layers, 8, 4, bucket_count=3)["host_commitment_bytes"] for layers in (2, 1))
    assert events[0] == ("fund", expected)
    assert [event[0] for event in events] == ["fund", "allocate", "allocate", "allocate"]
    assert prepared_owner.workspaces["decode"].descriptor_frames.layout["payload_bytes"] == 2400
    assert prepared_owner.workspaces["mtp"].descriptor_frames.layout["payload_bytes"] == 1328
    assert prepared_owner.workspaces["prefill"].descriptor_frames is None
    prepared_owner.close()


@pytest.mark.parametrize("count", [1, 8, 24])
@pytest.mark.parametrize("mtp", [False, True])
def test_once_frame_borrowed_prelude_rows_match_original_layer_invariant_table(count, mtp):
    weights, states, owner, packet, buffers, segments = prepared(count, owner_rows=8, mtp=mtp)
    layers = (weights.mtp.layer,) if mtp else weights.layers
    with packet.frame(segments, buffers):
        first_pointer = None
        for layer in layers:
            actual = packet.borrow_prelude(buffers, segments, buffers.rows, mtp=mtp)
            if first_pointer is None:
                first_pointer = actual.data_ptr()
            assert actual.data_ptr() == first_pointer
            expected = torch.empty((buffers.rows, 4), dtype=torch.int64)
            for state, first, last in segments:
                expected[first:last] = torch.tensor(((state.mtp_pos if mtp else state.pos_dev).data_ptr(),
                    state.kv_status.data_ptr(), first, last - first), dtype=torch.int64)
            assert torch.equal(actual, expected)
            for base in range(0, buffers.rows, owner.rows):
                kept = min(owner.rows, buffers.rows - base)
                owner.stage_native_descriptors(segments, layer, base, kept, mtp=mtp, bucket_rows=owner.native_bucket(kept))
    assert packet.borrowed_preludes == len(layers) and packet.submitted == 1
    with pytest.raises(RuntimeError, match="active descriptor frame"):
        packet.borrow_prelude(buffers, segments, buffers.rows, mtp=mtp)
    owner.close()


@pytest.mark.parametrize("count", [8, 24])
@pytest.mark.parametrize("mtp", [False, True])
def test_actual_batch_submit_borrows_packet_without_touching_legacy_dma_owner(monkeypatch, count, mtp):
    from tensorfold.families.qwen4_exp.cuda.batch_prelude import BatchPrelude
    from tensorfold.families.qwen4_exp.cuda import history_owner, compact_indexer
    from tensorfold.cuda.sm70 import family_native
    weights, states, owner, packet, buffers, segments = prepared(count, owner_rows=8, mtp=mtp)
    position = NS(inv_freq=torch.ones(1), sections=(1, 1, 1), amplitude=1.0)
    weights.positions = NS(target=position, mtp=position)
    weights.cfg.eps = 1e-6
    weights.meta.update(sm70_owner_prelude_submitted_batches=0, sm70_owner_prelude_submitted_mtp_batches=0,
                        sm70_owner_prelude_batch_counter_overflowed=False, sm70_owner_prelude_max_batch_segments=0)
    buffers.target_indexer_position = buffers.mtp_indexer_position = position
    buffers.pa = torch.zeros((buffers.rows, 2688), dtype=torch.float16)
    buffers.gated = torch.zeros((buffers.rows, 768), dtype=torch.float16)
    buffers.q = torch.empty((buffers.rows, 3, 256), dtype=torch.float16)
    buffers.iq = torch.empty((buffers.rows, 4, 128), dtype=torch.float16)
    monkeypatch.setattr(history_owner, "_native_prelude_position", lambda value: value is position)
    monkeypatch.setattr(compact_indexer, "_scalar_text_frame", lambda frame: True)
    published = []
    monkeypatch.setattr(history_owner, "_publish_native_frame", lambda state, layer, w, b, first, last, *args, **kw:
                        published.append((state, first, last)))
    controls = []
    def native(projection, descriptor, *args):
        assert descriptor is packet.batch_views[buffers.rows]
        controls.append(descriptor.clone())
    monkeypatch.setattr(family_native, "extension", lambda: NS(owner_prep_batch=native))
    batch = object.__new__(BatchPrelude)
    batch.rows, batch.pending, batch.error = buffers.rows, True, None
    batch.descriptor = torch.full((buffers.rows, 4), 91, dtype=torch.int64)
    batch.host = torch.full_like(batch.descriptor, 92)
    batch.view = batch.host.numpy()
    class ForbiddenDMA:
        def synchronize(self):
            pytest.fail("borrowed packet waited on legacy per-layer DMA")
        def record(self, stream):
            pytest.fail("borrowed packet recorded legacy per-layer upload")
    batch.upload = ForbiddenDMA()
    with packet.frame(segments, buffers):
        for layer in packet.layers:
            layer.attn = NS(q_scale=torch.ones(256), k_scale=torch.ones(256), iq_scale=torch.ones(128))
            assert batch.submit(weights, buffers, layer, segments, buffers.rows, mtp=mtp)
            for base in range(0, buffers.rows, owner.rows):
                kept = min(owner.rows, buffers.rows - base)
                owner.stage_native_descriptors(segments, layer, base, kept, mtp=mtp, bucket_rows=owner.native_bucket(kept))
    assert len(controls) == len(packet.layers) and len(published) == count * len(packet.layers)
    assert packet.borrowed_preludes == len(packet.layers) and packet.submitted == 1
    assert bool((batch.descriptor == 91).all()) and bool((batch.host == 92).all()) and batch.pending
    assert all(torch.equal(value, controls[0]) for value in controls)
    owner.close()


@pytest.mark.parametrize("slots,refused", [(23, True), (24, False)])
def test_actual_engine_final_admission_receives_future_frame_host_obligation_before_floor_publication(slots, refused):
    import ast
    import inspect
    import textwrap
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.prefix_policy import OWNER_FRAME_DESCRIPTORS_PROFILE
    method = ast.parse(textwrap.dedent(inspect.getsource(FlashNextEngine._initialize))).body[0]
    future = next(node for node in method.body if isinstance(node, ast.If)
        and any(isinstance(child, ast.Attribute) and child.attr == "frame_future_host_bytes" for child in ast.walk(node)))
    admitted = next(node for node in method.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "admit")
    floor = next(node for node in method.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        and node.test.id == "adaptive_sessions" and any(isinstance(child, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "streams" for target in child.targets) for child in node.body))
    publication = next(at for at, node in enumerate(floor.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "resident_windows" for target in node.targets))
    assert future.lineno < admitted.lineno < floor.lineno
    calls = []
    def actual_admit(*args, **kwargs):
        calls.append(kwargs)
        return {"session_capacity": {"resident_capacity": slots}}
    engine = NS(sm70_profile=OWNER_FRAME_DESCRIPTORS_PROFILE, streams="prior", kv_full_windows="prior")
    owner = NS(frame_rows=32, frame_future_host_bytes=123456, committed_bytes=lambda: 77)
    space = dict(self=engine, model_dir="fixture", max_len=262147, context_explicit=True,
        torch=NS(), geometry=object(), vision=False, media_requested=False, rank=0, workspace=0,
        weight_rule=object(), tp=8, sm70=True, adaptive_sessions=True, gather=None, exl3=False,
        automatic_capacity={}, numeric_capacity={}, rope_capacity={}, host_capacity={"host_workspace_bytes": 19},
        adaptive_capacity={}, nvfp4_source=False, nvfp4_preallocated=0, owner_preparation=owner,
        capacity_geometry=lambda *args: "geometry", admit=actual_admit, resident_windows="prior")
    code = compile(ast.Module(body=[future, admitted, *floor.body[:publication + 2]], type_ignores=[]),
                   inspect.getfile(FlashNextEngine), "exec")
    if refused:
        with pytest.raises(ValueError, match="at least24"):
            exec(code, space)
        assert engine.streams == engine.kv_full_windows == space["resident_windows"] == "prior"
    else:
        exec(code, space)
        assert engine.streams == engine.kv_full_windows == space["resident_windows"] == 24
    assert len(calls) == 1 and calls[0]["host_workspace_bytes"] == 19 + 123456
    assert calls[0]["preallocated_device_bytes"] == 77
