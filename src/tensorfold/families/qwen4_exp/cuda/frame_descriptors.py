"""One bounded descriptor upload per original target/MTP frame.

The actor prepares only pointer/control metadata here. Projection, cache/indexer
transactions, native execution and its last-consumer fence remain at each cut.
"""
from contextlib import contextmanager

import numpy as np
import torch

from tensorfold.cleanup import raise_failures



def envelope(layers, frame_rows, owner_rows, *, bucket_count=None):
    if any(type(v) is not int or v < 1 for v in (layers, frame_rows, owner_rows)) or owner_rows > 256:
        raise ValueError("frame descriptor bounds require positive declared layer/row geometry")
    if bucket_count is None:
        from .owner_execution import row_buckets
        bucket_count = len(row_buckets(owner_rows))
    if type(bucket_count) is not int or not 1 <= bucket_count <= owner_rows:
        raise ValueError("frame descriptor bucket count differs from its bounded catalogue")
    chunks = (frame_rows + owner_rows - 1) // owner_rows
    stride = (owner_rows * 132 + 4 + 7) // 8 * 8
    batch_payload = frame_rows * 4 * 8
    payload = batch_payload + layers * chunks * stride
    if payload > 2**63 - 1:
        raise OverflowError("frame descriptor payload exceeds signed64")
    # Fixed Python/NumPy/Tensor view catalogues, frame records and bounded weak
    # reader registrations. Actual native event/allocator residency is warmed
    # and included in the later physical baseline, never credited as free.
    metadata = 16384 + layers * chunks * (8192 + bucket_count * 4096) + 8192 * frame_rows
    return {"layers": layers, "frame_rows": frame_rows, "owner_rows": owner_rows,
            "chunks": chunks, "bucket_count": bucket_count, "stride": stride, "payload_bytes": payload, "batch_descriptor_bytes": batch_payload,
            "host_metadata_bytes_bound": metadata, "host_commitment_bytes": payload + metadata}


class FrameDescriptors:
    """Serialized frame lease with fixed pinned/device packets and view owners."""
    def __init__(self, owner, layers, frame_rows, *, mtp):
        self.owner, self.mtp = owner, mtp
        self.layout = self.host = self.device = self.numpy = None
        self.host_views = self.device_views = self.uploads = self.registered = self.batch_views = None
        self.batch_host = self.batch_device = None
        self.layer_ids, self.layers, self.buffers = (), (), None
        self.states, self.records, self.segment_rows, self.source_frame = (), None, (), None
        self.active = self.pending_upload = self.pending_done = False
        self.error = self.upload = self.done = self.upload_stream = self.done_stream = None
        self.closed = self.counter_overflowed = self.prepared = False
        self.cursor = self.chunks = self.rows = 0
        self.submitted = self.copied_layers = self.borrowed_preludes = 0
        self.layout = envelope(layers, frame_rows, owner.rows, bucket_count=len(owner.native_buckets))
        self.host_views, self.device_views, self.uploads, self.registered, self.batch_views = {}, {}, {}, {}, {}
        n = self.layout["payload_bytes"]
        self.host = torch.zeros(n, dtype=torch.uint8, pin_memory=owner.queries.is_cuda)
        self.device = torch.empty(n, dtype=torch.uint8, device=owner.queries.device)
        self.numpy = self.host.numpy()
        stride, b = self.layout["stride"], owner.rows
        prefix = self.layout["batch_descriptor_bytes"]
        self.batch_host = self.numpy[:prefix].view(np.int64).reshape(frame_rows, 4)
        self.batch_device = self.device[:prefix].view(torch.int64).view(frame_rows, 4)
        for rows in range(1, frame_rows + 1):
            self.batch_views[rows] = self.batch_device[:rows]
        for chunk in range(self.layout["chunks"]):
            for ordinal in range(layers):
                at = prefix + (chunk * layers + ordinal) * stride
                raw = self.numpy[at:at + stride]
                self.host_views[ordinal, chunk] = (raw[:b * 128].view(np.int64).reshape(b, 16),
                    raw[b * 128:b * 132].view(np.int32), raw[b * 132:b * 132 + 4].view(np.int32))
                d = self.device[at:at + b * 128].view(torch.int64).view(b, 16)
                lanes = self.device[at + b * 128:at + b * 132].view(torch.int32)
                layer = self.device[at + b * 132:at + b * 132 + 4].view(torch.int32)
                for bucket in owner.native_buckets:
                    self.device_views[ordinal, chunk, bucket] = (d[:bucket], lanes[:bucket], layer)
        for chunks in range(1, self.layout["chunks"] + 1):
            end = prefix + chunks * layers * stride
            self.uploads[chunks] = (self.host[:end], self.device[:end])
        if owner.queries.is_cuda:
            self.upload, self.done = torch.cuda.Event(), torch.cuda.Event()
            stream = torch.cuda.current_stream(owner.queries.device)
            for event in (self.upload, self.done):
                event.record(stream)
                event.synchronize()  # Materialize native handles before initial admission.
            self.upload_stream = self.done_stream = stream.cuda_stream
        self.prepared = True

    @property
    def nbytes(self):
        return 0 if self.device is None else self.device.nbytes

    def bind(self, weights, buffers):
        layers = (weights.mtp.layer,) if self.mtp else tuple(layer for layer in weights.layers if not layer.linear)
        if (len(layers) != self.layout["layers"] or buffers.history_owner is not self.owner
                or buffers.rows > self.layout["frame_rows"]):
            raise ValueError("frame packet geometry differs from its actual canonical role")
        ids = tuple(layer.index for layer in layers)
        if len(set(ids)) != len(ids) or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in ids):
            raise ValueError("frame packet layer controls require unique signed32 identities")
        self.layer_ids, self.buffers = ids, buffers
        self.layers = layers

    def eligible(self, buffers, segments):
        return (self.prepared and not self.closed and buffers is self.buffers and not buffers.prefill and bool(segments)
                and segments[-1][2] <= self.layout["frame_rows"])

    def poison(self, error):
        if self.error is None:
            self.error = error

    def wait_descriptor_frame(self, state):
        if self.active and any(state is current for current in self.states):
            raise RuntimeError("State mutation cannot overtake its active descriptor frame")
        if self.error is not None:
            raise RuntimeError("descriptor frame retirement is uncertain; retire the model") from self.error
        if id(state) in self.registered and self.pending_done and self.done is not None:
            stream = torch.cuda.current_stream(self.owner.queries.device)
            if stream.cuda_stream != self.done_stream:
                stream.wait_event(self.done)

    def wait_native_reuse(self):
        if self.active:
            raise RuntimeError("State mutation cannot overtake an active descriptor frame")
        if self.error is not None:
            raise RuntimeError("descriptor frame retirement is uncertain; retire the model") from self.error
        if self.pending_done and self.done is not None:
            stream = torch.cuda.current_stream(self.owner.queries.device)
            if stream.cuda_stream != self.done_stream:
                stream.wait_event(self.done)

    def _retire_registry(self):
        if self.registered is not None:
            for state in self.registered.values():
                state.history_native_readers.discard(self)
            self.registered.clear()

    def _check_state(self, state):
        record = self.records.get(id(state))
        if record is None or record[0] is not state:
            raise RuntimeError("descriptor frame borrowed a foreign State")
        _, version, slot, generation, pos, mtp_pos = record
        if (state.version != version or state.slot_generation != slot or state.history_generation != generation
                or state.pos != pos or state.mtp_len != mtp_pos):
            raise RuntimeError("original State changed during its descriptor frame")

    @contextmanager
    def frame(self, segments, buffers):
        if self.error is not None or self.active:
            raise RuntimeError("descriptor frame owner is failed or already active") from self.error
        secondary = [None]  # Reserve outcome ownership before any asynchronous submission.
        primary = None
        try:
            rows = segments[-1][2]
            if type(rows) is not int or not 0 < len(segments) <= rows <= self.layout["frame_rows"]:
                raise ValueError("descriptor frame requires bounded nonempty original row spans")
            if not self.eligible(buffers, segments):
                raise ValueError("descriptor frame exceeds its canonical prepared role")
            states = tuple(state for state, _, _ in segments)
            if len({id(state) for state in states}) != len(states):
                raise ValueError("descriptor frame requires each original State once")
            # Bound retained weak-reader registrations independently of request churn.
            if len(set(self.registered) | {id(state) for state in states}) > 2 * self.layout["frame_rows"]:
                if self.pending_done and self.done is not None:
                    self.done.synchronize()
                self._retire_registry()
            if self.pending_upload and self.upload is not None:
                self.upload.synchronize()  # Once per frame, never once per layer/chunk.
            for state in states:
                self.owner._validate_native_state(state)
            records = {id(state): (state, state.version, state.slot_generation, state.history_generation,
                                   state.pos, state.mtp_len) for state in states}
            for state in states:
                self.registered[id(state)] = state
                state.history_native_readers.add(self)
            self.states, self.records = states, records
            self.segment_rows = tuple(segments)
            self.source_frame = buffers.mtp_indexer_frame if self.mtp else buffers.target_indexer_frame
            self.rows = rows
            self.chunks = (rows + self.owner.rows - 1) // self.owner.rows
            self.cursor = 0
            self.active = True
            self.host.zero_()
            for state, first, last in segments:
                position = state.mtp_pos if self.mtp else state.pos_dev
                self.batch_host[first:last] = (position.data_ptr(), state.kv_status.data_ptr(), first, last - first)
            for ordinal, layer in enumerate(self.layers):
                for chunk in range(self.chunks):
                    base, count = chunk * self.owner.rows, min(self.owner.rows, rows - chunk * self.owner.rows)
                    bucket = self.owner.native_bucket(count)
                    spans = self.owner._native_descriptor_spans(segments, layer, base, count,
                        mtp=self.mtp, bucket_rows=bucket, frame=self)
                    table, lanes, control = self.host_views[ordinal, chunk]
                    control[0] = layer.index
                    for lane, (state, first, last, begin, end, cache, position, index_owner) in enumerate(spans):
                        lanes[lane] = last - first
                        live = self.owner.lane_rows.data_ptr() + lane * self.owner.lane_rows.element_size()
                        words = self.owner._native_descriptor_words(state, cache, position, live, 0, index_owner)
                        table[begin - base:end - base] = words
                        for row in range(begin, end):
                            table[row - base, 14] = row - first
            host, device = self.uploads[self.chunks]
            if self.done is not None:
                stream = torch.cuda.current_stream(self.owner.queries.device)
                if self.pending_done and self.done_stream != stream.cuda_stream:
                    stream.wait_event(self.done)
                device.copy_(host, non_blocking=True)
                self.pending_upload = True
                self.upload.record(stream)
                self.upload_stream = stream.cuda_stream
            else:
                device.copy_(host)
            if self.submitted < (1 << 63) - 1:
                self.submitted += 1
            else:
                self.counter_overflowed = True
            yield
            if self.cursor != len(self.layers) * self.chunks:
                raise RuntimeError("descriptor frame did not consume its exact prepared layer/chunk sequence")
        except BaseException as error:
            primary = error
            self.poison(error)
        try:
            if self.active and self.done is not None:
                self.pending_done = True
                stream = torch.cuda.current_stream(self.owner.queries.device)
                if self.owner._upload_pending and self.owner._descriptor_stream != stream.cuda_stream:
                    stream.wait_event(self.owner.descriptor_upload)
                self.owner.wait_native_reuse()
                self.done.record(stream)
                self.done_stream = stream.cuda_stream
        except BaseException as error:
            secondary[0] = error
            self.poison(error)
        self.active = False
        if primary is not None:
            raise_failures(primary, () if secondary[0] is None else secondary)
        if secondary[0] is not None:
            raise secondary[0]
        self.states, self.records, self.segment_rows, self.source_frame = (), None, (), None

    def borrow_prelude(self, buffers, segments, total_rows, *, mtp):
        """Borrow immutable position/status/span rows until the complete frame retires."""
        if not self.active or self.error is not None or buffers is not self.buffers or mtp is not self.mtp:
            raise RuntimeError("batch prelude lacks its canonical active descriptor frame")
        current = buffers.mtp_indexer_frame if mtp else buffers.target_indexer_frame
        if current is not self.source_frame or total_rows != self.rows or len(segments) != len(self.segment_rows):
            raise RuntimeError("batch prelude changed its issued frame or bounded row layout")
        for (state, first, last), (prior, a0, a1) in zip(segments, self.segment_rows):
            if state is not prior or first != a0 or last != a1:
                raise RuntimeError("batch prelude changed its original State row spans")
            self._check_state(state)
        if self.upload is not None:
            stream = torch.cuda.current_stream(self.owner.queries.device)
            if stream.cuda_stream != self.upload_stream:
                stream.wait_event(self.upload)
        if self.borrowed_preludes < (1 << 63) - 1:
            self.borrowed_preludes += 1
        else:
            self.counter_overflowed = True
        return self.batch_views[total_rows]

    def stage(self, segments, layer, base, count, *, mtp, bucket):
        if not self.active or self.error is not None or mtp is not self.mtp:
            raise RuntimeError("canonical cut lacks its active descriptor frame")
        ordinal, chunk = divmod(self.cursor, self.chunks)
        if (ordinal >= len(self.layer_ids) or layer.index != self.layer_ids[ordinal]
                or base != chunk * self.owner.rows or count != min(self.owner.rows, self.rows - base)
                or bucket != self.owner.native_bucket(count)):
            raise ValueError("descriptor cut changed its prepared layer/chunk geometry")
        current = self.buffers.mtp_indexer_frame if self.mtp else self.buffers.target_indexer_frame
        if current is not self.source_frame:
            raise RuntimeError("original coordinate frame changed during descriptor replay")
        if len(segments) != len(self.segment_rows):
            raise RuntimeError("descriptor frame changed its original row spans")
        for (state, first, last), (prior, a0, a1) in zip(segments, self.segment_rows):
            if state is not prior or first != a0 or last != a1:
                raise RuntimeError("descriptor frame changed its original row spans")
            self._check_state(state)
        table, lanes, control = self.device_views[ordinal, chunk, bucket]
        if self.upload is not None:
            stream = torch.cuda.current_stream(self.owner.queries.device)
            if stream.cuda_stream != self.upload_stream:
                stream.wait_event(self.upload)
        self.owner.wait_native_reuse()
        if self.owner.descriptor_upload is not None:
            stream = torch.cuda.current_stream(self.owner.queries.device)
            if self.owner._upload_pending and self.owner._descriptor_stream != stream.cuda_stream:
                stream.wait_event(self.owner.descriptor_upload)
        self.owner.descriptors[:bucket].copy_(table)
        self.owner.lane_rows[:bucket].copy_(lanes)
        self.owner.native_layer.copy_(control)
        if self.owner.descriptor_upload is not None:
            stream = torch.cuda.current_stream(self.owner.queries.device)
            self.owner.descriptor_upload.record(stream)
            self.owner._descriptor_stream = stream.cuda_stream
            self.owner._upload_pending = True
        self.cursor += 1
        if self.copied_layers < (1 << 63) - 1:
            self.copied_layers += 1
        else:
            self.counter_overflowed = True

    def retire(self, *, fenced):
        if self.active:
            raise RuntimeError("active descriptor frame must retire its CPU producer before storage")
        if not fenced and self.done is not None and (self.error is not None or self.pending_done):
            raise RuntimeError("descriptor frame retains uncertain consumers until a global fence") from self.error
        self.active = self.pending_done = self.pending_upload = False
        self._retire_registry()
        self.states, self.records, self.segment_rows, self.source_frame = (), None, (), None

    def close(self, *, fenced):
        self.retire(fenced=fenced)
        self.closed, self.prepared = True, False
        for views in (self.host_views, self.device_views, self.uploads, self.batch_views):
            if views is not None:
                views.clear()
        self.host = self.device = self.numpy = self.batch_host = self.batch_device = self.upload = self.done = None
        self.buffers, self.layers, self.layer_ids = None, (), ()

    def record(self):
        return {**(self.layout or {}), "submitted_frames": None if self.counter_overflowed else self.submitted,
                "copied_layer_chunks": None if self.counter_overflowed else self.copied_layers,
                "borrowed_preludes": None if self.counter_overflowed else self.borrowed_preludes,
                "prepared": self.prepared, "active": self.active, "poisoned": self.error is not None}
