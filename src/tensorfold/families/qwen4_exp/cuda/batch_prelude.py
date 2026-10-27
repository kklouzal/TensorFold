"""Prepared descriptor owner for exact multi-State native frame preparation."""
import torch


def workspace_bytes(rows):
    if type(rows) is not int or not 0 < rows <= 4096:
        raise ValueError("batch prelude rows must fit the fixed native envelope")
    payload = rows * 4 * 8
    return ((payload + 511) // 512) * 512, payload + 16384


class BatchPrelude:
    """One serialized actor owns host upload storage through event retirement."""
    __slots__ = ("rows", "descriptor", "host", "view", "upload", "pending", "error")

    def __init__(self, rows, device):
        workspace_bytes(rows)
        self.rows, self.pending, self.error = rows, False, None
        self.descriptor = torch.empty((rows, 4), dtype=torch.int64, device=device)
        self.host = torch.empty((rows, 4), dtype=torch.int64, pin_memory=True)
        self.view = self.host.numpy()
        self.upload = torch.cuda.Event()

    def submit(self, weights, buffers, layer, segments, total_rows, *, mtp):
        from .history_owner import _native_prelude_position, _publish_native_frame
        from .compact_indexer import _scalar_text_frame
        from tensorfold.cuda.sm70.family_native import extension

        if self.error is not None:
            raise RuntimeError("failed batch prelude owner requires model retirement") from self.error
        if not 0 < total_rows <= self.rows or total_rows > buffers.rows:
            raise ValueError("batch prelude row span differs from its admitted actor")
        c, binding = weights.cfg, weights.positions
        frame = buffers.mtp_indexer_frame if mtp else buffers.target_indexer_frame
        position = buffers.mtp_indexer_position if mtp else buffers.target_indexer_position
        if (binding is None or position is not (binding.mtp if mtp else binding.target)
                or not _native_prelude_position(position) or not _scalar_text_frame(frame)
                or (c.heads, c.head_dim, c.index_heads, c.index_dim) != (3, 256, 4, 128)
                or not buffers.pa.is_contiguous() or not buffers.gated.is_contiguous()):
            return False
        cursor = 0
        for state, first, last in segments:
            if first != cursor or not first < last <= total_rows:
                raise ValueError("batch prelude requires complete ordered State row segments")
            cursor = last
            if (state.image_positions is not None or state.rope_delta != 0
                    or state.media_profile_identity is not None or state.media_input_span is not None):
                return False
            ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
            cache = state.mtp_kc if mtp else state.kc[ai]
            if (cache.key_format.bits, cache.value_format.bits, cache.key_format.codec, cache.value_format.codec) != (8, 8, 0, 0):
                return False
            pos = state.mtp_len if mtp else state.pos
            if not 0 <= pos < 262144 or last - first > 262144 - pos:
                return False
        if cursor != total_rows:
            raise ValueError("batch prelude segments do not cover the producing frame")
        try:
            if self.pending:
                self.upload.synchronize()  # Host buffer remains borrowed until DMA completion.
            for state, first, last in segments:
                pointer = state.mtp_pos if mtp else state.pos_dev
                self.view[first:last] = (pointer.data_ptr(), state.kv_status.data_ptr(), first, last - first)
            self.descriptor[:total_rows].copy_(self.host[:total_rows], non_blocking=True)
            self.upload.record(torch.cuda.current_stream(weights.device))
            self.pending = True
            storage = buffers.gated.view(torch.uint8).view(-1)
            capacity, dim = buffers.rows, 256
            payload, scale, raw = capacity * dim, capacity * 16, capacity * 256
            key = storage[:payload].view(torch.int8).view(capacity, 1, dim)
            value = storage[payload:2 * payload].view(torch.int8).view(capacity, 1, dim)
            ks = storage[2 * payload:2 * payload + scale].view(torch.float16).view(capacity, 1, 8)
            vs = storage[2 * payload + scale:2 * payload + 2 * scale].view(torch.float16).view(capacity, 1, 8)
            ik = storage[2 * payload + 2 * scale:2 * payload + 2 * scale + raw].view(torch.float16).view(capacity, 128)
            extension().owner_prep_batch(buffers.pa[:total_rows], self.descriptor[:total_rows],
                layer.attn.q_scale, layer.attn.k_scale, layer.attn.iq_scale, position.inv_freq,
                buffers.q[:total_rows], key[:total_rows], value[:total_rows], buffers.iq[:total_rows],
                ik[:total_rows], ks[:total_rows], vs[:total_rows], layer.index,
                c.eps, position.amplitude, not buffers.prefill)
            for state, first, last in segments:
                _publish_native_frame(state, layer, weights, buffers, first, last, key, value, ks, vs, ik,
                                      position.inv_freq, position.sections, position.amplitude,
                                      mtp=mtp, frame_first=first)
            weights.meta["sm70_owner_prelude_submitted_batches"] += 1
            weights.meta["sm70_owner_prelude_max_batch_segments"] = max(
                weights.meta["sm70_owner_prelude_max_batch_segments"], len(segments))
        except BaseException as error:
            self.error = error  # Unknown submitted readers retain their owner until model fence.
            raise
        return True
