"""Private FP16 State batching; fixed CPU catalogue and by-value CUDA arguments.

Immutable State/Scratch pointers are validated on binding/generation changes.
Each frame refreshes only catalogue slots, spans and recurrence parity. CUDA
copies the selected pointers into its launch parameters (under Volta's4KiB
limit); no descriptor GPU allocation or upload is required. The serialized actor
retains actual States until the frame's consumer frontier, including failures.
"""
import weakref

import torch

from tensorfold.cleanup import raise_failures

MAX_STATES = 24
COLUMNS = 10
ENV = "TENSORFOLD_SM70_GDN_STATE_BATCH"


def workspace_bytes(layers):
    if type(layers) is not int or not 0 < layers <= 48:
        raise ValueError("GDN batch requires1..48 linear layers")
    payload = layers * MAX_STATES * COLUMNS * 8 + MAX_STATES * 4 * 8
    # CPU catalogue/frame, owner metadata, and bounded native validation stack.
    return 0, payload + 32768 + (128 << 10)


class GDNStateBatch:
    """One serialized target actor; cache weakly, borrow frames strongly."""
    def __init__(self, layers, device):
        workspace_bytes(layers)
        self.layers, self.device = layers, torch.device(device)
        self.host = torch.empty((layers, MAX_STATES, COLUMNS), dtype=torch.int64, device="cpu")
        self.frame = torch.empty((MAX_STATES, 4), dtype=torch.int64, device="cpu")
        self.mapping = self.frame.numpy()
        self.catalogue = [None] * MAX_STATES
        self.signatures = [None] * MAX_STATES
        self.owners = [None] * MAX_STATES
        self.frame_slots = [None] * MAX_STATES
        self.completed = torch.cuda.Event()
        self.stream = torch.cuda.current_stream(device).cuda_stream
        self.pending = self.active = self.used = False
        self.count = self.rows = self.submitted = 0
        self.error = None
        self.finish_errors = [None]  # One preallocated cold failure outcome; covered by owner metadata funding.

    def _stream(self):
        stream = torch.cuda.current_stream(self.device)
        if self.stream != stream.cuda_stream:
            raise ValueError("GDN State batch cannot cross its owning CUDA stream")
        return stream

    def retire(self):
        """After frame completion or before owner reuse, release only retired States."""
        if self.error is not None:
            raise RuntimeError("failed GDN batch owner requires model retirement") from self.error
        self._stream()
        if self.pending:
            self.completed.synchronize()
            self.pending = False
        for index in range(MAX_STATES):
            self.owners[index] = self.frame_slots[index] = None
        self.active = False

    def seal(self):
        """One event for the complete/partial frame, including failed forwards."""
        if self.used:
            try:
                self.completed.record(self._stream())
                self.pending = True
                self.used = False
            except BaseException as error:
                self.error = error
                raise

    def finish(self, primary=None):
        """Seal accepted work without replacing a target failure or its roots."""
        cause = None if primary is None else BaseException.__cause__.__get__(primary)
        context = None if primary is None else BaseException.__context__.__get__(primary)
        suppress = False if primary is None else BaseException.__suppress_context__.__get__(primary)
        try:
            self.seal()
        except BaseException as failure:
            if primary is None:
                raise
            self.finish_errors[0] = failure
            BaseException.__cause__.__set__(primary, cause)
            BaseException.__context__.__set__(primary, context)
            BaseException.__suppress_context__.__set__(primary, suppress)
            raise_failures(primary, self.finish_errors)

    def bind(self, segments):
        if self.error is not None:
            raise RuntimeError("failed GDN batch owner requires model retirement") from self.error
        self._stream()
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("GDN catalogue rebinding cannot occur inside graph capture")
        self.retire()
        count = len(segments)
        if count < 2 or count > MAX_STATES or any(not 1 <= last - first <= 3 for _, first, last in segments):
            return False  # N1/prefill/general shapes retain the original chain.
        for index, (state, first, last) in enumerate(segments):
            if state.rec.device != self.device or state.conv.device != self.device:
                raise ValueError("GDN State must belong to the batch device")
            if any(self.owners[prior] is state for prior in range(index)):
                raise ValueError("GDN batch requires independent States")
            signature = (state.slot_generation, state.history_generation, state.rec.data_ptr(),
                         state.conv.data_ptr(), id(state.scratch))
            slot = None
            for candidate in range(MAX_STATES):
                old = self.catalogue[candidate]
                if old is not None and old() is state:
                    slot = candidate
                    break
            if slot is None:
                # Never evict a slot already selected for this frame. A weak
                # catalogue owns no State/KV/tensor storage beyond frame drain.
                for candidate in range(MAX_STATES):
                    if candidate not in self.frame_slots[:index]:
                        old = self.catalogue[candidate]
                        if old is None or old() is None or all(old() is not selected[0] for selected in segments):
                            slot = candidate
                            break
            if slot is None:
                raise RuntimeError("GDN frame exceeds its24-State catalogue")
            changed = self.catalogue[slot] is None or self.catalogue[slot]() is not state or self.signatures[slot] != signature
            self.owners[index], self.frame_slots[index] = state, slot
            self.mapping[index, 0], self.mapping[index, 1] = slot, first
            self.mapping[index, 2], self.mapping[index, 3] = last - first, int(changed)
            # Publish weak cache identity only after native fill validation.
        from tensorfold.cuda.sm70.gdn_native import extension
        try:
            extension().fill_batch(self.host, self.frame, segments, self.layers)
            for index in range(count):
                state, slot = self.owners[index], self.frame_slots[index]
                if self.mapping[index, 3]:
                    self.catalogue[slot] = weakref.ref(state)
                    self.signatures[slot] = (state.slot_generation, state.history_generation,
                                             state.rec.data_ptr(), state.conv.data_ptr(), id(state.scratch))
            self.count, self.rows, self.active = count, segments[-1][2], True
            return True
        except BaseException as error:
            self.error = error
            raise

    def submit(self, layer, projection, weight, decay, bias, norm, eps, output, sums, *, negative_decay):
        if not self.active or torch.cuda.is_current_stream_capturing():
            return False  # Existing model graph paths retain their original fixed-State ABI.
        if self.error is not None:
            raise RuntimeError("failed GDN batch owner requires model retirement") from self.error
        self._stream()
        for index in range(self.count):
            state, slot = self.owners[index], self.frame_slots[index]
            signature = self.signatures[slot]
            if (state.slot_generation != signature[0] or state.history_generation != signature[1]
                    or state.rec.data_ptr() != signature[2] or state.conv.data_ptr() != signature[3]):
                raise ValueError("GDN State reset/migration/storage changed inside its prepared frame")
        from tensorfold.cuda.sm70.gdn_native import extension
        self.used = True  # Failure may follow accepted native submission; seal still fences it.
        extension().batch(projection, weight, decay, bias, norm, eps, output, sums,
                          self.host, self.frame, layer, self.count, negative_decay)
        self.submitted += 1
        return True
