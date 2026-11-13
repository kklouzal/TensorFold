"""Private paid host packet owner; blocking readback, serial bounded borrowing.

Only independent Python tokens/probabilities may leave a borrow. The caller
releases it before any full-source sampler fallback. Graphs retain their
original device packets and collectives; this owner is never captured.
"""
import sys
import threading
import weakref
from contextlib import contextmanager

import numpy as np
import torch

from .row_sampling import _signature

POLICY = 'owned-gathered-readback-v1'


def layout(world, rows):
    from .state import CAND
    if type(world) is not int or world != 8 or type(rows) is not int or not 0 < rows <= 256:
        raise ValueError('gathered readback requires TP8 and bounded 1..256 rows')
    packet = world * rows * (2 * CAND + 1) * 4
    ids = world * rows * CAND * 8
    return {'pinned_host_bytes': packet, 'pageable_host_bytes': ids, 'host_payload_bytes': packet + ids}


class GatheredReadback:
    def __init__(self, world, rows, role, device):
        self.closed = self.poisoned = False
        self.bound = False
        self.lock = threading.Lock()
        self.host = self.ids = self.words = None
        self.weights = self.buffer = self.packet = None
        self.world, self.rows, self.role = world, rows, role
        self.device = torch.device(device)
        self.cost = layout(world, rows)
        if role not in ('decode', 'mtp'):
            raise ValueError('gathered readback is only selected for target/MTP Buffer roles')
        self.host = torch.empty(self.cost['pinned_host_bytes'] // 4, dtype=torch.float32,
                                pin_memory=self.device.type == 'cuda')
        self.words = self.host.numpy()
        from .state import CAND
        self.ids = np.empty(world * rows * CAND, dtype=np.int64)
        self.ids_pointer = self.ids.ctypes.data
        self.original_words, self.original_ids = self.words, self.ids
        self.host_signature = _signature(self.host)

    def bind(self, weights, buffers):
        if self.closed or self.bound:
            raise RuntimeError('gathered readback binds once before State/capture')
        if (weights.meta.get('execution_provider') != 'sm70' or weights.meta.get('world') != self.world
                or weights.comm is None
                or weights.comm.world != self.world or buffers.world != self.world
                or self.world != 8 or buffers.rows != self.rows or buffers.owner_role != self.role
                or buffers.execution_dtype != torch.float16 or buffers.cand_all.device != self.device
                or buffers.cand_all.dtype != torch.float32 or not buffers.cand_all.is_contiguous()
                or buffers.cand_all.numel() != self.host.numel()):
            raise ValueError('gathered readback differs from its actual TP8 Half packet/Buffer role')
        self.weights, self.buffer, self.packet = weights, weakref.ref(buffers), buffers.cand_all
        self.comm = weights.comm
        self.packet_signature = _signature(self.packet)
        self.bound = True

    def validate_bindings(self, weights, packet):
        from .state import CAND
        buffers = None if self.buffer is None else self.buffer()
        if (self.closed or self.poisoned or not self.bound or self.weights is not weights
                or buffers is None or buffers.cand_all is not packet or packet is not self.packet
                or buffers.owner_role != self.role or buffers.rows != self.rows or buffers.world != self.world
                or buffers.execution_dtype != torch.float16
                or weights.meta.get('execution_provider') != 'sm70'
                or weights.meta.get('world') != self.world
                or weights.comm is not self.comm or weights.comm.world != self.world):
            raise RuntimeError('gathered readback requires its original live weights/Buffer/packet owner')
        if (self.words is not self.original_words or self.ids is not self.original_ids
                or self.ids.dtype != np.int64 or self.ids.shape != (self.world * self.rows * CAND,)
                or not self.ids.flags.c_contiguous or not self.ids.flags.writeable
                or self.ids.ctypes.data != self.ids_pointer
                or self.words.dtype != np.float32 or self.words.shape != (self.host.numel(),)
                or not self.words.flags.c_contiguous or self.words.ctypes.data != self.host.data_ptr()
                or packet.requires_grad or self.host.requires_grad or packet.is_neg() or packet.is_conj()
                or _signature(packet) != self.packet_signature or _signature(self.host) != self.host_signature):
            raise ValueError('gathered readback tensor/storage changed after cold binding')

    @contextmanager
    def borrow(self, weights, packet, rows):
        from .state import CAND
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('gathered readback already has an active borrower')
        try:
            self.validate_bindings(weights, packet)
            if self.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('gathered host readback cannot run inside capture')
            if type(rows) is not int or not 0 < rows <= self.rows:
                raise ValueError('gathered readback exceeds its paid row extent')
            count = self.world * rows * (2 * CAND + 1)
            try:
                # Preserve .cpu()'s blocking current-stream host-read contract.
                self.host[:count].copy_(packet[:count], non_blocking=False)
            except BaseException:
                self.poisoned = True
                raise
            g = self.words[:count].reshape(self.world, rows, 2 * CAND + 1)
            ids = self.ids[:self.world * rows * CAND].reshape(self.world, rows, CAND)
            np.copyto(ids, g[:, :, CAND:2 * CAND].view(np.int32))
            yield g[:, :, :CAND], ids, g[:, :, 2 * CAND]
        finally:
            self.lock.release()

    def record(self):
        metadata = (sys.getsizeof(self) + sys.getsizeof(vars(self)) + sys.getsizeof(self.lock)
                    + sys.getsizeof(self.host) + sys.getsizeof(self.words)
                    + sys.getsizeof(self.ids) - (0 if self.ids is None else self.ids.nbytes))
        return {'policy': POLICY, 'role': self.role, 'world': self.world, 'rows': self.rows,
                'host_metadata_bytes': metadata,
                **self.cost, 'bound': self.bound, 'closed': self.closed, 'poisoned': self.poisoned,
                'qualification': 'private-unqualified; native/model/P3 pending',
                'uncontrolled': 'NumPy per-row sampling temporaries, Python metadata and Torch internal copy machinery'}

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('cannot close gathered readback while borrowed')
        try:
            # Successful copies are blocking. A failed copy poisons this owner;
            # retain its pinned storage unless device retirement is proved.
            if self.poisoned and self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.host = self.words = self.ids = self.weights = self.buffer = self.packet = None
            self.comm = None
            self.original_words = self.original_ids = None
            self.closed = True
        finally:
            self.lock.release()
