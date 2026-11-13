"""Private fixed two-slot pinned descriptor upload ownership.

The GPU descriptor slab and its native-reader lease stay with OwnerBuffers.
Each host slot is immutable from enqueue through its own completion event;
rotating slots never permits overwriting an outstanding DMA source.
Calls inherit OwnerBuffers' serialized coordinator/stream contract. The busy
flag detects serial reentry; it is not a cross-thread synchronization primitive.
"""
import sys
from functools import partial

import torch

from tensorfold.cleanup import finish


POLICY = 'pinned-descriptor-ring2-v1'


class DescriptorUploadRing:
    policy = POLICY

    def __init__(self, host, words, event, device):
        # Journal each partial allocation before acquiring another resource.
        self.hosts, self.words, self.events = [host], [words], [event]
        self.pending, self.recorded, self.streams = [False], [False], [None]
        self.next = self.active = 0
        self.busy = False
        self.error = None
        self.device = device
        self.ready = False
        self._drains = (self._raise_error, partial(self._drain_slot, 0), partial(self._drain_slot, 1))
        self.hosts.append(torch.zeros(host.shape, dtype=host.dtype, pin_memory=event is not None))
        self.pending.append(False)
        self.recorded.append(False)
        self.streams.append(None)
        self.words.append(self.hosts[-1].numpy())
        self.events.append(torch.cuda.Event() if event is not None else None)
        if event is not None:
            stream = torch.cuda.current_stream(device)
            # Materialize both event resources in the cold preparation region.
            for item in self.events:
                item.record(stream)
            for item in self.events:
                item.synchronize()
        self.ready = True

    def acquire(self):
        if not self.ready:
            raise RuntimeError('descriptor upload ring construction did not complete')
        if self.error is not None:
            raise self.error
        if self.busy:
            raise RuntimeError('descriptor upload already owns a host writer')
        index = self.next
        if self.pending[index]:
            try:
                self.events[index].synchronize()
            except BaseException as error:
                self.error = error
                raise
            self.pending[index] = False
        self.active = index
        self.busy = True
        return self.hosts[index], self.words[index], self.events[index]

    def begin(self, stream):
        if not self.busy or (self.events[self.active] is not None and stream is None):
            raise RuntimeError('descriptor upload requires its acquired slot and CUDA stream')
        index = self.active
        # Even a failed copy/event record can have issued a DMA. Keep its
        # pinned source and stream until an independent completion fence.
        self.pending[index] = stream is not None
        self.recorded[index] = False
        self.streams[index] = stream

    def submitted(self, stream):
        if not self.busy or self.streams[self.active] is not stream:
            raise RuntimeError('descriptor completion differs from its acquired upload stream')
        index = self.active
        event = self.events[index]
        if event is not None:
            event.record(stream)
            self.recorded[index] = True
        self.next = (index + 1) % 2
        self.busy = False

    def fail(self, error):
        if self.error is None:
            self.error = error

    def _raise_error(self):
        if self.error is not None:
            raise self.error

    def _drain_slot(self, index):
        if index >= len(self.pending) or not self.pending[index]:
            return
        try:
            if self.recorded[index]:
                self.events[index].synchronize()
            else:
                self.streams[index].synchronize()
            self.pending[index] = False
        except BaseException as error:
            self.fail(error)
            raise

    def drain(self):
        # Attempt both independent slots, retaining the primary's native roots.
        finish(self._drains)
        self.busy = False

    @property
    def payload_bytes(self):
        return sum(host.nbytes for host in self.hosts)

    @property
    def metadata_bytes(self):
        return (sum(sys.getsizeof(words) + sys.getsizeof(words.base) for words in self.words)
                + sum(sys.getsizeof(value) for value in
                      (self, self.__dict__, self.hosts, self.words, self.events,
                       self.pending, self.recorded, self.streams, self._drains))
                + sum(sys.getsizeof(value) for value in (*self.events, *self._drains)))
