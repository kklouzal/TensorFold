"""Private original rowwise candidate capture component, without collectives.

Preparation requires process-wide capture quiescence and externally admitted
storage. This component is not enrolled in serving. Its graph pool allocation
and retained metadata must be funded by an integrating owner before admission.
"""
from __future__ import annotations

import gc

import torch

from tensorfold.cleanup import raise_failures

from .capture_lifetime import on_stream
from .state import CAND


def _binding(tensor):
    return tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


class CandidateRegion:
    """One exact role/row count over original FP32 head and packet storage."""

    def __init__(self, weights, buffers, logits, rows, *, id_map=None, offset=0, side=None, pool=None):
        self.weights, self.buffers, self.logits = weights, buffers, logits
        self.rows, self.id_map, self.offset = rows, id_map, offset
        self.side, self.pool, self.graph = side, pool, None
        self.failure, self.replay_stream = None, None
        self.prepared = self.closed = self.retiring = False
        self.hits = 0
        if (type(rows) is not int or not 1 <= rows <= 32 or type(offset) is not int
                or weights.meta.get('execution_provider') != 'sm70'):
            raise ValueError('candidate region requires canonical SM70 rows in1..32 and integer offset')
        if (logits.ndim != 2 or logits.shape[0] != rows or logits.shape[1] < 1
                or logits.dtype != torch.float32 or not logits.is_cuda or not logits.is_contiguous()
                or logits.requires_grad or buffers.prefill):
            raise ValueError('candidate region requires exact contiguous FP32 CUDA decode logits')
        if (buffers.cand.device != logits.device or buffers.cand.dtype != torch.float32
                or buffers.cand.shape[0] < rows or buffers.cand.shape[1] != 2 * CAND + 1
                or not buffers.cand.is_contiguous() or buffers.cand.requires_grad):
            raise ValueError('candidate region requires the original complete local packet storage')
        if logits.data_ptr() != buffers.logits.data_ptr() or logits.numel() > buffers.logits.numel():
            raise ValueError('candidate region must borrow the original head output prefix')
        if id_map is not None and (id_map.device != logits.device or id_map.ndim != 1
                or id_map.numel() != logits.shape[1] or id_map.dtype not in (torch.int32, torch.int64)
                or not id_map.is_contiguous() or id_map.requires_grad):
            raise ValueError('candidate region needs its immutable original vocabulary map')
        self.packet = buffers.cand
        self.input_owner = buffers.logits
        self.bindings = tuple(_binding(value) for value in (logits, self.packet))
        self.map_binding = None if id_map is None else _binding(id_map)

    def _body(self):
        from .forward import candidates_local

        return candidates_local(self.weights, self.buffers, self.logits, self.rows,
                                id_map=self.id_map, offset=self.offset)

    def _capture(self):
        secondary = [None]
        self.graph.capture_begin(pool=self.pool, capture_error_mode='thread_local')
        primary = None
        try:
            self._body()
        except BaseException as error:
            primary = error
        try:
            self.graph.capture_end()
        except BaseException as error:
            secondary[0] = error
        if secondary[0] is not None:
            raise_failures(primary, secondary)
        if primary is not None:
            raise primary

    @torch.no_grad()
    def prepare(self):
        if self.prepared or self.graph is not None or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError('candidate region cannot be prepared twice or after failure')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('candidate preparation requires process-wide capture quiescence')
        current = torch.cuda.current_stream(self.logits.device)
        self.replay_stream = current.cuda_stream
        try:
            if self.side is None:
                self.side = torch.cuda.Stream(device=self.logits.device)
            self.side.wait_stream(current)
            on_stream(torch.cuda, self.side, self._body)
            self.side.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            # Destroying another graph during Python collection can invalidate
            # active capture. Match the existing graph owners' cold guard.
            enabled = gc.isenabled()
            gc.disable()
            try:
                on_stream(torch.cuda, self.side, self._capture)
            finally:
                if enabled:
                    gc.enable()
            current.wait_stream(self.side)
            self.prepared = True
        except BaseException as error:
            self.failure = error
            raise

    def replay(self):
        if not self.prepared or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError('candidate region is not prepared and admitted') from self.failure
        try:
            if self.buffers.cand is not self.packet or self.buffers.logits is not self.input_owner:
                raise RuntimeError('candidate region lost its original buffer owners')
            if tuple(_binding(value) for value in (self.logits, self.packet)) != self.bindings:
                raise RuntimeError('candidate region lost its original storage layout')
            if self.id_map is not None and _binding(self.id_map) != self.map_binding:
                raise RuntimeError('candidate region lost its immutable vocabulary map')
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('candidate replay cannot nest inside another capture')
            current = torch.cuda.current_stream(self.logits.device)
            if current.cuda_stream != self.replay_stream:
                raise RuntimeError('candidate replay requires its original serialized CUDA stream')
            self.graph.replay()
            self.hits += 1
        except BaseException as error:
            self.failure = error
            raise

    def close(self):
        if self.closed:
            return
        self.retiring = True
        try:
            torch.cuda.synchronize(self.logits.device)
            if self.graph is not None:
                self.graph.reset()
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise
        self.graph = self.side = self.pool = None
        self.weights = self.buffers = self.logits = self.id_map = None
        self.packet = self.input_owner = None
        self.closed = True
