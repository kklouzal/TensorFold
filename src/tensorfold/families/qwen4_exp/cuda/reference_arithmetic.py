"""Private pre-admission exact-shape ATen islands, never whole-forward capture."""
import hashlib
from pathlib import Path
import time
import threading

import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sm70.family_ops import _matmul32

from .owner_preparation import ArithmeticRequest, ARITHMETIC_POLICY as POLICY


def statistics_step(query, key, mask, maximum, denominator, scale):
    score = _matmul32(query, key.T) * scale
    score.masked_fill_(~mask[None], -torch.inf)
    tile_max = score.amax(-1)
    next_max = torch.maximum(maximum, tile_max)
    safe_max = torch.where(torch.isfinite(next_max), next_max, 0)
    alpha = torch.where(torch.isfinite(maximum), torch.exp(maximum - safe_max), 0)
    probabilities = torch.where(mask[None], torch.exp(score - safe_max[:, None]), 0)
    return next_max, denominator * alpha + probabilities.sum(-1)


def probability_step(query, key, mask, maximum, denominator, scale):
    score = _matmul32(query, key.T) * scale
    score.masked_fill_(~mask[None], -torch.inf)
    return torch.where(mask[None], torch.exp(score - maximum[:, None]) / denominator[:, None], 0).half()


def value_step(probability, value, result):
    return result + _matmul32(probability, value)


class ReferenceArithmetic:
    """One serialized logical query; three graphs/role and three separate pools.

    Native keys/values and their generation/status checks stay OUTSIDE capture.
    K and V intentionally borrow the same Half slab, so probability must replay
    before values(), and PV only afterward. Returned recurrence tensors are
    copied into caller-paid destinations under one nonblocking operation lease;
    no mutable arithmetic recurrence/output view escapes that handoff.
    """
    def __init__(self, tiles, *, budget_bytes):
        self._operation_lease = threading.Lock()
        self.started = self.closed = False
        self.tiles = self.device = None
        self.graphs, self.pools, self.outputs, self.inputs = {}, {}, {}, {}
        self.side = self.prepared_stream = None
        self.prepared = False
        self.failure = None
        self.before = self.after = None
        self.charged_bytes = self.capture_s = 0
        self.private_spare_bytes = None
        self.tensor_leases = []
        self.replays = {'stats': 0, 'probability': 0, 'pv': 0}
        self.retained_tensor_bytes = 0
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.budget = ArithmeticRequest(budget_bytes).graph_budget_bytes
        self.tiles, self.device = tiles, tiles.workspace.device

    @staticmethod
    def _lease(tensor):
        storage = tensor.untyped_storage()
        signature = (tensor.data_ptr(), storage.data_ptr(), storage.nbytes(), tuple(tensor.shape),
                     tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype, tensor.device)
        return tensor, storage, signature

    @staticmethod
    def memory(device):
        return {'free': int(torch.cuda.mem_get_info(device)[0]),
                'allocated': int(torch.cuda.memory_allocated(device)),
                'reserved': int(torch.cuda.memory_reserved(device))}

    def prepare(self, *, quiescent):
        started = time.monotonic()
        try:
            if (self.prepared or self.closed or self.failure is not None or self.device.type != 'cuda'
                    or not callable(quiescent) or not quiescent()):
                raise RuntimeError('reference arithmetic capture requires one-time quiescent CUDA preparation')
            self.started = True  # Journal every subsequent CUDA/allocator effect.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            self.before = self.memory(self.device)
            self.prepared_stream = torch.cuda.current_stream(self.device)
            self.side = torch.cuda.Stream(device=self.device)
            self.inputs['query'] = torch.zeros((12, 256), dtype=torch.float16, device=self.device)
            self.inputs['maximum'] = torch.full((12,), -torch.inf, dtype=torch.float32, device=self.device)
            self.inputs['denominator'] = torch.zeros((12,), dtype=torch.float32, device=self.device)
            self.inputs['global_maximum'] = torch.zeros((12,), dtype=torch.float32, device=self.device)
            self.inputs['global_denominator'] = torch.ones((12,), dtype=torch.float32, device=self.device)
            self.inputs['result'] = torch.zeros((12, 256), dtype=torch.float32, device=self.device)
            self.tiles.half.zero_()
            self.tiles.mask.zero_()
            self.tensor_leases = [self._lease(t) for t in (self.tiles.half, self.tiles.mask, *self.inputs.values())]
            calls = {
                'stats': lambda: statistics_step(self.inputs['query'], self.tiles.half, self.tiles.mask,
                    self.inputs['maximum'], self.inputs['denominator'], .0625),
                'probability': lambda: probability_step(self.inputs['query'], self.tiles.half, self.tiles.mask,
                    self.inputs['global_maximum'], self.inputs['global_denominator'], .0625),
                'pv': lambda: value_step(self.outputs['probability'], self.tiles.half, self.inputs['result'])}
            for name, call in calls.items():
                self.side.wait_stream(self.prepared_stream)
                with torch.cuda.stream(self.side):
                    for _ in range(3):
                        call()
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                self.graphs[name] = graph  # Journal before fallible capture.
                self.pools[name] = torch.cuda.graph_pool_handle()
                with torch.cuda.graph(graph, pool=self.pools[name], stream=self.side):
                    self.outputs[name] = call()
                # Initialize captured outputs before another warmup reads them.
                with torch.cuda.stream(self.side):
                    graph.replay()
                self.prepared_stream.wait_stream(self.side)
                torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            self.after = self.memory(self.device)
            physical = self.before['free'] - self.after['free']
            active = self.after['allocated'] - self.before['allocated']
            if physical < 0 or active < 0:
                raise RuntimeError('reference arithmetic residency cannot be attributed under competing memory changes')
            self.charged_bytes = max(physical, active)
            if self.charged_bytes > self.budget:
                raise ValueError('reference arithmetic exceeds the explicit aggregate preparation allowance')
            for value in self.outputs.values():
                for tensor in value if type(value) is tuple else (value,):
                    self.tensor_leases.append(self._lease(tensor))
            self.retained_tensor_bytes = sum(lease[0].numel() * lease[0].element_size() for lease in self.tensor_leases[2:])
            self._account_pools()
            self.prepared, self.capture_s = True, time.monotonic() - started
        except BaseException as error:
            self.failure = error
            raise

    def _account_pools(self):
        from .service_memory import private_pool_spare
        try:
            if torch.cuda.memory.get_allocator_backend() != 'native':
                return
            spare = 0
            for pool in self.pools.values():
                segments = torch.cuda.memory_snapshot(mempool_id=pool, include_traces=False)
                spare += private_pool_spare(segments, pool, self.device.index or 0)
            self.private_spare_bytes = spare
        except (AttributeError, TypeError):
            return  # No unsupported allocator credit; physical-only accounting.

    def validate_adoption(self):
        if not self.prepared or self.closed or self.failure is not None:
            raise RuntimeError('reference arithmetic was not successfully prepared')
        current = [self.tiles.half, self.tiles.mask, *self.inputs.values()]
        for value in self.outputs.values():
            current.extend(value if type(value) is tuple else (value,))
        if len(current) != len(self.tensor_leases) or any(tensor is not lease[0] for tensor, lease in zip(current, self.tensor_leases)):
            raise RuntimeError('reference arithmetic tensor catalogue changed after capture')
        for tensor, storage, signature in self.tensor_leases:
            if self._lease(tensor)[2] != signature or storage.data_ptr() != signature[1]:
                raise RuntimeError('reference arithmetic borrowed tensor/storage changed after capture')

    def eligible(self, query, cache, selected, scale):
        if not self.prepared or self.closed or self.failure is not None:
            raise RuntimeError('reference arithmetic owner is unavailable')
        if torch.cuda.current_stream(self.device) != self.prepared_stream:
            raise RuntimeError('reference arithmetic requires its prepared current stream')
        return (cache.capacity > 0 and selected.numel() >= 128 and query.shape == (12, 256)
                and query.dtype == torch.float16 and query.is_contiguous() and query.device == self.device
                and query.data_ptr() % 512 == 0  # Preserve the captured GEMM input alignment class.
                and scale == .0625)

    def _validate_destinations(self, out, operation):
        if (operation == 'statistics' and type(out) is not tuple
                or operation == 'normalized' and not isinstance(out, torch.Tensor)):
            raise ValueError('reference arithmetic destination schema differs from its prepared operation')
        destinations = out if type(out) is tuple else (out,)
        shapes = ((12,), (12,)) if operation == 'statistics' else ((12, 256),)
        if len(destinations) != len(shapes):
            raise ValueError('reference arithmetic destination schema differs from its prepared operation')
        reads = [self.tiles.workspace, *self.inputs.values()]
        for value in self.outputs.values():
            reads.extend(value if type(value) is tuple else (value,))
        if operation == 'statistics' and torch._C._overlaps(*destinations):
            if not (destinations[0].stride() == destinations[1].stride() == (2,)
                    and abs(destinations[0].data_ptr() - destinations[1].data_ptr()) == 4):
                raise ValueError('reference arithmetic statistics destinations overlap')
        for tensor, shape in zip(destinations, shapes):
            if (tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != self.device
                    or tensor.requires_grad or tensor.is_neg() or tensor.is_conj()
                    or (operation == 'normalized' and not tensor.is_contiguous())
                    or (operation == 'statistics' and tensor.stride() not in ((1,), (2,)))
                    or any(torch._C._overlaps(tensor, source) for source in reads)):
                raise ValueError('reference arithmetic handoff requires disjoint caller-paid FP32 destinations')

    def _replayed(self, name):
        if self.replays[name] >= 2**63 - 1:
            raise OverflowError('reference arithmetic replay counter exhausted')
        self.graphs[name].replay()
        self.replays[name] += 1

    def statistics(self, query, cache, selected, visible, scale, context, out=None):
        if not self.prepared or self.closed or self.failure is not None:
            raise RuntimeError('reference arithmetic owner is unavailable')
        if out is None or selected.numel() < 128 or not cache.capacity:
            return None
        if not self._operation_lease.acquire(blocking=False):
            raise RuntimeError('reference arithmetic operation is already borrowed')
        try:
            if not self.eligible(query, cache, selected, scale):
                return None
            self._validate_destinations(out, 'statistics')
            self.inputs['query'].copy_(query)
            maximum, denominator = self.inputs['maximum'], self.inputs['denominator']
            maximum.fill_(-torch.inf)
            denominator.zero_()
            for start in range(0, selected.numel(), 128):
                with context.tile(cache, selected[start:start + 128], visible[start:start + 128]) as tile:
                    if tile.count == 128:
                        self._replayed('stats')
                        maximum.copy_(self.outputs['stats'][0])
                        denominator.copy_(self.outputs['stats'][1])
                    else:
                        maximum, denominator = statistics_step(query, tile.key, tile.mask, maximum, denominator, scale)
            out[0].copy_(maximum)
            out[1].copy_(denominator)
            return out
        except BaseException as error:
            self.failure = error
            raise
        finally:
            self._operation_lease.release()

    def normalized(self, query, cache, selected, visible, scale, maximum, denominator, context, out=None):
        if not self.prepared or self.closed or self.failure is not None:
            raise RuntimeError('reference arithmetic owner is unavailable')
        if out is None or selected.numel() < 128 or not cache.capacity:
            return None
        if not self._operation_lease.acquire(blocking=False):
            raise RuntimeError('reference arithmetic operation is already borrowed')
        try:
            if not self.eligible(query, cache, selected, scale):
                return None
            self._validate_destinations(out, 'normalized')
            self.inputs['query'].copy_(query)
            self.inputs['global_maximum'].copy_(maximum)
            self.inputs['global_denominator'].copy_(denominator)
            result = self.inputs['result']
            result.zero_()  # Keep the original first +0-result addition.
            for start in range(0, selected.numel(), 128):
                with context.tile(cache, selected[start:start + 128], visible[start:start + 128]) as tile:
                    if tile.count == 128:
                        self._replayed('probability')  # Last K read, stable Half P.
                        tile.values()  # SAME Half slab becomes V only now.
                        self._replayed('pv')
                        result.copy_(self.outputs['pv'])
                    else:
                        # Original eager tail order: QK, values, probability, PV.
                        score = _matmul32(query, tile.key.T) * scale
                        score.masked_fill_(~tile.mask[None], -torch.inf)
                        value = tile.values()
                        probability = torch.where(tile.mask[None], torch.exp(score - maximum[:, None]) /
                                                  denominator[:, None], 0).half()
                        result = result + _matmul32(probability, value)
            out.copy_(result)
            return out
        except BaseException as error:
            self.failure = error
            raise
        finally:
            self._operation_lease.release()

    def record(self):
        return {'policy': POLICY, 'prepared': self.prepared, 'closed': self.closed,
                'full_tile': 128, 'shape': [12, 256], 'graphs': list(self.graphs),
                'graph_budget_bytes': self.budget, 'charged_preparation_bytes': self.charged_bytes,
                'retained_tensor_payload_bytes': self.retained_tensor_bytes,
                'private_pool_spare_bytes': self.private_spare_bytes,
                'before': self.before, 'after': self.after, 'capture_seconds': self.capture_s,
                'replays': dict(self.replays), 'source_sha256': self.source_sha256,
                'staging_bytes_per_eligible_row': {'query_each_pass': 6144, 'global_ML': 96},
                'feedback_bytes_per_full_tile': {'statistics': 96, 'result': 12288},
                'qualification': 'source prototype; target bits/resources/model/P3 unqualified'}

    def close(self):
        if self.closed:
            return
        if not self._operation_lease.acquire(blocking=False):
            raise RuntimeError('reference arithmetic cannot retire an active operation')
        try:
            if self.started:
                torch.cuda.synchronize(self.device)  # All three graphs retire before borrowed K/V views.
            errors = []
            for name, graph in tuple(self.graphs.items()):
                try:
                    graph.reset()
                    del self.graphs[name]
                except BaseException as error:
                    errors.append(error)
            raise_failures(None, errors)
            self.outputs.clear()
            self.inputs.clear()
            self.pools.clear()
            self.tensor_leases.clear()
            self.tiles = self.side = self.prepared_stream = None
            self.closed = True
        except BaseException as error:
            self.failure = error
            raise
        finally:
            self._operation_lease.release()
