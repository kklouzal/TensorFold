"""Private canonical sampling packet scratch; original Torch row arithmetic.

One Buffers role owns these tensors before States/capture. Each score row and
temporary output is consumed on the caller's serialized current stream before
the next row overwrites it. Graphs borrow the same retained storages; this is
not a concurrent-stream workspace or another sampling algorithm.
"""
import sys
import weakref

import torch


POLICY = 'canonical-row-scratch-v1'
ALIGNMENT = 512


def workspace_bytes(width):
    from .state import CAND
    if type(width) is not int or width < 1:
        raise ValueError('row sampling requires a positive vocabulary shard width')
    return 4 * (width + ALIGNMENT // 4 - 1) + 16 * min(CAND, width)


def _signature(tensor):
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device, tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())


class RowSamplingScratch:
    def __init__(self, width, rows, role, device):
        from .state import CAND
        self.tensors, self.storage_owners = [], ()
        self.weights = self.buffer = self.anchors = self.signatures = None
        self.closed = self.bound = False
        self.width, self.rows, self.role = width, rows, role
        self.device = torch.device(device)
        self.expected_bytes = workspace_bytes(width)
        if type(rows) is not int or rows < 1 or role not in ('decode', 'mtp', 'prefill'):
            raise ValueError('row sampling requires its bounded actual Buffers role')
        kept = min(CAND, width)
        for shape, dtype in (((width + ALIGNMENT // 4 - 1,), torch.float32),
                             ((kept,), torch.int64), ((kept,), torch.int64)):
            self.tensors.append(torch.empty(shape, dtype=dtype, device=device))
        self.device = self.tensors[0].device
        self.storage_owners = tuple(t.untyped_storage() for t in self.tensors)
        first = (-self.tensors[0].data_ptr()) % ALIGNMENT // 4
        self.score = self.tensors[0][first:first + width]
        if self.score.data_ptr() % ALIGNMENT or self.payload_bytes != self.expected_bytes:
            raise ValueError('row sampling scratch alignment or byte extent differs from its cold forecast')
        self.original_score, self.score_signature = self.score, _signature(self.score)

    def bind(self, weights, buffers):
        if self.closed or self.bound:
            raise RuntimeError('row sampling scratch binds once before State/capture')
        if (weights.meta.get('execution_provider') != 'sm70' or weights.comm is None
                or weights.comm.world != 8 or buffers.world != 8 or weights.head.n != self.width
                or buffers.rows != self.rows or buffers.owner_role != self.role
                or buffers.execution_dtype != torch.float16 or buffers.logits.device != self.device
                or buffers.logits.dtype != torch.float32 or buffers.logits.shape[1] != self.width):
            raise ValueError('row sampling scratch differs from its admitted TP8 Half model/Buffer role')
        self.weights, self.buffer = weights, weakref.ref(buffers)
        self.comm = weights.comm
        self.heads = weights.head, weights.draft_head
        mapping = weights.draft_ids
        if mapping is not None and (type(mapping) is not torch.Tensor or mapping.ndim != 1
                                    or mapping.dtype != torch.int64 or mapping.device != self.device):
            raise ValueError('row sampling retains the actual original int64 device ID map')
        self.mapping = mapping
        self.mapping_version = None if mapping is None or mapping.is_inference() else mapping._version
        self.offset = int(weights.meta['vocab_offset'])
        self.draft_width = weights.head.n if weights.draft_head is None else weights.draft_head.n
        if not 0 < self.draft_width <= self.width or mapping is not None and mapping.numel() != self.draft_width:
            raise ValueError('draft sampling width/map exceeds the original shard scratch envelope')
        self.anchors = (buffers.logits, buffers.cand, buffers.cand_all, *self.tensors,
                        *((mapping,) if mapping is not None else ()))
        self.signatures = tuple(_signature(t) for t in self.anchors)
        self.storage_owners += tuple(t.untyped_storage() for t in self.anchors[:3])
        if mapping is not None:
            self.storage_owners += (mapping.untyped_storage(),)
        for i, left in enumerate(self.anchors):
            if any(torch._C._overlaps(left, right) for right in self.anchors[i + 1:]):
                raise ValueError('sampling scratch and packet/logit/map owners must be disjoint')
        self.bound = True

    def validate_bindings(self, weights, buffers):
        if (self.closed or not self.bound or self.weights is not weights or self.buffer() is not buffers
                or buffers.owner_role != self.role or buffers.rows != self.rows
                or weights.comm is not self.comm or weights.comm.world != 8 or buffers.world != 8
                or weights.head is not self.heads[0] or weights.draft_head is not self.heads[1]
                or len(self.tensors) != 3
                or any(current is not original for current, original in zip(self.tensors, self.anchors[3:6]))
                or self.score is not self.original_score or _signature(self.score) != self.score_signature
                or any(current is not original for current, original in
                       zip((buffers.logits, buffers.cand, buffers.cand_all), self.anchors[:3]))
                or weights.draft_ids is not self.mapping):
            raise RuntimeError('sampling scratch requires its original live weights and Buffers role')
        if any(_signature(t) != expected for t, expected in zip(self.anchors, self.signatures)):
            raise ValueError('sampling tensor/storage changed after cold binding')
        if self.mapping_version is not None and self.mapping._version != self.mapping_version:
            raise ValueError('sampling ID map changed after cold binding')

    def pack(self, weights, buffers, logits, rows, *, id_map, offset):
        from .state import CAND
        self.validate_bindings(weights, buffers)
        if not isinstance(logits, torch.Tensor):
            raise ValueError('canonical sampling requires original Tensor logits')
        width = logits.shape[1] if logits.ndim == 2 else 0
        expected_width = self.draft_width if self.role == 'mtp' or id_map is not None else self.width
        if (type(rows) is not int or not 0 < rows <= buffers.cand.shape[0]
                or logits.dtype != torch.float32 or logits.device != self.device or logits.ndim != 2
                or logits.shape[0] < rows or width != expected_width or logits.stride(1) != 1
                or logits.stride(0) < width or id_map is not None and id_map is not self.mapping
                or id_map is None and offset != self.offset
                or logits.requires_grad or any(t.requires_grad for t in self.anchors)
                or not torch.is_inference_mode_enabled() and any(t.is_inference() for t in self.anchors[1:6])):
            raise ValueError('unsupported canonical sampling row/map/layout/autograd state')
        first = logits.data_ptr() - buffers.logits.data_ptr()
        stop = first + ((rows - 1) * logits.stride(0) + width) * logits.element_size()
        if (first < 0 or first % (width * 4) or stop > buffers.logits.numel() * 4
                or logits.untyped_storage().data_ptr() != buffers.logits.untyped_storage().data_ptr()):
            raise ValueError('sampling rows must remain inside their retained logical logit owner')
        kept = min(CAND, width)
        scores = self.score[:width].view(1, width)
        indices = self.tensors[1][:kept].view(1, kept)
        global_ids = self.tensors[2][:kept].view(1, kept)
        for row in range(rows):
            packet = buffers.cand[row]
            values = packet[:kept].view(1, kept)
            ids = packet[CAND:CAND + kept].view(torch.int32).view(1, kept)
            lse = packet[2 * CAND:].view(1, 1)
            scores.copy_(logits[row:row + 1])
            torch.topk(scores, kept, dim=-1, sorted=False, out=(values, indices))
            if id_map is None:
                torch.add(indices, offset, out=global_ids)
            else:
                torch.index_select(id_map, 0, indices.view(-1), out=global_ids.view(-1))
            ids.copy_(global_ids)
            if kept < CAND:
                packet[kept:CAND].fill_(-float('inf'))
                packet[CAND + kept:2 * CAND].view(torch.int32).fill_(-1)
            torch.logsumexp(scores, dim=-1, keepdim=True, out=lse)

    @property
    def payload_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)

    def record(self):
        metadata = (sys.getsizeof(self) + sys.getsizeof(vars(self))
                    + sys.getsizeof(self.tensors) + sys.getsizeof(self.storage_owners)
                    + sum(sys.getsizeof(t) for t in self.tensors)
                    + (0 if not hasattr(self, 'score') else sys.getsizeof(self.score)))
        if self.signatures is not None:
            metadata += sys.getsizeof(self.anchors) + sys.getsizeof(self.signatures) + sys.getsizeof(self.buffer)
            metadata += sum(sys.getsizeof(signature) + sys.getsizeof(signature[2]) + sys.getsizeof(signature[3])
                            + sum(sys.getsizeof(signature[i]) for i in (0, 1, 6, 7))
                            for signature in self.signatures)
        return {'policy': POLICY, 'role': self.role, 'rows': self.rows, 'maximum_width': self.width,
                'device_tensor_bytes': self.payload_bytes, 'forecast_bytes': self.expected_bytes,
                'host_metadata_bytes': metadata,
                'bound': self.bound, 'closed': self.closed,
                'uncontrolled': 'Torch internal topk/logsumexp workspaces; no allocation-free claim',
                'qualification': 'private-unqualified; exact GPU packets/model gates pending'}

    def close(self):
        if self.closed:
            return
        if any(t.is_cuda for t in self.tensors):
            torch.cuda.synchronize(self.device)
        self.weights = self.buffer = self.anchors = self.signatures = None
        self.comm = self.heads = None
        self.original_score = self.score_signature = None
        self.tensors.clear()
        self.storage_owners = ()
        self.score = self.mapping = None
        self.closed = True
