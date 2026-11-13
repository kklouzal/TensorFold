"""Private paid host packet owner; blocking readback, serial bounded borrowing.

Only independent Python tokens/probabilities may leave a borrow. The caller
releases it before any full-source sampler fallback. Graphs retain their
original device packets and collectives; this owner is never captured.
"""
import sys
import struct
import threading
import weakref
import _thread
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


def _metadata_size(owner, *extra):
    """Distinct controlled Python/header objects; borrowed model storage excluded.

    Source-controlled dictionary keys, policy strings, dtype singletons and
    world/row/role arguments are borrowed. Torch/NumPy payloads are charged by
    layout(), once. C++ allocator/interpreter internals are not Python headers.
    IDs only deduplicate this strongly retained, bounded object graph.
    """
    seen=set();pending=[owner,*extra];total=scratch_peak=0
    pointer_integer_bytes=sys.getsizeof((1<<(8*struct.calcsize('P')))-1)
    while pending:
        value=pending.pop()
        if value is None or type(value) in (str,bool) or isinstance(value,torch.dtype):
            continue
        ident=id(value)
        if ident in seen:continue
        seen.add(ident)
        if value is owner and isinstance(value,GatheredReadback):
            total+=sys.getsizeof(value)+sys.getsizeof(vars(value))
            for name,item in vars(value).items():
                if name not in ('weights','packet','comm','world','rows','role'):pending.append(item)
        elif type(value) is dict:
            total+=sys.getsizeof(value);pending.extend(value.values())
        elif type(value) in (tuple,list):
            total+=sys.getsizeof(value);pending.extend(value)
        elif isinstance(value,np.ndarray):
            total+=sys.getsizeof(value)-(value.nbytes if value.flags.owndata else 0)
            if value.base is not None:pending.append(value.base)
        elif type(value) is torch.Tensor or isinstance(value,(torch.device,weakref.ReferenceType,_thread.LockType)):
            total+=sys.getsizeof(value)
        elif type(value) is int:
            total+=sys.getsizeof(value)
        else:
            raise ValueError('unbounded gathered readback metadata type: '+type(value).__name__)
        # Explicit traversal containers are temporary setup/record work, not
        # retained per-role storage. Their actual capacity is included in peak.
        scratch_peak=max(scratch_peak,sys.getsizeof(pending)+sys.getsizeof(seen)
            +len(seen)*pointer_integer_bytes)
    return total,total+scratch_peak


def _fresh_int(value):
    # Distinct runtime scalar objects matter: equal pointers in a signature
    # need not share their independently returned Python integer wrappers.
    return int(str(value))


def _signature_bound(elements,storage_bytes,device):
    pointer=(1<<(8*struct.calcsize('P')))-1
    return (_fresh_int(pointer),_fresh_int(pointer),(_fresh_int(elements),),tuple([1]),
            torch.float32,torch.device(device),_fresh_int(pointer),_fresh_int(storage_bytes))


def metadata_forecast(world,rows,role):
    """Pinned-runtime header bound, without CUDA or host payload materialization.

    The same initializer defines real and prototype state. Complete packet
    storages may exceed their one-dimensional views; nbytes uses size_t's full
    domain, and identity/pointer wrappers use the platform's pointer domain.
    A combined dictionary bounds the instance's possible split dictionary.
    """
    prototype=object.__new__(GatheredReadback)
    prototype._initialize(world,rows,role,'cuda')
    # Instance split dictionaries can shrink as more same-class instances are
    # created. Use the same complete insertion sequence in an unshared dict so
    # a forecast/TP-rank does not depend on earlier prototype history. The old
    # representation retires at replacement; bound its copy overlap, not its
    # history-dependent retained size.
    complete={}
    for key,value in vars(prototype).items():complete[key]=value
    dictionary_bound=sys.getsizeof(complete)
    if sys.getsizeof(vars(prototype))>dictionary_bound:
        raise ValueError('unsupported readback instance dictionary capacity')
    prototype.__dict__=complete
    prototype.host=torch.empty(0,dtype=torch.float32,device='cpu')
    prototype.words=prototype.host.numpy()
    prototype.ids=np.empty(0,dtype=np.int64)
    prototype.original_words,prototype.original_ids=prototype.words,prototype.ids
    pointer=(1<<(8*struct.calcsize('P')))-1
    size=(1<<(8*struct.calcsize('N')))-1
    prototype.ids_pointer=_fresh_int(pointer)
    prototype.metadata_bound_bytes=_fresh_int(size)
    prototype.metadata_scratch_bound_bytes=_fresh_int(size)
    elements=prototype.cost['pinned_host_bytes']//4
    prototype.host_signature=_signature_bound(elements,prototype.cost['pinned_host_bytes'],'cpu')
    prototype.packet_signature=_signature_bound(elements,size,'cuda')
    prototype.buffer=weakref.ref(prototype)  # Header only; its target is borrowed.
    bound,walk_peak=_metadata_size(prototype)
    result={'host_metadata_bound_bytes':bound,'host_traversal_scratch_bound_bytes':walk_peak-bound,
            'host_setup_metadata_peak_bytes':_fresh_int(size)}
    # Dictionary conversion and decimal pointer construction are setup-only.
    setup=_metadata_size(prototype,result)[1]+dictionary_bound+max(
        sys.getsizeof(str(size)),sys.getsizeof(str(pointer)),sys.getsizeof([1]))
    if bound>size or setup>size:
        raise OverflowError('gathered readback metadata exceeds the native size domain')
    result['host_setup_metadata_peak_bytes']=setup
    return result


def forecast(world,role_rows):
    if type(role_rows) is not dict or not role_rows or set(role_rows)-{'decode','mtp'}:
        raise ValueError('readback forecast requires its finite decode/MTP role map')
    return {role:{**layout(world,role_rows[role]),**metadata_forecast(world,role_rows[role],role)}
            for role in ('decode','mtp')if role in role_rows}


def forecast_resources(costs):
    """One-rank explicit peak; serial setup retires before ordinary service."""
    headers,forecast_walk_peak=_metadata_size(costs) if costs else (0,0)
    owner_map={}
    for role in costs:owner_map[role]=None  # Same insertion pattern; owners are separately charged.
    owner_map_headers=sys.getsizeof(owner_map) if costs else 0
    metadata=sum(cost['host_metadata_bound_bytes'] for cost in costs.values())
    payload=sum(cost['host_payload_bytes'] for cost in costs.values())
    peak=prefix=headers+owner_map_headers
    for cost in costs.values():
        # Prior roles stay live during the next empty actor/prototype setup.
        # The actor and prototype have distinct headers; no array payload is
        # present in the prototype. Forecast and owner-map headers are each paid once.
        peak=max(peak,prefix+cost['host_metadata_bound_bytes']+cost['host_setup_metadata_peak_bytes'])
        prefix+=cost['host_payload_bytes']+cost['host_metadata_bound_bytes']
    persistent=payload+metadata+headers+owner_map_headers
    scratch=max([forecast_walk_peak-headers,owner_map_headers,
        *(cost['host_traversal_scratch_bound_bytes']for cost in costs.values())])
    return {'host_payload_bytes':payload,'host_metadata_bound_bytes':metadata,
        'forecast_metadata_bytes':headers,'host_persistent_bytes':persistent,
        'owner_map_metadata_bound_bytes':owner_map_headers,
        'host_observation_scratch_bound_bytes':scratch,
        'host_admission_bytes':max(peak,persistent+scratch),
        'scope':'controlled headers and payload; native allocator/interpreter/copy internals separately observed'}


class GatheredReadback:
    def _initialize(self, world, rows, role, device):
        # Complete field layout is established before allocation/binding. Bind
        # changes values, never dictionary capacity; no external fields grow it.
        self.closed = self.poisoned = False
        self.bound = False
        self.lock = threading.Lock()
        self.host = self.ids = self.words = None
        self.weights = self.buffer = self.packet = self.comm = None
        self.original_words = self.original_ids = None
        self.host_signature = self.packet_signature = None
        self.ids_pointer = self.metadata_bound_bytes = self.metadata_scratch_bound_bytes = 0
        self.world, self.rows, self.role = world, rows, role
        self.device = torch.device(device)
        self.cost = layout(world, rows)
        if role not in ('decode', 'mtp') or self.device.type not in ('cpu','cuda'):
            raise ValueError('gathered readback is only selected for target/MTP Buffer roles')

    def __init__(self, world, rows, role, device):
        self._initialize(world,rows,role,device)
        planned=metadata_forecast(world,rows,role)
        self.metadata_bound_bytes=planned['host_metadata_bound_bytes']
        self.metadata_scratch_bound_bytes=planned['host_traversal_scratch_bound_bytes']
        del planned  # No prototype/forecast result survives payload allocation.
        self.host = torch.empty(self.cost['pinned_host_bytes'] // 4, dtype=torch.float32,
                                pin_memory=self.device.type == 'cuda')
        self.words = self.host.numpy()
        from .state import CAND
        self.ids = np.empty(world * rows * CAND, dtype=np.int64)
        self.ids_pointer = self.ids.ctypes.data
        self.original_words, self.original_ids = self.words, self.ids
        self.host_signature = _signature(self.host)
        self._check_metadata()

    def metadata_bytes(self):
        return _metadata_size(self)[0]

    def _check_metadata(self):
        try:
            measured,peak=_metadata_size(self)
            if (measured>self.metadata_bound_bytes or peak-measured>self.metadata_scratch_bound_bytes):
                raise MemoryError('gathered readback metadata exceeds its pre-materialization bound')
            return measured,peak-measured
        except BaseException:
            self.poisoned=True
            raise

    def bind(self, weights, buffers):
        if self.closed or self.poisoned or self.bound:
            raise RuntimeError('gathered readback binds once before State/capture')
        if (weights.meta.get('execution_provider') != 'sm70' or weights.meta.get('world') != self.world
                or weights.comm is None
                or weights.comm.world != self.world or buffers.world != self.world
                or self.world != 8 or buffers.rows != self.rows or buffers.owner_role != self.role
                or buffers.execution_dtype != torch.float16 or buffers.cand_all.device != self.device
                or buffers.cand_all.dtype != torch.float32 or buffers.cand_all.ndim!=1 or not buffers.cand_all.is_contiguous()
                or buffers.cand_all.numel() != self.host.numel()):
            raise ValueError('gathered readback differs from its actual TP8 Half packet/Buffer role')
        self.weights, self.buffer, self.packet = weights, weakref.ref(buffers), buffers.cand_all
        self.comm = weights.comm
        self.packet_signature = _signature(self.packet)
        self.bound = True
        self._check_metadata()

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
        metadata,scratch=self._check_metadata()
        return {'policy': POLICY, 'role': self.role, 'world': self.world, 'rows': self.rows,
                'host_metadata_bytes':metadata,'host_metadata_bound_bytes':self.metadata_bound_bytes,
                'host_metadata_walk_scratch_bytes':scratch,
                'host_metadata_walk_scratch_bound_bytes':self.metadata_scratch_bound_bytes,
                **self.cost, 'bound': self.bound, 'closed': self.closed, 'poisoned': self.poisoned,
                'qualification': 'private-unqualified; native/model/P3 pending',
                'metadata_scope':'controlled retained Python/NumPy/Tensor headers; disjoint payload charged once',
                'uncontrolled': 'NumPy per-row sampling temporaries and native allocator/interpreter/copy machinery'}

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
