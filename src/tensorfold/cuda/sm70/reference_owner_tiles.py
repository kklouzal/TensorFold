"""Exact reference-tile preparation; attention GEMMs/reductions remain owned by ATen.

Explicit preparation only. One paid communication receive slab is borrowed
between its collective uses on one stream. The Half tile is reused for V only
AFTER the QK consumer has been enqueued. Views expire when their tile lease ends;
no full-history decoded shadow or new device allocation is retained.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path
import threading

import torch

TILE = 128
DIM = 256
HALF_OFFSET = 1536  # mask128 + int64 indices1024, rounded up to a512-byte boundary
WORKSPACE_BYTES = HALF_OFFSET + TILE * DIM * 2


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources
    from .rotorquant_q4_tables import verify

    verify()
    prepared = _native()
    root = Path(__file__).parent
    sources = [root / "reference_owner_tiles_bindings.cpp", root / "reference_owner_tiles.cu"]
    headers = [root / n for n in ("packed_kv_decode.cuh", "rotorquant_q4.cuh", "rotorquant_q4_tables.cuh")]
    headers.append(root.parent / "kernels/tensor_contracts.cuh")
    digest = hashlib.sha256(prepared.__name__.encode())
    for p in [*sources, *headers]:
        digest.update(p.name.encode() + b"\0" + p.read_bytes())
    staged = stage_sources(sources, headers=headers, source_root=root.parent)
    return build.load("tf_sm70_reference_tiles_" + digest.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false", "--ftz=false"], extra_cflags=["-O3"])


def preparation_identity():
    """Fresh source/ELF identity, without changing policy or allocating model buffers."""
    module = extension()
    root = Path(__file__).parent
    paths = [Path(__file__), root / "reference_owner_tiles.cu", root / "reference_owner_tiles_bindings.cpp",
             *[root / n for n in ("packed_kv_decode.cuh", "rotorquant_q4.cuh", "rotorquant_q4_tables.cuh")],
             root.parent / "kernels/tensor_contracts.cuh"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    binary = Path(module.__file__)
    return {"policy": "reference-owner-tile128-preparation-v1", "source_sha256": digest.hexdigest(),
            "module_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "module_bytes": binary.stat().st_size, "module_code_physical_bytes": None,
            "accounting": "functions-loaded-before-owner-and-admission-free-snapshots"}


def prepare():
    extension().prepare()  # cudaFuncGetAttributes loads both entrypoints, including lazy modules.
    return preparation_identity()


class ReferenceTiles:
    """A bounded, exclusive borrow of already-accounted OwnerBuffers.partials.

    No allocation occurs here or in a tile operation. Construct/load before
    admission/warmup. Callers must finish previous slab reads on this stream
    before a lease; all tile consumers precede the next collective/slab writer.
    """
    def __init__(self, workspace, preparation=None):
        self._lease = threading.Lock()
        self._token = None
        self.closed = False
        self.workspace = self.mask = self.indices = self.half = None
        if (workspace.dtype != torch.float32 or not workspace.is_contiguous()
                or workspace.numel() * workspace.element_size() < WORKSPACE_BYTES
                or workspace.data_ptr() % 512):
            raise ValueError("reference tiles require an aligned paid contiguous FP32 receive slab")
        self.preparation = None if preparation is None else dict(preparation)
        if preparation is not None and preparation != preparation_identity():
            raise ValueError("reference tile prepared source/module identity changed")
        raw = workspace.reshape(-1).view(torch.uint8)
        self.workspace = workspace
        self.mask = raw[:TILE].view(torch.bool)
        self.indices = raw[TILE:TILE + TILE * 8].view(torch.int64)
        self.half = raw[HALF_OFFSET:WORKSPACE_BYTES].view(torch.float16).view(TILE, DIM)
        self.stream = torch.cuda.current_stream(workspace.device).cuda_stream if workspace.is_cuda else None

    def begin_retirement(self):
        if not hasattr(self, "_lease"):
            self.closed = True
            return
        if not self._lease.acquire(blocking=False):
            raise RuntimeError("cannot retire a borrowed reference tile receive slab")
        try:
            self.closed = True  # Stop new leases before the owner's existing device fence.
        finally:
            self._lease.release()

    def release_views(self):
        """Caller has completed its existing owner fence after begin_retirement."""
        if not self.closed:
            raise RuntimeError("reference tile views require proven owner retirement")
        self.workspace = self.mask = self.indices = self.half = None

    def record(self):
        return {"policy": "reference-owner-tile128-preparation-v1", "borrowed_device_bytes": WORKSPACE_BYTES,
                "additional_device_bytes": 0, "module_code_residency": "separate-unmeasured", "source": "already-paid owner.partials receive slab",
                "tile": TILE, "head_dim": DIM, "attention_arithmetic": "unchanged-ATen-reference"}

    def verify(self, workspace, preparation):
        if (self.closed or self.workspace is not workspace or self.preparation != preparation
                or self.mask.shape != (TILE,) or self.mask.dtype != torch.bool
                or self.indices.shape != (TILE,) or self.indices.dtype != torch.int64
                or self.half.shape != (TILE, DIM) or self.half.dtype != torch.float16
                or any(not tensor.is_contiguous() or tensor.device != workspace.device
                       for tensor in (self.mask, self.indices, self.half))
                or preparation != preparation_identity() or self.mask.data_ptr() != workspace.data_ptr()
                or self.indices.data_ptr() != workspace.data_ptr() + TILE
                or self.half.data_ptr() != workspace.data_ptr() + HALF_OFFSET):
            raise ValueError("actual reference tile preparer differs from its prepared owner/source")
        return {**self.record(), "preparation": dict(preparation), "receive_slab_bytes": workspace.nbytes}

    def context(self, *, status, generations, slot_generation, owner_generation, layer):
        return _Context(self, status, generations, slot_generation, owner_generation, layer)


class _Context:
    def __init__(self, owner, status, generations, slot, generation, layer):
        if (any(type(v) is not int or not 0 <= v < 2**63 for v in (slot, generation))
                or type(layer) is not int or not 0 <= layer < 2**31):
            raise ValueError("reference tile generations/layer require exact bounded integers")
        self.owner, self.status, self.generations = owner, status, generations
        self.slot, self.generation, self.layer = slot, generation, layer

    @contextmanager
    def tile(self, cache, positions, visible):
        owner = self.owner
        if not owner._lease.acquire(blocking=False):
            raise RuntimeError("reference tile receive-slab borrow is already active")
        tile = None
        try:
            if owner.closed:
                raise RuntimeError("reference tile receive slab is retired")
            if (not owner.workspace.is_cuda or torch.cuda.current_stream(owner.workspace.device).cuda_stream != owner.stream):
                raise RuntimeError("reference tile operations require their prepared CUDA stream")
            if cache.key_format != cache.value_format:
                raise ValueError("reference tile preparation requires homogeneous key/value formats")
            bits, codec = cache.key_format.bits, cache.key_format.codec
            token = object()
            owner._token = token
            extension().keys(cache.k, cache.ks, cache.page_offsets, positions, visible,
                self.generations, self.status, owner.indices, owner.mask, owner.half,
                cache.logical_capacity, cache.placement.page_tokens, bits, codec,
                self.slot, self.generation, self.layer)
            tile = _Tile(self, cache, positions.numel(), token)
            yield tile
        finally:
            if tile is not None:
                tile.indices = tile.mask = tile.key = tile.cache = None
            owner._token = None
            owner._lease.release()


class _Tile:
    def __init__(self, context, cache, count, token):
        self.context, self.cache, self.count, self.token = context, cache, count, token
        self.indices = context.owner.indices[:count]
        self.mask = context.owner.mask[:count]
        self.key = context.owner.half[:count]

    def values(self):
        context, owner = self.context, self.context.owner
        if owner._token is not self.token or torch.cuda.current_stream(owner.workspace.device).cuda_stream != owner.stream:
            raise RuntimeError("reference tile lease or stream has retired")
        cache = self.cache
        extension().values(cache.v, cache.vs, owner.indices, owner.mask, context.generations,
            context.status, owner.half, self.count, cache.value_format.bits, cache.value_format.codec,
            context.slot, context.generation, context.layer)
        return owner.half[:self.count]
