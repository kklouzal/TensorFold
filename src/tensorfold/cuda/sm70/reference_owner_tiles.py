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
        self.normalization_workspace = self.normalization_output = None
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
        self.normalization_workspace = self.normalization_output = None

    def bind_global_statistics(self, workspace, output):
        """Bind existing local/global slabs once, after native owner preparation."""
        if (self.closed or self.normalization_workspace is not None
                or output.ndim != 3 or output.shape[1:] != (12, 2) or output.shape[0] < 1
                or workspace.dtype != torch.float32 or output.dtype != torch.float32
                or workspace.device != self.workspace.device or output.device != self.workspace.device
                or not workspace.is_contiguous() or not output.is_contiguous() or workspace.data_ptr() % 512
                or workspace.nbytes < global_normalization_workspace_bytes(output.shape[0])
                or any(torch._C._overlaps(a, b) for a, b in
                       ((workspace, output), (workspace, self.workspace), (output, self.workspace)))):
            raise ValueError("global reference normalization needs fresh disjoint paid owner slabs")
        self.normalization_workspace, self.normalization_output = workspace, output

    def record(self):
        return {"policy": "reference-owner-tile128-preparation-v1", "borrowed_device_bytes": WORKSPACE_BYTES,
                "additional_device_bytes": 0, "module_code_residency": "separate-unmeasured", "source": "already-paid owner.partials receive slab",
                "tile": TILE, "head_dim": DIM, "attention_arithmetic": "unchanged-ATen-reference",
                "global_normalization": {
                    "policy": "scalar-reductions-batched-pointwise-v1",
                    "prepared": self.normalization_workspace is not None,
                    "additional_device_bytes": 0,
                    "rows": None if self.normalization_output is None else self.normalization_output.shape[0]}}

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


def global_normalization_workspace_bytes(rows):
    """Local-send/PV borrow; final M/S use the existing separate global slab."""
    if type(rows) is not int or rows < 1:
        raise ValueError("global normalization rows must be positive")
    floats = (3 * 128 + 4 * 12) * rows
    bools = (6 * 12 + 1) * rows + 2
    return ((floats * 4 + bools + 3) // 4 + 5) * 4


def merge_global_statistics(gathered, workspace, output, group_start, status_ranges, layer):
    """Batch independent normalization work; retain each scalar reduction ABI.

    The caller has submitted the statistics gather on this stream. `workspace`
    is the now-retired local-send/PV slab; gathered receive data stays read-only
    until every reduction/pointwise consumer is enqueued. Final M/S live in the
    separate paid `output` slab through all later tile/PV consumers. No device
    Tensor allocation occurs here. This is partial A21, not grouped QSA/top-k.
    """
    if gathered.ndim != 4 or type(status_ranges) not in (tuple, list):
        raise ValueError("global reference normalization requires row statistics and reusable State ranges")
    rows = gathered.shape[1]
    heads, stride = 12, 128  # 512-byte aligned rows, matching scalar CUDA outputs.
    if (gathered.dtype != torch.float32 or gathered.shape != (8, rows, heads, 2)
            or not gathered.is_contiguous() or type(group_start) is not int or group_start not in (0, 4)
            or type(layer) is not int or not 0 <= layer < 2**31
            or workspace.dtype != torch.float32 or not workspace.is_contiguous()
            or output.dtype != torch.float32 or not output.is_contiguous()
            or output.ndim != 3 or output.shape[1:] != (heads, 2) or not 0 < rows <= output.shape[0]
            or workspace.device != gathered.device or output.device != gathered.device
            or workspace.data_ptr() % 512
            or any(torch._C._overlaps(a, b) for a, b in
                   ((gathered, workspace), (gathered, output), (workspace, output)))):
        raise ValueError("global reference normalization requires disjoint paid FP32 owner slabs")
    # Three padded rows (max, denominator, products) and one abs-finite span.
    floats = (3 * stride + 4 * heads) * rows
    bools = (2 * heads + 4 * heads + 1) * rows + 2
    constant_start = (floats * 4 + bools + 3) // 4
    if workspace.nbytes < global_normalization_workspace_bytes(rows):
        raise ValueError("global reference normalization exceeds its paid local slab")
    cursor = 0
    for status, first, last in status_ranges:
        if (type(first) is not int or type(last) is not int or first != cursor or not first < last <= rows
                or status.dtype != torch.int32 or status.shape != (2,) or not status.is_contiguous()
                or status.device != gathered.device
                or any(torch._C._overlaps(status, slab) for slab in (gathered, workspace, output))):
            raise ValueError("global reference normalization needs complete ordered State status ranges")
        cursor = last
    if cursor != rows:
        raise ValueError("global reference normalization status ranges do not cover rows")
    raw = workspace.view(-1)
    maxima = raw[:rows * stride].as_strided((rows, heads), (stride, 1))
    denominators = raw[rows * stride:2 * rows * stride].as_strided((rows, heads), (stride, 1))
    product = raw[2 * rows * stride:3 * rows * stride].as_strided((rows, 4, heads), (stride, heads, 1))
    absolute = raw[3 * rows * stride:floats].view(rows, 4, heads)
    flags = raw.view(torch.uint8)[floats * 4:floats * 4 + bools].view(torch.bool)
    finite_max = flags[:rows * heads].view(rows, heads)
    finite_peer = flags[rows * heads:5 * rows * heads].view(rows, 4, heads)
    bad_elements = flags[5 * rows * heads:6 * rows * heads].view(rows, heads)
    bad_rows = flags[6 * rows * heads:6 * rows * heads + rows]
    bad_state, needs_layer = flags[-2:-1].view(()), flags[-1:].view(())
    constants = raw[constant_start:constant_start + 5]
    zero, one = constants[0].view(()), constants[1].view(())
    minus_two, layer_word, error_word = [constants[i:i + 1].view(torch.int32).view(()) for i in (2, 3, 4)]
    zero.zero_()
    one.fill_(1)
    minus_two.fill_(-2)
    layer_word.fill_(layer)
    target = output.view(-1)
    maximum = target[:output.shape[0] * heads].view(output.shape[0], heads)[:rows]
    denominator = target[output.shape[0] * heads:].view(output.shape[0], heads)[:rows]
    pieces = gathered[group_start:group_start + 4]
    # EXACT old input shape/stride: [4,12] from [8,R,12,2]. Padded
    # destinations and product rows retain scalar-output 512-byte alignment.
    for row in range(rows):
        torch.amax(pieces[:, row, :, 0], dim=0, out=maxima[row])
    # isfinite has no ordinary out overload. abs(x)<inf implements the same
    # finite predicate (including NaN/+-inf/+-0) using standard out operations.
    torch.abs(maxima, out=absolute[:, 0])
    torch.lt(absolute[:, 0], torch.inf, out=finite_max)
    torch.where(finite_max, maxima, zero, out=maximum)
    peer_maxima = pieces[..., 0].permute(1, 0, 2)
    torch.abs(peer_maxima, out=absolute)
    torch.lt(absolute, torch.inf, out=finite_peer)
    torch.sub(peer_maxima, maximum[:, None], out=product)
    torch.exp(product, out=product)
    torch.where(finite_peer, product, zero, out=product)
    torch.mul(product, pieces[..., 1].permute(1, 0, 2), out=product)
    for row in range(rows):
        torch.sum(product[row], dim=0, out=denominators[row])
    torch.abs(denominators, out=absolute[:, 0])
    torch.lt(absolute[:, 0], torch.inf, out=finite_max)
    torch.logical_not(finite_max, out=finite_max)
    torch.le(denominators, 0, out=bad_elements)
    torch.logical_or(finite_max, bad_elements, out=bad_elements)
    torch.any(bad_elements, dim=1, out=bad_rows)
    # Every row for this State uses the same layer; aggregate only the error
    # predicate, preserving sticky bits and the original first-failure layer.
    for status, first, last in status_ranges:
        torch.any(bad_rows[first:last], dim=0, out=bad_state)
        error_word.copy_(bad_state)
        error_word.mul_(16)
        status[0].bitwise_or_(error_word)
        torch.eq(status[1], minus_two, out=needs_layer)
        torch.logical_and(bad_state, needs_layer, out=needs_layer)
        torch.where(needs_layer, layer_word, status[1], out=status[1])
    torch.gt(denominators, 0, out=finite_max)
    torch.where(finite_max, denominators, one, out=denominator)
    return maximum, denominator
