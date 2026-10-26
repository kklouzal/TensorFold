"""One bounded batch of actual target input embeddings on the existing TP cohort.

CPU work must already be ready. Only rank zero borrows its owned CPU pixels;
FP16 patches travel to the one fixed encoder, and completed input rows broadcast
from that rank. Target and MTP borrow the same rows until their staging retires.
The encoder cache has fixed whole-unit slots, independent of resident sessions.
"""

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os

import torch

from tensorfold.cleanup import raise_failures

from ..media_plan import QwenMediaPlan
from ..position_profile import position_for
from .input_preparation import InputPreparationContext, KEY
from .media_inputs import ReadyInputSpan
from .position_coordinates import issue_coordinates


@dataclass(frozen=True)
class MediaInputSlice:
    plan: QwenMediaPlan
    generation: int
    start: int
    count: int


def producer_resources(*, rows, patches, cache_rows, hidden, world, encoder_rank):
    """Explicit buffer payload; vision weights/library workspace are separate.

    The same bound applies on cache misses. CPU pixels are an owned FP32 copy
    of one bounded decoder lease; budget that lease separately in MediaEnvelope.
    Encoder operator intermediates remain in the declared vision workspace.
    """
    if (any(type(v) is not int or v <= 0 for v in (rows, patches, hidden, world))
            or rows >= 1 << 31 or patches > (1 << 31) - 1 or patches % 4
            or type(cache_rows) is not int or cache_rows < 0
            or type(encoder_rank) is not int or not 0 <= encoder_rank < world):
        raise ValueError("media producer requires explicit bounded rows, patches, cache and placement")
    unit_rows, width = patches // 4, 3 * 2 * 16 * 16
    slots = cache_rows // unit_rows
    static = [rows * hidden * 2] * world
    temporary = [rows * 12] * world
    for rank in {0, encoder_rank}:
        static[rank] += patches * width * 2
    # Bounded decoded embedding output and the largest uncached projected unit.
    static[encoder_rank] += rows * 8 + slots * unit_rows * hidden * 2
    # Text lookup and whole-unit encoding execute sequentially on the same
    # coordinator stream. Prior slices' coordinates remain live through both.
    temporary[encoder_rank] += max(rows * hidden * 2, unit_rows * hidden * 2)
    per_rank = [base + work for base, work in zip(static, temporary, strict=True)]
    if max(per_rank) >= 1 << 63:
        raise ValueError("media producer byte products exceed the signed-int64 allocation contract")
    return {"device_bytes_by_rank": tuple(per_rank), "static_device_bytes_by_rank": tuple(static),
            "temporary_device_bytes_by_rank": tuple(temporary), "cache_slots": slots,
            "coordinate_device_bytes": rows * 12, "embedding_output_bytes": rows * hidden * 2,
            "projected_unit_output_bytes": unit_rows * hidden * 2,
            "cache_payload_rows": slots * unit_rows, "unit_rows": unit_rows,
            "cpu_pixel_copy_bytes": patches * width * 4,
            "cpu_pixel_cast_overlap_bytes": patches * width * 6,
            "cpu_coordinate_arrays_bytes": max(60 * rows + 24, 24 * rows + 32 * unit_rows + 24),
            "scope": "numeric buffer payload; runtime/allocator and library workspace accounted separately"}


class MediaInputProducer:
    @classmethod
    def create(cls, weights, *, encoder, encoder_rank, rows, patches, cache_rows, register):
        if not callable(register):
            raise ValueError("media producer requires its enclosing model's owner journal")
        owner = cls.__new__(cls)
        owner.weights, owner.encoder = weights, encoder
        owner.pid, owner.closed, owner.failed = os.getpid(), False, False
        owner.values = owner.ids = owner.pixels = owner.cache_values = owner.stream = None
        owner.spans, owner.coordinates, owner.cache = [], [], OrderedDict()
        owner.unit_output = owner.cpu_pixels = None
        owner.gpu_started = False
        owner.context = weights.meta.get(KEY)
        owner.rank, owner.world = weights.meta.get("rank", 0), weights.meta.get("world", 1)
        owner.device, owner.encoder_rank = weights.device, encoder_rank
        owner.rows, owner.patches, owner.hidden = rows, patches, weights.cfg.hidden
        owner.resources = producer_resources(rows=rows, patches=patches, cache_rows=cache_rows,
                                            hidden=owner.hidden, world=owner.world, encoder_rank=encoder_rank)
        if (type(owner.context) is not InputPreparationContext or weights.positions is None
                or weights.meta.get("media_inputs_enabled") is not True
                or weights.meta.get("coordinate_profile_identity") != owner.context.identity
                or owner.rank == encoder_rank and encoder is None or owner.rank != encoder_rank and encoder is not None
                or owner.device.type not in ("cpu", "cuda")
                or owner.device.type == "cuda" and owner.world != 8):
            raise ValueError("media producer lacks its admitted profile, actual placement or preparation owner")
        owner.position = position_for(weights)
        owner.roles = ("target", "mtp") if weights.mtp is not None else ("target",)
        owner.free_cache = list(range(owner.resources["cache_slots"]))
        register(owner)
        try:
            owner.gpu_started = owner.device.type == "cuda"
            if owner.gpu_started:
                owner.stream = torch.cuda.current_stream(owner.device)  # Borrow the serialized coordinator's stream.
            with owner._stream(), torch.inference_mode():
                owner.values = torch.empty((rows, owner.hidden), dtype=torch.float16, device=owner.device)
                if owner.rank == encoder_rank:
                    owner.ids = torch.empty(rows, dtype=torch.int64, device=owner.device)
                if owner.rank in (0, encoder_rank):
                    owner.pixels = torch.empty((patches, 1536), dtype=torch.float16, device=owner.device)
                if owner.rank == encoder_rank and owner.resources["cache_slots"]:
                    owner.cache_values = torch.empty((owner.resources["cache_slots"], patches // 4, owner.hidden),
                                                     dtype=torch.float16, device=owner.device)
        except BaseException:
            owner.failed = True
            raise
        return owner

    def _stream(self):
        return nullcontext() if self.stream is None else torch.cuda.stream(self.stream)

    def _check(self):
        if os.getpid() != self.pid or self.closed or self.failed:
            raise ValueError("media producer is retired, failed or belongs to another process")
        if self.context._active is not None:
            raise ValueError("media transport must precede the batch's nested input-preparation operation")

    def _fail(self, primary):
        self.failed = True
        if os.getpid() != self.pid:
            return  # Never touch an inherited native communicator in a child.
        try:
            self.context.fail(primary)
        except BaseException as notification:
            raise_failures(primary, (notification,))

    @staticmethod
    def _key(plan, unit):
        return plan.media_identity, unit.probe_index, unit.run.temporal_group

    def cached(self, plan, unit):
        self._check()
        return self._key(plan, unit) in self.cache

    def _unit(self, index, spec, unit, pixel_loader, contract):
        key, run = self._key(spec.plan, unit), unit.run
        if key in self.cache:
            slot, count = self.cache.pop(key)
            self.cache[key] = slot, count
            return None if self.rank != self.encoder_rank else self.cache_values[slot, :count]
        patch_count = run.grid[1] * run.grid[2]
        if patch_count > self.patches:
            raise ValueError("complete media unit exceeds the placed encoder's admitted patch bound")

        def pixels():
            if self.rank == self.encoder_rank:
                self.encoder._retire_inputs()  # Before reusing its receive/input buffer.
            if self.rank == 0:
                self.cpu_pixels = pixel_loader(index, unit)
                value = self.cpu_pixels
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
                        or value.shape != (patch_count, 1536) or not value.is_contiguous()):
                    raise ValueError("ready CPU media pixels differ from their admitted unit")
                self.pixels[:patch_count].copy_(value)
                self.cpu_pixels = None  # Synchronous pageable copy has finished reading it.

        self.context.prepare("media-pixels", pixels, contract=contract)
        if self.encoder_rank != 0:
            if self.rank == 0:
                self.weights.comm.send(self.pixels[:patch_count], self.encoder_rank)
            elif self.rank == self.encoder_rank:
                self.weights.comm.recv(self.pixels[:patch_count], 0)
        count = run.end - run.start

        def encode():
            if self.rank == self.encoder_rank:
                grid = torch.tensor([run.grid], dtype=torch.int64, device="cpu")
                self.unit_output = self.encoder.encode_device_unit(self.pixels[:patch_count], grid).features
                if (self.unit_output.shape != (count, self.hidden) or self.unit_output.dtype != torch.float16
                        or self.unit_output.device != self.device):
                    raise ValueError("placed vision output differs from its complete projected-unit contract")
            if self.resources["cache_slots"]:
                slot = self.free_cache.pop() if self.free_cache else self.cache.popitem(last=False)[1][0]
                if self.rank == self.encoder_rank:
                    self.cache_values[slot, :count].copy_(self.unit_output)
                    self.unit_output = None
                self.cache[key] = slot, count

        self.context.prepare("media-encode", encode, contract=contract)
        if self.rank != self.encoder_rank:
            return None
        return self.cache_values[self.cache[key][0], :count] if key in self.cache else self.unit_output

    def publish(self, slices, *, pixel_loader):
        """Publish a whole scheduled batch; the leader loader must never wait.

        It returns an owned CPU tensor from already completed MediaWork. All
        peers supply the same immutable slice descriptors. No consumer borrows
        a span until this method returns and all transfers are ordered ready.
        """
        try:
            self._check()
            if (type(slices) is not tuple or not 0 < len(slices) <= self.rows
                    or self.rank == 0 and not callable(pixel_loader)
                    or self.rank != 0 and pixel_loader is not None):
                raise ValueError("media batch requires bounded ordered slices and one leader CPU source")
            total = 0
            descriptor = []
            for spec in slices:
                if (type(spec) is not MediaInputSlice or type(spec.plan) is not QwenMediaPlan
                        or spec.plan.profile_identity != self.context.identity
                        or type(spec.generation) is not int or not 0 <= spec.generation < 1 << 63
                        or type(spec.start) is not int or spec.start < 0
                        or type(spec.count) is not int or spec.count <= 0
                        or spec.start > len(spec.plan.token_ids) - spec.count):
                    raise ValueError("media slice differs from its current admitted profile/generation/input")
                total += spec.count
                if total > self.rows:
                    raise ValueError("scheduled media slices exceed their shared input arena")
                ids = spec.plan.token_ids[spec.start:spec.start + spec.count]
                if any(type(value) is not int or not 0 <= value < self.weights.cfg.vocab for value in ids):
                    raise ValueError("media input token IDs differ from the selected vocabulary")
                descriptor.append((spec.plan.media_identity, spec.generation, spec.start, ids))
            contract = hashlib.sha256(json.dumps(descriptor, separators=(",", ":")).encode()).hexdigest()
            # Agree before divergent text/media/cache paths can choose their
            # next phase, and retire every prior reader before arena reuse.
            self.context.prepare("media-batch", self.retire, contract=contract)
            self.spans, self.coordinates = [None] * len(slices), [None] * len(slices)
            with self._stream(), torch.inference_mode():
                offset = 0
                for index, spec in enumerate(slices):
                    start, end = spec.start, spec.start + spec.count
                    view = self.values[offset:offset + spec.count]
                    if self.rank == self.encoder_rank:
                        from .forward import _embed
                        for run in spec.plan.positions.runs:
                            lo, hi = max(start, run.start), min(end, run.end)
                            if run.kind == "text" and lo < hi:
                                count = hi - lo
                                self.ids[:count].copy_(torch.tensor(spec.plan.token_ids[lo:hi], dtype=torch.int64, device="cpu"))
                                _embed(self.weights, self.ids[:count], 1, view[lo - start:hi - start])
                    for unit in spec.plan.units:
                        lo, hi = max(start, unit.run.start), min(end, unit.run.end)
                        if lo < hi:
                            features = self._unit(index, spec, unit, pixel_loader, contract)
                            if self.rank == self.encoder_rank:
                                view[lo - start:hi - start].copy_(features[lo - unit.run.start:hi - unit.run.start])
                            del features
                            self.unit_output = None
                    def coordinates(index=index, spec=spec):
                        host = spec.plan.positions.coordinates(spec.start, spec.count, max_rows=self.rows)
                        self.coordinates[index] = issue_coordinates(host, position=self.position,
                            profile_identity=self.context.identity, generation=spec.generation,
                            media_identity=spec.plan.media_identity, row_start=spec.start, max_rows=self.rows,
                            device=self.device)
                    self.context.prepare("media-coordinates", coordinates, contract=contract)
                    if self.weights.comm is not None:
                        self.weights.comm.broadcast(view, root=self.encoder_rank)
                    offset += spec.count

                def publish():
                    if self.rank == self.encoder_rank:
                        self.encoder._retire_inputs()  # Release completed vision-only metadata before target work.
                    offset = 0
                    for index, spec in enumerate(slices):
                        self.spans[index] = ReadyInputSpan(start=spec.start,
                            token_ids=spec.plan.token_ids[spec.start:spec.start + spec.count],
                            values=self.values[offset:offset + spec.count], positions=self.coordinates[index],
                            position=self.position, rope_delta=spec.plan.positions.rope_delta,
                            profile_identity=self.context.identity, generation=spec.generation,
                            media_identity=spec.plan.media_identity, max_rows=self.rows,
                            hidden_size=self.hidden, vocab_size=self.weights.cfg.vocab, roles=self.roles)
                        offset += spec.count
                self.context.prepare("media-publication", publish, contract=contract)
            return tuple(self.spans)
        except BaseException as primary:
            self._fail(primary)
            raise

    def retire(self, *, cancelled=False):
        """Reuse the arena only after every target/MTP staging reader retires."""
        for span in self.spans:
            if span is not None:
                span.retire(cancelled=cancelled)
        self.spans, self.coordinates = [], []

    def close(self):
        if self.closed:
            return
        if os.getpid() != self.pid:
            raise ValueError("media producer cannot retire its parent's CUDA scope")
        try:
            if self.gpu_started:
                torch.cuda.synchronize(self.device)
            self.retire(cancelled=True)
        except BaseException:
            self.failed = True
            raise
        self.cache.clear()
        self.values = self.ids = self.pixels = self.cache_values = self.unit_output = self.cpu_pixels = self.stream = None
        self.encoder = self.weights = self.context = self.position = None
        self.closed = True
