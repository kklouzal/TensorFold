"""Header-only Qwen MoE CT NVFP4 authority and transactional pool admission."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Callable

from tensorfold.cuda.capacity import Geometry, SIZES, config, headers
from tensorfold.cuda.geometry import gdn_geometry, layer_counts, stream_geometry
from tensorfold.cuda.nvfp4.host_layout import authority_fields_a4, bank_bases_a4, routed_tensor_a4

GIB = 1 << 30
_READ_PIECE = 64 << 20
_READ_ALIGN = 4096
_PAIR_WINDOW = 128


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def check(model_dir, gib, *, tp=1):
    """Validate the public budget before checkpoint IO; this cache owns one GPU."""
    if gib != "auto" and (isinstance(gib, bool) or not isinstance(gib, (int, float))
                          or (isinstance(gib, float) and not math.isfinite(gib)) or gib <= 0):
        raise ValueError("--vram-experts must be auto or a finite positive GiB count")
    if type(tp) is not int or tp != 1:
        raise ValueError("Qwen MoE --vram-experts supports one CUDA GPU (--tp 1)")


def _metadata_bytes(entries):
    # Include arbitrary validated extension metadata recursively. Count shared
    # objects repeatedly; interned keys/values never become an accidental
    # credit. Only scalar sizes escape the observer, never borrowed headers.
    size, pending = 0, [entries]
    while pending:
        value = pending.pop()
        size += sys.getsizeof(value)
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    return size


@dataclass(frozen=True)
class Layout:
    entry_bytes: int
    slots: int
    gpu_bytes: int
    host_bytes: int
    staging_bytes: int
    loading_bytes: int
    authority_allocations: tuple[int, ...]
    banks: tuple[str, ...]
    expert_count: int
    experts_per_bank: int
    runtime_workspace_bytes: int
    metadata_host_bytes: int
    control_host_bytes: int
    source_filesystem_bytes: int
    header_bytes: int
    largest_header_bytes: int
    automatic: bool
    format: str = "ct-nvfp4-a4"

    def routed_tensor(self, name):
        """Only fields of the banks completely validated at this boundary."""
        return routed_tensor_a4(name) and name.split(".experts.", 1)[0] in self.banks

    def transform(self, resident):
        """Remove CPU authority only; shared experts, router and backbone stay resident."""
        banks = frozenset(self.banks)
        def weights(name, info):
            routed = routed_tensor_a4(name) and name.split(".experts.", 1)[0] in banks
            return (0, 0) if routed else resident(name, info)
        return weights

    def receipt(self):
        return {"format": self.format, "entry_bytes": self.entry_bytes, "slots": self.slots,
                "gpu_bytes": self.gpu_bytes, "host_authority_bytes": self.host_bytes,
                "pinned_staging_payload_bound_bytes": self.staging_bytes,
                "loading_bytes": self.loading_bytes, "header_metadata_bound_bytes": self.metadata_host_bytes,
                "cache_control_host_bound_bytes": self.control_host_bytes,
                "prepared_activation_calibration_bytes": 12 * self.expert_count,
                "source_filesystem_working_bytes": self.source_filesystem_bytes,
                "safetensors_header_bytes": self.header_bytes,
                "largest_safetensors_header_bytes": self.largest_header_bytes,
                "native_runtime_workspace_bytes": self.runtime_workspace_bytes,
                "logical_expert_count": self.expert_count, "banks": len(self.banks),
                "automatic": self.automatic, "capturable": False}


def layout(model_dir: str | Path, gib: float | str, *, tp=1) -> Layout:
    """Validate original CT expert fields and size the final six-field authority.

    The shared expert remains a separately resident native CT operation. Every
    weight/input global contributes FP32 native calibration authority under
    CT's reciprocal convention, never ModelOpt calibration or BF16 metadata.
    """
    check(model_dir, gib, tp=tp)
    text = config(model_dir)
    for name in ("hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok", "num_hidden_layers"):
        _integer(text.get(name), name, 1)
    if text["num_experts_per_tok"] > text["num_experts"]:
        raise ValueError("num_experts_per_tok must not exceed num_experts")
    if text["num_experts"] > 1023:
        raise ValueError("Qwen MoE routing supports at most 1023 routed experts plus shared")
    dims, width, count = text["hidden_size"], text["moe_intermediate_size"], text["num_experts"]
    fields = authority_fields_a4(count, width, dims)
    metadata_objects = header_bytes = largest_header = 0

    def observe(offset, entries):
        nonlocal metadata_objects, header_bytes, largest_header
        metadata_objects += _metadata_bytes(entries)
        header_bytes += offset
        largest_header = max(largest_header, offset)

    entries = headers(model_dir, observer=observe)
    banks = bank_bases_a4(entries, text)
    allocations = tuple(math.prod(shape) * SIZES[dtype] for shape, dtype in fields)
    one = authority_fields_a4(1, width, dims)
    entry = 0
    for shape, dtype in one:
        entry = -(-entry // 16) * 16 + math.prod(shape) * SIZES[dtype]
    entry = -(-entry // 16) * 16
    logical = count * len(banks)
    requested = entry if gib == "auto" else (gib * GIB if type(gib) is int else
                                             gib.as_integer_ratio()[0] * GIB // gib.as_integer_ratio()[1])
    slots = min(logical, requested // entry)
    if not slots:
        raise ValueError(f"--vram-experts needs at least one native CT expert ({entry / GIB:.6f} GiB)")
    # The single (2,entry) Torch pinned allocation rounds within its ceil-log2
    # bucket on the pinned runtime, including configured nonrounded blocks.
    staging = 1 << (2 * entry - 1).bit_length()
    # Reader's two (64MiB+3*alignment) pinned blocks each round to 128MiB.
    # Buffered fallback's current 64MiB filesystem window is reclaimable; no
    # claim that the complete source checkpoint must stay resident is made.
    reader = 2 * (1 << (_READ_PIECE + _READ_ALIGN - 1).bit_length())
    loading = max(64 << 20, 8 * entry) + reader
    # Decoded raw, normalized capacity and loader dictionaries can coexist.
    # Three complete decoded inventories conservatively also cover parser
    # object-pair lists and span-sort work (their objects/keys are recounted).
    # On pinned CPython the raw bytes and decoded JSON input coexist, with
    # Unicode storage at most four bytes per input code point. Wire padding
    # is charged from observed offsets, including otherwise-filtered metadata.
    metadata = (3 * metadata_objects + 5 * largest_header
                + sys.getsizeof(b"") + sys.getsizeof("\U00010000"))
    # Histories, resident identities/NumPy rank arrays, per-bank views and the
    # bounded startup rank search are charged at the largest possible pool.
    # Three derived FP32 activation dequantizers per expert are prepared once
    # beside the original authority, including for the baseline CPU adapter.
    controls = (logical + 1 + 588 * logical + 4096 * len(banks) + 65536
                + 192 * _PAIR_WINDOW * (text["num_experts_per_tok"] + 1) + 65536)
    runtime = 16 * _PAIR_WINDOW * (dims + width) + 65536
    return Layout(entry, slots, slots * entry, sum(allocations) * len(banks), staging, loading,
                  allocations * len(banks), banks, logical, count, runtime, metadata, controls,
                  _READ_PIECE, header_bytes, largest_header, gib == "auto")


def _moe_buffers(rows, experts, slots, width, dims, prefill):
    """All retained power-of-two MoEBuffers through rows, including unused plan arrays.

    This mirrors owned cuda.moe.MoEBuffers/grouped.Plan allocation geometry,
    without importing Torch or a native extension during header admission.
    The native A4 bank does not consume the grouped plan, but the current
    common buffer constructor still allocates it. Decode and prompt modes
    have separate retained keys, even when their row buckets coincide.
    """
    total, bucket = 0, 16
    maximum = 1 << max(4, (rows - 1).bit_length())
    while bucket <= maximum:
        pairs = bucket * slots
        wide = pairs > 1024
        total += bucket * (4 * experts + 8 * slots + 2 * slots * width
                           + (2 if prefill else 4) * slots * dims)
        total += 4 * pairs + 12 * (min(pairs, experts) + pairs // 16) + 8
        total += 4 * (pairs if wide else 1)
        total += 4 * (((pairs + 1023) // 1024) * experts if wide else 1)
        bucket *= 2
    return total


@dataclass(frozen=True)
class RuntimeGeometry:
    """One future-only reservation, used unchanged by admission and pool sizing.

    ``cache_slots`` includes the widest verification window. All categories
    remain reserved before calibration; an allocator's cached/free block is
    not evidence that later state/workspace has already been materialized.
    """
    geometry: Geometry
    _components: Callable[[int], dict]

    def components(self, cache_slots):
        return self._components(_integer(cache_slots, "cache_slots"))


def runtime_geometry(text, *, streams, keep, max_rows, prompt_rows) -> RuntimeGeometry:
    """Bound eager target state, full future KV and retained activation shapes.

    Single-stream clones share their attention list. Concurrent private and
    kept states may retain independent full-size KV allocations, so their
    initial 256-row allocation is never used as a future-growth credit.
    Recurrent copies include current/commit plus every retained prefix. The
    shared dense geometry supplies record and operator workspace bounds;
    explicit MoE buffers and native W4A4 temporary arrays are additional.
    Secondary-model weights/KV/workspace belong to its separate geometry.
    """
    _integer(streams, "streams", 1)
    _integer(keep, "keep")
    _integer(max_rows, "max_rows", 1)
    _integer(prompt_rows, "prompt_rows", 512)
    if max_rows > 128:
        raise ValueError("max_rows must be an integer in 1..128")
    t = dict(text)
    for name in ("hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok",
                 "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size",
                 "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                 "linear_value_head_dim", "linear_conv_kernel_dim"):
        _integer(t.get(name), name, 1)
    if t["num_experts_per_tok"] > t["num_experts"] or t["num_experts"] > 1023:
        raise ValueError("routing needs top-k <= num_experts <= 1023")
    if t.get("head_dim") is None:
        if t["hidden_size"] % t["num_attention_heads"]:
            raise ValueError("hidden_size must divide evenly into attention heads when head_dim is absent")
        t["head_dim"] = t["hidden_size"] // t["num_attention_heads"]
    _integer(t["head_dim"], "head_dim", 1)
    if "layer_types" in t:
        kinds = t["layer_types"]
        if not isinstance(kinds, (list, tuple)) or len(kinds) != t["num_hidden_layers"] or any(
                kind not in ("linear_attention", "full_attention") for kind in kinds):
            raise ValueError("layer_types must describe every linear_attention/full_attention layer")
    else:
        _integer(t.get("full_attention_interval", 4), "full_attention_interval", 1)
    d, width, slots = t["hidden_size"], t["moe_intermediate_size"], t["num_experts_per_tok"] + 1
    shared = _integer(t.get("shared_expert_intermediate_size", width), "shared_expert_intermediate_size", 1)
    # Dense prompt bounds require intermediate_size; routed/shared activations
    # coexist, so use every ordered slot's largest declared intermediate.
    t["intermediate_size"] = slots * max(width, shared)
    t["moe_intermediate_size"] = max(width, shared)
    rows = max(16, max_rows) * streams  # includes MultiDecoder's 16-row/lane calibration grid
    for count in (rows, prompt_rows):
        if (1 << max(4, (count - 1).bit_length())) * slots > (1 << 31) - 1:
            raise ValueError("MoE row/slot geometry exceeds the signed-int32 grouped-plan boundary")
    linear, attention = layer_counts(t)
    nk, nv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    dk, dv, conv = t["linear_key_head_dim"], t["linear_value_head_dim"], t["linear_conv_kernel_dim"]
    state = linear * (nv * dk * dv * 4 + (conv - 1) * (2 * nk * dk + nv * dv) * 2)
    kv_row = t["num_key_value_heads"] * t["head_dim"] * 4
    single = gdn_geometry(t, 1, max_rows, rows=rows, prompt=prompt_rows, evicts=True)
    workspace = single.bytes_at(0) - 4 * state - (attention + 1) * 1024 * kv_row
    copies = keep + 2
    if streams > 1:
        copies = 2 * streams + keep + 2
        base = stream_geometry(t, 1, streams, keep, first=256)
        workspace = max(workspace, base.bytes_at(0) - copies * state)
    workspace += _moe_buffers(rows, t["num_experts"] + 1, slots, width, d, False)
    workspace += _moe_buffers(prompt_rows, t["num_experts"] + 1, slots, width, d, True)
    workspace += 16 * _PAIR_WINDOW * (d + max(width, shared)) + 65536

    def components(capacity):
        # grow() may start at 1024 rows. reserve/private() then cap future
        # growth at State.limit, which is the admitted cache-slot capacity.
        active = max(1024, capacity)
        live = (streams * attention + 1) * active * kv_row
        snapshots = copies * state
        if streams > 1:
            snapshots += (keep + 1) * attention * active * kv_row
        scratch = rows * t["num_attention_heads"] * (t["head_dim"] + 2) * ((capacity + 511) // 512) * 4
        return {"future_kv_bytes": live, "future_snapshot_bytes": snapshots,
                "future_target_workspace_bytes": workspace + scratch}

    def bytes_at(capacity):
        return sum(components(_integer(capacity, "cache_slots")).values())

    return RuntimeGeometry(Geometry(bytes_at, max_rows), components)


def plan_pool(plan: Layout, *, raw_free_bytes: int, existing_pool_bytes: int,
              future_kv_bytes: int, future_snapshot_bytes: int, future_target_workspace_bytes: int,
              future_draft_workspace_bytes: int, device_reserve_bytes: int,
              allocation_size: Callable[[int], int]) -> dict:
    """Select a startup pool after fixed target/draft/service allocations.

    ``raw_free_bytes`` already excludes the backbone, prepared buffers and live
    initial pool. Only FUTURE allocations belong in the named reservations.
    ``allocation_size`` is the preparation owner's nondecreasing allocator
    storage upper bound, proved for its active pinned runtime/configuration;
    this module never guesses a CUDA segment size. A replacement coexists with
    the old pool until commit, so old storage is not credited to transaction
    free space. Reusing an existing maximum-fitting pool allocates nothing.
    """
    names = ("raw_free_bytes", "existing_pool_bytes", "future_kv_bytes", "future_snapshot_bytes",
             "future_target_workspace_bytes", "future_draft_workspace_bytes", "device_reserve_bytes")
    values = (raw_free_bytes, existing_pool_bytes, future_kv_bytes, future_snapshot_bytes,
              future_target_workspace_bytes, future_draft_workspace_bytes, device_reserve_bytes)
    for name, value in zip(names, values):
        _integer(value, name)
    if existing_pool_bytes % plan.entry_bytes or existing_pool_bytes > plan.expert_count * plan.entry_bytes:
        raise ValueError("existing expert pool must contain whole supported expert cells")
    if not callable(allocation_size):
        raise ValueError("pool sizing requires a proved allocator storage bound")
    target = max(future_target_workspace_bytes, plan.runtime_workspace_bytes)
    reserved = future_kv_bytes + future_snapshot_bytes + target + future_draft_workspace_bytes + device_reserve_bytes
    room = raw_free_bytes - reserved
    if room < 0:
        raise ValueError("mandatory KV/state/draft/native workspace exceeds remaining CUDA memory")

    def allocation(cells):
        requested = cells * plan.entry_bytes
        result = allocation_size(requested)
        if type(result) is not int or result < requested:
            raise ValueError("allocator bound must return integer storage no smaller than its request")
        return result

    limit = plan.expert_count if plan.automatic else plan.slots
    low, high = 0, limit
    while low < high:
        middle = (low + high + 1) // 2
        if allocation(middle) <= room:
            low = middle
        else:
            high = middle - 1
    existing = existing_pool_bytes // plan.entry_bytes
    reuse = existing > 0 and existing >= low and (plan.automatic or existing == limit)
    cells = existing if reuse else low
    if not cells or (not plan.automatic and cells != limit):
        raise ValueError("CUDA remaining memory cannot fit the requested expert pool and mandatory workspace")
    requested = cells * plan.entry_bytes
    allocated = 0 if reuse else allocation(cells)
    return {"policy": "qwen-ct-nvfp4-after-fixed-state-v1", "raw_free_bytes": raw_free_bytes,
            "existing_pool_bytes": existing_pool_bytes, "existing_pool_not_credited": True,
            "future_kv_bytes": future_kv_bytes, "future_snapshot_bytes": future_snapshot_bytes,
            "future_target_workspace_bytes": target, "future_draft_workspace_bytes": future_draft_workspace_bytes,
            "device_reserve_bytes": device_reserve_bytes, "reserved_bytes": reserved,
            "slots": cells, "requested_pool_bytes": requested, "new_allocation_bound_bytes": allocated,
            "replace_pool": not reuse, "transaction_free_after_reservations_bytes": room - allocated,
            "final_free_after_reservations_bytes": room - allocated + (0 if reuse else existing_pool_bytes),
            "snapshot_is_reservation": False}
