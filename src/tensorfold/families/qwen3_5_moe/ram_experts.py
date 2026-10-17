"""Header-only Qwen MoE CT NVFP4 authority and transactional pool admission."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Callable

from tensorfold.cuda.capacity import SIZES, config, headers
from tensorfold.cuda.nvfp4.host_layout import authority_fields_a4, bank_bases_a4, routed_tensor_a4
from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES

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
                          or not math.isfinite(gib) or gib <= 0):
        raise ValueError("--vram-experts must be auto or a finite positive GiB count")
    if type(tp) is not int or tp != 1:
        raise ValueError("Qwen MoE --vram-experts supports one CUDA GPU (--tp 1)")


def _metadata_bytes(entries):
    # Header decoding, normalized capacity metadata and reader metadata can
    # coexist. Count shared objects repeatedly, so interned names/keys do not
    # turn into an accidental credit. The wire header can contain padding up
    # to the safetensors protocol cap, independently of decoded object sizes.
    size = sys.getsizeof(entries)
    for name, info in entries.items():
        size += sys.getsizeof(name) + sys.getsizeof(info)
        for key, value in info.items():
            size += sys.getsizeof(key) + sys.getsizeof(value)
            if isinstance(value, (tuple, list)):
                size += sum(sys.getsizeof(item) for item in value)
    return 3 * size + MAX_HEADER_BYTES


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
                "source_filesystem_working_bytes": self.source_filesystem_bytes,
                "native_runtime_workspace_bytes": self.runtime_workspace_bytes,
                "logical_expert_count": self.expert_count, "banks": len(self.banks),
                "automatic": self.automatic, "capturable": False}


def layout(model_dir: str | Path, gib: float | str, *, tp=1) -> Layout:
    """Validate original CT expert fields and size the final six-field authority.

    The shared expert remains a separately resident native CT operation. Every
    weight/input global is preserved as FP32 authority, never reconstructed
    from ModelOpt calibration or converted through BF16.
    """
    check(model_dir, gib, tp=tp)
    text, entries = config(model_dir), headers(model_dir)
    for name in ("hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok", "num_hidden_layers"):
        _integer(text.get(name), name, 1)
    if text["num_experts_per_tok"] > text["num_experts"]:
        raise ValueError("num_experts_per_tok must not exceed num_experts")
    if text["num_experts"] > 1023:
        raise ValueError("Qwen MoE routing supports at most 1023 routed experts plus shared")
    dims, width, count = text["hidden_size"], text["moe_intermediate_size"], text["num_experts"]
    banks = bank_bases_a4(entries, text)
    fields = authority_fields_a4(count, width, dims)
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
    # Reader's two (64MiB+alignment) pinned pieces each round to 128MiB.
    # Buffered fallback's current 64MiB filesystem window is reclaimable; no
    # claim that the complete source checkpoint must stay resident is made.
    reader = 2 * (1 << (_READ_PIECE + _READ_ALIGN - 1).bit_length())
    loading = max(64 << 20, 8 * entry) + reader
    metadata = _metadata_bytes(entries)
    # Histories, resident identities/NumPy rank arrays, per-bank views and the
    # bounded startup rank search are charged at the largest possible pool.
    controls = logical + 1 + 576 * logical + 4096 * len(banks) + 65536
    runtime = 16 * _PAIR_WINDOW * (dims + width) + 65536
    return Layout(entry, slots, slots * entry, sum(allocations) * len(banks), staging, loading,
                  allocations * len(banks), banks, logical, count, runtime, metadata, controls,
                  _READ_PIECE, gib == "auto")


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
