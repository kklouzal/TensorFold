"""Header-only CUDA startup estimates and a shared rank capacity decision."""

from __future__ import annotations

import math
import os
import re
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
# safetensors dtype names -> bytes a value (FP8: the FP4 checkpoints' block scales)
SIZES = {"U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
         "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "U32": 4, "I32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
LIMIT_ENV = "TENSORFOLD_CUDA_MEMORY_LIMIT_GB"


def itemsize(info: dict, name: str) -> int:
    """Bytes a value for one tensor header; an unknown dtype names the tensor instead of raising a bare KeyError."""

    dtype = info.get("dtype")
    item = SIZES.get(dtype) if isinstance(dtype, str) else None
    if item is None:
        raise ValueError(f"checkpoint tensor {name} has dtype {dtype!r}, which the startup estimate cannot "
                         f"size; known dtypes: {', '.join(sorted(SIZES))}")
    return item


@dataclass(frozen=True)
class Weights:
    resident: int
    staging: int
    mapped: int = 0  # read-only, unpinned file pages, reclaimable by the OS


@dataclass(frozen=True)
class Geometry:
    """Allocated cache, recurrence and bounded scratch bytes at a cache-slot capacity."""

    bytes_at: Callable[[int], int]
    reserve: int
    minimum_slots: int = 0
    breakdown_at: Callable[[int], dict[str, int]] | None = None

    def needed(self, window: int) -> int:
        return int(self.bytes_at(max(self.minimum_slots, window + self.reserve)))


def fit_sessions(window: int, budget: int, weights: Weights,
                 geometry: Callable[[int], Geometry], *, ceiling: int | None = None,
                 additional_fits: Callable[[int], bool] | None = None) -> int:
    """Largest full-window session count for a monotone, exact rank geometry.

    Appending a slot must preserve preceding owners and cannot release their
    storage. Every session owns at least one byte; the usable byte budget is
    therefore a physical search bound, not an execution-bucket/session cap.
    Loading overlaps serving only as described by the existing Weights plan.
    """
    if type(window) is not int or window <= 0 or type(budget) is not int or budget < 0:
        raise ValueError("session planning needs a positive context and nonnegative integer byte budget")
    if ceiling is not None and (type(ceiling) is not int or ceiling < 1):
        raise ValueError("session ceiling must be a positive integer or automatic")
    if any(type(n) is not int or n < 0 for n in (weights.resident, weights.staging, weights.mapped)):
        raise ValueError("session weight estimates must be nonnegative integer bytes")
    if weights.resident + weights.staging > budget:
        return 0
    bound = min(budget - weights.resident, ceiling) if ceiling is not None else budget - weights.resident

    def fits(count):
        cost = geometry(count).needed(window)
        if type(cost) is not int or cost < count:
            raise ValueError("session geometry must retain at least one byte per session")
        return weights.resident + cost <= budget and (additional_fits is None or additional_fits(count))

    low, high = 0, min(1, bound)
    while high and fits(high):
        low = high
        if high == bound:
            return high
        high = min(bound, high * 2)
    while low < high:
        count = (low + high + 1) // 2
        if fits(count):
            low = count
        else:
            high = count - 1
    return low


@dataclass(frozen=True)
class Plan:
    native: int
    requested: int | None
    explicit: bool
    fitting: int
    budget: int
    weights: Weights
    geometry: Geometry
    keeps_tables: bool | None = None   # the window leaves the mapped tables their pages (None: nothing to keep)
    largest: int = 0                   # the largest window the budget fits up to the native one: what a restart gets
    resident: int = 0                  # the largest window that leaves the mapped tables their pages

    @property
    def settings(self) -> list[int]:
        return [self.native, -1 if self.requested is None else self.requested, int(self.explicit)]

    def receipt(self, window: int) -> dict:
        receipt = {"native_window": self.native, "requested_context": self.requested,
                "explicit_context": self.explicit, "context_window": window,
                "cache_slots": max(self.geometry.minimum_slots, window + self.geometry.reserve),
                "budget_bytes": self.budget, "weight_bytes_estimate": self.weights.resident,
                "loading_bytes_estimate": self.weights.staging, "mapped_table_bytes": self.weights.mapped,
                "cache_workspace_bytes_estimate": self.geometry.needed(window),
                "startup_peak_bytes_estimate": self.weights.resident + self.weights.staging,
                "serving_peak_bytes_estimate": self.weights.resident + self.geometry.needed(window),
                "total_bytes_estimate": self.weights.resident + max(self.weights.staging, self.geometry.needed(window)),
                "full_mapped_working_set_bytes_estimate": self.weights.resident +
                max(self.weights.staging, self.geometry.needed(window)) + self.weights.mapped,
                "mapped_pages_reclaimable": True, "mapped_tables_resident": self.keeps_tables}
        if self.geometry.breakdown_at is not None:
            receipt["resource_ledger_bytes"] = self.geometry.breakdown_at(
                max(self.geometry.minimum_slots, window + self.geometry.reserve))
        return receipt


def config(model_dir: str | Path) -> dict:
    from tensorfold.checkpoint_metadata import read_config

    raw = read_config(model_dir)
    selected = raw.get("text_config")
    if selected is not None and not isinstance(selected, dict):
        raise ValueError("checkpoint text_config must be an object or null")
    text = dict(selected or raw)
    if raw.get("model_type") in ("qwen3_5", "qwen3_5_moe"):
        from tensorfold.families.qwen3_5.config_metadata import resolve_layer_types, validate_cuda_metadata

        validate_cuda_metadata(raw)
        if text.get("layer_types") is not None:
            text["layer_types"] = list(resolve_layer_types(text, layers=int(text["num_hidden_layers"])))
        else:
            text.pop("layer_types", None)  # generic geometry uses the arithmetic interval pattern
    for name in ("quantization", "quantization_config"):
        if raw.get(name) is not None and not isinstance(raw[name], dict):
            raise ValueError(f"checkpoint {name} metadata must be an object or null")
    quantization = raw.get("quantization") or raw.get("quantization_config") or {}
    text["_quantization"] = quantization
    if quantization.get("quant_method") == "gguf":
        text["_replicate_kv_heads"] = True
    return text


def headers(model_dir: str | Path, *, rank: int | None = None, files: list[Path] | None = None,
            observer: Callable[[int, dict], None] | None = None) -> dict:
    """Inspect one operation's checkpoint metadata without caching its namespace.

    Indexed targets are authorized before opening headers, resolved again once
    per distinct shard for the mapping, and checked before publication. The
    reader's input-immutability contract covers this metadata operation; these
    checks detect observed namespace changes, not every transient filesystem
    mutation. No resolved path survives this invocation.

    ``observer`` runs once per completely validated raw header, before its
    entries are copied/filtered. Its offset includes the eight-byte prefix.
    The callback may inspect borrowed entries (including ``__metadata__``),
    but must neither retain nor mutate them. It must own its own accumulated
    scalar result; headers never stores the callback beyond this operation.
    """
    from .tensor_file import checkpoint_path, read_header, read_metadata_json

    path = Path(model_dir)
    if observer is not None and not callable(observer):
        raise ValueError("header observer must be callable or null")
    if rank is not None and (type(rank) is not int or rank < 0):
        raise ValueError("checkpoint rank must be a nonnegative integer or null")
    files = list(files or []) or (sorted(path.glob(f"*.rank{rank}.safetensors")) if rank is not None else [])
    mapping = None
    if not files:
        other = sorted(path.glob("*.rank*.safetensors"))
        if other:
            raise ValueError("checkpoint contains another rank's split weights; use this rank's folder")
        index = checkpoint_path(path, "model.safetensors.index.json")
        if index.exists():
            mapping = read_metadata_json(index).get("weight_map")
            if (not isinstance(mapping, dict) or not mapping
                    or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                           for k, v in mapping.items())):
                raise ValueError("checkpoint index must contain a nonempty string-to-string weight_map")
            files = [path / n for n in sorted(set(mapping.values()))]
        else:
            files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError("startup memory estimate needs the checkpoint tensor headers")
    if mapping is not None:
        for shard in sorted(set(mapping.values())):
            checkpoint_path(path, shard)
    authorized_files = []
    lexical_root = Path(os.path.abspath(path))
    for file in files:
        file = Path(file)
        absolute = Path(os.path.abspath(file))
        try:
            relative = absolute.relative_to(lexical_root)
        except ValueError as error:
            try:
                relative = absolute.relative_to(path.resolve())
            except ValueError:
                raise ValueError(f"{file}: explicit checkpoint file must lie in the model directory") from error
        authorized = checkpoint_path(path, str(relative))
        authorized_files.append((file, authorized))
    # Prove every indexed relative target before opening any payload header.
    # Explicit files retain their lexical names for the rank/split contract.
    out, located = {}, {}
    for file, authorized in authorized_files:
        offset, entries = read_header(authorized, SIZES)
        if observer is not None:
            observer(offset, entries)
        for name, info in entries.items():
            if name == "__metadata__":
                continue
            if name in out:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            out[name] = {**info, "split": ".rank" in file.name}
            located[name] = authorized
    if mapping is not None and len(mapping) <= 2 * len(authorized_files):
        # Small indexes do not amortize two distinct-shard walks; keep their
        # direct original checks rather than adding cache state and IO.
        for name, shard in mapping.items():
            if located.get(name) != checkpoint_path(path, shard):
                raise ValueError(f"{name}: checkpoint index does not point to its declared tensor shard")
    elif mapping is not None:
        targets = {}
        for name, shard in mapping.items():
            target = targets.get(shard)
            if target is None:
                target = targets[shard] = checkpoint_path(path, shard)
            if located.get(name) != target:
                raise ValueError(f"{name}: checkpoint index does not point to its declared tensor shard")
        # Recheck after the full mapping walk so a persistent retarget during
        # that walk cannot publish a stale operation-local authorization.
        for shard, target in targets.items():
            if checkpoint_path(path, shard) != target:
                raise ValueError(f"{shard}: checkpoint target changed during header inspection")
    return out


def estimate_weights(model_dir: str | Path, transform: Callable, *, rank: int | None = None,
                     files: list[Path] | None = None) -> Weights:
    return _estimate_weights_from_headers(headers(model_dir, rank=rank, files=files), transform)


def _estimate_weights_from_headers(entries: dict, transform: Callable) -> Weights:
    """The generic estimate from metadata already validated by ``headers``."""

    layers: dict[str, int] = {}
    resident = mapped = largest = 0
    for name, info in entries.items():
        size, host = transform(name, info)
        size, host = int(size), int(host)
        if min(size, host) < 0:
            raise ValueError("negative startup weight estimate")
        resident += size
        mapped += host
        largest = max(largest, size)
        match = re.search(r"(?:layers|blocks)\.(\d+)\.", name)
        group = match.group(1) if match else name
        layers[group] = layers.get(group, 0) + size
    # CPU expert lists/stack, GPU uploads and tiled outputs can coexist during one layer load.
    staging = 3 * max([largest, *layers.values()], default=0)
    return Weights(resident, staging, mapped)


def _meminfo() -> dict | None:
    try:
        rows = Path("/proc/meminfo").read_text().splitlines()
        memory = {key.rstrip(":"): int(value) * 1024 for key, value, *_ in (row.split() for row in rows)}
    except (OSError, ValueError):
        return None
    return memory if {"MemTotal", "MemAvailable"} <= memory.keys() else None


def unified(torch) -> bool:
    """A GPU on the host's memory (GB10): its free figure is MemFree, which counts the page cache as used."""

    try:
        return bool(torch.cuda.get_device_properties(0).is_integrated)
    except (AttributeError, AssertionError, RuntimeError):
        return False


def reserve_bytes(total: int, *, host: bool = False) -> int:
    """Memory the startup keeps free: max(4 GiB, a tenth), or TENSORFOLD_MEMORY_RESERVE_GIB (>= 2).

    Preserve host integer-floor and device ceiling arithmetic from the fork."""

    value = os.environ.get("TENSORFOLD_MEMORY_RESERVE_GIB", "").strip()
    if not value:
        return max(4 * GIB, total // 10 if host else math.ceil(total / 10))
    try:
        gib = float(value)
    except ValueError:
        gib = math.nan
    if not 2 <= gib <= total / GIB:
        raise ValueError(f"TENSORFOLD_MEMORY_RESERVE_GIB={value}: a number of GiB from 2 to the memory's size")
    return int(gib * GIB)


def cuda_reserve_bytes(total: int, *, default: int | None = None) -> int:
    """Discrete CUDA's floor; an explicit device override may be zero.

    Unset preserves the legacy global reserve, or a caller's existing explicit
    default. This policy never changes host/unified memory reservations.
    """
    if type(total) is not int or total < 0:
        raise ValueError("CUDA memory size must be nonnegative integer bytes")
    value = os.environ.get("TENSORFOLD_CUDA_MEMORY_RESERVE_GIB")
    if value is None:
        if default is not None and (type(default) is not int or not 0 <= default <= total):
            raise ValueError("default CUDA memory reserve must be integer bytes within device memory")
        return reserve_bytes(total) if default is None else default
    try:
        gib = float(value)
    except ValueError:
        gib = math.nan
    if not math.isfinite(gib) or gib < 0:
        raise ValueError(f"TENSORFOLD_CUDA_MEMORY_RESERVE_GIB={value}: finite GiB from 0 to device memory size")
    numerator, denominator = gib.as_integer_ratio()
    scaled = numerator * GIB
    if scaled > total * denominator:
        raise ValueError(f"TENSORFOLD_CUDA_MEMORY_RESERVE_GIB={value}: finite GiB from 0 to device memory size")
    return scaled // denominator


def host_stream_reserve_bytes(total: int) -> int:
    """Existing host staging reserve, shared by global and NUMA admission."""
    if type(total) is not int or total <= 0:
        raise ValueError("host physical memory must be positive integer bytes")
    return reserve_bytes(total, host=True) if os.environ.get("TENSORFOLD_MEMORY_RESERVE_GIB", "").strip() else 2 * GIB


def host_stream_bytes() -> int | None:
    """Host staging room, with a 2-GiB default reserve or the explicit startup reserve override."""

    memory = _meminfo()
    if memory is None:
        return None
    reserve = host_stream_reserve_bytes(memory["MemTotal"])
    return max(0, memory["MemAvailable"] - reserve)


def host_memory_snapshot() -> dict:
    """Advisory startup observation, never an allocation grant or reservation.

    A telemetry read failure remains visible without turning a loaded model's
    diagnostic into another admission decision. Cancellation still propagates.
    """
    memory = _meminfo()
    error = None
    try:
        room = host_cgroup_bytes()
    except (OSError, ValueError) as failure:
        room, error = None, str(failure)
    return {"mem_available_bytes": None if memory is None else memory["MemAvailable"],
            "cgroup_remaining_bytes": room, "cgroup_read_error": error,
            "snapshot_is_reservation": False}


def host_cgroup_bytes() -> int | None:
    """Smallest remaining cgroup-v2 hard allowance along this process's visible ancestry.

    A private container cgroup namespace reports ``/``; ordinary host processes
    can belong to a nested systemd/job cgroup. Ancestors account for all their
    descendants, so another child can consume the limiting ancestor's room.
    No exposed memory controller returns None; an exposed limit must be readable.
    """

    root = Path("/sys/fs/cgroup")
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    paths = [row[3:] for row in memberships if row.startswith("0::")]
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError("invalid cgroup-v2 process membership")
    relative = Path(paths[0])
    if not relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid cgroup-v2 process membership")
    current = root.joinpath(*relative.parts[1:])
    allowance = None
    while True:
        try:
            limit = (current / "memory.max").read_text().strip()
        except FileNotFoundError:
            limit = "max"                      # no memory controller here (including the real hierarchy root)
        except OSError as exc:
            raise ValueError(f"cannot read cgroup-v2 memory limit at {current}: {exc}") from exc
        if limit != "max":
            try:
                used = (current / "memory.current").read_text().strip()
            except OSError as exc:
                raise ValueError(f"cannot read cgroup-v2 memory usage at {current}: {exc}") from exc
            if not limit.isdecimal() or not used.isdecimal():
                raise ValueError(f"invalid cgroup-v2 memory allowance at {current}")
            remaining = max(0, int(limit) - int(used))
            allowance = remaining if allowance is None else min(allowance, remaining)
        if current == root:
            return allowance
        current = current.parent


def cuda_limit_bytes(environ: Mapping[str, str] | None = None) -> int | None:
    """The CUDA admission budget's explicit GiB cap in bytes, or None when unset.

    ``TENSORFOLD_CUDA_MEMORY_LIMIT_GB`` caps the grant the same absolute way ``TENSORFOLD_MEMORY_LIMIT_GB``
    caps the MLX budget. ValueError, naming the variable, for a nonpositive, non-finite, or non-numeric value.
    """

    value = (os.environ if environ is None else environ).get(LIMIT_ENV)
    if value is None:
        return None
    try:
        gib = float(value)
    except ValueError:
        raise ValueError(f"{LIMIT_ENV} must be a positive number in GiB") from None
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError(f"{LIMIT_ENV} must be a positive number in GiB")
    numerator, denominator = gib.as_integer_ratio()
    # GiB is a power of two: this is the same truncation without overflowing
    # the float intermediate for large, otherwise valid finite caps.
    return numerator * GIB // denominator


def available_bytes(torch, *, reserve: int | None = None) -> int:
    """What admission and the runtime gate read as live: the pool's free memory less its floor, under the explicit cap."""

    free, total = map(int, torch.cuda.mem_get_info())
    if reserve is not None and (type(reserve) is not int or not 0 <= reserve <= total):
        raise ValueError("explicit CUDA memory reserve must be integer bytes within device memory")
    integrated = unified(torch)
    memory = _meminfo() if integrated else None
    if memory is not None:
        granted = memory["MemAvailable"] - (reserve_bytes(memory["MemTotal"], host=True) if reserve is None else reserve)     # one pool: page cache counts as free
    elif integrated:
        # Missing /proc data retains the unified fallback's legacy floor;
        # a discrete-only override must not lower a shared host reservation.
        granted = free - (reserve_bytes(total) if reserve is None else reserve)
    else:
        granted = free - cuda_reserve_bytes(total, default=reserve)        # a discrete card (or no /proc/meminfo): the floor comes off the card
    limit = cuda_limit_bytes()
    return max(0, min(granted, limit)) if limit is not None else max(0, granted)


def total_bytes(torch) -> int:
    """The GPU's memory (a GB10's is the host's): the same on every rank, so what it sizes agrees without a gather."""

    return int(torch.cuda.mem_get_info()[1])


def page_room(torch) -> int | None:
    """What caches and mapped read-only tables share on a unified GPU (MemAvailable); None on a discrete GPU."""

    memory = _meminfo()
    return memory["MemAvailable"] if memory is not None and unified(torch) else None


def make_plan(native: int, requested: int | None, explicit: bool, budget: int,
              weights: Weights, geometry: Geometry, room: int | None = None) -> Plan:
    native = int(native)
    requested = None if requested is None else int(requested)
    if requested is not None and requested < 0:
        raise ValueError("context must be 0 or a positive token count")
    target = requested if requested else native
    if target <= 0:
        raise ValueError("checkpoint has no native window; give an explicit positive --context")
    upper = min(target, native) if native > 0 else target

    def fit(ceiling: int, top: int = upper) -> int:
        low, high = 0, 0 if weights.resident + weights.staging > budget else top
        while low < high:
            middle = (low + high + 1) // 2
            if weights.resident + geometry.needed(middle) <= ceiling:
                low = middle
            else:
                high = middle - 1
        return low

    fitting, keeps, resident = fit(budget), None, 0
    if weights.mapped and room is not None:
        # windows up to ``resident`` keep mapped tables in the page cache; past it they page
        resident = fit(min(budget, room - weights.mapped))
        if explicit:
            keeps = 0 < resident >= upper
        else:
            fitting, keeps = (resident, True) if resident else (fitting, False)
    largest = fit(budget, native if native > 0 else target)
    return Plan(native, requested, bool(explicit), fitting, int(budget), weights, geometry, keeps, largest, resident)


def choose(plan: Plan, peers: list[list[int]] | None = None, *, window_kind: str = "native") -> int:
    """Choose one window for every rank; an explicit nonfit request refuses on every rank."""

    rows = peers if peers is not None else [plan.settings + [plan.fitting]]
    if any(row[:3] != plan.settings for row in rows):
        raise ValueError(f"CUDA ranks have different {window_kind} windows or context flags; "
                         "start both with the same flags")
    fitting = min(row[3] for row in rows)
    target = plan.requested if plan.requested else plan.native
    if plan.explicit and plan.requested and plan.native > 0 and plan.requested > plan.native:
        raise ValueError(f"requested --context {plan.requested} exceeds the checkpoint's {plan.native}-token "
                         f"{window_kind} window; "
                         f"estimated fitting prompt-plus-reply capacity is {fitting} tokens; reduce --context and "
                         "the prompt/reply reserve, or free memory/use smaller weights")
    if fitting <= 0 or (plan.explicit and plan.requested and target > fitting):
        kind = window_kind if target == plan.native else "default"
        wanted = (f"requested context {target}" if plan.explicit and plan.requested
                  else f"the {target}-token {kind} window or any smaller one")
        raise ValueError(f"CUDA startup memory budget cannot fit {wanted}; estimated largest fitting "
                         f"prompt-plus-reply window: {fitting} tokens across the ranks. " +
                         (f"Use --context {fitting} with a smaller prompt/reply reserve, or " if fitting else "Please ") +
                         "free memory or use smaller/quantized weights; no model weights "
                         "or KV caches have been loaded. KV precision is unchanged.")
    return min(target, fitting)


def floor(model_dir: str | Path) -> tuple[int, int]:
    """The compute capability a checkpoint's kernels need: 8.9 for every format (clusters are taken where present)."""

    from tensorfold.cuda import build

    from tensorfold.checkpoint_metadata import read_config

    path = Path(model_dir)
    if not (path / "config.json").is_file() and path.suffix.lower() != ".gguf" and not any(path.glob("*.gguf")):
        return build.MIN_CAPABILITY
    raw = read_config(model_dir)
    if (raw.get("quantization_config") or raw.get("quantization") or {}).get("quant_method") == "gguf":
        return (7, 0)
    return build.MIN_CAPABILITY


def admit(model_dir: str | Path, requested: int | None, explicit: bool | None, torch,
          geometry: Geometry | Callable, transform: Callable, *, rank: int = 0, world: int = 1,
          gather: Callable | None = None, draft_dir: Path | None = None,
          draft_geometry: Geometry | Callable | None = None, startup_copies: int = 0,
          extra_files: tuple[Path, ...] = (), files: list[Path] | None = None,
          draft_transform: Callable | None = None,
          draft_weights: Callable[[Path], Weights] | None = None,
          weight_estimator: Callable[[Path], tuple[Weights, int]] | None = None,
          context_limit: int | None = None, original_context: int | None = None,
          host_resident: int = 0, host_extra_staging: int = 0, resident_extra: int = 0,
          device_workspace_bytes: int = 0, host_workspace_bytes: int = 0,
          memory_reserve: int | None = None,
          session_geometry: Callable[[dict, int], Geometry] | None = None,
          session_ceiling: int | None = None,
          session_host_workspace: Callable[[int], int] | None = None,
          preallocated_device_bytes: int = 0) -> dict:
    """One refusal or capacity on both ranks before allocating.

    ``weight_estimator`` optionally returns the backend's retained GPU weights,
    GPU load overlap, and a separate host staging allowance, as ``(Weights,
    host_bytes)``. The default uses the generic estimate for both staging
    domains. A backend estimate must be a nonnegative integer byte contract.
    The draft model uses ``draft_weights`` or a transform.
    Additional serving workspaces exclude buffers already charged by geometry.
    Device and host workspaces coexist during serving. On discrete GPUs the
    host workspace replaces loading scratch in the host peak; on unified GPUs
    both serving workspaces share the cache/weight budget. The separate host
    availability/cgroup check keeps its conservative legacy host-extra charge
    in both phases; that argument's unified loading-only charge is unchanged.
    ``preallocated_device_bytes`` are proven nonreusable owners already absent
    from physical free; subtract them only from an explicit absolute cap.
    """

    from tensorfold.cuda import build

    build.refuse_old_gpu(floor(model_dir))          # an old GPU is refused here, before any weight loads
    error = None
    plan = None
    session_fitting, session_factory, session_context = 0, None, 0
    try:
        if type(preallocated_device_bytes) is not int or preallocated_device_bytes < 0:
            raise ValueError("preallocated device commitments must be nonnegative integer bytes")
        preallocation_limit = cuda_limit_bytes() if preallocated_device_bytes else None
        if preallocation_limit is not None and preallocated_device_bytes > preallocation_limit:
            raise ValueError("preallocated device commitments exceed the absolute CUDA budget")
        text = config(model_dir)
        geometry = geometry(text) if callable(geometry) else geometry
        if weight_estimator is None:
            weights = estimate_weights(model_dir, transform, rank=rank, files=files)
            host_staging = weights.staging
        else:
            result = weight_estimator(Path(model_dir))
            if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], Weights):
                raise ValueError("weight_estimator must return (Weights, host staging bytes)")
            weights, host_staging = result
            if any(type(n) is not int or n < 0 for n in
                   (weights.resident, weights.staging, weights.mapped, host_staging)):
                raise ValueError("backend weight estimates must be nonnegative integer bytes")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
               for n in (host_resident, host_extra_staging, resident_extra,
                         device_workspace_bytes, host_workspace_bytes)):
            raise ValueError("additional resident, staging and workspace byte estimates must be nonnegative integers")
        weights = Weights(weights.resident + resident_extra, weights.staging, weights.mapped)
        if unified(torch):
            weights = Weights(weights.resident + host_resident,
                              weights.staging + host_extra_staging, weights.mapped)
        if extra_files:                      # files outside the index, same layout (Nemotron's MTP head, EXL3 tables)
            more = estimate_weights(model_dir, transform, files=list(extra_files))
            host_staging = max(host_staging, more.staging)
            weights = Weights(weights.resident + more.resident, max(weights.staging, more.staging),
                              weights.mapped + more.mapped)
        weights = Weights(weights.resident, weights.staging + startup_copies * weights.resident, weights.mapped)
        if draft_dir is not None:
            if session_geometry is not None and draft_geometry is not None:
                raise ValueError("automatic session geometry must include the draft history for every session")
            draft = draft_weights(draft_dir) if draft_weights is not None else estimate_weights(
                draft_dir, draft_transform or (lambda name, info: (math.prod(info["shape"]) * max(4, itemsize(info, name)),
                                                                   0)))
            host_staging = max(host_staging, draft.staging)
            # the drafter loads after the target: the peak is the larger of either load's
            weights = Weights(weights.resident + draft.resident, max(weights.staging - draft.resident, draft.staging),
                              weights.mapped)
            if draft_geometry is not None:
                draft_geometry = draft_geometry(config(draft_dir)) if callable(draft_geometry) else draft_geometry
                main = geometry
                geometry = Geometry(lambda slots: main.bytes_at(slots) + draft_geometry.bytes_at(slots),
                                    main.reserve, main.minimum_slots)
        serving_workspace = device_workspace_bytes + (host_workspace_bytes if unified(torch) else 0)
        if session_geometry is not None:
            def session_factory(count):
                base = session_geometry(text, count)
                from .geometry import with_fixed

                return with_fixed(base, serving_workspace, label="device_library_workspace")
        if serving_workspace:
            base_geometry = geometry
            geometry = Geometry(lambda slots: base_geometry.bytes_at(slots) + serving_workspace,
                                base_geometry.reserve, base_geometry.minimum_slots)
        host_needed = host_resident + host_extra_staging + max(host_staging, host_workspace_bytes)
        allowance = host_cgroup_bytes() if host_resident or host_extra_staging or host_workspace_bytes or session_geometry is not None else None
        if host_resident or host_extra_staging or host_workspace_bytes:
            if allowance is not None and host_needed > allowance:
                label = "host weights/staging/workspace" if host_workspace_bytes else "host expert weights/staging"
                raise ValueError(f"{label} need {host_needed / GIB:.2f} GiB, "
                                 f"but the container/cgroup has only {allowance / GIB:.2f} GiB remaining")
        host_free = None
        if not unified(torch):
            host_free = host_stream_bytes()
            if host_free is not None and host_needed > host_free:
                label = "host weights/staging/workspace" if host_workspace_bytes else (
                    "host weights/staging" if host_resident or host_extra_staging else "host staging")
                raise ValueError(f"{label} needs an estimated {host_needed / GIB:.2f} GiB, "
                                 f"but only {host_free / GIB:.2f} GiB is available after its reserve; "
                                 "free host memory or use a checkpoint with smaller loading buffers")
        native = int(text.get("max_position_embeddings") or 0)
        if context_limit is not None:
            if isinstance(context_limit, bool) or not isinstance(context_limit, int) or context_limit <= 0:
                raise ValueError("effective RoPE context limit must be a positive integer")
            if original_context is None or isinstance(original_context, bool) or not isinstance(original_context, int) \
                    or original_context <= 0 or original_context > context_limit:
                raise ValueError("original context must be positive and no greater than the effective RoPE limit")
        effective_native = native if context_limit is None else context_limit
        budget = available_bytes(torch) if memory_reserve is None else available_bytes(torch, reserve=memory_reserve)
        if preallocation_limit is not None:
            # Physical free already excludes these owners. Only the absolute
            # cap needs their remaining obligation removed, exactly once.
            budget = min(budget, preallocation_limit - preallocated_device_bytes)
        if session_factory is not None:
            session_context = requested or effective_native
            if (type(session_context) is not int or session_context <= 0
                    or effective_native > 0 and session_context > effective_native):
                raise ValueError("automatic sessions require a positive context within the checkpoint position limit")
            def host_fits(count):
                extra = 0 if session_host_workspace is None else session_host_workspace(count)
                if type(extra) is not int or extra < 0:
                    raise ValueError("session host workspace must be nonnegative integer bytes")
                needed = host_resident + host_extra_staging + max(host_staging, host_workspace_bytes + extra)
                return (host_free is None or needed <= host_free) and (
                    allowance is None or needed <= allowance)

            session_fitting = fit_sessions(session_context, budget, weights, session_factory,
                                           ceiling=session_ceiling, additional_fits=host_fits)
            geometry = session_factory(max(1, session_fitting))
        plan = make_plan(effective_native, requested,
                         requested is not None if explicit is None else explicit,
                         budget,
                         weights, geometry, room=None if session_factory is not None else page_room(torch))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        error = f"{type(exc).__name__}: {exc}"     # name the cause: its text alone has hidden a dtype's KeyError
    status = [1 if error else 0, *(plan.settings + [plan.fitting, plan.largest] if plan else [0, -1, 0, 0, 0])]
    if session_geometry is not None:
        ceiling_status = (-1 if session_ceiling is None else session_ceiling
                          if type(session_ceiling) is int and session_ceiling > 0 else -2)
        status += [session_fitting, ceiling_status]
    both = gather(status) if world > 1 else [status]
    if any(row[0] for row in both):
        raise ValueError("CUDA startup memory geometry could not be established on every rank: " +
                         (error or "another rank could not read its checkpoint; check both folders/configs"))
    if session_geometry is not None:
        if any(len(row) != 8 or row[7] != status[7] for row in both):
            raise ValueError("CUDA ranks selected different automatic session ceilings")
        sessions = min(row[6] for row in both)
        if sessions < 1:
            raise ValueError(f"CUDA memory budget cannot retain one complete {session_context}-token session "
                             "with the selected method and execution resources; context and quality are unchanged")
        geometry = session_factory(sessions)
        plan = make_plan(plan.native, requested, plan.explicit, plan.budget, weights, geometry)
        if plan.fitting < session_context:
            raise RuntimeError("session geometry no longer preserves the admitted full-context commitment")
        # Peers' fitting windows were computed at different local session counts;
        # only the agreed complete window governs this shared session profile.
        both = [row[:4] + [session_context, session_context] + row[6:] for row in both]
    window = choose(plan, [row[1:] for row in both],
                    window_kind="native" if context_limit is None else "RoPE")
    receipt = {**plan.receipt(window), "largest_window": min(row[5] for row in both)}
    if preallocated_device_bytes:
        receipt["device_preallocation"] = {
            "already_committed_bytes": preallocated_device_bytes,
            "absolute_limit_bytes": preallocation_limit,
            "remaining_absolute_limit_bytes": None if preallocation_limit is None else
                                                preallocation_limit - preallocated_device_bytes,
            "included_in_physical_free_baseline": True,
            "total_project_peak_bytes_estimate": receipt["total_bytes_estimate"] + preallocated_device_bytes,
        }
    if session_geometry is not None:
        ledger = {} if geometry.breakdown_at is None else geometry.breakdown_at(
            max(geometry.minimum_slots, session_context + geometry.reserve))
        ledger_names = sorted(ledger)
        local = [rank, plan.budget, weights.resident, weights.staging,
                 geometry.needed(session_context), session_factory(sessions + 1).needed(session_context),
                 int(host_fits(sessions + 1)), *(ledger[name] for name in ledger_names)]
        ranks = gather(local) if world > 1 else [local]
        if any(len(row) != len(local) or row[0] != peer for peer, row in enumerate(ranks)):
            raise RuntimeError("rank resource ledgers do not match the admitted session profile")
        resource = ("administrator-ceiling" if sessions == session_ceiling else
                    "host-memory" if any(not row[6] for row in ranks) else "rank-device-memory")
        receipt["session_capacity"] = {
            "policy": "fixed-context-worst-rank-v1", "evidence": "header-plan-unqualified",
            "context_tokens": session_context, "configured_ceiling": session_ceiling,
            "resident_capacity": sessions, "full_context_commitments": sessions,
            "binding_ranks": [i for i, row in enumerate(both) if row[6] == sessions]
                             if session_ceiling is None or sessions < session_ceiling else [],
            "limiting_resource": resource,
            "ranks": [{"rank": row[0], "budget_bytes": row[1], "resident_weight_bytes": row[2],
                       "loading_overlap_bytes": row[3], "session_execution_bytes": row[4],
                       "next_session_execution_bytes": row[5],
                       "resource_ledger_bytes": dict(zip(ledger_names, row[7:]))} for row in ranks],
        }
        if session_host_workspace is not None:
            host_workspace_bytes += session_host_workspace(sessions)
    if device_workspace_bytes or host_workspace_bytes:
        receipt.update(device_workspace_bytes_estimate=device_workspace_bytes,
                       host_workspace_bytes_estimate=host_workspace_bytes)
    if context_limit is not None:
        receipt.update(native_window=original_context, rope_context_limit=context_limit)
    native_note = str(receipt["native_window"])
    if context_limit is not None:
        native_note += f", RoPE limit {context_limit}"
    print(f"[tensorfold] CUDA rank {rank} startup estimate {receipt['total_bytes_estimate'] / GIB:.2f} GiB "
          f"within {plan.budget / GIB:.2f} GiB; native {native_note}, allocated prompt/reply window {window}, "
          f"cache slots {receipt['cache_slots']}", flush=True)
    note = tables_note(plan)
    if note:
        print(f"[tensorfold] {note}", flush=True)
    return receipt


def tables_note(plan: Plan) -> str | None:
    """What startup says when the window leaves the mapped tables no room (their lookups then read the disk)."""

    if plan.keeps_tables is not False:
        return None
    fix = (f"a --context of {plan.resident} or less, or fewer --parallel streams, keeps them resident"
           if plan.explicit and plan.resident else "free memory to keep them resident")
    return (f"the {plan.weights.mapped / GIB:.1f} GiB of mapped tables do not fit beside the weights and caches: "
            f"lookups will page them from disk, which slows prompts ({fix})")


def gather_ints(torch, gather: Callable, values: list[int], world: int = 2) -> list[list[int]]:
    send = torch.tensor(values, dtype=torch.int64, device="cuda")
    receive = torch.empty((world * len(values),), dtype=torch.int64, device="cuda")
    gather(send, receive)
    return receive.view(world, -1).tolist()
