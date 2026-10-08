"""Host-expert configuration and header-only memory accounting; no accelerator imports."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path

from tensorfold.cuda.capacity import config, headers

GIB = 2**30
AUTO_HEADROOM_BYTES = 512 * 2**20
MAX_ROUTED_EXPERTS = 1023  # experts.cu's EMAX=1024 includes the shared expert.


def check(model_dir: str | Path, gib: float | str, *, tp: int = 1) -> None:
    """One-GPU packed affine4 group32 or EXL3 weights; budget units are GiB."""

    if gib != "auto" and (isinstance(gib, bool) or not isinstance(gib, (int, float)) or not math.isfinite(gib) or gib <= 0):
        raise ValueError("--vram-experts must be auto or a finite positive GiB count")
    if tp != 1:
        raise ValueError("--vram-experts currently supports one CUDA GPU (--tp 1)")
    from tensorfold.cuda.exl3.format import is_exl3
    from tensorfold.quantization import checkpoint_specs

    if is_exl3(Path(model_dir)):
        return
    import json

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    specs = [spec for spec in checkpoint_specs(raw).values() if spec is not None]
    if not specs or any(spec.bits != 4 or spec.group_size != 32 for spec in specs):
        raise ValueError("--vram-experts supports MLX affine 4-bit weights in groups of 32 only")
    quant = raw.get("quantization_config") or raw.get("quantization") or {}
    if quant.get("quant_method") == "modelopt":
        raise ValueError("--vram-experts does not support NVFP4; use affine 4-bit group32 weights")


def expert_tensor(name: str) -> bool:
    """The packed projections; router and shared routing gate remain resident."""

    return ".switch_mlp." in name or ".shared_expert." in name


def plan_scratch(text: dict, rows: int, prompt_rows: int, mtp: bool) -> tuple[int, int]:
    """Additional device/pinned-host plan windows for decode, MTP and mixed prefill buffers."""

    experts, slots = int(text["num_experts"]) + 1, int(text["num_experts_per_tok"]) + 1
    device = host = 0
    for count in [rows] * (1 + int(mtp)) + [prompt_rows + rows]:
        pairs = count * slots
        values = 3 * (min(pairs, experts) + pairs // 16)
        device += (2 * values + 4) * 4
        host += (values + 2) * 4
    return device, host


@dataclass(frozen=True)
class Layout:
    entry_bytes: int
    slots: int
    gpu_bytes: int
    host_bytes: int
    staging_bytes: int
    loading_bytes: int
    format: str = "affine4"
    metadata_device_bytes: int = 0
    metadata_host_bytes: int = 0
    control_device_bytes: int = 0
    control_host_bytes: int = 0
    automatic: bool = False

    def transform(self, resident):
        """Exclude expert projections; the fixed GPU pool is admitted as resident_extra."""

        def weights(name, info):
            if self.format == "exl3" and exl3_expert_tensor(name):
                if name.endswith((".trellis", ".mcg", ".mul1")):
                    return 0, 0
                # Prepared sign vectors occupy FP16, regardless of whether
                # the checkpoint stores individual scales or packed signs.
                if name.endswith((".su", ".sv")):
                    if resident(name, info) == (0, 0):
                        return 0, 0
                    return math.prod(info["shape"]) * 16 * 2, 0
                return resident(name, info)
            if expert_tensor(name):
                return 0, 0
            return resident(name, info)
        return weights


def layout(model_dir: str | Path, gib: float | str, *, mtp: bool = True) -> Layout:
    """Size exact packed affine4 buffers, including shared and optional MTP experts."""

    check(model_dir, gib)
    text = config(model_dir)
    for field in ("moe_intermediate_size", "hidden_size", "shared_expert_intermediate_size",
                  "num_hidden_layers", "num_experts", "num_experts_per_tok"):
        if type(text.get(field)) is not int or text[field] <= 0:
            raise ValueError(f"--vram-experts: {field} must be a positive integer")
    if text["num_experts_per_tok"] > text["num_experts"]:
        raise ValueError("--vram-experts: num_experts_per_tok must not exceed num_experts")
    if text["num_experts"] > MAX_ROUTED_EXPERTS:
        raise ValueError(f"--vram-experts: the CUDA plan supports at most {MAX_ROUTED_EXPERTS} routed experts")
    width, dims = int(text["moe_intermediate_size"]), int(text["hidden_size"])
    if width <= 0 or dims <= 0 or width % 32 or dims % 32:
        raise ValueError("--vram-experts requires expert width and hidden size divisible by 32")
    if int(text["shared_expert_intermediate_size"]) != width:
        raise ValueError("--vram-experts requires matching routed and shared expert widths")
    from tensorfold.cuda.exl3.format import is_exl3

    if is_exl3(Path(model_dir)):
        return _exl3_layout(model_dir, gib, text, mtp)
    # 160 int32 words per 32x32 tile: 4-bit weights plus BF16 affine scales/biases.
    entry = 3 * (width // 32) * (dims // 32) * 160 * 4
    bundles: dict[str, dict] = {}
    for name, info in headers(model_dir).items():
        if not mtp and (name.startswith("mtp.") or ".mtp." in name):
            continue
        if not expert_tensor(name):
            continue
        marker = ".switch_mlp." if ".switch_mlp." in name else ".shared_expert."
        base, part = name.split(marker, 1)
        bundle = bundles.setdefault(base, {})
        bundle[(marker, part)] = info
    count = 0
    for base, bundle in bundles.items():
        for marker in (".switch_mlp.", ".shared_expert."):
            for projection, rows, cols in (("gate_proj", width, dims), ("up_proj", width, dims),
                                            ("down_proj", dims, width)):
                for field, dtype, last in (("weight", {"U32", "I32"}, cols // 8),
                                            ("scales", {"BF16"}, cols // 32),
                                            ("biases", {"BF16"}, cols // 32)):
                    info = bundle.get((marker, projection + "." + field))
                    expected = ([int(text["num_experts"])] if marker == ".switch_mlp." else []) + [rows, last]
                    if info is None or info["shape"] != expected or info["dtype"] not in dtype or \
                            any(type(n) is not int or n <= 0 for n in info["shape"]):
                        raise ValueError(f"--vram-experts: invalid affine4 projection {base}{marker}{projection}.{field}")
        count += int(text["num_experts"]) + 1
    if not count:
        raise ValueError("--vram-experts: checkpoint has no complete affine4 expert layers")
    main_layers = {int(match.group(1)) for base in bundles
                   if (match := re.search(r"(?:^|\.)model\.layers\.(\d+)\.mlp$", base))}
    if main_layers != set(range(int(text["num_hidden_layers"]))):
        raise ValueError("--vram-experts: checkpoint must contain every configured main expert layer")
    requested = entry if gib == "auto" else math.floor(gib) * GIB + int((gib % 1) * GIB)
    slots = min(count, requested // entry)
    if not slots:
        raise ValueError(f"--vram-experts needs at least one packed expert ({entry / GIB:.6f} GiB)")
    # Two pinned upload entries; two-expert CPU chunks need at most 8*entry +256KiB of live arrays.
    return Layout(entry, slots, slots * entry, count * entry, 2 * entry,
                  max(64 * 2**20, 8 * entry + 256 * 2**10), automatic=gib == "auto")


def exl3_expert_tensor(name: str) -> bool:
    """EXL3 expert projections; routing weights and shared gates stay resident."""

    return ".mlp.experts." in name or ".mlp.shared_expert." in name


def _exl3_layout(model_dir: str | Path, gib: float | str, text: dict, mtp: bool) -> Layout:
    """Compact original trellises on CPU, maximum-size GPU cells, resident scales.

    All configured layers are validated from headers before payload allocation.
    A GPU cell preserves the original gate/up/down streams and their bit widths;
    no dequantization or requantization changes the checkpoint precision.
    """

    from tensorfold.cuda.exl3.format import PARTS, parse_group

    if text["num_experts_per_tok"] + 1 > 32:
        raise ValueError("--vram-experts: native EXL3 grouping supports at most 32 routed/shared slots per row")

    projections: dict[str, dict] = {}
    for name, info in headers(model_dir).items():
        if not mtp and (name.startswith("mtp.") or ".mtp." in name):
            continue
        if not exl3_expert_tensor(name):
            continue
        prefix, _, part = name.rpartition(".")
        if part not in PARTS or any(type(n) is not int or n <= 0 for n in info["shape"]):
            raise ValueError(f"--vram-experts: invalid EXL3 expert field {name}")
        projections.setdefault(prefix, {})[part] = info
    bases = {prefix.split(".experts.", 1)[0] if ".experts." in prefix
             else prefix.split(".shared_expert.", 1)[0] for prefix in projections}
    main = {int(match.group(1)) for base in bases
            if (match := re.search(r"(?:^|\.)model(?:\.language_model)?\.layers\.(\d+)\.mlp$", base))}
    if main != set(range(text["num_hidden_layers"])):
        raise ValueError("--vram-experts: checkpoint must contain every configured main EXL3 expert layer")
    if any(not (re.search(r"(?:^|\.)model(?:\.language_model)?\.layers\.\d+\.mlp$", base)
                or re.fullmatch(r"mtp\.layers\.0\.mlp", base)) for base in bases):
        raise ValueError("--vram-experts: unrecognized EXL3 expert layer")
    width, dims, total, largest, count = text["moe_intermediate_size"], text["hidden_size"], 0, 0, 0
    expected = set()
    for base in sorted(bases):
        codebooks = set()
        for expert in [f"{base}.experts.{i}" for i in range(text["num_experts"])] + [f"{base}.shared_expert"]:
            entry = 0
            for projection, inputs, outputs in (("gate_proj", dims, width), ("up_proj", dims, width),
                                                ("down_proj", width, dims)):
                prefix = f"{expert}.{projection}"
                expected.add(prefix)
                parts = projections.get(prefix)
                if parts is None:
                    raise ValueError(f"--vram-experts: missing EXL3 projection {prefix}")
                meta = parse_group(prefix, parts)
                if (meta.k, meta.n) != (inputs, outputs) or meta.bias:
                    raise ValueError(f"--vram-experts: incompatible EXL3 projection {prefix}")
                codebooks.add(meta.codebook)
                entry = -(-entry // 16) * 16 + meta.trellis_bytes
                total += meta.trellis_bytes
            largest = max(largest, -(-entry // 16) * 16)
            count += 1
        if len(codebooks) != 1:
            raise ValueError(f"--vram-experts: mixed EXL3 codebooks in {base}")
    if set(projections) != expected:
        raise ValueError("--vram-experts: unexpected EXL3 expert projection")
    requested = largest if gib == "auto" else math.floor(gib) * GIB + int((gib % 1) * GIB)
    slots = min(count, requested // largest)
    if not slots:
        raise ValueError(f"--vram-experts needs at least one packed EXL3 expert ({largest / GIB:.6f} GiB)")
    # Three logical pointer/width tables per layer; prepared FP16 scales are
    # already included by the resident weight transform.
    return Layout(largest, slots, slots * largest, total, 2 * largest,
                  max(64 * 2**20, 2 * largest), "exl3", count * (3 * 8 + 3 * 4), count * 28,
                  slots * 32, slots * 64, gib == "auto")


def auto_pool_bytes(available: int, entry_bytes: int, expert_count: int, *, control_bytes: int = 0,
                    growth_bytes: int = 0, snapshot_bytes: int = 0, workspace_bytes: int = 0) -> dict:
    """One startup allocation from actual remaining device bytes, before any lease.

    The caller has already allocated shared weights and every fixed decoder
    buffer. It supplies worst future full-context cache growth/copy overlap,
    retained/in-flight snapshots and unallocated prompt workspace separately.
    Packed weights and per-cell publication controls share the remainder; the
    explicit 512-MiB headroom is additional to those model-owned allocations.
    """
    values = (available, entry_bytes, expert_count, control_bytes, growth_bytes, snapshot_bytes, workspace_bytes)
    if any(type(value) is not int or value < 0 for value in values) or not entry_bytes or not expert_count:
        raise ValueError("auto expert sizing requires nonnegative integer byte counts and positive geometry")
    reserved = AUTO_HEADROOM_BYTES + growth_bytes + snapshot_bytes + workspace_bytes
    cells = min(expert_count, max(0, available - reserved) // (entry_bytes + control_bytes))
    if not cells:
        raise ValueError("CUDA remaining memory cannot fit one automatic expert cell after full service workspace")
    return {"policy": "remaining-device-bytes-after-fixed-service-state-v1", "available_bytes": available,
            "headroom_bytes": AUTO_HEADROOM_BYTES, "full_context_growth_and_copy_bytes": growth_bytes,
            "snapshot_bytes": snapshot_bytes, "unallocated_workspace_bytes": workspace_bytes,
            "reserved_bytes": reserved, "entry_bytes": entry_bytes, "control_bytes_per_cell": control_bytes,
            "logical_expert_count": expert_count, "slots": cells, "gpu_bytes": cells * entry_bytes,
            "control_device_bytes": cells * control_bytes,
            "unused_after_reservations_bytes": available - reserved - cells * (entry_bytes + control_bytes)}
