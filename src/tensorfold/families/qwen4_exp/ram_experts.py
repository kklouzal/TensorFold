"""Host-expert configuration and header-only memory accounting; no accelerator imports."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path

from tensorfold.cuda.capacity import config, headers

GIB = 2**30
MAX_ROUTED_EXPERTS = 1023  # experts.cu's EMAX=1024 includes the shared expert.


def check(model_dir: str | Path, gib: float, *, tp: int = 1) -> None:
    """One-GPU affine4 group32 weights, with finite, positive packed GPU cache bytes."""

    if isinstance(gib, bool) or not isinstance(gib, (int, float)) or not math.isfinite(gib) or gib <= 0:
        raise ValueError("--ram-experts must be a finite positive GiB count")
    if tp != 1:
        raise ValueError("--ram-experts currently supports one CUDA GPU (--tp 1)")
    from tensorfold.cuda.exl3.format import is_exl3
    from tensorfold.quantization import checkpoint_specs

    if is_exl3(Path(model_dir)):
        raise ValueError("--ram-experts does not support EXL3 packs; use affine 4-bit group32 weights")
    import json

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    specs = [spec for spec in checkpoint_specs(raw).values() if spec is not None]
    if not specs or any(spec.bits != 4 or spec.group_size != 32 for spec in specs):
        raise ValueError("--ram-experts supports MLX affine 4-bit weights in groups of 32 only")
    quant = raw.get("quantization_config") or raw.get("quantization") or {}
    if quant.get("quant_method") == "modelopt":
        raise ValueError("--ram-experts does not support NVFP4; use affine 4-bit group32 weights")


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

    def transform(self, resident):
        """Exclude expert projections; the fixed GPU pool is admitted as resident_extra."""

        def weights(name, info):
            if expert_tensor(name):
                return 0, 0
            return resident(name, info)
        return weights


def layout(model_dir: str | Path, gib: float, *, mtp: bool = True) -> Layout:
    """Size exact packed affine4 buffers, including shared and optional MTP experts."""

    check(model_dir, gib)
    text = config(model_dir)
    for field in ("moe_intermediate_size", "hidden_size", "shared_expert_intermediate_size",
                  "num_hidden_layers", "num_experts", "num_experts_per_tok"):
        if type(text.get(field)) is not int or text[field] <= 0:
            raise ValueError(f"--ram-experts: {field} must be a positive integer")
    if text["num_experts_per_tok"] > text["num_experts"]:
        raise ValueError("--ram-experts: num_experts_per_tok must not exceed num_experts")
    if text["num_experts"] > MAX_ROUTED_EXPERTS:
        raise ValueError(f"--ram-experts: the CUDA plan supports at most {MAX_ROUTED_EXPERTS} routed experts")
    width, dims = int(text["moe_intermediate_size"]), int(text["hidden_size"])
    if width <= 0 or dims <= 0 or width % 32 or dims % 32:
        raise ValueError("--ram-experts requires expert width and hidden size divisible by 32")
    if int(text["shared_expert_intermediate_size"]) != width:
        raise ValueError("--ram-experts requires matching routed and shared expert widths")
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
                        raise ValueError(f"--ram-experts: invalid affine4 projection {base}{marker}{projection}.{field}")
        count += int(text["num_experts"]) + 1
    if not count:
        raise ValueError("--ram-experts: checkpoint has no complete affine4 expert layers")
    main_layers = {int(match.group(1)) for base in bundles
                   if (match := re.search(r"(?:^|\.)model\.layers\.(\d+)\.mlp$", base))}
    if main_layers != set(range(int(text["num_hidden_layers"]))):
        raise ValueError("--ram-experts: checkpoint must contain every configured main expert layer")
    requested = math.floor(gib) * GIB + int((gib % 1) * GIB)
    slots = min(count, requested // entry)
    if not slots:
        raise ValueError(f"--ram-experts needs at least one packed expert ({entry / GIB:.6f} GiB)")
    # Two pinned upload entries; two-expert CPU chunks need at most 8*entry +256KiB of live arrays.
    return Layout(entry, slots, slots * entry, count * entry, 2 * entry,
                  max(64 * 2**20, 8 * entry + 256 * 2**10))
