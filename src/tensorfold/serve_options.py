"""Serve options a backend or family has no path for, refused before any weight is downloaded."""

from __future__ import annotations

import argparse
import inspect
import math
from typing import Any


def check_numbers(args: argparse.Namespace) -> None:
    """Validate CLI floats before downloads or accelerator/resource acquisition.

    Finite temperature/top-p retain their existing greedy/off-range policies;
    memory budgets and decode-share have nonnegative units, SSD pools positive.
    Integer clamping/default policies remain with their existing owners.
    """

    from tensorfold.server.request_limits import optional_limit

    for name in ("max_http_connections", "max_pending_requests", "max_engine_calls"):
        optional_limit(getattr(args, name, None), "--" + name.replace("_", "-"))
    budgets = ("prompt_cache_gib", "spill_gib", "pass_cache_gib", "mlx_cache_gib", "ssd_experts")
    for name in (*budgets, "temperature", "top_p", "min_p", "decode_share", "mtp_confidence", "yarn_factor"):
        value = getattr(args, name, None)
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{flag} must be a finite number")
        if name in budgets or name == "decode_share":
            if value < 0 or (name == "ssd_experts" and value == 0):
                raise ValueError(f"{flag} must be {'positive' if name == 'ssd_experts' else 'nonnegative'}")
        if name in ("min_p", "mtp_confidence") and not 0 <= value <= 1:
            raise ValueError(f"{flag} must be between 0 and 1")


def check(args: argparse.Namespace, family: Any, backend: str, config_dir: Any = None) -> None:
    """Refuse KV cache, draft rule, image, share, slot and precision options the backend or family can't serve."""

    if getattr(args, "max_engine_calls", None) is not None and backend != "mlx":
        raise ValueError("--max-engine-calls bounds the MLX engine RPC queue and is supported on MLX only")
    if getattr(args, "trust_model_code", False):
        if backend != "mlx" or not getattr(family.package, "MLX_MODEL_FILE", False):
            raise ValueError("--trust-model-code requires an MLX family recipe that uses the provider's model_file")
    ram = getattr(args, "vram_experts", None)
    if getattr(args, "ssd_experts", None) is not None and backend == "cuda":
        raise ValueError("--ssd-experts is supported on MLX only; Flash Next CUDA supports --vram-experts GIB")
    if ram is not None:
        if ram != "auto" and (isinstance(ram, bool) or not isinstance(ram, (int, float))
                              or not math.isfinite(ram) or ram <= 0):
            raise ValueError("--vram-experts must be auto or a finite positive GiB count")
        if getattr(args, "ssd_experts", None) is not None:
            raise ValueError("--vram-experts and --ssd-experts cannot be combined")
        check_ram = getattr(family.package, "check_vram_experts", None)
        if backend != "cuda" or check_ram is None:
            raise ValueError("--vram-experts is supported by Flash Next affine 4-bit on CUDA only")
        check_ram(config_dir, ram, tp=getattr(args, "tp", 1))
    if getattr(args, "yarn_factor", None) is not None:
        if backend != "cuda" or not hasattr(family.package, "rope_parameters"):
            raise ValueError("--yarn-factor is supported by Flash Next on CUDA only")
        family.package.rope_parameters(config_dir, args.yarn_factor)
    if getattr(args, "vision_urls", False) and not getattr(args, "vision", False):
        raise ValueError("--vision-urls needs --vision")
    images = getattr(args, "vision_max_images", None)
    if images is not None:
        if not isinstance(images, int) or isinstance(images, bool) or images < 1:
            raise ValueError("--vision-max-images must be a positive integer")
        if not getattr(args, "vision", False):
            raise ValueError("--vision-max-images needs --vision")
    if getattr(args, "vision", False):             # only --vision reads the config here
        if family.model_type == "glm5_next" and backend != "mlx":
            raise ValueError("GLM-5.3-Flash image input is currently MLX-only")
        from tensorfold.families import read_config
        from tensorfold.vision.config import validate_vision_config

        validate_vision_config(read_config(config_dir) if config_dir else {}, family.model_type)
        if family.model_type == "qwen4_exp" and backend != "cuda":
            raise ValueError("--vision for Flash Next runs on the CUDA engine; the MLX path has no image tower yet")
    share = getattr(args, "decode_share", None)
    if share is not None and backend == "cuda" and not getattr(family.package, "CUDA_DECODE_SHARE", False):
        raise ValueError("--decode-share sets the Mac server's share, and Flash Next's on CUDA; this CUDA engine runs "
                         "a round after each 1,024 prompt rows")
    if share is not None and share < 0:
        raise ValueError(f"--decode-share is 0 (whole prompts first) or more, not {share}")
    kv = getattr(args, "kv_dtype", "bf16")
    if kv != "bf16" and backend != "cuda":
        raise ValueError(f"--kv-dtype {kv} is a CUDA engine option: the MLX path caches keys and values as bf16")
    supported = getattr(family.package, "CUDA_KV_DTYPES", ("bf16",))
    if kv not in supported:
        raise ValueError(f"{family.title} on CUDA serves a {' or '.join(supported)} KV cache, not --kv-dtype {kv}")
    for name in ("kv_key_dtype", "kv_value_dtype"):
        value = getattr(args, name, None)
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if backend != "cuda" or not getattr(family.package, "CUDA_KV_PAIRS", False):
            raise ValueError(f"{flag} is supported by Flash Next on CUDA only")
        if value not in supported:
            raise ValueError(f"{flag} {value!r}: choose {' or '.join(supported)}")
    slots = getattr(args, "checkpoint_slots", None)
    if slots is not None and backend == "cuda" and getattr(family.package, "CUDA_CHECKPOINT_SLOTS", False):
        if slots < 1:
            raise ValueError(f"--checkpoint-slots is 1 or more, not {slots}")
        if _cuda_streams(getattr(args, "parallel", "auto")) < 2:
            raise ValueError(f"--checkpoint-slots sets the prompt states {family.title}'s concurrent decoder keeps on "
                             "CUDA (--parallel 2 or more); one stream keeps 4, which share its attention buffer")
    fp8 = getattr(family.package, "CUDA_PREFILL_FP8", False) and backend == "cuda"
    if getattr(args, "prefill_fp8", None) and not fp8:              # asked for by name, not a default
        raise ValueError(f"--prefill-fp8 picks FP8 prompt kernels on CUDA; {family.title} on "
                         f"{'CUDA' if backend == 'cuda' else 'MLX'} has none (its prompts run bf16 activations)")
    confidence = getattr(args, "mtp_confidence", None)
    if confidence is None:
        return
    engine = getattr(family.package, "cuda_engine", None) if backend == "cuda" else None
    if engine is None or "mtp_confidence" not in inspect.signature(engine).parameters:
        raise ValueError(f"--mtp-confidence sets where a CUDA engine's MTP chains stop; {family.title} on "
                         f"{'CUDA' if backend == 'cuda' else 'MLX'} has no such rule")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"--mtp-confidence is a probability from 0 to 1, not {confidence}")


def _cuda_streams(value: Any) -> int:
    """The streams a CUDA engine serves for ``--parallel`` (auto: one), or 2 for a value the serve command refuses itself."""

    text = str(value).strip().lower()
    if text == "auto":
        return 1
    try:
        return max(1, int(text))
    except ValueError:
        return 2


def vision_options(args: argparse.Namespace) -> dict[str, Any]:
    """``--vision`` and ``--vision-urls`` as a family's load options."""

    if not getattr(args, "vision", False):
        return {}
    return {"vision": True, "vision_urls": bool(getattr(args, "vision_urls", False))}


__all__ = ["check", "check_numbers", "vision_options"]
