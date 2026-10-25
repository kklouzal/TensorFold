"""Serve options a backend or family has no path for, refused before any weight is downloaded."""

from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path
from typing import Any


def check_numbers(args: argparse.Namespace) -> None:
    """Validate CLI floats before downloads or accelerator/resource acquisition.

    A finite temperature at or below 0 stays greedy; top-p, like min-p, must lie in [0, 1], the range every request
    and both servers' samplers require (a default outside it would fail each request that inherits it);
    memory budgets and decode-share have nonnegative units, SSD pools positive.
    Integer clamping/default policies remain with their existing owners.
    """

    from tensorfold.server.request_limits import optional_limit
    from tensorfold.server.request_options import PENALTY_FIELDS

    eos = getattr(args, "eos_token_ids", None)
    if eos is not None and (not eos or any(type(token) is not int or token < 0 for token in eos)):
        raise ValueError("--eos-token-id must be a nonnegative integer (repeat for multiple IDs)")

    for name in ("max_http_connections", "max_pending_requests", "max_engine_calls"):
        optional_limit(getattr(args, name, None), "--" + name.replace("_", "-"))
    budgets = ("prompt_cache_gib", "spill_gib", "pass_cache_gib", "mlx_cache_gib", "ssd_experts")
    for name in (*budgets, "temperature", "top_p", "min_p", *PENALTY_FIELDS, "decode_share", "mtp_confidence",
                 "yarn_factor"):
        value = getattr(args, name, None)
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{flag} must be a finite number")
        if name in budgets or name == "decode_share":
            if value < 0 or (name == "ssd_experts" and value == 0):
                raise ValueError(f"{flag} must be {'positive' if name == 'ssd_experts' else 'nonnegative'}")
        if name in ("top_p", "min_p", "mtp_confidence") and not 0 <= value <= 1:
            raise ValueError(f"{flag} must be between 0 and 1")
        if name in ("presence_penalty", "frequency_penalty") and not -2 <= value <= 2:
            raise ValueError(f"{flag} must be between -2 and 2")
        if name == "repetition_penalty" and value <= 0:
            raise ValueError(f"{flag} must be greater than 0")


def _served_name(args: argparse.Namespace) -> str:
    """``--name``, else a repo id's last segment or the model path's name; no I/O (matches ``cli._serve_cuda``)."""

    if getattr(args, "name", ""):
        return args.name
    from tensorfold import hub

    model = str(args.model).rstrip("/")
    return model.split("/")[-1] if hub.is_repo_id(str(args.model)) else Path(model).expanduser().resolve().name


def name_priority(args: argparse.Namespace) -> dict[str, str]:
    """``--name-priority ID=background`` parsed: the id and the priority word it defaults to. No I/O."""

    parsed: dict[str, str] = {}
    for entry in getattr(args, "name_priority", None) or []:
        model_id, sep, priority = str(entry).partition("=")
        if not sep or not model_id.strip() or priority.strip().lower() != "background":
            raise ValueError(f"--name-priority takes ID=background, not {entry!r}")
        parsed[model_id.strip()] = "background"
    return parsed


def check(args: argparse.Namespace, family: Any, backend: str, config_dir: Any = None) -> None:
    """Refuse KV cache, draft rule, image, tokenizer, share, slot and precision options the backend or family can't
    serve."""

    method = getattr(args, "draft_method", "auto")
    if method not in ("auto", "none", "mtp", "dspark"):
        raise ValueError("--draft-method must be auto, none, mtp or dspark")
    if method != "auto" and (backend != "cuda" or method not in getattr(family.package, "CUDA_DRAFT_METHODS", ())):
        raise ValueError("--draft-method requires a CUDA model with the selected proposal method")
    draft_rows = getattr(args, "draft_rows", None)
    if draft_rows is not None:
        if (backend != "cuda" or not getattr(family.package, "CUDA_DRAFT_ROWS", False)
                or type(draft_rows) is not int or not 2 <= draft_rows <= 128):
            raise ValueError("--draft-rows requires supported CUDA DFlash and an integer from 2 through 128")
        if _cuda_streams(getattr(args, "parallel", "auto")) > 1 and draft_rows > 16:
            raise ValueError("--draft-rows supports at most 16 rows with --parallel")
    ranks = getattr(args, "expert_ranks", 1)
    shards = getattr(args, "expert_shards", None)
    if ranks != 1 or shards is not None:
        if backend != "cuda" or getattr(family, "model_type", "") != "qwen4_exp":
            raise ValueError("expert parallelism is supported by Flash Next CUDA only")
        if getattr(args, "tp", 1) != 1 or getattr(args, "vram_experts", None) is None:
            raise ValueError("expert parallelism requires --tp 1 and --vram-experts GIB|auto")
        from tensorfold.cuda.capacity import config
        from tensorfold.cuda.expert_parallel import ownership
        ownership(ranks, shards, getattr(args, "rank", 0), int(config(config_dir)["num_experts"]))
    history = getattr(args, "indexer_history", "gpu")
    if history not in ("gpu", "ram"):
        raise ValueError("--indexer-history must be gpu or ram")
    if history == "ram":
        if backend != "cuda" or getattr(family, "model_type", "") not in ("qwen4_exp", "qwen3_8_flash_next"):
            raise ValueError("--indexer-history ram requires Flash Next GGUF on CUDA/SM70")
        from tensorfold.families.qwen4_exp.gguf import resolve_gguf

        source = config_dir if config_dir is not None else getattr(args, "model", None)
        if source is None or resolve_gguf(source) is None:
            raise ValueError("--indexer-history ram requires a local Flash Next GGUF checkpoint")
    serving_root = getattr(args, "serving_profile_root", None)
    if serving_root is not None:
        from tensorfold.families.qwen4_exp.cuda.serving_profiles import profile_root
        from tensorfold.server.request_limits import require_finite_endpoint_limits

        require_finite_endpoint_limits(max_pending_requests=getattr(args, "max_pending_requests", None),
                                      max_http_connections=getattr(args, "max_http_connections", None))
        profile_root(serving_root)
    manifest = getattr(args, "execution_profile_manifest", None)
    if manifest is not None or serving_root is not None:
        from tensorfold.families.qwen4_exp.gguf import resolve_gguf

        source = config_dir if config_dir is not None else getattr(args, "model", None)
        if (backend != "cuda" or getattr(family, "model_type", "") not in ("qwen4_exp", "qwen3_8_flash_next")
                or history != "gpu" or getattr(args, "vram_experts", None) is not None
                or source is None or resolve_gguf(source) is None):
            raise ValueError("execution/serving profile selection requires resident Flash Next GGUF GPU histories on SM70")
    windows = getattr(args, "kv_full_windows", None)
    if windows is not None:
        if backend != "cuda" or getattr(family, "model_type", "") not in ("qwen4_exp", "qwen3_8_flash_next"):
            raise ValueError("--kv-full-windows is supported by Flash Next CUDA only")
        from tensorfold.families.qwen4_exp.ram_experts import check_kv_full_windows
        from tensorfold.families.qwen4_exp.gguf import resolve_gguf
        resident_gguf = config_dir is not None and resolve_gguf(config_dir) is not None
        adaptive_gguf = resident_gguf and history == "gpu" and getattr(args, "vram_experts", None) is None
        if windows == "auto":
            if not adaptive_gguf:
                raise ValueError("--kv-full-windows auto requires resident GGUF GPU histories and automatic capacity")
        elif type(windows) is not int or windows < 1:
            check_kv_full_windows(windows, streams=1, vram_experts=None)
        elif not adaptive_gguf:
            check_kv_full_windows(windows, streams=_cuda_streams(getattr(args, "parallel", "auto")),
                                  vram_experts=getattr(args, "vram_experts", None), tp=getattr(args, "tp", 1),
                                  resident_gguf=resident_gguf)
    if backend == "cuda":
        _cuda_topology(args, family)
    if backend == "mlx" and getattr(family, "model_type", "") == "qwen3_5_moe" and not getattr(args, "no_drafts", False):
        cap = getattr(args, "mtp_drafts", None)
        if cap is not None:
            if type(cap) is not int or cap < 0:
                raise ValueError("Qwen3.6 MoE --mtp-drafts must be a nonnegative integer")
            if getattr(args, "drafter", "auto") not in ("auto", "none", ""):
                raise ValueError("Qwen3.6 MoE --mtp-drafts selects its included MTP layer; it cannot be combined with "
                                 "an explicit separate --drafter")
    if getattr(args, "max_engine_calls", None) is not None and backend != "mlx":
        raise ValueError("--max-engine-calls bounds the MLX engine RPC queue and is supported on MLX only")
    if getattr(args, "trust_model_code", False):
        if backend != "mlx" or not getattr(family.package, "MLX_MODEL_FILE", False):
            raise ValueError("--trust-model-code requires an MLX family recipe that uses the provider's model_file")
    ram = getattr(args, "vram_experts", None)
    pin = getattr(args, "vram_experts_pin", "off")
    if pin not in ("off", "auto", "on") or pin != "off" and (ram is None or backend != "cuda"):
        raise ValueError("--vram-experts-pin requires supported CUDA --vram-experts and off, auto or on")
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
            raise ValueError("--yarn-factor is supported by Flash Next, Qwen3.8 dense and Qwen3.6 on CUDA only")
        family.package.rope_parameters(config_dir, args.yarn_factor)
    if getattr(args, "vision_urls", False) and not getattr(args, "vision", False):
        raise ValueError("--vision-urls needs --vision")
    images = getattr(args, "vision_max_images", None)
    if images is not None:
        if not isinstance(images, int) or isinstance(images, bool) or images < 1:
            raise ValueError("--vision-max-images must be a positive integer")
        if not getattr(args, "vision", False):
            raise ValueError("--vision-max-images needs --vision")
    tokens = getattr(args, "vision_image_tokens", None)
    if tokens is not None:
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= 65536:
            raise ValueError("--vision-image-tokens is a number of tokens from 1 to 65,536")
        if not getattr(args, "vision", False):
            raise ValueError("--vision-image-tokens needs --vision")
        if backend != "cuda":
            raise ValueError("--vision-image-tokens sets the CUDA Qwen image budget; the MLX towers size their "
                             "workspace for 4,096 visual tokens")
    if getattr(args, "vision_offload", False):
        if not getattr(args, "vision", False):
            raise ValueError("--vision-offload needs --vision")
        if backend != "cuda":
            raise ValueError("--vision-offload is for the CUDA backend; the Mac's image tower already shares host memory")
    if getattr(args, "tokenizer_file", None) is not None:
        if backend != "cuda":
            raise ValueError("--tokenizer is read by the CUDA server; MLX serves the checkpoint's tokenizer")
        if getattr(args, "vision", False):         # the vision processors encode image prompts with their own tokenizer
            raise ValueError("--tokenizer encodes text prompts only: image prompts would keep the checkpoint's "
                             "tokenizer, so it cannot be combined with --vision")
        if int(getattr(args, "tp", 1) or 1) > 1 or int(getattr(args, "expert_ranks", 1) or 1) > 1:
            raise ValueError("--tokenizer is admitted in one process: several ranks would each compile grammars from "
                             "their own tokenizer, unchecked against rank 0's, so it serves one rank only")
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
    if "fp16" in (kv, getattr(args, "kv_key_dtype", None), getattr(args, "kv_value_dtype", None)):
        from tensorfold.families.qwen4_exp.gguf import resolve_gguf

        if (backend != "cuda" or getattr(family, "model_type", "") != "qwen4_exp"
                or resolve_gguf(config_dir or args.model) is None):
            raise ValueError("--kv-dtype fp16 requires the Flash Next GGUF FP16 execution provider")
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
    if getattr(args, "drafter_bits", None) is not None and backend == "cuda" \
            and not getattr(family.package, "CUDA_DRAFTER_BITS", False):
        raise ValueError(f"--drafter-bits: {family.title} on CUDA runs its draft model at one fixed precision; drop "
                         "the flag")
    fp8 = getattr(family.package, "CUDA_PREFILL_FP8", False) and backend == "cuda"
    if getattr(args, "prefill_fp8", None) and not fp8:              # asked for by name, not a default
        raise ValueError(f"--prefill-fp8 picks FP8 prompt kernels on CUDA; {family.title} on "
                         f"{'CUDA' if backend == 'cuda' else 'MLX'} has none (its prompts run bf16 activations)")
    priorities = name_priority(args)
    if priorities and backend != "cuda":
        raise ValueError("--name-priority is a CUDA server option: the MLX server has its own background rule")
    if priorities:
        served = _served_name(args)
        ids = {served, *(str(a).strip() for a in getattr(args, "alias", None) or () if str(a).strip())}
        bad = sorted(set(priorities) - ids)
        if bad:
            raise ValueError(f"--name-priority names {', '.join(bad)}, not --name or an --alias "
                             f"({', '.join(sorted(ids))})")
    confidence = getattr(args, "mtp_confidence", None)
    if confidence is None:
        return
    engine = getattr(family.package, "cuda_engine", None) if backend == "cuda" else None
    if engine is None or "mtp_confidence" not in inspect.signature(engine).parameters:
        raise ValueError(f"--mtp-confidence sets where a CUDA engine's MTP chains stop; {family.title} on "
                         f"{'CUDA' if backend == 'cuda' else 'MLX'} has no such rule")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"--mtp-confidence is a probability from 0 to 1, not {confidence}")


def _cuda_topology(args: argparse.Namespace, family: Any) -> None:
    """Validate rank topology and the selected Flash Next execution envelope before weights are resolved."""

    tp, rank = getattr(args, "tp", 1), getattr(args, "rank", 0)
    from tensorfold.families.qwen4_exp.gguf import resolve_gguf

    gguf = (getattr(family, "model_type", "") in ("qwen4_exp", "qwen3_8_flash_next")
            and resolve_gguf(getattr(args, "model", "")) is not None)
    if type(tp) is not int or tp not in (1, 2, 4, 8) or tp == 8 and not gguf:
        raise ValueError("--tp must be 1, 2 or 4; Flash Next GGUF additionally supports 8")
    if type(rank) is not int or rank < 0:
        raise ValueError("--rank must be a nonnegative integer")
    if tp > 1 and not getattr(args, "master", ""):
        raise ValueError("--tp 2/4/8 needs --master: rank 0's address on the link between the ranks")
    if rank >= max(tp, getattr(args, "expert_ranks", 1)):
        raise ValueError(f"--rank {rank} needs --tp {rank + 1} or more")
    if tp == 4 and not gguf and not getattr(family.package, "CUDA_TP4", False):
        raise ValueError(f"--tp 4: {family.title} runs on one or two GPUs on CUDA")
    port = getattr(args, "master_port", 29551)
    # Every follower connects to this same CLI value; no channel advertises a leader's ephemeral port.
    if max(tp, getattr(args, "expert_ranks", 1)) > 1 and (type(port) is not int or not 1 <= port <= 65535):
        raise ValueError("--master-port must be an integer from 1 to 65535 for multiple ranks")
    if getattr(family, "model_type", "") == "qwen4_exp":
        streams = _cuda_streams(getattr(args, "parallel", "auto"))
        if tp > 1 and streams > 1 and not gguf:
            raise ValueError("--parallel decodes several Flash Next requests together on one GPU; with --tp 2 it "
                             "serves one request at a time for now, so drop --parallel")
        if getattr(args, "vision", False) and (streams < 2 or tp != 1):
            raise ValueError("image input on Flash Next runs on one GPU with --parallel 2 or more")


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
    """``--vision``, ``--vision-urls`` and ``--vision-offload`` as a family's load options."""

    if not getattr(args, "vision", False):
        return {}
    return {"vision": True, "vision_urls": bool(getattr(args, "vision_urls", False)),
            "vision_offload": bool(getattr(args, "vision_offload", False))}


__all__ = ["check", "check_numbers", "name_priority", "vision_options"]
