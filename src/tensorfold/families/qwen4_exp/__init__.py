"""Qwen3.8 Flash Next forward pass and fused decode with exact MTP drafting on Metal or one or two CUDA GPUs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .kv_formats import DTYPES

CUDA_DRAFT_METHODS = ("auto", "none", "mtp", "dspark", "dflash2")
CUDA_ADAPTIVE_SESSIONS = True  # GGUF provider resolves auto; modern CUDA retains its existing one-stream default.
CUDA_KV_DTYPES = DTYPES          # the KV cache dtypes the CUDA engine can allocate (``--kv-dtype``)

MODEL_TYPES = ("qwen4_exp", "qwen3_8_flash_next")   # the second: the name newer exports (Mia-AiLab's NVFP4) carry
TITLE = "Qwen3.8 Flash Next"
LANES = True
# with their MTP head: MLX affine (4-bit the default; oQ4e, oQ5e, 6- and 8-bit read too), EXL3 and NVFP4
MODELS = ("TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP", "turboderp/Qwen3.8-Flash-Next-exl3",
          "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "RadixArk/Qwen3.8-Flash-Next-NVFP4")
NVFP4_MODELS = MODELS[2:]
QUANT_METHODS = {"cuda": ("mlx", "exl3", "modelopt", "gguf")}  # each format retains its own byte reader/provider
EXL3_VARIANT = "any"                           # every EXL3 codebook and width (tensorfold.families.EXL3_VARIANT_ANY)
KERNEL_PACKAGE = "tensorfold.kernels.qwen.flash_next.v1"
KERNEL_VERSION = "v1"
# The CLI sets defaults before MLX starts so expert bindings do not end each command buffer.
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


def rope_parameters(model_dir: str | Path, yarn_factor: float | None = None):
    """Resolve text RoPE without accelerator imports or modifying the checkpoint."""

    from tensorfold.families import read_config

    from .rope import RopeParameters

    return RopeParameters.from_config(read_config(Path(model_dir)), yarn_factor)


def has_mtp(model_dir: Path) -> bool:
    """Whether the checkpoint kept the MTP head's weights (``mtp.*``)."""

    import json

    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    return any(".mtp." in name or name.startswith("mtp.") for name in json.loads(index.read_text())["weight_map"])


def check_quantization(config: dict[str, Any], backend: str) -> None:
    """Refuse from config.json alone an MLX format the kernels do not read: affine 2-8 bits in groups of 32-128."""

    from tensorfold.quantization import checkpoint_specs

    specs = [spec for spec in checkpoint_specs(config).values() if spec is not None]
    if not specs:
        raise ValueError(f"{TITLE} reads MLX affine-quantized weights; this checkpoint has none. Use {MODELS[0]}.")
    formats = {(spec.bits, spec.group_size) for spec in specs}
    if backend == "cuda" and formats != {CUDA_QUANTIZATION}:
        bits, group = CUDA_QUANTIZATION
        raise ValueError(f"{TITLE}'s CUDA kernels read {bits}-bit weights in groups of {group}; this checkpoint has "
                         f"{', '.join(f'{b}-bit g{g}' for b, g in sorted(formats))}. Use {MODELS[0]}.")


def check(model_dir: Path) -> None:
    from tensorfold.families import EXL3_QUANT, OWN_MODEL_HELP, describe_quantization, quant_method, read_config

    config = read_config(model_dir)
    if quant_method(config) == "gguf":
        from .gguf import GGUFCheckpoint, resolve_gguf

        GGUFCheckpoint.open(resolve_gguf(model_dir))
        return
    if quant_method(config) == EXL3_QUANT:
        # an EXL3 pack (the CUDA engine, any codebook and per-tensor width): only the MTP head to report
        if (Path(model_dir) / "model.safetensors.index.json").is_file() and not has_mtp(model_dir):
            print("[tensorfold] this EXL3 checkpoint has no MTP head: decoding without MTP drafts", flush=True)
        return
    if quant_method(config) == "modelopt":
        from tensorfold.vision.qwen_checkpoint import vision_key

        # NVFP4 experts and n-gram tables; other linears are bf16, MXFP8 or block FP8
        found = config.get("quantization") or config.get("quantization_config") or {}
        algo = str(found.get("quant_algo") or "NVFP4").upper()
        # FP8 is read in the MTP drafter's experts (dequantized and re-quantized at load: they only draft) and in the
        # n-gram tables (their own FP8 reader, host_table.FP8Table); FP8 elsewhere stays refused
        def fp8_read(name: str) -> bool:
            return {"mtp", "experts"} <= set(name.split(".")) or ".ple.ple_embedding.ngram_embedding." in name + "."

        layers = {str(v.get("quant_algo", "")).upper() for k, v in (found.get("quantized_layers") or {}).items()
                  if not (str(v.get("quant_algo", "")).upper() == "FP8" and fp8_read(k))}
        algos = layers if algo == "MIXED_PRECISION" else {algo}
        weights = [g.get("weights") or {} for g in (found.get("config_groups") or {}).values()]
        fp4 = {int(w.get("group_size", 16)) for w in weights if int(w.get("num_bits", 4)) == 4}
        if not algos <= {"NVFP4", "W4A16_NVFP4", "MXFP8", "FP8_PB_WO"} or fp4 - {16}:
            raise ValueError(f"TensorFold's Flash Next kernels read NVFP4 (ModelOpt FP4) weights in blocks of 16, the "
                             f"other linears bf16, MXFP8 or 128x128-block FP8 ({', '.join(NVFP4_MODELS)}); "
                             "this checkpoint has "
                             + describe_quantization(config) + f". {OWN_MODEL_HELP}")
        # N-gram tables have their own NVFP4 reader; other non-expert layers would cast packed bytes to bf16.
        outside = sorted(name for name, layer in (found.get("quantized_layers") or {}).items()
                         if "NVFP4" in str(layer.get("quant_algo", "")).upper() and "experts" not in name.split(".")
                         and ".ple.ple_embedding.ngram_embedding." not in name + "."
                         and vision_key(name) is None)
        if outside:
            raise ValueError("TensorFold's Flash Next kernels read NVFP4 in routed experts and n-gram tables only; "
                             f"this checkpoint has it on {len(outside)} other layer(s), e.g. {outside[0]}. "
                             f"{OWN_MODEL_HELP}")
        if (Path(model_dir) / "model.safetensors.index.json").is_file() and not has_mtp(model_dir):
            print("[tensorfold] this NVFP4 checkpoint has no MTP head: decoding without MTP drafts", flush=True)
        return
    check_quantization(config, "mlx")
    # Config-only preflight cannot establish whether the MTP head is missing; wait for weights or their index.
    if ((Path(model_dir) / "model.safetensors.index.json").is_file()
            or any(Path(model_dir).glob("model*.safetensors"))) and not has_mtp(model_dir):
        print(f"[tensorfold] this checkpoint has no MTP head: decoding without MTP drafts ({MODELS[0]} has one)",
              flush=True)


# a table's tensors in the checkpoint: weights, scales and biases of each shard, the only bytes read on the host
_TABLE = r"language_model\.model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.(weight|scales|biases)"


def ple_bytes(model_dir: Path) -> int:
    """Bytes of the checkpoint's n-gram (PLE) tables, which stay on the host (mapped, or on SSD with --ple-on-ssd)."""

    import re

    from tensorfold.families.qwen4_exp.host_table import read_header

    return sum(entry["data_offsets"][1] - entry["data_offsets"][0]
               for path in Path(model_dir).glob("model*.safetensors")
               for name, entry in read_header(path).items() if re.fullmatch(_TABLE, name))


def expert_bytes(model_dir: Path) -> int:
    """Bytes of the decoder layers' routed expert stacks, which --ssd-experts leaves on disk."""

    from tensorfold.streaming.checkpoint import tensor_bytes

    return tensor_bytes(Path(model_dir), lambda name: name.startswith("language_model.model.layers.")
                        and ".mlp.switch_mlp." in name)


def weight_bytes(model_dir: Path, ple_on_ssd: bool = False) -> int:
    """The bytes MLX loads: the checkpoint, less its n-gram tables where the loader keeps them on the host."""

    from tensorfold.families.qwen4_exp.host_table import ngrams_on_host

    size = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    return size - ple_bytes(model_dir) if ngrams_on_host(model_dir, ple_on_ssd) else size


def load(model_dir: Path, *, mtp_drafts: int | None = None, ple_on_ssd: bool = False,
         ssd_experts: float | None = None, **_: Any) -> tuple[Any, Any]:
    if rope_parameters(model_dir).rope_type != "default":
        raise ValueError("Flash Next YaRN text RoPE is supported on CUDA only")
    from tensorfold.families.qwen4_exp.runtime import load as load_runtime

    drafts = mtp_drafts if has_mtp(Path(model_dir)) else 0
    return load_runtime(Path(model_dir), drafts=drafts, ple_on_ssd=ple_on_ssd, ssd_experts=ssd_experts)


def engine_settings(model: Any) -> dict[str, Any]:
    """The widest exact window, prompt chunks the memory allows (8,192 where tensor units run MLX's gathers)."""

    from tensorfold.families.qwen3_5 import tensor_units

    width = int(getattr(model, "exact_width", 1) or 1)
    steps = (8192, 4096, 2048) if tensor_units() else (4096, 2048)
    return {"max_rows": width, "max_draft": max(0, width - 1), "prefill_steps": steps}


def kernel_version(model: Any) -> str:
    """Include the loaded model's prompt-attention selection modes in its kernel fingerprint."""

    from tensorfold.families import families, kernel_source_version

    modes = [str(int(bool(getattr(layer.self_attn, "kernel_select", False))))
             for layer in getattr(model, "layers", ()) if hasattr(layer, "self_attn")]
    source = kernel_source_version(families()["qwen4_exp"])
    prefill = getattr(model, "prefill_key", None)                     # the prefill path, matmul route and GPU
    return f"{source}|prompt_attention={','.join(modes)}" + (f"|{prefill}" if prefill else "")


# the CUDA engine reads MLX affine weights of this (bits, group size), or NVFP4 (ModelOpt) routed experts
CUDA_QUANTIZATION = (4, 32)
CUDA_DECODE_SHARE = True           # --parallel rounds size their prompt pass by --decode-share (0: whole passes)
CUDA_PREFILL_FP8 = True            # --prefill-fp8: an NVFP4 checkpoint's MXFP8 linears have an FP8 prompt kernel
CUDA_KV_PAIRS = True               # independent key/value storage, including asymmetric quantization


def check_vram_experts(model_dir: str | Path, gib: float | str, *, tp: int = 1) -> None:
    from .ram_experts import check

    check(model_dir, gib, tp=tp)

def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                mtp_confidence: float | None = None, context: int | None = None, ple_on_ssd: bool = False,
                kv_dtype: str = "bf16", decode_share: float | None = None, yarn_factor: float | None = None,
                vram_experts: float | str | None = None, kv_key_dtype: str | None = None,
                kv_value_dtype: str | None = None,
                expert_ranks: int = 1, expert_shards=None,
                vram_experts_pin: str = "off",
                **options: Any):
    """Verify MTP on one or two CUDA GPUs; start rank 1 first for ``tp=2``, with any ``CUDA_KV_DTYPES`` KV storage."""

    if "ram_experts" in options:
        raise TypeError("ram_experts was renamed to vram_experts; specify the GPU expert-cache budget with that keyword")
    from tensorfold.cuda.exl3.format import is_exl3

    if is_exl3(Path(model_dir)):
        print("[tensorfold] EXL3 packs are experimental: replies are exact; see "
              "docs/recipes/qwen3.8-flash-next.md#exl3-checkpoints-experimental for how they compare", flush=True)
    from .gguf import resolve_gguf
    gguf = resolve_gguf(model_dir) is not None
    family_manifest = options.get("family_manifest")
    sm70 = gguf or family_manifest is not None
    if gguf and family_manifest is not None:
        raise ValueError("family_manifest selects the explicit SM70 NVFP4 recipe; a GGUF source has its own representation")
    history = options.get("indexer_history", "gpu")
    if history not in ("gpu", "ram"):
        raise ValueError("indexer_history must be gpu or ram")
    if history == "ram" and not sm70:
        raise ValueError("indexer_history ram requires the GGUF SM70 provider")
    serving_root = options.get("serving_profile_root")
    execution_manifest = options.get("execution_profile_manifest")
    position_manifest = options.get("position_profile_manifest")
    media_manifest = options.get("media_profile_manifest")
    if media_manifest is not None and (not gguf or history != "gpu" or tp != 8
            or not options.get("vision", False) or options.get("graphs", True)):
        raise ValueError("media_profile_manifest requires --vision, --no-graphs and resident Flash Next GGUF TP8 GPU histories")
    if position_manifest is not None and not sm70:
        raise ValueError("position_profile_manifest requires an explicit Flash Next SM70 provider")
    if (execution_manifest is not None or serving_root is not None) and not (gguf and history == "gpu" and vram_experts is None):
        raise ValueError("execution_profile_manifest requires resident Flash Next GGUF GPU histories on SM70")
    method = options.get("draft_method", "auto")
    if method not in CUDA_DRAFT_METHODS:
        raise ValueError("Flash Next draft_method must be auto, none, mtp, dspark or dflash2")
    if method in ("dspark", "dflash2") and not gguf:
        label = "DSpark" if method == "dspark" else "DFlash2"
        raise ValueError(f"{label} currently requires the Flash Next GGUF SM70 provider")
    if method == "none":
        no_drafts, drafter = True, ""
    if drafter and not sm70:
        raise ValueError(f"{TITLE} drafts with its own MTP head on CUDA: a separate draft model does not apply")
    from .cuda import CONFIDENCE, DEPTH
    from .cuda.engine import FlashNextEngine
    from .kv_formats import get_pair

    pair = get_pair(kv_dtype, kv_key_dtype, kv_value_dtype)  # before weights or Torch are loaded
    if not sm70 and any(side.compact is not None for side in (pair.key, pair.value)):
        raise ValueError("compact Planar4/Iso4 KV requires the GGUF SM70 execution provider")
    depth = 0 if no_drafts else DEPTH if mtp_drafts is None else int(mtp_drafts)
    if method == "mtp" and depth == 0:
        raise ValueError("explicit MTP requires a positive maximum depth; use runtime proposal_limit 0 to retain "
                         "warm MTP capability, or select --draft-method none to disable it")
    if method in ("dspark", "dflash2") and not drafter:
        raise ValueError("DSpark/DFlash2 require their checkpoint directory through --drafter")
    if depth and not drafter and not has_mtp(Path(model_dir)):
        raise ValueError(f"this checkpoint has no MTP head, which {TITLE}'s CUDA engine drafts with ({MODELS[0]} "
                         "has one): without it every round would decode one token. Serve a checkpoint with the "
                         "head, or pass --no-drafts for the serial reference")
    confidence = CONFIDENCE if mtp_confidence is None else float(mtp_confidence)
    if type(expert_ranks) is not int or not 1 <= expert_ranks <= 4:
        raise ValueError("--expert-ranks must be 1 through 4")
    if expert_shards is not None:
        from tensorfold.cuda.capacity import config
        from tensorfold.cuda.expert_parallel import ownership
        ownership(expert_ranks, expert_shards, rank, int(config(model_dir)["num_experts"]))
    # Prepare every conversion/optional argument before acquiring a live expert
    # session. The guarded handoff below also covers constructor allocation and
    # argument-unpacking failures before a consumer can own its rollback.
    directory = Path(model_dir)
    parallel = options.get("parallel")
    parallel_auto = options.get("parallel_auto", parallel is None or parallel == "auto")
    if type(parallel_auto) is not bool:
        raise ValueError("parallel_auto must be boolean")
    automatic = parallel_auto
    if sm70 and not automatic and (type(parallel) is not int or parallel < 1):
        raise ValueError("the GGUF session ceiling must be a positive integer or auto")
    streams = 1 if automatic else int(parallel)
    engine_options = {"depth": depth, "confidence": confidence, "max_len": context,
                      "context_explicit": options.get("context_explicit"), "tp": int(tp), "rank": int(rank),
                      "master": master, "port": int(master_port), "streams": max(1, streams),
                      "ple_on_ssd": ple_on_ssd, "kv_dtype": kv_dtype,
                      "kv_key_dtype": kv_key_dtype, "kv_value_dtype": kv_value_dtype,
                      "share": 0.0 if decode_share is None else float(decode_share),
                      "vision": bool(options.get("vision", False)),
                      "vision_urls": bool(options.get("vision_urls", False)), "yarn_factor": yarn_factor,
                      "vram_experts": vram_experts}
    if family_manifest is not None:
        engine_options["family_manifest"] = Path(family_manifest)
        if options.get("_prepared_family_plan") is not None:
            engine_options["_prepared_family_plan"] = options["_prepared_family_plan"]
    if position_manifest is not None:
        engine_options["position_profile_manifest"] = Path(position_manifest)
    if media_manifest is not None:
        engine_options["media_profile_manifest"] = Path(media_manifest)
    if "graphs" in options:
        engine_options["graphs"] = options["graphs"]
    if sm70 and history == "gpu":
        # Preserve auto intent through the CLI's ordinary CUDA one-stream
        # default. Integer settings constrain the solved full-context capacity.
        engine_options.update(adaptive_sessions=True,
                              session_ceiling=None if automatic else int(parallel))
        if options.get("execution_rows") is not None:
            engine_options["execution_rows"] = options["execution_rows"]
        if serving_root is not None:
            engine_options["serving_profile_root"] = Path(serving_root)
        if execution_manifest is not None:
            engine_options["execution_profile_manifest"] = Path(execution_manifest)
    if vram_experts_pin != "off":
        engine_options["vram_experts_pin"] = vram_experts_pin
    if history != "gpu":
        engine_options["indexer_history"] = history
    if sm70 and drafter:
        from .cuda.speculation import checkpoint_method
        kind = checkpoint_method(drafter)
        if method != "auto" and method != kind:
            raise ValueError(f"--draft-method {method} does not match the selected {kind} drafter checkpoint")
        key = {"dspark": "dspark_model_dir", "dflash2": "dflash2_model_dir", "mtp": "mtp_model_dir"}[kind]
        engine_options[key] = Path(drafter)
        if kind == "dflash2":
            if (tp != 8 or history != "gpu" or execution_manifest is None
                    or bool(options.get("vision", False)) and media_manifest is None or expert_ranks != 1 or depth > 3):
                raise ValueError("DFlash2 requires fixed TP8 text-only resident GGUF GPU histories, "
                                 "an execution-profile manifest with exact draft_pairing, and at most three proposals")
    elif method in ("dspark", "dflash2"):
        raise ValueError("DSpark/DFlash2 require their checkpoint directory through --drafter")
    if "draft_vocab" in options:
        engine_options["draft_vocab"] = options["draft_vocab"]
    full_windows = options.get("kv_full_windows")
    if full_windows == "auto":
        if not (gguf and history == "gpu" and vram_experts is None):
            raise ValueError("kv_full_windows auto requires resident GGUF GPU histories and automatic capacity")
    elif full_windows is not None:
        engine_options["kv_full_windows"] = full_windows
    for name in ("eos_token_ids", "chat_template_file", "chat_template_name"):
        if options.get(name) is not None:
            engine_options[name] = options[name]
    if expert_ranks > 1:
        if tp != 1:
            raise ValueError("expert parallelism and tensor parallelism cannot be combined")
        from tensorfold.cleanup import rollback

        from .cuda.expert_parallel import Follower, Session
        session = Session(model_dir, ranks=expert_ranks, shards=expert_shards, rank=rank,
                          master=master or "127.0.0.1", port=master_port, budget=vram_experts, mtp=depth > 0,
                          pin_mode=vram_experts_pin,
                          **({"eos_token_ids": options["eos_token_ids"]} if options.get("eos_token_ids") is not None else {}))
        try:
            if rank:
                return Follower(session)
            engine_options["expert_session"] = session
            return FlashNextEngine(directory, **engine_options)
        except BaseException as error:  # noqa: BLE001 - rollback preserves cancellation and the original failure
            rollback(session, error, lambda: session.close(abort=True))
    return FlashNextEngine(directory, **engine_options)
