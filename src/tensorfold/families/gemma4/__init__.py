"""Gemma 4 (model_type ``gemma4`` or text-only ``gemma4_text``): the MoE checkpoints, on the lane engine."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

MLX_MODEL_FILE = True

MODEL_TYPES = ("gemma4", "gemma4_text")
TITLE = "Gemma 4"
LANES = True
MODELS = ("mlx-community/gemma-4-26b-a4b-it-4bit",)
CUDA_DRAFTER = "z-lab/gemma-4-26B-A4B-it-DFlash"
CUDA_DRAFTER_BITS = True
CUDA_PREFIX_CACHE_BYTES = True
CUDA_LOSSLESS_HEAD = True
CUDA_DELTA4_HEAD = True
CUDA_PREPARE_GLOBAL_PROMPT = True
CUDA_DFLASH_HEAD_W4A16 = True
CUDA_CLIP_SAMPLED_VERIFICATION = True
CUDA_SPARK_GLOBAL_GQA = True
CUDA_SPARK_SAMPLED_Q = True
CUDA_DRAFT_ROWS = True
QUANT_METHODS = {"cuda": ("mlx", "compressed-tensors", "modelopt", None), "mlx": ("mlx",)}
KERNEL_PACKAGE = "tensorfold.kernels.gemma.v1"
KERNEL_VERSION = "v1"
# the decode projections run another family's matmul kernels
KERNEL_DEPENDENCIES = ("tensorfold.kernels.qwen.dense.v1.lane_qmm", "tensorfold.kernels.nemotron.lightning.v1.rows")


def check(model_dir: str | Path) -> None:
    """Validate the common MoE layout; backend-specific quantization is checked separately."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quant_method, quantization, read_config

    config = read_config(model_dir)
    from .cuda.e2b_metadata import E2BConfig, is_dense_gemma
    if is_dense_gemma(config):
        E2BConfig.from_checkpoint(config)
        return
    if quant_method(config) in ("compressed-tensors", "modelopt"):
        check_metadata(config, "cuda")
        return
    text = config.get("text_config") or config
    experts, top_k = int(text.get("num_experts") or 0), int(text.get("top_k_experts") or 0)
    missing = [what for what, ok in (
        ("a MoE block in every layer", bool(text.get("enable_moe_block"))),
        ("no per-layer inputs", not int(text.get("hidden_size_per_layer_input") or 0)),
        ("no shared-KV layers", not int(text.get("num_kv_shared_layers") or 0)),
        ("head dims a multiple of 64", all(int(text.get(k) or 64) % 64 == 0 for k in ("head_dim", "global_head_dim"))),
        ("a positive multiple of 32 experts and top_k in 1..expert count",
         experts > 0 and experts % 32 == 0 and 1 <= top_k <= experts),
    ) if not ok]
    if missing:
        raise ValueError(f"TensorFold's Gemma 4 kernels cover the MoE checkpoints ({MODELS[0]}); this one lacks "
                         + ", ".join(missing) + f". {OWN_MODEL_HELP}")
    bits, group = quantization(config)
    if bits != 4 or group not in (32, 64):
        raise ValueError(f"Gemma 4's kernels read MLX 4-bit weights in groups of 32 or 64 ({MODELS[0]}); this "
                         f"checkpoint has {describe_quantization(config)}. {OWN_MODEL_HELP}")
    # CUDA also reads original mixed4/6 checkpoints. Backend discovery invokes
    # check_quantization before this shared, backend-independent check.


def check_metadata(config: dict, backend: str) -> None:
    """Native CT activation math and architecture are declared before downloads."""
    from tensorfold.families import quant_method
    from .cuda.e2b_metadata import E2BConfig, is_dense_gemma
    if is_dense_gemma(config):
        if backend != "cuda":
            raise ValueError("Gemma dense E2B/E4B/31B is implemented by the CUDA backend")
        E2BConfig.from_checkpoint(config)
        return
    if backend == "cuda" and quant_method(config) == "modelopt":
        from .cuda.weights import Config
        from tensorfold.cuda.nvfp4.format import config_block
        Config.from_dict(config)
        block = config_block(config)
        if (not block or block.get('quant_algo') != 'NVFP4' or block.get('kv_cache_scheme') is not None
                or block.get('has_zero_point', False) is not False
                or block.get('transform_config') or block.get('sparsity_config') or block.get('pre_quant_scale')
                or block.get('lora_rank')):
            raise ValueError('Gemma4 MoE ModelOpt requires untransformed NVFP4 with BF16 KV and native BF16 exceptions')
        return
    if backend == "cuda" and quant_method(config) is None:
        raise ValueError("Unquantized Gemma4 CUDA support requires the complete dense E2B/E4B architecture")
    if backend == "cuda" and quant_method(config) == "compressed-tensors":
        from .cuda.weights import Config
        Config.from_dict(config)
        block = config.get("quantization_config") or {}
        if (block.get("format") != "nvfp4-pack-quantized" or block.get("kv_cache_scheme") is not None
                or block.get("transform_config") or block.get("sparsity_config")):
            raise ValueError("Gemma4 CUDA reads original untransformed CT NVFP4 with BF16 KV")


def check_quantization(config: dict, backend: str) -> None:
    from tensorfold.families import OWN_MODEL_HELP

    found = config.get("quantization") or config.get("quantization_config") or {}
    if backend == "cuda":
        from .cuda.weights import quant_spec, Config, tensor_contract

        cfg = Config.from_dict(config)
        for name in tensor_contract(cfg)[0]:
            quant_spec(config, name)
        return
    for name, spec in found.items():
        if isinstance(spec, dict) and not (name.endswith("router.proj") and int(spec.get("bits", 0)) == 8):
            raise ValueError(f"Gemma 4's kernels read 4-bit projections with an 8-bit router; {name} has "
                             f"{spec.get('bits')}-bit weights. {OWN_MODEL_HELP}")


def check_cuda_modelopt_moe_metadata(config: dict, *, flag: str) -> None:
    """Shared actual-checkpoint boundary for opt-in MoE CUDA features."""
    from tensorfold.families import quant_method

    text = config.get("text_config") or config
    if (not isinstance(text, dict) or quant_method(config) != "modelopt"
            or text.get("enable_moe_block") is not True):
        raise ValueError(f"{flag} requires Gemma 4 ModelOpt MoE on CUDA; dense Gemma and other formats are unsupported")


def check_cuda_delta4_moe_metadata(config: dict, *, flag: str) -> None:
    """Delta4's existing loader supports CT NVFP4 MoE, separately from Sym8."""
    from tensorfold.families import quant_method

    text = config.get('text_config') or config
    block = config.get('quantization_config') or {}
    if (config.get('model_type') not in MODEL_TYPES or not isinstance(text, dict)
            or text.get('enable_moe_block') is not True or quant_method(config) != 'compressed-tensors'
            or not isinstance(block, dict)
            or block.get('format') != 'nvfp4-pack-quantized'):
        raise ValueError(f'{flag} requires Gemma 4 CT NVFP4 MoE on CUDA; dense Gemma and other formats are unsupported')



def check_cuda_prepared_global_prompt_metadata(config: dict, *, flag: str) -> None:
    """Expose only the existing native MoE prompt-scratch constructor path."""
    from tensorfold.families import quant_method

    if config.get('model_type') not in MODEL_TYPES:
        raise ValueError(f'{flag} requires Gemma 4 CT NVFP4 or ModelOpt MoE on CUDA')
    method = quant_method(config)
    if method == 'compressed-tensors':
        check_cuda_delta4_moe_metadata(config, flag=flag)
    elif method == 'modelopt':
        check_cuda_modelopt_moe_metadata(config, flag=flag)
    else:
        raise ValueError(f'{flag} requires Gemma 4 CT NVFP4 or ModelOpt MoE on CUDA; affine and dense formats are unsupported')


def check_cuda_clipped_verification_metadata(config: dict, *, flag: str) -> None:
    """Clipping uses the existing non-dense Gemma CUDA DFlash loop."""
    from .cuda.e2b_metadata import is_dense_gemma

    if config.get('model_type') not in MODEL_TYPES or is_dense_gemma(config):
        raise ValueError(f'{flag} requires Gemma 4 MoE on CUDA with enabled DFlash')


def check_cuda_prefix_cache_metadata(config, *, flag="--prefix-cache-bytes"):
    """The shared archive belongs to the MoE engine, not the dense E2B route."""
    from .cuda.e2b_metadata import is_dense_gemma
    text = config.get("text_config") or config
    if (config.get("model_type") not in MODEL_TYPES or not isinstance(text, dict)
            or text.get("enable_moe_block") is not True or is_dense_gemma(config)):
        raise ValueError(f"{flag} requires Gemma 4 MoE on CUDA")


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0,
                master: str = "", master_port: int = 29551, no_drafts: bool = False,
                context: int | None = None, **options: Any):
    """Original mixed4/6 affine or CT NVFP4 target, DFlash (BF16, or private FP8 copies at --drafter-bits 8) and
    native image/video input."""
    if tp != 1 or rank != 0:
        raise ValueError("Gemma4 CUDA currently runs on one GPU; use --tp 1 --rank 0")
    allowed = {"context_explicit", "parallel", "draft_rows", "drafter_bits", "vision", "vision_urls",
               "vision_offload", "eos_token_ids", "chat_template_file", "chat_template_name", "lossless_head",
               "spark_global_gqa", "spark_sampled_q", "delta4_head", "prepare_global_prompt", "dflash_head_w4a16", "clip_sampled_verification", "prefix_cache_bytes"}
    unsupported = sorted(set(options) - allowed)
    if unsupported:
        raise ValueError("Gemma4 CUDA does not implement: " + ", ".join(unsupported))
    cache_bytes = options.get("prefix_cache_bytes")
    if cache_bytes is not None and (type(cache_bytes) is not int or not 0 <= cache_bytes <= sys.maxsize):
        raise ValueError("prefix_cache_bytes must be a nonnegative native byte count")
    if options.get("vision_offload"):
        raise ValueError("Gemma4's vision tower stays resident; drop --vision-offload")
    clip_sampled_verification = options.get('clip_sampled_verification', False)
    if type(clip_sampled_verification) is not bool:
        raise ValueError('clip_sampled_verification must be a bool')
    if clip_sampled_verification and (not drafter or no_drafts):
        raise ValueError('--clip-sampled-verification requires enabled DFlash')
    dflash_head_w4a16 = options.get('dflash_head_w4a16', False)
    if type(dflash_head_w4a16) is not bool:
        raise ValueError('dflash_head_w4a16 must be a bool')
    if dflash_head_w4a16 and (not drafter or no_drafts or type(options.get('drafter_bits',16)) is not int
                            or options.get('drafter_bits',16)!=8):
        raise ValueError('--dflash-head-w4a16 requires enabled DFlash and --drafter-bits 8')
    prepare_global_prompt = options.get('prepare_global_prompt', False)
    if type(prepare_global_prompt) is not bool:
        raise ValueError('prepare_global_prompt must be a bool')
    delta4_head = options.get('delta4_head', False)
    if type(delta4_head) is not bool:
        raise ValueError('delta4_head must be a bool')
    if delta4_head and options.get('lossless_head', False):
        raise ValueError('--delta4-head and --lossless-head are mutually exclusive')
    from .cuda.e2b_metadata import is_dense_gemma
    from tensorfold.families import read_config
    raw = read_config(model_dir)
    head_options = {}
    if cache_bytes is not None:
        check_cuda_prefix_cache_metadata(raw)
        head_options["prefix_cache_bytes"] = cache_bytes
    if clip_sampled_verification:
        check_cuda_clipped_verification_metadata(raw, flag="--clip-sampled-verification")
        head_options["clip_sampled_verification"] = True
    if dflash_head_w4a16:
        check_cuda_prepared_global_prompt_metadata(raw, flag='--dflash-head-w4a16')
        head_options['dflash_head_w4a16'] = True
    if prepare_global_prompt:
        check_cuda_prepared_global_prompt_metadata(raw, flag='--prepare-global-prompt')
        head_options['prepare_global_prompt'] = True
    if delta4_head:
        check_cuda_delta4_moe_metadata(raw, flag='--delta4-head')
        head_options['delta4_head'] = True
    for option in ("lossless_head", "spark_global_gqa", "spark_sampled_q"):
        value = options.get(option, False)
        if type(value) is not bool:
            raise ValueError(f"{option} must be a bool")
        if value:
            check_cuda_modelopt_moe_metadata(raw, flag="--" + option.replace("_", "-"))
            head_options[option] = True
    if is_dense_gemma(raw):
        from .cuda.e2b_engine import E2BEngine
        rows = options.get("draft_rows")
        return E2BEngine(Path(model_dir), drafter=drafter, no_drafts=no_drafts,
                         depth=None if rows is None else rows - 1, context=context,
                         streams=options.get("parallel", 1),
                         **{name: options[name] for name in ("context_explicit", "drafter_bits", "vision",
                                                           "vision_urls", "eos_token_ids", "chat_template_file",
                                                           "chat_template_name") if name in options})
    if options.get("vision_urls"):
        raise ValueError("Gemma4 currently accepts embedded images/video; drop --vision-urls")
    from tensorfold.families import quant_method
    if quant_method(raw) in ("compressed-tensors", "modelopt"):
        from tensorfold.cuda import precision
        if precision.mode() != precision.CHECKPOINT:
            raise ValueError("Gemma4 native NVFP4 requires --precision checkpoint with calibrated activation math")
    from .cuda.engine import Gemma4Engine

    rows = options.get("draft_rows")
    return Gemma4Engine(Path(model_dir), drafter=drafter, no_drafts=no_drafts,
                        depth=None if rows is None else rows - 1, context=context,
                        streams=options.get("parallel", 1), **head_options,
                        **{name: options[name] for name in ("context_explicit", "drafter_bits", "vision",
                              "eos_token_ids", "chat_template_file", "chat_template_name") if name in options})


def load(model_dir: Path, *, lane_kernels: str = "auto", drafter: str = "", drafter_bits: int = 8,
         trust_model_code: bool = False, prefix_cache_bytes: int | None = None, **_: Any) -> tuple[Any, Any]:
    """MLX's qmv loop a row by default; ``lane_kernels`` "on": the lane matmul; ``drafter``: a DFlash model's chains."""

    if prefix_cache_bytes is not None:
        raise ValueError("prefix_cache_bytes requires Gemma 4 MoE on CUDA")
    from tensorfold.families import read_config
    from .cuda.e2b_metadata import is_dense_gemma
    if is_dense_gemma(read_config(model_dir)):
        raise ValueError("Gemma E2B/E4B requires the CUDA backend")
    from tensorfold.families.gemma4.model import load as load_model

    backend = "lane" if str(lane_kernels) == "on" else "rows"
    return load_model(Path(model_dir), backend=backend, drafter=drafter, drafter_bits=drafter_bits,
                      trust_model_code=trust_model_code)


def cuda_drafter(model_dir):
    from tensorfold.families import read_config
    from .cuda.e2b_metadata import is_dense_gemma
    from .cuda.e2b_metadata import E2BConfig
    raw = read_config(model_dir)
    if is_dense_gemma(raw):
        return "rasyosef/gemma-4-E2B-it-dflash2" if E2BConfig.from_checkpoint(raw).profile == "E2B" else ""
    from tensorfold.families import quant_method
    return "" if quant_method(raw) == "modelopt" else CUDA_DRAFTER


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}
