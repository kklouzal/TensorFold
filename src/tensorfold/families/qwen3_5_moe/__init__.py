"""Qwen3.6 MoE (qwen3_5_moe): DeltaNet, attention, and routed experts."""
# CUDA uses the checkpoint's MTP; Macs retain DFlash by default and offer explicit MTP on the row decoder.

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen3_5_moe",)
TITLE = "Qwen3.6 MoE"
LANES = True
# MLX 4-bit, groups of 64, routers 8-bit, MTP layer in mtp-4bit.safetensors (mlx-community's files take it too)
MODELS = ("TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP", "mlx-community/Qwen3.6-35B-A3B-4bit")
REQUIRED_FILES = {MODELS[0]: ("mtp-4bit.safetensors",)}
DRAFTER = "z-lab/Qwen3.6-35B-A3B-DFlash"      # Preserve the fork's implicit Mac drafter pending native MTP selection.
CUDA_DRAFTER = ""                             # CUDA drafts with the checkpoint's MTP layer.
DFLASH = DRAFTER
KERNEL_PACKAGE = "tensorfold.kernels.qwen.dense.v1"
KERNEL_VERSION = "v1"
MLX_MODEL_FILE = True
# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)
CUDA_PREFILL_FP8 = True            # --prefill-fp8: the attention and DeltaNet projections' FP8 prompt kernel


def check(model_dir: str | Path) -> None:
    """One GPU, MLX 4-bit weights in groups of 64."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quantization, read_config

    if quantization(read_config(model_dir)) != CUDA_QUANTIZATION:
        raise ValueError(f"{TITLE}'s CUDA engine reads MLX 4-bit weights in groups of 64 ({MODELS[0]}); this "
                         f"checkpoint has {describe_quantization(read_config(model_dir))}. {OWN_MODEL_HELP}")


def mtp_file(model_dir: Path) -> Path | None:
    """The MTP layer: TF_QWEN36_MTP (a path, or 0 for none), the checkpoint's side file, else the tested repo's."""

    import os

    from tensorfold import hub

    from .mtp import MTP_FILE, find

    choice = os.environ.get("TF_QWEN36_MTP", "")
    found = find(Path(model_dir), choice)
    if found is not None or choice == "0":
        return found
    tested = hub.cached(MODELS[0])          # mlx-community's 4-bit shards are byte-identical to this repo's
    return tested / MTP_FILE if tested is not None and (tested / MTP_FILE).is_file() else None


def load(model_dir: Path, *, drafter: str = "", drafter_bits: int = 4, mtp_drafts: int | None = None,
         trust_model_code: bool = False, mtp_head: str | None = None, **_: Any) -> tuple[Any, Any]:
    """The fork row decoder; explicit positive MTP caps opt into the new head family. Text only."""
    # Lane kernels take no routed experts. row_forward.moe computes one row's bits at any width.
    if mtp_drafts is not None and (type(mtp_drafts) is not int or mtp_drafts < 0):
        raise ValueError("MTP draft cap must be a nonnegative integer or None")

    from tensorfold.families.qwen3_5 import lane_family, load_lane_model

    if not drafter and mtp_drafts is not None and mtp_drafts > 0:
        from tensorfold.cleanup import raise_failures

    model, tokenizer = load_lane_model(Path(model_dir), trust_model_code=trust_model_code)
    if drafter or mtp_drafts is None or mtp_drafts == 0:
        return lane_family(model, lanes=False, drafter=drafter, drafter_bits=drafter_bits, title=TITLE,
                           use=MODELS[1]), tokenizer
    try:
        from functools import partial

        from .family import Qwen36Family

        # The CLI binds external head bytes (or explicit absence) in its identity
        # receipt; never reselect an environment/cache path after that capture.
        path = (None if mtp_head == "" else
                Path(mtp_head) if mtp_head is not None else mtp_file(Path(model_dir)))
        if path is None:
            print(f"[tensorfold] {TITLE}: no MTP layer for this checkpoint, so one token a round. To draft, fetch it "
                  f"(0.5 GB): hf download {MODELS[0]} mtp-4bit.safetensors", flush=True)
        make = partial(Qwen36Family, mtp_path=path, drafts=mtp_drafts)
        return lane_family(model, lanes=False, drafter="", drafter_bits=drafter_bits, title=TITLE, use=MODELS[0],
                           make=make), tokenizer
    except BaseException as primary:
        # Ownership has not reached the caller. Keep the raw model in this
        # traceback until both default-stream fences retire it or teardown.
        cause = BaseException.__cause__.__get__(primary)
        context = BaseException.__context__.__get__(primary)
        suppressed = BaseException.__suppress_context__.__get__(primary)
        import_error = gpu_error = cpu_error = None
        try:
            import mlx.core as mx
        except BaseException as failure:
            import_error = failure
        else:
            # LaneEngine.drain's supported streams; one failure cannot skip the other.
            try:
                mx.synchronize(mx.default_stream(mx.gpu))
            except BaseException as failure:
                gpu_error = failure
            try:
                mx.synchronize(mx.default_stream(mx.cpu))
            except BaseException as failure:
                cpu_error = failure
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        BaseException.__suppress_context__.__set__(primary, suppressed)
        raise_failures(primary, [error for error in (import_error, gpu_error, cpu_error) if error is not None])


def engine_settings(model: Any) -> dict[str, Any]:
    from tensorfold.families.qwen3_5 import engine_settings as dense

    return dense(model)


def kernel_version(model: Any) -> str:
    """Names the row decoder that computed a prefix snapshot; TF_MOE_ROWS's two paths give other bits."""

    import hashlib

    from tensorfold.kernels.qwen.dense.v1 import row_forward, row_matmul

    parts = [getattr(row_matmul.BACKEND, "name", "simd_qmm"), f"row_attention={row_forward.ROW_ATTENTION}",
             f"moe_rows={row_forward.MOE_ROWS}",
             *(path.read_text() for path in sorted(Path(row_forward.__file__).parent.glob("*.py")))]
    return f"qwen3_5_moe-{KERNEL_VERSION}-" + hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                context: int | None = None, **options: Any):
    """One GPU: MTP chains verified exactly, or the serial reference when no_drafts is set."""
    # parallel above 1 decodes that many requests together.

    if drafter:
        raise ValueError(f"{TITLE} drafts with its own MTP layer on CUDA: a separate draft model does not apply")
    if int(tp) != 1:
        raise ValueError(f"{TITLE} runs on one GPU: drop --tp")
    from .cuda import DEPTH
    from .cuda.engine import Qwen36Engine

    depth = 0 if no_drafts else DEPTH if mtp_drafts is None else int(mtp_drafts)
    streams = max(1, int(options.get("parallel") or 1))
    if streams > 1 and not 0 <= depth <= 15:
        raise ValueError(f"--parallel verifies up to 16 rows a stream: --mtp-drafts 0 to 15, not {depth}")
    return Qwen36Engine(Path(model_dir), depth=depth, context=context, context_explicit=options.get("context_explicit"),
                        streams=streams)
