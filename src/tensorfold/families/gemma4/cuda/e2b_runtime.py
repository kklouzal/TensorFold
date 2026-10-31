"""Qualified E2B runtime boundary; the existing launcher owns provider proof."""
from pathlib import Path
import os

TORCH_COMMIT = "ef46daf1aa35294a3cb297ef3f59b5a6463063f8"
TRANSFORMERS_VERSION = "5.17.0"
TRITON_VERSION = "3.9.0+gitfc8d31a5"


def require_runtime(torch, transformers_version, triton_distribution_version):
    """Refuse unsupported execution before checkpoint allocation or global policy.

    The phase/state descriptors are a launch contract, not a replacement for
    foundation_guard's seal, driver and mapped-provider verification. Triton's
    distribution version carries its pinned git suffix; module.__version__ does not.
    """
    def refuse(detail):
        raise RuntimeError(
            "Gemma E2B requires the protected GB10 CUDA image "
            "(SM121, qualified Torch/Transformers/Triton). " + detail +
            " Build/run deploy/gb10 with its cuda_entrypoint; keep the GPU entrypoint intact.")

    state = os.environ.get("TF_FOUNDATION_STATE")
    if not state or not Path(state).is_file() or os.environ.get("TF_FOUNDATION_RUNTIME_PHASE") != "cuda":
        refuse("The managed CUDA bootstrap descriptor is missing. ")
    if (getattr(torch.version, "git_version", None) != TORCH_COMMIT
            or getattr(torch.version, "cuda", None) != "13.4"):
        refuse("The qualified Torch commit/CUDA13.4 build is required. ")
    if transformers_version != TRANSFORMERS_VERSION or triton_distribution_version != TRITON_VERSION:
        refuse("Use Transformers5.17.0 and Triton3.9.0+gitfc8d31a5 from the pinned image. ")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1):
        refuse("Only the qualified GB10 SM121 target is admitted. ")
    try:
        matmul = torch.backends.cuda.matmul
        reduced = matmul.allow_bf16_reduced_precision_reduction
        split = matmul.allow_bf16_reduced_precision_reduction_split_k
    except AttributeError:
        refuse("Native BF16 reduction/split-K controls are unavailable. ")
    if type(reduced) is not bool or type(split) is not bool:
        refuse("Native BF16 reduction/split-K controls have an unsupported API. ")
