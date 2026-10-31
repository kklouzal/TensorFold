"""Producer API admission only: no model weights, CUDA initialization or GEMM."""
import importlib.metadata as md
import json
import os
from pathlib import Path

import torch

if torch.cuda.is_initialized():
    raise RuntimeError("CPU producer smoke began with initialized CUDA")

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

recipes = ("NVFP4_DEFAULT_CFG", "NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG",
           "NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG", "NVFP4_FOUR_OVER_SIX_CFG")
for recipe in recipes:
    if not isinstance(getattr(mtq, recipe), dict):
        raise RuntimeError(f"missing ModelOpt recipe: {recipe}")
if not callable(mtq.quantize) or not callable(export_hf_checkpoint):
    raise RuntimeError("producer quantization/export API is unavailable")
if torch.cuda.is_initialized():
    raise RuntimeError("producer API import initialized CUDA")
from tensorfold_foundation_runtime import mapped_runtime  # noqa: E402 - supplied by managed foundation activation.

state = json.loads(Path(os.environ["TF_FOUNDATION_STATE"]).read_text())
providers = mapped_runtime(state["runtime"], allow_vision=True)
print(json.dumps({"scope": "CPU imports/API/recipe availability only; no calibration or model proof",
    "modelopt": md.version("nvidia-modelopt"), "torch": torch.__version__,
    "transformers": md.version("transformers"), "recipes": recipes,
    "cuda_initialized": False, "mapped_providers": providers}, sort_keys=True))
