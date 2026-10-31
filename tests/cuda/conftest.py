"""The CUDA engines' tests: collected only where PyTorch sees an NVIDIA GPU (DGX Spark, in NVIDIA's container)."""

import gc
import importlib.util
import os
import sys

import pytest

# GLM's engines keep the MTP head beside DFlash2 here (TF_GLM_MTP=1), so both drafters stay under test
os.environ.setdefault("TF_GLM_MTP", "1")


def _cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


collect_ignore_glob = [] if _cuda() else ["test_*.py"]


@pytest.fixture(autouse=True, scope="module")
def _give_back_gpu_memory():
    """After each module: free its tensors and torch's cached blocks, so the next engine admits as a fresh process."""

    yield
    if not _cuda():
        return
    import torch

    if torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def _pdl_policy():
    """Each test leaves the process-wide PDL launch policy (``tensorfold.cuda.pdl.ON``) as it found it: a concurrent
    Flash Next decoder sets it at construction and each round, and every later launch of a shared kernel follows it."""

    module = sys.modules.get("tensorfold.cuda.pdl")
    before = module.ON if module is not None else False
    yield
    module = sys.modules.get("tensorfold.cuda.pdl")
    if module is not None:
        module.ON = before


@pytest.fixture(params=[False, True], ids=["bf16", "fp8"])
def fp8(request):
    """Each test that takes it once with bf16 prompts (the default) and once with --prefill-fp8."""

    from tensorfold.cuda import prompt_precision

    with prompt_precision.using(request.param):
        yield request.param
