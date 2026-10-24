"""Metadata-only boundary for the private pinned single-query attention candidate.

The caller must check this before graph-state preparation. Returning false means
the existing Torch attention path, and eager execution when no graph is qualified.
"""

from collections.abc import Mapping
from functools import lru_cache

REQUIRED_IDENTITY = {
    "torch": "2.14.1+cu126",
    "torch_git": "5c4886908584029761b579af026dcfb627c84070",
    "cuda": "12.6",
    "cublas": "12.6.4.1",
    "blas_backend": "_BlasBackend.Cublas",
    "fp16_reduced_precision": True,
    "fp16_accumulation": False,
    "workspace_config": None,
    "workspace_bytes": 8519680,
    "sms": 80,
}


def supports(*, rows, heads, kv_heads, head_dim, context, first_end, identity,
             capability, prompt=False):
    """Use host-owned positions and immutable device/runtime metadata only."""
    dimensions = (rows, heads, kv_heads, head_dim, context, first_end)
    if any(type(value) is not int for value in dimensions):
        return False
    if not isinstance(identity, Mapping) or not isinstance(capability, (tuple, list)):
        return False
    return (not prompt and tuple(capability) == (7, 0)
            and 1 <= rows <= 24 and heads == 3 and kv_heads == 1 and head_dim == 256
            and 65 <= first_end <= context <= 8192
            and first_end + rows - 1 <= context
            and all(key in identity and identity[key] == value for key, value in REQUIRED_IDENTITY.items()))


@lru_cache(maxsize=None)
def runtime_identity(device):
    """Warm outside capture; all repeated decisions use metadata only."""
    import importlib.metadata
    import os
    import torch

    return {"torch": torch.__version__, "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda, "cublas": importlib.metadata.version("nvidia-cublas-cu12"),
            "blas_backend": str(torch.backends.cuda.preferred_blas_library()),
            "fp16_reduced_precision": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation,
            "workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "workspace_bytes": torch._C._cuda_getCublasWorkspaceSize(),
            "sms": torch.cuda.get_device_properties(device).multi_processor_count}


def active():
    from . import family_ops

    return family_ops._NATIVE_FAMILY and family_ops._NATIVE_ATTENTION


def metadata_prepare(w):
    """Record the immutable runtime descriptor before graph staging/warm-up."""
    identity = dict(runtime_identity(w.device))
    descriptor = {"first_end": 65, "last_end": 8192, "max_rows": 24,
                  "heads": 3, "kv_heads": 1, "head_dim": 256, "runtime": identity}
    previous = w.meta.setdefault("attention_qualification", descriptor)
    if previous != descriptor:
        raise ValueError("attention qualification identity changed after preparation")
    return descriptor


def allows(w, positions, rows, context):
    """One existing owner/graph cohort's real host positions, before staging."""
    if not active() or getattr(w.cfg, "execution_dtype", "bfloat16") != "float16":
        return True
    import torch

    identity = runtime_identity(w.device)
    capability = torch.cuda.get_device_capability(w.device)
    return all(supports(rows=rows, heads=w.cfg.heads, kv_heads=w.cfg.kv_heads,
                        head_dim=w.cfg.head_dim, context=context, first_end=position+1,
                        identity=identity, capability=capability) for position in positions)
