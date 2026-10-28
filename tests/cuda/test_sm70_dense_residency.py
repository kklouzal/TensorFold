"""Load existing Dense kernel code before any tensor work under CUDA lazy loading."""
from pathlib import Path
import hashlib
import os

import pytest

torch = pytest.importorskip("torch")


def prepare_before_tensor_work():
    from tensorfold.cuda.sm70 import gguf_ops

    if os.environ.get("CUDA_MODULE_LOADING") != "LAZY":
        raise RuntimeError("Dense residency qualification requires explicit CUDA_MODULE_LOADING=LAZY")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("Dense residency qualification requires actual SM70")
    # Construct no model or staging tensors before the driver's lazy-code query.
    module = gguf_ops.extension()
    free_before, total = torch.cuda.mem_get_info()
    loaded = module.prepare_dense32()
    free_after, total_after = torch.cuda.mem_get_info()
    assert loaded == 4 and total_after == total
    path = Path(module.__file__)
    return {"loaded_kernel_variants": loaded, "loading": os.environ["CUDA_MODULE_LOADING"],
            "free_before": free_before, "free_after": free_after, "total": total,
            "module": str(path), "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "explicit lazy function-attribute residency before tensor warmup/capture; no math/performance claim"}


def test_actual_sm70_lazy_dense_function_residency_before_warmup():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires actual SM70")
    assert prepare_before_tensor_work()["loaded_kernel_variants"] == 4
