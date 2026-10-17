"""The NVFP4 CPU controls skip before importing unavailable backend SDKs."""

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_loading_controls_skip_without_triton_before_backend_import_or_cuda_initialization():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("the isolated collection control requires an installed Torch")
    target = Path(__file__).with_name("test_nvfp4_loading_pack_control.py")
    script = textwrap.dedent("""
        import json
        from pathlib import Path
        import runpy
        import sys
        import pytest
        import torch

        assert not torch.cuda.is_initialized()
        initialized = []
        def forbidden(*args, **kwargs):
            initialized.append(True)
            raise AssertionError("collection attempted CUDA initialization")
        torch.cuda.init = forbidden
        torch.cuda._lazy_init = forbidden
        if hasattr(torch._C, "_cuda_init"):
            torch._C._cuda_init = forbidden
        sys.modules["triton"] = None
        sys.path.insert(0, str(Path(sys.argv[1]).parent))
        try:
            runpy.run_path(sys.argv[1], run_name="_isolated_nvfp4_collection")
        except pytest.skip.Exception as skipped:
            assert "triton" in str(skipped)
        else:
            raise AssertionError("missing Triton did not skip the module")
        assert not initialized and not torch.cuda.is_initialized()
        assert "nvfp4_a4_inputs" not in sys.modules
        assert "test_nvfp4_a4_host_control" not in sys.modules
        assert "tensorfold.cuda.nvfp4.host_experts" not in sys.modules
        print(json.dumps({"skipped_for": "triton", "cuda_initialized": False,
                          "backend_imported": False}))
    """)
    result = subprocess.run([sys.executable, "-c", script, str(target)], check=False,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"skipped_for": "triton", "cuda_initialized": False,
                                        "backend_imported": False}
