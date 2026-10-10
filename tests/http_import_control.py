"""Prove HTTP imports avoid accelerator runtimes in a fresh stdlib-only process.

Other tests may already have imported Torch or NumPy in the parent pytest
process. The child inspects the same TensorFold package location, with SDK
imports denied before loading any project module and site initialization off.
"""

from pathlib import Path
import subprocess
import sys

import tensorfold


_IMPORT_CHECK = r'''
import importlib
import importlib.abc
from pathlib import Path
import sys

forbidden = {"torch", "numpy", "mlx", "triton", "xgrammar", "tokenizers", "ctypes"}

class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden or fullname == "tensorfold._fd_owner":
            raise RuntimeError("accelerator/native import during HTTP import check: " + fullname)

sys.meta_path.insert(0, NoSDK())
package = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(package.parent))
for name in sys.argv[2:]:
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(package):
        raise AssertionError("HTTP import resolved outside the selected package: " + name)
if any(name.split(".")[0] in forbidden or name == "tensorfold._fd_owner" for name in sys.modules):
    raise AssertionError("HTTP imports initialized an accelerator/native runtime")
'''


def assert_accelerator_free_imports(*modules):
    package = Path(tensorfold.__file__).resolve().parent
    try:
        subprocess.run(
            [sys.executable, "-I", "-S", "-W", "error", "-c", _IMPORT_CHECK, str(package), *modules],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except subprocess.CalledProcessError as error:
        raise AssertionError("isolated HTTP import check failed:\n" + error.stdout + error.stderr) from error
