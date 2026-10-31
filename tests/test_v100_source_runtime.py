"""Actual installed source-image application binding; no GPU work or qualification."""

import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(not Path("/opt/tensorfold-pins/v100-source-foundation.json").is_file(),
                                reason="requires the selected source-image receipt")


def test_installed_source_application_uses_the_owned_nccl_provider():
    from tensorfold.cuda.comm import _library

    expected = Path("/opt/tensorfold-native/nccl-2.32.3/lib/libnccl.so.2")
    assert os.environ.get("TF_NCCL_LIB") == str(expected)
    library = _library()
    assert Path(library._name).resolve(strict=True) == expected.resolve(strict=True)
    version = ctypes.c_int()
    assert library.ncclGetVersion(ctypes.byref(version)) == 0 and version.value == 23203


def test_current_process_audit_binds_an_actual_additional_shared_module(tmp_path):
    compiler = shutil.which("cc")
    assert compiler is not None, "the selected development foundation supplies its host compiler"
    source, binary = tmp_path / "audit.c", tmp_path / "extra_audit_fixture.so"
    source.write_text("int audit_fixture_value(void) { return 37; }\n")
    subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(binary)], check=True,
                   capture_output=True, timeout=30)
    library = ctypes.CDLL(str(binary))
    assert library.audit_fixture_value() == 37
    path = Path("/opt/tensorfold-pins/source_foundation.py")
    spec = importlib.util.spec_from_file_location("selected_source_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.read_json(path.parent / "v100-source-foundation.json")[0]
    pins = module.read_json(path.parent / "source-pins.json")[0]
    result = module.audit_process(manifest, pins)
    assert result["scope"] == "current-process" and result["release_qualified"] is False
    entry = next(row for row in result["additional_process_libraries"] if row["path"] == str(binary.resolve()))
    assert entry == {"path": str(binary.resolve()), **module.file_identity(binary)}
