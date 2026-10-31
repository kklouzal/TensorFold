"""CPU boundary-fault controls for probe teardown; these do not qualify CUDA execution."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def retirement_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    include = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"
    if compiler is None or not (include / "cuda_runtime.h").is_file():
        pytest.skip("the selected native source builder supplies the host compiler and CUDA headers")
    binary = tmp_path_factory.mktemp("probe-retirement") / "retirement"
    result = subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(include),
                             "-I" + str(ROOT / "deploy/v100/probes"),
                             str(ROOT / "tests/v100_probe_retirement_fixture.cpp"), "-o", str(binary)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("mode,code,retired", [("success", 0, True), ("fence-failure", 1, False),
                                               ("graph-failure", 1, False)])
def test_uncertain_retirement_keeps_host_and_native_borrowers_until_process_exit(retirement_probe, tmp_path,
                                                                              mode, code, retired):
    marker = tmp_path / "host-destruction"
    result = subprocess.run([str(retirement_probe), mode, str(marker)], capture_output=True, text=True, timeout=5)
    assert result.returncode == code, result.stderr
    assert marker.exists() is retired
    if not retired:
        assert "terminal process teardown" in result.stderr
