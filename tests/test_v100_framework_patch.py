"""Actual installed-source-wheel CMake consumer gates; no GPU qualification."""

import importlib.metadata
from pathlib import Path
import shutil
import subprocess

import pytest

try:
    TORCH_VERSION = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError:
    TORCH_VERSION = None

pytestmark = pytest.mark.skipif(TORCH_VERSION != "2.14.1+tfcu129sm70",
                                reason="requires the actual selected source-built framework")


def consume(prefix, temporary):
    cmake = shutil.which("cmake")
    assert cmake is not None, "the source foundation must supply its pinned CMake consumer tool"
    script = temporary / "consume.cmake"
    script.write_text('''find_package(ATen CONFIG REQUIRED PATHS "${ATEN_PREFIX}/share/cmake/ATen" NO_DEFAULT_PATH)
get_filename_component(expected "${ATEN_PREFIX}/include" ABSOLUTE)
if(NOT "${ATEN_INCLUDE_DIR}" STREQUAL "${expected}")
  message(FATAL_ERROR "ATen includes escaped the installed/relocated prefix: ${ATEN_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ATEN_INCLUDE_DIR}/ATen/ATen.h")
  message(FATAL_ERROR "ATen consumer resolved nonexistent installed headers")
endif()
''')
    result = subprocess.run([cmake, "-DATEN_PREFIX=" + str(prefix), "-P", str(script)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def installed_prefix():
    return Path(importlib.metadata.distribution("torch").locate_file("torch")).resolve(strict=True)


def test_installed_aten_config_resolves_actual_headers(tmp_path):
    consume(installed_prefix(), tmp_path)


def test_installed_aten_config_survives_prefix_relocation(tmp_path):
    source = installed_prefix()
    relocated = tmp_path / "relocated prefix"
    (relocated / "share/cmake/ATen").mkdir(parents=True)
    shutil.copyfile(source / "share/cmake/ATen/ATenConfig.cmake",
                    relocated / "share/cmake/ATen/ATenConfig.cmake")
    (relocated / "include").symlink_to(source / "include", target_is_directory=True)
    consume(relocated, tmp_path)
