"""Generated kernel authority, packaging declaration and producer error ownership."""

import fnmatch
import importlib.util
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("bank_exchange_generator", ROOT / "tools/generate_bank_exchange_kernel.py")
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_current_header_reproduces_exactly_from_h_authority():
    assert generator.generate(generator.source_bytes(generator.SOURCE)) == generator.source_bytes(generator.OUTPUT)


def test_unexpected_source_boundary_cannot_produce_a_kernel():
    raw = generator.source_bytes(generator.SOURCE)
    with pytest.raises(ValueError, match="markers"):
        generator.generate(raw.replace(b"template <unsigned Tiles>", b"// changed boundary"))


def test_short_write_and_close_failure_retain_exact_primary():
    primary, secondary = OSError("write primary"), OSError("close secondary")

    class Stream:
        def write(self, _data):
            raise primary

        def close(self):
            raise secondary

    class Output:
        def open(self, mode):
            assert mode == "wb"
            return Stream()

    with pytest.raises(OSError) as outcome:
        generator.write_output(Output(), b"generated")
    assert outcome.value is primary
    # The adopted transport uses the sole cleanup error directly as cause;
    # grouping is reserved for multiple retained roots.
    assert primary.__cause__ is secondary


def test_wheel_package_data_declares_every_required_native_header():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = metadata["tool"]["setuptools"]["package-data"]["tensorfold.cuda.nvfp4"]
    for name in ("bank_exchange_core.h", "bank_exchange_kernel.h", "bank_exchange.cpp", "bank_exchange_pybind.cpp"):
        assert any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
