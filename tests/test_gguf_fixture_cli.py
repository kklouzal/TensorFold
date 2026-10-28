"""Actual generator CLI plans cannot overwrite the full-range corpus by default."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/generate_gguf_mixed_fixture.py"
FIXTURES = ROOT / "tests/fixtures"


@pytest.mark.parametrize("finite", [False, True])
def test_real_cli_dry_run_selects_mode_specific_default_and_preserves_both_corpora(tmp_path, finite):
    paths = [FIXTURES / "gguf_mixed_ggml.npz", FIXTURES / "gguf_mixed_projection_ggml.npz"]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    argv = [sys.executable, str(TOOL), "--quants", str(tmp_path / "read-only-reference.c"),
            "--common", str(tmp_path / "read-only-reference.h"), "--dry-run"]
    if finite:
        argv.append("--finite-projections")
    result = subprocess.run(argv, text=True, capture_output=True, check=True)
    record = json.loads(result.stdout)
    assert record["output"] == str(paths[int(finite)])
    assert record["finite_projections"] is finite
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before


def test_real_cli_explicit_output_remains_an_override_and_dry_run_never_creates_it(tmp_path):
    output = tmp_path / "custom.npz"
    result = subprocess.run([sys.executable, str(TOOL), "--quants", "reference.c", "--common", "reference.h",
                             "--finite-projections", "--output", str(output), "--dry-run"],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["output"] == str(output)
    assert not output.exists()
