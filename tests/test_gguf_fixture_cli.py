"""Generator CLI plans keep finite controls separate from the full-range corpus."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/generate_gguf_mixed_fixture.py"
FIXTURES = ROOT / "tests/fixtures"


@pytest.mark.parametrize("finite", [False, True])
def test_real_cli_dry_run_selects_mode_specific_default_and_preserves_both_corpora(tmp_path, finite):
    paths = [FIXTURES / "gguf_mixed_ggml.npz", FIXTURES / "gguf_mixed_projection_ggml.npz"]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    argv = [sys.executable, str(TOOL), "--quants", str(tmp_path / "missing-reference.c"),
            "--common", str(tmp_path / "missing-reference.h"), "--dry-run"]
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
