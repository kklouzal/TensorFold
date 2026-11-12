"""Packaging recapture cannot change any completed native build fact."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("capture_bridge", ROOT / "tools/bind_v100_capture_provenance.py")
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def captures(tmp_path):
    original = {"schema": "tensorfold.v100-source-build", "version": 1,
        "producer_inputs": {"recipe": {"sha256": "a" * 64}}, "wheel": {"sha256": "b" * 64},
        "targets": {"cuda": "12.9.2"}, "native_prefix": {"library": {"sha256": "c" * 64}},
        "probes": {"oracle": {"sha256": "d" * 64}}, "cmake_configuration": {"arch": "70"}}
    current = copy.deepcopy(original)
    current["producer_inputs"]["recipe"]["sha256"] = "e" * 64
    old = tmp_path / "historical.json"
    old.write_text(json.dumps(original))
    inputs = tmp_path / "build-inputs"
    inputs.mkdir()
    new = inputs / "source-build.json"
    new.write_text(json.dumps(current))
    phase = inputs / "original-native-phase-v6"
    phase.mkdir()
    (phase / "input-manifest.json").write_bytes(b"actual frozen compile snapshot identity")
    return old, new, phase, inputs / "capture-provenance.json", current


def test_capture_time_revision_binds_both_receipts_and_original_phase_bytes(tmp_path):
    old, new, phase, output, current = captures(tmp_path)
    old_bytes, new_bytes = old.read_bytes(), new.read_bytes()
    result = BRIDGE.bind(old, new, phase, output)
    assert old.read_bytes() == old_bytes and new.read_bytes() == new_bytes
    assert (output.parent / result["original_capture"]["path"]).read_bytes() == old_bytes
    assert result["original_capture"]["sha256"] == hashlib.sha256(old_bytes).hexdigest()
    assert result["current_capture"]["sha256"] == hashlib.sha256(new_bytes).hexdigest()
    assert "no native rebuild" in result["scope"]
    assert result["native_phase"]["files"]["input-manifest.json"]["sha256"] == hashlib.sha256(
        (phase / "input-manifest.json").read_bytes()).hexdigest()
    assert BRIDGE.bind(old, new, phase, output) == result


@pytest.mark.parametrize("field", ["wheel", "targets", "native_prefix", "probes", "cmake_configuration", "added-null"])
def test_capture_time_revision_refuses_any_native_field_change(field, tmp_path):
    old, new, phase, output, current = captures(tmp_path)
    current[field] = None if field == "added-null" else {"different": True}
    new.write_text(json.dumps(current))
    with pytest.raises(ValueError, match="native build identities changed"):
        BRIDGE.bind(old, new, phase, output)
    assert not output.exists()
