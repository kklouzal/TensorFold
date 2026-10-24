"""The portable component receipt reports evidence without a serving claim."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from gguf_fixture import tiny_model
from tools import v100_probe


def test_native_receipt_hashes_actual_binary_and_requires_sm70_cubin(tmp_path, monkeypatch):
    binary = tmp_path / "native.so"
    binary.write_bytes(b"the native module under qualification")
    module = SimpleNamespace(__file__=str(binary), __name__="tf_native", decode=lambda: None)
    monkeypatch.setattr(v100_probe.subprocess, "check_output", lambda *a, **k: "ELF file: kernels.sm_70.cubin")
    result = v100_probe.native_receipt(module)
    assert result["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert result["module"] == "tf_native" and result["operators"] == ["decode"]
    monkeypatch.setattr(v100_probe.subprocess, "check_output", lambda *a, **k: "ELF file: kernels.sm_80.cubin")
    with pytest.raises(RuntimeError, match="SM70 cubin"):
        v100_probe.native_receipt(module)


def test_checkpoint_only_receipt_keeps_module_availability_and_qualification_distinct(tmp_path, monkeypatch, capsys):
    path = tiny_model(tmp_path / "tiny.gguf")
    monkeypatch.setattr(v100_probe, "find_spec", lambda name: None if name.endswith("gdn_native") else object())
    monkeypatch.setattr("sys.argv", ["v100_probe.py", "--checkpoint", str(path)])
    v100_probe.main()
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == 2 and result["qualification"] == "component-receipt"
    assert result["components"]["mixed_gguf_weights"]["available"]
    assert not result["components"]["native_gdn"]["available"]
    assert result["checkpoint"]["architecture"] == "qwen4exp"
    assert "provider" not in result and "engine_ready" not in result and "pending" not in result
