"""Native format selection and original-payload admission at family boundaries."""

import json
import math
import sys
from types import ModuleType

import pytest

from gemma_cuda_target_nvfp4_metadata_test import fixture
from tensorfold import families
from tensorfold.cuda import capacity as common
from tensorfold.families.gemma4.cuda import capacity


def test_native_checkpoint_is_readable_only_by_its_cuda_backend(tmp_path):
    raw, _ = fixture()
    (tmp_path / "config.json").write_text(json.dumps(raw))
    family = families.detect(tmp_path)
    families.require_readable(family, raw, "cuda")
    family.package.check(tmp_path)
    with pytest.raises(ValueError, match="does not read"):
        families.require_readable(family, raw, "mlx")


def test_admission_retains_one_bf16_tied_head_and_original_native_payloads(tmp_path, monkeypatch):
    raw, headers = fixture()
    metadata = ModuleType("tensorfold.checkpoint_metadata")
    metadata.read_config = lambda directory: raw
    runtime = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "tensorfold.checkpoint_metadata", metadata)
    monkeypatch.setitem(sys.modules, "torch", runtime)
    monkeypatch.setattr(common, "headers", lambda directory: headers)
    captured = {}

    def admitted(directory, context, explicit, torch, geometry, rule, *, weight_estimator):
        captured["weights"], captured["host_staging"] = weight_estimator(directory)
        return {"context_window": context}

    monkeypatch.setattr(common, "admit", admitted)
    assert capacity.admit(tmp_path, 128, True) == {"context_window": 128}
    # All original fields are consumed, with no duplicated tied output head,
    # float expansion of the native experts, or affine scales/biases invented.
    original = sum(math.prod(v["shape"]) * common.SIZES[v["dtype"]]
                   for v in headers.values())
    assert captured["weights"].resident == original
    assert captured["weights"].staging == captured["host_staging"] > 0
