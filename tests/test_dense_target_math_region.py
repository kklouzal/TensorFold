"""Target-region fixture rejects mismatches without mutating either math flag."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def fixture_module():
    path = Path(__file__).parent / "cuda/test_sm70_large_dense_tiles.py"
    spec = importlib.util.spec_from_file_location("dense_target_region_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reduction,accumulation,valid", [(True,False,True),(False,False,False),(True,True,False)])
def test_exact_target_flags_are_observed_or_refused_without_mutation(monkeypatch,reduction,accumulation,valid):
    module = fixture_module()
    matmul = NS(allow_fp16_reduced_precision_reduction=reduction, allow_fp16_accumulation=accumulation)
    monkeypatch.setattr(module,"torch",NS(backends=NS(cuda=NS(matmul=matmul))))
    if valid:
        assert module.target_math_policy() == {"allow_fp16_reduced_precision_reduction":True,"allow_fp16_accumulation":False}
    else:
        with pytest.raises(RuntimeError,match="target FP16"):
            module.target_math_policy()
    assert (matmul.allow_fp16_reduced_precision_reduction,matmul.allow_fp16_accumulation) == (reduction,accumulation)
