"""CPU checks of the prefill prompt-attention selection, its environment switch and its native record."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.cuda import history_owner  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.owner_execution import KERNEL_FAMILY, RN_KERNEL_FAMILY  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402
from tests.test_native_owner_adapter import _weights  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = "TENSORFOLD_PRIVATE_SM70_PROMPT_ATTENTION"


def _import_with(value):
    env = {**os.environ, VARIABLE: value, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    return subprocess.run([sys.executable, "-c", "import tensorfold.families.qwen4_exp.cuda.history_owner"],
                          env=env, capture_output=True, text=True, cwd=ROOT, timeout=300)


@pytest.mark.parametrize("value", ["0", "fused-v1"])
def test_declared_values_import(value):
    result = _import_with(value)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", ["1", "", "fused", "FUSED-V1", "off"])
def test_undeclared_values_fail_at_import(value):
    result = _import_with(value)
    assert result.returncode != 0 and VARIABLE in result.stderr


def _q8(bits=8, codec=0):
    return NS(key=NS(bits=bits, codec=codec))


@pytest.mark.parametrize("prefill,family,pair,switch,expected", [
    (True, RN_KERNEL_FAMILY, _q8(), "fused-v1", True),
    (True, RN_KERNEL_FAMILY, _q8(), "0", False),
    (False, RN_KERNEL_FAMILY, _q8(), "fused-v1", False),
    (True, KERNEL_FAMILY, _q8(), "fused-v1", False),
    (True, RN_KERNEL_FAMILY, _q8(4, 0), "fused-v1", False),
    (True, RN_KERNEL_FAMILY, None, "fused-v1", False),
])
def test_selection_requires_prefill_rn_q8_and_the_switch(monkeypatch, prefill, family, pair, switch, expected):
    monkeypatch.setattr(history_owner, "_PROMPT_ATTENTION", switch)
    owner = NS(prefill=prefill, native_kernel_family=family, native_pair=pair)
    assert OwnerBuffers._prompt_attention_selected(owner) is expected


@pytest.mark.parametrize("switch", ["fused-v1", "0"])
def test_native_record_reports_the_selected_local_family_and_its_borrow(monkeypatch, switch):
    monkeypatch.setattr(history_owner, "_PROMPT_ATTENTION", switch)
    weights = _weights()
    owner = OwnerBuffers(weights, 8, NS(idw=2052), prefill=True)
    try:
        owner.prepare_native([State(weights, 257, 4, "int8", slot=0)])
        assert (owner.native_pair.key.bits, owner.native_pair.key.codec) == (8, 0)
        owner.native_kernel_family, owner.native_heads_per_block = RN_KERNEL_FAMILY, 12
        record = owner.native_record()
        if switch == "0":
            assert record["kernel_family"] == RN_KERNEL_FAMILY and "rn_probability_borrow_bytes" in record
            assert not any(name.startswith("prompt_") for name in record)
            return
        from tensorfold.cuda.sm70.prompt_attention import FAMILY

        assert record["kernel_family"] == FAMILY and record["merge_family"] == RN_KERNEL_FAMILY
        assert not any(name.startswith("rn_") for name in record)
        offset, size = record["prompt_unnormalized_borrow_offset_bytes"], record["prompt_unnormalized_borrow_bytes"]
        # Disjoint from the gathered statistics prefix and inside the paid receive slab.
        assert offset == 8 * owner.rows * 12 * 2 * 4 and size == owner.rows * 12 * 256 * 4
        assert offset + size <= owner.partials.numel() * owner.partials.element_size()
        assert record["prompt_unnormalized_additional_device_bytes"] == 0
    finally:
        owner.close()
