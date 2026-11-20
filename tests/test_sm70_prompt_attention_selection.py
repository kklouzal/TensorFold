"""CPU checks of the prefill prompt-attention selection and its native record."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.owner_execution import KERNEL_FAMILY, RN_KERNEL_FAMILY  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402
from tests.test_native_owner_adapter import _weights  # noqa: E402

def _q8(bits=8, codec=0):
    half = NS(bits=bits, codec=codec)
    return NS(key=half, value=half)


@pytest.mark.parametrize("prefill,family,pair,expected", [
    (True, RN_KERNEL_FAMILY, _q8(), True),
    (False, RN_KERNEL_FAMILY, _q8(), False),
    (True, KERNEL_FAMILY, _q8(), False),
    (True, RN_KERNEL_FAMILY, _q8(4, 0), False),
    (True, RN_KERNEL_FAMILY, None, False),
])
def test_selection_requires_prefill_rn_and_an_owned_codec(prefill, family, pair, expected):
    codec = OwnerBuffers.native_codec.fget(NS(native_pair=pair))
    owner = NS(prefill=prefill, native_kernel_family=family, native_codec=codec)
    assert OwnerBuffers._prompt_attention_selected(owner) is expected


def test_native_record_reports_the_selected_local_family_and_its_borrow():
    weights = _weights()
    owner = OwnerBuffers(weights, 8, NS(idw=2052), prefill=True)
    try:
        owner.prepare_native([State(weights, 257, 4, "int8", slot=0)])
        assert (owner.native_pair.key.bits, owner.native_pair.key.codec) == (8, 0)
        owner.native_kernel_family, owner.native_heads_per_block = RN_KERNEL_FAMILY, 12
        record = owner.native_record()
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
