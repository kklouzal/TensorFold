"""Fixed DFlash compute blocks and useful target rows are different resources."""

import pytest

from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry


@pytest.mark.parametrize("requested", [0, 1, 2, 3])
def test_fixed_four_compute_keeps_selected_proposal_count_independent(requested):
    method = MethodGeometry("dflash2", 3)
    assert method.cap_proposals(requested, reply_tokens=10, context_tokens=10, target_rows=32) == requested
    assert method.target_rows(requested) == requested + 1
    receipt = method.receipt()
    assert receipt["draft_compute_block_rows"] == 4
    assert receipt["stock_mtp_cache_layers"] == 0
    assert method.draft_tail_rows is None
    assert receipt["zero_proposal_maintains_conditioning"]
    assert receipt["sampling"] == "standard rejection with actual conditional selector q"


@pytest.mark.parametrize("name,arguments", [
    ("reply", {"reply_tokens": 1, "context_tokens": 10, "target_rows": 32}),
    ("context", {"reply_tokens": 10, "context_tokens": 1, "target_rows": 32}),
    ("execution", {"reply_tokens": 10, "context_tokens": 10, "target_rows": 1}),
])
def test_commitment_and_execution_envelope_cap_before_proposal_draw(name, arguments):
    assert MethodGeometry("dflash2", 3).cap_proposals(3, **arguments) == 0, name


def test_no_claim_that_larger_proposal_count_changes_the_fixed_operator():
    with pytest.raises(ValueError, match="at most three"):
        MethodGeometry("dflash2", 4)
