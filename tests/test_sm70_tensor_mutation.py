"""The actual selected Torch counter cannot wrap at an owned writer boundary."""

import pytest
import torch

from tensorfold.cuda.sm70.tensor_mutation import MAX_MUTATION_VERSION, require_mutation_budget


@pytest.mark.parametrize("writes", [False, True, 0, -1, 1.5, MAX_MUTATION_VERSION + 1])
def test_invalid_write_budgets_are_refused(writes):
    with pytest.raises(ValueError, match="positive bounded"):
        require_mutation_budget(torch.zeros(2, 3), writes)


def test_inference_storage_cannot_silently_disable_mutation_checks():
    with torch.inference_mode():
        untracked = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="normal tracked"):
        require_mutation_budget(untracked, 1)


def test_raw_publication_refuses_actual_exhausted_counter_before_submission(monkeypatch):
    from tensorfold.cuda.sm70 import compact_indexer as native

    destination = torch.zeros(4, 3, dtype=torch.int64)
    torch._C._autograd._unsafe_set_version_counter([destination], [MAX_MUTATION_VERSION])
    monkeypatch.setattr(native, "extension", lambda: pytest.fail("raw submission after exhausted counter"))
    with pytest.raises(ValueError, match="version exhausted"):
        native.publish(torch.ones_like(destination), destination, 0, torch.zeros(2, dtype=torch.int64),
                       (0, 0), torch.tensor([0, -2], dtype=torch.int32), 0)
    assert destination._version == MAX_MUTATION_VERSION and torch.count_nonzero(destination) == 0


def test_compact_persistent_destinations_and_oracle_clone_remain_tracked_under_inference_mode():
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tests.test_native_owner_adapter import _weights

    with torch.inference_mode():
        state = State(_weights(), 32, 4, "int8")
        cloned = state.clone()
    for owner in (state, cloned):
        for tensor in [*owner.pooled, owner.mtp_pooled]:
            require_mutation_budget(tensor, 1)
        for entry in owner.compact_indexer.entries:
            if entry is not None:
                for name in ("raw", "coordinates", "protected", "protected_coordinates", "proposal_raw", "proposal_coordinates"):
                    require_mutation_budget(getattr(entry, name), 1)
