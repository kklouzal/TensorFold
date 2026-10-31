"""Concurrent device admission must not charge NUMA archives as VRAM."""

from tensorfold.families.qwen4_exp.cuda.multi import _tensors
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_flashnext_indexer_state import runtime


def test_concurrent_accounting_enumerates_each_archive_device_buffer_once():
    # CPU tensors substitute for device allocations only in this ownership
    # oracle. The explicit device_tensors contract remains the authority.
    with runtime() as (weights, _, _):
        state = State(weights, 37, 4, "int8", limit=128)
        owner = state.indexer_history_owner
        tensors = list(_tensors(state))
        identities = [id(tensor) for tensor in tensors]
        for tensor in owner.device_tensors():
            assert identities.count(id(tensor)) == 1
        for tensor in (owner.archive, owner.restore_host, owner.origins_host,
                       *(frame.host for frame in owner.frames.values())):
            assert id(tensor) not in identities
