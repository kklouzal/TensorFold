# ruff: noqa: E402 - optional Torch precedes the actual communicator module.
"""Persistent native metadata reservations without GPU/model execution."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
pytest.importorskip("torch")

from tensorfold.cuda import comm
from test_nccl_buffer_boundary import source


def test_unopened_or_closed_communicator_cannot_reserve_native_state_slots():
    owner = comm.NCCL()
    assert owner.native_state_slots_highwater == 0
    with pytest.raises(RuntimeError, match="open communicator"):
        owner.note_native_state_slots(1)
    owner.close()
    with pytest.raises(RuntimeError, match="open communicator"):
        owner.note_native_state_slots(1)
    assert owner.native_state_slots_highwater == 0


@pytest.mark.parametrize("count", [False, 0, -1, 1.5, "8", 1 << 63])
def test_invalid_native_state_reservation_never_changes_highwater(count):
    owner = comm.NCCL()
    with pytest.raises(ValueError, match="signed-63"):
        owner.note_native_state_slots(count)
    assert owner.native_state_slots_highwater == 0


def test_same_transport_keeps_maximum_across_model_drains_and_wrapper_borrows():
    cls, lib, _streams, distributed = source()
    with patch.dict(sys.modules, {distributed.__name__: distributed}):
        owner = cls()
        owner.open(1, 2, "fixture", 1234)
    wrapper = comm.Transport(owner)
    try:
        wrapper.bind_native_nofile(1024)
        wrapper.bind_native_nofile(1024)
        with pytest.raises(ValueError, match="fixed for this communicator"):
            wrapper.bind_native_nofile(512)
        assert owner.native_nofile_hard_limit == 1024
        wrapper.note_native_state_slots(40)
        wrapper.note_native_state_slots(3)
        assert wrapper.native_state_slots_highwater == owner.native_state_slots_highwater == 40
        wrapper.note_native_state_slots(64)
        assert owner.native_state_slots_highwater == 64
        assert len(lib.ncclCommInitRank.calls) == 1  # No new communicator for a new model count.
    finally:
        owner.close()
    assert owner.native_state_slots_highwater == 64
    with pytest.raises(RuntimeError, match="open communicator"):
        wrapper.note_native_state_slots(65)


def test_unqualified_extension_refuses_before_any_adaptive_native_acquisition(monkeypatch):
    monkeypatch.setitem(comm.BACKENDS, "unqualified-test", lambda owner: owner)
    monkeypatch.setattr(comm, "NCCL", lambda: pytest.fail("unsupported provider acquired native owner"))
    with pytest.raises(ValueError, match="plain NCCL"):
        comm.open_comm(0, 8, "fixture", 1234, backend="unqualified-test", use_libuv=False)


def test_generic_default_keeps_the_existing_registered_transport_contract(monkeypatch):
    events = []
    owner = SimpleNamespace(store=None, open=lambda *_a, **kw: events.append(("open", kw)))
    monkeypatch.setattr(comm, "NCCL", lambda: owner)
    monkeypatch.setitem(comm.BACKENDS, "generic-test", lambda value:
                        events.append(("wrapped", value is owner)) or value)
    assert comm.open_comm(0, 1, "fixture", 1234, backend="generic-test") is owner
    assert events == [("open", {}), ("wrapped", True)]
