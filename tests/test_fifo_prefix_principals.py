"""Selected FIFO prefix authorization; numerical/state copy is the existing controlled boundary."""

import pytest

from tensorfold.families.qwen4_exp.cuda import prefixes
from test_flashnext_adaptive_retention import owner


def test_fifo_scope_does_not_borrow_another_principal_and_identical_ids_do_not_replace_it():
    decoder = owner(2)
    decoder.admission_policy = "fifo"
    source = decoder.free.pop()
    source.data[:] = bytes([31]) * len(source.data)
    source.cache_principal = "trusted-a"
    decoder._remember([7] * 31, source, {"pos": 31, "mtp_len": 30}, None)
    destination, resume, cached = decoder._slot_for([7] * 32, True, principal="trusted-b")
    assert destination is not source and resume is None and cached == 0
    assert bytes(source.data) == bytes([31]) * len(source.data)
    destination.cache_principal = "trusted-b"
    decoder._remember([7] * 31, destination, {"pos": 31, "mtp_len": 30}, None)
    assert len(decoder.kept) == 2
    assert prefixes._best(decoder.kept, [7] * 32, principal="trusted-a")[1] is source
    assert prefixes._best(decoder.kept, [7] * 32, principal="trusted-b")[1] is destination
    assert prefixes._best(decoder.kept, [7] * 32, principal=None) is None


def test_explicit_no_auth_domain_does_not_reinterpret_untagged_legacy_entries():
    legacy = ([1] * 8, object(), {"pos": 8}, None)
    local = ([1] * 4, object(), {"pos": 4, "prefix_principal": None}, None)
    other = ([1] * 7, object(), {"pos": 7, "prefix_principal": "trusted-a"}, None)
    assert prefixes._best([legacy, local, other], [1] * 9) is legacy  # Unchanged legacy policy.
    assert prefixes._best([legacy, local, other], [1] * 9, principal=None) is local
    assert prefixes._best([legacy, local, other], [1] * 9, principal="trusted-a") is other


def test_scoped_publication_requires_the_same_actual_snapshot_principal():
    decoder = owner(1)
    state = decoder.free.pop()
    with pytest.raises(ValueError, match="actual request principal"):
        prefixes.remember(decoder, [1], state, {"pos": 1, "prefix_principal": "trusted-b"}, None,
                          principal="trusted-a")
    assert decoder.kept == []
