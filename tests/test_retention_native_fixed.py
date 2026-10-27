"""Known native payload owners and the actual finite descriptor capability."""

import resource
import sys

import pytest

from tensorfold.families.qwen4_exp.cuda.retention_resources import native_sync_fixed_payload


def test_fixed_rows_cover_actual_sdk_layout_and_each_protocol_owner():
    receipt = native_sync_fixed_payload(world=2, ram=True, slots=3, fd_limit=8)
    # Two ranks: base2+worker2+control2+sender1+receiver1. Four prefix
    # boundaries, two RPC keys, six initial keys, and44 retention keys. Input
    # preparation adds two persistent slots plus five overlapping epoch keys.
    assert receipt["current_client_owner_bound"] == 8
    assert receipt["active_caller_bound"] == 5
    assert receipt["client_key_slots_per_array"] == 9
    assert receipt["prefix_boundary_key_bound"] == 4
    assert receipt["stored_key_bound"] == 56 + 2 + 5
    assert receipt["value_map_key_highwater_bound"] == 56 + 2 + 5
    terms = receipt["terms"]
    # The sealed C++20 SDK measured node72, string32, vector24, Store216,
    # TCPClient8, daemon288 and shared TCPServer control block32.
    assert terms["stored_key_chars_and_nodes"] == (56 + 2 + 5) * (8193 + 72)
    assert terms["client_and_handler_string_objects"] == 32 * 32
    assert terms["client_and_handler_vector_objects"] == 29 * 24
    assert terms["current_tcpstore_and_client_objects"] == 8 * 224
    assert terms["daemon_and_poll_vector_objects"] == 352
    assert receipt["global_native_fixed_payload_bytes"] == sum(terms.values())


def test_server_descriptor_metadata_is_global_and_includes_reallocation():
    small = native_sync_fixed_payload(world=2, ram=True, slots=3, fd_limit=8)
    tp8 = native_sync_fixed_payload(world=8, ram=True, slots=3, fd_limit=8)
    # Five accepted sockets fit beside listener+two pipe FDs. Both socket
    # and seven-entry poll vectors can own old4+new8 elements at resize.
    for receipt in (small, tp8):
        assert receipt["accepted_socket_bound"] == 5
        assert receipt["terms"]["accepted_socket_vector_storage"] == 12 * 8
        assert receipt["terms"]["poll_vector_storage"] == 12 * 8
        assert receipt["terms"]["accepted_fd_bookkeeping_nodes"] == 5 * 40
    for name in ("accepted_fd_bookkeeping_nodes", "accepted_fd_bucket_pointer_storage",
                 "accepted_socket_vector_storage", "poll_vector_storage"):
        assert small["terms"][name] == tp8["terms"][name]


def test_actual_hard_limit_is_paid_even_when_soft_limit_is_smaller(monkeypatch):
    observed = []

    def limit(which):
        observed.append(which)
        return 8, 64

    monkeypatch.setattr(resource, "getrlimit", limit)
    receipt = native_sync_fixed_payload(world=2, ram=True, slots=3)
    assert observed == [resource.RLIMIT_NOFILE]
    assert receipt["nofile_hard_limit"] == 64
    assert receipt["accepted_socket_bound"] == 61


def test_new_descriptor_highwater_keeps_old_poll_array_during_growth():
    before = native_sync_fixed_payload(world=2, ram=True, slots=3, fd_limit=17)
    after = native_sync_fixed_payload(world=2, ram=True, slots=3, fd_limit=18)
    # The17-entry poll vector needs new32 while old16 still exists.
    assert before["terms"]["poll_vector_storage"] == (8 + 16) * 8
    assert after["terms"]["poll_vector_storage"] == (16 + 32) * 8
    assert after["global_native_fixed_payload_bytes"] > before["global_native_fixed_payload_bytes"]


def test_bucket_highwater_is_not_credited_after_ram_profile_retirement():
    ram = native_sync_fixed_payload(world=8, ram=True, slots=4, fd_limit=128)
    retired = native_sync_fixed_payload(world=8, ram=False, slots=4, fd_limit=128)
    assert retired["value_map_key_highwater_bound"] == ram["value_map_key_highwater_bound"]
    assert retired["terms"]["value_and_wait_bucket_pointer_storage"] == ram["terms"][
        "value_and_wait_bucket_pointer_storage"]
    assert retired["stored_key_bound"] < ram["stored_key_bound"]


def test_larger_state_envelope_pays_additional_outstanding_prefix_keys():
    first = native_sync_fixed_payload(world=8, ram=True, slots=4, fd_limit=128)
    more = native_sync_fixed_payload(world=8, ram=True, slots=5, fd_limit=128)
    assert more["prefix_boundary_key_bound"] - first["prefix_boundary_key_bound"] == 7
    assert more["stored_key_bound"] - first["stored_key_bound"] == 7
    assert more["terms"]["stored_key_chars_and_nodes"] - first["terms"][
        "stored_key_chars_and_nodes"] == 7 * (8193 + 72)


def test_sparse_native_prime_table_gap_is_in_the_bucket_bound():
    # Adjacent entries from the sealed runtime14 table. The successor exceeds
    # twice its predecessor by211; a generic P(n)<2n bound is insufficient.
    previous, successor = 2251799813685119, 4503599627370449
    accepted = (previous + 1) // 2
    receipt = native_sync_fixed_payload(world=2, ram=True, slots=3, fd_limit=accepted + 3)
    old_and_new_upper = accepted + successor
    assert receipt["terms"]["accepted_fd_bucket_pointer_storage"] >= 2 * 8 * old_and_new_upper


def test_actual_native_hostname_extent_and_constructor_copy_are_paid_once():
    inline = native_sync_fixed_payload(world=2, ram=True, slots=3, fd_limit=128, host_bytes=15)
    heap = native_sync_fixed_payload(world=2, ram=True, slots=3, fd_limit=128, host_bytes=16)
    # Eight Store.addr_.host owners and two concurrent Python constructor
    # caster strings; each selected non-SSO allocation requests16+oneNUL.
    assert inline["terms"]["configured_host_string_storage"] == 0
    assert heap["terms"]["configured_host_string_storage"] == 10 * 17
    assert heap["native_host_bytes"] == 16
    assert heap["global_native_fixed_payload_bytes"] - inline["global_native_fixed_payload_bytes"] == 170


@pytest.mark.parametrize("change", [
    {"world": 0}, {"world": True}, {"slots": 0}, {"slots": None},
    {"ram": 1}, {"fd_limit": 3}, {"fd_limit": -1}, {"fd_limit": True},
    {"host_bytes": -1}, {"host_bytes": True},
])
def test_fixed_payload_rejects_missing_or_nonfinite_contracts(change):
    arguments = {"world": 2, "ram": True, "slots": 3, "fd_limit": 128}
    arguments.update(change)
    with pytest.raises(ValueError):
        native_sync_fixed_payload(**arguments)


def test_fixed_payload_rejects_unrepresentable_total_before_allocation():
    with pytest.raises(ValueError, match="resource sum"):
        native_sync_fixed_payload(world=2, ram=True, slots=3, fd_limit=sys.maxsize)
