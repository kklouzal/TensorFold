"""Selected default C JSON encoder payload, separate from parser/streaming."""

from dataclasses import replace
import json
import sys

import pytest

pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.retention_resources import (
    _json_one_shot_payload, _json_payload_policy,
)


@pytest.fixture(scope="module")
def policy():
    return _json_payload_policy()


def test_actual_encoder_source_is_bound_and_old_authority_cannot_grant_this_scope(policy):
    assert ("json-encoder", 16070,
            "af7bd40a0d0d0a3e726a9b4b3a2a543019f6ab97a340d0162a9c29ca9da97869") in policy.files
    old = replace(policy, files=tuple(row for row in policy.files if row[0] != "json-encoder"))
    with pytest.raises(ValueError, match="captured encoder"):
        _json_one_shot_payload(1024, old)


@pytest.mark.parametrize("value", [None, [127, "\x7f" * 100], {"é": [1e15, True, ["𝄞"]]},
                                  [{str(i): [i, None]} for i in range(64)]])
def test_default_wire_and_ascii_owners_have_an_independent_payload_floor(policy, value):
    text = json.dumps(value)
    wire = text.encode("utf-8")
    record = _json_one_shot_payload(len(wire), policy)
    assert len(text) == len(wire) and text.isascii()
    assert record["phase_bytes"]["returned_ascii_to_utf8"] >= sys.getsizeof(text) + sys.getsizeof(wire)
    assert record["total_bytes"] == max(record["phase_bytes"].values())


@pytest.mark.parametrize("count", [1, 5, 16, 64, 10000])
def test_int_address_markers_pay_general_tables_and_integer_objects(policy, count):
    markers = {(1 << 63) + i: None for i in range(count)}
    record = _json_one_shot_payload(2 * count, policy)
    assert record["c_marker_depth_bound"] == count
    assert record["metadata"]["marker_dictionary_and_resize"] >= sys.getsizeof(markers)
    assert record["metadata"]["marker_address_integers"] >= sum(map(sys.getsizeof, markers))


def test_c_depth_is_independent_of_python_generator_recursion_limit(policy):
    shallow_python = replace(policy, recursion_limit=20, frame_work=20 * 764 + 36)
    assert _json_one_shot_payload(1 << 20, shallow_python)["c_marker_depth_bound"] == 10000
    assert _json_one_shot_payload(1 << 20, shallow_python) == _json_one_shot_payload(1 << 20, policy)


@pytest.mark.parametrize("size", [0, -1, True, 1 << 63])
def test_one_shot_bound_rejects_invalid_sizes_before_any_encoding(policy, size):
    with pytest.raises(ValueError):
        _json_one_shot_payload(size, policy)


def test_full_phase_arithmetic_must_remain_representable(policy):
    with pytest.raises(ValueError):
        _json_one_shot_payload(sys.maxsize, policy)
