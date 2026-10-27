"""Selected native requested-layout bounds, with independent rational arithmetic."""

from fractions import Fraction
from types import SimpleNamespace as NS
import math
import sys

import pytest

from tensorfold.families.qwen4_exp.cuda import tokenizer_payload as payload


def bound(**changes):
    fields = dict(decoder="tokenizers-0.23.3/bytelevel-concat-v2", piece_utf8_bytes=8,
        decoded_utf8_bytes_per_token=3, decoded_utf8_fixed_bytes=0, working_utf8_fixed_bytes=0,
        lexical_id_wire_ratio=(8, 3), decoded_id_wire_ratio=(3, 3), escaped_id_wire_ratio=(5, 3))
    return NS(**{**fields, **changes})


@pytest.fixture
def equations_only(monkeypatch):
    # Arithmetic tests do not impersonate an actual runtime/admission receipt.
    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: {"scope": "unit arithmetic only"})


def test_native_phase_lifetimes_use_maximum_and_separate_python_output(equations_only):
    result = payload.bytelevel_decode_payload(bound(), context=4)
    assert result["phase_heap_requested_bytes"] == {
        "token_lookup_clone_and_header_growth": 432,
        "piece_and_flat_bytes": 292,
        "lossy_and_to_string": 120,
        "decoder_join": 64,
    }
    assert result["native_heap_requested_bytes"] == 432
    assert result["result_to_python_payload_bytes"] == 268
    assert result["json_escaped_string_bytes"] == 74


@pytest.mark.parametrize("wire", [0, 1, 2, 5, 10, 101, 2**53 + 7])
def test_decimal_wire_bounds_use_exact_rationals_without_context_clipping(equations_only, wire):
    selected = bound(lexical_id_wire_ratio=(11, 7), decoded_id_wire_ratio=(5, 7),
                     escaped_id_wire_ratio=(13, 7))
    result = payload.bytelevel_decode_payload(selected, encoded_id_bytes=wire)
    n = (wire + 1) // 2
    assert result["input_ids"] == n
    assert result["lexical_bytes"] == min(8 * n, math.ceil(Fraction((wire + 1) * 11, 7)))
    expected = min(3 * n, math.ceil(Fraction((wire + 1) * 5, 7)))
    assert result["decoded_utf8_bytes"] == expected
    assert result["json_escaped_content_bytes"] == min(6 * expected,
        math.ceil(Fraction((wire + 1) * 13, 7)))
    assert "context_ids" not in result["budget"]


def test_empty_decode_still_pays_returned_string_vector_header(equations_only):
    result = payload.bytelevel_decode_payload(bound(), context=0)
    assert result["native_heap_requested_bytes"] == 24
    assert result["result_to_python_payload_bytes"] == 160
    assert result["json_escaped_string_bytes"] == 2


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65])
def test_native_bound_covers_independent_constructible_vector_growth(equations_only, count):
    # Simulate successive pushes: actual old+new buffers coexist during growth.
    # Zero lexical payload isolates the String-vector header responsibility.
    capacity = peak = 0
    for required in range(1, count + 1):
        if required > capacity:
            next_capacity = max(4, 2 * capacity, required)
            peak = max(peak, capacity + next_capacity)
            capacity = next_capacity
    result = payload.bytelevel_decode_payload(bound(piece_utf8_bytes=0,
        decoded_utf8_bytes_per_token=0), context=count)
    assert result["phase_heap_requested_bytes"]["token_lookup_clone_and_header_growth"] >= 4 * count + 24 * peak


@pytest.mark.parametrize("kwargs", [{}, {"context": 1, "encoded_id_bytes": 1}, {"context": True},
                                  {"context": -1}, {"encoded_id_bytes": 1.0}, {"context": sys.maxsize}])
def test_invalid_or_overflowing_budgets_fail_before_runtime_selection(monkeypatch, kwargs):
    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: pytest.fail("invalid bound selected runtime"))
    with pytest.raises(ValueError):
        payload.bytelevel_decode_payload(bound(), **kwargs)


@pytest.mark.parametrize("ratio", [None, [1, 2], (1, 0), (-1, 2), (True, 2), (1, True), (sys.maxsize + 1, 2)])
def test_invalid_wire_ratios_are_not_rounded_or_defaulted(equations_only, ratio):
    with pytest.raises(ValueError, match="exact nonnegative rational"):
        payload.bytelevel_decode_payload(bound(decoded_id_wire_ratio=ratio), encoded_id_bytes=100)


@pytest.mark.parametrize("changes", [{"decoder": "Sequence"}, {"decoded_utf8_fixed_bytes": 1},
                                    {"working_utf8_fixed_bytes": 1}, {"piece_utf8_bytes": False}])
def test_unsupported_decoder_domains_are_not_given_bytelevel_credit(equations_only, changes):
    with pytest.raises(ValueError):
        payload.bytelevel_decode_payload(bound(**changes), context=4)


def test_selected_native_artifact_and_toolchain_are_bound():
    pytest.importorskip("tokenizers")
    actual = payload.bytelevel_payload_policy()
    assert actual["rust"] == "1.99.0/b940084d7" and actual["pyo3"] == "0.29.2"
    assert {name: (size, digest) for name, size, digest in actual["files"]} == {
        "tokenizers-native": (11326992, "4b1458a379296762a034d3bc89299c0c35b4eefb0577acca922f481b262c65a1"),
        "tokenizers-sbom": (151305, "6faa23324b2b53b4a8da9b451a675610cc45e1a23e8c5bb0740c080064e1ba0d"),
    }


def test_changed_native_artifact_is_refused(monkeypatch):
    pytest.importorskip("tokenizers")
    from tensorfold.engine import model_identity
    original = model_identity.capture_model_identity
    def changed(*args, **kwargs):
        captured = original(*args, **kwargs)
        return NS(files=[NS(logical_name=item.logical_name, size=item.size, sha256="0" * 64)
                         for item in captured.files])
    monkeypatch.setattr(model_identity, "capture_model_identity", changed)
    with pytest.raises(ValueError, match="actual decoder/toolchain artifact"):
        payload.bytelevel_payload_policy()
