import json

import pytest
import torch

from tests.paged_attention_diagnostic import differences


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_exact_and_adjacent_ieee_values(dtype):
    value = torch.tensor([-1., 0., 1.], dtype=dtype)
    same = differences(value, value.clone())
    assert same["exact_bits"] and same["max_ulp_finite"] == 0
    following = torch.nextafter(value, torch.full_like(value, torch.inf))
    result = differences(value, following)
    assert not result["exact_bits"] and result["different_bits"] == 3
    assert result["max_ulp_finite"] == 1
    assert result["zero_reference_nonzero_candidate"] == 1
    json.dumps(result, allow_nan=False)


def test_signed_zero_and_nonfinite_are_not_hidden_by_numeric_equality():
    value = torch.tensor([0., float("inf"), float("nan")])
    other = value.clone()
    other[0] = -0.
    result = differences(value, other)
    assert result["different_bits"] == 1 and result["max_ulp_finite"] == 1
    assert result["reference_nonfinite"] == result["candidate_nonfinite"] == 2
    assert result["max_absolute_finite"] == 0
    json.dumps(result, allow_nan=False)


def test_relative_zero_and_large_difference_are_explicit_and_json_finite():
    reference = torch.tensor([0., -torch.finfo(torch.float32).max, 2.])
    candidate = torch.tensor([1., torch.finfo(torch.float32).max, 3.])
    result = differences(reference, candidate)
    assert result["zero_reference_nonzero_candidate"] == 1
    assert result["max_relative_nonzero_reference"] == 2
    assert result["max_absolute_finite"] > torch.finfo(torch.float32).max
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("reference,candidate", [
    (torch.zeros(2), torch.zeros(3)),
    (torch.zeros(2), torch.zeros(2, dtype=torch.float16)),
    (torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)),
])
def test_operand_protocol_is_checked(reference, candidate):
    with pytest.raises(ValueError):
        differences(reference, candidate)
