"""Finite cold capture selection does not narrow supported eager shapes."""
from copy import deepcopy

import pytest

from tensorfold.families.qwen4_exp.cuda.exact_graph_policy import parse_policy


def policy():
    return {'policy': 'paid-subset-v1', 'keys': [['verify', 5, 1], ['mtp', 1, 1]],
            'evidence_sha256': 'a' * 64}


def test_subset_is_canonical_copied_and_idempotent():
    request = policy()
    selected = parse_policy(request)
    request['keys'][0][1] = 99
    assert selected['keys'] == [['mtp', 1, 1], ['verify', 5, 1]]
    assert parse_policy(selected) == selected
    assert parse_policy(None) is None


@pytest.mark.parametrize('change', [
    {'policy': 'full'}, {'evidence_sha256': 'A' * 64}, {'evidence_sha256': 'a' * 63},
    {'evidence_sha256': None}, {'keys': []}, {'keys': [['verify', True, 1]]},
    {'keys': [['verify', 1, True]]}, {'keys': [['verify', 0, 1]]},
    {'keys': [['verify', 5, 2]]}, {'keys': [['verify', 1, 2]]},
    {'keys': [['prefill', 1, 1]]}, {'keys': [['mtp', 1, 1], ['mtp', 1, 1]]},
    {'keys': [['verify', 1]]}, {'extra': True},
])
def test_invalid_authority_geometry_and_duplicates_fail(change):
    with pytest.raises(ValueError, match='exact graph subset'):
        parse_policy({**deepcopy(policy()), **change})


def test_observed_demand_accepts_ragged_totals_without_widening_legacy_policy():
    request = {**policy(), 'policy': 'observed-demand-v1',
               'keys': [['verify', 28, 8], ['mtp', 3, 3]]}
    assert parse_policy(request)['keys'] == [['mtp', 3, 3], ['verify', 28, 8]]
    assert parse_policy(parse_policy(request)) == parse_policy(request)
    with pytest.raises(ValueError):
        parse_policy({**request, 'policy': 'paid-subset-v1'})
