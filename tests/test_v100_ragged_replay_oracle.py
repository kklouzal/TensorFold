"""CPU controls for the native oracle's fixed workload and exact comparator.

These do not execute or qualify graph replay. The real idle TP8 decoder is the
only execution path of the native component gate.
"""
import pytest
import torch

from tools.v100_ragged_replay_oracle import LAYOUTS, SEED_LENGTHS, _equal


def test_layouts_preserve_requested_key_and_change_cold_last_row_maps():
    seed = (4, 4, 4, 4, 3, 3, 3, 3)
    maps = set()
    for counts in (seed, *LAYOUTS):
        assert len(counts) == 8 and sum(counts) == 28
        assert min(counts) >= 1 and max(counts) <= 5
        cursor, last = 0, []
        for count in counts:
            cursor += count
            last.append(cursor - 1)
        maps.add(tuple(last))
    assert len(maps) == 3


def test_exact_comparator_accepts_independent_same_values():
    expected = {'logits': torch.tensor([0.0, 1.0]), 'pos': 3}
    _equal({'logits': expected['logits'].clone(), 'pos': 3}, expected)


@pytest.mark.parametrize('actual', [
    {'logits': torch.tensor([0.0, 1.000001]), 'pos': 3},
    {'logits': torch.tensor([0.0, 1.0]), 'pos': 4},
    {'logits': torch.tensor([0.0, 1.0])},
    {'logits': torch.tensor([float('nan'), 1.0]), 'pos': 3},
    {'logits': torch.tensor([-0.0, 1.0]), 'pos': 3},
])
def test_exact_comparator_refuses_changed_output_metadata_or_nonfinite(actual):
    with pytest.raises(AssertionError):
        _equal(actual, {'logits': torch.tensor([0.0, 1.0]), 'pos': 3})


def test_fixed_operation_input_is_closed_bounded_and_duplicate_free(tmp_path):
    import json

    from tools.v100_owned_qualification import _ragged_input
    data = {'schema': 'tensorfold.native-ragged-replay-input-v1', 'key': [28, 8],
            'layouts': [list(counts) for counts in LAYOUTS], 'slots': 10,
            'maximum_context_rows': 10, 'comparison': 'bit-exact'}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    assert len(_ragged_input(path)) == 64
    for raw in (json.dumps({**data, 'slots': 10.0}), json.dumps({**data, 'unknown': 1}),
                '{"slots":10,"slots":10}', ' ' * 4097):
        path.write_text(raw)
        with pytest.raises(ValueError):
            _ragged_input(path)


def test_seed_and_ragged_windows_fit_prepaid_short_prefix_with_unequal_contexts():
    contexts = tuple(length + (index % 2 == 0) for index, length in enumerate(SEED_LENGTHS))
    assert min(contexts) > 0 and len(set(contexts)) > 1
    for counts in LAYOUTS:
        assert max(context + count for context, count in zip(contexts, counts)) <= 10
