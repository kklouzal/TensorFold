"""Native attention's retained arrays have an exact CPU-only admission forecast."""

import pytest

from tensorfold.cuda.sm70.attention_native import workspace_bytes


@pytest.mark.parametrize("geometry,expected", [
    ((1, 3, 256, 256), 10752),
    ((18, 3, 256, 8192), 6193152),
    ((256, 3, 256, 65536), 88080384),
    ((2, 6, 128, 257), 39168),
    ((0, 3, 256, 8192), 0),
])
def test_workspace_bytes_counts_f32_scores_half_probabilities_and_f32_partials(geometry, expected):
    assert workspace_bytes(*geometry) == expected


@pytest.mark.parametrize("geometry", [(-1, 3, 256, 8192), (1, True, 256, 8192), (1, 3, 256, 8192.0)])
def test_workspace_bytes_rejects_noninteger_or_negative_geometry(geometry):
    with pytest.raises(ValueError, match="nonnegative integer"):
        workspace_bytes(*geometry)
