"""Actual immutable runtime/row-policy provenance, never a forged HOLD schema."""
import os
from pathlib import Path
import pytest
from tools import v100_backend_probe as probe
from tools.moe_screen_coverage import held_hot_rows


def test_exact_current_hold_schema_and_inherited_row_policy_before_stage():
    runtime, parent = (Path(os.environ[name]) for name in
                       ('TENSORFOLD_TEST_RUNTIME_HOLD', 'TENSORFOLD_TEST_ROW_POLICY_HOLD'))
    wheel = Path(os.environ['TENSORFOLD_TEST_INSTALLED_WHEEL'])
    runtime_sha = 'b5c78eb23e5d4c5111a707506a8afb5623cef81c0dd6412c70c1376786560f3a'
    parent_sha = '8021154e213e1383b2a26290d14cb4d5318dfb0c80eee7cb003e3f9fc5e2979d'
    result = probe.validate_profile_row_policy(runtime, parent, wheel,
        runtime_sha=runtime_sha, row_sha=parent_sha)
    assert result['APP'] == '4b0ea6b51047f93b2b973ed5a467d8c279b94f6d85c683c4258a712ff6edd99f'
    assert result['rows'] == list(held_hot_rows(parent, parent_sha))
    assert len(result['rows']) == 16
    # Preserve the original failure classification: descriptor schema is not
    # the row-policy parser's schema and must never be renamed/substituted.
    with pytest.raises(ValueError, match='held hot16 policy'):
        held_hot_rows(runtime, runtime_sha)
    with pytest.raises(ValueError, match='runtime HOLD identity'):
        probe.validate_profile_row_policy(parent, parent, wheel,
            runtime_sha=runtime_sha, row_sha=parent_sha)
    with pytest.raises(ValueError, match='held hot-row source policy hash'):
        probe.validate_profile_row_policy(runtime, parent, wheel,
            runtime_sha=runtime_sha, row_sha=runtime_sha)
