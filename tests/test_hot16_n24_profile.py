"""Current hot16 diagnostic admission/label controls; no GPU/model-speed claim."""
import os
from pathlib import Path
import pytest
from tools import v100_backend_probe as probe
from tests.test_joint_single_profile import diagnostic_args


def test_actual_n24_parser_keeps_current_optins_and_private_reader_funding(tmp_path, monkeypatch):
    raw = diagnostic_args(tmp_path, monkeypatch)
    raw[raw.index('--profile-concurrency') + 1] = '24'
    monkeypatch.setenv('TENSORFOLD_SM70_HC_MOE_REGION_GRAPHS', '1')
    monkeypatch.setenv('TENSORFOLD_SM70_HC_MOE_SHARED_POOL', '1')
    args = probe.arguments(raw)
    assert args.profile_concurrency == 24 and args.tokens == 2048
    assert args.session_ceiling == 24 and args.startup_content_workers == 3
    assert args.gdn_state_batch_trial and args.packed_shared_fusion_trial
    assert not args.quality_spotchecks
    assert probe.PROFILE_READER_BYTES_PER_RANK == 6 << 30
    with pytest.raises(SystemExit):
        probe.arguments(raw + ['--quality-spotchecks'])


def test_actual_normal_current_wholewheel_covers_hot16_sources():
    wheel = os.environ['TENSORFOLD_TEST_INSTALLED_WHEEL']
    record = probe.installed_identity(Path(wheel))
    for path in ('hc_moe_region.py', 'moe_region_catalogue.py', 'multi_graphs.py',
                 'candidate_region_graphs.py', 'candidate_region_catalogue.py'):
        assert 'tensorfold/families/qwen4_exp/cuda/' + path in record['affected_files']


def test_actual_hc_override_label_preserves_admission_failure_and_exit_fault(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.hc_moe_region import HCReadoutMoERegion
    from tools import concurrent_observers
    from tests.test_v100_backend_probe import retained_failure
    monkeypatch.setattr(concurrent_observers, '_owner', None)
    diagnostic = OSError('label exit')
    class Label:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            pass
        def __exit__(self, *_args):
            raise diagnostic
    plan = probe.project_observers(Label)
    matches = [row for row in plan if row[0] is HCReadoutMoERegion and row[1] == 'replay_prevalidated']
    assert len(matches) == 1 and len(plan) <= 128
    owner, name, original, wrapper = matches[0]
    assert vars(owner)[name] is original and wrapper.__wrapped__ is original
    region = object.__new__(HCReadoutMoERegion)
    region.prepared, region.failure = False, None
    with pytest.raises(RuntimeError, match='not an admitted prepared component') as caught:
        wrapper(region)
    assert retained_failure(caught.value, diagnostic)
    assert vars(owner)[name] is original  # Planning never installs/changes APP.
