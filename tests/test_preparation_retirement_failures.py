"""A semantic refusal cannot conceal unretired preparation ownership."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("helper", ["stage", "draft", "media"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_expected_refusal_preserves_cleanup_failure_and_contains_uncertain_cohort(helper, cleanup_fails, monkeypatch):
    from tensorfold.cuda import cohort
    from tensorfold.families.qwen4_exp.cuda import execution_identity as identity, media_runtime, stage_topology

    primary = identity._ManifestRefusal("expected preparation refusal")
    retirement = OSError("preparation key retirement failed")
    notifications, retired = [], []

    def phase(*args, **kwargs):
        raise primary

    def retire(owner, *args, **kwargs):
        retired.append(owner)
        if cleanup_fails:
            raise retirement

    monkeypatch.setattr(identity, "_phase", phase)
    monkeypatch.setattr(identity, "_retire_store", retire)
    monkeypatch.setattr(cohort, "notify_failure", lambda: notifications.append(True))
    engine = SimpleNamespace(rank=0, tp=8, _startup_store=object(), _abort_comm=False)
    with pytest.raises(identity._ManifestRefusal) as caught:
        if helper == "stage":
            stage_topology.prepare_pp1(engine, None)
        elif helper == "draft":
            identity.prepare_dflash_pairing(engine, {}, target_config={}, plan=None, context=1)
        else:
            media_runtime.prepare_media_source(engine, None)
    assert caught.value is primary and len(retired) == 1
    assert engine._abort_comm is cleanup_fails
    assert bool(notifications) is cleanup_fails
    assert identity._unretired(primary) is cleanup_fails
    if cleanup_fails:
        assert retired[0]._execution_manifest_uncertain
        assert primary.__cause__ is not None  # Cleanup failure and retained owner remain reachable.
