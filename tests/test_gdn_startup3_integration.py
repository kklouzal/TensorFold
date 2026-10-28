"""Private GDN + three-whole-file-hash composition; no GPU or model load."""
import json

from tools import v100_backend_probe as probe
from tests.test_v100_backend_probe import cli
from tensorfold.families.qwen4_exp.cuda import startup_content


def test_exact_global_startup_funding_keeps_default_one_and_private_three(tmp_path):
    files = {name: tmp_path / name for name in ("target-a", "target-b", "draft")}
    one = startup_content.StartupContentPlan(files)
    three = startup_content.StartupContentPlan(files, workers=3)
    assert one.workers == 1 and one.host_bytes == 75501568
    assert three.workers == 3 and three.host_bytes == 226504704
    assert three.record()["global_host_bytes"] == 226504704


def test_actual_launcher_inherits_both_gdn_and_three_startup_worker_intents(tmp_path, monkeypatch):
    from tensorfold.cuda import expert_launch

    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "source-dense-owner-batch-prelude-trial")
    monkeypatch.setenv("TENSORFOLD_SM70_GDN_STATE_BATCH", "1")
    monkeypatch.delenv("TENSORFOLD_STARTUP_CONTENT_OVERLAP", raising=False)
    monkeypatch.delenv("TENSORFOLD_STARTUP_CONTENT_WORKERS", raising=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema":"tensorfold.execution-profile-manifest","version":1,"artifacts":{}}))
    raw = cli(tmp_path) + ["--execution-profile-manifest",str(manifest),"--overlap-content-capture",
        "--startup-content-workers","3","--gdn-state-batch-trial","--owner-batch-prelude-trial",
        "--native-dense-trial","--session-ceiling","24","--concurrency-grid","--fast-grid",
        "--fast-grid-protocol","fast-grid-v2","--tokens","2048","--quality-spotchecks"]
    args = probe.arguments(raw)
    assert args.startup_content_workers == 3 and args.gdn_state_batch_trial
    observed = []
    def supervise(commands, environments, **_kwargs):
        for command, environment in zip(commands,environments,strict=True):
            assert environment["TENSORFOLD_STARTUP_CONTENT_WORKERS"] == "3"
            assert environment["TENSORFOLD_STARTUP_CONTENT_OVERLAP"] == "1"
            assert environment["TENSORFOLD_SM70_GDN_STATE_BATCH"] == "1"
            assert environment["TENSORFOLD_SM70_PROFILE"] == "source-dense-owner-batch-prelude-trial"
            assert "--startup-content-workers" in command and "--gdn-state-batch-trial" in command
            observed.append(True)
        return 0
    monkeypatch.setattr(probe,"installed_identity",lambda _wheel:{})
    monkeypatch.setattr(expert_launch,"supervise",supervise)
    assert probe.launch(args,raw) == 0 and len(observed) == 8
