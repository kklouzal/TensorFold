"""Private composition boundaries; numerical acceptance requires actual model tasks."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from tensorfold.cuda.sm70 import gguf_ops, router_blas
from tensorfold.families.qwen4_exp.cuda import prefix_policy
from tools import v100_backend_probe as probe
from tests.test_v100_backend_probe import cli


def test_private_epoch_and_temporary_operating_credit_are_explicit_before_admit():
    assert prefix_policy.WMMA80_PROFILE in router_blas.PROFILES
    assert prefix_policy.identity(prefix_policy.WMMA80_PROFILE) != prefix_policy.identity(router_blas.PROFILE)
    assert prefix_policy.state_epoch(prefix_policy.WMMA80_PROFILE) != prefix_policy.state_epoch(router_blas.PROFILE)
    device, host = gguf_ops.wmma80_workspace_bytes()
    assert device == (20 + 2) << 20 and host == 64 << 10
    assert 32 * 33 * 80 * (8 * 2 * 4 + 2) == 5575680 < device
    source = Path(__file__).parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(source.read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.sm70_profile == WMMA80_TRIAL")
    options = {"device_workspace_bytes": 0, "host_workspace_bytes": 0}
    scope = dict(self=NS(sm70_profile=prefix_policy.WMMA80_PROFILE),
                 WMMA80_TRIAL=prefix_policy.WMMA80_PROFILE, options=options, tp=8)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"), scope)
    assert options == {"device_workspace_bytes": 22 << 20, "host_workspace_bytes": 8 * (64 << 10)}
    admission = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                     and n.value.func.id == "admit")
    assert block.end_lineno < admission.lineno


def test_explicit_private_flag_and_current_all6_quality_intents_must_agree(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", prefix_policy.WMMA80_PROFILE)
    monkeypatch.setenv("TENSORFOLD_SM70_GDN_STATE_BATCH", "1")
    raw = cli(tmp_path) + ["--native-dense-trial", "--owner-batch-prelude-trial", "--router-blas-trial",
        "--gdn-state-batch-trial", "--session-ceiling", "24", "--concurrency-grid", "--fast-grid",
        "--fast-grid-protocol", "fast-grid-v2", "--tokens", "2048", "--quality-spotchecks"]
    with pytest.raises(SystemExit):
        probe.arguments(raw)
    args = probe.arguments(raw + ["--wmma80-trial"])
    assert args.wmma80_trial and args.router_blas_trial and args.gdn_state_batch_trial
    assert args.quality_spotchecks and args.session_ceiling == 24 and args.tokens == 2048
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", router_blas.PROFILE)
    with pytest.raises(SystemExit):
        probe.arguments(raw + ["--wmma80-trial"])
