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
    assert prefix_policy.SHARED_FUSION_PROFILE in router_blas.PROFILES
    assert prefix_policy.identity(prefix_policy.SHARED_FUSION_PROFILE) != prefix_policy.identity(router_blas.PROFILE)
    assert prefix_policy.state_epoch(prefix_policy.SHARED_FUSION_PROFILE) != prefix_policy.state_epoch(router_blas.PROFILE)
    device, host = gguf_ops.shared_fusion_workspace_bytes()
    assert device == 2 << 20 and host == 64 << 10
    assert 32 * 33 * 80 * 2 == 168960 < device
    source = Path(__file__).parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(source.read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.sm70_profile == SHARED_FUSION_TRIAL")
    options = {"device_workspace_bytes": 0, "host_workspace_bytes": 0}
    scope = dict(self=NS(sm70_profile=prefix_policy.SHARED_FUSION_PROFILE),
                 SHARED_FUSION_TRIAL=prefix_policy.SHARED_FUSION_PROFILE, options=options, tp=8)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"), scope)
    assert options == {"device_workspace_bytes": 2 << 20, "host_workspace_bytes": 8 * (64 << 10)}
    admission = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                     and n.value.func.id == "admit")
    assert block.end_lineno < admission.lineno


def test_explicit_private_flag_and_current_all6_quality_intents_must_agree(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", prefix_policy.SHARED_FUSION_PROFILE)
    monkeypatch.setenv("TENSORFOLD_SM70_GDN_STATE_BATCH", "1")
    raw = cli(tmp_path) + ["--native-dense-trial", "--owner-batch-prelude-trial", "--router-blas-trial",
        "--gdn-state-batch-trial", "--session-ceiling", "24", "--concurrency-grid", "--fast-grid",
        "--fast-grid-protocol", "fast-grid-v2", "--tokens", "2048", "--quality-spotchecks"]
    with pytest.raises(SystemExit):
        probe.arguments(raw)
    args = probe.arguments(raw + ["--packed-shared-fusion-trial"])
    assert args.packed_shared_fusion_trial and args.router_blas_trial and args.gdn_state_batch_trial
    assert args.quality_spotchecks and args.session_ceiling == 24 and args.tokens == 2048
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", router_blas.PROFILE)
    with pytest.raises(SystemExit):
        probe.arguments(raw + ["--packed-shared-fusion-trial"])



def test_actual_expert_constructor_count_is_routed_not_total_slots():
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts
    bank = GGUFExperts(object(), object(), object(), (), count=512, width=80, dims=2560)
    assert bank.count == 513 and bank.count - 1 == 512


def test_cuda_private_credit_rejects_hip_allocator_alias_before_warm(monkeypatch):
    from tensorfold.cuda.sm70 import execution_policy as policy
    metadata = dict(policy._FAMILY_GEOMETRY, dtype="float16", mamba_ssm_dtype="float32", output_gate_type="sigmoid")
    monkeypatch.setattr(policy, "_family_runtime", lambda _torch: None)
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "large_segment_size_mb:40")
    def forbidden():
        pytest.fail("nondefault allocator reached software/native preparation")
    monkeypatch.setattr(policy, "_family_software", forbidden)
    target = NS(cuda=NS(memory=NS(get_allocator_backend=lambda: "native")))
    with pytest.raises(ValueError, match="default native CUDA allocator"):
        policy._prepare_family_trial(metadata, 8, target, packed=True, shared_fusion=True)
