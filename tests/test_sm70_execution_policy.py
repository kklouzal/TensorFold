"""The launch profile selects admitted existing operators before model ownership."""
from types import SimpleNamespace as NS
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def policy(monkeypatch, tmp_path):
    from tensorfold.cuda.sm70 import execution_policy as policy, family_ops, routing_native
    from tensorfold.cuda.sm70 import attention_qualification as qualification
    from tensorfold.families.qwen4_exp.cuda import dspark_context_graphs, dspark_proposal_graphs, gguf_prefill, gguf_weights, padded_linears
    for owner,name in [(policy,"_SELECTED"),(policy,"_FAMILY_SOFTWARE"),(family_ops,"_NATIVE_FAMILY"),(family_ops,"_NATIVE_ATTENTION"),
                       (routing_native,"_NATIVE_LOGITS"),(padded_linears,"_ENABLED"),
                       (dspark_context_graphs,"_ENABLED"),(dspark_proposal_graphs,"_ENABLED"),
                       (gguf_prefill,"WORKSPACE_BUDGET_BYTES"),(gguf_weights.GGUFExperts,"implementation"),
                       (gguf_weights.GGUFConcat,"implementation")]:
        monkeypatch.setattr(owner,name,getattr(owner,name))
    monkeypatch.setattr(qualification,"runtime_identity",lambda device:dict(qualification.REQUIRED_IDENTITY))
    monkeypatch.setattr(torch.cuda,"current_device",lambda:0)
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda *a:(7,0))
    for name in (*policy._GRAPH_KEYS, "NCCL_ENV_PLUGIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NCCL_CONF_FILE", str(tmp_path / "user.conf"))
    monkeypatch.setattr(policy, "_SYSTEM_CONF", str(tmp_path / "system.conf"))
    return NS(module=policy,family=family_ops,routing=routing_native,padded=padded_linears,
              proposal=dspark_proposal_graphs,context=dspark_context_graphs,prefill=gguf_prefill,weights=gguf_weights,qualification=qualification)


def text():
    return dict(hidden_size=2560,num_hidden_layers=48,num_attention_heads=24,num_key_value_heads=2)


@pytest.mark.parametrize("profile",["reference","optimized"])
def test_explicit_profile_selects_same_engine_operators_and_memory_policy(policy,monkeypatch,profile):
    monkeypatch.setenv(policy.module.PROFILE_ENV,profile)
    assert policy.module.configure(text(),tp=8,torch=torch)==profile
    enabled=profile=="optimized"
    assert policy.family._NATIVE_FAMILY==policy.family._NATIVE_ATTENTION==enabled
    assert policy.routing._NATIVE_LOGITS==policy.padded._ENABLED==enabled
    assert policy.proposal._ENABLED==policy.context._ENABLED==enabled
    assert policy.weights.GGUFExperts.implementation==("packed_exact" if enabled else "reference")
    assert policy.weights.GGUFConcat.implementation==("native32" if enabled else "reference")
    assert policy.prefill.WORKSPACE_BUDGET_BYTES==((1<<30) if enabled else None)


@pytest.mark.parametrize("change",["geometry","tp","runtime"])
def test_unqualified_optimized_profile_refuses_before_changing_operators(policy,monkeypatch,change):
    monkeypatch.setenv(policy.module.PROFILE_ENV,"optimized")
    before=policy.weights.GGUFExperts.implementation
    metadata=text()
    tp=8
    if change=="geometry":
        metadata["hidden_size"]=4096
    if change=="tp":
        tp=4
    if change=="runtime":
        monkeypatch.setattr(policy.qualification,"runtime_identity",lambda d:{})
    with pytest.raises(ValueError,match="qualified"):
        policy.module.configure(metadata,tp=tp,torch=torch)
    assert policy.weights.GGUFExperts.implementation==before


def test_unknown_profile_refuses_before_cuda_or_model_ownership(policy,monkeypatch):
    monkeypatch.setenv(policy.module.PROFILE_ENV,"typo")
    with pytest.raises(ValueError,match="must be"):
        policy.module.configure({},tp=8,torch=NS())


def test_loaded_weights_receive_only_the_selected_profile_metadata(policy,monkeypatch):
    calls=[]
    monkeypatch.setattr(policy.module,"_SELECTED","optimized")
    monkeypatch.setattr(policy.qualification,"metadata_prepare",lambda w:calls.append(w))
    w=NS(meta={})
    policy.module.prepare_weights(w)
    assert calls==[w]
    assert w.meta==dict(sm70_profile="optimized",capture_ready=True,device_greedy_mtp_ready=True,device_greedy_native_ready=True)


@pytest.mark.parametrize("value", ["1", "+1", "01", "0x1", "  1suffix", "", "0" * 5000 + "1"])
def test_documented_nccl_integer_prefix_defaults_are_preserved(policy, monkeypatch, value):
    for name in policy.module._GRAPH_KEYS:
        monkeypatch.setenv(name, value)
    record = policy.module.graph_inputs()
    assert record["effective"] == {"graph_usage_mode": 2, "graph_stream_ordering": 1,
                                   "graph_mixing_support": 1}


@pytest.mark.parametrize("name", ["NCCL_GRAPH_MIXING_SUPPORT", "NCCL_GRAPH_STREAM_ORDERING"])
@pytest.mark.parametrize("value", ["0", "-0", "2", "0b1", "0x0", "invalid"])
def test_unqualified_nccl_inputs_refuse_before_operator_or_journal_mutation(policy, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    before, records = policy.weights.GGUFExperts.implementation, []
    with pytest.raises(ValueError, match=name):
        policy.module.configure(text(), tp=8, torch=torch, record_graph_inputs=records.append)
    assert policy.weights.GGUFExperts.implementation == before and records == []


def test_native_file_precedence_ignores_unrelated_values_and_records_no_raw_secrets(policy, tmp_path, monkeypatch):
    import json
    user, system = tmp_path / "user.conf", tmp_path / "system.conf"
    user.write_text("#NCCL_GRAPH_STREAM_ORDERING=0\nUNRELATED=fixture-private-value\n"
                    "NCCL_GRAPH_STREAM_ORDERING=1\nNCCL_GRAPH_STREAM_ORDERING=0\n"
                    "NCCL_GRAPH_MIXING_SUPPORT=0\n")
    system.write_text("NCCL_GRAPH_STREAM_ORDERING=0\nNCCL_GRAPH_MIXING_SUPPORT=0\n")
    monkeypatch.setenv("NCCL_GRAPH_MIXING_SUPPORT", "1")
    record = policy.module.graph_inputs()
    assert len(record["files"]) == 2 and all(item["sha256"] for item in record["files"])
    assert "fixture-private-value" not in json.dumps(record)
    assert record["qualification"] == "validated-inputs-only"
    monkeypatch.delenv("NCCL_GRAPH_MIXING_SUPPORT")
    with pytest.raises(ValueError, match="NCCL_GRAPH_MIXING_SUPPORT"):
        policy.module.graph_inputs()


def test_native_fgets_chunk_and_value_buffer_boundaries_are_not_ini_lines(policy, tmp_path):
    user = tmp_path / "user.conf"
    user.write_bytes(b"x" * 4095 + b"NCCL_GRAPH_MIXING_SUPPORT=0\n")
    with pytest.raises(ValueError, match="NCCL_GRAPH_MIXING_SUPPORT"):
        policy.module.graph_inputs()
    user.write_bytes(b"NCCL_GRAPH_MIXING_SUPPORT=" + b"0" * 1022 + b"1\r\n")
    policy.module.graph_inputs()
    user.write_bytes(b"NCCL_GRAPH_MIXING_SUPPORT=" + b"0" * 1023 + b"1\n")
    with pytest.raises(ValueError, match="NCCL_GRAPH_MIXING_SUPPORT"):
        policy.module.graph_inputs()


def test_explicit_environment_plugin_is_an_unqualified_input(policy, tmp_path, monkeypatch):
    (tmp_path / "user.conf").write_text("NCCL_ENV_PLUGIN=unqualified-provider\n")
    with pytest.raises(ValueError, match="plugin requires separate"):
        policy.module.graph_inputs()
    monkeypatch.setenv("NCCL_ENV_PLUGIN", "none")
    policy.module.graph_inputs()


def test_config_version_binding_refuses_restored_mtime_and_new_missing_file(policy, tmp_path):
    import os
    user = tmp_path / "user.conf"
    first = policy.module.graph_inputs()
    user.write_text("NCCL_GRAPH_STREAM_ORDERING=1\n")
    with pytest.raises(ValueError, match="file changed"):
        policy.module.verify_graph_inputs(first)
    record, before = policy.module.graph_inputs(), user.stat()
    user.write_text("NCCL_GRAPH_STREAM_ORDERING=0\n")
    os.utime(user, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert user.stat().st_size == before.st_size and user.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="file changed"):
        policy.module.verify_graph_inputs(record)


def test_config_read_bound_is_checked_before_open(policy, tmp_path, monkeypatch):
    from tensorfold.cuda import tensor_file
    monkeypatch.setattr(tensor_file, "MAX_HEADER_BYTES", 8)
    (tmp_path / "user.conf").write_bytes(b"x" * 9)
    monkeypatch.setattr(tensor_file, "_regular_stream", lambda *_a: pytest.fail("oversized config was opened"))
    with pytest.raises(ValueError, match="read bound"):
        policy.module.graph_inputs()


def test_absolute_config_does_not_bind_unused_home_or_working_directory(policy, monkeypatch):
    record = policy.module.graph_inputs()
    assert "HOME" not in record["environment"] and "cwd" not in record["environment"]
    def unavailable():
        raise FileNotFoundError("working directory was removed")
    with monkeypatch.context() as changed:
        changed.setattr(policy.module.os, "getcwd", unavailable)
        policy.module.graph_inputs()
        policy.module.verify_graph_inputs(record)


def test_native_relative_path_buffer_limit_precedes_working_directory_resolution(policy, tmp_path, monkeypatch):
    from pathlib import Path
    raw = "/".join(["a" * 230] * 4 + ["b" * 120])
    native = Path(raw[:1023])  # NCCL snprintf's 1024-byte destination includes its terminating NUL.
    path = tmp_path / native
    path.parent.mkdir(parents=True)
    path.write_text("NCCL_GRAPH_STREAM_ORDERING=0\n")
    assert len(str(path)) > 1023 and not (tmp_path / raw).exists()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NCCL_CONF_FILE", raw)
    with pytest.raises(ValueError, match="NCCL_GRAPH_STREAM_ORDERING"):
        policy.module.graph_inputs()


def test_changed_opened_config_is_refused_before_policy_journal(policy, tmp_path, monkeypatch):
    from contextlib import contextmanager
    from tensorfold.cuda import tensor_file
    path = tmp_path / "user.conf"
    path.write_text("NCCL_GRAPH_MIXING_SUPPORT=1\n")
    original, changed, records = tensor_file._regular_stream, [], []

    @contextmanager
    def mutating(filename):
        with original(filename) as source:
            class Stream:
                def fileno(self):
                    return source.fileno()

                def readline(self, size):
                    data = source.readline(size)
                    if not changed:
                        path.write_text("NCCL_GRAPH_MIXING_SUPPORT=0\n")
                        changed.append(True)
                    return data

                def read(self, size):
                    return source.read(size)
            yield Stream()

    monkeypatch.setattr(tensor_file, "_regular_stream", mutating)
    with pytest.raises(ValueError, match="changed"):
        policy.module.configure(text(), tp=8, torch=torch, record_graph_inputs=records.append)
    assert changed == [True] and records == []


def test_policy_journal_failure_precedes_operator_changes(policy):
    before = policy.weights.GGUFExperts.implementation
    primary = MemoryError("policy journal failed")
    def failed(record):
        assert record["effective"]["graph_usage_mode"] == 2
        assert policy.weights.GGUFExperts.implementation == before
        raise primary
    with pytest.raises(MemoryError) as caught:
        policy.module.configure(text(), tp=8, torch=torch, record_graph_inputs=failed)
    assert caught.value is primary and policy.weights.GGUFExperts.implementation == before


def test_single_rank_without_nccl_does_not_validate_unused_nccl_inputs(policy, monkeypatch):
    monkeypatch.setenv(policy.module.PROFILE_ENV, "reference")
    monkeypatch.setattr(policy.module, "graph_inputs", lambda: pytest.fail("single rank consulted NCCL"))
    records = []
    assert policy.module.configure(text(), tp=1, torch=torch, record_graph_inputs=records.append) == "reference"
    assert records == [None]
