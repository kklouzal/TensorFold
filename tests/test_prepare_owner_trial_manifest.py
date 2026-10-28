"""Loader-derived real-file manifests, without model execution or CUDA."""
import hashlib
import json
import shutil

import pytest

from tools import prepare_owner_trial_manifest as prepare


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    from test_serving_profile_manifest import profile_files

    root, _, _ = profile_files(tmp_path)
    target = root / "target"
    draft = root / "draft"
    draft.mkdir()
    shutil.copyfile(target / "model.gguf", draft / "mtp.gguf")
    monkeypatch.setattr(prepare.hub, "resolve", lambda model, *, download: target)
    monkeypatch.setattr(prepare, "drafter_metadata_source", lambda mtp: draft)
    return target, draft, root / "trial.json"


@pytest.mark.torch
def test_prepared_trial_hashes_real_loader_closure_and_passes_shipping_preload(inputs):
    target, draft, output = inputs
    document = prepare.create(target, draft / "mtp.gguf", output)
    selected = prepare.SimpleNamespace(model_dir=target, mtp_model_dir=draft,
                                       method_geometry=prepare.SimpleNamespace(method="mtp"))
    files, _ = prepare.execution_identity.artifact_files(selected)
    assert document["artifacts"] == {name: {"size": path.stat().st_size,
                                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                      for name, path in files.items()}
    assert "draft/mtp.gguf" in files and "metadata/tokenizer.json" in files
    assert set(document) == {"schema", "version", "artifacts", "preparation"}
    assert document["preparation"] == {"kernel_family": "sm70-paged-owner-q8-compact-v1",
        "graph_family": "uncaptured", "collective_family": "tp8-allgather-v1",
        "graph_budget_bytes": 0, "trial": True}
    assert prepare.execution_identity.begin_execution_manifest(selected, output) == document
    assert json.loads(output.read_text()) == document


@pytest.mark.torch
def test_missing_tokenizer_fails_in_real_loader_inventory_before_publication(inputs):
    target, draft, output = inputs
    (target / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer assets"):
        prepare.create(target, draft / "mtp.gguf", output)
    assert not output.exists()


@pytest.mark.torch
def test_manifest_cannot_overwrite_a_loader_input(inputs):
    target, draft, _ = inputs
    original = (target / "config.json").read_bytes()
    with pytest.raises(ValueError, match="replace a selected loader input"):
        prepare.create(target, draft / "mtp.gguf", target / "config.json")
    assert (target / "config.json").read_bytes() == original


@pytest.mark.torch
def test_changed_input_after_hashing_refuses_before_publication(inputs, monkeypatch):
    target, draft, output = inputs
    original = prepare.artifact_identity
    def change(path):
        result = original(path)
        if path.name == "mtp.gguf":
            path.write_bytes(path.read_bytes() + b"changed after complete hash")
        return result
    monkeypatch.setattr(prepare, "artifact_identity", change)
    with pytest.raises(ValueError, match="changed during"):
        prepare.create(target, draft / "mtp.gguf", output)
    assert not output.exists()


@pytest.mark.torch
def test_explicit_graph_manifest_keeps_real_artifact_census_and_full8(inputs):
    target, draft, output = inputs
    document = prepare.create(target, draft / "mtp.gguf", output, graph_budget_bytes=512 << 20)
    assert document["preparation"] == {"kernel_family": "sm70-paged-owner-q8-compact-v1",
        "graph_family": "owner-attention-v1", "collective_family": "tp8-allgather-v1",
        "graph_budget_bytes": 512 << 20, "trial": True}
    assert document["artifacts"]
    assert not any(name in document for name in ("evidence", "correctness", "performance"))
    selected = prepare.SimpleNamespace(model_dir=target, mtp_model_dir=draft,
                                       method_geometry=prepare.SimpleNamespace(method="mtp"))
    assert prepare.execution_identity.begin_execution_manifest(selected, output) == document


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_graph_allowance_refuses_before_input_or_output_access(tmp_path, bad):
    with pytest.raises(ValueError, match="nonnegative integer"):
        prepare.create(tmp_path / "absent-target", tmp_path / "absent-draft", tmp_path / "out", graph_budget_bytes=bad)
    assert not (tmp_path / "out").exists()
