"""Metadata is optional only when absent, and never authorizes arbitrary files."""

import errno
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold import checkpoint_metadata as metadata
from tensorfold.cli import _generation_config, _reply_tokens, _sampling_defaults, build_parser


def test_optional_files_and_host_only_import(tmp_path):
    assert metadata.read_json(tmp_path, "optional.json") is None
    assert metadata.read_text(tmp_path, "optional.jinja") is None
    with pytest.raises(FileNotFoundError):
        metadata.read_json(tmp_path, "config.json", required=True)
    code = (f"import sys,tensorfold; tensorfold.__path__.insert(0,{str(Path(metadata.__file__).parent)!r}); "
            "import tensorfold.checkpoint_metadata; "
            "assert not any(k in sys.modules for k in ('torch', 'mlx', 'triton'))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("payload", ['[]', '{"x":1,"x":2}', '{"x":NaN}', '{"x":"\\ud800"}', ''])
def test_malformed_optional_json_is_not_absent(tmp_path, payload):
    (tmp_path / "optional.json").write_text(payload)
    with pytest.raises((ValueError, UnicodeError)):
        metadata.read_json(tmp_path, "optional.json")


def test_links_and_nonregular_inputs(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "missing.json").symlink_to("broken.json")
    with pytest.raises(FileNotFoundError):
        metadata.read_json(model, "missing.json")
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    (model / "escape.json").symlink_to(outside)
    for name in ("escape.json", "../outside.json", str(outside)):
        with pytest.raises(ValueError, match="authorized|traverse|absolute"):
            metadata.read_json(model, name)
    (model / "directory.json").mkdir()
    with pytest.raises((ValueError, OSError)):
        metadata.read_json(model, "directory.json")


@pytest.mark.parametrize("reader, name", [(metadata.read_json, "default.json"),
                                         (metadata.read_text, "default.jinja")])
def test_missing_optional_leaf_does_not_hide_broken_parent_or_model_root(tmp_path, reader, name):
    model = tmp_path / "model"
    model.mkdir()
    assert reader(model, f"additional/{name}") is None
    (model / "additional").symlink_to("missing-directory", target_is_directory=True)
    with pytest.raises(FileNotFoundError):
        reader(model, f"additional/{name}")
    with pytest.raises(FileNotFoundError):
        reader(tmp_path / "absent-model", name)
    (tmp_path / "file-root").write_text("file")
    with pytest.raises(NotADirectoryError):
        reader(tmp_path / "file-root", name)


@pytest.mark.parametrize("failure", [PermissionError(errno.EACCES, "denied"), OSError(errno.EIO, "read failed")])
def test_optional_metadata_preserves_access_and_io_failures(tmp_path, monkeypatch, failure):
    path = tmp_path / "optional.json"
    original = Path.lstat

    def checked(self):
        if self == path:
            raise failure
        return original(self)

    monkeypatch.setattr(Path, "lstat", checked)
    with pytest.raises(OSError) as raised:
        metadata.read_json(tmp_path, path.name)
    assert raised.value is failure


def test_text_utf8_bound_and_same_read(tmp_path, monkeypatch):
    (tmp_path / "template.jinja").write_bytes('café'.encode())
    assert metadata.read_text(tmp_path, "template.jinja") == 'café'
    monkeypatch.setattr(metadata, "MAX_HEADER_BYTES", 4)
    with pytest.raises(ValueError, match="metadata size"):
        metadata.read_text(tmp_path, "template.jinja")
    (tmp_path / "invalid.jinja").write_bytes(b'\xff')
    with pytest.raises(UnicodeError):
        metadata.read_text(tmp_path, "invalid.jinja")


def test_reply_limit_uses_reply_only_metadata_and_explicit_override(tmp_path):
    args = SimpleNamespace(max_tokens=None)
    assert _reply_tokens(args, tmp_path) == 4096
    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps({"max_length": 20}))
    assert _reply_tokens(args, tmp_path) == 4096  # total length is not a reply limit
    path.write_text(json.dumps({"max_new_tokens": 777}))
    assert _reply_tokens(args, tmp_path) == 777
    assert _reply_tokens(SimpleNamespace(max_tokens=42), tmp_path) == 42
    for value in (0, -1, True, 1.5, "17"):
        path.write_text(json.dumps({"max_new_tokens": value}))
        with pytest.raises(ValueError, match="positive integer"):
            _reply_tokens(args, tmp_path)
        assert _reply_tokens(SimpleNamespace(max_tokens=42), tmp_path) == 42


def test_sampling_override_wins_before_numeric_validation(tmp_path):
    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps({"temperature": "invalid", "top_k": 20, "top_p": .9, "do_sample": False}))
    args = SimpleNamespace(temperature=.7, top_k=0, top_p=0, min_p=None)
    assert _sampling_defaults(args, tmp_path) == {"temperature": .7, "top_k": 0, "top_p": 0.}
    path.write_text('{"do_sample": "false"}')
    with pytest.raises(ValueError, match="do_sample"):
        _generation_config(tmp_path)


def test_model_override_flags_preserve_omission_and_zero():
    parser = build_parser()
    omitted = parser.parse_args(["serve", "owner/model"])
    assert omitted.max_tokens is None and omitted.eos_token_ids is None
    assert omitted.chat_template_file is None and omitted.chat_template_name is None
    explicit = parser.parse_args(["serve", "owner/model", "--eos-token-id", "0", "--eos-token-id", "7",
                                  "--chat-template-name", "tools", "--max-tokens", "11"])
    assert explicit.eos_token_ids == [0, 7] and explicit.max_tokens == 11
    assert explicit.chat_template_name == "tools"
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "owner/model", "--chat-template", "custom.jinja", "--chat-template-name", "tools"])


def test_hub_fetches_ancillary_files_before_preflight_at_cached_revision(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants

    from tensorfold import hub

    snapshot = tmp_path / "models--Vontra--name--with-dashes" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{}')
    monkeypatch.setattr(hub, "cached", lambda *a, **kw: snapshot)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    calls = []
    infos = []

    def info(self, repo_id, **options):
        infos.append((repo_id, options))
        return SimpleNamespace(sha=snapshot.name,
                               siblings=[SimpleNamespace(rfilename=name) for name in
                                         ("config.json", "hf_quant_config.json")])

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", info)

    def fetch(repo_id, **options):
        calls.append((repo_id, options))
        (snapshot / "hf_quant_config.json").write_text('{"producer": {"name": "modelopt"}}')
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    assert hub.metadata("TensorFold/name--with-dashes") == snapshot
    repo_id, options = calls.pop()
    assert repo_id == "Vontra/name--with-dashes"
    assert infos == [(repo_id, {"revision": snapshot.name})]
    assert options["revision"] == snapshot.name
    assert {"hf_quant_config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
            "chat_template.jinja", "additional_chat_templates/*.jinja", "chat_template.json",
            "quantization_config.json", "quant_config.json"} <= \
        set(options["allow_patterns"])
    assert not any("safetensors" in name for name in options["allow_patterns"])
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    assert hub.metadata("TensorFold/name--with-dashes") == snapshot
    assert calls == []
    assert len(infos) == 1


def test_online_metadata_failure_cannot_become_cached_absence(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants

    from tensorfold import hub

    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{}')
    failure = ConnectionError("inventory unavailable")

    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub, "cached", lambda *a, **kw: snapshot)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", unavailable)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: snapshot)
    with pytest.raises(ConnectionError) as raised:
        hub.metadata("owner/model")
    assert raised.value is failure


def test_latest_metadata_resolves_new_revision_then_pins_download(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants

    from tensorfold import hub

    old = tmp_path / "models--owner--model" / "snapshots" / ("a" * 40)
    new = old.with_name("b" * 40)
    new.mkdir(parents=True)
    (new / "config.json").write_text('{}')
    infos, downloads = [], []

    def info(self, repo_id, **options):
        infos.append((repo_id, options))
        return SimpleNamespace(sha=new.name, siblings=[SimpleNamespace(rfilename="config.json")])

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub, "cached", lambda *a, **kw: old)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", info)
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo_id, **kw: downloads.append((repo_id, kw)) or new)
    assert hub.metadata("owner/model", latest=True) == new
    assert infos == [("owner/model", {"revision": None})]
    assert downloads[0][1]["revision"] == new.name


@pytest.mark.parametrize("scenario", ["missing-sidecar", "missing-config", "wrong-download", "wrong-info", "invalid-path"])
def test_hub_metadata_inventory_and_revision_are_binding(tmp_path, monkeypatch, scenario):
    import huggingface_hub
    from huggingface_hub import constants

    from tensorfold import hub

    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{}')
    files = ["config.json", "hf_quant_config.json"]
    if scenario == "missing-config":
        files = []  # A leftover cache file cannot replace the authoritative inventory.
    if scenario == "invalid-path":
        files = ["additional_chat_templates/../outside.jinja"]
    info = SimpleNamespace(sha="b" * 40 if scenario == "wrong-info" else snapshot.name,
                           siblings=[SimpleNamespace(rfilename=name) for name in files])
    downloads = []
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub, "cached", lambda *a, **kw: snapshot)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **kw: info)
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **kw: downloads.append(kw) or
                        (snapshot.with_name("b" * 40) if scenario == "wrong-download" else snapshot))
    with pytest.raises((FileNotFoundError, ValueError)):
        hub.metadata("owner/model")
    assert bool(downloads) == (scenario in ("missing-sidecar", "wrong-download"))


def test_cache_preserves_selected_partial_revision_and_io_error(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import LocalEntryNotFoundError

    from tensorfold import hub

    snapshot = tmp_path / "models--owner--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{}')
    newer = snapshot.with_name("b" * 40)
    newer.mkdir()
    (newer / "config.json").write_text('{}')
    os.utime(snapshot, (1, 1))
    os.utime(newer, (2, 2))
    partial = LocalEntryNotFoundError("selected snapshot is partial")
    partial.snapshot_path = str(snapshot)

    def fail(error):
        def download(*args, **kwargs):
            raise error
        return download

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail(partial))
    assert hub.cached("owner/model", cache_dir=tmp_path) == snapshot
    failure = PermissionError(errno.EACCES, "cannot read selected ref")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail(failure))
    with pytest.raises(PermissionError) as raised:
        hub.cached("owner/model", cache_dir=tmp_path)
    assert raised.value is failure


def test_offline_metadata_never_looks_up_unknown_revision(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants

    from tensorfold import hub

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hub, "cached", lambda *a, **kw: None)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **kw: pytest.fail("offline network access"))
    with pytest.raises(FileNotFoundError, match="offline"):
        hub.metadata("owner/model", latest=True)


@pytest.mark.parametrize("filename", ["quantization_config.json", "quant_config.json"])
@pytest.mark.parametrize("method", ["EXL3", "gptq", "awq"])
def test_explicit_sidecar_only_format_is_visible_to_early_preflight(tmp_path, filename, method):
    from tensorfold.families import quant_method

    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}')
    (tmp_path / filename).write_text(json.dumps({"quant_method": method, "bits": 3, "head_bits": 6,
                                               "codebook": "mul1", "tensor_storage": {"large": [1, 2, 3]}}))
    effective = metadata.read_config(tmp_path)
    assert quant_method(effective) == method.lower()
    block = effective["quantization_config"]
    assert block["bits"] == 3
    assert ("tensor_storage" in block) == (method.lower() != "exl3")


@pytest.mark.parametrize("location", ["quantization", "quantization_config", "text_config"])
@pytest.mark.parametrize("method", ["exl3", "unknown"])
def test_declared_main_format_does_not_open_redundant_inventory_sidecars(tmp_path, monkeypatch, location, method):
    block = {"quant_method": method, "bits": 3, "codebook": "mul1"}
    raw = {"text_config": {"quantization_config": block}} if location == "text_config" else {location: block}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    # A sparse oversized file makes accidental redundant reads fail without
    # materializing the real checkpoint's ~96-MB tensor inventory in this test.
    with (tmp_path / "quantization_config.json").open("wb") as stream:
        stream.truncate(metadata.MAX_HEADER_BYTES + 1)
    original = metadata.read_json
    read = []

    def checked(directory, filename, **options):
        read.append(filename)
        return original(directory, filename, **options)

    monkeypatch.setattr(metadata, "read_json", checked)
    assert metadata.read_config(tmp_path)["quantization_config"] == block
    assert read == ["config.json", "hf_quant_config.json"]


def test_eligible_sidecars_reconcile_representation_and_refuse_conflicts(tmp_path):
    (tmp_path / "config.json").write_text('{}')
    first = {"quant_method": "exl3", "bits": 3, "tensor_storage": {"first": "inventory"}}
    second = {"quant_method": "EXL3", "head_bits": 6, "tensor_storage": {"second": "inventory"}}
    (tmp_path / "quantization_config.json").write_text(json.dumps(first))
    path = tmp_path / "quant_config.json"
    path.write_text(json.dumps(second))
    assert metadata.read_config(tmp_path)["quantization_config"] == {"quant_method": "exl3", "bits": 3, "head_bits": 6}
    for incompatible in ({"quant_method": "gptq"}, {"quant_method": "exl3", "bits": 4}):
        path.write_text(json.dumps(incompatible))
        with pytest.raises(ValueError, match="conflict"):
            metadata.read_config(tmp_path)


def test_sidecar_without_method_does_not_infer_a_format(tmp_path):
    (tmp_path / "config.json").write_text('{}')
    (tmp_path / "quantization_config.json").write_text('{"bits":4}')
    assert "quantization_config" not in metadata.read_config(tmp_path)


@pytest.mark.parametrize("raw", [{}, {"quantization": None}, {"quantization_config": {}},
                                  {"text_config": {"quantization_config": {}}}])
def test_empty_main_blocks_preserve_sidecar_format_fallback(tmp_path, raw):
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "quant_config.json").write_text('{"quant_method":"exl3","codebook":"mul1"}')
    assert metadata.read_config(tmp_path)["quantization_config"] == {"quant_method": "exl3", "codebook": "mul1"}


@pytest.mark.parametrize("raw", [{"quantization": False}, {"quantization_config": []},
                                  {"text_config": []}, {"text_config": {"quantization": "exl3"}}])
def test_malformed_main_block_fails_before_secondary_io(tmp_path, monkeypatch, raw):
    def read(directory, filename, **options):
        assert filename == "config.json", "secondary metadata was read before refusing a malformed declaration"
        return raw

    monkeypatch.setattr(metadata, "read_json", read)
    with pytest.raises(ValueError, match="object"):
        metadata.read_config(tmp_path)


@pytest.mark.parametrize("method", ["", "  ", False, 7, []])
def test_invalid_explicit_sidecar_method_is_not_absence(tmp_path, method):
    (tmp_path / "config.json").write_text('{}')
    (tmp_path / "quant_config.json").write_text(json.dumps({"quant_method": method}))
    with pytest.raises(ValueError, match="quant_method"):
        metadata.read_config(tmp_path)


def test_hub_full_pull_uses_preflight_revision(tmp_path, monkeypatch):
    from tensorfold import hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{}')
    (snapshot / "model.safetensors").write_bytes(b'weights')
    calls = []
    monkeypatch.setattr(hub, "cached", lambda *a, **kw: None)
    monkeypatch.setattr(hub, "pull", lambda *a, **kw: calls.append(kw) or snapshot)
    assert hub.resolve("owner/model", revision="a" * 40) == snapshot
    assert calls == [{"cache_dir": None, "revision": "a" * 40}]


@pytest.mark.parametrize("flags, reason", [(["--eos-token-id", "8"], "vocabulary"),
                                          (["--chat-template-name", "missing"], "selected chat template")])
def test_invalid_explicit_model_binding_stops_before_weights(tmp_path, monkeypatch, flags, reason):
    from tensorfold import cli, families, hub

    (tmp_path / "config.json").write_text('{"vocab_size":8}')
    family = SimpleNamespace(title="fixture", model_type="fixture",
                             package=SimpleNamespace(cuda_engine=lambda *a, **k: pytest.fail("engine acquired")))
    monkeypatch.setattr(families, "detect", lambda directory: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weights resolved"))
    args = build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-update-check", *flags])
    with pytest.raises(ValueError, match=reason):
        cli.cmd_serve(args)
