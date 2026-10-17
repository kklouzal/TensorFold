"""Actual family factories and early constructor refusal, without CUDA execution."""
import importlib
import json
import os

import pytest
import torch
from jinja2 import TemplateSyntaxError

from tensorfold.cuda.chat_template import prepare_chat_template


FAMILIES = [
    ("qwen3_5", "cuda.engine", "Qwen27Engine", {}),
    ("qwen3_5_moe", "cuda.engine", "Qwen36Engine", {}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {}),
    ("glm5_next", "cuda.engine", "GlmEngine", {"tp": 2, "master": "127.0.0.1"}),
    ("nemotron_h", "cuda.app", "NemotronEngine", {}),
]


@pytest.mark.parametrize("family,relative,classname,options", FAMILIES)
@pytest.mark.parametrize("metadata", [{"eos_token_ids": [0, 3]}, {"chat_template_name": "tool_use"},
                                     {"chat_template_file": "operator.jinja"}])
def test_factory_forwards_each_metadata_control_without_silently_swallowing_it(
        tmp_path, monkeypatch, family, relative, classname, options, metadata):
    package = importlib.import_module(f"tensorfold.families.{family}")
    backend = importlib.import_module(f"tensorfold.families.{family}.{relative}")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": family}))
    monkeypatch.setattr(backend, classname, lambda *args, **kwargs: kwargs)
    result = package.cuda_engine(tmp_path, no_drafts=True, **options, **metadata)
    for key, value in metadata.items():
        assert result[key] == value


@pytest.mark.parametrize("family,relative,classname,options", FAMILIES)
def test_invalid_selected_template_is_compiled_and_refused_before_device_setup(
        tmp_path, monkeypatch, family, relative, classname, options):
    backend = importlib.import_module(f"tensorfold.families.{family}.{relative}")
    bad = tmp_path / "bad.jinja"
    bad.write_text("{% if")
    def unexpected(*args, **kwargs):
        pytest.fail("invalid template reached CUDA device setup")
    monkeypatch.setattr(torch.cuda, "set_device", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    constructor = getattr(backend, classname)
    args = (tmp_path, None) if family == "qwen3_5" else (tmp_path,)
    keywords = {"rank": 0, "master": "127.0.0.1", "port": 29551} if family == "glm5_next" else {}
    with pytest.raises(TemplateSyntaxError):
        constructor(*args, chat_template_file=bad, **keywords)


@pytest.mark.parametrize("family,relative,classname,options", FAMILIES)
@pytest.mark.parametrize("kind", ["directory", "dangling", "fifo", "outside"])
def test_present_invalid_tokenizer_cannot_bypass_early_template_admission(
        tmp_path, monkeypatch, family, relative, classname, options, kind):
    path = tmp_path / "tokenizer.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "dangling":
        path.symlink_to(tmp_path / "missing.json")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("POSIX checkpoint FIFO boundary")
        os.mkfifo(path)
    else:
        outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
        outside.write_text("{}")
        path.symlink_to(outside)
    backend = importlib.import_module(f"tensorfold.families.{family}.{relative}")
    def unexpected(*args, **kwargs):
        pytest.fail("invalid tokenizer reached CUDA device setup")
    monkeypatch.setattr(torch.cuda, "set_device", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    constructor = getattr(backend, classname)
    args = (tmp_path, None) if family == "qwen3_5" else (tmp_path,)
    keywords = {"rank": 0, "master": "127.0.0.1", "port": 29551} if family == "glm5_next" else {}
    with pytest.raises((ValueError, FileNotFoundError)):
        constructor(*args, **keywords)


def test_true_absence_still_allows_bare_model_api_without_template(tmp_path):
    assert prepare_chat_template(tmp_path) is None


def test_absent_tokenizer_does_not_hide_invalid_model_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_chat_template(tmp_path / "missing-root")
