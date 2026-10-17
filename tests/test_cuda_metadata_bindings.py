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


@pytest.mark.parametrize("family,relative,classname,options", [
    ("qwen3_5", "cuda.engine", "Qwen27Engine", {"tree_rows": "invalid"}),
    ("qwen3_5", "cuda.engine", "Qwen27Engine", {"keep": "invalid"}),
    ("qwen3_5_moe", "cuda.engine", "Qwen36Engine", {"depth": "invalid"}),
    ("qwen3_5_moe", "cuda.engine", "Qwen36Engine", {"confidence": "invalid"}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"depth": -1}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"confidence": 2}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"tp": 3}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"rank": 2}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"streams": 2, "tp": 2}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"tp": 2}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"vram_experts_pin": "on"}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"vram_experts": 0}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"vram_experts": True}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"vram_experts": "invalid"}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"vision": True}),
    ("qwen4_exp", "cuda.engine", "FlashNextEngine", {"kv_dtype": "unknown"}),
    ("glm5_next", "cuda.engine", "GlmEngine", {"policy": "unknown"}),
    ("glm5_next", "cuda.engine", "GlmEngine", {"world": 3}),
    ("glm5_next", "cuda.engine", "GlmEngine", {"rank": 2}),
    ("nemotron_h", "cuda.app", "NemotronEngine", {"drafts": 9}),
    ("nemotron_h", "cuda.app", "NemotronEngine", {"confidence": 2}),
    ("nemotron_h", "cuda.app", "NemotronEngine", {"rank": 2}),
    ("nemotron_h", "cuda.app", "NemotronEngine", {"tp": 2}),
])
def test_cheap_invalid_options_refuse_before_template_or_device_work(
        tmp_path, monkeypatch, family, relative, classname, options):
    import tensorfold.cuda.chat_template as templates

    def unexpected(*args, **kwargs):
        pytest.fail("invalid launch options reached template metadata or CUDA device setup")
    monkeypatch.setattr(templates, "prepare_chat_template", unexpected)
    monkeypatch.setattr(torch.cuda, "set_device", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    backend = importlib.import_module(f"tensorfold.families.{family}.{relative}")
    constructor = getattr(backend, classname)
    missing = tmp_path / "unavailable-checkpoint"
    args = (missing, None) if family == "qwen3_5" else (missing,)
    keywords = {"rank": 0, "master": "127.0.0.1", "port": 29551} if family == "glm5_next" else {}
    keywords.update(options)
    with pytest.raises(ValueError):
        constructor(*args, **keywords)
