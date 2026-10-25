"""The optional raw-indexer archive uses ordinary launch and family boundaries."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli, serve_options


def test_history_default_and_explicit_storage_modes():
    parser = cli.build_parser()
    assert parser.parse_args(["serve", "owner/model"]).indexer_history == "gpu"
    assert parser.parse_args(["serve", "owner/model", "--indexer-history", "ram"]).indexer_history == "ram"
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "owner/model", "--indexer-history", "automatic"])


@pytest.mark.parametrize("backend,model_type", [("mlx", "qwen4_exp"), ("cuda", "gemma4")])
def test_archive_rejects_other_providers_before_model_or_accelerator_work(backend, model_type):
    args = cli.build_parser().parse_args(["serve", "owner/model", "--indexer-history", "ram", "--no-drafts"])
    family = NS(model_type=model_type, title="test", package=NS())
    with pytest.raises(ValueError, match="GGUF.*CUDA/SM70"):
        serve_options.check(args, family, backend)


def test_archive_requires_local_gguf_at_serve_boundary(monkeypatch, tmp_path):
    from tensorfold.families.qwen4_exp import gguf

    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--indexer-history", "ram", "--no-drafts"])
    family = NS(model_type="qwen4_exp", title="test", package=NS())
    with pytest.raises(ValueError, match="local Flash Next GGUF"):
        serve_options.check(args, family, "cuda", tmp_path)


@pytest.mark.torch
@pytest.mark.parametrize("mode", ["gpu", "ram"])
def test_family_factory_forwards_only_the_opt_in_history_mode(monkeypatch, mode):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine

    monkeypatch.setattr(exl3, "is_exl3", lambda _: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: Path("target.gguf"))
    calls = []
    monkeypatch.setattr(engine, "FlashNextEngine", lambda directory, **kw: calls.append(kw) or NS(options=kw))
    family.cuda_engine("target", no_drafts=True, kv_dtype="fp16", indexer_history=mode)
    assert len(calls) == 1
    if mode == "gpu":
        assert "indexer_history" not in calls[0]
    else:
        assert calls[0]["indexer_history"] == "ram"


def test_family_rejects_ram_on_other_checkpoint_formats(monkeypatch):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf

    monkeypatch.setattr(exl3, "is_exl3", lambda _: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: None)
    with pytest.raises(ValueError, match="GGUF SM70"):
        family.cuda_engine("target", no_drafts=True, indexer_history="ram")
