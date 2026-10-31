"""Normal Flash Next --drafter recognition, without constructing CUDA resources."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def factory_function():
    import tensorfold.families.qwen4_exp as family
    return family.cuda_engine


@pytest.mark.parametrize(
    "architecture,expected",
    [(["DSparkDraftModel"], "dspark_model_dir"), (["StockMTP"], "mtp_model_dir"), (None, "mtp_model_dir")],
)
@pytest.mark.parametrize("disabled", [False, True])
def test_external_drafter_uses_its_declared_architecture_before_engine_allocation(
    tmp_path, monkeypatch, architecture, expected, disabled
):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine

    monkeypatch.setattr(exl3, "is_exl3", lambda path: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda path: Path("target.gguf"))
    calls = []

    def construct(directory, **kw):
        calls.append((directory, kw))
        return SimpleNamespace(options=kw)

    monkeypatch.setattr(engine, "FlashNextEngine", construct)
    drafter = tmp_path / "draft"
    drafter.mkdir()
    if architecture is not None:
        (drafter / "config.json").write_text(json.dumps({"architectures": architecture}))
    result = factory_function()("target", drafter=str(drafter), mtp_drafts=3, no_drafts=disabled, kv_dtype="fp16")
    assert len(calls) == 1 and result.options[expected] == drafter
    assert ("mtp_model_dir" if expected == "dspark_model_dir" else "dspark_model_dir") not in result.options
    assert result.options["depth"] == (0 if disabled else 3)
