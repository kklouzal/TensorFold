"""Optional MTP admission preserves plain/DFlash load behavior without importing numerical SDKs."""

from pathlib import Path
from types import ModuleType
import sys

import pytest

from tensorfold.families import qwen3_5, qwen3_5_moe


def loaders(monkeypatch):
    loaded, made = [], []
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path, **options: loaded.append((path, options)) or ("model", "tokenizer"))
    monkeypatch.setattr(qwen3_5, "lane_family", lambda model, **options: made.append((model, options)) or "family")
    return loaded, made


@pytest.mark.parametrize("cap,drafter", [(None, ""), (0, ""), (None, "selected-dflash"), (4, "selected-dflash")])
def test_plain_and_external_dflash_keep_the_original_wrapper_without_mtp_import(monkeypatch, cap, drafter):
    loaded, made = loaders(monkeypatch)
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5_moe.family", None)
    monkeypatch.setattr(qwen3_5_moe, "mtp_file", lambda path: pytest.fail("baseline mode selected optional head"))
    result = qwen3_5_moe.load(Path("checkpoint"), drafter=drafter, mtp_drafts=cap, trust_model_code=True)
    assert result == ("family", "tokenizer")
    assert loaded == [(Path("checkpoint"), {"trust_model_code": True})]
    assert made == [("model", {"lanes": False, "drafter": drafter, "drafter_bits": 4,
                               "title": qwen3_5_moe.TITLE, "use": qwen3_5_moe.MODELS[1]})]


@pytest.mark.parametrize("bound", [None, "", "bound-head.safetensors"])
def test_explicit_positive_cap_binds_one_head_choice_or_absence(monkeypatch, bound):
    loaded, made = loaders(monkeypatch)
    package = ModuleType("tensorfold.families.qwen3_5_moe.family")

    class Family:
        pass

    package.DRAFTS, package.Qwen36Family = 4, Family
    monkeypatch.setitem(sys.modules, package.__name__, package)
    selections = []
    monkeypatch.setattr(qwen3_5_moe, "mtp_file", lambda path: selections.append(path) or Path("selected-head.safetensors"))
    qwen3_5_moe.load(Path("checkpoint"), mtp_drafts=3, mtp_head=bound)
    factory = made[0][1]["make"]
    assert factory.func is Family and factory.keywords["drafts"] == 3
    assert factory.keywords["mtp_path"] == (Path("selected-head.safetensors") if bound is None else Path(bound) if bound else None)
    assert selections == ([Path("checkpoint")] if bound is None else [])
    assert loaded[0][1] == {"trust_model_code": False}


@pytest.mark.parametrize("cap", [-1, True, "3", 1.5])
def test_invalid_mtp_cap_fails_before_model_loading(monkeypatch, cap):
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda *args, **options: pytest.fail("invalid cap reached weights"))
    with pytest.raises(ValueError, match="nonnegative integer"):
        qwen3_5_moe.load(Path("checkpoint"), mtp_drafts=cap)
