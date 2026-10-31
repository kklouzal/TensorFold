"""Offline CLI planning is local, lazy, and cannot imply completed conversion."""
import json
from pathlib import Path

import pytest

from tensorfold import cli
from tensorfold import offline_quantization


def test_quantize_plan_prints_machine_readable_unexecuted_plan(monkeypatch, capsys):
    calls = []

    def plan(source, **options):
        calls.append((source, options))
        return {"status": "plan-only", "execution_implemented": False, "source": str(source)}

    monkeypatch.setattr(offline_quantization, "plan_nvfp4", plan)
    assert cli.main(["quantize", "plan", "./original-bf16"]) == 0
    assert calls == [(Path("./original-bf16"), {})]
    assert json.loads(capsys.readouterr().out) == {
        "status": "plan-only", "execution_implemented": False, "source": "original-bf16"}


def test_explicit_recipe_is_not_silently_replaced(monkeypatch, capsys):
    def reject(source, **options):
        assert options == {"recipe": "unknown-recipe"}
        raise ValueError("unsupported explicit quantization recipe")

    monkeypatch.setattr(offline_quantization, "plan_nvfp4", reject)
    assert cli.main(["quantize", "plan", "./original-bf16", "--recipe", "unknown-recipe"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "unsupported explicit quantization recipe" in output.err


def test_no_conversion_command_is_advertised_or_accepted(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["quantize", "run", "./original-bf16"])
    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_missing_local_source_is_reported_without_a_conversion(monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise FileNotFoundError("original local BF16 directory required")

    monkeypatch.setattr(offline_quantization, "plan_nvfp4", missing)
    assert cli.main(["quantize", "plan", "./missing-bf16"]) == 1
    assert "original local BF16 directory required" in capsys.readouterr().err
