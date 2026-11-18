"""Flash Next prompt-piece rows: `serve --prefill-rows`, then TENSORFOLD_PREFILL_ROWS, then the default
(1,024 on SM70, otherwise the engine's own plan); one validation for all three."""

import argparse

import pytest

from tensorfold.cuda.geometry import SM70_PREFILL_ROWS, indexed_prefill_rows, prefill_rows_value


def test_unset_or_empty_leaves_the_engine_its_plan_except_on_sm70(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    assert indexed_prefill_rows() is None
    assert indexed_prefill_rows((12, 1)) is None
    assert indexed_prefill_rows((7, 0)) == SM70_PREFILL_ROWS == 1024
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", " ")
    assert indexed_prefill_rows() is None
    assert indexed_prefill_rows((7, 0)) == 1024


@pytest.mark.parametrize("value, rows", [("256", 256), ("2048", 2048), (" 4096 ", 4096), ("16384", 16384)])
@pytest.mark.parametrize("capability", [None, (7, 0), (12, 1)])
def test_a_set_value_is_the_piece_rows_on_every_device(value, rows, capability, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", value)
    assert indexed_prefill_rows(capability) == rows


@pytest.mark.parametrize("value", ["255", "16385", "0", "-2048", "2k", "4096.0"])
def test_anything_else_refuses_at_startup(value, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", value)
    with pytest.raises(ValueError, match="TENSORFOLD_PREFILL_ROWS"):
        indexed_prefill_rows((7, 0))
    with pytest.raises(ValueError, match="--prefill-rows"):
        prefill_rows_value(value, "--prefill-rows")


def serve_args(*argv):
    from tensorfold.cli import build_parser

    return build_parser().parse_args(["serve", "some/model", *argv])


@pytest.mark.parametrize("value", ["255", "16385", "2k"])
def test_the_flag_refuses_what_the_environment_refuses(value, capsys):
    with pytest.raises(SystemExit):
        serve_args("--prefill-rows", value)
    assert "--prefill-rows: 256 to 16,384 rows" in capsys.readouterr().err


@pytest.mark.parametrize("environment, flag, rows", [
    (None, None, 1024), ("2048", None, 2048), (None, "4096", 4096), ("2048", "512", 512)])
def test_flag_then_environment_then_default(environment, flag, rows, monkeypatch):
    from tensorfold.cli import _apply_prefill_rows

    monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    if environment is not None:
        monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", environment)
    args = serve_args(*(["--prefill-rows", flag] if flag is not None else []))
    assert args.prefill_rows == (None if flag is None else int(flag))
    _apply_prefill_rows(args)
    assert indexed_prefill_rows((7, 0)) == rows
