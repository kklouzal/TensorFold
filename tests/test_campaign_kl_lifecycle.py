"""CLI completion and temporary-hook ownership for the finite teacher-forced adapter; no model or GPU."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location("campaign_kl_lifecycle", ROOT / "tools" / "campaign_kl.py")
    kl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kl)
finally:
    sys.path.pop(0)


@pytest.fixture
def boundary(monkeypatch):
    package, cuda, server, cli = (ModuleType(name) for name in
                                  ("tensorfold", "tensorfold.cuda", "tensorfold.cuda.server", "tensorfold.cli"))
    package.__path__, cuda.__path__ = [], []
    package.cuda, package.cli, cuda.server = cuda, cli, server
    original = object()
    server.serve = original
    for module in (package, cuda, server, cli):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(server=server, cli=cli, original=original, app=object())


def test_success_normalizes_private_argv_and_restores_hook(boundary):
    argv = ["serve", "/model", "--host", "0.0.0.0", "--port", "80"]
    seen = []
    def main(args):
        seen.append(args)
        boundary.server.serve(boundary.app, "127.0.0.1", 8890)
        return 0
    boundary.cli.main = main
    work = []
    kl.with_app(argv, 8890, work.append)
    assert work == [boundary.app]
    assert seen == [["serve", "/model", "--host", "127.0.0.1", "--port", "8890"]]
    assert argv[-3:] == ["0.0.0.0", "--port", "80"]
    assert boundary.server.serve is boundary.original


@pytest.mark.parametrize("status", [1, 4, 130])
def test_nonzero_cli_after_completed_dump_is_failure(boundary, tmp_path, status):
    partial = tmp_path / "completed-dump"
    def main(args):
        boundary.server.serve(boundary.app, "127.0.0.1", 8890)
        return status
    boundary.cli.main = main
    with pytest.raises(SystemExit) as error:
        kl.with_app(["serve", "/model"], 8890, lambda app: partial.write_bytes(b"complete scoring"))
    assert error.value.code == status
    assert partial.read_bytes() == b"complete scoring"
    assert boundary.server.serve is boundary.original


@pytest.mark.parametrize("status", [None, False, 0.0, "0"])
def test_non_integer_cli_result_is_not_success(boundary, status):
    def main(args):
        boundary.server.serve(boundary.app, "127.0.0.1", 8890)
        return status
    boundary.cli.main = main
    with pytest.raises(RuntimeError, match="integer status"):
        kl.with_app(["serve", "/model"], 8890, lambda app: None)
    assert boundary.server.serve is boundary.original


@pytest.mark.parametrize("where", ["cli", "work"])
def test_original_exception_and_cause_survive_restoration(boundary, where):
    cause = OSError("transport")
    primary = RuntimeError("workload")
    primary.__cause__ = cause
    def work(app):
        if where == "work":
            raise primary
    def main(args):
        if where == "cli":
            raise primary
        boundary.server.serve(boundary.app, "127.0.0.1", 8890)
        return 0
    boundary.cli.main = main
    with pytest.raises(RuntimeError) as error:
        kl.with_app(["serve", "/model"], 8890, work)
    assert error.value is primary and error.value.__cause__ is cause
    assert boundary.server.serve is boundary.original


def test_missing_hook_and_parser_exit_restore_previous_binding(boundary):
    boundary.cli.main = lambda argv: 0
    with pytest.raises(SystemExit, match="hook never ran"):
        kl.with_app(["serve", "/model"], 8890, lambda app: None)
    assert boundary.server.serve is boundary.original
    def parse_error(argv):
        raise SystemExit(2)
    boundary.cli.main = parse_error
    with pytest.raises(SystemExit) as error:
        kl.with_app(["serve", "/model"], 8890, lambda app: None)
    assert error.value.code == 2 and boundary.server.serve is boundary.original


def test_real_cli_main_cleanup_value_error_cannot_become_dump_success(boundary, capsys):
    # Execute the maintained CLI's actual main body with only model construction/parser replaced by fixtures.
    path = ROOT / "src" / "tensorfold" / "cli.py"
    node = next(node for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    calls = []
    def handler(args):
        boundary.server.serve(boundary.app, "127.0.0.1", 8890)
        raise ValueError("engine cleanup failed")
    namespace = {"sys": sys, "Path": Path, "COMMANDS": ("serve",),
                 "build_parser": lambda: SimpleNamespace(parse_args=lambda argv: SimpleNamespace(func=handler))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - owned CLI AST
    boundary.cli.main = namespace["main"]
    with pytest.raises(SystemExit) as error:
        kl.with_app(["serve", "/model"], 8890, calls.append)
    assert error.value.code == 1 and calls == [boundary.app]
    assert "engine cleanup failed" in capsys.readouterr().err
    assert boundary.server.serve is boundary.original
