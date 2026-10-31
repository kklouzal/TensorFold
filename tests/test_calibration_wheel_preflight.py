"""Cheap prerequisite failures must precede backend imports and source writes."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "deploy/gb10/calibration/wheel_preflight.py"
spec = importlib.util.spec_from_file_location("wheel_preflight_control", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    executable = tmp_path / "git"
    executable.write_text("# fixture; never executed")
    executable.chmod(0o755)
    monkeypatch.setattr(module, "verify_git_role", lambda: None)
    monkeypatch.setattr(module.shutil, "which", lambda name: str(executable))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="git version 2.43.0\n"))
    return source, tmp_path / "output", executable


def test_missing_git_precedes_any_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Git addon"):
        module.preflight(tmp_path / "absent", tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_valid_preflight_records_executable_version_without_writes(tmp_path, monkeypatch):
    source, output, executable = inputs(tmp_path, monkeypatch)
    assert module.preflight(source, output) == (source, output, executable, "git version 2.43.0")
    assert not output.exists()


@pytest.mark.parametrize("failure", ["source_permission", "missing_metadata", "existing_output", "bad_version"])
def test_prerequisite_failures_do_not_create_build_output(tmp_path, monkeypatch, failure):
    source, output, _ = inputs(tmp_path, monkeypatch)
    if failure == "source_permission":
        original = module.os.access
        monkeypatch.setattr(module.os, "access", lambda path, mode: False if Path(path) == source else original(path, mode))
    elif failure == "missing_metadata":
        (source / ".git").rmdir()
    elif failure == "existing_output":
        output.mkdir()
    else:
        monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="not Git"))
    with pytest.raises(RuntimeError):
        module.preflight(source, output)
    assert not output.exists() or not list(output.iterdir())


def test_real_stdlib_cli_missing_git_does_not_import_backend(tmp_path):
    script = f'''import os,runpy,sys
os.environ['PATH']=''
sys.argv=['wheel_preflight.py','--source',{str(tmp_path / 'absent')!r},'--output',{str(tmp_path / 'new')!r}]
try:
    runpy.run_path({str(SOURCE)!r},run_name='__main__')
except RuntimeError as error:
    assert 'Git addon' in str(error)
else:
    raise AssertionError('missing Git was accepted')
assert 'setuptools.build_meta' not in sys.modules
print('real CLI refused missing Git before backend import')
'''
    result = subprocess.run([sys.executable, "-c", script], check=True, text=True, capture_output=True)
    assert "before backend import" in result.stdout
