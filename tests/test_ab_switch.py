"""Benchmark A/B switches: unset means defaults; set, a changed file is re-read at the next refresh."""

import importlib
import json
import os

import pytest

from tensorfold.cuda import ab_switch


def test_unset_returns_defaults(monkeypatch):
    monkeypatch.delenv(ab_switch.ENV, raising=False)
    mod = importlib.reload(ab_switch)
    mod.refresh()
    assert mod.get("select", "old") == "old" and mod.snapshot() == {}


def test_changed_file_is_reread(monkeypatch, tmp_path):
    path = tmp_path / "ab.json"
    path.write_text(json.dumps({"select": "T"}))
    monkeypatch.setenv(ab_switch.ENV, str(path))
    mod = importlib.reload(ab_switch)
    mod.refresh()
    assert mod.get("select", "old") == "T"
    path.write_text(json.dumps({"select": "old", "graphs": True}))
    os.utime(path, ns=(1, 1))                      # a distinct modification time
    mod.refresh()
    assert mod.get("select", "x") == "old" and mod.get("graphs", False) is True
    path.unlink()
    mod.refresh()
    assert mod.get("select", "default") == "default"
    monkeypatch.delenv(ab_switch.ENV)
    importlib.reload(ab_switch)


def test_stage_switches_default_to_new_and_reject_other_values(monkeypatch, tmp_path):
    """``new``: a stage's switch is "new" (its behaviour, also when unset or absent) or "old" (the path it replaces);
    any other value is refused by name. ``active``: a control file is configured (an A/B run), even before the
    first refresh and after the file goes."""

    monkeypatch.delenv(ab_switch.ENV, raising=False)
    mod = importlib.reload(ab_switch)
    mod.refresh()
    assert not mod.active() and mod.new("b1") is True
    path = tmp_path / "ab.json"
    path.write_text(json.dumps({"b1": "old", "b2": "new", "b3": "NEW", "graphs": True, "pdl": None}))
    monkeypatch.setenv(ab_switch.ENV, str(path))
    mod = importlib.reload(ab_switch)
    assert mod.active() and mod.new("b1") is True       # not read yet: defaults
    mod.refresh()
    assert mod.new("b1") is False and mod.new("b2") is True and mod.new("absent") is True
    for key in ("b3", "graphs", "pdl"):
        with pytest.raises(ValueError, match=f'"{key}" must be "new" or "old"'):
            mod.new(key)
    path.unlink()
    mod.refresh()
    assert mod.active() and mod.new("b3") is True
    monkeypatch.delenv(ab_switch.ENV)
    importlib.reload(ab_switch)
