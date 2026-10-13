"""Benchmark A/B switches: unset means defaults; set, a changed file is re-read at the next refresh."""

import importlib
import json
import os

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
