"""CPU controls for GPU-fixture diagnostics/retirement, not kernel numerics."""
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("fault", [None, "render", "marker", "head-close", "full-close", "cancel"])
def test_owner_fixture_reports_primary_before_independent_retirement_and_keeps_watchdog(monkeypatch, fault):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "cuda"))
    subject = importlib.import_module("test_sm70_owner_graphs")
    from tensorfold.cuda import comm

    primary, secondary = ValueError("head cohort initialization failed"), MemoryError("diagnostic/retirement fault")
    events, live = [], [False]

    def arm(seconds, *, repeat, file):
        assert seconds == 60 and repeat and file is subject.sys.stderr
        live[0] = True
        events.append("watchdog-armed")

    def cancel():
        assert events.index("full-close") > events.index("head-close")
        events.append("watchdog-cancel")
        live[0] = False
        if fault == "cancel":
            raise secondary

    class Communicator:
        def __init__(self, name):
            self.name, self.store = name, object()

        def open(self, *_args, **_kwargs):
            events.append(self.name + "-open")
            if self.name == "head":
                raise primary

        def close(self, *, abort):
            assert abort and live[0]
            events.append(self.name + "-close")
            if fault == self.name + "-close":
                raise secondary

    communicators = iter((Communicator("full"), Communicator("head")))
    monkeypatch.setattr(comm, "NCCL", lambda: next(communicators))
    monkeypatch.setattr(torch, "set_num_threads", lambda _threads: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(subject.faulthandler, "dump_traceback_later", arm)
    monkeypatch.setattr(subject.faulthandler, "cancel_dump_traceback_later", cancel)

    def phase(_rank, head4, name, *, rows=None):
        assert head4
        events.append(name)
        if fault == "marker" and name == "head-retirement-begin":
            raise secondary

    def render(error, *, file):
        assert error is primary and file is subject.sys.stderr and live[0]
        events.append("primary-render")
        if fault == "render":
            raise secondary

    monkeypatch.setattr(subject, "_phase", phase)
    monkeypatch.setattr(subject.traceback, "print_exception", render)
    with pytest.raises(ValueError) as caught:
        subject._worker(0, 29627, True)
    assert caught.value is primary
    assert events.index("primary-render") < events.index("head-close") < events.index("full-close")
    assert events[-1] == "watchdog-cancel" and not live[0]
    if fault is not None:
        assert primary.__cause__ is secondary
