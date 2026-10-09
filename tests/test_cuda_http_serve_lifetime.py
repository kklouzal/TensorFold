"""CUDA serving cleanup keeps primary identity without importing a model SDK."""

import signal
from types import SimpleNamespace
from unittest.mock import patch

from tensorfold.cuda.http import serve


class Opaque(BaseException):
    def __str__(self):
        raise AssertionError("foreign failure formatted")

    def add_note(self, note):
        raise AssertionError("foreign failure note called")


def serving(primary=None, cleanup=None):
    calls = []

    def run():
        calls.append("serve")
        if primary is not None:
            raise primary

    def close():
        calls.append("close")
        if cleanup is not None:
            raise cleanup

    server = SimpleNamespace(serve_forever=run, server_close=close)
    with patch.object(signal, "signal"):
        try:
            serve(None, "127.0.0.1", 0, server=server)
        except BaseException as error:
            return calls, error
    return calls, None


def test_normal_serving_and_termination_drain_the_supplied_owner():
    for primary in (None, KeyboardInterrupt()):
        calls, error = serving(primary)
        assert calls == ["serve", "close"] and error is None


def test_primary_identity_and_cleanup_context_survive_cuda_serve():
    primary, cleanup = Opaque(), Opaque()
    calls, error = serving(primary, cleanup)
    assert calls == ["serve", "close"]
    assert error is primary and error.__cause__ is cleanup


def test_cleanup_failure_after_normal_serving_or_termination_is_reported():
    for primary in (None, KeyboardInterrupt()):
        cleanup = Opaque()
        calls, error = serving(primary, cleanup)
        assert calls == ["serve", "close"] and error is cleanup


def test_identical_primary_and_cleanup_failure_do_not_self_chain():
    primary = Opaque()
    calls, error = serving(primary, primary)
    assert calls == ["serve", "close"] and error is primary
    assert primary.__cause__ is None
