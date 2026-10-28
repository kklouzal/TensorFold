"""Startup capture ownership, current-process authority and boundary controls."""
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import threading
import time

import pytest

from tensorfold.families.qwen4_exp.cuda import startup_content as startup
from tensorfold.families.qwen4_exp.cuda import execution_identity as identity
from tensorfold.engine import model_identity


def plan():
    return startup.StartupContentPlan({"target/unit":Path("/tmp/qualification-operand")})


def test_native_pthread_stack_queries_use_real_typed_attributes_without_global_changes():
    from tensorfold._fd_owner import _thread_stack_layout, _current_thread_stack_layout
    before = threading.stack_size()
    stack, guard = _thread_stack_layout(before)
    assert 0 < stack <= startup.STACK_LIMIT and guard >= 0
    assert _thread_stack_layout(1 << 20)[0] == 1 << 20
    actual = []
    thread = threading.Thread(target=lambda: actual.append(_current_thread_stack_layout()))
    thread.start()
    thread.join()
    assert actual and 0 < actual[0][0] <= stack and actual[0][1] <= guard
    assert threading.stack_size() == before
    for bad in (True, -1, "bytes"):
        with pytest.raises((TypeError, OverflowError)):
            _thread_stack_layout(bad)


@pytest.mark.parametrize("profile", ["source-owner-pieces-trial", "source-owner-target-greedy-trial", "source-owner-target-guard-trial", "source-owner-target-prelude-trial", "source-owner-frame-descriptors-trial"])
def test_default_is_serial_and_enabled_mode_refuses_wrong_scope(tmp_path, monkeypatch, profile):
    monkeypatch.delenv(startup.ENV, raising=False)
    assert startup.prepare(object(), None) is None
    monkeypatch.setenv(startup.ENV, "0")
    with pytest.raises(ValueError, match="accepts only1"):
        startup.prepare(object(), None)
    monkeypatch.setenv(startup.ENV, "1")
    with pytest.raises(ValueError, match="explicit private"):
        startup.prepare(NS(_profile_build=None, tp=1), {})
    engine=NS(_profile_build=None,tp=8,method_geometry=NS(method="mtp"),
        media_profile=None,sm70_profile=profile,
        _execution_preload={"capture_files":{"target/model":tmp_path/"model"}})
    prepared=startup.prepare(engine,{})
    assert prepared.host_bytes == startup.STACK_LIMIT + prepared.guard_bytes + startup.HASH_WORKING_BYTES


def test_worker_hashes_complete_expected_bytes_and_reuses_only_issued_authority(tmp_path):
    path=tmp_path/"model.gguf"
    raw=b"complete immutable payload"*1000
    path.write_bytes(raw)
    files={"target/model.gguf":path}
    before=identity._opened_stats(files)
    expected={"target/model.gguf":{"size":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}}
    capture=startup.StartupContentCapture(plan(),
        lambda cancelled: identity._capture(files,before,expected,cancelled=cancelled)[0])
    capture.start()
    authority=capture.result()
    assert authority.bytes_read == len(raw) and authority.hashes_reused == 0
    timing=capture.record()
    assert timing["worker_operation_retired"] and timing["registration_wait_ns"] >= 0
    assert timing["started_monotonic_ns"] <= timing["finished_monotonic_ns"]
    authorities,contents,read,reused=identity._capture_reusing(files,before,expected,(authority,))
    assert authorities == (authority,) and contents == expected and read == 0 and reused == 1
    capture.close()
    assert capture.closed and not capture.thread.is_alive()
    with pytest.raises(RuntimeError,match="retired"):
        capture.result()
    path.write_bytes(b"X"+raw[1:])
    with pytest.raises(ValueError,match="changed"):
        authority.verify_unchanged()


def test_worker_error_is_observed_and_close_joins_without_duplicate_failure():
    error=ValueError("original hash refusal")
    def failed(_cancelled):
        raise error
    capture=startup.StartupContentCapture(plan(),failed)
    capture.start()
    with pytest.raises(ValueError) as caught:
        capture.result()
    assert caught.value is error
    capture.close()
    assert capture.closed and not capture.thread.is_alive()


def test_early_close_requests_cancellation_and_waits_for_actual_retirement():
    accepted=threading.Event()
    def operation(cancelled):
        accepted.set()
        while not cancelled():
            time.sleep(.001)
        raise CancelledError("owned cancellation")
    capture=startup.StartupContentCapture(plan(),operation)
    capture.start()
    assert accepted.wait(3)
    capture.close()
    assert capture.closed and capture.cancel_observed and not capture.thread.is_alive()


def test_unobserved_real_failure_remains_cleanup_failure():
    error=OSError("read failed")
    capture=startup.StartupContentCapture(plan(),lambda _cancelled: (_ for _ in ()).throw(error))
    capture.start()
    capture.thread.join()
    with pytest.raises(OSError) as caught:
        capture.close()
    assert caught.value is error and capture.closed
    capture.close()


def test_model_identity_cancel_between_reads_preserves_descriptor_cleanup(tmp_path, monkeypatch):
    path=tmp_path/"model.gguf"
    path.write_bytes(b"x"*(model_identity._READ_BYTES+1))
    real=model_identity.os.read
    calls=[]
    def read(fd,size):
        calls.append(size)
        return real(fd,size)
    monkeypatch.setattr(model_identity.os,"read",read)
    with pytest.raises(CancelledError):
        model_identity.capture_model_identity({"model":path},runtime_identity={},
            authorize=lambda _n,p:p.resolve(),max_file_bytes=path.stat().st_size,
            max_total_bytes=path.stat().st_size,cancelled=lambda:bool(calls))
    assert calls == [model_identity._READ_BYTES]
    # The owned descriptor has closed: replacing the input is valid and a
    # subsequent independent ordinary capture consumes the complete new bytes.
    path.write_bytes(b"new")
    authority=model_identity.capture_model_identity({"model":path},runtime_identity={},
        authorize=lambda _n,p:p.resolve(),max_file_bytes=3,max_total_bytes=3)
    assert authority.bytes_read == 3


def test_actual_engine_retirement_joins_capture_before_withdrawing_its_owner():
    from test_flashnext_engine_lifetime import Engine
    accepted=threading.Event()
    def operation(cancelled):
        accepted.set()
        while not cancelled():
            time.sleep(.001)
        raise CancelledError("owned cancellation")
    capture=startup.StartupContentCapture(plan(),operation)
    owner=NS(_startup_content_capture=capture)
    capture.start()
    assert accepted.wait(3)
    Engine._stop_startup_content(owner)
    assert owner._startup_content_capture is None and capture.closed and not capture.thread.is_alive()
    Engine._stop_startup_content(owner)


def test_actual_plan_checks_preload_identity_before_starting_its_sole_worker(tmp_path, monkeypatch):
    path=tmp_path/"model.gguf"
    path.write_bytes(b"immutable")
    files={"target/model.gguf":path}
    before=identity._opened_stats(files)
    expected={"target/model.gguf":{"size":9,"sha256":hashlib.sha256(b"immutable").hexdigest()}}
    engine=NS(_execution_preload={"artifacts":before,"capture_files":files})
    monkeypatch.setattr(identity,"artifact_files",lambda _engine:pytest.fail("GGUF metadata reparsed after admission"))
    prepared=startup.StartupContentPlan(files)
    capture=prepared.start(engine,{"artifacts":expected})
    assert capture.thread.ident is None and not capture.start_attempted
    capture.start()
    assert capture.result().bytes_read == 9
    capture.close()
    path.write_bytes(b"mutated!!")
    with pytest.raises(ValueError,match="changed"):
        prepared.start(engine,{"artifacts":expected})


def test_capture_never_starts_after_retirement_or_failed_creation(monkeypatch):
    capture=startup.StartupContentCapture(plan(),lambda _cancelled:pytest.fail("retired owner started"))
    capture.close()
    with pytest.raises(RuntimeError,match="one-shot"):
        capture.start()
    capture=startup.StartupContentCapture(plan(),lambda _cancelled:pytest.fail("failed owner started"))
    error=RuntimeError("can't start new thread")
    monkeypatch.setattr(capture.thread,"start",lambda:(_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError) as caught:
        capture.start()
    assert caught.value is error
    with pytest.raises(RuntimeError,match="before it is started"):
        capture.close()
    assert not capture.closed


def test_expected_sha_failure_is_not_replaced_by_a_stat_match(tmp_path):
    path=tmp_path/"model.gguf"
    path.write_bytes(b"actual")
    files={"target/model.gguf":path}
    before=identity._opened_stats(files)
    expected={"target/model.gguf":{"size":6,"sha256":"0"*64}}
    capture=startup.StartupContentCapture(plan(),
        lambda cancelled:identity._capture(files,before,expected,cancelled=cancelled)[0])
    capture.start()
    with pytest.raises(ValueError,match="declared SHA256"):
        capture.result()
    capture.close()


def test_start_exception_after_native_acceptance_still_joins_owned_work(monkeypatch):
    accepted=threading.Event()
    def operation(cancelled):
        accepted.set()
        while not cancelled():
            time.sleep(.001)
        raise CancelledError("owned cancellation")
    capture=startup.StartupContentCapture(plan(),operation)
    actual=capture.thread.start
    error=RuntimeError("interrupted after native acceptance")
    def interrupted():
        actual()
        raise error
    monkeypatch.setattr(capture.thread,"start",interrupted)
    with pytest.raises(RuntimeError) as caught:
        capture.start()
    assert caught.value is error and accepted.wait(3)
    capture.close()
    assert capture.closed and not capture.thread.is_alive()


@pytest.mark.skipif(not hasattr(__import__("signal"), "setitimer"), reason="POSIX signal control")
def test_real_join_interrupt_cannot_publish_closed_before_actual_operation_cleanup():
    import signal
    from tensorfold.cleanup import rollback
    accepted, release = threading.Event(), threading.Event()
    def operation(_cancelled):
        accepted.set()
        release.wait()
        return NS(bytes_read=1)
    capture=startup.StartupContentCapture(plan(),operation)
    capture.start()
    assert accepted.wait(3)
    primary=KeyboardInterrupt("actual join interrupted")
    current=[primary]
    def interrupted(_signum,_frame):
        raise current[0]
    old=signal.signal(signal.SIGALRM,interrupted)
    try:
        signal.setitimer(signal.ITIMER_REAL,.03)
        with pytest.raises(KeyboardInterrupt) as caught:
            capture.result()
        assert caught.value is primary
        assert not capture.operation_retired.is_set() and capture.finished_ns is None
        # Reproduce an interrupted cleanup as well: actual rollback must
        # retain the owner and original primary until the hash operation ends.
        current[0]=KeyboardInterrupt("actual cleanup interrupted")
        signal.setitimer(signal.ITIMER_REAL,.03)
        with pytest.raises(KeyboardInterrupt) as caught:
            rollback(capture,primary,capture.close)
        assert caught.value is primary
        assert any(owner is capture for owner in primary._tensorfold_retained_owners)
        assert not capture.closed and not capture.operation_retired.is_set()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)
        release.set()
        capture.close()
    assert capture.closed and capture.operation_retired.is_set() and capture.finished_ns is not None


@pytest.mark.parametrize("hash_failure", [False, True])
def test_failing_final_timing_still_publishes_retirement_and_preserves_hash_primary(monkeypatch, hash_failure):
    primary=ValueError("hash refusal")
    diagnostic=MemoryError("final timestamp allocation")
    def operation(_cancelled):
        if hash_failure:
            raise primary
        return NS(bytes_read=1)
    capture=startup.StartupContentCapture(plan(),operation)
    monkeypatch.setattr(startup.time,"monotonic_ns",lambda:(_ for _ in ()).throw(diagnostic))
    capture.start()
    with pytest.raises(ValueError if hash_failure else MemoryError) as caught:
        capture.result()
    assert caught.value is (primary if hash_failure else diagnostic)
    assert capture.operation_retired.is_set() and capture.finished_ns is None
    assert capture.finish_errors == [diagnostic]
    if hash_failure:
        assert primary.__cause__ is diagnostic
    capture.close()
    assert capture.closed


@pytest.mark.parametrize("case", ["unicode", "control", "aggregate", "files"])
def test_unprovable_inventory_working_bounds_refuse_before_worker_allocation(tmp_path, monkeypatch, case):
    path=tmp_path/"model.gguf"
    path.write_bytes(b"data")
    if case == "unicode":
        files={"target/mödél":path}
    elif case == "control":
        files={"target/control\n":path}
    elif case == "aggregate":
        files={("target/"+str(index)+"x"*2000):path for index in range(40)}
    else:
        files={str(index):path for index in range(65)}
    monkeypatch.setattr(startup,"StartupContentCapture",lambda *_args:pytest.fail("worker allocated before inventory refusal"))
    with pytest.raises(ValueError,match="startup"):
        startup.StartupContentPlan(files)


@pytest.mark.parametrize("phase", ["result", "close"])
def test_outcome_argument_preparation_failure_keeps_original_hash_primary(monkeypatch, phase):
    import sys
    import inspect
    primary=ValueError("hash failure before outcome preparation")
    allocation=MemoryError("outcome list allocation")
    capture=startup.StartupContentCapture(plan(),lambda _cancelled:(_ for _ in ()).throw(primary))
    capture.start()
    capture.operation_retired.wait()
    code=capture._raise_outcome.__func__.__code__
    source,first=inspect.getsourcelines(capture._raise_outcome)
    allocation_line=first+next(index for index,line in enumerate(source) if "errors = [error" in line)
    fired=[]
    def trace(frame,event,_arg):
        if frame.f_code is code and event == "line" and frame.f_lineno == allocation_line:
            fired.append(True)
            raise allocation
        return trace
    old=sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(ValueError) as caught:
            getattr(capture,phase)()
        assert fired == [True] and caught.value is primary and primary.__cause__ is allocation
    finally:
        sys.settrace(old)
        capture.close()
    assert capture.closed and capture.operation_retired.is_set()
