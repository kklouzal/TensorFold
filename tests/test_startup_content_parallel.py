"""Whole-file parallel identity, bounded ownership and adverse failures."""
from concurrent.futures import CancelledError
from types import SimpleNamespace as NS
import hashlib
import signal
import threading
import time

import pytest

from tensorfold.cleanup import rollback
from tensorfold.families.qwen4_exp.cuda import startup_content as startup
from tensorfold.families.qwen4_exp.cuda import execution_identity as identity


def inputs(tmp_path):
    files={}
    for name,size in (("target/a",19000),("target/b",13000),("draft/c",7000),("metadata/config",32)):
        path=tmp_path/name.replace("/","-")
        path.write_bytes((name.encode()*(-(-size//len(name))))[:size])
        files[name]=path
    stats=identity._opened_stats(files)
    expected={name:{"size":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
              for name,path in files.items()}
    return files,stats,expected


def group_plan(files):
    return startup.StartupContentPlan(files,workers=3)


def test_parallel_full_file_authorities_equal_serial_content_without_rehash(tmp_path):
    files,before,expected=inputs(tmp_path)
    plan=group_plan(files)
    assert plan.host_bytes==3*(startup.STACK_LIMIT+plan.guard_bytes+startup.HASH_WORKING_BYTES)
    engine=NS(_execution_preload={"capture_files":files,"artifacts":before})
    capture=plan.start(engine,{"artifacts":expected})
    assert len(capture.captures)==3 and not any(c.start_attempted for c in capture.captures)
    capture.start()
    authorities=capture.authorities()
    serial,serial_contents=identity._capture(files,before,expected)
    retained,contents,read,reused=identity._capture_reusing(files,before,expected,authorities)
    assert contents==serial_contents==expected and read==0 and reused==len(files)
    assert len(retained)==3 and sum(a.bytes_read for a in authorities)==serial.bytes_read
    assert all(a.runtime_json==serial.runtime_json for a in authorities)
    record=capture.record()
    assert record["bytes_hashed"]==serial.bytes_read and record["workers_global"]==3
    assert all(row["hash_operation_started_monotonic_ns"]<=row["finished_monotonic_ns"] for row in record["workers"])
    assert record["worker_operation_retired"]
    capture.close()
    assert capture.closed and all(c.closed and c.operation_retired.is_set() for c in capture.captures)
    assert startup.partition_files(dict(reversed(list(files.items()))),before,3)==capture.partitions
    path=files["target/a"]
    raw=path.read_bytes()
    path.write_bytes(b"X"+raw[1:])
    with pytest.raises(ValueError):
        identity._capture_reusing(files,before,expected,authorities)


def test_three_owned_operations_really_overlap_without_executor(tmp_path):
    files,_before,_expected=inputs(tmp_path)
    barrier=threading.Barrier(3)
    def operation(_cancelled):
        barrier.wait(3)
        return NS(bytes_read=1)
    capture=startup.StartupContentGroup(group_plan(files),[operation]*3,[("a",),("b",),("c",)])
    capture.start()
    assert len(capture.authorities())==3
    record=capture.record()
    assert max(row["hash_operation_started_monotonic_ns"] for row in record["workers"]) <= min(
        row["finished_monotonic_ns"] for row in record["workers"])
    capture.close()
    assert capture.closed


def test_genuine_hash_failure_cancels_siblings_and_is_primary(tmp_path):
    files,_before,_expected=inputs(tmp_path)
    barrier=threading.Barrier(3)
    primary=ValueError("partition hash mismatch")
    def wait(cancelled):
        barrier.wait(3)
        while not cancelled():
            time.sleep(.001)
        raise CancelledError("sibling hash failed")
    def failed(_cancelled):
        barrier.wait(3)
        raise primary
    capture=startup.StartupContentGroup(group_plan(files),[wait,wait,failed],[("a",),("b",),("c",)])
    capture.start()
    with pytest.raises(ValueError) as caught:
        capture.authorities()
    assert caught.value is primary
    capture.close()
    assert capture.closed and all(c.operation_retired.is_set() for c in capture.captures)


def test_partial_start_after_native_acceptance_is_cancelled_and_joined(tmp_path,monkeypatch):
    files,_before,_expected=inputs(tmp_path)
    def wait(cancelled):
        while not cancelled():
            time.sleep(.001)
        raise CancelledError("startup aborted")
    capture=startup.StartupContentGroup(group_plan(files),[wait]*3,[("a",),("b",),("c",)])
    original=capture.captures[1].thread.start
    primary=RuntimeError("interruption after second native acceptance")
    def interrupted():
        original()
        raise primary
    monkeypatch.setattr(capture.captures[1].thread,"start",interrupted)
    with pytest.raises(RuntimeError) as caught:
        capture.start()
    assert caught.value is primary
    capture.close()
    assert capture.closed and all(c.closed for c in capture.captures)
    assert all(c.operation_retired.is_set() for c in capture.captures[:2])
    assert not capture.captures[2].start_attempted


@pytest.mark.skipif(not hasattr(signal,"setitimer"),reason="actual POSIX signal")
def test_real_signal_join_and_cleanup_interrupt_retain_all_file_owners(tmp_path):
    files,_before,_expected=inputs(tmp_path)
    entered=threading.Event()
    release=threading.Event()
    def blocked(_cancelled):
        entered.set()
        release.wait()
        return NS(bytes_read=1)
    capture=startup.StartupContentGroup(group_plan(files),[blocked]*3,[("a",),("b",),("c",)])
    capture.start()
    assert entered.wait(3)
    primary=KeyboardInterrupt("actual parallel result join interrupted")
    current=[primary]
    def interrupt(_signum,_frame):
        raise current[0]
    old=signal.signal(signal.SIGALRM,interrupt)
    try:
        # Every Group result error requests all cancellations, but these
        # deliberately blocked operations retain their file borrowers.
        signal.setitimer(signal.ITIMER_REAL,.03,.03)
        with pytest.raises(KeyboardInterrupt) as caught:
            capture.authorities()
        assert caught.value is primary and not capture.closed
        assert not any(c.operation_retired.is_set() for c in capture.captures)
        current[0]=KeyboardInterrupt("actual parallel cleanup interrupted")
        with pytest.raises(KeyboardInterrupt) as caught:
            rollback(capture,primary,capture.close)
        assert caught.value is primary
        assert any(owner is capture for owner in primary._tensorfold_retained_owners)
        assert not capture.closed
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)
        release.set()
        capture.close()
    assert capture.closed and all(c.operation_retired.is_set() for c in capture.captures)


def test_partial_start_before_acceptance_retains_uncertain_owner_and_original(tmp_path,monkeypatch):
    files,_before,_expected=inputs(tmp_path)
    def wait(cancelled):
        while not cancelled():
            time.sleep(.001)
        raise CancelledError("startup failed")
    capture=startup.StartupContentGroup(group_plan(files),[wait]*3,[("a",),("b",),("c",)])
    primary=RuntimeError("cannot create second thread")
    monkeypatch.setattr(capture.captures[1].thread,"start",lambda:(_ for _ in ()).throw(primary))
    with pytest.raises(RuntimeError) as caught:
        capture.start()
    assert caught.value is primary
    with pytest.raises(RuntimeError) as caught:
        rollback(capture,primary,capture.close)
    assert caught.value is primary and not capture.closed
    assert any(owner is capture for owner in primary._tensorfold_retained_owners)
    assert capture.captures[0].closed and capture.captures[0].operation_retired.is_set()
    assert not capture.captures[1].closed and capture.captures[2].closed


def test_worker_budget_and_zero_size_file_partition_rules(tmp_path,monkeypatch):
    files,before,_expected=inputs(tmp_path)
    assert group_plan(files).record()["workers_global"]==3
    for key in before:
        before[key]["stat"][2]=0
    groups=startup.partition_files(files,before,3)
    assert len(groups)==3 and all(group for group in groups)
    assert sorted(name for group in groups for name in group)==sorted(files)
    for workers in (0,2,4,True):
        with pytest.raises(ValueError,match="workers"):
            startup.StartupContentPlan(files,workers=workers)
    monkeypatch.delenv(startup.ENV,raising=False)
    monkeypatch.setenv(startup.WORKERS_ENV,"3")
    with pytest.raises(ValueError,match="requires enabled"):
        startup.prepare(object(),None)


def test_cancel_failure_survives_later_success_until_reported(tmp_path,monkeypatch):
    files,_before,_expected=inputs(tmp_path)
    capture=startup.StartupContentGroup(group_plan(files),[lambda _cancelled:NS(bytes_read=1)]*3,[("a",),("b",),("c",)])
    original=capture.captures[1].cancel.set
    primary=RuntimeError("first cancellation failed")
    calls=[]
    def once():
        calls.append(True)
        if len(calls)==1:
            raise primary
        original()
    monkeypatch.setattr(capture.captures[1].cancel,"set",once)
    capture._cancel_all()
    capture._cancel_all()
    assert capture.cancel_errors[1] is primary
    with pytest.raises(RuntimeError) as caught:
        capture.close()
    assert caught.value is primary and capture.closed


def test_real_wrong_partition_hash_never_publishes_partial_authorities(tmp_path):
    files,before,expected=inputs(tmp_path)
    expected["target/b"]["sha256"]="0"*64
    capture=group_plan(files).start(NS(_execution_preload={"capture_files":files,"artifacts":before}),{"artifacts":expected})
    capture.start()
    with pytest.raises(ValueError,match="declared SHA256"):
        capture.authorities()
    capture.close()
    assert capture.closed and all(c.operation_retired.is_set() for c in capture.captures)


def test_group_outcome_compaction_allocation_keeps_original_refs():
    import inspect
    import sys
    primary=ValueError("original partition hash failure")
    secondary=OSError("cancel failure")
    allocation=MemoryError("group outcome compaction")
    source,first=inspect.getsourcelines(startup._raise_slots)
    line=first+next(index for index,text in enumerate(source) if "errors = [error" in text)
    fired=[]
    def trace(frame,event,_arg):
        if frame.f_code is startup._raise_slots.__code__ and event=="line" and frame.f_lineno==line:
            fired.append(True)
            raise allocation
        return trace
    old=sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(ValueError) as caught:
            startup._raise_slots(primary,[primary,None,None],[secondary,None,None])
    finally:
        sys.settrace(old)
    assert fired==[True] and caught.value is primary and primary.__cause__ is allocation
    frame=primary.__traceback__
    while frame is not None and frame.tb_frame.f_code is not startup._raise_slots.__code__:
        frame=frame.tb_next
    assert frame is not None and frame.tb_frame.f_locals["fourth"] is secondary


def retained_failure(error,target):
    seen=set()
    pending=[error]
    while pending:
        current=pending.pop()
        if current is target:
            return True
        if id(current) in seen:
            continue
        seen.add(id(current))
        for value in (BaseException.__cause__.__get__(current),BaseException.__context__.__get__(current)):
            if value is not None:
                pending.append(value)
        if isinstance(current,BaseExceptionGroup):
            pending.extend(current.exceptions)
    return False


@pytest.mark.parametrize("kind",["finish","wait","transport"])
def test_cancelled_sibling_diagnostic_is_not_lost_behind_genuine_partition_error(tmp_path,monkeypatch,kind):
    import inspect
    import sys
    files,_before,_expected=inputs(tmp_path)
    barrier=threading.Barrier(3)
    primary=ValueError("genuine failed partition")
    secondary=MemoryError("cancelled sibling diagnostic/transport")
    def wait(cancelled):
        barrier.wait(3)
        while not cancelled():
            time.sleep(.001)
        raise CancelledError("sibling cancelled")
    def failed(_cancelled):
        barrier.wait(3)
        raise primary
    capture=startup.StartupContentGroup(group_plan(files),[wait,wait,failed],[("a",),("b",),("c",)])
    original_clock=startup.time.monotonic_ns
    calls=[]
    def clock():
        if threading.current_thread() is capture.captures[0].thread:
            calls.append(True)
            if len(calls)==2:
                raise secondary
        return original_clock()
    if kind=="finish":
        monkeypatch.setattr(startup.time,"monotonic_ns",clock)
    elif kind=="wait":
        capture.captures[0].wait_errors[0]=secondary
    source,first=inspect.getsourcelines(startup.StartupContentCapture._raise_outcome)
    line=first+next(index for index,text in enumerate(source) if "errors = [error" in text)
    fired=[]
    def trace(frame,event,_arg):
        if (kind=="transport" and event=="line" and frame.f_code is startup.StartupContentCapture._raise_outcome.__code__
                and frame.f_lineno==line and frame.f_locals["self"] is capture.captures[0]):
            fired.append(True)
            raise secondary
        return trace
    capture.start()
    previous=sys.gettrace()
    try:
        if kind=="transport":
            sys.settrace(trace)
        with pytest.raises(ValueError) as caught:
            capture.authorities()
    finally:
        sys.settrace(previous)
        capture.close()
    assert caught.value is primary and retained_failure(primary,secondary)
    assert capture.captures[0].cancel_observed and capture.captures[0].error_observed
    assert capture.closed
    if kind=="transport":
        assert fired==[True]
