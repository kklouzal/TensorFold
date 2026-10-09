# Linux native descriptor fault qualification

`fd_close_fault.c` is a project-owned test shim. It is compiled separately and
preloaded into one isolated test process. It is never linked into the production
extension or installed over a dependency. `fd_owner_fault_native.py` loads
the exact Root-approved ABI3 provider binary and checks its source metadata.

The setter arms one live ordinary regular-file descriptor, its device/inode and
the arming thread. A mutex orders setter, close, snapshots and reset. The first
matching close invokes the actual next libc `close`, proves the old number is
invalid, and opens a distinct test-owned replacement with `O_CLOEXEC`. The
replacement must naturally obtain the exact old number. The shim never uses
`dup2` to overwrite another descriptor. Only then does it return the selected
`EIO` or `EINTR`; mode zero tests successful native return. A real close failure,
identity mismatch or failed/nondeterministic reuse fails qualification. Every
unmarked close uses the next libc close with its original errno behavior.

The process contract excludes outside descriptor reassignment and foreign
pthread cancellation. No borrower is using the marked descriptor during close.
Only the arming thread may reset. Reset awaits the native sequence, disarms, and
does not close the replacement. The worker independently proves its inode and
explicitly closes it after reset. The marker counts any later close of the old
number on any thread. Completed parallel repeated owner closes and native
reference-counted destruction must leave that count zero and the replacement
readable. Independent sentinel descriptors and an unrelated concurrent-close
control check scope. End-of-process descriptor identity sets must match.

SIGINT cases send `os.kill(os.getpid(), SIGINT)`, observe the signal number in a
scoped Python signal handler, and invoke the normal interrupt handler. Scoped
profiling events place the actual signal before native `close` entry or after
its successful return. The journaled open case uses CPython 3.12+ local instruction
monitoring, or the declared 3.11 opcode tracing facility, to send SIGINT after
native `owner.open` and before the caller stores its `None` result. A closed
protocol2 slot was already published in the operation journal before entry.
The interrupt therefore leaves that exact live owner available for explicit
cleanup. Zero/EIO/EINTR modes prove consumed-state reuse and preservation of the
original interruption plus every explicit close error. No destructor or
unraisable diagnostic receives operation-status credit. These are deterministic
boundary signals, not arbitrary asynchronous signal stress.

The Root-only packet is `native-fd-fault-source-09`. It pins Root's protocol2
source03 provider approval: fifteen ordinary native tests passed without skips,
source `e4f9e187`, binary `f4408452`. Root supplies that actual binary's mounted path.
The maintained ROOT generic worker runs the phase in CPU mode with
`entry_function="run"`, the fresh result directory as its sole path argument,
and `TENSORFOLD_ROOT_REMOTE_QUALIFICATION=fd-owner-fault-v1`. It injects its
authenticated `ROOT_NATIVE_HELPER`; the phase borrows that context for compiler
and child logs, bounded reads and atomic receipts. The outer drains the same
context after every child and log has retired. The CLI remains an isolated
controller source seam; the generic function phase is the current helper gate.

```text
TENSORFOLD_ROOT_REMOTE_QUALIFICATION=fd-owner-fault-v1
argv=["fd09/run_root_native.py"]
entry_function="run"
path_arguments=["/results/fd-fault"]
```

The driver checks its sealed source manifest, compiles only the shim with strict
C11 warnings, pthread support, explicit exports and hardened shared-library
link flags, saves full compiler/ELF output, and launches the worker with only
that shim in `LD_PRELOAD`. Root supplies 8 CPU cores/MAX_JOBS=8, 12 GiB RAM,
zero swap, no GPU, and a 600-second process deadline. The bounded internal
deadlines total at most 560 seconds, plus at most five seconds for an owned
process-group cleanup on failure. Output scope uses resolved parent paths;
The controller establishes failed child status before closing either log
stream, attempts every owned stream close and process-group drain, and retains
all failures using native exception causes. Receipt serialization/write/close
and optional exception formatting have the same primary-preserving contract;
they never call opaque annotation hooks. Receipt publication failure still
fails the job even when a native control file was successfully produced.
Repeated exception identities are deduplicated, and a primary never appears
in its own cause group. A nonzero child status observed while draining an
interrupted wait is retained alongside the original interruption.
The worker attempts independent sentinel cleanup even if marked cleanup fails,
and attempts all signal/profile/opcode/monitor restorations. A failed reset
retains the exact dependent native owner and fails qualification; it never
closes an unproved replacement. Operation acquisition uses `OwnedFD()` followed
by journal publication and `owner.open`; constructor acquisition is refused by
protocol2. Destruction is a misuse backstop, never operation error transport.
This packet does not claim Apple, another
ABI/toolchain, sanitizer, final canonical wheel/RECORD, model, or performance
qualification. Any changed provider/shim/source/runtime invalidates reuse.
The standalone worker lives under `tests/native` without the `test_` discovery
prefix; ordinary pytest collection runs only its stdlib source controls.

Primary references: [Linux close semantics](https://man7.org/linux/man-pages/man2/close.2.html),
[CPython profile events](https://docs.python.org/3/library/sys.html#sys.setprofile),
[Python signal delivery](https://docs.python.org/3/library/signal.html),
[instruction monitoring](https://docs.python.org/3.12/library/sys.monitoring.html).
Linux releases the numeric descriptor before late close errors. This test
targets that Linux contract; it makes no generic POSIX EINTR promise.

The standalone transport block is generated from held `owned_cleanup_errors_v4.py`
by `generate_fd_fault_error_transport_v1.py`. Every independent cleanup captures
the operation's native cause and context before invoking foreign callbacks;
those exact roots survive callback mutation and failed group publication.

The controller validates the actual Root receipt producer's typed
`actual_native_methods`, ownership API, source/binary identity and qualification
status before output creation or compiler invocation. The exact receipt SHA
is part of the sealed operation boundary.
