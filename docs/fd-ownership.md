# Native descriptor ownership

`tensorfold._fd_owner.OwnedFD()` constructs a valid closed descriptor slot on the
declared Linux and macOS CPython targets. Construction acquires no descriptor.
The caller must publish that slot into its operation journal **before** calling
`owner.open(path, flags, mode=0o600, *, dir_fd=None)`. Open returns `None`; acquired
ownership is already stored inside the journaled native slot when it returns.
Interruption before storing the constructor result has no resource to close.
Interruption at native open return or before storing its result leaves the live
descriptor in the existing journal for explicit cleanup and error observation.
An acquiring factory which returns a newly opened owner violates this protocol.

The extension uses the CPython 3.11 Stable ABI and C11. Source builds require a C
compiler and Python development headers; matching `cp311-abi3` platform wheels
supply the compiled extension. The CUDA development container supplies the
toolchain. Native Apple execution remains a separate qualification.

`open` converts path and integer arguments before acquisition, normalizes an
exact bytes path, rejects embedded NULs and permission modes outside
`0..0o7777`, and calls `openat` with `O_CLOEXEC`. Flags retain their OS meaning;
callers select nonblocking, no-follow, directory and exclusive creation policy.
A failed validation or OS open leaves the slot closed and retryable under the
caller's explicit policy. A successful acquisition is one-shot: opening again
raises `RuntimeError`, including after close. Closing a fresh slot has no effect
and does not prevent its first acquisition. There is one OS attempt per call;
neither acquisition nor close silently retries an OS error.

The native acquiring state is set before argument callbacks. Reentrant or
concurrent open and close during acquisition raise `RuntimeError`; `fileno`
raises `ValueError` until publication. The syscall releases the GIL, then its
descriptor and used state are published under the reacquired GIL before any
fallible Python operation or return. Operation owners serialize lifetime changes
and observe acquisition completion before borrowing or cleanup.

`dir_fd` optionally borrows a parent descriptor through open completion. Relative
paths use that parent directly; absolute paths retain the POSIX `openat` behavior
of ignoring it. Callers validate each path component and use
`O_DIRECTORY`/`O_NOFOLLOW` at traversal boundaries. Successful child acquisition
owns a new independent descriptor and never consumes the parent.

`fileno()` returns a borrowed descriptor. Callers must finish all borrowed I/O
before `close()` and never externally close, reassign, guard or transfer it.
Close retires native ownership and executes OS close within the same C call.
Read-only `closed` stays true afterward, including when close reports an OS
error. Repeated close calls have no effect; a reused numeric FD is never retried.
`closed` describes absence of borrowable ownership, including before first open;
it does not prove completion of a released-GIL open or close. Observe the original
operation's return or exception before releasing surrounding resources.

The type is immutable and cannot be subclassed. Instances have no dictionary,
weak references or owned Python containers. Every operation close failure must
be returned explicitly by the caller's cleanup protocol, preserving its primary
failure and any remaining journal. Destruction is only a misuse backstop; its
unraisable diagnostic is never accepted as operation failure transport.

Startup supplier checks bind `_ownership_version == 2`,
`_limited_api == 0x030B0000`, and `_source_sha256` to the installed maintained C
source. They also verify the binary's wheel `RECORD` hash, installed path,
platform/ABI tag and approved source manifest. These metadata describe build
inputs and do not authenticate an untrusted binary. Protocol1 constructor
acquisition was task-scoped and has no retained compatibility path.
