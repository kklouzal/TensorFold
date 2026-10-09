# Flash Next CUDA ownership

`FlashNextEngine` owns its communicator, decoder plans, expert pool and table
read journal. Startup failure observes every accepted table read before releasing
dependent resources. An interrupted unfinished read or failed native retirement
retains the engine with the primary exception for recovery or process containment.

`close()` stops new admission and drains accepted requests and the scheduler.
Normal rank-zero close then sends the existing peer stop frame before retiring
its communicator. The CLI already drains its serving handlers before calling
engine close. An explicit rank-zero `shutdown()` sends that same frame, stops
new local requests, and must run after the current serial request finishes.
It is safe to call `close()` after `shutdown()`.

Rank one's `follow()` owns each store receive and its resulting decode until
completion. Its receive marker belongs to that operation's native scope lock;
matching retirement clears it, including cancellation before the receive starts.
A rejected overlapping follower cannot retire another operation's marker.
Closing from another thread during an idle receive fails immediately
with a retained-resource error; it does not cancel the native wait, poll, write a
local request key, or release the communicator. Ask rank zero to shut down or
depart, join `follow()`, and retry close. The original one-hour acknowledged store
timeout also ends a receive after local admission closes. Other store errors
propagate. Rank-zero network departure keeps its graceful `follow()` return,
but selects abort before fencing because the peer may be absent or aborting.
An acknowledged normal stop frame keeps ordinary teardown.

`close(abort=True)` selects failed-operation teardown. Failed startup or accepted
request work makes this policy sticky. Abort precedes device fencing; ordinary
communicator retirement fences before destroy. A native destroy/abort whose
completion is unknown is not retried. Source fixtures do not qualify two-GPU
NCCL, model output, or performance; those require their actual runtime gates.
