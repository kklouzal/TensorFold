# Outside-core CPU diagnostic

`--cpu-counters` is an opt-in FASTv2 probe diagnostic requiring the current
GPU-local worker configuration. It adds no GPU API, performance package, polling
thread or model restart. The coordinator captures cumulative CPU counters before
calling each cohort and after that call has returned or raised. The existing
cohort implementation freezes its core endpoints and retires its callers and
backend owners before returning. Postterminal analysis must verify that every
collection bracket is disjoint from all six measured core intervals.

The snapshots bracket setup/readiness, decode, cancel/drain and reporting. Their
deltas are **whole-cohort CPU**, not exact CPU utilization inside a ten-second
core. Sequential reads are not simultaneous. A file watcher observing the prior
case's JSON would not guarantee an outside-core point before the next case.

The reader records cgroup-v2 `cpu.stat`, `cpu.max`, `pids.max`, membership and
identity; each rank and the supervisor's `/proc` CPU ticks/start time; and live
TID CPU ticks and `schedstat`. It pins process/thread directories, validates
start times, and records vanished TIDs. Missing scheduler statistics are explicit
and do not mean zero scheduler wait. It does not reset counters, alter sysctls,
affinity, memory policy or quota. Kernel CPU tick rate and monotonic read brackets
are stored with the raw data. Container CPU totals overlap process/TID totals.

One snapshot is capped at the existing 2,424,832-byte aggregate receipt allowance;
twelve phases permit at most 29,097,984 output bytes. A global allowance of
31,260,672 bytes adds 2 MiB for transient reader state and 64 KiB for descriptors.
It is charged once through the existing finite-caller host-control admission,
including a conservative charge for output/page-cache growth. Output is streamed;
reports retain small descriptors rather than snapshot histories. The existing
512 KiB leader and 256 KiB follower report limits remain. Serialized bounds do
not establish equal RSS; local allocation controls and target resource admission
remain required. No graph/vendor/VRAM credit is borrowed and full24 Q8 funding is
preserved.

The current implementation supports the model container's verified cgroup-v2
namespace root. If its CPU/PID controllers, membership or fields are unavailable,
it stops before launching model workers rather than silently reading an ancestor
or inventing zeros. New/retired TIDs, changed identities, counter resets and quota
changes must not be clamped into valid deltas. Parent and model PID namespaces
must be interpreted using the recorded identities and clock source.

Target CPU-counter collection is unrun. The qualification host storage incident
blocks remote staging and GPU execution; existing NUMA model evidence remains a
separate completed reference. Counter-enabled throughput is diagnostic and does
not inherit an uninstrumented comparison's repeatability or qualification.
