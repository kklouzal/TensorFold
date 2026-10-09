# CUDA indexed checkpoint startup

Large Safetensors indexes resolve each distinct shard target once during a
header operation, then recheck targets before publishing the complete mapping.
Index/schema validation, authorization of every shard, checks before opening
each file, and the native header reader remain unchanged. Small indexes keep
the original per-entry checks. No-index and explicit-file reads keep their
original paths. The operation retains no process-wide path cache.

Checkpoint metadata and file targets must remain immutable during an operation,
as required by the header reader. Publication checks reject a persistent
symlink retarget; neither the original nor optimized reader provides an atomic
filesystem snapshot or detects every transient retarget.

On the RTX PRO 2000 Blackwell host with 64 GB system RAM, the actual
Qwen3.8-Flash-Next EXL3 index contained 304,105 entries. Eight alternating paired
header calls consumed identical complete dictionaries. Mean relative latency
decreased 84.81%; the conditional paired 95% interval was 83.61–86.01% faster.

Four fresh unprofiled public constructors ran in baseline/candidate/candidate/
baseline order with four slots, context 2,048, YaRN factor 2, int8 K/V, automatic
expert caching, eager execution, and prefetch disabled. Initial private compiler
caches were empty. Constructor times were 277.232/231.081/230.387/279.883 seconds.
The candidate/baseline geometric latency ratio was 0.8283; its conditional
paired 95% interval was 0.7650–0.8969. This is two starts per arm and two paired
log ratios, with an independence/normality assumption; it supplies no population
tail guarantee.

All four runs had identical capacity plans and 1,767 initialized weight records
covering 36,063,143,264 observed tensor bytes. Both serial and MTP 64-token
teacher outputs matched in each run. The byte scope excludes mapped PLE payload,
mutable scratch, dynamic pointers and cache storage; tensor aliases can be
counted twice. Resource budgets, no-swap/OOM checks and clean retirement passed.
This qualifies indexed-header startup in that configuration, without a
generation-throughput or unrelated loader-lifecycle claim.

Full raw receipts and independent admission are retained under
`task-artifacts/quality-preserving-optimization`: the metadata actual02 and
constructor actual02–05 artifacts, plus `capacity-index-canonical-disposition-actual02-05.json`.
