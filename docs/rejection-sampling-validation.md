# CUDA DFlash rejection sampling

This integration increment implements the previously selected standard rejection
sampler for sampled Gemma 4 DFlash, Qwen dense DFlash2 and Qwen3.6 CT NVFP4 DFlash
requests. It is an implementation candidate; its final installed/native/model and
performance gates are pending. Existing target precision, grammar, penalties,
context limits and supported hardware remain the governing contracts.

A proposal token is drawn from its recorded distribution `q`, independently of
acceptance and residual noise. Verification accepts it with probability
`min(1, p(token) / q(token))`; rejection draws from the normalized positive part
of `p - q`. Here `p` is the target distribution after the actual grammar mask,
penalties, temperature, top-k, top-p and min-p filters. A truncated proposal
vocabulary can define `q`; it does not restrict target support. The algorithm
follows [Leviathan, Kalman and Matias (ICML 2023)](https://proceedings.mlr.press/v202/leviathan23a.html).

Sampled drafting uses fixed request-local linear blocks. Greedy drafting retains
its existing tree/chain path. Proposal positions and random domains do not depend
on a physical batch slot or scheduler packing. Background requests replay their
original prompt and check already emitted tokens. A fixed request seed and engine
configuration must remain reproducible across cache reuse and supported batching.
Sampled serial/drafted tokens and different proposal configurations may differ;
greedy serial identity remains required. Other families and Apple paths retain
their separate sampling contracts.

The migration assessment reproduced a concrete failure in naive archive wiring:
with a fixed seed and position, changing whether a node had a child changed the
emitted token. Consequently, scheduler-dependent proposal truncation cannot be
treated as a harmless performance choice. The implementation must preserve the
logical proposal and its actual `q` through grammar pruning, row limits and replay.
The archived core's fallback after an empty residual is not an accepted recovery
policy.

Probability rows represent categorical laws: the stored nonnegative FP64 values
are normalized by their represented sum. Acceptance accounts for both row sums.
An outward-rounded interval resolves ordinary comparisons; ambiguous comparisons
use exact integer arithmetic on the represented values. A rejected proposal uses
the exact positive difference between those normalized laws before converting the
normalized residual back to FP64. An empty residual cannot silently select from
the original target row. The keyed binary64 uniform grid has 52 midpoint bits,
so every generated uniform is representable and strictly between zero and one.

The migration also corrects a pre-existing stochastic numerical defect. With
large finite equal logits such as `[1e30, 1e30]`, adding Gumbel noise to absolute
scaled scores rounded the noise away and selected the first token every time.
Stochastic scores are now centered before temperature, normalization and noise;
sharded normalization retains a centered representation. Greedy selection is
unchanged. Seeded outputs affected by the old defect are corrected, rather than
promised to match older builds. The shared CPU/MLX reference and active Metal
sampling path receive the same correction; native Apple validation is pending.

Cache qualification also corrects its eviction preparation. Qwen's cache retains
reused entries before unused ones; distinct one-shot eviction prompts could leave
the original hot entry cached. The gate now reuses each existing eviction prompt
before moving to the next. Its cold/warm coverage thresholds and token equality
checks are unchanged. The original desktop attempt is retained as incomplete,
with all token comparisons matching but no demonstrated cold repeat.

## Evidence and remaining gates

Pre-migration source `cefd26d30b9bc653b0a3a03d44b28747579372e8` passed
10,164 installed CPU tests on each of ARM64 and AMD64, with 204 documented skips
per host. Its installed GB10 component batch passed 303 tests, with two deliberate
device-trap skips. These results establish the preceding integration checkpoint;
they do not qualify the migrated sampler.

The same installed checkpoint completed all five 4k HTTP identity/mechanics
profiles: Gemma, Qwen3.8 and Qwen3.6 general-thinking, coding-thinking and instruct.
The controls produced 48 complete 512-token benchmark replies, 12 greedy and
12 sampled serial/drafted comparisons, and 36 repeated pairs. All matched under
that checkpoint's coupled sampler. Completed-answer/tools, long-context and comparative
performance are separate gates. Current GB10 qualification uses VBoost 0 and the
host's ten-CPU container partition. Historical results retain their own conditions.

Migration completion requires effective-probability and independent categorical
oracle checks; room/EOS/grammar/penalty/error cases; native proposal lifetime,
greedy identity, repeat/cache/batch/replay and supported shard checks; and the
affected installed full-model API, quality, context and performance gates. No
acceptance-rate or throughput improvement is established by the algorithm alone.
