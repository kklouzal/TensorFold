# Experimental DFlash logits-only graph

`Gemma4Engine(..., dflash_graph=True)` is a default-off internal constructor
candidate for an enabled private-FP8 DFlash companion with a block of2..16 rows.
Depth4 uses five block rows. The ordinary eager path, BF16 companion profiles,
other depths and public serving defaults retain their existing behavior when
the option is false. There is no public CLI flag or deployment selection.

The experimental checkpoint tested source
`2c3d7b0efddc7d871aedba86683f33b9c3645568985c78edd6fb0d544f6c3d04`,
runtime image
`sha256:d83610b5c2b7fcf3f205bae109017edff3934141a9c76b483d00594f67148bdd`
and verification image
`sha256:5a9521b520e782da5cb0300c65b419566ab70077ef640553d2a42fc9ed552804`.
It passed 519 source CPU tests with 15 documented CUDA skips, 188 installed CPU
tests and 37 native GPU tests without skips. These immutable build identities
precede this documentation-only update; runtime and maintained test bytes are
unchanged. Earlier07d results are not inherited or relabeled.

Same-image graph OFF/ON workers used AEON revision
`90c8601b1a439497bc15d8fba9e52990b45ea0cb`, DFlash revision
`77d4202772dfe50b2396ec7bac9cfffc7b9e7057`, private FP8/depth4, Delta4R16,
cache NONE and a131072-token window on GB10/SM121. Both matched all 11 real
progressive prefills through committed position131071, 17 proposal/logit/q/state
probes, two target-teacher comparisons and 13 generation cases. Required
rejection, native-EOS, grammar/masked-verification and position-sensitive
witnesses were present, with no witness gaps; both workers retired safely and
the enabled graph owner closed its retained fields. This is the recorded
component/full-model correctness scope, not HTTP/media/cancellation or usable
full-window reply-room qualification.

No performance or production selection follows. Production requests advanced
from147 to152 during correctness, and private completed-cache copies confound
startup comparisons. Elapsed worker times and allocator observations are not
controlled throughput, startup or physical UMA-residency evidence.

The graph contains the existing raw embedding, companion layers and private
FP8 vocabulary projection, including the original softcap and every rounding,
attention, reduction and noncausal block operation. Actual-q extraction, host
readback, canonical tie repair, CPU draws, target verification and accepted
tap append remain outside capture. In particular, sampled proposals retain the
same complete block even near a reply's final emitted token.

One DFlash instance retains at most one graph generation, keyed by actual
DraftState identity, block length and exact full-attention slot count. A key
change synchronizes the current exclusive stream and retires the old graph and
its pool before warming/capturing another. This charges capture and transition
cost instead of retaining a growing graph dictionary. No model history is
copied for capture. Warmup and replay write only the same reserved speculative
suffix as an eager proposal, without advancing committed length; the current
position, anchor and mask rows are staged before use.

Graph output storage remains private. Every `DFlash.logits()` result is a fresh
owning FP32 clone, so later replay, pool replacement and full-row q tie repair
cannot mutate or retire a caller's previous result. That copy and its memory
belong in the end-to-end comparison. Close first drains the exclusive engine
stream. An unknown native failure retains its graph/state/exception owners,
blocks further engine work and refuses another CUDA retirement attempt.
The owning engine also retains unexpected faults first observed after replay,
including proposal readback, target probability processing or stream drain.
The first fault keeps its original exception/cause/context; later retirement
cannot overwrite it. Cooperative `RequestCancelled` remains recoverable, and
ordinary prompt validation still runs before entering the mutable lane.
Callers of standalone `DFlash.logits()` own containment of faults observed by
their later external consumers; the engine cannot intercept those operations.

Header admission adds a one-generation tensor-payload estimate: all projection,
normalization, residual/MLP and attention outputs without reuse credit, optional
split-K scratch, fixed token IDs, head/softcap outputs and the fresh return.
The native W8A16 lane path has no activation quantizer or expert plan. The
estimate is not physical UMA residency or a bound on allocator fragmentation,
CUDA graph metadata or reserved pool bytes. Exact-image allocation/reservation,
host peaks and actual capture compatibility must be qualified separately.

Maintained tests are `tests/test_gemma_dflash_graph.py` and
`tests/cuda/test_gemma_dflash_graph_cuda.py`. CPU adapters exercise policy,
state/geometry dispatch, fresh-result lifetime, failure containment and actual
header integration. Native tests compare exact eager/graph FP8 logits and q,
changing anchors/positions/geometry, restored rejected branches and context
tails, and retain allocator readback. The tiny component is not full-model,
vision/service or physical-RAM qualification.

Before selection, complete new-image serving/media/cancellation and usable
full-window reply-room gates, then controlled complete-round and serving
comparisons including warmup/capture, transitions, clone/readback, startup,
prefill, decode and memory. Keep the six sampling/workload regions separate.
The passed component/full-model checks do not establish a speed gain,
execution neutrality, physical-memory capacity or a shipping deployment.
