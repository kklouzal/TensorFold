# FP8 fragment packing qualification, 2026-10-21

The selected loading candidate replaces indexed fragment gathering with a
strided view and a direct copy into the final contiguous byte layout. Row
chunks remain bounded at 4,096. Padding, byte order, ownership and native
consumer arithmetic retain their existing contracts. Requantization retains
its conservative historical admission bound; the comments document how it
covers the reduced working set.

## Completed comparison

The immutable installed experiment used source
`f1a8d5d66a4f72e8b40c5302b78339e4d95f21b6`, verification image
`sha256:81db17bd18b91c2576c6d7dba4672d89fe412d9097027e991545cb8bc5d636ff`,
and wheel `e37e9ee4cbf43719c28066257a806f3424366446a8499b6a53f5217ce0a09704`.
Both arms used that image. A bound the exact historical fragment function from
`3b8cee825b21b2c88197cc811efa2544858d7ad7`; B used the installed candidate.
No installed files were edited. Timed arms had direct function bindings without
a packing counter, wrapper, profiler or trace observer.

The model was AEON-7 Qwen3.8-27B mixed NVFP4/FP8, target revision
`282e6775538f770fbda26154f78fb4321e070de3`, with z-lab DFlash2 revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. The tokenizer override SHA256 was
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
Context was native 262,144, with checkpoint precision, BF16 KV, one stream,
the default DFlash2 proposal policy, thinking enabled and no thinking-token
limit. Hardware/runtime: NVIDIA GB10 SM12.1, VBoost0, CPUs
`5-9,15-19`, ten compute/build threads, 16 GiB host reserve, Python3.12.3,
Torch2.16.0.dev20261014+cu134, CUDA toolkit13.4.3 and driver580.178.04.

Two excluded prewarms established all 232 startup packing calls and matching
geometry. The timed campaign then completed 24 fresh-process starts in six
fixed blocks: ABBA, BAAB, ABBA, BAAB, ABBA, BAAB. Compiler and file caches were
warm; this is not a cold-storage result. No model, build or weight-hashing job
ran between the timed blocks. The ordinary production container was restored
and its exact identity/configuration verified after each guarded job; speech
services retained their original identities and health.

The primary objective was production CLI entry-to-ready latency. Arithmetic
means were A11.484128697 s and B10.792733406 s. Every block favored B. The
geometric A/B ratio was 1.064050151, with two-sided95% Student-t5 interval
[1.056396242,1.071759515]. The model assumes suitable independent block
distributions; drift and autocorrelation remain statistical limitations.
Interpreter, tool validation/imports and guard activation precede this timer.

Each position retained four complete 64-token requests: short/history,
first/repeated. Rendered prompts were46/8,451 tokens; cache counts were0/45 and
0/8,444. Complete output IDs and cache/speculation counters matched in every
arm. Request TTFT and complete wall time showed no established loss; their
intervals crossed zero, so no request equivalence or speedup is claimed.
Engine prefill/decode phase counters are diagnostic. All secondary intervals
are pointwise/unadjusted and do not provide joint95% coverage.

The raw report SHA256 is
`c6ec9273e037d3033e979d320ea0cbcc0af7730a98e1d452cd2d5b1a3643b899`.
Its historical `STARTUP_GAIN_REQUEST_VERIFICATION_PENDING` status applied an
extra secondary-equivalence veto. Separate policy review corrected that
interpretation: the predeclared loading gate requires six positive startup
blocks and a positive lower interval bound, while established required-region
losses block selection. Statistics and raw records were preserved unchanged.
[The public receipt](fp8-packing-validation-20261021.json) retains all timing
positions, metrics, control digests and artifact identities without prompt text.

## Release integration and pending gates

The experiment's other source candidates were common to both arms and are not
qualified by this result. In particular, it does not qualify the omitted Qwen
prefill group sums. This release integrates the measured packing change,
admission documentation and independent tests onto the original accepted source, alongside
the exercised measurement procedure. Other optimization prototypes remain in
their development branch.

The isolated final source still needs its pinned CPU/native checks, immutable
build/installed audits and final full-model comparison before publication.
The historical installed experiment does not qualify a different image.
No Gemma/Qwen3.6/Flash Next, HTTP, concurrency, cold-storage or universal
performance conclusion follows from this Qwen3.8 loading result.
