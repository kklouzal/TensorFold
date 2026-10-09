# Nemotron CUDA sampling contracts

Nemotron's sampler keys each positive-temperature draw by the existing masked
63-bit request seed, absolute position `meta[0] + row + 1 + offset`, and global
token ID. Key hashing, FP64 temperature/Gumbel arithmetic, min-p threshold,
rank-by-rank top-k mass accumulation, and confidence rounding are unchanged.

Greedy selection examines the complete original-dtype row. Without a map, the
first physical maximum is also its lowest global ID. With a map, equal maxima
select the lowest global ID, including maps whose IDs are not in column order.
Greedy confidence retains the original approximation: the top 20 FP32 candidate
values, temperature one, FP64 reduction, rounded to FP32 before output conversion.
It is not a full-vocabulary probability. Positive top-k likewise retains its
FP32 candidate-score domain; nucleus with top-k off uses the existing FP64 row
and full-mass rule. Raw callers must prove the relevant normalized score domain
is finite; `-inf` masks remain legal with at least one finite score per row.

Candidate extraction keeps the original `torch.topk` values in their physical
slots. A complete stable `(score descending, global ID ascending)` order repairs
only the retained boundary's IDs, so a tie wider than the margin cannot hide an
eligible ID. Strict-score inputs keep original output and confidence bytes.
Nucleus orders original FP64 values before temperature, resolving equal values
by global ID, before its unchanged exponent, sum,
and cumulative mass operations. Top-p zero or one disables the nucleus cut,
as in the declared keyed sampling policy. No host score readback or graph
capture synchronization is introduced.

The split target and split MTP head share exact candidate-word exchange: FP32
scores and signed64 IDs travel as three signed32 words per candidate, with rank
zero's rows first. Shape, dtype and device are checked before reinterpretation.
The MTP head retains enough candidates for the request's top-k, including top-k
above 20; top-k zero carries the complete local head to the full nucleus.
The default top-20 confidence approximation remains the same.

`MTPHead` validates external draft IDs before upload or head indexing: distinct
integer IDs, in the full model vocabulary, representable by signed32 outputs,
and a positive multiple of 64 (128 for a split head). Raw device map/candidate
APIs require that same value proof; metadata checks do not read device values.
Device inputs and maps stay immutable until queued consumers finish, and callers
establish completion of a producer on another stream. Native candidate inputs
are contiguous; legal strided output/confidence vectors use contiguous staging
and stream-ordered copies. Position and settings buffers remain model-owned
through capture and replay. MTP confidence readback rejects nonfinite or
out-of-range values before adaptive draft policy consumes them.

Qualification distinguishes guarded source/metadata proofs from real CUDA
execution. The maintained primitive tests use an independent complete-row FP64
and Python-integer hash oracle, original valid-region byte comparison, corrected
metadata failures, exact integer-word exchange, and capture replay. Two synthetic
shards on one GPU do not establish NCCL, two-GPU model parity, or model performance.
