# Bounded CPU media frontend

The additive SM70 frontend is an internal composition API. It has no CLI
activation or shipping resource defaults. The existing eager media frontend for
other providers retains its behavior. A profile owner must first admit the
complete media envelope and fixed vision encoder, use the combined source/media
software authority at bootstrap, and supply the selected GGUF tokenizer's exact
expanded-token planner. These source controls do not qualify the SC117 visual
weights, eight V100s, target/draft positions, task quality or service capacity.

## Ownership and interfaces

`tensorfold.vision.replay.MediaEnvelope` requires every resource bound explicitly:
image/video counts, per-item and total encoded upload bytes, dimensions,
per-frame and cumulative decoded pixels/bytes, selected frames, duration,
expanded visual rows, worker/thread count, per-worker virtual address space,
hard CPU time, worker and whole-staging wall deadlines, single-allocation limit,
probe bytes/time, scanned frames/packets, live staging bytes, encoded replay
bytes/items/leases/borrows, manifest/result bytes, redirects and URL length,
retirement grace, and an explicit rational assumed-FPS policy or `None`.
Limits express distinct resources; replay and staging do not supply KV capacity.
The profile owner must account for concurrent worker address-space peaks and
later GPU/pinned transfer staging in the complete host/resource ledger.

`ReplayStore(root, envelope).stage(sources, allow_urls=..., cancelled=...)`
preserves ordered `ImageSource | VideoSource` inputs. The shared root must exist,
be canonical, owned by the service and protected from other users' writes.
Stage reuses the existing bounded data-URL parser and public-HTTPS fetcher,
including redirect, address-pinning and MIME rules. Parsing and download run in
a fresh supervised CPU process. Encoded files are sealed read-only before
publication. `ReplayRef.record()` is serializable identity, not ownership.

The leader retains each `ReplayLease` until request, State and rank consumers
retire. Each such consumer holds `lease.borrow()`; `lease.close()` refuses
outstanding borrowers. References contain encoded SHA256, size, kind, detail and
media ordinal. Workers open without following a final symlink and verify the
same descriptor's immutable metadata and encoded hash before decoding. This is
a cooperating shared-filesystem ownership contract, not a new persistent cache.
Retirement enters a phase that rejects new borrowers before destructive work;
failed closes retain their owners for retry/containment. Quotas derive from
bounded owner reservation entries, so retrying removal cannot subtract twice.
Directory removal is one-shot: an uncertain removal retains the failed lease
and quota and marks the owning store for process containment. Retrying it cannot delete a subsequently
reused pathname or another lease's live claims.

`MediaExecutor(store).probe_replay(ref, cancelled=...)` returns an immutable
`MediaProbe`: selected source indices and rational actual timestamps, source
dimensions/count, global resized shape/grid and processor/content identity.
Probe scans all frames under cumulative work/byte bounds, without retaining
decoded history. Unknown/invalid FPS or PTS requires an explicit assumed-FPS
policy; there is no 24-FPS fallback. One-frame clips, changing resolutions and
multiple video streams require an explicit different policy and are rejected.
Duration admission includes the final frame's presentation endpoint or an
authoritative video-stream endpoint; missing actual endpoints are refused.
An unrelated audio/container duration does not substitute for video metadata.
Accepted container families remain MP4/QuickTime and Matroska/WebM.

`executor.pixel_unit(probe, index, cancelled=...)` creates a `PixelUnitLease`
for a complete image or temporal pair, with normalized CPU FP32 patch rows of
shape `[grid_h * grid_w, 1536]`. The fixed encoder acquires `pixels.borrow()`
and reads its read-only NumPy `values`. Keep this borrow through asynchronous
transfer completion; do not mutate pixels or cast away read-only protection.
Both outstanding borrowers and exported array buffers prevent mapping release.
Retiring the pixel lease also retires its replay borrow and staging reservation.
There is no normalized/projected full-history archive. Replay misses may decode
again within the same explicit bounds.

`ReplayMediaFrontend(store, executor, planner, cancelled=..., allow_urls=...)`
supplies the opt-in `prepare_sources` hook, selected before the old eager decode
path. The planner receives the rendered template and ordered probes plus
`max_prompt_tokens` and `max_visual_tokens`; it must validate canonical token
expansion, timestamps/delimiters, visual spans, expanded input-plus-output
allowance and positions before GPU execution. The resulting
`PreparedReplayMedia` owns plan/probes/replay, exposes `token_ids`,
`borrow_replay()` and `close()`. The Engine/profile owner is responsible for
request failure/cancellation, retained State and cross-rank lifetime wiring.
The bounded hook also requires `check_media_health()`. It checks containment
before projecting errors into HTTP input/capacity refusals; a poisoned internal
failure cannot become an ordinary 400/503. CPU readiness owners can call
`executor.check_health()` after a completed operation before retrying `MediaBusy`.

## Process and processor contracts

Workers use fresh exec, never a warm Python fork. The existing
`expert_launch.supervise` owns their sessions and descendants through retirement.
Hard CPU/address-space/file-size limits precede JSON parsing and native imports.
CUDA devices are hidden and transforms explicitly use CPU. Worker concurrency
and staging are shared by all executors for a store, with fail-fast admission.
Temporary worker-slot or staging pressure raises `MediaBusy`; the bounded HTTP
hook maps it to the existing 503 capacity response. A unit larger than the
entire configured staging capacity remains a permanent input refusal.
Diagnostic output has a separately sealed-size memory file. Cancellation waits
for session retirement before releasing resources. Unresolved retirement poisons
the store, preserves reservations/borrowers, and requires process containment;
it never becomes a successful cancellation or capacity release.
A normal supervisor return proves session retirement, including returned
cancellation/resource statuses. Every supervisor exception is conservatively
contained; diagnostic exception types do not prove retirement.
The existing native `OwnedFD` gains only a Linux sealable-memfd acquisition;
its existing file acquisition/close contracts remain unchanged. Replay reads
reuse the existing native regular-stream owner and hash at most the admitted
extent plus one EOF byte. Workers resolve the normal installed app package;
there is no source-test provider loader or raw-descriptor fallback.

FFmpeg's process-global `av_max_alloc` binds the same owned libavutil loaded by
PyAV, with codec `max_pixels`, probe-size/analyze-duration options, explicit
thread limits and packet/frame scan ceilings. Tensor allocation sizes are
checked structurally before preprocessing; the per-worker address-space limit
contains library transients. For images, decoded bytes conservatively charge
four bytes per source pixel; video charges actual decoded plane buffers.

Transformers reference `50e35cc0ca18ec19b80653690de553e6decf74f5` and both
official configuration hashes are pinned. The exact TorchVision
`Qwen2VLImageProcessor` and `Qwen3VLVideoProcessor` run on CPU FP32. Still images
use supplied-PIL RGB conversion: encoded orientation, alpha dropped and
grayscale expanded to RGB. This differs from the legacy eager decoder's EXIF
transpose and white compositing and is covered separately.

Video uses the pinned 2-FPS automatic rounded-uniform sampler and enforces the
administrator's selected-frame bound. The complete selected frame count
determines `smart_resize` once with the pinned temporal-spatial budget.
Individual pairs then use exact TorchVision bicubic/antialias resize followed by
`video_processor(do_resize=False, cap_pixels_per_frame=False)`; odd final frames
repeat. The explicit `actual-pts-v1` adapter averages rational first/last temporal
timestamps and preserves the pinned `<{time:.1f} seconds>` format. Actual VFR
timestamps are retained instead of substituting source-index/FPS timestamps.

## Verification scope

Maintained controls are `tests/test_media_replay.py` and
`tests/test_media_frontend_runtime.py`; the latter requires the immutable pinned
CPU media image. They exercise exact whole-official image/video processor
parity, alpha/grayscale behavior, temporal order/padding, VFR PTS, explicit
assumptions, malformed/changed inputs, quotas, cancellation, hard resource
failure and lifetime/poisoning. They use small independent lossless RGB fixtures.
They are software controls, not a measured maximum media envelope, service
activation, SC117/mmproj or V100 qualification.

## Serving integration

`--media-profile-manifest PATH --vision --no-graphs` selects the source-stage
GGUF SM70 TP8 adaptive unique-history integration. The manifest parser requires
the complete versioned `tensorfold.sm70-media-profile` envelope, pinned processor
and vision source identity, encoder placement, input/cache rows and workspace.
There are no inferred CPU or media resource limits. The existing text-only
providers keep their selection.

The leader uses the prepared native target tokenizer and its complete model
token-ID range, including nonlexical UNUSED rows. The same effective token
roles and media/position identity reach every rank. One bounded `MediaWork`
lookahead prepares CPU pixels before slot eligibility; the coordinator consumes
ready units without waiting for a decoder. Only the fixed encoder receives
pixels, and it broadcasts the completed input rows over the existing cohort.
Target and MTP borrow those same rows until their staging readers retire.

The logical App request owns replay across every generation/thinking subrun.
Backend completion retires its claim; final App closure releases replay only
after CPU work and all rank consumers retire. Counting and preparation failures
also release their acquired owners. These software controls do not qualify
V100/NCCL execution, model output, capacity, numerical quality or performance.

The current v6 FIFO, complete retained-state backing, standard API lifecycle
and capture/accounting requirements have separate implementation gates.
