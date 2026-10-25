# Explicit software NVFP4 on SM70

The maintained candidate imports a manifest-declared ModelOpt recipe into the
same Flash Next `Weights`, layer execution, scheduler, history allocator and
capacity planner used by the GGUF provider. It retains row-major E2M1 bytes,
E4M3 block scales per16 inputs and the source's direct `weight_scale_2` factor.
Original floating exclusions convert to the manifest's FP16 representation.
This is software W4A16. It uses no native NVFP4/FP8 matrix instruction and does
not expand the complete target to FP16.

The source recipe is deliberately explicit: known Flash Next operators,
ModelOpt direct global scales, supported group16 shards, and actual floating
exclusions. Unsupported FP8/MXFP8 projections, compressed-tensors reciprocal
scales, unknown transforms and unknown tensor roles fail before device
allocation. They are not silently cast or interpreted as another encoding.
The existing Blackwell/native NVFP4 routes retain their current behavior when
no family manifest is selected.

The numerical profile `nvfp4-e2m1-e4m3-fp16-fp32-v1` fixes FP32 scale arithmetic
in `(e2m1 * e4m3) * global` order, then FP16 operand rounding. Fixed M32/N64/K256
products and increasing K-tile sums accumulate in FP32. `--precision full`
disables declared activation quantization; it does not request expanded weight
storage. Departures from source W4A4/QAT/BF16 computation remain **unqualified**
until the selected artifact's numerical, model-quality, speculative, resource
and V100 gates pass. Compilation and small decoded-tile checks cannot establish
those results.

Prepare a manifest from complete, immutable local files. Supply the actual
source repository/revision and normalization convention: offset0 means stored
gamma; offset1 means a centered parameter to which the operator adds one. The
tool does not infer the convention from a publisher or weight means.

```bash
python tools/prepare_sm70_nvfp4.py "$MODEL_ROOT" \
  --repository "$SOURCE_REPOSITORY" --revision "$SOURCE_REVISION" \
  --norm-offset "$SOURCE_NORM_OFFSET" --linear-kernel reference \
  --expert-kernel reference --output "$FAMILY_MANIFEST"
```

Preparation hashes source tensors, config and actual default tokenizer/template
inputs with bounded host buffers. It validates the family tensor graph before
publishing a new manifest without overwriting an existing file. Reuse it for
the same source/recipe identity. Any changed file, recipe or transformation
requires new preparation and invalidates affected qualification. No import or
conversion runs on request admission. The first recipe pins the checkpoint's
default tokenizer/template; external overrides need a different implemented
profile.

The initial reference invocation uses the normal serve command:

```bash
tensorfold serve "$MODEL_ROOT" --backend cuda --family-manifest "$FAMILY_MANIFEST" \
  --precision full --kv-dtype fp16 --no-drafts \
  --max-pending-requests "$MAX_PENDING_REQUESTS" \
  --max-http-connections "$MAX_HTTP_CONNECTIONS"
```

TP1/2/4/8 sharding is admitted only when the manifest's heads, vocabulary,
expert widths and group16 boundaries support it. The existing launcher supplies
the normal rank/master flags. Unique histories still require the verified TP8,
two-global-KV-head geometry; other geometries retain their existing layout.
An external validated stock MTP companion uses the existing shared head and
MTP path. Unimplemented embedded-draft layouts remain an explicit refusal.

The engine prepares the new decode/GEMM workspace and measures its persistent
Torch/physical residency before solving capacity, then adopts that exact
workspace during import. Loader plans report source encodings, each rank's
retained bytes/shards, excluded conversions, host PLE mapping, bounded import
spans, matrix/expert workspace, and actual preparation observations separately.
Preparation exercises both FP16 and FP32 linear output entrypoints before its
memory observation. Offline compile/link/import checks passed with the source
Torch CUDA12.9 SDK and protected CCCL3.5/CUTLASS4.8.1 headers; they establish
build compatibility, while physical residency and device behavior stay pending.
The reference expert provider uses a bounded CPU selection snapshot and fixed
activation tiles. The separately named `--linear-kernel wmma` preparation choice
uses fused M32/N64/K256 FP16 shared tiles with FP32 accumulation, eliminating
the reference's per-tile decode/GEMM dispatches. `--expert-kernel device` uses
GPU CUB counts, stable members and actual group work tiles, with fused gate/up,
FP32 SiLU/product, FP16 activation storage and down WMMA. Both are maintained
unqualified candidates. Their workspace, residency and complete-step behavior
must be compared under the capacity-first mission before selection.
Whole-forward CUDA graphs and optimized execution profiles remain declined.
Encoded PLE rows reconstruct directly into FP16 through the existing mapped or
positional reader lifetimes; import streams every table's finite-range checks.
Host object, allocator, worker and Future effects remain explicitly unqualified
runtime measurements alongside construction-derived array payloads.

Retained weight parts and their storage bindings are immutable. Preparation
validates one shared workspace/device and excludes source/scratch aliases;
execution rejects overlapping input, selection or output storage before either
provider can mutate it. The source interval catalogue is built once, so output
validation does not scan all experts on each call. Hashing consumes at most the
opened file size plus one EOF byte and refuses changed input identity.
Embedding outputs also reject nonempty writes into retained table storage;
empty-row and ordinary fresh-output behavior is preserved.

The public serving-profile integration must include `family_manifest` and its
`required_source_files()` closure in prepared profile identities. Structural
A-to-B-to-A changes must retire the old weights/workspace/readers before
replacement allocation. This source increment is not production deployment or
full-model/V100 performance evidence.

The required initial SC117 IQ3_S image/video profile and pinned processor are
separate acceptance work. This additional NVFP4 source candidate currently
refuses visual operators and does not establish the multimodal profile. Native
or extended RoPE/YaRN selections have distinct effective identities; larger
coordinate/context quality, MTP/draft positioning and complete resource gates
remain unqualified rather than inferred from a scaling factor.
