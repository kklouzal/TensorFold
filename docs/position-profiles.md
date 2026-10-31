# Flash Next positional profiles

The SM70 family resolves an immutable target attention/indexer policy and a
separate MTP policy. DFlash keeps its own full-width scalar RoPE, absolute
position limit and sliding window. A target YaRN override never supplies an
undeclared draft extension. This source increment does not qualify native
SC117 or any extended artifact/codec/method/media pairing.

New bound execution initially requires adaptive unique TP8 GPU histories with
two actual global KV heads. `--position-profile-manifest PATH` selects an
explicit policy there. A positive `--context` must match its declared rows;
omitted context or zero uses the file's declared allowance, without fitting it
to a shorter window. The ordinary unique native path resolves source defaults
without requiring this extra file. Legacy replicated native/CLI-YaRN and
DSpark paths keep their existing unbound behavior. An explicit new positional
manifest on those unsupported layouts/methods refuses before retention.

The input manifest has this structure (replace both hashes with the actual
selected configuration identities):

```json
{
  "schema_version": 1,
  "family": "qwen4_exp",
  "epoch": 0,
  "requested_context": 524288,
  "qualification": "unqualified-candidate",
  "target": {
    "source_config_sha256": "ACTUAL_TARGET_CONFIGURATION_SHA256",
    "rope": {
      "rope_type": "yarn", "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "mrope_section": [11, 11, 10], "mrope_interleaved": true,
      "original_max_position_embeddings": 262144, "factor": 2
    },
    "coordinate_limits": [524288, 524288, 524288]
  },
  "mtp": {
    "source_config_sha256": "ACTUAL_COMPANION_CONFIGURATION_SHA256",
    "rope": {
      "rope_type": "yarn", "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "mrope_section": [11, 11, 10], "mrope_interleaved": true,
      "original_max_position_embeddings": 262144, "factor": 2
    },
    "coordinate_limits": [524288, 524288, 524288]
  },
  "dflash2": null
}
```

This illustrates an unqualified 2x candidate, not an approved MTP pairing or
released context maximum. Null companion declarations retain their explicit
source-native defaults; the actual selected companion configuration still
participates in the input closure. Stock MTP configurations that omit positions
use the target's original source reference/theta/layout, with native factor and
amplitude, rather than inheriting a runtime target override. Any explicit
companion parameters remain authoritative. DFlash extensions require a
separately implemented draft recipe; its physical ring does not extend absolute
position support.

The public preparation API is
`tensorfold.families.qwen4_exp.position_profile.resolve_position_profile`.
It accepts `requested_context`, optional `yarn_factor`/`manifest_path`, and
`companion_configs` mapping actual MTP/DFlash source paths. Its resolved role
records expose the actual configuration SHA256 values, source input closure,
complete defaults, bounds and implementation identities. The returned receipt
is diagnostic JSON, distinct from the input manifest and from release evidence.
Raw manifest qualification claims cannot authorize execution.

Metadata resolution and module import need no accelerator runtime. Before model
or device retention, both Engine and structural-profile preparation must call
`prepare_position_vectors(profile, mtp_active=...)`. That check runs the existing
exact CPU `RopeParameters.inverse_frequencies` algorithm, including every FP32
YaRN intermediate, and refuses zero/nonfinite results. Finite scalar parameters
alone are insufficient: factor times a large FP32 frequency can overflow. Bind
requires the checked, unchanged prepared operands; it does not defer this failure
until the end of weight loading. Target/MTP share equal frequency formulas;
a distinct MTP vector is one small model-owned FP32 allocation, not a per-session
phase table. Its extra payload is charged once.

Rows and coordinates have separate units. Expanded input, retained context and
reply share the selected row allowance. Actual inclusive THW extrema must fit
each exclusive axis limit; many visual rows may share small THW coordinates.
Lookahead row counts do not shorten a THW domain or rewrite row allowance.
Padded rows are not semantic inputs. The trusted producer validates actual live
positions before integer conversion and binds proofs to their real storage,
version, request and resident model/profile identity. Callers cannot turn JSON
bounds into authority. DFlash validates its scalar absolute rows separately.

RoPE amplitude multiplies only rotary cosine/sine operands. It therefore scales
the rotary Q/K contribution by the amplitude squared, while nonrotary channels
stay unchanged. Target/indexer use the declared compatible transform and pooled
keys retain group-first coordinates. Native frequency construction preserves
the existing float64-power/float32-storage policy; static YaRN preserves the
pinned HF float32 construction. No global full-score scaling substitutes for it.

Frequencies and algorithms remain fixed for resident-state lifetime. A changed
transform/reference drains and retires affected state and graphs before
in-process activation; updating constants or rotating existing quantized keys
is not a state conversion. The Engine issues a fresh full resident activation
namespace before bound State construction/warmup. It provides isolation, not a
content or qualification certificate. Actual prepared content authorities are
bound when present. Bound text does not enable media HC/coordinate boundaries;
only a trusted admitted media envelope, encoder and resource plan can do so.

CPU/source/HF checks do not establish V100 quality, memory residency, acceptance,
throughput or latency. Those gates remain required for native and extended
points, both KV codecs, MTP, DFlash and required image/video continuation.
