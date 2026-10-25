# SM70 process software binding

`tensorfold.cuda.sm70.software_binding` binds the administrator-owned source
foundation to the current process. It invokes the same protected
`/opt/tensorfold-pins/source_foundation.py::audit_process` implementation;
TensorFold does not contain a second audit algorithm.

```python
from tensorfold.cuda.sm70.software_binding import issue_source_binding, verify_source_binding

authority = issue_source_binding(required=True)  # after supported preparation/warmup
software_record = verify_source_binding(authority)  # before profile promotion
```

The fixed protected inputs are `v100-source-foundation.json`, `source-pins.json`
and `source_foundation.py` under `/opt/tensorfold-pins`. They must be complete,
bounded, administrator-owned regular files in the immutable protected hierarchy.
The producer's exact bytes must match its sealed artifact before execution.
The source-image producer seals these administrator metadata files to mode 0444; a
source candidate copied with group-writable metadata cannot issue authority
until its deployment recipe establishes this immutable hierarchy.
Issuance binds their raw digests, the process audit/runtime digests, current PID
and actual immutable artifact/process-input file identities. Serialized records
cannot recreate an authority. A fork cannot reuse its parent's authority.

Only absence of all three protected files permits
`issue_source_binding(required=False)` to return explicit unverified-reference
status. Partial, malformed, changed or failed-audit inputs raise. Reference
status cannot pass `verify_source_binding`.

Metadata file retirement reuses the native checkpoint-file stream owner. An
unretired close preserves `_tensorfold_checkpoint_file_scope` on the original
failure. Promotion callers must apply the existing retained-owner fatal/containment
classification rather than treating that failure as a safely retired profile
refusal; there is no reference downgrade or hidden close retry here.

Verification is an explicit preparation/promotion operation, never a per-request
or kernel-launch check. It reuses the process-issued object only while its actual
input identities remain unchanged and reruns the protected process audit. Changed
files, loaded runtime or PID invalidate reuse; the caller must explicitly issue
again after warmup rather than silently rebinding the old authority. Callers
privately own the authority and invoke these operations at their coordinated
state boundary. `record()` is observational and does not replace verification.

Records remain software-only with `release_qualified=false` and pending
compiler/device/model/media gates. They preserve the source specification and
component identities actually recorded by the image. Existing text-scope source
artifacts do not qualify the revised SC117 image/video target, processor stack,
resource envelope or acceptance. Media dependency software has a separate
combined authority below. Profile/evidence promotion is owned by the runtime;
software binding does not clear compiler issue 6111901.

## Required source and media software together

`tensorfold.cuda.sm70.media_software_binding` joins that source authority with
the same protected media producer's `audit_install` implementation. A source-only
authority or a serialized software record cannot satisfy the combined verifier.

```python
from tensorfold.cuda.sm70.media_software_binding import (
    issue_media_binding, reissue_media_binding, verify_media_binding,
)

authority = issue_media_binding()  # coordinated load/preparation; no reference downgrade
record = verify_media_binding(authority)  # before adopting/replacing a vision dependency scope
# After supported model/JIT warmup adds process code, explicitly observe new state:
authority = reissue_media_binding(authority)
record = verify_media_binding(authority)
```

Issuance requires the fixed producer
`/opt/tensorfold-pins/media_foundation.py` and `media-install.json`,
`media-pins.json`, `media-build.json`, `preprocessor_config.json`, and
`video_preprocessor_config.json` under `/opt/tensorfold-media-pins`. Missing,
partial, aliased, writable, malformed or inconsistent metadata refuses. Producer
bytes match both the sealed installation artifact and completed-build inputs;
the build's source-producer identity must match the actual source preparation.
The media producer receives that privately verified source helper module without
replacing or trusting the process's same-name `sys.modules` entry.

The protected media audit checks added package/native/configuration bytes and
materializes its declared CPU metadata/providers before final source-process
issuance. Both protected directory hierarchies and all immutable input identities
are checked around the operation. Metadata uses the existing descriptor-bound
native stream owner, retaining primary failures and any unretired file scope.
No target/vision weights are loaded by this software adapter.

`verify_media_binding` checks the exact issued type/issuer/PID before filesystem
or producer work, then verifies both source and media audits. It never repairs a
stale object. `reissue_media_binding` also checks the old immutable anchor first:
package, provider, metadata or configuration file changes cannot be repaired by
reissuing. Only supported new warm process observations with unchanged original
files can create a replacement authority. Previously observed files remain in
that immutable anchor even if their process mappings disappear. The cumulative
anchor is bounded by three existing 100,000-entry inventory namespaces (source,
media and process observations); exhaustion refuses rather than evicting an old
identity. The caller must retain/quiesce its
scope at these preparation/adoption boundaries. These calls perform file audits
and must never be added to a request or kernel-launch path; `record()` only
returns a detached observational copy.

Combined records retain all inherited pending gates and add explicit
service/media/SC117/position/YaRN/device/resource/quality/performance gates. The
verified CPU software dependency list does not clear the broader inherited
image/video stack gate. They remain `software_only=true` and
`release_qualified=false`: source/processor/decoder dependency verification alone
does not qualify a vision model, service media policy or release. Production
vision callers must adopt this exact combined authority, with their remaining
model/device/quality/resource gates checked separately.
