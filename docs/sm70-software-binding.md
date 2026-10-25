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
resource envelope or acceptance. TorchVision/PyAV/processor provisioning is a
separate pending increment. Profile/evidence promotion is owned by the runtime;
software binding does not clear compiler issue 6111901.
