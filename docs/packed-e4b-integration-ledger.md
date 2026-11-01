# Packed E4B integration checkpoint

Worktree `TensorFold-packed-integration`, branch `gb10-e4b-packed-integration`,
base `761342ed`. Root owns integration back into canonical python-0.6; unrelated
BF16 optimization WIP is excluded. Scope: current ModelOpt NVFP4 W4A4 E4B storage
only, existing Engine/Target/cache/media/assistant contracts, BF16 behavior kept.

Selected structural implementation: reuse dense_loader, no second inference
engine, no BF16 projection facade and no full dequantized mirror. Format/shape/
manifest preparation is repeated by the loader at byte-consumption lifetime.
Admission uses explicit prepared NVFP4 storage and original CPU placeholder
owners, with no reuse/page-commit credit. Unknown FP8/transformed/KV layouts and
incompatible projection overrides are refused. Terminal loader retirement holds
all previous owners and BLAS lease; no CUDA retry or release. Packed constructor
failure after publication drains before release or retains terminal ownership.

Evidence at this checkpoint: source implementation only. Parent's earlier
private loader smoke is not an Engine/Target or serving qualification. CPU checks
and dedicated D13/D14 review are pending; no GPU/model/container/build executed by
this owner. Actual same-packed width1..8, greedy/drafted, sampled replay, media,
131K capacity, HTTP and BF16 quality comparisons remain required. Performance
selection uses decode first, prefill second and memory third, with matched
complete-workload evidence; no performance or equivalence claim is made here.

## CPU checkpoint

The first original2a source-overlay run recorded443 passes,18 skips and12
failures: two new test mocks used the wrong positional Geometry argument, and
10 inherited reader paths lacked the source worktree's compiled `_fd_owner`.
No parser fallback or weakened assertion was introduced. Corrected the test
index and built the existing owned C extension via `setup.py build_ext --inplace`
in original2a with its managed compiler, network disabled, no GPU,4GiB RAM/no
swap, quota2 on host-managed CPUs5-9,15-19. The same17-file suite then passed
**456 tests with17 optional processor skips**, CUDA uninitialized. Build changed
no tracked source; C SHA `f85d2c923adfaa14e819250e8def3c3166b8c8624295a37572916e62ad20c1f2`.
The inherited setuptools license deprecations are packaging advisories outside
this storage integration; no compiler warning/error occurred. Changed Python
scope Ruff and diff whitespace checks pass. Exact commands, full logs, selected
tests and binary receipt are retained by the primary under
`integration-preparation/packed-integration-cpu-v{1,2}.*`.

The reader overlap bound includes two64MiB host owners plus8192 alignment bytes:
`Reader._to_host` can clone a misaligned view while its aligned input lives.
No allocator-reuse credit is inferred. GPU/model/HTTP/quality/performance checks
and D13/D14 source review remain pending at this CPU checkpoint.

## Bounded review remediation (CPU rerun pending)

D13/D14 identified two resource gaps: native media topology could exceed old
header sizing before CPU construction, and host-loading-only estimates did not
activate generic UMA cgroup admission. Preparation now pins the original E4B
media allocation profile before constructor loops and uses a native META shape
model to validate complete headers via the same validator used at real binding.
Actual nonpersistent shapes and native CPU placeholder bytes drive estimates;
META values are never used to initialize the real model. An explicit existing
`admit_host_commitment` check validates current physical/cgroup allowance before
Target allocation without adding a second UMA charge. The dead packed header
rescan in the existing admission function was removed; BF16 flow is unchanged.

The exception retained-owner marker is not a reliable negative retirement proof
when annotation itself exhausts. A caller-preallocated `DenseLoadStatus` now
records unsafe device binding and proved retirement/ownership transfer, and
Target holds its BLAS lease if safe cleanup is unproved even without a marker.
Needed tests cover actual loader status and marker-absent terminal handling,
media growth before constructors, missing native media inventory and cgroup
refusal before device planning. Source scope Ruff/diff checks pass; these
remediations await a new granted CPU run and reviewer recheck.

## Remediation CPU checkpoint

The same17-file suite on original2a passed **464 tests with17 optional-media
skips**, CUDA uninitialized, no warnings. All actual parameters/nonpersistent
buffers from the allocation-free HF5.17 topology model were META; complete native
schema, altered-media preconstructor refusal, actual current-cgroup admission,
no duplicate UMA charge, marker-absent unsafe load ownership and transport-OOM
primary/native-root preservation controls passed. Source hashes were unchanged
during execution. The strong model/source slots are preallocated in load status
and cleared only after successful transfer or proved retirement; exhaustion
before rollback tuple/call retains actual owners without a CUDA retry. Source
Ruff/whitespace checks pass. Exact v3 command/log/hash receipt is retained under
`integration-preparation/packed-integration-cpu-v3.*` by the primary. This does
not qualify actual integrated GPU arithmetic, media, serving or performance.
