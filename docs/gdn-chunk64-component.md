# Explicit GDN chunk64 numerical trial

The maintained `tools/v100_gdn_native_envelope.py` runs the numerical helper
with an explicitly hashed, bounded DATA selection. It checks the actual GPU
UUID, SM70 capability, CPU affinity and installed component hashes, records
resource snapshots, and retains a bounded compiler cache before process exit.
Snapshot, retention and publication failures are transported with the original
kernel failure. This replaces the private Source27 envelope for future trials;
that failed historical packet is preserved. CPU envelope controls and actual
native/component/model qualification remain distinct.
Initial-state preservation uses FP32 storage-bit equality through both the
continuation and alias-refusal controls; value equality alone misses signed
zero changes. Nonzero-state cases include negative zero as an adverse input.
The serial oracle consumes the saved pre-launch initial bank. Observed
preservation failures remain sticky through later alias-refusal checks.
The selected TileLang0.1.10 compile API receives the explicit target dictionary
`{'kind': 'cuda', 'arch': 'sm_70'}`. The legacy target string was rejected at the
first native preparation before GDN compilation; that failure is retained.
The corrected Source36 run completed the 18 main native cases and both split
continuations. Outputs/final states were finite and initial storage bits stayed
unchanged. Maximum observed absolute errors against the serial F64 oracle were
`1.1128783187982355e-5` for output and `3.8669565944969564e-5` for final state;
these observations establish no accepted error tolerance or model accuracy.
Its nondefault-stream diagnostic stopped before profiling because the symbol
parser counted one forward declaration and definition twice. The corrected
parser requires one unique entry; the actual stream proof remains a native gate.

The helper defaults to `--phase full`, retaining all18 main cases and both
continuations before the stream diagnostic. `--phase nondefault-stream` runs
the same stream diagnostic alone and emits `phase: nondefault-stream`,
`main_numerical_cases: unrun` and an empty `results` list. It claims no main
numerical pass. Complete component evidence requires the separately retained,
source-bound main-case and split results as well as an actual stream proof;
a stream-only process exit cannot replace either numerical or quality gates.

The private candidate in `tensorfold.cuda.sm70.gdn_chunk64` owns a minimal
prefill component around three original FlashQLA SM70 TileLang generators.
It has no engine hook, CLI serving flag, automatic selection or dependency
installation. Source implementation is separate from protected-SDK compilation,
native numerical validation and model/performance qualification; these are
pending. Decode, MTP verification, native chain and Value8 retain their current
implementations.

The original algorithm computes chunk64 log-gate prefix sums, a KKT/WY
triangular solve and chunk state/output updates through Volta-oriented GEMMs.
It differs from the existing scalar recurrence: FP16 q/k/v/A and shared state,
FP32 accumulation/final state, exp2 fast math and block reassociation are a new
numerical mode. No bit identity, tolerance, quality equivalence or speedup is
claimed. Only the upstream `T.gemm_v1` spelling becomes pinned TileLang0.1.10
`T.gemm`; the upstream initializer makes this same alias globally. The owned
module avoids changing the installed language module. The imported MIT source
and exact original/adapted version/extraction identities are
in `gdn_chunk64_LICENSE.txt` and `gdn_chunk64_provenance.json`.

## Boundary and ownership

`layout(rows, cuts=())` specifies exact tensor scratch credit: `792*rows+21`
bytes. Runtime/compiler/allocator retention is separate and must be contained
and measured in the component process. Rows are positive int32; only one
sequence, Hq2/HV6, D128, 128 threads is supported. Partial64-token tails retain
the original masked kernel. Nonempty interior cuts are refused before any
allocation or mutation. Graph capture is refused by this first component;
no unverified replay/version-counter guarantee is implied.

`prepare(rows, device, workspace_budget_bytes=..., compiler=Compiler(...))`
is cold, outside capture. It requires the protected CUDA12.9 nvcc path/hash,
preconfigured CUDA_HOME/PATH and TL_CUTLASS_PATH selecting the protected
CUTLASS4.8.1 include root, an explicitly attested owned TileLang artifact,
verified protected FFI0.1.16 and
preserved Triton3.2.0.
It records and checks installed TileLang base requirements without installing
anything. It does not import vLLM/FlashQLA, change global compiler callbacks,
patch installed libraries or select pip CUDA headers. The selected foundation header manifest is verified. Public
`get_tir`/`tilelang.compile` explicitly select SM70 and the TVM-FFI backend;
its NVCC callback takes the selected CCCL3.5 include before implicit toolkit
headers. This supported configuration changes no callbacks. Actual compilation
and generated-header evidence remain admission gates. Run this optional compiler in a dedicated component process, never the
live server. Do not install the upstream full requirements stack.

`prepared.bind(q,k,v,raw_log_gate,beta,initial,output,final)` validates one
fixed set of contiguous, no-grad CUDA tensors and disjoint write storage.
q/k are normalized **unscaled** FP16 `[1,R,2,128]`, v/output FP16
`[1,R,6,128]`, raw log gates and beta FP32 `[1,R,6]`, initial/final FP32 VLK
`[1,6,128,128]`. q scaling is applied once inside the core. The trusted producer
must supply finite nonpositive raw log gates directly; `log(saved_exp_gate)`
loses information after underflow and is not a valid adapter. Beta rounding
must be explicitly retained or qualified by that producer's numerical contract.
Any arbitrary FP32 initial state is read from its own bank; final state is
written separately. The core does not change TF conv history, select current
banks, preserve interior cuts or publish speculative state.

`bound.run()` is inference-only, serial on the preparation CPU thread/CUDA
stream. It enqueues three kernels with no tensor allocations or compiler calls and
returns the caller output/final. Its supported stream scope creates a small
Python context object. It updates PyTorch mutation versions, retains
all tensor/kernel owners, and does not synchronize. Caller must not resize,
replace or mutate storage ownership until stream completion; bind again after
replacement. On CUDA failure the run is poisoned and must be contained without
publishing final state. Inputs may be refreshed by preceding work on the same
stream. There is no concurrent workspace reuse or multi-stream contract.

## Required validation and finite adoption gates

The focused maintained CPU boundary checks are `tests/test_sm70_gdn_chunk64.py`.
The explicit GPU trial `tools/v100_gdn_chunk64_trial.py` compares component
outputs/final states with an independent CPU float64 serial delta rule at
rows1/63/64/65/128/129, zero and nonzero initial states, and raw gates whose exp
underflows. It checks immutable initial-bank ownership, alias refusal,
finite/full outputs and exact zero known answers. The nonzero-initial controls
retain separate-bank continuations at128=65+63 and129=65+64. Numerical errors are
reported without an invented tolerance; a zero process exit does not qualify
the numerical mode. Use only a coordinator-granted GPU/resource lease and keep
all failures. Compiler failure produces a structured failure report.

Before integration, establish a guarded TF frontend exposing raw gates and
preserving conv/norm/scale/beta behavior; retain current-bank ownership and
saved prefix/cut/conv-tail semantics or keep refusing unsupported cuts. Actual
protected-toolchain generated instructions/shared memory/retention, native
state continuity and graph behavior need evidence. Model output/PPL/MTP
acceptance and controlled end-to-end prefill/decode comparisons under required
TP8/ETP4/context/capacity remain required. No different-region upstream
whole-model TPS observation qualifies this component.

## Candidate ledger

- Objective/region: prefill-only component discovery for required V100 TP8
  Hq2/HV6/D128, before any engine selection. The named oracle is the independent
  float64 serial recurrence on the component's actual inputs. Existing TF
  scalar/native recurrence remains the admissible serving baseline.
- Candidate: exact source-scoped original chunk64 algorithm, new numerical
  trial. Status: source implemented; unproven, not adopted.
- Hard invariants: protected CUDA12.9/Triton3.2; distinct FP32 VLK initial/final;
  no committed-bank mutation, no hidden cuts/conv publication, no log(exp),
  fixed workspace credit, single-owner/thread/stream lifetime.
- Decisive integration blockers: guarded raw-gate/normalization frontend,
  retained-cut/conv-tail state contracts, dependency/compiler admissibility,
  authorized numerical bounds and actual native/model results.
- Verification schedule: implementation precedes tests. Source23 paired
  measured windows prohibit heavyweight CPU compilation/tests; coordinator
  leases and Root GPU authorization are required before those actions.
  Focused CPU checks passed14 tests in2.42s using image2cad with the isolated
  source overlay,4CPU/8GiB/node1/runc/NVIDIAvoid/netnone. Ruff for all four
  affected Python files, AST and diff checks pass. These are cold boundary and
  independent-reference checks. D13 source review cleared the standalone
  unqualified checkpoint, including resolved failure-evidence and pinned-API
  findings; no reviewer build/GPU tests. Compile output and native errors
  remain pending. All CPU check containers were inspected and removed.
- Finite stop: component source plus focused resource/state/numerical evidence;
  no full engine hook, no automatic dependency changes, no serving adoption.

The exact optional TileLang0.1.10 CPython Linux x86_64 wheel inspected for this
boundary has published SHA256
`85180d1a96defeecdf52d5d075a31c3fc551d8485981e6b636762a9cd7eb02fe`.
Its base metadata requires FFI `>=0.1.10,~=0.1.0`, so the owned FFI0.1.16 is
metadata-admissible; it does not require Triton or CUDA13. The optional `nvcc`
extra requires CUDA13 and is outside this trial. Tensor/FFI integration with the
owned Torch2.16 developer build still requires actual component execution.

## Owned artifact admission

The unpatched public TileLang0.1.10 with the protected FFI0.1.16 actually aborted
while importing its embedded TVM reflection registrations. Metadata dependency
compatibility did not establish runtime compatibility. `Compiler.inspect()`
now refuses that unattested tuple and every other unattested version before
TileLang import. The existing two compiler arguments and trial invocation
surface remain; there is no automatic version-prefix admission or FFI downgrade.
An actual owned wheel/version is not yet established by this source change.

`Compiler` accepts an explicit `OwnedArtifact` selection: immutable attestation
path, exact metadata byte count/SHA256, retained owned-wheel path and aggregate
hash-byte credit. The metadata envelope is at most16MiB. One cold inspection
hashes the wheel and all declared owned/FFI installed members; the caller funds
their measured sum, separately from metadata, existing SDK/header inspection
and compiler/runtime/allocator memory. No guessed wheel size or version is
supplied. Package contents exclude wheel `RECORD` and runtime bytecode; package
extra/missing files and symlinks are refused. The exact protected FFI wheel,
256-member inventory and core remain fixed to the observed0.1.16 foundation.

The closed schema `tensorfold.g3-owned-tilelang-attestation-v1` has exactly:

- `source_pins`: actual TileLang, TVM, build-header FFI, DLPack, libbacktrace,
  CUTLASS and composable-kernel commits.
- `build`: actual proposal/archive SHA256 and successful build image ID, observed
  externally by the build controller.
- `tilelang` and `protected_ffi`: exact metadata version, wheel bytes/SHA256 and
  complete relative installed-member bytes/SHA256.
- `compiler_environment`: the existing verified `cuda_home`, `nvcc_sha256` and
  selected-header descriptor, obtained without importing TileLang through
  `Compiler.environment()`.
- `evidence`: exact maintained producer hash and the successful postbuild audit,
  TileLang-first and FFI/Torch-first receipt identities.
- `schema` and `qualification: artifact-identity-only`.

Attestation `tensorfold.g3-owned-tilelang-attestation-v2` retains those eight
fields and adds one closed `derivation` object. Top-level `source_pins` and
`build` continue to identify the native708bc2d/a8d1a61 build and its archive;
`tilelang` and the audit/import `evidence` identify the resulting derived
artifact. The derivation records the complete native base wheel inventory,
reviewed Python07827c6/772f65b pins, the exact approved DictAttrs before/after
member, packaging proposal/producer/tool/recipe-input/image identities and
the embedded provenance member. It permits this one reviewed change, not an
arbitrary Python patch or unknown source tree.
The registered derived-provenance member cannot be admitted under v1 by
dropping the derivation field; valid single-build v1 selections remain unchanged.

V2 requires the explicit retained `--owned-native-base-wheel` in addition to
the five existing selection flags. V1 rejects that unused extra selection.
Both retained wheels and all declared archive member payloads are hashed;
normalized dist-info renaming must preserve every unaffected/native member.
METADATA may change only Version; WHEEL may change only Generator to the
reviewed public-wheel0.48.0 derivation label. Both RECORD files must cover the
actual complete member identities. The sole added provenance JSON must match
the selected native/Python/packager/recipe chain exactly, excluding its own
digest, resulting wheel hash and image so that the identity is noncircular.

V2 SHA256 credit additionally covers the base wheel and both decompressed
member inventories. Each RECORD is separately bounded metadata of at most16MiB;
CRC work and ZIP-library IO are not claimed as SHA256 byte credit. The dedicated
process's host-memory, wall-time and cache grants remain required. Structural
normalization and an ordinary import report do not prove actual bundled-TVM
origin: the consumer's existing Python/native mapping checks still run directly
after import and before compilation. Actual attestation publication waits for
the successful derived audit and both fresh import-order receipts; a future
matching application image is required for v2 admission.

An external SHA256 selection is administrator authority, not a signature or
proof of numerical quality. The producer must first establish the actual
successful build, all three postbuild outcomes, immutable sources and positive
retirement. The maintained trial helper's `attest-owned` mode accepts the bounded
immutable candidate DATA through `--input`, `--input-bytes`, `--input-sha256`
and a new `--output`. It uses the same closed parser and requires its own actual
source hash in `evidence.producer`; it performs structural normalization only,
preserves existing files, and reports the minimum hashing credit and derived
cache namespace. It does not manufacture or validate successful import receipts.

The native trial adds five paired flags:
`--owned-tilelang-attestation`, `--owned-attestation-bytes`,
`--owned-attestation-sha256`, `--owned-tilelang-wheel` and
`--owned-hash-byte-credit`. Their absence produces a clean identity-admission
failure before operand allocation or TileLang import. The launcher must
preselect `TILELANG_CACHE_DIR` with the reported identity namespace as its leaf;
an explicit `TILELANG_TMP_DIR` must stay inside that namespace. The key binds
the complete attestation and actual compiler/header descriptor. Code does not
rewrite environment variables or callbacks. Actual imported compiler/header/
cache settings are checked again before compilation. Both package descendants
and the top-level bundled TVM must have verified Python origins. TVM Python and
library-path overrides may select only the attested trees. Existing foreign
TileLang/TVM/FFI mappings fail before TileLang import; compilation additionally
requires the exact mapped TileLang, TVM runtime/compiler and protected FFI
libraries. Installed packages, wheel and cache ownership remain
immutable/exclusive for the dedicated component lifetime.

The compiler report retains the admitted identity, source/build provenance,
hashing credit and cache namespace. Controlled CPU fixtures exercise this
contract without qualifying a real owned wheel. Changes to the application
consumer require a matching derived image/source gate; old installed code or
the failed upstream purpose image cannot qualify this new path. Actual owned
artifact admission, generated SM70 code, stream/math/state and model/quality/P3
gates remain pending.

The focused source-overlay gate passed61 controlled CPU cases in0.70s on the
genuine Source30 SDK/application image `0ad2d11d`, with its installed FD
extension, UID1000,2CPU/4GiB, runc/VOID and no network. TileLang remained
unimported and CUDA unavailable. Scoped Ruff and whitespace checks pass;
source hashes held and the check containers were inspected and
removed. These fixtures validate the identity/credit/origin/error boundaries;
they are not an actual owned artifact or compiler/native result.
The bundled CUTLASS4.1.0 is not adopted: the configured selected4.8.1 headers
must compile successfully. No dependency lock or shipping image is changed.

Cold compilation makes exactly three public `tilelang.compile` calls without
an autotuner or optional variants. It must execute in a coordinator-bounded
component process with predeclared wall-time, memory/CPU and cache-disk grants;
TileLang's compiler/library allocation is not bounded by the tensor-credit
number. Retain the process's actual compile logs and failure status. Native
trial authorization must supply those grants; this checkpoint requests none
implicitly and has not run a compiler.

## Stream assurance increment

The pinned TileLang TVM-FFI adapter describes current-stream thunks in prose,
but its actual `_convert_torch_func` comments out that thunk and calls the
runtime executable directly. The CUDA module launches with
`TVMFFIEnvGetStream`. Torch2.16's C-DLPack exchange API exposes its current
stream, but that alone is not proof that this adapter installs it. The owned
boundary explicitly enters the supported FFI0.1.16 `use_raw_stream` scope for
the admitted Torch stream and restores the prior FFI context on every exit.
This supports the caller's original nondefault stream; no default-only
restriction or global compiler/runtime callback is introduced.

Focused CPU scope/restoration controls are added for success and launch failure;
they are not actual device-stream evidence. After Source24 retirement, the
coordinator-granted CPU-only source-overlay check passed18 tests in2.06s,
including all four scope/restoration cases; Ruff for all four affected Python
files and syntax/whitespace checks passed. The check container retired and was
removed. No compiler or GPU job ran. The native procedure now requires a distinct
`--stream-trace-output`: create a nondefault Torch stream, prepare/bind there,
enqueue a producer, G3 and a consumer there, wait its completion event and check
the exact zero known answer. The first profiler initialization requests CPU+CUDA.
Native generated entry names must match exactly three recorded G3 kernel events;
additional Torch control kernels must share the same recorded GPU stream and
device, otherwise stream coverage remains unproven. Preserve the raw closed
trace/hash and every failure; no trace header or CPU annotation suffices.
Actual compiler and GPU execution are still pending separate leases.

The [prefill integration design](gdn-chunk64-prefill-integration.md) defines the
raw-log/qk/beta producer, epilogue, row-capacity and prefix-cut/alternate-bank
work still needed before an ordinary serving hook.

The earlier stream admission explicitly required `apache-tvm-ffi==0.1.16` before
using its stream API. TileLang's broader base dependency floor establishes
package compatibility, not a verified stream contract for every allowed FFI
version. Other FFI versions are refused; no SDK patch or dependency install is
performed by this guard. At that checkpoint a mocked-toolkit control checked rejection of
0.1.10/0.1.15/0.1.17/0.2.0 and progression to remaining dependency inspection
for0.1.16; it did not establish actual compiler admission. The owned-artifact
guard above now additionally requires complete explicit identity evidence
before that dependency inspection can continue.

The five new version controls passed in0.37s in the genuine Source25 SDK image
with the isolated source overlay (UID1000,2CPU/4GiB,Node1,runc/NVIDIAvoid/netnone);
18 prior unaffected checks were deselected and reused. Ruff for the three
checked Python files, syntax and whitespace checks pass. No installed-source
qualification, compiler execution, SDK hashing or GPU result is implied.
