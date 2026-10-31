# Producer integration checkpoint

Latest verified stage: owned runtime fork `a851b028` produced wheel
`0.48.0rc1.dev268+ga851b028e`, SHA256
`1c99f1a5b0109104a97bdebefcc78fa496ea5a1b59d3d71ef1498759490723db`.
Producer image
`sha256:d6b54ef6eb5f8e98d29ad4e2f733a04b23285c170e02e30b0dc58de648aff86d`
passed CPU API, dependency, namespace, seal and original-provider admission.
On GB10 SM121, its guarded synthetic Local-Hessian calibration passed four
batches and 64 activation rows per pass, including the native Hessian weight sweep and
the receipt-bound FP8 helper. This establishes that small execution path,
not full-model calibration, quality, throughput or peak-memory qualification.

The focused native GPU regression attempt collected all 39 cases: 36 passed
and three new ragged-layout cases failed their static-versus-dynamic comparison.
A subsequent guarded diagnostic completed all 12 combinations of two input
generator precisions, three output precisions and two ragged shapes. Dynamic
results were bitwise identical across contiguous, padded, singleton-row and
three tile layouts, and both paths matched their respective adopted helpers.
The mismatch was isolated to a one-ULP block-scale difference crossing an FP4
rounding boundary, rather than an addressing discrepancy. The original failure
and complete diagnostic remain retained in the host task ledger.

Local test-only fork checkpoint `f8a02ef4` corrects that new test's false
cross-path equivalence assumption. It compares independently gathered aligned
1-by-16 blocks using the same dynamic quantizer and global scale, with exact
equality and explicit FP32/BF16 generators. All production bytes and earlier
tests remain unchanged. The corrected 39-case native GPU rerun passed on test
image `a0b914aa`, with no skipped, retried or expected-failure cases; outer
producer, original-provider, resource-guard and retirement checks also passed.
The run took 1.43 seconds inside pytest, which is a regression-test duration,
not an inference performance result. The runtime producer remains pinned to
`a851b028`. No calibrated E4B artifact
or quantized E4B serving/performance result exists at this checkpoint.

The first full E4B attempt completed both 1,024-batch passes and all 258
Local-Hessian weight refinements, then failed while hashing the recipe for its
receipt: raw ModelOpt block maps mix integer axes with named string attributes,
which Python's sorted JSON encoder cannot compare. No export was started and
no OOM event occurred; the E2B service was restored. The coordinator now uses
ModelOpt's public `QuantizeConfig.model_dump(mode="json")` representation and
computes its digest before loading weights. An actual D6 CPU preflight
reproduced the original failure, serialized the complete 258-projection recipe
without changing its numerical configuration, and passed final producer audits.
The corrected focused CPU suite passed 254 checks. A short 64-sample conversion
diagnostic precedes another full 1,024-sample attempt; it is not a substitute
for final calibration or quality/performance qualification.

Historical verified stage: the isolated Git/build-tools image
`sha256:ee54ef39fa38f8fff5cdde2e9948d1d75bc8f5fe05f1ba428c68c5ca2fc28877`
passed its original SDK, added OS/Python seals and CPU backend smoke. The public
Setuptools backend built the clean pinned ModelOpt checkout as
`0.48.0rc1.dev267+ge352caca3`, wheel SHA256
`1e2ebe573fb81882adefb6dc257578200cc75b220e5c323ff7b2f6c522783d9b`.
This was an ARM64 CPU-only, network-isolated build with 4 GiB/no extra swap,
requested CPU affinity 0,1 and a two-CPU quota. Source tracked files remained unchanged;
the original base image ID remained unchanged. No SCM override was used.
The Git stage binds the inherited dpkg exclusion policy and permits only its
declared nonruntime documentation omissions; required files remain mandatory.
Its expanded boundary suite passed 51 controls. The failed earlier image,
which incorrectly expected excluded documentation, remains a failed result.

The real native resolver selected six additions against all unchanged baseline
versions: the source-built ModelOpt wheel, nvidia-ml-py13.615.71,
OmegaConf2.4.0, PuLP3.3.2, Ninja1.13.2 and SciPy1.18.1. All artifacts matched
the report's SHA256 values and the addition-only manifest/lock generator passed.
The producer image
`sha256:91bc4fdbedbd7d00d212e0b8d1d713d65d2a845de3f3e40decc5c99b730033ed`
passed actual CPU quantize/export imports, all four required recipe checks,
combined dependency validation, independent seals, final namespace validation
and original provider audits. CUDA remained uninitialized; 19 mapped provider
records were verified. Git and its OS role are absent from this producer.
The first producer build failed because the owned smoke script changed
`sys.path`; the correction reuses the already activated runtime helper and
retains the namespace check. That failed result is preserved separately.
The CPU-only CUDA-runtime warning and upstream TorchScript deprecation warning
were retained; neither grants CUDA/JIT execution qualification or requires a
vendor patch. Calibration, model export, GPU and quality gates remain pending.
The FP8 scaling helper subsequently built and loaded against the same managed
SDK, using its original NVIDIA source, C++20 and SM121 code generation. Source
SHA256 is `90de92d83c61f4f36167d4a1d7928f99c41d4154b490645c2dcf1c1ddd17c8bf`;
binary SHA256 is `85b78c49d487800dfb079415358f35588e672c8e50092b3ae855ac05b349c842`.
Final role/provider/seal checks passed, CUDA stayed uninitialized, and the CPU
cgroup peaked at 3,582,009,344 bytes below its 4 GiB limit, with no OOM event.
Actual affinity was `5-9,15-19`, enforced by the pre-existing host Docker policy;
the two-CPU quota remained active. Earlier affinity values here describe requested
flags, not verified effective placement. These functional checks establish no
CPU or GPU performance gain. The first helper attempt stopped before compilation
because its cache expectation ignored the protected startup mapping; that failure
is retained. The successful attempt preserved that mapping and isolated only
the host cache backing. CUDA numerical qualification remains pending.
The historical checkpoints below retain their original validation limits.

Objective: containerized, highest-quality NVFP4 calibration preparation while
preserving the current BF16 serving/runtime foundation. This first increment
covers dependency isolation and CPU producer API admission, not calibration,
approximation quality, model export compatibility or serving selection.

Fixed inputs: ARM64 CPython3.12 verification base
`2a772b064f593ae9a4436562c4c26ceaa60cec55709992cbf966b74b16bc649c`;
ModelOpt commit `e352caca3c36236fc15adea2627a9d42788380c4`.
Source: [pinned upstream metadata](https://github.com/NVIDIA/Model-Optimizer/blob/e352caca3c36236fc15adea2627a9d42788380c4/pyproject.toml),
[pinned import contract](https://github.com/NVIDIA/Model-Optimizer/blob/e352caca3c36236fc15adea2627a9d42788380c4/modelopt/torch/__init__.py).
Version ranges admit Torch2.16 and Transformers5.17; this is a necessary metadata
condition, not proof of the actual combined stack.

Chosen candidate architecture: separately sealed additive producer prefix;
original foundation guards/policies/manifests/providers remain unchanged.
D13 and D14 preliminary design reviews agree this is legitimate conditional on
concrete closure/content/provider validation. No production policy exception
or FA4-style unsealed diagnostic scope is used.

Implemented: baseline all-version constraints and distribution payload hashes;
actual native resolver report -> addition-only wheel-hash lock; fixed-point
dependency/extra validation; overlap/activation/provider checks; extension file
seal; combined public `pip check`; original managed CPU/GPU launch requirements;
zero-CUDA API smoke. Installation is offline/no-index and materializes an already
validated full solution. D13 caught and source remediation fixed a local-variable
shadow in audit; a successful-audit control was added before any execution.

Verified CPU boundary checks: `tests/test_calibration_role.py`, 11 passed in
0.08 seconds, container exit0 on the exact base above. Execution used runc,
networknone, no NVIDIA devices, memory4GiB/no swap, pids128, oom-score1000,
requested CPUs0,1 (effective placement not captured for that run), read-only
canonical source. Command after the CPU guard was
`python3 -m pytest --noconftest -o pythonpath=tests -p no:cacheprovider -q
tests/test_calibration_role.py` with plugin autoload disabled; no unrelated
conftest/plugins, Torch import, package installation, compiler or GPU workload.
AST syntax parsing of all three Python files and `git diff --check` passed.
These are boundary controls, not a successful ModelOpt import or image build.

Subsequent D13 source findings were remediated: final audits now run
independently and preserve all cleanup/primary failures; executable tools and
requirements lock join the independent seal; provider checks require actual
same-process bootstrap ownership/profile identity before and after every
consumer; ELF checks reuse installed sealed helpers without path mutation;
literal consumer arguments use run-only subparser remainder. A parser control
also prevents preparation flags from being swallowed as consumer arguments.
The expanded 16-control CPU rerun passed0.08s/container exit0 with the same bounded
configuration. The subsequent explicit build-tools kind passed19 controls in
0.08s/exit0; Draft's offered unused brief slot was released immediately. Full
main() CLI integration controls bring the role suite to20; four additional
source-builder boundary controls cover pinned Git, forbidden SCM overrides,
backend-supplied wheel version/hash and unsatisfied dynamic build dependencies.
The subsequent real Python3.12 metadata check found invalid explicit
`md.distributions(path=None)` use, then repeated identical search-path metadata.
Both were fixed with default-argument omission and deduplication of identical
resolved METADATA file origins only; distinct installations remain fatal.
All26 controls passed1.23s/container exit0. Full CLI control uses actual native
stdlib metadata and substitutes only expensive payload hashing. Root preserves
the failed snapshot and immutable source/test receipts. No result for an earlier
revision is represented as successor test execution.

Bootstrap source-only increment: same sealed role, explicit build-tools kind
with setuptools-scm>=8/vcs-versioning allowlist, no ModelOpt-wheel prerequisite.
Actual root resolver selects10.3.4/2.6.0; their closure is satisfied by unchanged
baseline packaging26.3/setuptools81. Root acquired/hash-verified both wheels.
Root pinned source archive SHA256 is
`4b8badbf8c35850b505de52a5aefeadd19e2aba6ea8d0d4e30860320a161511b`;
clean full-history checkout describes `0.48.0rc0-267-ge352caca`. The actual wheel
version remains unmeasured until root executes the public build backend.
Root prepared the concrete baseline/manifest and attempted the offline
build-tools image. Installation/audits succeeded and actual backend imports
printed setuptools81/wheel0.47/scm10.3.4; final namespace audit then rejected
duplicate wheel metadata. No image was published and no ModelOpt wheel was built.
The complete failed log and actual pre-fix role bytes are retained externally
as `build-tools-image-v1.log` and `build-tools-image-v1-role.source.txt` (0444).
The latter SHA256 is
`2d8ddfaf7005fdecb4aa7692d71b12d1fc329e9e99a5ed7f8d5f8a850ac492b3`.
Public dynamic PEP517 build requirements are
queried and checked in the build consumer; optional compiler execution is not
ruled out by import smoke. Actual source build remains root-only/network-isolated.

Remediation: capture the whole original sys.path, allow only the sealed
Setuptools vendor append, verify the five affected public build-module origins,
then restore the original list/binding before independent final audits.
An actual fresh managed-base consumer proved public wheel0.47.0 and packaging
still load from the original application prefix; only metadata discovery also
sees vendor wheel0.46.3. Setuptools81 __init__.py SHA256 is
`f94b531bda5d7846646222054faf51724302ab151d5482e38f6e20dea789feb7`.
All28 controls pass3.56s/exit0 with full actual path/origin evidence in
`calibration-import-context-controls-v4.log`. Public wheel alias to vendor code,
path changes, genuine duplicate installations and cleanup failures stay fatal.
Ruff/diff checks pass; image retry/source build remain unrun for this successor.

At that namespace-fix checkpoint, the build-tools retry, source wheel and
producer closure were pending; their subsequent results are recorded above.
Producer image/API smoke subsequently passed as recorded above; actual
calibration remains pending. No agent
package installation, image build or GPU job has occurred.
No generated lock with invented versions/hashes is checked in.

Separate later gates: actual target and Google-assistant coverage, chosen
calibration corpus/method/quality contract, source/runtime export consumption,
no batch-dependent target activation scale, resource guards and full multimodal
and native-context acceptance. Optional ModelOpt CUDA/JIT operators require
explicit approved compilation/resource/provider evidence. Optional real GEMM's
TensorRT-LLM dependency is not admitted by this CPU producer role.
