# CUDA native qualification evidence

Root's six `actual02` jobs qualified the held extra CUDA packet on one NVIDIA
RTX PRO 2000 Blackwell device, SM 120, with PyTorch
`2.16.0.dev20261006+cu134` and CUDA 13.4. The independent stdlib admission checks
the complete run receipt, worker receipt, raw result and normalized result;
their source hashes, counts and resource records agree.

| Changed surface | Original valid byte cases | Corrected input rejections | Captures | Other checks |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5 B16 | 90 | 13 | 10 | Three prompt specializations |
| GLM EXL3 grouped divisor | 10 | 9 | 1 | Positive representable divisor and output bounds |
| Nemotron grouped scan | 40 | 4 | 8 | Recurrence outputs, chunk boundaries and final state |
| Qwen3.5 local QMM | 97 | 8 | 8 | Declared layouts and dtype boundaries |
| Draft attention | 36 | 11 | 0 | Existing eager operation contract |
| GLM split | 72 | 0 | 1 | 12 original split rules |
| Total | 345 | 45 | 28 | 12 rules |

Malformed inputs ran only through corrected entries. The recorded number of
malformed original calls is zero. The C++/CUDA comparisons bind independently
compiled original and corrected libraries, their compiler flags and source
hashes. Nemotron's isolated symbol names are reversed mechanically to its exact
held sources. QMM, draft attention and split bind their held source/oracle
inputs; the split comparison includes a separate byte-slice oracle.

Every job used a 12 GiB CPU memory cap, zero swap, eight CPU cores,
`MAX_JOBS=8`, a 4 GiB GPU allocator budget and an 1800-second compilation deadline.
Observed cgroup and allocator peaks remained within those bounds, with zero
recorded OOM or memory-limit events. All six exited successfully, retained no
failed build owners, reaped their launch processes and removed their containers.
The 8 GiB free-memory admission used by these isolated qualification jobs is a
test resource condition, without an added production hardware requirement.

The 15 relevant canonical native/runtime files still match the held packet.
Its copied `tensor_file.py` is unused by these six workers; subsequent changes
to that reader belong to its separate current-source qualification. The jobs
make no full-model, serving performance, later installed-wheel or two-device
claim. Draft attention has no capture credit. B16's checked-setter failure and
concurrent configuration checks belong to the separate shared configuration
receipt; these six jobs alone do not qualify that failure path.

The reproducible admission controller is
`task-artifacts/quality-preserving-optimization/check_extra_cuda_actual_evidence_v1.py`.
Its receipt is
`task-artifacts/quality-preserving-optimization/extra-cuda-actual02-independent-cuda-admission-01.json`;
that receipt binds all 36 result/log/resource artifacts and the exact staged
input maps. Reusing this evidence requires the relevant source, runtime,
compiler flags, device and selected numerical contracts to remain unchanged.
