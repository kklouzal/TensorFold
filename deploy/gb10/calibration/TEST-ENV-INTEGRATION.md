# Test-only application integration

The separate test application leaves the existing producer Dockerfile, role,
pins, prefixes and SDK helpers unchanged. Native resolution, acquisition of six
test wheels, isolated materialization and CPU framework admission have passed.
Image `dd0f41f0` collected all 39 native GPU cases: 36 passed and three new
ragged cases failed an invalid cross-path equivalence assumption. The corrected
test-only source checkpoint passed all 39 cases on image `a0b914aa`, including
outer integrity/resource/retirement checks; see [evidence](EVIDENCE.md).

1. In the actual admitted fork producer image, original managed CPU interpreter:
   run `test_environment.py snapshot --pyproject SOURCE/pyproject.toml
   --producer-image sha256:ACTUAL_ID --output /work/test-baseline`.
   This validates originals, records source/config files and exact full app
   closure, and emits six upstream requirements plus old-version constraints
   with only pytest replaceable. Do not request all dev extras.
2. Root runs native full pip dry-run with that requirements/constraints pair,
   acquires exact report-selected native wheels, then runs `prepare --baseline
   .../baseline.json --report ... --wheels ... --output ...`. The manifest
   explicitly authorizes pytest9.2 replacement and new required plugins, never
   a protected accelerator replacement. Coverage's exact `a1_coverage.pth`
   source/hash must be reviewed as a TEST-only startup hook before install.
3. Build the separate [test image recipe](../Dockerfile.calibration-test) from
   the admitted producer with the reviewed `test-inputs` context. Its managed
   materialization and CPU smoke commands are authoritative.

   Both image identity and final separate test-app seal require root/D13/D14
   admission. Materialization copies the old application site into a new venv,
   removes old pytest-owned files ONLY in that copy, installs the actual solved
   set offline, and audits old originals even on failure. No hardlinks or old
   prefix writes are used. Startup files are byte-identical copies of the
   original managed activation. The old application site is absent from the
   new interpreter, leaving one effective pytest distribution. The separate
   ModelOpt prefix is then appended under the new explicit test baseline.
4. For GPU tests preserve outer host guard/quota/profile identity and
   original `cuda_entrypoint.py` → NVIDIA → `checked-run`, selecting the new
   test interpreter as its explicit final executable:

   ```text
   ... cuda_entrypoint.py python3 -u
       /opt/tensorfold-test-tools/test_environment.py start --output /receipt/out/test39
       /receipt/qualified_gpu_bootstrap_v3_guarded.py RECORD OPERATOR ARGS...
   ```

   The operator runs this tool's `pytest` command after the existing bootstrap
   has installed the CUDA quota. It accepts no arbitrary pytest options/nodes;
   it selects the three affected Testfp4 methods, preserves project addopts,
   conftest/timeouts/plugins and the nox conditional coverage setting, and routes
   only cache/report outputs away from the read-only source tree.
   When `COVERAGE_PROCESS_START` is enabled, set `COVERAGE_FILE` to the exact
   output-directory `.coverage` path BEFORE the test interpreter starts, so
   upstream startup tracing never writes into the original/source tree.
   The OLD-interpreter `start` boundary verifies the sealed prefix BEFORE its
   startup hooks can execute, refuses initial `COVERAGE_PROCESS_CONFIG`, validates
   the START config against the frozen pyproject, records the actual initial
   settings, and execs the fixed new test Python/tool/run chain. CONFIG generated
   later by upstream coverage for its subprocesses is not suppressed or patched.
   Original physical-prefix audits run in their clean interpreter pre/post;
   the test namespace has its own file/catalog/provider/module-origin checks.
   Pytest's legitimate temporary source-path additions are restored before
   final namespace validation. GPU collection is not attempted in CPU smoke.

Runtime uses native JUnit plus supported pytest collection/report hooks to
reconcile all39 source-bound cases (27/6/6). It rejects empty/missing/duplicate
cases, skipped/failed/error/xfail/rerun outcomes, incomplete phases and XML/report
contradictions. Every native report, including an initial rerun failure, is saved
unchanged as outcome metadata; existing retry/timeout/plugin behavior still runs.
Exit0 alone is never numerical qualification. The tool executes the reviewed,
source-bound tests without rewriting their numerical assertions. Direct
root/gpu/quantization conftests are source-pinned. A test-only source revision
requires new source hashes and a matching test-app manifest even when producer
runtime bytes, resolver output and wheels are unchanged.

Pending: full-model calibration, export and consumer/quality qualification.
The >2^31-element Testfp4 node remains capacity-blocked under the current 2 GiB
cap; the focused selection does not mark it passed or remove it.
