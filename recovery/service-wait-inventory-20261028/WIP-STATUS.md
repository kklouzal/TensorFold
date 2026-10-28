# Emergency recovery checkpoint — UNVERIFIED WIP

User paused development and tests for the V100 storage emergency. No V100 access,
new tests, builds, experiments or deployment are authorized by this checkpoint.

The counter prototype in `tools/cpu_counters.py` and its optional probe hooks are
UNVERIFIED and unfinished. No counter-specific tests, staged import, resource
bound proof or D13 review has passed. Raw capture, PID/thread/cgroup failure
controls, offline delta/alignment checks and target collection still require
completion after explicit resumption. Do not stage or run this WIP flag.

The last verified required NUMA implementation remains f11b36c2 / tool HOLD205b /
normal APP4b0e, with the separate source D13 a4403468 and actual7294 audit9a2353b0.
Its six-tier first screen was32.9/45.5/76.1/112.1/127.3/129.9, unchanged QA/current
prefixes and full24Q8/resource/counter evidence. This is a required provisional
configuration, not a repeatable superiority or release claim.

The JSON/source files in this directory are immutable emergency copies of owned
assessment/plan artifacts; their original paths and hashes are in the manifest.
The primary owns the full external raw-evidence backup and master register.
Application source under `src/tensorfold` was not changed by this counter WIP.

All pending owned counter source and plans are preserved as-is for recovery;
no remediation or new development was performed after the pause instruction.
