# GB10 progress IO scope

`tools/gb10_progress.py` preserves the existing single-process `Progress` and
`HotGate` contracts from the retained `campaign_bench.py` source `42eeacfa`.
The only changes are the progress writer and hot-timeout reason reader using
the shared `gb10_io.opened_file` owner. JSON formatting, ordinary buffered UTF-8
write/truncation, temporary-file naming, atomic replace, cadence, state fields,
clocks and thermal refusal stay unchanged. Optional timeout-read recovery applies
only after its owner has retired: a single read/open `FileNotFoundError` still
means cooling, and another single read/open `OSError` retains the original
unreadable-reason refusal. Independent close failures are fatal and cannot be
mistaken for cooling or flattened into reason text; raw roots remain visible.
No new fsync, permission, memory or timing policy is added.

The profile-specific `campaign_bench.py` derivative must remove the old two
class definitions and import these canonical classes. It must not retain
shadowed definitions, replace installed TensorFold callables, or change the
declared optional host-observation `OSError`-to-`None` recovery.

The targeted tests cover the owned integration: exact heartbeat format/cadence,
disabled work, existing temporary-file truncation/mode, write/read with close
failure, no publication after failed retirement, and hot-timeout refusal/cooling.
They do not reimplement filesystem or JSON algorithms. All **13 cases passed**
in the combined 58-case host run on Linux CPython 3.14.4 with pytest 9.0.2.
The changed progress code and tests add no Ruff diagnostics; seven inherited
controller diagnostics remain separately scoped. No profile or hardware run is
qualified by these host checks. Historical successful
captures and measurements retain their original source and validity.
