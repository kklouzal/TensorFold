# Managed CUDA toolkit-state IO

The managed `TF_FOUNDATION_STATE` reader closes its stream exactly once and
retains a read failure together with an independent close failure. Failed
streams and raw error statuses remain owned by the error path. Cleanup
transport is imported lazily only on failure, after resource retirement is
attempted; successful reads gain no new transport dependency.

The existing `Path.open("rb")`, `64 MiB + 1` read, 64 MiB refusal limit,
`json.loads`, symlink handling and toolkit-validation rules are unchanged.
Native sources, architecture flags, projection arithmetic and model routing
are unchanged.

Thirteen targeted CPU cases passed on Linux CPython 3.14.4 with pytest 9.0.2,
covering read/close combinations, interruption, hostile exception getters,
transport-import failure, size limits, symlinks and invalid state. Thirty
unaffected cases were deselected in that focused run. The source and test
files remained unchanged during execution; no compiler, CUDA or model ran.

Fresh installed-image and model qualification remains separate. The published
Gemma throughput receipt retains its original source `16b4c087`; this IO fix
does not claim another performance gain or a new loading-time measurement.
