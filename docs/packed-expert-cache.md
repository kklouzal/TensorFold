# Packed affine expert cache

Flash Next's RAM-backed affine4/group-32 loader can retain the exact packed
expert bytes between starts. Enable it explicitly in a writable persistent
service cache, outside the read-only model mount:

```bash
export TENSORFOLD_PACKED_EXPERT_CACHE=/cache/packed-experts
export TENSORFOLD_PACKED_EXPERT_CACHE_BYTES=<positive-byte-budget>
```

Both variables are required when enabling persistence. Set the byte budget from
the selected checkpoint's packed host-authority size and available disk, allowing
one layer's publication workspace plus metadata. If a layer exceeds the budget,
startup reports that it was not stored and continues with canonical packing.
Unset the directory variable to disable persistence. This is affine expert
persistence; the EXL3 loader retains its original compact source format.

The directory must belong to the service UID and must not be writable by group
or other users. Give processes sharing a cache root the same budget. Managed
artifacts and the single writer's temporary artifact stay within that byte
budget; at most 4096 complete artifacts are retained. Older artifacts are evicted
by publication time, including when a later start reduces the configured budget.
An externally populated inventory larger than that bound is refused explicitly.
The bound excludes filesystem block/metadata overhead and unrelated directory
contents. Active access is serialized with a process advisory lock. A killed
publisher leaves a private temporary file that the next exclusive owner removes.

Each artifact binds the full tensor names, validated shapes/dtypes, exact owned
expert order, shared-expert selection, and group/layout version to SHA-256 of
all contributing source byte ranges. Source bytes are hashed again at each
startup; saved mtimes or hash receipts do not authorize a hit. A rank reads and
allocates only its owned routed experts and, when requested, the shared expert.
Main and MTP banks have separate identities. Checkpoint files must stay immutable
through loading, as required by the ordinary checkpoint reader.

The file contains a bounded canonical JSON header and two little-endian int32
arrays in the native grouped expert layout: group size 32, 160-word blocks,
gate/up interleaved, BF16 scale/bias bits unchanged. Cached payloads are size and
SHA-256 checked while copying into final **owned** host authority. Their bytes
are never used directly through a memory map. Authority allocation follows the
model's selected pinned/pageable policy without a second whole-bank copy.

Publication uses a private temporary file, exact writes, file fsync, atomic
replacement and directory fsync. Missing, stale or damaged artifacts are reported
and canonically repacked. Source changes, unauthorized access, I/O failures,
resource exhaustion and cleanup failures stop startup with the original failure
context; they are not silently converted to a cache miss. Cache digests detect
corruption; the private service-owned directory is the authentication boundary.

The reported 271-second startup and proposed sub-minute target concern a
72-GiB affine authority on a different desktop. Small synthetic model and native
packing checks on the local Blackwell host cannot establish that result.
