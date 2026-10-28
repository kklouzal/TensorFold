# Emergency TensorFold recovery — 2026-10-28

Development and V100 access were stopped at the user's emergency pause. This is a recovery checkpoint, not a tested release.

All local branch heads are preserved in upstream recovery refs; see final-source-ref-verification.json and recovery-heads.json. Older dirty worktrees were snapshotted with a temporary Git index into emergency/20261028/wip/* commits without changing their original files, staging area, or branches. Their manifest records the recovery commits. Active authors separately committed and pushed their WIP. Untested and failed-test states remain explicitly labeled. No force push or history rewrite was used.

The latest measured configuration before the storage failure was c9284c50 / APP4b0e with required GPU-local NUMA bootstrap f11b36c2. Its six-tier short-screen rates were 32.9 /45.5 /76.1 /112.1 /127.3 /129.9 aggregate tokens/s. These are provisional results, not full qualification. Both85single and600aggregate goals remain unmet; two historical quality spot-check failures remain. Tail-only full-model testing was UNRUN: the host filesystem became read-only before its container launched. Tail2GPU and Pool45GPU component checks passed. Other current WIP is documented in active-source-state.json and author reports.

The authoritative design document and handoffs are included. The evidence archive preserves local small source scripts, JSON receipts/results, logs, reviews, and reproduction metadata with per-file hashes. It excludes model weights, wheels, binaries, large raw traces, symlinks, and potential credential material. This is a source/research recovery backup, not a full disk image. Some raw traces existed only on the V100 host and are not claimed backed up here.

Reassemble and verify the evidence archive:

```sh
cat portable-evidence.tar.gz.part-* > portable-evidence.tar.gz
sha256sum portable-evidence.tar.gz
tar -xzf portable-evidence.tar.gz
```

Compare its SHA256 with archive-parts.json. Fetch a source recovery branch with git fetch origin and create a local branch from the recorded remote ref. Preserve each candidate's original verification status.
