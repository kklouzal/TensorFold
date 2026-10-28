# GPU-local worker CPU scheduling

For the explicit eight-V100 profile, each GPU worker uses its GPU-local socket's
CPU allowance. The four GPUs on each socket share the entire local allowance;
this does not reserve disjoint cores. The approved host currently reports
GPU0–3 on CPUs `0-15,32-47` and GPU4–7 on `16-31,48-63`. The bootstrap resolves
GPU UUID/PCI identity and verifies the live PCI NUMA node and node CPU list
against the SHA256-held policy before selecting their intersection with the
child's inherited CPU allowance. An empty intersection or changed topology fails
startup. Parent/supervisor affinity and the container's CPU quota are unchanged.

Use the owned `tools/gpu_local_worker.py` as the child entrypoint, before loading
TensorFold, Torch, CUDA, the model, or worker helper threads. An ordinary external
TP8 launcher uses the same entrypoint for each of its eight independently owned
rank sessions (substitute the matching rank, UUID, policy, digest and receipt):

```sh
CUDA_VISIBLE_DEVICES=GPU-... python tools/gpu_local_worker.py \
  --rank 0 --policy /configuration/worker-affinity-policy.json \
  --policy-sha POLICY_SHA256 --receipt /evidence/rank-0-affinity-bootstrap.json \
  -- python -m tensorfold serve MODEL --tp 8 --rank 0 --master 127.0.0.1 --master-port 29607
```

The supervisor still owns all eight processes, exit statuses and retirement.
The exec keeps the child PID/session and inherited failure pipe; there is no
`preexec_fn`, parent affinity mutation, second target replica, or expert-only
launcher substitution. A per-rank receipt must be a new file. This entrypoint is
an explicit profile configuration, not an automatic policy for unrelated hosts.

The finite backend probe accepts `--worker-affinity gpu-local` plus
`--worker-affinity-policy` and `--worker-affinity-policy-sha`; it wraps only its
rank child commands. It records actual TID masks before backend imports, after
engine preparation and after the rank0 finite-caller pool is fully materialized,
outside the measured loops. Still-live helper masks outside the selected local
allowance fail the diagnostic. Boundary censuses do not prove continuous
all-time affinity; helper threads already retired at a census are not described
as live observations. The explicit `observe-reference` mode is only a matched
experimental reference and never the configured production policy.

CPU affinity does not set memory policy, migrate or lock pages, alter NUMA
balancing, or relocate previously shared file/PLE/page-cache pages. Startup
first-touch may change physical placement incidentally; no local-memory claim
follows. NCCL, PLE/preload worker counts, BLAS thread settings, all graph families,
24 full Q8 windows and numerical policies are unchanged. Actual eight-rank
helper/locality, resource, quality and all-six throughput verification is pending;
no performance, HTTP deployment or release qualification is claimed here.

The Linux interface is per-thread; children retain affinity across exec and new
threads inherit their creator's mask. See the maintained
[sched_setaffinity manual](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)
and [pthread affinity manual](https://man7.org/linux/man-pages/man3/pthread_setaffinity_np.3.html).
