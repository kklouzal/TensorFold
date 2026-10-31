# Automatic model configuration: local verification

Application source `8cf97219722a391203ca7602e6047cfeaaa643b5` was built through
the ordinary Dockerfile runtime and verification targets. The installed package
matched all 517 shipped source files; no application source, installed dependency
or checkpoint was patched for qualification. The
[receipt](verification-model-metadata-20261017.json) records immutable images,
inputs, checks, captured requests and resource observations.

The host has one RTX PRO 2000 Blackwell/SM120, 16 GB VRAM, an eight-thread
Xeon E3-1285 v6 and about 62.68 GiB visible system RAM, with no swap. Runtime
versions were Python 3.12.3, PyTorch `2.16.0.dev20261006+cu134`, CUDA 13.4,
Transformers 5.17.0, Hugging Face Hub 1.24.0 and NCCL 2.30.7.

## Resource policy

These runs set `TENSORFOLD_MEMORY_RESERVE_GIB=8` and
`TENSORFOLD_CUDA_MEMORY_RESERVE_GIB=0`, with absolute memory caps unset. Container
RAM, CPU, PID, memlock and stack limits were removed. Live host cgroup checks
verified unlimited CPU/RAM/PIDs through the container's ancestor hierarchy;
an owned Docker update removed an inherited PID ceiling. Host-wide defaults
were untouched. All visible GPUs were exposed; this machine has one.

Required model, KV, copy, snapshot and workspace allocations remained accounted
for. The 8 GiB host floor used sampled `MemAvailable`, including reclaimable
cache, with a 250 ms monitor that could stop only its owned validation container.
It is not a continuous reservation against other processes. The lowest sampled
availability across the three main model runs was 21,920,694,272 bytes, about
20.42 GiB. Each server exited with SIGINT status zero, without OOM or cleanup
failure. The embedding container was left stopped.

## Real NVFP4 checkpoint

`surogate/Qwen3.5-9B-NVFP4` at revision
`3410fbc4f4075661f58f6b222d05b6dde646b8c1` is a dense ModelOpt export. Its complete
9,361,048,680-byte weight file was rehashed and matched the pinned publisher
SHA256 `a1304acf8325cee90075bf8eb7f5e973ef893c7d497ded92e7d720f0e7178749`.

The automatic run omitted precision, EOS, template and launch reply-limit flags.
It selected declared FP4 activation math, EOS `[248044, 248046]`, the checkpoint's
Jinja template and the 4096-token reply default. The integer-answer request
returned `323` and stopped on the tokenizer's chat EOS, without a request stop
workaround. The template source hash was
`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.

A separate run explicitly selected `--precision full`, replacement EOS `248046`,
the mounted template file and a 64-token launch reply default. Startup and stats
reported those overrides; requests could still supply their own reply lengths.
Both runs used four slots, a controlled 16,384-token window, BF16 KV, drafts off,
FP8 prefill off and thinking off. Each completed 29 captured HTTP operations,
including three 512-token coding replies.

| Precision | Observed decode rates, three coding requests | Token SHA |
| --- | --- | --- |
| Automatic checkpoint math | 23.74 / 23.94 / 23.99 tok/s | `253ca1e75134` |
| Explicit BF16 activations | 33.97 / 34.22 / 34.22 tok/s | `8b1e791ef6ca` |

Complete coding-token arrays matched the prior same-checkpoint, same-precision
reference captures from the [earlier qualification](small-nvfp4-validation.md).
That reference used an explicit chat-stop workaround; these requests did not.
The coding replies reached their 512-token limit before EOS. The two precision
modes have different numerical contracts and different outputs. These unpaired
observations, after functional requests and reused compiler caches, establish no
speedup, performance equivalence, generated-code correctness or broad model quality.
The historical dense reference is a composite qualification: its v2 coding
samples completed before a later family-inapplicable harness assertion failed.
That original attempt remains failed; only its completed coding captures serve
as numerical references here. The receipt seals their paths, hashes and producer.

Seeded repeated/serial and concurrent/solo token identity, SSE framing/usage,
token counting, model discovery and strict JSON output passed. This export's
JSON-call template correctly refused the XML-only strict function-tool compiler,
including the unchanged 31-tool Harness catalog. Enforcement was preserved.

An additional startup omitted context and all the launch overrides above. It
fitted **118,149 tokens** beneath the model's native 262,144-token ceiling: its
startup estimate was 14.78 GiB within a 14.78 GiB admission budget. Caches grow
on demand; the short smoke's sampled GPU peak was 8573 vendor MB. Models,
tokenization and stats agreed on
the fitted window. A short request also omitted its reply limit and stop strings;
it completed `323` with the expected EOS and the 4096-token default configured.
This verifies admission and short generation, not a filled 118k context.

## Flash Next regression and kernel checks

The unchanged `turboderp/Qwen3.8-Flash-Next-exl3` revision
`65c895314393431c09050b2e04e250836b3a6eb4` completed 24 HTTP checks with four slots,
context 16,384, RotorQuant8-norm KV, MTP-4/confidence 0.5 and pageable RAM experts.
Its pool grew from the prior 3.05 GiB configuration to **3.55 GiB**, with 3056
resident cells and zero optional headroom; all four full-window growths and
24 snapshots remained reserved. Loading took 88.1 seconds with reused caches.
This is not a paired startup or throughput comparison.

Repeated/drafted/serial and concurrent/solo identity, streaming progress, strict
function calls and the 31-tool catalog passed. Its reference comparison uses
the prior same requests and precision; request identities are checked separately
from completion ordering in the concurrent batch.

The final installed verification image's focused affected-contract gates passed
**1,423 CPU tests** and **114 native CUDA tests**; the receipt lists their selected
modules. This was not a whole-repository suite rerun. Twelve model-plus-matching-drafter cases were skipped because their
inputs were unavailable. Native coverage includes NVFP4/FP8 numerical oracles,
EXL3/RAM-expert operations and actual one-rank NCCL broadcast, int64 EOS agreement
and expert-session lifecycle. The affected-file Ruff comparison had no new
findings against the original baseline.

Two physical GPU ranks, mixed SM89/SM120 execution, Apple MLX execution and the
full Flash Next ModelOpt export remain unqualified on this host. The latter's
63.3 GiB expert authority alone cannot fit here while retaining 8 GiB of RAM.
NCCL implementation and one-rank execution do not substitute for a two-GPU run.
Historical failed constructor and resource-proof attempts are retained in the
evidence packet; the completed runs above are separate clean attempts.
