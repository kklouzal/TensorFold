# Explicit native expandable allocation

`tensorfold serve ... --native-expandable-segments` maps to the dense CUDA
factory/constructor keyword `native_allocator="expandable"`. This one positive
option initially supports Qwen dense ModelOpt/compressed-tensors checkpoint math
on SM12.1, with `tp=1`, `rank=0`, one stream and bf16 prompts. It changes allocation policy,
not model precision, numerical kernels, weights, context or cache semantics.
MLX, other families, affine/EXL3, full-precision mode, FP8 prompts and other topology/device
regions reject the explicit request before the setter or weight payloads.

Omit the option to retain existing behavior. Omission does not inspect or modify
allocator configuration, reject preconfigured/embedded/multiple-engine callers,
or disable the existing multi-stream policy. CT MoE, Flash and Gemma receive no
new allocator policy through this option.

The opt-in caller owns a newly exec'd process with the reviewed import path, no
previous allocator setting calls, no other CUDA allocator consumers and no
concurrent initialization. The command is also an importable Python function;
calling `cli.main` or a constructor inside an arbitrary existing process does
not prove that precondition. Empty segments/default last text corroborate it but
cannot reveal every persistent native field. The strong ownership requirement
is specific to this optional process-global policy, not a general engine rule.

Before configuration, the implementation requires the native backend, exact
known exposed default settings, zero allocator segments and absence of allocator
aliases, cache-disabling and IPC overrides (including empty values). Existing
operator configuration is not deleted, appended to, normalized or reset. Only
after device/metadata/precision admission does it call the already adopted
`torch._C._accelerator_setAllocatorSettings("expandable_segments:True")` once,
before weight tensors, capacity allocations, graphs or request workers. It
requires effective expansion and unchanged unrelated settings afterward.

The independently versioned Torch interface and schema are pinned by the
native runtime qualification. Unsupported warnings, missing capabilities and
partial parse errors fail startup; there is no warning fallback, retry or
settings restoration. After a setter attempt the owner must terminally retire
the failed startup process. Successful settings remain process-global and fixed
through all consumers' lifetimes, including graph replay and teardown; engine
close does not reset another owner's allocator. No extra proof tensor or global
import-time setting is inserted in normal startup.

The optional source API/CLI passed 198 CPU checks in 2.59 seconds in pinned AMD64
image `509b4135` with GPUs hidden. These cover configuration admission, direct
constructor/factory/CLI forwarding, bf16-prompt restrictions, omitted/preconfigured
and multiple-engine behavior, and failure propagation. All 1,313 admitted source/test
files remained unchanged, 137 imported project origins matched, and the exact test
container retired. Both dedicated source and policy reviews closed cleanly.
These source controls use CPU weights and labeled CUDA boundaries; they do not
establish execution of the new policy in a native CUDA engine or a model gain.

The separate installed capability smoke used source `c8e2d0eb`, image `8bbf384f`, Torch
`2.16.0.dev20261014+cu134`/git `ef46daf1`, CUDA13.4 and one GB10 SM12.1. It verified
one setting with all other exposed defaults unchanged, an actual 3 MiB first
allocation in a 20 MiB expandable segment, retained strided-view bytes, three
complete A/B/A graph replays with an actual private graph pool, and zero final
segments/retained owners. Observed mapping topology reached 24 MiB and requested
payload 4 MiB; its conservative explicit payload bound was 4.5 MiB under a 7 MiB
cap. Three virtual reservations totaling 440,766,824,448 bytes were virtual
address space, not physical mappings or UMA-residency evidence. This functional
probe does not qualify a later source/image or imply a model performance gain.

The new source requires immutable installed/native and full-model qualification,
including the actual CLI option, default and conflict paths, complete token and
cache/speculation controls, graph lifetime/capacity and startup/request regions.
Whole-process startup must include policy/init/mapping costs. Comparison is to
the accepted old-fragment/native complete design; allocator-only and combined
packing designs are distinct. The [isolated packing rejection](fp8-packing-validation-20261021.md)
remains unchanged, with no transfer of historical f1/c8 timing intervals.
