"""Programmatic dependent launch (PDL) for Flash Next's decode kernels.

A kernel launched with PDL may be scheduled while its stream predecessor finishes. Every kernel launched that way
waits (``griddepcontrol.wait``: the predecessor grid complete and its memory visible) as its first operation on every
CTA path, before any global access (the lane matmul alone stages its immutable weights first), and only then lets its
own dependents launch: so each kernel still sees all of its predecessors' writes (a completed kernel implies its
predecessors completed), arithmetic is unchanged, and only launch latency and kernel tails overlap. Torch operations
and copies between them launch normally, which is a full dependency. CUDA graphs captured from PDL launches keep them
as programmatic edges.

``select`` resolves the chip's policy once per decoder (GB10, compute capability 12.1, on: grouped QMM's measured
policy; other chips off until measured; ``TENSORFOLD_PDL`` = 1 or 0 forces it on PDL-capable chips). ``ON`` is what
every launch site passes (its kernels' ``PDL`` constexpr and Triton's ``launch_pdl``, the extensions' attribute). It
is one process global, with this contract:

- Writers: only the Flash Next concurrent decoder. ``MultiDecoder.__init__`` and ``MultiDecoder.round`` (at each
  round's start) set the decoder's policy, and ``RoundGraphs.capture`` sets it for the captured keys and restores
  the value it found. The initial value (False) launches every site's original code.
- Readers: every launch of a converted kernel in the process follows its current value: the rounds, and also the
  prompt passes run between rounds, warm-up and capture forwards, and other engines' launches of the shared kernels
  (``tensorfold.cuda.kernels.qmm``/``gdn``, ``tensorfold.cuda.moe``/``experts``). Either value gives the same results,
  so it changes launch overlap only; passes between rounds keep the last round's value.
- Threads: a launch site reads ``ON`` twice (the kernel's constexpr and the launch attribute); a change between the
  reads could launch a kernel compiled without its wait with the attribute. It is therefore written only on the
  thread that launches the decode kernels and never while a launch is being built: the constructing thread through
  construction, warm-up and capture, then the scheduler's worker, which ``Thread.start`` orders after them.
- Lifetime: the process. No teardown or reset is needed in production (one engine a process; every value is valid at
  every site); a test that writes it restores it (``tests/cuda/conftest.py`` does after every CUDA test).
"""

from __future__ import annotations

import os

import triton
import triton.language as tl

ENV = "TENSORFOLD_PDL"
ON = False


def select(device: int = 0) -> bool:
    """This chip's PDL policy (sm_90 and later only): TENSORFOLD_PDL 1/0, else on for GB10 (12.1)."""

    import torch

    value = os.environ.get(ENV, "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1, not {value!r}")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 9:
        return False
    return value == "1" if value else capability == (12, 1)


@triton.jit
def wait(PDL: tl.constexpr):
    """A PDL-launched kernel's first operation: its predecessor complete and visible, then its dependents may launch
    (they wait here in turn). Nothing when the kernel is compiled without PDL."""

    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
