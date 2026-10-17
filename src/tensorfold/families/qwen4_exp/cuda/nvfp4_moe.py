"""Flash Next's NVFP4 MoE: routed experts on the grouped NVFP4 kernel, the bf16 shared expert on the FP4 tables."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.expert_parallel import owned_items
from tensorfold.cuda.nvfp4 import experts as nvx

from . import nvfp4


@dataclass
class Expert4:
    """One expert: gate/up rows in one grid (gate rows first), and the down grid."""

    gu: nvfp4.FP4         # [2*NI, K]
    down: nvfp4.FP4       # [K, NI]


@dataclass
class MoE4:
    """One layer's experts: the routed ones as grouped NVFP4 blocks, the shared one as bf16 tables."""

    routed_experts: nvx.Experts4
    shared: Expert4
    kernel: str = "nvfp4"

    capturable = True     # the plan stays on the device

    @property
    def routed(self) -> int:
        return self.routed_experts.count

    @property
    def count(self) -> int:
        return self.routed + 1

    @property
    def width(self) -> int:
        return self.routed_experts.width                     # NI: a gate row's inputs, down's outputs

    @property
    def dims(self) -> int:
        return self.routed_experts.dims                      # D: x's inputs, down's outputs

    def nbytes(self) -> int:
        return self.routed_experts.nbytes() + self.shared.gu.nbytes() + self.shared.down.nbytes()

    def shared_act(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        """x [R, D] bf16 -> the shared expert's SwiGLU [R, NI] bf16, bf16(bf16(silu(g)) * u) on bf16 g and u."""

        g = _shared(self.shared.gu, x, prompt, False)
        gate, up = g[:, :self.width].to(torch.float32), g[:, self.width:].to(torch.float32)
        return ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32) * up).to(torch.bfloat16)


class CachedMoE4:
    """Host-backed routed banks plus the leader's unchanged resident shared expert.

    A bank can be empty on an EP follower. Local routed IDs are compact, the
    optional shared ID follows them, and -1 in ``run_picks`` is an unowned pair.
    The caller serializes bank use; the underlying cache serializes leases.
    """

    kernel = "nvfp4"
    capturable = False

    def __init__(self, routed, shared, width, dims, count, slots, *, workspace=None):
        self.routed_experts, self.shared = routed, shared
        self.width, self.dims, self.routed = width, dims, count
        self.slots = slots
        # EP passes one model-owned dictionary to all main/MTP banks. Its
        # callers serialize every consumer on the same CUDA compute stream.
        # Standalone use owns one bank's profile; the ordinary MoE path uses
        # the existing shared MoEBuffers and never allocates this workspace.
        self._workspace = {} if workspace is None else workspace

    @property
    def count(self):
        return self.routed + int(self.shared is not None)

    def device_tensors(self):
        scratch = tuple(t for plan, act in self._workspace.values()
                        for t in (plan.members, plan.items, plan.counts, plan.rank, plan.hist, act))
        return ((self.routed_experts.device_tensors() if self.routed_experts is not None else ()), self.shared, scratch)

    shared_act = MoE4.shared_act

    def run_picks(self, x, picks, out, *, prefill):
        """Execute local picks into complete unweighted slots; unowned slots are zero.

        Bounded chunks retain each native pair's original inner arithmetic.
        Invalid IDs are refused before a plan can index any expert storage.
        """
        if (x.device.type != "cuda" or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != self.dims
                or x.stride(-1) != 1 or x.stride(0) % 4 or x.data_ptr() % 16
                or picks.device != x.device or picks.dtype != torch.int32
                or picks.ndim != 2 or picks.shape[0] != x.shape[0] or not picks.is_contiguous()
                or out.device != x.device or tuple(out.shape) != (*picks.shape, self.dims)
                or out.dtype != (torch.bfloat16 if prefill else torch.float32) or not out.is_contiguous()
                or out.data_ptr() % 4
                or (self.routed_experts is not None and x.device != self.routed_experts.cache.device)):
            raise ValueError("NVFP4 local picks require BF16 rows, contiguous INT32 picks and matching output slots")
        slots = picks.shape[1]
        if not 0 < slots <= 1024 or slots != self.slots:
            raise ValueError("NVFP4 local picks must use the configured routed/shared slot count")
        def overlaps(source):
            if not source.numel() or not out.numel():
                return False
            end = source.data_ptr() + (sum((n - 1) * s for n, s in zip(source.shape, source.stride())) + 1) * source.element_size()
            return source.data_ptr() < out.data_ptr() + out.numel() * out.element_size() and out.data_ptr() < end

        if (overlaps(x) or overlaps(picks)
                or (self.routed_experts is not None and overlaps(self.routed_experts.cache._pool))):
            raise ValueError("NVFP4 output slots must not overlap input rows, picks or expert storage")
        out.zero_()
        for lo in range(0, x.shape[0], 128):
            hi = min(x.shape[0], lo + 128)
            cpu = picks[lo:hi].cpu()
            if bool(((cpu < -1) | (cpu >= self.count)).any()):
                raise ValueError("NVFP4 local picks contain an out-of-range expert")
            key = (x.device, self.width, self.dims, self.routed, slots)
            scratch = self._workspace.get(key)
            if scratch is None:
                with torch.inference_mode(False):
                    plan = grouped.Plan(128, slots, max(1, self.routed), x.device)
                    act = torch.empty((128 * slots, self.width), dtype=torch.bfloat16, device=x.device)
                scratch = self._workspace[key] = (plan, act)
            plan, act = scratch
            plan.prefill = bool(prefill)
            plan.tile = nvx.PREFILL_TILE if prefill else grouped.TILE
            # Deterministic expert/row-slot ordering agrees with grouped.route;
            # unowned and resident-shared pairs never enter the routed plan.
            values = [expert if expert < self.routed else -1 for expert in cpu.reshape(-1).tolist()]
            members, items = owned_items(values, max(1, self.routed), plan.tile)
            if members:
                plan.members[:len(members)].copy_(torch.tensor(members, dtype=torch.int32))
                _run_cached_items(x[lo:hi], self.routed_experts, plan, act, out[lo:hi].view(-1, self.dims),
                                  torch.tensor(items, dtype=torch.int32))
            if self.shared is not None:
                pairs = torch.nonzero(cpu == self.routed, as_tuple=False)
                if pairs.numel():
                    index = pairs[:, 0].to(x.device)
                    value = self.shared_act(x[lo:hi].index_select(0, index), prefill)
                    result = _shared(self.shared.down, value, prefill, not prefill)
                    out[lo + index, pairs[:, 1].to(x.device)] = result


def _run_cached_items(x, cached, plan, act, out, items):
    """Lend distinct routed experts in bounded item windows, preserving members."""
    at = 0
    while at < items.shape[0]:
        end, ids, seen = at, [], set()
        while end < items.shape[0]:
            expert = int(items[end, 0])
            if expert not in seen:
                if len(ids) == cached.capacity:
                    break
                ids.append(expert)
                seen.add(expert)
            end += 1
        with cached.lease(ids) as (hot, mapping):
            block = items[at:end].clone()
            block[:, 0] = torch.tensor([mapping[int(i)] for i in block[:, 0]], dtype=torch.int32)
            plan.items[:end - at].copy_(block)
            plan.counts.copy_(torch.tensor((end - at, len(ids)), dtype=torch.int32))
            plan.experts = cached.capacity
            nvx.gate_up(x, hot, plan, act, x.shape[0])
            nvx.down(act, hot, plan, out, x.shape[0])
        at = end


def _shared(lin, x: torch.Tensor, prompt: bool, f32: bool) -> torch.Tensor:
    """The shared expert's matmul: bf16 weights on the FP4 kernel's exact tables, MXFP8 ones on the lane matmul."""

    if isinstance(lin, nvfp4.FP4):
        return nvfp4.matmul(x, lin, f32=f32)
    return lin.prefill(x) if prompt else lin(x)


def expert4_from_bf16(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> Expert4:
    """The shared expert from bf16 weights (torch linear layout [out, in]): identity-scaled FP4 tables."""

    gu = nvfp4.fp4_from_bf16(torch.cat([gate, up], dim=0).contiguous())
    return Expert4(gu, nvfp4.fp4_from_bf16(down.contiguous()))


def moe4_from_checkpoint(gate: tuple, up: tuple, down: tuple,
                         shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """A layer from stacked checkpoint arrays (words, e4m3 scales, per-expert scales) and the bf16 shared expert."""

    return MoE4(nvx.make(gate, up, down), shared if isinstance(shared, Expert4) else expert4_from_bf16(*shared))


def moe4_from_bf16(gate_up: torch.Tensor, down: torch.Tensor,
                   shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """The MTP layer's bf16 experts (gate_up [E, 2NI, K], down [E, K, NI]) in NVFP4: they only draft."""

    ni = gate_up.shape[1] // 2
    gate, up = nvx.quantize(gate_up[:, :ni]), nvx.quantize(gate_up[:, ni:])
    return MoE4(nvx.make(gate, up, nvx.quantize(down)),
                shared if isinstance(shared, Expert4) else expert4_from_bf16(*shared))


def moe4_from_experts(gate: list, up: list, down: list, shared: tuple) -> MoE4:
    """Per-expert FP4 triples and the bf16 shared expert -> a layer (the test path)."""

    def stack(items):
        return (torch.stack([t[0] for t in items]), torch.stack([t[1].view(torch.uint8) for t in items]),
                torch.stack([torch.as_tensor(t[2], dtype=torch.float32).reshape(()) for t in items]).to(
                    items[0][0].device))

    return moe4_from_checkpoint(stack(gate), stack(up), stack(down), shared)


def moe(x: torch.Tensor, xs: torch.Tensor, router_rows: torch.Tensor, ex: MoE4, buf, cfg) -> None:
    """Route rows x [R, D] and run their experts into buf (the shared ``moe`` contract); ``xs`` unused: no biases."""

    from tensorfold.cuda import moe as moe_mod

    rows, top_k = x.shape[0], int(cfg.num_experts_per_tok)
    moe_mod.router(x, router_rows, buf.logits[:rows])
    moe_mod.select(buf.logits[:rows], buf, top_k, ex.routed, nvx.PREFILL_TILE)
    prompt = buf.y.dtype != torch.float32                  # a prompt's buffers keep bf16 slots
    if isinstance(ex, CachedMoE4):
        from .forward import _ExpertWindow

        scratch = getattr(buf, "expert_window", None)
        if scratch is None:
            scratch = buf.expert_window = _ExpertWindow(buf.plan)
        items = scratch.read(buf.plan)
        # The shared expert has its own exact resident representation; filter
        # it before compact slot remapping rather than treating it as FP4.
        items = items[items[:, 0] < ex.routed]
        _run_cached_items(x, ex.routed_experts, scratch.plan, buf.act.view(-1, ex.width),
                          buf.y.view(-1, ex.dims), items)
        buf.act[:rows, top_k] = ex.shared_act(x, prompt)
        buf.y[:rows, top_k] = _shared(ex.shared.down, buf.act[:rows, top_k], prompt, not prompt)
        return buf
    nvx.gate_up(x, ex.routed_experts, buf.plan, buf.act.view(-1, ex.width), rows, skip=ex.routed)
    buf.act[:rows, top_k] = ex.shared_act(x, prompt)
    nvx.down(buf.act.view(-1, ex.width), ex.routed_experts, buf.plan, buf.y.view(-1, ex.dims), rows, skip=ex.routed)
    buf.y[:rows, top_k] = _shared(ex.shared.down, buf.act[:rows, top_k], prompt, not prompt)
    return buf
