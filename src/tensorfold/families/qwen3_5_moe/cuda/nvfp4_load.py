"""Original compressed-tensors Qwen MoE weights, with a separately owned expert bank."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from tensorfold.checkpoint_metadata import read_config
from tensorfold.cuda.moe import Routed
from tensorfold.families.qwen3_5.cuda.nvfp4_load import SUFFIXES
from tensorfold.families.qwen3_5.cuda.nvfp4_load import load_nvfp4 as load_backbone


@dataclass(frozen=True)
class SharedExpert:
    gate: object
    up: object
    down: object

    def nbytes(self) -> int:
        return sum(projection.nbytes() for projection in (self.gate, self.up, self.down))


class NativeMoE:
    """Prepared router/bank/shared execution with the family's ordered combine.

    The bank writes only routed slots; shared output occupies the final slot.
    Decode retains FP32 down sums, while prompt slots retain BF16. No weight
    conversion or format discovery occurs during execution. The factory owns
    bank/authority lifetime; this operator owns the resident shared projections.
    """

    def __init__(self, bank, shared: SharedExpert) -> None:
        from tensorfold.cuda.moe import swiglu_bf16

        self.shared, self.swiglu = shared, swiglu_bf16
        self.capturable = getattr(bank, "capturable", False)
        # Fp4Linear.__call__ defaults to BF16 even with an FP32 destination in
        # checkpoint mode. Select the explicit FP32 epilogue before execution.
        if shared.down.act is not None:
            from tensorfold.cuda.nvfp4 import checkpoint

            self.down32 = partial(checkpoint.matmul, checkpoint.A4, lin=shared.down, f32=True)
        else:
            self.down32 = shared.down

    def __call__(self, x, router_rows, bank, buf, top_k, experts):
        from tensorfold.cuda import moe

        rows = x.shape[0]
        moe.router(x, router_rows, buf.logits[:rows])
        prefill = buf.plan.prefill
        if self.capturable:
            from tensorfold.cuda.experts import TILE

            # Resident banks group the routed and shared picks on the device;
            # the plan's shared cell (id ``experts``) is skipped by the bank.
            moe.select(buf.logits[:rows], buf, top_k, experts, TILE)
            bank.run_plan(x, buf, prefill=prefill)
        else:
            moe.select_rows(buf.logits[:rows], buf, top_k, experts)
            bank.run(x, buf.pick[:rows, :top_k], buf.y[:rows, :top_k], prefill=prefill)
        if prefill:
            gate, up = self.shared.gate.prefill(x), self.shared.up.prefill(x)
            activation = self.swiglu(gate, up)
            shared = self.shared.down.prefill(activation)
        else:
            gate, up = self.shared.gate(x), self.shared.up(x)
            activation = self.swiglu(gate, up)
            shared = torch.empty((rows, self.shared.down.n), dtype=torch.float32, device=x.device)
            self.down32(activation, out=shared)
        buf.y[:rows, top_k].copy_(shared)
        return buf


def _parts(context, name):
    return {suffix: context.info[f"{name}.{suffix}"] for suffix in SUFFIXES
            if f"{name}.{suffix}" in context.info}


def _router(context, prefix, cfg):
    rows = []
    for field, count in (("gate", cfg.experts), ("shared_expert_gate", 1)):
        name = prefix + field
        parts = _parts(context, name)
        kind, activation = context.metadata.validate(name, parts, checkpoint=context.checkpoint_math)
        if kind != "bf16" or activation is not None or parts != {"weight": ("BF16", [count, cfg.hidden])}:
            raise ValueError(f"{name}: Qwen NVFP4 routing requires original BF16 rows without quantized activations")
        rows.append(context.read(name + ".weight"))
    return torch.cat(rows, dim=0).contiguous()


def _shared(context, prefix, cfg, width):
    projections = []
    for field, n, k in (("gate_proj", width, cfg.hidden), ("up_proj", width, cfg.hidden),
                        ("down_proj", cfg.hidden, width)):
        name = prefix + "shared_expert." + field
        parts = _parts(context, name)
        kind, _ = context.metadata.validate(name, parts, checkpoint=context.checkpoint_math)
        weight = parts.get("weight", parts.get("weight_packed"))
        if kind != "nvfp4" or weight != ("U8", [n, k // 2]):
            raise ValueError(f"{name}: Qwen NVFP4 shared expert requires original NVFP4 weights of shape [{n}, {k}]")
        projections.append(context.linear(name))
    return SharedExpert(*projections)


def load_nvfp4(model_dir: str | Path, device: str = "cuda", *, expert_factory: Callable,
               eos_token_ids=None):
    """Read the ordinary NVFP4 backbone and delegate each routed bank lazily.

    ``expert_factory(context, absolute_mlp_prefix, config)`` returns a bank with
    ``run(x, picks, out, prefill=...)``. Picks/out are strided routed-slot views:
    their row stride includes the final shared slot. The factory must preserve
    original packed codes and reciprocal CT scales, consume every routed field
    exactly once through ``context.read``, and own its authority until close.
    """
    if not callable(expert_factory):
        raise TypeError("NVFP4 MoE loading requires an expert-bank factory")
    raw = read_config(model_dir)
    text = raw.get("text_config", raw)
    width = int(text.get("shared_expert_intermediate_size", text["moe_intermediate_size"]))
    if width <= 0 or width % 64:
        raise ValueError("Qwen NVFP4 shared expert width must be a positive multiple of64")

    def mlp(prefix, context, cfg):
        router = _router(context, prefix, cfg)
        shared = _shared(context, prefix, cfg, width)
        bank = expert_factory(context, prefix.removesuffix("."), cfg)
        return {"moe": Routed(router, bank, cfg.top_k, run=NativeMoE(bank, shared))}

    return load_backbone(model_dir, device, mlp=mlp,
                         **({"eos_token_ids": eos_token_ids} if eos_token_ids is not None else {}))
