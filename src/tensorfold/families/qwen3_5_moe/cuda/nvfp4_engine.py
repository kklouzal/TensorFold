"""Qwen CT NVFP4 MoE with original A4 math and separately verified DFlash proposals."""
from __future__ import annotations

import threading
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from tensorfold.cleanup import rollback
from tensorfold.families.qwen3_5.cuda.engine import KEEP, KEEP_ONE, Qwen27Engine


class QwenNvfp4MoeEngine(Qwen27Engine):
    """One target worker; accepted calls retain all expert and draft owners through completion."""

    def __init__(self, model_dir, draft_dir=None, *, context=None, context_explicit=None, streams=1,
                 vram_experts=None, vram_experts_pin="off", drafter_bits=16, max_rows=None,
                 allow_copy=True, eos_token_ids=None, chat_template_file=None, chat_template_name=None):
        if type(drafter_bits) is not int or drafter_bits not in (0, 4, 8, 16):
            raise ValueError("Qwen DFlash draft precision must be 0/16 (BF16), 8 (proposal-only FP8) or 4 "
                             "(proposal-only affine4)")
        if type(allow_copy) is not bool or vram_experts_pin not in ("off", "auto", "on"):
            raise ValueError("copy proposals require a bool; expert pin policy must be off, auto or on")
        self._lifecycle = threading.Condition()
        self._serial = threading.Lock()
        self._calls = {}
        self._closing = self._closed = self._close_running = False
        self.scheduler = self.expert_cache = None
        try:
            self._start(Path(model_dir), None if draft_dir is None else Path(draft_dir),
                        context=context, context_explicit=context_explicit, streams=streams,
                        vram_experts=vram_experts, vram_experts_pin=vram_experts_pin,
                        drafter_bits=drafter_bits, max_rows=max_rows, allow_copy=allow_copy, eos_token_ids=eos_token_ids,
                        chat_template_file=chat_template_file, chat_template_name=chat_template_name)
        except BaseException as primary:  # noqa: BLE001 -- Interrupts must also retire partial owners; rollback reraises.
            rollback(self, primary, self.close)

    def _start(self, model_dir, draft_dir, *, context, context_explicit, streams, vram_experts,
               vram_experts_pin, drafter_bits, max_rows, allow_copy, eos_token_ids, chat_template_file, chat_template_name):
        import torch

        from tensorfold.cuda import moe, precision, prompt_precision
        from tensorfold.cuda.capacity import admit, available_bytes, config, cuda_reserve_bytes, total_bytes, unified
        from tensorfold.cuda.chat_template import prepare_chat_template
        from tensorfold.cuda.geometry import prompt_row_bytes, prompt_rows
        from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4
        from tensorfold.families.qwen3_5.cuda.nvfp4_load import weight_bytes
        from tensorfold.families.qwen3_5.cuda.nvfp4_memory import weight_estimate

        from ..dflash_metadata import DraftConfig, draft_geometry, draft_weights
        from ..ram_experts import layout, plan_pool, runtime_geometry
        from .weights import load

        if precision.mode() != precision.CHECKPOINT:
            raise ValueError("Qwen CT NVFP4 MoE currently requires --precision checkpoint and native FP4 activation math")
        if prompt_precision.fp8():
            raise ValueError("Qwen CT NVFP4 MoE uses the checkpoint's A4 activations; drop --prefill-fp8")
        if type(streams) is not int or streams < 1:
            raise ValueError("parallel streams must be a positive integer")
        text = config(model_dir)
        draft_config = DraftConfig.read(draft_dir) if draft_dir is not None else None
        rows = (draft_config.block if draft_config is not None else 16) if max_rows is None else max_rows
        minimum_rows = 2 if draft_config is not None else 1
        if type(rows) is not int or not minimum_rows <= rows <= (16 if streams > 1 else 128):
            raise ValueError("draft verification rows require at least 2 with a secondary model, at most 16 with "
                             "--parallel, or at most 128 on one stream")
        self.max_rows = self.tree_rows = rows
        bits = 16 if drafter_bits == 0 else drafter_bits
        self.chat_template = prepare_chat_template(model_dir, template_file=chat_template_file,
                                                   template_name=chat_template_name)
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability(0)[0] != 12:
            raise ValueError("Qwen CT NVFP4 MoE requires a CUDA SM 12.x GPU for the checkpoint's native A4 math")
        self.torch = torch
        self.tp, self.rank, self.allow_copy = 1, 0, bool(allow_copy)
        self.vision = self.draft = self.room = None
        # MoE metadata has no dense intermediate_size. Prompt sizing charges
        # the actual routed activation width instead of inventing a dense MLP.
        prompt_text = {**text, "intermediate_size": text["num_experts_per_tok"] * text["moe_intermediate_size"]}
        chunk = prompt_rows(total_bytes(torch), prompt_row_bytes(prompt_text))
        # Resident native A4 banks plan routes on the device and retain that scratch; one resident stream also
        # verifies its chain rounds in CUDA graphs over a fixed state. Decide residency with those charges.
        keep = KEEP if streams > 1 else KEEP_ONE
        runtime = runtime_geometry(text, streams=streams, keep=keep, max_rows=rows, prompt_rows=chunk,
                                   graphs=streams == 1, planned=True)
        draft_runtime = (draft_geometry(config(draft_dir), streams=streams,
                                        kept=KEEP + 1 if streams > 1 else KEEP_ONE,
                                        block=rows, prompt_rows=chunk) if draft_dir is not None else None)
        resident, host_weight_staging = weight_estimate(model_dir, weight_bytes)
        draft_held = draft_weights(draft_dir, bits=bits).resident if draft_dir is not None else 0
        minimum = runtime.geometry.needed(1) + (draft_runtime.needed(1) if draft_runtime is not None else 0)
        automatic_offload = vram_experts is None and resident.resident + draft_held + minimum > available_bytes(torch)
        plan = layout(model_dir, "auto" if automatic_offload else vram_experts) if (
            automatic_offload or vram_experts is not None) else None
        if plan is not None and unified(torch):
            raise ValueError("Qwen CT NVFP4 host expert pools currently require discrete CUDA memory; "
                             "on unified-memory GPUs use resident experts without --vram-experts")
        graphs = streams == 1 and plan is None
        if plan is not None:            # leased host-pool banks: host windows, no device plan, no graphs
            runtime = runtime_geometry(text, streams=streams, keep=keep, max_rows=rows, prompt_rows=chunk)
        transform = plan.transform(weight_bytes) if plan is not None else weight_bytes
        host_extra = (plan.loading_bytes + plan.staging_bytes + plan.metadata_host_bytes + plan.control_host_bytes
                      + plan.source_filesystem_bytes
                      if plan is not None else 0)
        if plan is not None:
            from tensorfold.cuda.allocation_bound import pool_allocation_bound
            from tensorfold.cuda.expert_pin import ExpertPinPolicy
            from tensorfold.cuda.nvfp4.cache_a4 import A4ExpertCache

            before_pool = torch.cuda.memory._snapshot()
            pins = ExpertPinPolicy(plan.host_bytes, mode=vram_experts_pin,
                                   future_bytes=host_extra + host_weight_staging,
                                   allocation_bytes=plan.authority_allocations)
            try:
                self.expert_cache = A4ExpertCache(plan.entry_bytes, plan.entry_bytes, "cuda", pin_policy=pins,
                                                  width=text["moe_intermediate_size"], dims=text["hidden_size"])
            except BaseException as primary:  # noqa: BLE001 -- Preserve pinned ownership through interrupt rollback.
                rollback(pins, primary, pins.close)
            allocate = pool_allocation_bound(torch, self.expert_cache._pool, before_snapshot=before_pool)
            allocator = allocate.receipt
        # The observed bootstrap allocation is already excluded from available
        # memory. Explicit pool bytes are still FUTURE until transactional
        # replacement; auto chooses its final size after fixed weights load.
        self.capacity_plan = admit(
            model_dir, context, context_explicit, torch, runtime.geometry, transform,
            weight_estimator=partial(weight_estimate, transform=transform),
            host_resident=plan.host_bytes if plan is not None else 0, host_extra_staging=host_extra,
            resident_extra=allocate(plan.gpu_bytes) if plan is not None and not plan.automatic else 0,
            draft_dir=draft_dir, draft_weights=partial(draft_weights, bits=bits), draft_geometry=draft_runtime)
        self.context_window = self.capacity_plan["context_window"]
        if draft_config is not None and self.context_window > draft_config.rope.context_limit:
            raise ValueError("target context exceeds this DFlash checkpoint's declared rotary context")
        # Resident banks carry a routed-form A4 shared expert as their last cell (one device plan per layer);
        # any other shared expert loads as separate projections.
        factory = partial(load_host_experts_a4, cache=self.expert_cache, device="cuda", swiglu=moe.swiglu_bf16,
                          shared=self.expert_cache is None)
        self.w = load(model_dir, expert_factory=factory,
                      **({"eos_token_ids": eos_token_ids} if eos_token_ids is not None else {}))
        self.w.prompt_rows = chunk
        if draft_dir is not None:
            from .dflash import STOP_SCORE, DFlash

            # The proposal cut was selected for one stream; with --parallel the shared tree allocator already sizes
            # each stream's window from the same drafter scores and the calibrated verify cost.
            self.draft = DFlash(draft_dir, self.w, context_limit=self.context_window, block=rows, bits=bits,
                                stop_score=STOP_SCORE if streams == 1 else None)
        if plan is not None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            allocate.validate_current()
            components = runtime.components(self.capacity_plan["cache_slots"])
            self.expert_pool_plan = plan_pool(
                plan, raw_free_bytes=int(torch.cuda.mem_get_info()[0]),
                existing_pool_bytes=self.expert_cache.gpu_bytes, **components,
                future_draft_workspace_bytes=draft_runtime.bytes_at(self.capacity_plan["cache_slots"])
                if draft_runtime is not None else 0,
                device_reserve_bytes=cuda_reserve_bytes(total_bytes(torch)), allocation_size=allocate)
            self.expert_pool_plan["allocator"] = allocator
            self.expert_layout = plan.receipt()
            if self.expert_pool_plan["replace_pool"]:
                self.expert_cache.configure_before_use(self.expert_pool_plan["requested_pool_bytes"])
            print(f"[tensorfold] Qwen NVFP4 RAM experts: {self.expert_cache.gpu_bytes / 2**30:.2f} GiB, "
                  f"{self.expert_cache.capacity}/{plan.expert_count} cells; "
                  f"{plan.host_bytes / 2**30:.2f} GiB host authority; {vram_experts_pin} pin policy; "
                  f"{self.context_window}-token context", flush=True)
        self._prepare_serving(model_dir, streams=streams, keep=KEEP, max_rows=self.max_rows)
        if graphs:
            if not all(layer.moe is None or getattr(layer.moe.run, "capturable", False) for layer in self.w.layers):
                raise RuntimeError("resident Qwen CT expert banks must plan routes on the device")
            from .chain_graphs import ChainGraphs

            self.runner = ChainGraphs(self.w, self.capacity_plan["cache_slots"])
            # Capture every chain width's smallest-span graph before serving (larger spans still capture on first use).
            self.runner.prepare(range(1, rows + 1), taps=self.draft is not None)
        if plan is not None and self.multi is None:
            from tensorfold.cuda.streams import KVRoom

            self.room = KVRoom(self.cache, components["future_kv_bytes"])
        if self.multi is not None and self.multi.memory_gate is not None:
            # Its available_bytes live source already subtracts the selected
            # host/device floor. Do not subtract a second reserve here.
            self.multi.memory_gate.reserve = 0

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True, *,
                 vision=None, constraint=None, background=False, penalties=None):
        token = object()
        try:
            with self._lifecycle:
                if self._closing or self._closed:
                    raise RuntimeError("Qwen NVFP4 engine is closing or closed")
                if threading.get_ident() in self._calls.values():
                    raise RuntimeError("Qwen NVFP4 engine cannot recursively enter an active call")
                self._calls[token] = threading.get_ident()
            with nullcontext() if self.concurrent else self._serial:
                return super().generate(prompt, max_tokens, sampling, on_tokens, draft=draft, stop_eos=stop_eos,
                                        vision=vision, constraint=constraint, background=background,
                                        penalties=penalties)
        finally:
            with self._lifecycle:
                self._calls.pop(token, None)
                self._lifecycle.notify_all()

    def close(self):
        """Reject new calls, join the worker and retire every accepted consumer before unpinning."""
        lease = threading.Lock()
        with lease:
            try:
                with self._lifecycle:
                    scheduler = self.scheduler
                    if (threading.get_ident() in self._calls.values()
                            or threading.current_thread() is getattr(scheduler, "thread", None)):
                        raise RuntimeError("Qwen NVFP4 engine cannot close from an active call or decoding worker")
                    self._closing = True
                    while self._close_running:
                        if not self._close_running.locked():
                            self._close_running = False
                            break
                        self._lifecycle.wait(0.1)
                    if self._closed:
                        return
                    self._close_running = lease
                if scheduler is not None:
                    scheduler.close()
                    self.scheduler = None
                with self._lifecycle:
                    while self._calls:
                        self._lifecycle.wait()
                if hasattr(self, "torch"):
                    self.torch.cuda.synchronize()
                if self.expert_cache is not None:
                    self.expert_cache.close()
                with self._lifecycle:
                    self._closed = True
            finally:
                with self._lifecycle:
                    if self._close_running is lease:
                        self._close_running = False
                    self._lifecycle.notify_all()
