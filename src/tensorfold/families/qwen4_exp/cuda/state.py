"""Flash Next's static per-window buffers, so a step can be graph-captured, and one sequence's committed state."""

from __future__ import annotations

import traceback
import weakref

import numpy as np
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda import experts as grouped
from tensorfold.cuda import moe as moe_mod

from . import attention as attn_mod
from . import gdn as gdn_mod
from . import kvcache
from .weights import Weights


# -- per-window buffers --------------------------------------------------------------------------------
CAND = 32      # tensor parallel: candidates a rank gathers per row for sampling (top-k 20 plus the sampler's margin 8)
ATT_ROWS = 256  # prompt attention runs in blocks of this many rows (its partials scale with rows x context)
ENDS = 16       # prompts one prompt pass can end (each ending prompt's last row gets the head)


def execution_dtype(w: Weights):
    policy = w.meta.get("execution_dtype", getattr(w.cfg, "execution_dtype", "bfloat16"))
    if policy not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported Flash Next execution dtype {policy!r}")
    return getattr(torch, policy)


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    def __init__(self, w: Weights, rows: int, capacity: int, *, prefill: bool = False,
                 moe_prefill: bool | None = None, owner_role: str | None = None) -> None:
        c = w.cfg
        dev = w.device
        from .history_placement import for_weights

        placement = for_weights(w)
        wide = c.streams * c.hidden
        self.rows, self.prefill = rows, prefill
        self.rotor_lookup = False  # operation-owned constexpr selection; graphs and unqualified requests stay original
        self.owner_kernel_family = "torch-reference"
        self.owner_graph_family = "uncaptured"
        self.owner_graphs = None
        self.hc_plans = None
        head_rows = ENDS if prefill else rows
        bf, f32 = execution_dtype(w), torch.float32
        self.execution_dtype = bf
        from .padded_linears import create as padded_linears

        self.padded_linears = padded_linears(w, rows, prefill=prefill, dtype=bf)

        def row_tensor(name, shape, *, dtype=bf, zero=False):
            if self.padded_linears is not None:
                return self.padded_linears.tensor(name, shape, dtype, zero=zero)
            return (torch.zeros if zero else torch.empty)(shape, dtype=dtype, device=dev)
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.last = torch.zeros((rows,), dtype=torch.int32, device=dev)          # each stream's last row
        self.last_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        self.ple_late = None           # n-gram rows ``stage(late=True)`` left for the forward (forward.stage_late)
        self.split = None              # while a forward is captured in pieces: ends one at each per-step insert
        self.h = torch.empty((rows, wide), dtype=bf, device=dev)
        self.pss = torch.empty((rows, c.hidden // 256, c.streams), dtype=f32, device=dev)
        self.normed = row_tensor("normed", (rows, wide), dtype=bf)
        self.xs_normed = torch.empty((rows, wide // 32), dtype=f32, device=dev)
        self.dn = torch.empty((rows, c.low + c.streams), dtype=bf, device=dev)
        self.dn_mix = torch.empty((rows, c.low), dtype=bf, device=dev)      # a mixer's down (no inject rows)
        self.act = row_tensor("act", (rows, c.low), dtype=bf)
        self.xs_act = torch.empty((rows, c.low // 32), dtype=f32, device=dev)
        self.inj_a = torch.empty((rows, c.streams), dtype=bf, device=dev)
        self.inj_m = torch.empty((rows, c.streams), dtype=bf, device=dev)
        self.up = row_tensor("up", (rows, wide), dtype=bf)
        self.mixed = row_tensor("mixed", (rows, c.hidden), dtype=bf)
        self.xs_mixed = torch.empty((rows, c.hidden // 32), dtype=f32, device=dev)
        self.branch = row_tensor("branch", (rows, c.hidden), dtype=bf)
        attn_width = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
        self.pa = row_tensor("pa", (rows, attn_width), dtype=bf)
        self.q = torch.empty((rows, c.heads, c.head_dim), dtype=bf, device=dev)
        self.iq = torch.empty((rows, c.index_heads, c.index_dim), dtype=bf, device=dev)
        self.attn = attn_mod.AttnScratch(min(rows, ATT_ROWS) if prefill else rows, c.heads, c.head_dim, capacity,
                                         dev, budget=c.index_budget, ratio=c.index_ratio, dtype=bf,
                                         selection_only=placement is not None)
        self.gated = row_tensor("gated", (rows, c.heads * c.head_dim), dtype=bf)
        self.xs_gated = torch.empty((rows, c.heads * c.head_dim // 32), dtype=f32, device=dev)
        # the experts' prefill arithmetic for decode windows too (``moe_prefill``): their rows can share a pass's launch
        self.moe = moe_mod.MoEBuffers(rows, _MoECfg(c), dev, prefill=prefill if moe_prefill is None else moe_prefill,
                                      tile=grouped.PREFILL_TILE if prefill else (8 if bf == torch.float16 else grouped.window_tile(dev)))
        # DeltaNet projections and outputs and attention outputs; ``commit`` reads the projections' conv channels
        lin = 1 if prefill else sum(1 for layer in w.layers if layer.linear)     # a prompt chunk commits each layer
        self.proj = row_tensor("proj", (lin, rows, gdn_mod.widths(c.nk, c.nv)[1]), dtype=bf, zero=True)
        # Prompt chunks can exceed decode max_rows. Their temporary replay rows
        # are shared layer-by-layer; committed State scratch stays decode-sized.
        self.prefill_gdn_scratch = (gdn_mod.GDNScratch(rows, dev, c.nk, c.nv, dtype=bf)
                                   if prefill and bf == torch.float16 else None)
        if prefill:                    # the DeltaNet rows' conv taps ([conv state (3) | the chunk's rows]) and stream
            self.windows = (torch.arange(rows, dtype=torch.int32, device=dev)[:, None]
                            + torch.arange(c.conv_kernel, dtype=torch.int32, device=dev)).contiguous()
            self.sid = torch.zeros((rows,), dtype=torch.int32, device=dev)
            self.conv_ptr = torch.zeros((1,), dtype=torch.int64, device=dev)
            self.pos_blk = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.gout = row_tensor("gout", (rows, c.nv * gdn_mod.DV), dtype=bf)
        self.gxs = torch.empty((rows, c.nv * gdn_mod.DV // 32), dtype=f32, device=dev)
        self.attn_o = torch.empty((rows, c.heads, c.head_dim), dtype=bf, device=dev)
        self.streams = torch.empty((rows, wide), dtype=bf, device=dev)
        # fp32 logits: the head's fp32 sums reach the sampler unrounded (bf16 steps are 1/8 at logits 16-32)
        self.logits = row_tensor("logits", (head_rows, w.head.n), dtype=f32)
        world = int(w.meta.get("world", 1))
        self.world = world
        if world > 1:                  # tensor parallel: fp32 partials and their rank-ordered gathers
            self.part_branch = row_tensor("part_branch", (rows, c.hidden), dtype=f32)
            self.part_moe = torch.empty((rows, c.hidden), dtype=f32, device=dev)
            self.g_branch = torch.empty((world * rows * c.hidden,), dtype=f32, device=dev)
            self.g_moe = torch.empty((world * rows * c.hidden,), dtype=f32, device=dev)
            self.cand = torch.empty((head_rows, 2 * CAND + 1), dtype=f32, device=dev)
            self.cand_all = torch.empty((world * head_rows * (2 * CAND + 1),), dtype=f32, device=dev)
        # n-gram embedding
        nrow = rows * 2 * c.heads_per_ngram
        dh = c.ple_dim // (2 * c.heads_per_ngram)
        self.ple_w = torch.zeros((nrow, dh // 8), dtype=torch.int32, device=dev)
        self.ple_s = torch.zeros((nrow, dh // 32), dtype=bf, device=dev)
        self.ple_b = torch.zeros((nrow, dh // 32), dtype=bf, device=dev)
        pin = torch.cuda.is_available()
        self.ple_hw = torch.zeros((nrow, dh // 8), dtype=torch.int32, pin_memory=pin)
        self.ple_hs = torch.zeros((nrow, dh // 32), dtype=torch.int16, pin_memory=pin)
        self.ple_hb = torch.zeros((nrow, dh // 32), dtype=torch.int16, pin_memory=pin)
        self.ple_v = torch.zeros((nrow, dh), dtype=bf, device=dev)          # a bf16 table's rows, as they ship
        self.ple_hv = torch.zeros((nrow, dh), dtype=bf, pin_memory=pin)
        self.ple_emb = row_tensor("ple_emb", (rows, c.ple_dim), dtype=bf)
        self.xs_ple = torch.empty((rows, c.ple_dim // 32), dtype=f32, device=dev)
        self.ple_keys = row_tensor("ple_keys", (rows, wide), dtype=bf)
        self.ple_vals = row_tensor("ple_vals", (rows, c.hidden), dtype=bf)
        self.ple_gated = torch.empty((rows, wide), dtype=bf, device=dev)
        self.ple_pss = torch.empty((rows, c.streams), dtype=f32, device=dev)
        self.ple_nrow = torch.empty((rows, wide), dtype=bf, device=dev)
        # split-K partials for the largest matmul of a window (prompt chunks take K whole)
        self.part = torch.empty((1 if prefill else 32 * max(rows, 4) * 2560,), dtype=f32, device=dev)
        # MTP
        self.mtp_e = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.mtp_eo = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.mtp_hn = torch.empty((rows, wide), dtype=bf, device=dev)
        self.mtp_xh = torch.empty((rows, wide // 32), dtype=f32, device=dev)
        self.mtp_hs = torch.empty((rows * c.streams, c.hidden), dtype=bf, device=dev)
        self.mtp_in = torch.empty((rows, wide), dtype=bf, device=dev)          # the MTP's input streams
        draft = getattr(w, "dspark", None)
        self.dspark_taps = draft.new_taps(rows, dev) if draft is not None else None
        self.history_owner = None
        if placement is not None:
            from .history_owner import OwnerBuffers

            role = ("prefill" if prefill else "decode") if owner_role is None else owner_role
            prepared = w.meta.get("owner_workspaces")
            if prepared is None:
                self.history_owner = OwnerBuffers(w, min(rows, ATT_ROWS), self.attn, prefill=prefill)
            else:
                if not isinstance(prepared, dict) or set(prepared).difference({"decode", "mtp", "prefill"}) or role not in prepared:
                    raise ValueError("prepared owner workspaces require the exact declared execution role")
                workspace = prepared[role]
                if (not isinstance(workspace, OwnerBuffers) or not workspace._base_ready
                        or workspace.rows < min(rows, ATT_ROWS)
                        or workspace.heads != c.heads or workspace.head_dim != c.head_dim
                        or workspace.prefill != prefill
                        or workspace.selection.shape[1] != self.attn.idw + 2
                        or workspace.rank != w.meta["rank"] or workspace.comm is not w.comm
                        or workspace.queries.device != torch.device(dev)):
                    raise ValueError("prepared owner workspace differs from the actual model/rank execution envelope")
                self.history_owner = workspace


def _rows(t: torch.Tensor, rows: int, keep: int) -> torch.Tensor:
    """``t`` reallocated with ``rows`` rows, its first ``keep`` copied."""

    other = torch.zeros((rows, *t.shape[1:]), dtype=t.dtype, device=t.device)
    keep = min(keep, t.shape[0], rows)
    other[:keep] = t[:keep]
    return other


class _MoECfg:
    def __init__(self, c) -> None:
        self.num_experts_per_tok = c.top_k
        self.num_experts = c.experts
        self.moe_intermediate_size = c.moe_width
        self.hidden_size = c.hidden
        self.execution_dtype = getattr(c, "execution_dtype", "bfloat16")


# -- committed state -----------------------------------------------------------------------------------
class KVNumericError(RuntimeError):
    """A RotorQuant frame failed validation; its sequence requires explicit reset."""


class KVResizeError(RuntimeError):
    """A failed cache resize requires exclusive retirement before this slot can be reused."""


class State:
    """Committed caches of one sequence (and the MTP head's attention layer); grown by ``ensure`` up to ``limit``."""

    def __init__(self, w: Weights, capacity: int, max_rows: int, kv_dtype: str = "bf16", *,
                 limit: int | None = None, kv_pair=None, kv_key_dtype: str | None = None,
                 kv_value_dtype: str | None = None, slot: int = 0) -> None:
        c = w.cfg
        dev = w.device
        bf = execution_dtype(w)
        self.execution_dtype = bf
        from .history_placement import for_weights

        self.history_placement = for_weights(w)
        if type(slot) is not int or slot < 0:
            raise ValueError("history session slot must be a nonnegative integer")
        self.history_slot, self.history_rank = slot, int(w.meta.get("rank", 0))
        self.history_comm = getattr(w, "comm", None) if self.history_placement is not None else None
        self.history_generation = self.slot_generation = 0
        self.history_generations = (torch.zeros((2,), dtype=torch.int64, device=dev)
                                    if self.history_placement is not None else None)
        self.history_native_readers = weakref.WeakSet()
        self._history_error = None
        self.indexer_history_mode = w.meta.get("indexer_history", "gpu")
        if self.indexer_history_mode not in ("gpu", "ram"):
            raise ValueError("indexer history must be gpu or ram")
        if self.indexer_history_mode == "ram" and (
                bf != torch.float16 or w.meta.get("execution_provider") != "sm70"):
            raise ValueError("RAM indexer history requires the GGUF SM70 FP16 execution provider")
        self.indexer_history_owner = None
        kvcache.check(kv_dtype)
        if kv_pair is not None:
            if type(kv_pair) is not kvcache.KVPairFormat:
                raise ValueError("kv_pair must be a registered KVPairFormat")
            if kv_key_dtype is not None or kv_value_dtype is not None:
                raise ValueError("explicit KV pair cannot be combined with side dtype overrides")
            if kv_dtype != "bf16" and kv_pair != kvcache.get_pair(kv_dtype):
                raise ValueError("KV dtype shorthand conflicts with the explicit pair")
        self.kv_pair = kv_pair if kv_pair is not None else kvcache.get_pair(kv_dtype, kv_key_dtype, kv_value_dtype)
        if bf == torch.float16:
            from .sm70_kv import check_pair

            working_policy = check_pair(self.kv_pair)
        else:
            if any(side.compact is not None for side in (self.kv_pair.key, self.kv_pair.value)):
                raise ValueError("RotorQuant planar4/iso4 SM70 codec requires the explicit FP16 execution provider")
            working_policy = "stored-basis-native64-v1"
        self.kv_pair.row_bytes(c.kv_heads, c.head_dim)
        self.kv_key_dtype, self.kv_value_dtype = self.kv_pair.key_dtype, self.kv_pair.value_dtype
        self.kv_working_policy = working_policy
        self.kv_identity = (self.kv_pair.identity, self.kv_working_policy)
        if self.history_placement is not None:
            plan = self.history_placement
            self.kv_identity += (("unique-history-v1", plan.world, plan.global_kv_heads,
                                  plan.target_layers, plan.page_tokens),)
        if self.indexer_history_mode == "ram":
            self.kv_identity += (("indexer-history", "lossless-fp16-ram-archive-v1"),)
        self.kv_key_bits = 0 if self.kv_pair.key.bits == 16 else self.kv_pair.key.bits
        self.kv_value_bits = 0 if self.kv_pair.value.bits == 16 else self.kv_pair.value.bits
        self.kv_key_codec, self.kv_value_codec = self.kv_pair.key.codec, self.kv_pair.value.codec
        # Graph kernels only OR error bits and publish the first failing layer.
        # This allocation keeps its address through growth, restore and reset.
        self.kv_status = (torch.tensor([0, -2], dtype=torch.int32, device=dev)
                          if (not self.kv_pair.symmetric or self.kv_key_codec or self.kv_value_codec
                              or (bf == torch.float16 and self.kv_pair.key.quantized)
                              or self.indexer_history_mode == "ram"
                              or self.history_placement is not None) else None)
        world = int(w.meta.get("world", 1))
        self._kv_peer_status = (torch.empty((world, 2), dtype=torch.int32, device=dev)
                                if self.kv_status is not None and world > 1 else None)
        self._kv_pending = False
        self._kv_error: KVNumericError | KVResizeError | None = None
        self._resize_error: BaseException | None = None
        self._resize_frames_pending = None
        self.limit = int(capacity if limit is None else limit)     # the rows this sequence may grow to
        capacity = min(int(capacity), self.limit)
        self.capacity = capacity
        self.version = 0                 # counts reallocations: a graph's pointer table refreshes on a change
        self.pos = 0
        self.target_tentative_length = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        # text after an image prompt rotates at its cache position plus this offset (0: no images, the plain path)
        self.image_positions, self.image_rows, self.image_features = None, (), None
        self.rope_delta = 0
        self.rope_delta_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        lin = [layer for layer in w.layers if layer.linear]
        att = [layer for layer in w.layers if not layer.linear]
        self.lin_index = {layer.index: i for i, layer in enumerate(lin)}
        self.att_index = {layer.index: i for i, layer in enumerate(att)}
        n = len(lin)
        self.conv = torch.zeros((n, c.conv_kernel - 1, c.conv_dim), dtype=bf, device=dev)
        self.rec = torch.zeros((2, n, c.nv, c.dv, c.dk), dtype=torch.float32, device=dev)
        self.cur = [0] * n
        self.scratch = [gdn_mod.GDNScratch(max_rows, dev, c.nk, c.nv, dtype=bf) for _ in range(n)]
        self.ratio, self.index_dim = c.index_ratio, c.index_dim
        self.layers = len(att) + (w.mtp is not None)              # attention caches: the layers', the MTP head's
        if self.indexer_history_mode == "ram":
            session = w.meta.get("indexer_history_session")
            if session is None:
                raise ValueError("RAM indexer history requires its admitted host allocation session")
            if (session.plan.layers, session.plan.index_dim, session.plan.ratio) != (
                    self.layers, c.index_dim, c.index_ratio):
                raise ValueError("RAM indexer history plan disagrees with the model's attention geometry")
            # Host history owns the complete declared window from startup.
            # GPU KV growth never reallocates or aliases this independent owner.
            self.indexer_history_owner = session.new(self.limit)
            if not self.layers or self.indexer_history_owner.device_nbytes % self.layers:
                raise ValueError("raw indexer device allocation disagrees with its per-layer byte plan")
            self.indexer_raw_device_bytes = self.indexer_history_owner.device_nbytes // self.layers
        else:
            self.indexer_raw_device_bytes = None
        self.row_bytes = self.kv_pair.row_bytes(c.kv_heads, c.head_dim) + (
            c.index_dim * 2 if getattr(self, "indexer_history_owner", None) is None else 0)
        if self.history_placement is not None:
            from .history_owner import OwnedKVCache, allocate_target_caches

            def allocate(layer):
                return OwnedKVCache(capacity, c.kv_heads, c.head_dim, dev, pair=self.kv_pair,
                                    placement=self.history_placement, slot=slot, layer=layer, rank=self.history_rank)
            self.kc = allocate_target_caches(capacity, c.head_dim, dev, pair=self.kv_pair,
                                             placement=self.history_placement, slot=slot, rank=self.history_rank)
        else:
            self.kc = [kvcache.KVCache(capacity, c.kv_heads, c.head_dim, dev, pair=self.kv_pair) for _ in att]
        def indexer_rows(layer, rows):
            if self.history_placement is None:
                return rows
            return (rows if self.history_rank == self.history_placement.indexer_owner(
                slot, layer, include_mtp=w.mtp is not None) else 0)
        self.ikc = ([torch.zeros((indexer_rows(i, capacity), c.index_dim), dtype=bf, device=dev)
                     for i in range(len(att))]
                    if getattr(self, "indexer_history_owner", None) is None else
                    [self.indexer_history_owner.layer(i) for i in range(len(att))])
        nb = -(-capacity // c.index_ratio)
        self.pooled = [torch.zeros((indexer_rows(i, nb), c.index_dim), dtype=bf, device=dev)
                       for i in range(len(att))]
        wide = c.streams * c.hidden
        self.ple_tail = torch.zeros(((c.ple_kernel - 1) * c.ngram_size, wide), dtype=bf, device=dev)
        self.ple_history = c.ngram(0).initial_history() if c.ple_layers else None
        self.ple_last: tuple[np.ndarray, np.ndarray] | None = None
        # MTP head (its own attention cache; ``mtp_len`` entries, the last ``mtp_drafted`` of them chained drafts)
        self.mtp_len = 0
        self.mtp_drafted = 0
        self.mtp_pos = torch.zeros((1,), dtype=torch.int32, device=dev)
        if w.mtp is not None:
            self.mtp_kc = (allocate(len(att)) if self.history_placement is not None else
                           kvcache.KVCache(capacity, c.kv_heads, c.head_dim, dev, pair=self.kv_pair))
            self.mtp_ikc = (torch.zeros((indexer_rows(len(att), capacity), c.index_dim), dtype=bf, device=dev)
                            if getattr(self, "indexer_history_owner", None) is None else self.indexer_history_owner.layer(len(att)))
            self.mtp_pooled = torch.zeros((indexer_rows(len(att), -(-capacity // c.index_ratio)), c.index_dim), dtype=bf,
                                          device=dev)

        self.dspark_runtime = getattr(w, "dspark", None)
        self.dspark = None
        if self.dspark_runtime is not None:
            self.dspark_runtime.init_state(self)

    @property
    def kv_dtype(self) -> str:
        """Legacy symmetric format name; independent sides require their named properties."""

        if not self.kv_pair.symmetric:
            raise ValueError("mixed KV state has no single dtype; use kv_key_dtype/kv_value_dtype")
        return self.kv_pair.key_dtype

    def kv_begin(self) -> None:
        """Mark GPU writes unchecked; never clear a previous numeric failure."""

        error = (getattr(self, "_history_error", None) or
                 (self._kv_error if self._kv_error is not None else self._resize_error))
        if error is not None:
            raise error
        if self.kv_status is None:
            return
        self._kv_pending = True

    def kv_check(self) -> None:
        """Validate pending writes before publication, outside CUDA graph capture.

        A successful check consumes one two-word readback; subsequent checks
        without new GPU writes do not synchronize. Numeric failures require
        ``reset``; an ambiguous migration requires exclusive retirement and
        cannot be cleared by reset. CUDA/runtime failures propagate unchanged.
        """

        error = (getattr(self, "_history_error", None) or
                 (self._kv_error if self._kv_error is not None else self._resize_error))
        if error is not None:
            raise error
        if getattr(self, "indexer_history_owner", None) is not None:
            self.indexer_history_owner.check()
        status = self.kv_status
        if status is None:
            return
        if not self._kv_pending:
            return
        if status.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        self._kv_latch(*status.cpu().tolist())

    def kv_check_words(self, bits: int, layer: int) -> None:
        """``kv_check`` from status words a caller read back after this frame's last status write (its own wait):
        the same latch, without a readback of its own."""

        error = (getattr(self, "_history_error", None) or
                 (self._kv_error if self._kv_error is not None else self._resize_error))
        if error is not None:
            raise error
        if self.kv_status is None:
            return
        if self._kv_pending:
            self._kv_latch(bits, layer)

    def _kv_latch(self, bits: int, layer: int) -> None:
        self._kv_pending = False
        if bits:
            names = [name for bit, name in ((1, "Q"), (2, "K"), (4, "V"), (8, "RMS"), (16, "index"), (32, "descriptor"))
                     if bits & bit]
            self._kv_error = KVNumericError(
                f"K={self.kv_key_dtype}, V={self.kv_value_dtype} numeric validation failed: {','.join(names)} (bits={bits}), "
                f"layer={'MTP' if layer == -1 else layer}, pos={self.pos}, mtp_len={self.mtp_len}; "
                "this sequence requires reset")
            raise self._kv_error
        if getattr(self, "indexer_history_owner", None) is not None:
            try:
                # The consumed status readback follows this State's submitted
                # D2H stages on the producing stream. No additional wait.
                self.indexer_history_owner.validated(fenced=True)
            except BaseException as error:
                self._kv_error = error
                raise

    def kv_snapshot_metadata(self) -> dict:
        """Pinned codec identity and validated status for an in-memory snapshot."""

        self.kv_check()
        return {"kv_identity": self.kv_identity, "kv_status": (0, -2)}

    def kv_validate_snapshot(self, snap: dict) -> None:
        """Reject incompatible/unvalidated RotorQuant snapshots before mutation."""

        self.kv_check()
        if type(snap) is not dict:
            raise ValueError("a KV snapshot must be a dictionary")
        status = snap.get("kv_status")
        if (type(status) is not tuple or len(status) != 2 or any(type(word) is not int for word in status)
                or status != (0, -2) or snap.get("kv_identity") != self.kv_identity):
            raise ValueError("a KV snapshot needs matching ordered formats, working policy and validated status")
        if any(type(snap.get(name)) is not int or not 0 <= snap[name] <= self.capacity
               for name in ("pos", "mtp_len")):
            raise ValueError("KV snapshot positions must be integer rows within its sequence cache")

    def set_pos(self, pos: int) -> None:
        self.kv_check()
        if getattr(self, "indexer_history_owner", None) is not None:
            self._indexer_commit(pos, self.mtp_len)
        self.pos = pos
        self.target_tentative_length = pos
        self.pos_dev.fill_(pos)

    def discard_target_tentative(self) -> None:
        """Resolve rejected/uncommitted rows without advancing any target state."""
        self.kv_check()
        self.target_tentative_length = self.pos

    def _indexer_commit(self, pos: int, mtp_len: int) -> None:
        lengths = [pos] * len(self.kc) + ([mtp_len] if hasattr(self, "mtp_kc") else [])
        try:
            self.indexer_history_owner.commit(lengths)
        except BaseException as error:
            self._kv_error = error
            raise

    def _publish_indexer_positions(self, pos: int, mtp_len: int) -> None:
        """Publish a restored/reset pair without mixing old and new extents."""
        self._indexer_commit(pos, mtp_len)
        try:
            self.pos, self.mtp_len = pos, mtp_len
            self.target_tentative_length = pos
            self.pos_dev.fill_(pos)
            self.mtp_pos.fill_(mtp_len)
        except BaseException as error:
            self._kv_error = error
            raise

    def indexer_snapshot_metadata(self, pos: int, mtp_len: int) -> dict:
        """Exact raw-history extent after validation, including a cut before commit.

        Ordinary ``kv_snapshot_metadata`` remains the numeric-status authority.
        Raw bytes need only be available, not published at the main cursor yet.
        """
        if getattr(self, "indexer_history_owner", None) is None:
            return {}
        if any(type(value) is not int or not 0 <= value <= self.capacity for value in (pos, mtp_len)):
            raise ValueError("raw-history snapshot positions must fit this State's device cache")
        lengths = [pos] * len(self.kc) + ([mtp_len] if hasattr(self, "mtp_kc") else [])
        return {"indexer_history": self.indexer_history_owner.snapshot(lengths)}

    def cache_bytes(self, rows: int | None = None) -> int:
        """Bytes of the caches that grow with context at ``rows`` rows (now: ``capacity``), the MTP head's included."""

        if rows is None and getattr(self, "_resize_error", None) is not None:
            # A failed layer-at-a-time replacement can leave different layers
            # at different sizes. The old capacity is no longer an accounting
            # authority, even though no such geometry may be used for inference.
            return self.owned_cache_bytes()
        placement = getattr(self, "history_placement", None)
        if placement is None:
            total = self.layers * self.layer_bytes(rows)
        else:
            total = placement.rank_bytes(self.history_slot, self.capacity if rows is None else int(rows),
                                         self.history_rank, self.kv_pair.row_bytes(1, self.kc[0].head_dim),
                                         self.kv_pair.dummy_bytes, index_dim=self.index_dim,
                                         ratio=self.ratio, include_mtp=hasattr(self, "mtp_kc"))
        if getattr(self, "dspark_runtime", None) is not None:
            total += (self.capacity if rows is None else int(rows)) * self.dspark_runtime.plan.cache_row_bytes
        return total

    def owned_cache_bytes(self) -> int:
        """Actual attached cache storage; used only for failed resize accounting and retirement."""

        total = sum(cache.nbytes for cache in self.kc)
        total += sum(array.nbytes for array in self.pooled)
        total += (sum(array.nbytes for array in self.ikc) if getattr(self, "indexer_history_owner", None) is None else
                  self.indexer_history_owner.device_nbytes)
        if hasattr(self, "mtp_kc"):
            total += self.mtp_kc.nbytes + self.mtp_pooled.nbytes
            if getattr(self, "indexer_history_owner", None) is None:
                total += self.mtp_ikc.nbytes
        if getattr(self, "dspark", None) is not None:
            total += self.dspark.nbytes
        return total

    def layer_bytes(self, rows: int | None = None) -> int:
        """One attention layer's share of ``cache_bytes``: what a resize holds twice at once."""

        rows = self.capacity if rows is None else int(rows)
        placement = getattr(self, "history_placement", None)
        if placement is not None:
            return max(placement.layer_rank_bytes(
                self.history_slot, layer, rows, self.history_rank,
                self.kv_pair.row_bytes(1, self.kc[0].head_dim), self.kv_pair.dummy_bytes,
                index_dim=self.index_dim, ratio=self.ratio, include_mtp=hasattr(self, "mtp_kc"))
                for layer in range(self.layers))
        raw = getattr(self, "indexer_raw_device_bytes", None) or 0
        return rows * self.row_bytes + self.kv_pair.dummy_bytes + -(-rows // self.ratio) * self.index_dim * 2 + raw

    def layer_resize_bytes(self, rows: int) -> int:
        """Replacement GPU storage; the RAM owner's fixed ring/stages survive resize."""
        if getattr(self, "history_placement", None) is not None:
            return 0  # Full-window arena is immutable; migration has its separate bounded overlap.
        return self.layer_bytes(rows) - (getattr(self, "indexer_raw_device_bytes", None) or 0)

    def host_cache_bytes(self) -> int:
        return 0 if getattr(self, "indexer_history_owner", None) is None else self.indexer_history_owner.host_nbytes

    def ensure(self, rows: int, step: int = 8192) -> int:
        """Grow every context cache to hold ``rows`` rows (a ``step`` at a time, at most ``limit``); returns the new bytes."""

        self.kv_check()
        rows = int(rows)
        if rows <= self.capacity:
            return 0
        if rows > self.limit:
            raise ValueError(f"context of {rows} rows past this sequence's {self.limit}-row window")
        return self.resize(min(self.limit, -(-rows // step) * step))

    def resize(self, rows: int) -> int:
        """Reallocate the context caches at ``rows`` rows, keeping every committed row; returns the bytes it added."""

        self.kv_check()
        rows = int(rows)
        if getattr(self, "history_placement", None) is not None and rows != self.capacity:
            raise ValueError("unique histories retain fixed full-window arenas; context resize needs a separately admitted transition")
        if getattr(self, "history_placement", None) is not None:
            return 0
        before = self.cache_bytes()
        keep = max(self.pos, self.mtp_len)
        if rows < keep:
            raise ValueError(f"a {rows}-row cache can't keep {keep} committed rows")
        try:
            self._resize_caches(rows, self.pos, self.mtp_len)
            self.capacity = rows
            self.version += 1
            if getattr(self, "dspark_runtime", None) is not None:
                self.dspark_runtime.resize(self)
            return self.cache_bytes() - before
        except BaseException as error:
            self._poison_resize(error)
            raise

    def _resize_caches(self, rows: int, pos: int, mtp_len: int) -> None:
        """Keep the original component allocation/release order and one-layer copy-peak bound.

        Failure may leave individual components at different sizes; poison and
        actual attached-byte accounting handle that state without grouping new
        allocations and extending the old index/pool buffers' lifetimes.
        """

        blocks, kept_blocks = -(-rows // self.ratio), -(-max(pos, mtp_len) // self.ratio)
        def owner_rows(layer, count):
            placement = getattr(self, "history_placement", None)
            return count if placement is None or self.history_rank == placement.indexer_owner(
                self.history_slot, layer, include_mtp=hasattr(self, "mtp_kc")) else 0
        for i in range(len(self.kc)):
            self.kc[i] = self.kc[i].resized(rows, pos)
            if getattr(self, "indexer_history_owner", None) is None:
                self.ikc[i] = _rows(self.ikc[i], owner_rows(i, rows), pos)
            self.pooled[i] = _rows(self.pooled[i], owner_rows(i, blocks), kept_blocks)
        if hasattr(self, "mtp_kc"):
            self.mtp_kc = self.mtp_kc.resized(rows, mtp_len)
            if getattr(self, "indexer_history_owner", None) is None:
                self.mtp_ikc = _rows(self.mtp_ikc, owner_rows(len(self.kc), rows), mtp_len)
            self.mtp_pooled = _rows(self.mtp_pooled, owner_rows(len(self.kc), blocks), -(-mtp_len // self.ratio))

    def _poison_resize(self, error: BaseException) -> None:
        self._resize_error = error
        self._kv_error = None
        if self._resize_frames_pending is None:
            self._resize_frames_pending = error.__traceback__
        failures = []
        try:
            self._kv_error = KVResizeError("cache resize failed; this sequence requires exclusive cache retirement")
            self._kv_error.__cause__ = error
        except BaseException as diagnostic:
            failures.append(diagnostic)
        try:
            self.version += 1
        except BaseException as diagnostic:
            failures.append(diagnostic)
        # Retain the traceback's stack, while releasing failed helpers' local
        # replacement tensors (including KVCache.resized and _rows). Attached
        # arrays stay owned and charged until explicit retirement succeeds.
        try:
            traceback.clear_frames(self._resize_frames_pending)
        except BaseException as diagnostic:
            failures.append(diagnostic)
        else:
            self._resize_frames_pending = None
        if failures:
            raise_failures(error, failures)

    def recover_resize(self, w: Weights, rows: int) -> int:
        """Exclusively discard a failed sequence and rebuild its cache geometry.

        The decoder removes kept aliases before calling, then returns the slot
        to its free pool only on success. Fence pending copies/consumers before
        replacing storage. A failed fence/rebuild/reset retains poison and all
        attached ownership; its caller preserves the primary and cleanup errors.
        """

        if self._resize_error is None:
            raise RuntimeError("cache retirement requires a failed resize")
        if type(rows) is not int or not 0 < rows <= self.limit:
            raise ValueError("retired cache rows must be a positive integer within the slot limit")
        before = self.owned_cache_bytes()
        try:
            if self.pos_dev.is_cuda:
                torch.cuda.current_stream(self.pos_dev.device).synchronize()
            if self._resize_frames_pending is not None:
                # These frames may retain replacements not attached to State.
                # Do not allocate or publish another geometry until every old
                # temporary is retired after its stream's consumers complete.
                traceback.clear_frames(self._resize_frames_pending)
                self._resize_frames_pending = None
            self._resize_caches(rows, 0, 0)
            self.capacity = rows
            self._resize_error = self._kv_error = None
            self.reset(w)
            self.version += 1
            if getattr(self, "dspark_runtime", None) is not None:
                self.dspark_runtime.resize(self)
            return self.cache_bytes() - before
        except BaseException as error:
            self._poison_resize(error)
            raise

    def begin_sequence_generation(self) -> int:
        """Retire prior GPU readers before the first write of a new admission.

        The coordinator first resolves its deferred CPU replay/callback work.
        Native reader events cover the final consumer, even across streams.
        Ordinary same-request rollback does not advance this admission epoch.
        """
        if self._resize_error is not None:
            raise self._kv_error if self._kv_error is not None else self._resize_error
        if self._history_error is not None:
            raise self._history_error
        generation = self.slot_generation
        if type(generation) is not int or not 0 <= generation < 2**63 - 1:
            raise RuntimeError("history slot generation is invalid or exhausted; retire the slot without reuse")
        try:
            for reader in tuple(self.history_native_readers):
                reader.wait_native_reuse()
            generations = self.history_generations
            if generations is not None and generations.is_cuda:
                torch.cuda.current_stream(generations.device).synchronize()
            if generations is not None:
                generations[0].fill_(generation + 1)
            self.slot_generation = generation + 1
        except BaseException as error:
            # Poison existing fields before allocating diagnostic context. A
            # failed wrapper allocation cannot make ambiguous device writes
            # look healthy or permit this slot to be reused.
            self._history_error = self._kv_error = error
            try:
                failure = RuntimeError("history generation publication failed; this slot requires exclusive retirement")
                failure.__cause__ = error
            except BaseException as diagnostic:
                raise_failures(error, [diagnostic])
            self._history_error = self._kv_error = failure
            raise
        return self.slot_generation

    def reset(self, w: Weights, *, new_sequence: bool = True) -> None:
        """An empty sequence in the same buffers (captured graphs keep pointing at them)."""

        if self._resize_error is not None:
            raise self._kv_error if self._kv_error is not None else self._resize_error
        if getattr(self, "_history_error", None) is not None:
            raise self._history_error
        if type(new_sequence) is not bool:
            raise ValueError("reset admission generation policy must be explicit boolean")
        if new_sequence:
            self.begin_sequence_generation()
        if getattr(self, "indexer_history_owner", None) is not None:
            self.indexer_history_owner.reset()
        if self.kv_status is not None:
            self.kv_status.zero_()
            self.kv_status[1].fill_(-2)
            self._kv_pending = False
            self._kv_error = None
        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0] * len(self.cur)
        self.ple_tail.zero_()
        self.ple_history = w.cfg.ngram(0).initial_history() if w.cfg.ple_layers else None
        self.ple_last = None
        if getattr(self, "indexer_history_owner", None) is not None:
            self._publish_indexer_positions(0, 0)
        else:
            self.set_pos(0)
        self.set_rope_delta(0)
        self.image_positions, self.image_rows, self.image_features = None, (), None
        self.mtp_drafted = 0
        if getattr(self, "indexer_history_owner", None) is None:
            self.set_mtp_len(0)
        if getattr(self, "dspark_runtime", None) is not None:
            self.dspark_runtime.reset(self)

    def history_record(self) -> dict:
        """Published CPU scalars only; safe for observability without a GPU fence."""
        placement = getattr(self, "history_placement", None)
        result = {"layout": "replicated" if placement is None else "unique",
                  "slot": getattr(self, "history_slot", 0),
                  "slot_generation": getattr(self, "slot_generation", 0),
                  "ownership_generation": getattr(self, "history_generation", 0),
                  "target_cached_length": self.pos,
                  "target_tentative_length": getattr(self, "target_tentative_length", self.pos),
                  "draft_valid_length": self.mtp_len - self.mtp_drafted,
                  "draft_tentative_length": self.mtp_len,
                  "incomplete_pool_rows": self.pos % self.ratio,
                  "context_rows": self.capacity,
                  "history_poisoned": getattr(self, "_history_error", None) is not None,
                  "allocated_history_bytes": self.owned_cache_bytes()}
        if placement is not None:
            result.update(attention_degree=placement.degree,
                          mtp_attention_degree=4 if hasattr(self, "mtp_kc") else None,
                          page_tokens=placement.page_tokens, rank=self.history_rank,
                          qualification="source-only; SM70 distributed qualification pending")
            pending = getattr(self, "_history_migration", None)
            result.update(pending_attention_degree=None if pending is None else pending["target"].degree,
                          migration_blocked_reason=None if pending is None else pending.get("blocked_reason"),
                          migration_bytes=getattr(self, "history_migration_bytes", 0),
                          holder_ranks=sorted({owner for cache in self.kc for owner in cache.owners}))
            arena = self.kc[0].arena
            if arena is not None:
                result["target_page_arena"] = {"rank": arena.rank, "page_tokens": arena.page_class.tokens,
                                               "page_bytes": arena.page_class.nbytes,
                                               "page_capacity": arena.page_capacity,
                                               "reserved_pages": arena.reserved_pages,
                                               "available_pages": arena.free_pages}
        return result

    def history_page(self, layer: int, head: int, page: int) -> dict:
        """Canonical logical identity; relocation never participates in codec identity."""
        placement = self.history_placement
        if placement is None:
            raise ValueError("replicated history has no unique-owner page table")
        placement._indices(self.history_slot, layer, head, page)
        if layer == placement.target_layers and not hasattr(self, "mtp_kc"):
            raise ValueError("this profile has no MTP history page")
        start = page * placement.page_tokens
        if start >= self.capacity:
            raise ValueError("logical page is beyond this session's context commitment")
        committed = self.pos if layer < placement.target_layers else self.mtp_len - self.mtp_drafted
        tentative = (getattr(self, "target_tentative_length", self.pos)
                     if layer < placement.target_layers else self.mtp_len)
        return {"slot": self.history_slot, "slot_generation": self.slot_generation,
                "layer": layer, "global_head": head, "logical_start": start,
                "logical_rows": placement.page_rows(self.capacity, page),
                "physical_rows": placement.physical_page_rows(self.capacity, page),
                "committed_rows": min(placement.page_tokens, max(0, committed - start)),
                "tentative_rows": min(placement.page_tokens, max(0, tentative - start)),
                "codec_identity": self.kv_identity,
                "owner_rank": ((self.kc[layer] if layer < placement.target_layers else self.mtp_kc).owners[page] % 4
                               + head * 4),
                "ownership_generation": self.history_generation}

    def begin_history_migration(self, degree: int, *, overlap_bytes: int | None = None) -> None:
        from .history_owner import begin_migration

        begin_migration(self, degree, overlap_bytes=overlap_bytes)

    def migrate_history_bundle(self) -> bool:
        from .history_owner import migrate_next_bundle

        return migrate_next_bundle(self)

    def set_rope_delta(self, delta: int) -> None:
        self.rope_delta = int(delta)
        self.rope_delta_dev.fill_(int(delta))

    def clone(self) -> "State":
        """An independent copy (tests and A/B checks)."""

        self.kv_check()
        import copy

        other = copy.copy(self)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(other, name, value.clone())
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                setattr(other, name, [v.clone() for v in value])
            elif isinstance(value, kvcache.KVCache):
                setattr(other, name, value.clone())
            elif isinstance(value, list) and value and isinstance(value[0], kvcache.KVCache):
                setattr(other, name, [v.clone() for v in value])
        other.cur = list(self.cur)
        other.history_native_readers = weakref.WeakSet()
        # An independent oracle clone owns new cache allocations and actual
        # page maps, but cannot borrow a live transition's mutable progress.
        other._history_migration = None
        if getattr(self, "indexer_history_owner", None) is not None:
            other.indexer_history_owner = self.indexer_history_owner.clone()
            other.ikc = [other.indexer_history_owner.layer(i) for i in range(len(self.kc))]
            if hasattr(self, "mtp_kc"):
                other.mtp_ikc = other.indexer_history_owner.layer(len(self.kc))
        other.scratch = [copy.copy(sc) for sc in self.scratch]
        for sc_new, sc in zip(other.scratch, self.scratch):
            for name, value in vars(sc).items():
                setattr(sc_new, name, value.clone())
        if getattr(self, "dspark_runtime", None) is not None:
            self.dspark_runtime.clone_state(self, other)
        return other

    def set_mtp_len(self, n: int) -> None:
        self.kv_check()
        if getattr(self, "indexer_history_owner", None) is not None:
            self._indexer_commit(self.pos, n)
        self.mtp_len = n
        self.mtp_pos.fill_(n)

    def copy_prefix(self, source: "State", pos: int, mtp_len: int) -> None:
        """Copy only valid cache rows and complete pools; the caller restores the kept point's recurrent snapshot."""

        source.kv_check()
        self.kv_check()
        if getattr(self, "dspark_runtime", None) is not None:
            self.dspark_runtime.validate_copy(self, source, pos)
        if self is source or self.kv_identity != source.kv_identity or self.ratio != source.ratio:
            raise ValueError("a prefix copy needs distinct slots with matching cache formats")
        if (type(pos) is not int or type(mtp_len) is not int
                or not 0 <= pos <= min(self.capacity, source.pos)
                or not 0 <= mtp_len <= min(self.capacity, source.mtp_len)):
            raise ValueError("a prefix copy must fit the destination and the source's committed rows")
        if len(self.kc) != len(source.kc):
            raise ValueError("a prefix copy needs matching attention layers")
        pairs = list(zip(self.kc, source.kc))
        if mtp_len:
            pairs.append((self.mtp_kc, source.mtp_kc))
        for dst, src in pairs:
            if dst.pair != src.pair or (dst.kv_heads, dst.head_dim) != (src.kv_heads, src.head_dim):
                raise ValueError("a prefix copy needs matching cache formats, dimensions and rotation tables")
        if getattr(self, "history_placement", None) is not None:
            from .history_owner import copy_prefix

            copy_prefix(self, source, pos, mtp_len)
            if getattr(self, "dspark_runtime", None) is not None:
                self.dspark_runtime.copy_prefix(self, source, pos)
            return
        lengths = [pos] * len(self.kc) + ([mtp_len] if hasattr(self, "mtp_kc") else [])
        if getattr(self, "indexer_history_owner", None) is not None:
            # The owner validates both full archives before its first write.
            # Do this before any main-KV mutation, without validating twice.
            self.indexer_history_owner.copy_prefix(source.indexer_history_owner, lengths)
        def copy_cache(dst, src, rows):
            dst.k[:rows].copy_(src.k[:rows])
            dst.v[:rows].copy_(src.v[:rows])
            if dst.key_format.quantized:
                dst.ks[:rows].copy_(src.ks[:rows])
            if dst.value_format.quantized:
                dst.vs[:rows].copy_(src.vs[:rows])
        for i, cache in enumerate(self.kc):
            copy_cache(cache, source.kc[i], pos)
            if getattr(self, "indexer_history_owner", None) is None:
                self.ikc[i][:pos].copy_(source.ikc[i][:pos])
            self.pooled[i][:pos // self.ratio].copy_(source.pooled[i][:pos // self.ratio])
        if mtp_len:
            copy_cache(self.mtp_kc, source.mtp_kc, mtp_len)
            if getattr(self, "indexer_history_owner", None) is None:
                self.mtp_ikc[:mtp_len].copy_(source.mtp_ikc[:mtp_len])
            self.mtp_pooled[:mtp_len // self.ratio].copy_(source.mtp_pooled[:mtp_len // self.ratio])

        if getattr(self, "dspark_runtime", None) is not None:
            self.dspark_runtime.copy_prefix(self, source, pos)

    def snapshot(self) -> dict:
        """The committed state outside the cache rows; ``restore`` needs the cache rows below ``pos`` still in place."""

        metadata = self.kv_snapshot_metadata()
        p = self.cur[0] if self.cur else 0
        if any(c != p for c in self.cur):
            raise RuntimeError("DeltaNet layers out of step")
        history = self.indexer_snapshot_metadata(self.pos, self.mtp_len - self.mtp_drafted)
        return {"pos": self.pos, "rec": self.rec[p].clone(), "conv": self.conv.clone(),
                "ple_tail": self.ple_tail.clone(),
                "ple_history": None if self.ple_history is None else self.ple_history.copy(),
                "mtp_len": self.mtp_len - self.mtp_drafted, **metadata, **history,
                **(self.dspark_runtime.snapshot(self) if getattr(self, "dspark_runtime", None) is not None else {})}

    def restore(self, snap: dict) -> None:
        self.kv_validate_snapshot(snap)
        lengths = [snap["pos"]] * len(self.kc) + ([snap["mtp_len"]] if hasattr(self, "mtp_kc") else [])
        if getattr(self, "dspark_runtime", None) is not None:
            self.dspark_runtime.validate_snapshot(self, snap)
        if getattr(self, "indexer_history_owner", None) is not None:
            self.indexer_history_owner.restore(snap.get("indexer_history"), lengths)
        self.rec[0].copy_(snap["rec"])
        self.cur = [0] * len(self.cur)
        self.conv.copy_(snap["conv"])
        self.ple_tail.copy_(snap["ple_tail"])
        self.ple_history = None if snap["ple_history"] is None else snap["ple_history"].copy()
        self.ple_last = None
        if getattr(self, "indexer_history_owner", None) is not None:
            self._publish_indexer_positions(snap["pos"], snap["mtp_len"])
        else:
            self.set_pos(snap["pos"])
        self.set_rope_delta(0)                       # kept prompts are text only
        self.image_positions, self.image_rows, self.image_features = None, (), None
        self.mtp_drafted = 0
        if getattr(self, "indexer_history_owner", None) is None:
            self.set_mtp_len(snap["mtp_len"])
        if getattr(self, "dspark_runtime", None) is not None:
            self.dspark_runtime.restore(self, snap)
