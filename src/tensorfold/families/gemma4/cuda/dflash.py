"""Owned Gemma DFlash proposals over persistent, target-accepted context.

The companion has causal sliding layers and a noncausal full layer by default.
Its noise block never enters committed context: a proposal writes the block's
keys and values into slots beyond the committed length, which the next
accepted context overwrites. The official Torch generator uses raw target
embedding rows; the borrowed target head owns softcap30 once.

The target verifies every proposal (probability-ratio rejection for sampled
requests), preserving its output law, so
the companion's arithmetic (BF16 row-reuse projections, or at bits 8 private
FP8 row-scaled copies of the projections and the borrowed head; kernel
geometry) must be deterministic for the same prefix, including when its prompt came from a partial cache hit.
Committed prompt projections therefore keep prompt arithmetic even for a tiny suffix. One DraftState belongs to
one request lane; its storage is allocated once for the admitted context.
"""

from __future__ import annotations

import gc
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl

from tensorfold.cleanup import rollback
from tensorfold.cuda import speculative
from tensorfold.cuda.b16_sym8_layout import SCHEMA as SYM8_SCHEMA
from tensorfold.cuda.b16_delta4 import SCHEMA as DELTA4_SCHEMA
from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.cuda.requant import Head, TokenFp8Head, prefix, requantize
from tensorfold.tensor_file import checkpoint_path
from tensorfold.families.qwen3_5.cuda import b16

from ..dflash_metadata import checked_headers, positive
from . import attention
from .capacity import DFLASH_FULL_CHUNK as FULL_CHUNK, check_dflash_graph, check_dflash_head_a8, check_dflash_head_w4a16
from .forward import rms_norm
from .state import PREFILL_ROWS, companion_rings, sliding_slots


class _LogitsGraph:
    """One graph generation, serialized by its DFlash caller on one stream.

    A geometry/state transition retires the old pool before capturing another.
    Only private outputs borrow that pool; callers receive an owning clone.
    Unknown native failures retain the entire generation and exception frames.
    """

    def __init__(self, head):
        self.head = head
        self.ids = torch.empty((head.block,), dtype=torch.int32, device=head.device)
        self.state = self.key = self.graph = self.output = self.pool = None
        self.failure = None
        self.closed = False

    def check_live(self):
        if self.failure is not None:
            raise RuntimeError('DFlash logits graph failed; retain its owner for process containment') from self.failure
        if self.closed:
            raise RuntimeError('DFlash logits graph is closed')

    def close(self):
        """The engine has drained its exclusive stream before releasing graph storage."""
        if self.closed:
            return
        self.check_live()
        self.closed = True
        self.state = self.key = self.graph = self.output = self.pool = self.ids = None

    def forward(self, state, anchor, length, slots):
        self.check_live()
        try:
            key = (length, slots)
            changed = self.state is not state or self.key != key
            stream = torch.cuda.current_stream(self.head.device)
            if changed:
                # A previous result's owning clone may still be consumed on this
                # stream. Complete it before retiring its source graph/pool.
                stream.synchronize()
                self.state = self.key = self.graph = self.output = self.pool = None
                self.state = state
            ids = self.ids[:length]
            ids.fill_(self.head.mask_id)
            ids[0] = anchor
            state.pos.fill_(state.length)

            def run():
                return self.head._logits_staged(state, self.head.raw_embed_tokens(ids), length, slots)

            if changed:
                self.pool = torch.cuda.graph_pool_handle()
                self.graph = torch.cuda.CUDAGraph()
                # The retained side stream is ordered both ways, including on
                # failure through the retained Python exception frame.
                warm = torch.cuda.Stream(device=self.head.device)
                warm.wait_stream(stream)
                with torch.cuda.stream(warm):
                    run()
                stream.wait_stream(warm)
                warm.synchronize()
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.graph(self.graph, pool=self.pool, capture_error_mode='thread_local'):
                        self.output = run()
                finally:
                    if enabled:
                        gc.enable()
                self.key = key
            self.graph.replay()
            return self.output.clone()
        except BaseException as error:
            # Publish to the existing field without entering another helper
            # that could fail before retaining the original native authority.
            if self.failure is None:
                self.failure = error
            raise


@triton.jit
def _draft_prep(SRC, STRIDE, QW, KW, FREQ, POS, QO, KC, VC, RING,
                H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    """Qwen3 companion head norms and BF16 RoPE for one (row, head) of fused [k | v | q] rows.

    Programs j < HK publish normalized, rotated keys and HK <= j < 2HK raw
    values into slot position % RING; later programs rotate query heads into QO.
    The arithmetic mirrors the Torch companion: BF16 normalization rounding,
    BF16 weight product, BF16 cos/sin and BF16 rounding of each RoPE product.
    """
    row = tl.program_id(0)
    j = tl.program_id(1)
    half: tl.constexpr = D // 2
    d = tl.arange(0, half)
    pos = tl.load(POS) + row
    src = SRC + row * STRIDE + j * D
    x1 = tl.load(src + d)
    x2 = tl.load(src + half + d)
    if (j >= HK) & (j < 2 * HK):
        dst = VC + ((pos % RING) * HK + j - HK) * D
        tl.store(dst + d, x1)
        tl.store(dst + half + d, x2)
    else:
        f1, f2 = x1.to(tl.float32), x2.to(tl.float32)
        inv = tl.rsqrt((tl.sum(f1 * f1, 0) + tl.sum(f2 * f2, 0)) / D + EPS)
        if j < HK:
            weight = KW
        else:
            weight = QW
        y1 = ((f1 * inv).to(tl.bfloat16).to(tl.float32) * tl.load(weight + d).to(tl.float32)).to(tl.bfloat16)
        y2 = ((f2 * inv).to(tl.bfloat16).to(tl.float32) * tl.load(weight + half + d).to(tl.float32)).to(tl.bfloat16)
        angle = pos.to(tl.float32) * tl.load(FREQ + d)
        c = tl.cos(angle).to(tl.bfloat16).to(tl.float32)
        s = tl.sin(angle).to(tl.bfloat16).to(tl.float32)
        y1, y2 = y1.to(tl.float32), y2.to(tl.float32)
        a = ((y1 * c).to(tl.bfloat16).to(tl.float32) + (-(y2 * s)).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        b = ((y2 * c).to(tl.bfloat16).to(tl.float32) + (y1 * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        if j < HK:
            dst = KC + ((pos % RING) * HK + j) * D
        else:
            dst = QO + (row * H + j - 2 * HK) * D
        tl.store(dst + d, a)
        tl.store(dst + half + d, b)


def _project(x: torch.Tensor, weight: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """BF16 projection; prompt rows retain the prompt GEMM regardless of a partial cache hit's row count."""
    m = x.shape[0]
    if prompt or m > 16:
        return b16.prompt(x, weight)
    return b16.matmul_rows(x, weight, max(2, 1 << (m - 1).bit_length()))


def _project8(x: torch.Tensor, linear, *, prompt: bool = False) -> torch.Tensor:
    """FP8-copy projection: explicit prompt rows retain prompt arithmetic even for a one-row cache suffix."""
    return linear.prefill(x) if prompt or x.shape[0] > 16 else linear(x)


def _packed_fp8_head(table, *, softcap):
    """DF-only private FP8 copy from exact RAW packed rows, never scaled rows.

    Named locals own every pending buffer through each successful fence. Any
    unexpected failure escapes without cleanup/drain retry; Engine retains the
    native exception frames and complete constructor owner until teardown.
    """
    from tensorfold.cuda.b16_sym8 import Sym8Table
    from .head import Delta4Table
    from tensorfold.cuda.requant import CHUNK_ROWS, requant_rows
    from tensorfold.cuda.nvfp4.linear import Fp8Linear
    if (not ((type(table) is Sym8Table and table.layout == SYM8_SCHEMA)
             or (type(table) is Delta4Table and table.layout == DELTA4_SCHEMA)) or table.k % 64
            or table.device.type != 'cuda'):
        raise ValueError('DF FP8 proposal copy requires an actual owned packed raw BF16 table')
    chunk = min(CHUNK_ROWS, table.n, table.lookup_limit)
    codes = scales = ids = raw_rows = part_codes = part_scales = linear = result = None
    # All destinations are final private FP8 source fields, not a BF16 mirror.
    codes = torch.empty((table.n, table.k), dtype=torch.float8_e4m3fn, device=table.device)
    scales = torch.empty((table.n,), dtype=torch.float32, device=table.device)
    for begin in range(0, table.n, chunk):
        end = min(table.n, begin + chunk)
        ids = torch.arange(begin, end, dtype=torch.int64, device=table.device)
        raw_rows = table.embedding(ids)
        part_codes, part_scales = requant_rows(raw_rows)
        codes[begin:end].copy_(part_codes)
        scales[begin:end].copy_(part_scales)
        torch.cuda.synchronize(table.device)
        ids = raw_rows = part_codes = part_scales = None
    linear = Fp8Linear.from_checkpoint(codes, scales)
    result = Head(linear, 'fp8', softcap=softcap)
    torch.cuda.synchronize(table.device)
    return result


def _cpu_nvfp4_head(model_dir, device, shape, *, softcap):
    """Original owned CPU BF16 -> one whole-tensor CPU recipe -> owned W4A16 head.

    CPU source/quantization finishes before GPU uploads. No FP8 source,
    per-chunk global scale, target-table mutation or full BF16 GPU mirror.
    Named destinations survive unknown native failures in constructor frames.
    """
    from tensorfold.cuda.requant import nvfp4_rows
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    # Finish the preceding FP8-transformer conversion before claiming this
    # CPU allocation phase is disjoint. A failed fence escapes, never retries.
    torch.cuda.synchronize(device)
    from tensorfold.checkpoint_metadata import read_json
    model_dir = Path(model_dir)
    name = 'model.language_model.embed_tokens.weight'
    index = read_json(model_dir, 'model.safetensors.index.json')
    mapping = None if index is None else index.get('weight_map')
    if index is not None and (not isinstance(mapping, dict) or not mapping
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items())
            or name not in mapping):
        raise ValueError('W4A16 original head requires its authoritative safetensors weight_map')
    file = checkpoint_path(model_dir, 'model.safetensors' if mapping is None else mapping[name])
    reader = SafeTensors([file])
    original = codes = scales = global_scale = gpu_codes = gpu_scales = linear = result = None
    try:
        original = reader.get(name)
        if original.device.type != 'cpu' or original.dtype != torch.bfloat16 or tuple(original.shape) != tuple(shape):
            raise ValueError('W4A16 private head requires the original BF16 CPU checkpoint head')
        codes, scales, global_scale = nvfp4_rows(original)
        original = None
    except BaseException as primary:
        try:
            reader.close()
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    else:
        reader.close()
    # The quantizer's returned fields own storage independently of the owned CPU
    # source. CPU temporaries/source are retired before this GPU packing phase.
    gpu_codes = codes.to(device=device)
    gpu_scales = scales.to(device=device)
    linear = Fp4Linear.from_checkpoint(gpu_codes, gpu_scales, float(global_scale), act=None)
    result = Head(linear, 'nvfp4', softcap=softcap)
    torch.cuda.synchronize(device)
    return result


class DraftState:
    """One lane's committed companion context: persistent per-layer K/V and its length.

    Sliding layers keep a ring of max(window + block, rows) slots (rounded to
    64): the window plus a proposal block, and one append's rows on distinct
    slots. The full layer keeps every position of the admitted window.
    ``length`` counts committed target positions; ``pos`` is its device copy.
    """

    def __init__(self, cfg, device, window: int, rows: int, block: int):
        self.rings = companion_rings(cfg.layer_types, cfg.sliding_window, block, rows, window)
        self.keys = [torch.empty((n, cfg.kv_heads, cfg.head_dim), dtype=torch.bfloat16, device=device)
                     for n in self.rings]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.rows, self.length = rows, 0
        self.layer_types = tuple(cfg.layer_types)
        self.pos = torch.zeros((1,), dtype=torch.int32, device=device)

    def reset(self) -> None:
        self.length = 0

    def _index(self, layer: int, length: int):
        count = min(length, self.rings[layer])
        return torch.arange(length - count, length, device=self.pos.device) % self.rings[layer]

    def snapshot(self):
        """Detached copies of every retained committed row, in position order."""
        held = []
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            index = self._index(i, self.length)
            held.append((k.index_select(0, index), v.index_select(0, index)))
        return self.length, tuple(held)

    def restore(self, snapshot) -> None:
        length, held = snapshot
        for i, (k, v) in enumerate(held):
            index = self._index(i, length)
            self.keys[i].index_copy_(0, index, k)
            self.values[i].index_copy_(0, index, v)
        self.length = length

    def cache_group_bytes(self, rows: int) -> dict[str, int]:
        import math
        if type(rows) is not int or not 0 < rows <= self.rows:
            raise ValueError('DFlash cache rows exceed the admitted append')
        return {group: rows * sum(math.prod(t.shape[1:]) * t.element_size()
                                 for i, kind in enumerate(self.layer_types) if (kind == 'sliding_attention') == sliding
                                 for t in (self.keys[i], self.values[i]))
                for group, sliding in (('draft_full', False), ('draft_window', True))}

    def cache_export_rows(self, start: int, stop: int, *, groups=None) -> dict:
        selected = ('draft_full', 'draft_window') if groups is None else tuple(groups)
        if (not selected or len(set(selected)) != len(selected)
                or any(group not in ('draft_full', 'draft_window') for group in selected)):
            raise ValueError('explicit supported companion archive groups required')
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= self.length
                or stop - start > self.rows
                or 'draft_window' in selected and any(start < max(0, self.length - ring)
                    for ring, kind in zip(self.rings, self.layer_types) if kind == 'sliding_attention')):
            raise ValueError('DFlash archive rows lack committed canonical append authority')
        result = {group: [] for group in selected}
        for i, kind in enumerate(self.layer_types):
            group = 'draft_window' if kind == 'sliding_attention' else 'draft_full'
            if group not in result:
                continue
            if group == 'draft_full':
                result[group].extend((self.keys[i][start:stop].clone(), self.values[i][start:stop].clone()))
            else:
                index = torch.arange(start, stop, device=self.pos.device) % self.rings[i]
                result[group].extend((self.keys[i].index_select(0, index), self.values[i].index_select(0, index)))
        return {group: tuple(values) for group, values in result.items()}

    def cache_import_rows(self, group: str, start: int, tensors: tuple, offset: int, rows: int) -> None:
        if (group not in ('draft_full', 'draft_window')
                or any(type(n) is not int for n in (start, offset, rows))
                or not 0 <= start < start + rows <= max(self.rings) or offset < 0):
            raise ValueError('DFlash archive import range differs')
        layers = [i for i, kind in enumerate(self.layer_types)
                  if (kind == 'sliding_attention') == (group == 'draft_window')]
        if len(tensors) != 2 * len(layers):
            raise ValueError('DFlash archive layer inventory differs')
        for ordinal, i in enumerate(layers):
            for destination, source in ((self.keys[i], tensors[2 * ordinal]),
                                        (self.values[i], tensors[2 * ordinal + 1])):
                value = source[offset:offset + rows]
                if group == 'draft_full':
                    destination[start:start + rows].copy_(value)
                else:
                    index = torch.arange(start, start + rows, device=self.pos.device) % self.rings[i]
                    destination.index_copy_(0, index, value)



class DFlash:
    """One shared companion; each lane owns a DraftState of accepted K/V only."""

    _logits_graph = None

    def __init__(self, draft_dir, target, *, context_limit, block=None, bits=16, rows=PREFILL_ROWS,
                 dflash_graph=False, dflash_head_native_a8=False, dflash_head_w4a16=False, head_source_dir=None):
        check_dflash_graph(dflash_graph, bits, 2 if block is None else block, 'dflash')
        check_dflash_head_a8(dflash_head_native_a8, bits, 2 if block is None else block, 'dflash', dflash_graph)
        check_dflash_head_w4a16(dflash_head_w4a16, bits, 'dflash', dflash_graph, dflash_head_native_a8)
        if dflash_head_w4a16 and head_source_dir is None:
            raise ValueError('W4A16 private head requires its authoritative original target checkpoint')
        if type(bits) is not int or bits not in (8, 16):
            raise ValueError("Gemma DFlash proposal weights are BF16 (16) or private FP8 row-scaled copies (8)")
        head = getattr(getattr(target, "weights", None), "embed", None)
        lossless = getattr(head, 'layout', None) in (SYM8_SCHEMA, DELTA4_SCHEMA)
        if bits == 8 and lossless:
            from tensorfold.cuda.b16_sym8 import Sym8Table
            from .head import Delta4Table
            if not ((type(head) is Sym8Table and head.layout == SYM8_SCHEMA)
                    or (type(head) is Delta4Table and head.layout == DELTA4_SCHEMA)):
                raise ValueError('DF FP8 proposal copy requires an actual owned packed raw BF16 table')
        elif bits == 8 and getattr(getattr(head, "weight", None), "dtype", None) != torch.bfloat16:
            raise ValueError("Gemma DFlash FP8 proposals need the target's BF16 tied head")
        cfg, _ = checked_headers(draft_dir)
        if ((target.config.hidden, target.config.layers, target.config.vocab)
                != (cfg.hidden, cfg.target_layers, cfg.vocab)):
            raise ValueError("Gemma DFlash hidden width, target layer count and vocabulary must match the target")
        if cfg.tap_layers != target.tap_layers:
            raise ValueError("Gemma DFlash target tap outputs must match the companion's trained layer IDs")
        self.window = positive(context_limit, "context limit")
        if self.window > cfg.rope.context_limit:
            raise ValueError("Gemma DFlash context exceeds the declared rotary context")
        self.block = cfg.block if block is None else positive(block, "block")
        if not 2 <= self.block <= 128:
            raise ValueError("Gemma DFlash blocks require 2 through128 rows, matching target tree verification")
        if self.block > cfg.sliding_window:
            raise ValueError("Gemma DFlash sliding blocks must fit the declared sliding window")
        check_dflash_graph(dflash_graph, bits, self.block, 'dflash')
        if (cfg.heads % cfg.kv_heads or attention.QUERY_TILE % (cfg.heads // cfg.kv_heads)
                or cfg.head_dim not in (64, 128, 256)):
            raise ValueError("Gemma DFlash attention needs a query group dividing 16 and head width 64/128/256")
        if dflash_head_native_a8:
            if (cfg.hidden, cfg.vocab) != (2816, 262144):
                raise ValueError('native A8 proposal head requires hidden2816/vocabulary262144')
            if tuple(torch.cuda.get_device_capability(target.device)) != (12, 1):
                raise ValueError('native A8 proposal head prototype requires SM121')
        self.rows = positive(rows, "context rows")
        # Kernel offsets are int32: cached key rows and full-layer chunk partials must index below 2**31.
        slots = max(companion_rings(cfg.layer_types, cfg.sliding_window, self.block, self.rows, self.window))
        if max(slots * cfg.kv_heads, self.block * cfg.heads * -(-self.window // FULL_CHUNK)) * cfg.head_dim >= 2**31:
            raise ValueError("Gemma DFlash attention offsets exceed int32 at this context")
        self.config, self.device, self.bits = cfg, target.device, bits
        self.hidden = cfg.hidden
        self.tap_layers, self.mask_id = cfg.tap_layers, cfg.mask_id
        self.raw_embed_tokens, self.project_logits = target.raw_embed_tokens, target.project_logits
        reader = SafeTensors([checkpoint_path(Path(draft_dir), "model.safetensors")])
        try:
            loaded = {name: reader.get(name).to(self.device).contiguous() for name in cfg.shapes()}
        except BaseException as primary:  # noqa: BLE001 - preserve primary and reader cleanup failures
            try:
                rollback(reader, primary, reader.close)
            except BaseException as transport:
                if transport is primary:
                    raise
                raise primary from transport
        else:
            reader.close()
        # Fuse projections that share an input: [k | v | q] (the [k | v] prefix
        # projects committed context) and [gate | up]. Fusion keeps every
        # output column's own dot product.
        for layer in range(cfg.layers):
            base = f"layers.{layer}."
            attn = base + "self_attn."
            loaded[base + "kvq"] = torch.cat([loaded.pop(attn + f"{p}_proj.weight") for p in "kvq"])
            loaded[base + "gate_up"] = torch.cat([loaded.pop(base + f"mlp.{p}_proj.weight") for p in ("gate", "up")])
        # Projections: BF16 on the row-reuse kernel, or (bits 8) private FP8 copies (tensorfold.cuda.requant: E4M3
        # codes, one FP32 scale an output row); with FP8 the borrowed head is an FP8 copy of the target's BF16 head
        # as well (the target keeps its own head). Each BF16 source is freed once its copy exists.
        names = list(cfg.fused_shapes())
        kv_width = 2 * cfg.kv_heads * cfg.head_dim
        self.linears = {}
        for name in names:
            weight = loaded.pop(name) if bits == 8 else loaded[name]
            copy = requantize(weight, "fp8") if bits == 8 else None
            self.linears[name] = partial(_project8, linear=copy) if bits == 8 else partial(_project, weight=weight)
            if name.endswith(".kvq"):
                kv = name[:-3] + "kv"
                self.linears[kv] = (partial(_project8, linear=prefix(copy, kv_width)) if bits == 8
                                    else partial(_project, weight=weight[:kv_width]))
            del weight, copy
        if bits == 8:
            self.project_logits = (_cpu_nvfp4_head(head_source_dir, self.device, (cfg.vocab, cfg.hidden),
                                                  softcap=target.config.softcap) if dflash_head_w4a16 else
                                   _packed_fp8_head(head, softcap=target.config.softcap) if lossless else
                                   Head(head, "fp8", softcap=target.config.softcap))
            if dflash_head_native_a8:
                self.project_logits = TokenFp8Head(self.project_logits)
        self.weights = loaded
        # Transformers5.6 Qwen3 default RoPE constructs the exponent in FP32.
        # Reusing Flash Next's FP64 default construction would change frequencies.
        exponent = torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim
        self.inv_freq = (1.0 / (cfg.rope.theta ** exponent)).to(self.device)
        if bits == 8:
            # Prepare only the providers selected by these validated private
            # copies. First-use compilation belongs to loading, before ready.
            from tensorfold.cuda.nvfp4.linear import _ext, _prompt_ext

            _prompt_ext()
            _ext()
            if dflash_head_native_a8:
                from tensorfold.cuda.nvfp4 import checkpoint

                checkpoint._ext()
                checkpoint.sm_count()
        self._logits_graph = _LogitsGraph(self) if dflash_graph else None

    def make_state(self) -> DraftState:
        if self._logits_graph is not None:
            self._logits_graph.check_live()
        return DraftState(self.config, self.device, self.window, self.rows, self.block)

    def _publish(self, fused, state, layer, query: torch.Tensor | None):
        cfg = self.config
        base = f"layers.{layer}.self_attn."
        programs = 2 * cfg.kv_heads + (cfg.heads if query is not None else 0)
        _draft_prep[(fused.shape[0], programs)](
            fused, fused.stride(0), self.weights[base + "q_norm.weight"], self.weights[base + "k_norm.weight"],
            self.inv_freq, state.pos, query if query is not None else fused, state.keys[layer], state.values[layer],
            state.rings[layer], H=cfg.heads, HK=cfg.kv_heads, D=cfg.head_dim, EPS=cfg.eps, num_warps=1,
            enable_fp_fusion=False)

    @torch.no_grad()
    def append(self, state: DraftState, taps: torch.Tensor, *, prompt: bool = False) -> None:
        """Commit accepted taps. Prompt rows always use prompt arithmetic, including a tiny cache suffix.

        Cached prompt K/V must equal fresh prefill K/V: sampled rejection depends on the companion's actual q.
        The lane projection and prompt GEMM have different bits, so row count cannot determine a prompt's role.
        Decode appends retain the existing fast lane projection for at most16 rows.
        """
        if self._logits_graph is not None:
            self._logits_graph.check_live()
        cfg = self.config
        if (not isinstance(taps, torch.Tensor) or taps.ndim != 2 or taps.shape[1] != len(self.tap_layers) * self.hidden
                or taps.dtype != torch.bfloat16 or taps.device != self.device
                or state.length + taps.shape[0] > self.window or taps.shape[0] > state.rows):
            raise ValueError("Gemma DFlash accepted taps must match the target and admitted positions")
        if not taps.shape[0]:
            return
        context = rms_norm(self.linears["fc.weight"](taps.contiguous(), prompt=prompt),
                           self.weights["hidden_norm.weight"],
                           cfg.eps)
        state.pos.fill_(state.length)
        for layer in range(cfg.layers):
            self._publish(self.linears[f"layers.{layer}.kv"](context, prompt=prompt), state, layer, None)
        state.length += taps.shape[0]

    @torch.no_grad()
    def propose(self, state: DraftState, anchor: int, count: int, history=None, *, sampled=False, sampling=None
                ) -> speculative.Chains | None:
        """Up to ``count`` proposals after the anchor from committed context only: the shared chain utility's
        argmax for greedy requests or actual sampled q candidates. Both use the preceding proposal path's
        penalties; sampled packets retain their real distribution for target rejection verification."""
        logits = self.logits(state, anchor, count)
        return None if logits is None else speculative.Chains(logits, [history], logits.shape[0], sampled=sampled,
                                                             sampling=sampling)

    @torch.no_grad()
    def logits(self, state: DraftState, anchor: int, count: int) -> torch.Tensor | None:
        """Softcapped FP32 companion logits [n, vocab] for block rows 1..n, n <= ``count``."""
        if self._logits_graph is not None:
            self._logits_graph.check_live()
        cfg = self.config
        if type(count) is not int or not 0 <= count <= 127:
            raise ValueError("Gemma DFlash proposal count must be an integer from zero through127")
        length = min(self.block, count + 1, self.window - state.length)
        if length < 2 or state.length == 0:
            return None
        if type(anchor) is not int or not 0 <= anchor < cfg.vocab:
            raise ValueError("Gemma DFlash anchors must be native target-vocabulary token IDs")
        slots = (state.length + length - 1) // FULL_CHUNK + 1
        if self._logits_graph is not None:
            return self._logits_graph.forward(state, anchor, length, slots)
        ids = torch.full((length,), self.mask_id, dtype=torch.int32, device=self.device)
        ids[0] = anchor
        x = self.raw_embed_tokens(ids)
        state.pos.fill_(state.length)
        return self._logits_staged(state, x, length, slots)

    def _logits_staged(self, state, x, length, full_slots):
        """The original arithmetic with position supplied by the persistent device scalar."""
        cfg = self.config
        # Noncausal layers see the whole block: every row shares [start, start + length).
        block_start = state.pos.repeat(length)
        block_end = block_start + length
        scale = cfg.head_dim ** -0.5
        group = cfg.heads // cfg.kv_heads
        for layer, kind in enumerate(cfg.layer_types):
            base = f"layers.{layer}."
            h = rms_norm(x, self.weights[base + "input_layernorm.weight"], cfg.eps)
            fused = self.linears[base + "kvq"](h)
            q = torch.empty((length, cfg.heads, cfg.head_dim), dtype=torch.bfloat16, device=self.device)
            self._publish(fused, state, layer, q)
            noncausal = not cfg.causal_layers[layer]
            if kind == "sliding_attention":
                window, chunk = cfg.sliding_window, attention.SLIDING_CHUNK
                slots = sliding_slots(window, length, group, length if noncausal else 0)
            else:
                window, chunk = 0, FULL_CHUNK
                slots = full_slots
            out = attention.sliding(q, state.keys[layer], state.values[layer], state.pos, block_start, block_end,
                                    window=window, slots=slots, blocks=noncausal, scale=scale, chunk=chunk)
            x = x + self.linears[base + "self_attn.o_proj.weight"](out.reshape(length, -1))
            h = rms_norm(x, self.weights[base + "post_attention_layernorm.weight"], cfg.eps)
            gate, up = self.linears[base + "gate_up"](h).split(cfg.width, dim=-1)
            x = x + self.linears[base + "mlp.down_proj.weight"](torch.nn.functional.silu(gate) * up)
        h = rms_norm(x[1:], self.weights["norm.weight"], cfg.eps)
        return self.project_logits(h).float()
