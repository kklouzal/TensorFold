"""Gemma 4 requests with exact target-verified DFlash chains.

The single request lane owns persistent target and companion caches, reset or
restored per request. A single CUDA stream owns all requests. The prefix entry
includes the visual input identity and holds detached snapshots at prefill
positions (row-invariant forwards make a snapshot at any position equal to a
fresh prefill of that prefix): one at the whole prompt (with its next-token
logits) and one before the prompt's last turn marker, where the next chat turn's
history rendering departs from this turn's generation prompt.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

from .state import PREFILL_ROWS, widest_forward

PREFIX_SNAPSHOTS = 3     # resident prefix snapshots at most: an entry's two checkpoints + the next prompt's boundary
TURN_MARKER = "<|turn>"


@dataclass(frozen=True)
class _Checkpoint:
    position: int
    state: Any                  # detached target Snapshot at ``position``
    companion: Any              # detached DraftState snapshot, or None without a drafter
    logits: Any                 # next-token logits when ``position`` is the whole prompt, else None


@dataclass(frozen=True)
class _Prefix:
    tokens: tuple[int, ...]
    media: tuple
    checkpoints: tuple[_Checkpoint, ...]   # ascending positions; the last covers the whole prompt


def _prompt_ids(prompt, vocab: int) -> list[int]:
    if not isinstance(prompt, (list, tuple)) or not prompt:
        raise ValueError("Gemma4 needs a nonempty sequence of prompt token IDs")
    if any(type(token) is not int or not 0 <= token < vocab for token in prompt):
        raise ValueError("Gemma4 prompt token IDs must be integers inside the target vocabulary")
    return list(prompt)


def _prefill_end(start: int, length: int, width: int, blocks) -> int:
    """Never divide a visual block's bidirectional attention across forwards."""
    end = min(length, start + width)
    for begin, stop in blocks:
        if begin < start < stop:
            raise ValueError("Gemma4 cached prefix ends inside a visual block")
        if begin < end < stop:
            # End text before the block; starting on its first row may widen
            # to that block alone, never text+block beyond admission's bound.
            end = begin if begin > start else stop
            break
    return end


def _media_key(vision) -> tuple:
    if vision is None:
        return ()
    return (tuple(vision.image_hashes), tuple(vision.video_hashes), tuple(vision.blocks))


def _accepted(sampled, inputs, room: int, eos) -> list[int]:
    """Sample after the anchor; a proposal survives only its parent's choice."""
    chosen = []
    for row, token in enumerate(sampled):
        chosen.append(token)
        if len(chosen) >= room or token in eos or row + 1 >= len(inputs) or token != inputs[row + 1]:
            break
    return chosen


class Gemma4Engine:
    """Native CUDA target and DFlash (BF16, or private FP8 copies at --drafter-bits 8) on one exclusive CUDA
    stream."""

    tp = 1
    graphs = None          # text decode/verify CUDA graph runner (None: every forward eager)
    _clip_sampled_verification_to_room = False
    _tap_features = True
    _dflash_graph = False

    def __init__(self, model_dir: Path, *, drafter: str = "", drafter_bits: int = 16,
                 depth: int | None = None, context: int | None = None,
                 context_explicit: bool | None = None, streams: int = 1,
                 no_drafts: bool = False, vision: bool = False, eos_token_ids=None,
                 chat_template_file=None, chat_template_name=None,
                 prepare_global_prompt: bool = False, lossless_head: bool = False,
                 spark_global_gqa: bool = False, spark_sampled_q: bool = False, global_trig: bool = False,
                 global_trig_prompt: bool = True, delta4_head: bool = False, delta4_reuse: int = 16, global_cache: str = "none", global_cache_prompt: bool = False, dflash_graph: bool = False, dflash_head_native_a8: bool = False, dflash_head_w4a16: bool = False, clip_sampled_verification: bool = False) -> None:
        if type(clip_sampled_verification) is not bool:
            raise ValueError('clip_sampled_verification must be a bool')
        if clip_sampled_verification and (not drafter or no_drafts):
            raise ValueError('clipped sampled verification requires enabled DFlash')
        if type(dflash_head_w4a16) is not bool:
            raise ValueError('dflash_head_w4a16 must be a bool')
        if dflash_head_w4a16 and (drafter_bits != 8 or not drafter or no_drafts or dflash_graph or dflash_head_native_a8):
            raise ValueError('W4A16 private head requires enabled FP8 DFlash without draft graph/native A8')
        if type(dflash_head_native_a8) is not bool:
            raise ValueError('dflash_head_native_a8 must be a bool')
        if dflash_head_native_a8 and (drafter_bits != 8 or not drafter or no_drafts or depth != 4 or dflash_graph):
            raise ValueError('native A8 proposal head requires enabled FP8 DFlash depth4 without dflash_graph')
        if type(dflash_graph) is not bool:
            raise ValueError('dflash_graph must be a bool')
        self._dflash_graph = dflash_graph
        from .state import check_global_cache
        check_global_cache(global_cache, prompt=global_cache_prompt,
                           prepared_prompt=prepare_global_prompt, trig=global_trig)
        if type(delta4_head) is not bool:
            raise ValueError('delta4_head must be a bool')
        if type(delta4_reuse) is not int or delta4_reuse not in (1, 2, 4, 8, 16):
            raise ValueError('Delta4 reuse rows must be1/2/4/8/16')
        if not delta4_head and delta4_reuse != 16:
            raise ValueError('Delta4 reuse investigation requires delta4_head')
        if type(global_trig_prompt) is not bool:
            raise ValueError("global_trig_prompt must be a bool")
        if type(global_trig) is not bool:
            raise ValueError("global_trig must be a bool")
        if type(spark_sampled_q) is not bool:
            raise ValueError("spark_sampled_q must be a bool")
        if type(spark_global_gqa) is not bool:
            raise ValueError("spark_global_gqa must be a bool")
        if type(lossless_head) is not bool:
            raise ValueError("lossless_head must be a bool")
        if type(prepare_global_prompt) is not bool:
            raise ValueError("prepare_global_prompt must be a bool")
        if type(streams) is not int or streams != 1:
            raise ValueError("Gemma4 CUDA currently requires --parallel 1; concurrent lanes are not qualified")
        if type(drafter_bits) is not int or drafter_bits not in (8, 16):
            raise ValueError("Gemma4 CUDA DFlash runs BF16 (--drafter-bits 16) or private FP8 copies (8) "
                             "of the companion and its borrowed head")
        if depth is not None and (type(depth) is not int or not 1 <= depth <= 127):
            raise ValueError("Gemma4 DFlash draft depth must be between 1 and 127")
        if depth is not None and (not drafter or no_drafts):
            raise ValueError("Gemma4 draft depth requires an enabled DFlash companion")

        import torch
        from tensorfold.cuda.chat_template import prepare_chat_template
        from tensorfold.token_metadata import load_tokenizer, resolve_eos
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.families import quant_method
        from .capacity import admit, check_dflash_graph, check_dflash_head_a8, check_dflash_head_w4a16, companion_kind
        from .forward import Gemma4Target
        from .weights import Gemma4Weights

        raw = read_config(model_dir)
        method = quant_method(raw)
        if delta4_head and method != 'compressed-tensors':
            raise ValueError('Delta4 tied table is a CT-only investigation candidate')
        if global_cache != 'none' and method != 'compressed-tensors':
            raise ValueError('persistent global cache requires native CT arithmetic')
        if global_trig and method != 'compressed-tensors':
            raise ValueError('global trig requires CT arithmetic')
        draft_dir = Path(drafter) if drafter and not no_drafts else None
        kind = companion_kind(draft_dir) if draft_dir is not None else None
        if clip_sampled_verification:
            if kind != 'dflash':
                raise ValueError('clipped sampled verification requires a DFlash companion')
            # Resolve once during preparation; omitted policy preserves subclass defaults.
            self._clip_sampled_verification_to_room = True
        self._head_kind = kind
        if spark_sampled_q and (kind != 'sparkulator' or method != 'modelopt'):
            raise ValueError('Spark sampled q requires the actual Sparkulator/ModelOpt pair')
        if spark_global_gqa and (kind != 'sparkulator' or method != 'modelopt'):
            raise ValueError('Spark global GQA requires the actual Sparkulator/ModelOpt pair')
        if spark_global_gqa and (type(context) is not int or not 0 < context <= 131072):
            raise ValueError('Spark global GQA requires an explicit context up to131072')
        if spark_global_gqa:
            from ..sparkulator_metadata import validate_global_gqa_torch
            validate_global_gqa_torch(torch)
        spark_cfg = None
        if kind == 'sparkulator':
            from ..sparkulator_metadata import SparkulatorConfig
            spark_cfg = SparkulatorConfig.from_dict(read_config(draft_dir))
            if drafter_bits != 16:
                raise ValueError('Sparkulator baseline requires bits16')
        if lossless_head and method != 'modelopt':
            raise ValueError('lossless_head is a ModelOpt-only investigation candidate')
        if prepare_global_prompt and method not in ('compressed-tensors', 'modelopt'):
            raise ValueError('prepared prompt KV requires CT or ModelOpt arithmetic')
        if method == 'modelopt':
            from tensorfold.cuda import precision
            if precision.mode() != precision.CHECKPOINT:
                raise ValueError('Gemma4 ModelOpt MoE requires checkpoint activation math')
        torch.cuda.set_device(0)
        self.chat_template = prepare_chat_template(model_dir, template_file=chat_template_file,
                                                   template_name=chat_template_name)
        self.max_rows = (spark_cfg.block if spark_cfg is not None and depth is None else
                         1 if method == 'modelopt' and draft_dir is None else
                         16 if depth is None else depth + 1)
        check_dflash_graph(dflash_graph, drafter_bits, self.max_rows, kind)
        check_dflash_head_a8(dflash_head_native_a8, drafter_bits, self.max_rows, kind, dflash_graph)
        check_dflash_head_w4a16(dflash_head_w4a16, drafter_bits, kind, dflash_graph, dflash_head_native_a8)
        if spark_sampled_q and self.max_rows > 16:
            raise ValueError('Spark sampled q verification currently admits at most16 rows')
        if lossless_head and self.max_rows > 16:
            raise ValueError('lossless target head verification currently admits at most16 rows')
        self.capacity_plan = admit(model_dir, context, context_explicit, drafter_dir=draft_dir,
                                   vision=vision, max_rows=self.max_rows, streams=streams, keep=PREFIX_SNAPSHOTS,
                                   drafter_bits=drafter_bits,
                                   **({"prepare_global_prompt": True} if prepare_global_prompt else {}),
                                   **({"global_trig": True} if global_trig else {}),
                                   **({"global_cache": global_cache, "global_cache_prompt": global_cache_prompt}
                                      if global_cache != "none" else {}),
                                   **({"lossless_head": True} if lossless_head else {}),
                                   **({"delta4_head": True} if delta4_head else {}),
                                   **({"spark_global_gqa": True} if spark_global_gqa else {}),
                                   **({"spark_sampled_q": True} if spark_sampled_q else {}),
                                   **({'dflash_graph': True} if dflash_graph else {}),
                                   **({'dflash_head_native_a8': True} if dflash_head_native_a8 else {}),
                                   **({'dflash_head_w4a16': True} if dflash_head_w4a16 else {}))
        self.context_window = self.capacity_plan["context_window"]
        self.max_rows = min(self.max_rows, self.context_window)
        text = raw.get("text_config") or raw
        self.eos = resolve_eos(model_dir, text.get("eos_token_id"),
                               vocab_size=text["vocab_size"], override=eos_token_ids)
        if method == "compressed-tensors":
            from .nvfp4_weights import load
            from .modelopt_weights import ModelOptLoadStatus
            self._ct_load_status = ModelOptLoadStatus()
            try:
                self.w = load(model_dir, vision=vision, status=self._ct_load_status,
                              **({'delta4_head': True, 'delta4_reuse': delta4_reuse} if delta4_head else {}))
            except BaseException as primary:
                if not self._ct_load_status.safe_to_release:
                    self._retain_modelopt_failure(primary)
                raise
        elif method == 'modelopt':
            from .modelopt_weights import load, ModelOptLoadStatus
            self._modelopt_load_status = ModelOptLoadStatus()
            try:
                self.w = load(model_dir, vision=vision, status=self._modelopt_load_status,
                              **({"lossless_head": True} if lossless_head else {}))
            except BaseException as primary:
                if not self._modelopt_load_status.safe_to_release:
                    self._retain_modelopt_failure(primary)
                raise
        else:
            self.w = Gemma4Weights.load(model_dir, vision=vision)
        try:
            self.target = Gemma4Target(
                self.w,
                **({"prompt_context": self.context_window} if prepare_global_prompt or global_trig else {}),
                **({"prepare_global_prompt": True} if prepare_global_prompt else {}),
                **({"global_trig": True, "global_trig_prompt": global_trig_prompt} if global_trig else {}),
                **({"global_cache": global_cache, "global_cache_prompt": global_cache_prompt}
                   if global_cache != "none" else {}))
            self.head = None
            self.vision = None
            longest = 0
            if vision:
                from .vision import GemmaCudaVision, GemmaVisionProcessor, processor_geometry
                self.vision = GemmaCudaVision(self.w.raw_config, self.w.vision, self.w.projector,
                                              frontend=GemmaVisionProcessor.from_directory(model_dir))
                media = processor_geometry(model_dir)
                longest = max(media.image_soft_tokens, media.video_soft_tokens)
            forward_rows = widest_forward(self.max_rows, longest)
            if draft_dir is not None and kind != 'sparkulator':
                # Before any cache: the companion's load transients are admission staging.
                from .dflash import DFlash
                self.head = DFlash(draft_dir, self.target, context_limit=self.context_window,
                                   block=self.max_rows, bits=drafter_bits, rows=forward_rows,
                                   **({'dflash_graph': True} if dflash_graph else {}),
                                   **({'dflash_head_native_a8': True} if dflash_head_native_a8 else {}),
                                   **({'dflash_head_w4a16': True, 'head_source_dir': model_dir} if dflash_head_w4a16 else {}))
            # Spark borrows this exact main state; its native loading status is
            # caller-owned before construction, including an unreturned head.
            self._state = self.target.make_state(self.context_window, rows=forward_rows)
            if kind == 'sparkulator':
                from .sparkulator import Sparkulator, LoadStatus
                self._spark_load_status = LoadStatus()
                self.head = Sparkulator(draft_dir, self.target, self.context_window,
                                        block=self.max_rows, status=self._spark_load_status,
                                        **({"global_gqa": True} if spark_global_gqa else {}),
                                        **({"sampled_q": True} if spark_sampled_q else {}))
                self.head.bind_state(self._state)
            self._draft = self.head.make_state() if self.head is not None else None
            self._tap_features = not (self.head is not None and
                                     getattr(self.head, 'feature_source', 'taps') == 'final_hidden')
            from .graphs import Graphs
            # Text decode/verify rounds replay CUDA graphs over the engine-lifetime state; prefill stays eager. Every
            # row count's smallest-grid graph is captured now (before the load event below orders serving after it).
            # Larger attention grids capture on first use; their outputs and pool are covered by admission.
            self.graphs = Graphs(self.target, self._state, max_rows=self.max_rows)
            self.graphs.prepare(taps=(False, True) if self.head is not None and self._tap_features else (False,))
            # Absent from a tokenizer: only whole-prompt checkpoints are kept.
            self._turn = load_tokenizer(model_dir).token_to_id(TURN_MARKER)
            self._prefix = None
            self._prefix_lock = threading.Lock()
            self._run_lock = threading.Lock()
            self._closed_lock = threading.Lock()
            self._closed = False
            self._owner = None
            self._stream = torch.cuda.Stream(device=self.target.device)
            self.concurrent = False
            self.scheduler = None
            # All model uploads/packing occur on the constructing stream. Publish
            # them before a request can read weights from an independent lane.
            self._loaded = torch.cuda.Event()
            self._loaded.record()
            self._stream.wait_event(self._loaded)
            # Return load-time staging blocks to the device: the persistent caches are allocated, and the
            # serving snapshots need large blocks that these fragments cannot satisfy (admission counts no cache).
            torch.cuda.empty_cache()
        except BaseException as primary:
            if method in ('modelopt', 'compressed-tensors') or kind == 'sparkulator':
                # Unknown constructor/graph retirement is terminal for this
                # candidate process; retain actual owners, never retry CUDA.
                self._retain_modelopt_failure(primary)
            raise

    def _retain_modelopt_failure(self, primary):
        from tensorfold.cleanup import rollback
        _native_cause = BaseException.__cause__.__get__(primary)
        _native_context = BaseException.__context__.__get__(primary)
        authority = primary
        try:
            def unresolved():
                raise authority
            rollback(self, primary, unresolved)
        except BaseException as transport:
            if transport is not primary:
                raise primary from transport
            raise

    @contextmanager
    def _lane(self, background: bool):
        """The server's Turns owns priority; direct engine calls are exclusive."""
        if self._owner == threading.get_ident():
            raise RuntimeError("Gemma4 generation cannot reenter its active request callback")
        with self._run_lock:
            self._check_open()
            if self._dflash_graph:
                # Resolve the expected exception before any operation in this
                # lane; import failure must not replace a later native fault.
                from tensorfold.server.cancellation import RequestCancelled
            self._owner = threading.get_ident()
            try:
                yield self._stream
            except BaseException as primary:
                if self._dflash_graph:
                    # Request parameters are validated before entering this
                    # mutable GPU lane. Normal cooperative cancellation remains
                    # recoverable; an unexpected operation/readback fault may
                    # be the first observation of an asynchronous CUDA failure.
                    if not isinstance(primary, RequestCancelled):
                        graph = self.head._logits_graph
                        if graph.failure is None:
                            graph.failure = primary
                raise
            finally:
                self._owner = None

    def drain(self) -> None:
        """Wait for active request work before reclaiming its shared weights."""
        if self._owner == threading.get_ident():
            raise RuntimeError("cannot drain Gemma4 from its active request callback")
        with self._run_lock:
            graph = self.head._logits_graph if self._dflash_graph else None
            if graph is not None and (not graph.closed or graph.failure is not None):
                graph.check_live()
            try:
                self._stream.synchronize()
            except BaseException as primary:
                # Final appends can fault after the last sampling readback.
                # Never retry retirement after its first failed observation.
                if graph is not None and graph.failure is None:
                    graph.failure = primary
                raise

    def close(self) -> None:
        if self._owner == threading.get_ident():
            raise RuntimeError("cannot close Gemma4 from its active request callback")
        with self._closed_lock:
            self._closed = True
        self.drain()
        with self._prefix_lock:
            self._prefix = None
        if getattr(self, '_head_kind', None) == 'sparkulator' and self.head is not None:
            # The stream is known retired. Native head close owns no retry or
            # main-cache mutation; derived bridge storage may now be released.
            self.head.close()
            self._draft = None
        if self._dflash_graph and self.head is not None:
            self.head._logits_graph.close()

    def _check_open(self) -> None:
        with self._closed_lock:
            if self._closed:
                raise RuntimeError("Gemma4 engine is closed")
        if self._dflash_graph:
            self.head._logits_graph.check_live()

    def _taps(self, result, count=None):
        if getattr(self.head, 'feature_source', 'taps') == 'final_hidden':
            return result.hidden[:count]
        import torch
        return torch.cat([result.taps[layer][:count] for layer in self.head.tap_layers], dim=-1)

    def _append_head(self, state, result, input_ids, count=None, *, prompt=False):
        self.head.append(state, self._taps(result, count), prompt=prompt)

    def _find_prefix(self, prompt, media):
        """The longest checkpoint whose tokens prefix this prompt and leaves a forward or exact logits."""
        with self._prefix_lock:
            hit = self._prefix
            if hit is None or hit.media != media:
                return None
            for checkpoint in reversed(hit.checkpoints):
                p = checkpoint.position
                if (p < len(prompt) or (p == len(prompt) and checkpoint.logits is not None)) \
                        and tuple(prompt[:p]) == hit.tokens[:p]:
                    return checkpoint
            return None

    def _checkpoint(self, state, logits=None):
        return _Checkpoint(state.position, state.snapshot(),
                           self._draft.snapshot() if self._draft is not None else None,
                           None if logits is None else logits.clone())

    def _turn_boundary(self, prompt, cached: int, blocks) -> int:
        """Position of the prompt's last turn marker when a checkpoint there can serve the next turn, else 0."""
        if self._turn is None:
            return 0
        # From the end down to the cached prefix only: the last marker is normally a few tokens back.
        at = next((i for i in range(len(prompt) - 1, cached, -1) if prompt[i] == self._turn), 0)
        return 0 if any(begin < at < stop for begin, stop in blocks) else at

    def _prefill(self, prompt, vision, progress, started, *, cache_prompt=True):
        import torch
        media = _media_key(vision)
        hit = self._find_prefix(prompt, media)
        state = self._state
        cached = hit.position if hit is not None else 0
        logits = None
        if hit is not None:
            state.restore(hit.state)
            if self._draft is not None:
                self._draft.restore(hit.companion)
            logits = hit.logits
        else:
            state.reset()
            if self._draft is not None:
                self._draft.reset()
        encoded = None
        if vision is not None and cached < len(prompt):
            if self.vision is None:
                raise ValueError("Gemma4 image/video input requires --vision")
            encoded = self.vision.encode(vision, prompt)
        blocks = () if encoded is None else encoded.blocks
        boundary = self._turn_boundary(prompt, cached, blocks)
        kept = []
        start = cached
        try:
            while start < len(prompt):
                self._check_open()
                end = _prefill_end(start, len(prompt), PREFILL_ROWS, blocks)
                if start < boundary < end:
                    end = boundary
                ids = prompt[start:end]
                embeddings = None
                if encoded is not None:
                    visual = [(row, i) for i, row in enumerate(encoded.rows) if start <= row < end]
                    if visual:
                        tokens = torch.tensor(ids, dtype=torch.int64, device=self.target.device)
                        embeddings = self.target.embed_tokens(tokens)
                        at = torch.tensor([row - start for row, _ in visual], device=self.target.device)
                        take = torch.tensor([i for _, i in visual], device=self.target.device)
                        embeddings.index_copy_(0, at, encoded.features.index_select(0, take))
                result = self.target.forward(ids, state, embeddings=embeddings, vision_blocks=blocks,
                                             output_logits=False, output_taps=self.head is not None and self._tap_features,
                                             trusted_tokens=True, prompt=True)
                if self.head is not None:
                    self._append_head(self._draft, result, ids, prompt=True)
                if end == len(prompt):
                    logits = self.target.project_logits(result.hidden[-1:])
                elif end == boundary and cache_prompt:
                    kept.append(self._checkpoint(state))
                start = end
                if progress is not None and progress(len(prompt), cached, start,
                                                     (time.perf_counter() - started) * 1000):
                    from tensorfold.server.cancellation import RequestCancelled
                    raise RequestCancelled("request cancelled during Gemma4 prefill")
        except BaseException:
            # Nothing is published, and a traceback keeps this frame alive: release the unpublished
            # checkpoint so it never coexists with the next prefill's beyond PREFIX_SNAPSHOTS.
            kept.clear()
            raise
        if logits is None:
            raise RuntimeError("Gemma4 prefill produced no next-token logits")
        if cached < len(prompt) and cache_prompt:
            # All reads of the old entry have been issued on the same stream. Admission
            # reserves PREFIX_SNAPSHOTS: the old entry's two plus this prompt's boundary
            # checkpoint; evicting first keeps the whole-prompt snapshot within them.
            hit = None
            with self._prefix_lock:
                self._prefix = None
                kept.append(self._checkpoint(state, logits))
                self._prefix = _Prefix(tuple(prompt), media, tuple(kept))
        return state, logits, cached

    def generate(self, prompt: list[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None], draft: bool = True,
                 stop_eos: bool = True, *, vision=None, constraint=None,
                 background: bool = False, progress=None, penalties=None, cache_prompt: bool = True) -> dict[str, Any]:
        if type(cache_prompt) is not bool:
            raise ValueError("cache_prompt must be a bool")
        import torch
        from tensorfold.cuda.sampling import effective_rows, sample_rows
        from tensorfold.engine.penalties import History
        from tensorfold.engine.rejection import ChainPacket, verify

        prompt = _prompt_ids(prompt, self.target.config.vocab)
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Gemma4 max_tokens must be a positive integer")
        if len(prompt) + max_tokens > self.context_window:
            raise ValueError(f"Gemma4 prompt and reply exceed the {self.context_window}-token admitted context")
        if vision is not None and tuple(prompt[:len(vision.token_ids)]) != tuple(vision.token_ids):
            raise ValueError("Gemma4 prepared media does not belong to this prompt")
        history = History(penalties, prompt) if penalties is not None and penalties.active else None
        queued = time.perf_counter()
        with self._lane(background) as stream, torch.cuda.stream(stream), torch.inference_mode():
            started = time.perf_counter()
            state, logits, cached = self._prefill(prompt, vision, progress, started,
                                                  **({"cache_prompt": False} if not cache_prompt else {}))
            # Grammar masking is in place; immutable prefix logits are never
            # handed to it, including an exact full-prompt cache hit.
            logits = logits.clone() if constraint is not None else logits
            if constraint is not None:
                constraint.mask(logits)
            if history is not None:              # never in place on the immutable prefix entry's logits
                logits = history.apply(logits, copy=constraint is None)
            first = sample_rows(logits, [len(prompt)], sampling)[0]
            if constraint is not None:
                constraint.advance([first])
            if history is not None:
                history.extend([first])
            prefill_s = time.perf_counter() - started
            decoded = time.perf_counter()
            out = [first]
            rounds = drafted = accepted = 0
            min_rows = 0
            eos = self.eos if stop_eos else ()
            stopped = bool(on_tokens([first]))
            active_drafts = bool(draft and self.head is not None)
            collect_taps = active_drafts and self._tap_features
            stochastic = active_drafts and sampling is not None and sampling.temperature > 0
            while not stopped and len(out) < max_tokens and out[-1] not in eos \
                    and not (constraint is not None and constraint.finished):
                self._check_open()
                room = max_tokens - len(out)
                inputs = [out[-1]]
                packet = None
                if active_drafts and (stochastic or room > 1):
                    # Sampled noncausal proposal geometry belongs to this request's deployment, never its
                    # remaining output room. Even the final emitted slot tests the same full q packet.
                    limit = self.max_rows - 1 if stochastic else min(self.max_rows - 1, room - 1)
                    if stochastic:
                        chains = self.head.propose(self._draft, inputs[0], limit, history, sampled=True,
                                                   sampling=sampling)
                        first_position = state.position + 1
                        packet = (chains.sample(0, limit, sampling, first_position, history) if chains is not None
                                  else ChainPacket((), (), first_position))
                        guesses = packet.tokens
                    else:
                        chains = self.head.propose(self._draft, inputs[0], limit, history)
                        guesses = chains.finish(0, limit, history)[0] if chains is not None else []
                    chains = None  # synchronous packet readback completed; retire decoded launch staging now
                    if len(guesses) > limit or any(type(t) is not int or not 0 <= t < self.target.config.vocab
                                                   for t in guesses):
                        raise RuntimeError("Gemma4 DFlash returned an invalid proposal")
                    inputs.extend(guesses)
                if stochastic and self._clip_sampled_verification_to_room:
                    # E2B owns only the logical context. Unreachable target
                    # rows need no cache/RoPE capacity; the full q packet above
                    # remains unchanged, including its noncausal geometry.
                    inputs = inputs[:room]
                window = None
                if constraint is not None:
                    window = constraint.window(inputs, [-1, *range(len(inputs) - 1)])
                    inputs = window.tokens
                position = state.position
                if self.graphs is not None:
                    result = self.graphs.forward(inputs, state, output_taps=collect_taps)
                else:
                    result = self.target.forward(inputs, state, output_taps=collect_taps,
                                                 trusted_tokens=True)
                logits = result.logits
                if constraint is not None:
                    constraint.mask(logits, window)
                if history is not None:          # row r follows the reply and the drafts inputs[1..r]
                    logits = history.apply(logits, inputs)
                if stochastic:
                    # A linear grammar window retains a prefix. Only the unreachable original suffix is absent;
                    # a rejected/terminated proposal is tested under its parent's masked effective p row.
                    probabilities = effective_rows(logits, sampling)
                    probabilities.extend([None] * (len(packet.tokens) + 1 - len(inputs)))
                    path, terminal = verify(packet, probabilities, sampling.seed, room=room, eos=eos)
                    chosen = [inputs[row] for row in path[1:]] + [terminal]
                    del probabilities
                    packet = None
                else:
                    sampled = sample_rows(logits, list(range(position + 1, position + len(inputs) + 1)), sampling)
                    chosen = _accepted(sampled, inputs, room, eos)
                if history is not None:
                    history.extend(chosen)
                state.rollback(position + len(chosen))
                if active_drafts:
                    self._append_head(self._draft, result, inputs, len(chosen))
                if constraint is not None:
                    constraint.advance(chosen)
                rounds += 1
                drafted += len(inputs) - 1
                accepted += len(chosen) - 1
                min_rows = len(inputs) if not min_rows else min(min_rows, len(inputs))
                out.extend(chosen)
                stopped = bool(on_tokens(chosen))
            return {"prefill_s": round(prefill_s, 4), "decode_s": round(time.perf_counter() - decoded, 4),
                    "queue_s": round(started - queued, 4), "cached": cached, "drafts": active_drafts,
                    "rounds": rounds, "drafted": drafted, "accepted": accepted, "min_rows": min_rows}
