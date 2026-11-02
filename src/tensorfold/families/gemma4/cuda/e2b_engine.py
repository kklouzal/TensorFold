"""Single-lane dense Gemma E2B/E4B using TensorFold's existing request sampler."""
from __future__ import annotations

from pathlib import Path
import math
import threading
import time
from typing import Any, Callable

from .engine import Gemma4Engine, _Prefix
from .e2b_metadata import E2BConfig
from .e2b_target import E2BTarget

PREFILL_ROWS = 512
PREFIX_SNAPSHOTS = 3


def prefill_boundaries(prompt, turn, rows=PREFILL_ROWS, *, blocks=()):
    """Canonical native-GEMM shapes: every turn, then at most ``rows``.

    Extension may change the previous final partial chunk, so that checkpoint
    is reusable only when it is also a boundary of the extended prompt.
    """
    if type(rows) is not int or rows not in (512, 1024, 2048):
        raise ValueError("Gemma prefill rows must be512,1024 or2048")
    previous = 0
    for span in blocks:
        if (not isinstance(span, (tuple,list)) or len(span) != 2
                or any(type(value) is not int for value in span)
                or not previous <= span[0] < span[1] <= len(prompt)):
            raise ValueError('bidirectional vision blocks must be ordered nonoverlapping prompt spans')
        previous = span[1]
    stops = [i for i, token in enumerate(prompt) if i and token == turn] if turn is not None else []
    stops.append(len(prompt))
    ends = []
    start = 0
    for stop in stops:
        while start < stop:
            end = min(start + rows, stop)
            for lo, hi in blocks:
                if lo < end < hi:
                    end = lo if lo > start else hi
                    break
            start = end
            ends.append(start)
    return tuple(ends)


def companion_metadata(directory, target):
    """CPU-owned discriminator; no unrelated head may borrow target features."""
    if target.profile == "E2B":
        from ..e2b_dflash_metadata import checked_headers
        cfg, _ = checked_headers(directory)
        kind, width = "dflash2", cfg.hidden
    else:
        from tensorfold.checkpoint_metadata import read_config
        architecture = read_config(directory).get("architectures")
        if architecture == ["Gemma4AssistantForCausalLM"]:
            from ..assistant_metadata import checked_headers
            kind = "assistant"
        elif architecture == ["Eagle3DraftModel"]:
            from ..eagle3_metadata import checked_headers
            kind = "eagle3"
        else:
            raise ValueError("Gemma E4B requires its compatible native assistant or EAGLE3 metadata")
        cfg, _ = checked_headers(directory)
        width = cfg.backbone_hidden if kind == "assistant" else cfg.hidden
    if width != target.hidden or cfg.vocab != target.vocab:
        raise ValueError("Gemma companion target width/vocabulary differs")
    return kind, cfg


def assistant_q_host_bytes(rows, width, *, proposal_policy=False):
    """Actual full-support immutable q owners beside target p/residual work.

    One proposal row stores independent int64 IDs/FP64 weights in immutable
    bytes-backed ndarrays. Retain every row until target verification completes.
    Packet construction temporarily also owns the two append-grown lists.
    Existing effective_workspace covers one-step preparation/draw scratch; q
    preparation and target extraction are sequential, while these owners overlap.
    """
    import sys
    import numpy as np
    from tensorfold.engine.rejection import ChainPacket, ProbabilityRow
    from .assistant_sampled import RecordedSampledChain

    if any(type(value) is not int or value < 1 for value in (rows, width)):
        raise ValueError("positive assistant proposal count/vocabulary required")
    example = ProbabilityRow([0], [1.0])
    row_header = sys.getsizeof(example) + sum(sys.getsizeof(array) + sys.getsizeof(array.base) - array.nbytes
                                             for array in (example.ids, example.probs))
    row_data = width * (np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize)
    packet = ChainPacket((), (), 0)
    record = RecordedSampledChain(packet, None, None)
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    append_list = []
    for _ in range(rows):
        append_list.append(None)
    containers = (sys.getsizeof(packet) + sys.getsizeof(packet.first_position)
                  + 2*sys.getsizeof(()) + 2*rows*pointer
                  + sys.getsizeof(record) + sys.getsizeof(record.__dict__) + 2*sys.getsizeof(append_list))
    if proposal_policy:
        from tensorfold.engine.exact_sampling import Sampling
        derived = Sampling(seed=0)
        # dataclasses.replace owns one Sampling/dict and the newly multiplied
        # temperature float. Other fields borrow immutable request/config data.
        containers += sys.getsizeof(derived) + sys.getsizeof(derived.__dict__) + sys.getsizeof(derived.temperature)
    return rows*(row_data + row_header + sys.getsizeof(width-1)) + containers


def admit(model_dir, cfg, context, explicit, torch, draft_dir, max_rows, media, drafter_bits=16,
          *, assistant_sampled_q=False, assistant_q_policy=False, full_static_cache=False,
          prefill_rows=PREFILL_ROWS, packed_storage=None):
    """Account target/draft owners before allocating, using the shared CUDA policy."""
    from tensorfold.cuda.capacity import Geometry, Weights, admit as common, headers, itemsize
    from tensorfold.cuda.sampling import effective_workspace
    from tensorfold.cuda.speculative import chain_workspace
    from tensorfold.engine.penalties import bias_workspace

    if packed_storage is None:
        entries = headers(model_dir)
        target_bytes = sum(itemsize(v, k) * math.prod(v["shape"]) for k, v in entries.items())
    else:
        target_bytes = packed_storage.resident
    draft = None
    draft_bytes = 0
    draft_peak = 0
    kind = None
    if draft_dir is not None:
        kind, draft = companion_metadata(draft_dir, cfg)
        draft_bytes, draft_peak = draft.memory_bytes(drafter_bits)
    sampling = effective_workspace(max_rows, cfg.vocab, top_k=0) if draft else {"device_bytes": 0, "host_bytes": 0}
    proposal = chain_workspace(max_rows - 1, cfg.vocab, streams=1, penalized=True) if draft else {"device_bytes": 0, "host_bytes": 0}
    bias = bias_workspace(max_rows, cfg.vocab)
    retained_q = 0
    if assistant_sampled_q:
        if kind != "assistant":
            raise ValueError("sampled q requires the native official assistant")
        # Q is prepared one row at a time before target p exists. The same
        # effective_rows workspace may be reused, not double-counted; immutable
        # full-support q rows remain live alongside all p/residual workspace.
        one_q = effective_workspace(1, cfg.vocab, top_k=0)
        sampling = {name: max(sampling[name], one_q[name]) for name in sampling}
        retained_q = assistant_q_host_bytes(max_rows-1, cfg.vocab, proposal_policy=assistant_q_policy)

    def geometry(slots):
        slots = min(slots, cfg.max_position)
        cache = cfg.cache_bytes(slots, prefill_rows)
        companion = draft.state_bytes(slots) if draft else 0
        if kind == "assistant":
            # Last native shared SWA window (512, not persistent511) plus
            # accepted final hidden, including immutable prefix snapshots.
            companion += min(slots, cfg.window)*cfg.kv_heads*cfg.head_dim*4 + cfg.hidden*2
        # Live request plus bounded immutable prefix/snapshot owners.
        if full_static_cache:
            # One permanently sized live full slab; prefix snapshots own ONLY
            # logical rows (worst case the admitted capacity). Native SWA
            # recording belongs only to the live state, not every snapshot.
            snapshot_cache = cfg.cache_bytes(slots, 0)
            retained = cache + PREFIX_SNAPSHOTS*snapshot_cache + (PREFIX_SNAPSHOTS+1)*companion
            restore_swa = sum(min(slots, cfg.window-1)*cfg.kv_heads*cfg.head_dim*4
                              for kind in cfg.donor_layers if kind == "sliding_attention")
            retained += restore_swa + (5*slots if media else 0)  # live/snapshot/restore masks
        else:
            retained = (PREFIX_SNAPSHOTS + 1) * (cache + companion)
        # Conservative math-SDPA owner: scores and probabilities, plus causal mask.
        # A fused library provider may use less; no Flash dispatch is assumed here.
        attention = prefill_rows * (slots + prefill_rows) * cfg.heads * 9
        activations = prefill_rows * (cfg.hidden * 40 + cfg.intermediate * 8) * 4
        logits = max_rows * cfg.vocab * 4
        # Main embeddings before/after media replacement are retained through chunked prefill.
        embeddings = slots * cfg.hidden * 4 if media else 0
        draft_work = (draft.block * (draft.vocab * 8 + draft.hidden * 64)) if draft else 0
        if kind == "assistant":
            draft_work += min(slots, cfg.window)*cfg.kv_heads*cfg.head_dim*4 + cfg.hidden*2
        packed_work = packed_storage.workspace(max(prefill_rows, max_rows)) if packed_storage is not None else 0
        repeat = cfg.prompt_kv_repeat_bytes(slots, prefill_rows) if cfg.profile == '31B' else 0
        math_acc = cfg.prompt_math_accumulator_bytes(slots, prefill_rows) if cfg.profile == '31B' else 0
        return retained + attention + activations + logits + embeddings + draft_work + packed_work + repeat + math_acc

    resident = target_bytes + draft_bytes
    if packed_storage is None:
        peak = max(target_bytes * 2, target_bytes + draft_peak)
        host_loading = max(target_bytes, draft.parameter_bytes if draft else 0)
    else:
        from tensorfold.cuda.capacity import unified
        # CPU placeholders coexist with streaming CUDA binding on UMA. Charge
        # them without physical-page/reuse credit; host cgroup checks remain.
        stage = packed_storage.device_staging
        if unified(torch):
            stage += packed_storage.host_staging
        peak = max(target_bytes + stage, target_bytes + draft_peak)
        host_loading = max(packed_storage.host_staging, draft.parameter_bytes if draft else 0)
    if packed_storage is not None:
        from tensorfold.cuda.capacity import admit_host_commitment
        # Check actual current cgroup/physical allowance before placeholders.
        # This is an admission check, not another UMA charge/reservation.
        admit_host_commitment(host_loading)
    result = common(
        model_dir, context, explicit, torch, Geometry(geometry, 0), lambda _name, _info: (0, 0),
        weight_estimator=lambda _: (Weights(resident, max(0, peak - resident)),
                                     host_loading),
        device_workspace_bytes=sampling["device_bytes"] + proposal["device_bytes"] + bias["device_bytes"],
        host_workspace_bytes=sampling["host_bytes"] + proposal["host_bytes"] + retained_q + bias["host_bytes"])
    result["logit_bias_workspace"] = bias
    return result


class E2BEngine(Gemma4Engine):
    """Own one target/draft transaction lane; reuse target-law rejection and API contracts."""

    graphs = None
    prefill_rows = PREFILL_ROWS
    # Qualification candidate; enable only after an integrated P3 comparison.
    _clip_sampled_verification_to_room = False

    def __init__(self, model_dir: Path, *, drafter="", drafter_bits=16, depth=None, context=None,
                 context_explicit=None, streams=1, no_drafts=False, vision=False,
                 eos_token_ids=None, chat_template_file=None, chat_template_name=None, vision_urls=False,
                 fuse_prompt_norms=False, prompt_sdpa_gqa=False,
                 assistant_attention_gqa=False, assistant_device_chain=False, assistant_sampled_q=False,
                 attention_query_rows=2, assistant_q_temperature_factor=1.0, assistant_q_top_k=None,
                 full_static_cache=False, prefill_rows=PREFILL_ROWS):
        import torch
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.cuda.chat_template import prepare_chat_template
        from tensorfold.token_metadata import load_tokenizer, resolve_eos

        if type(prefill_rows) is not int or prefill_rows not in (512, 1024, 2048):
            raise ValueError("Gemma prefill rows must be512,1024 or2048")
        if type(streams) is not int or streams != 1:
            raise ValueError("Gemma E2B currently requires one exclusive request lane")
        if type(attention_query_rows) is not int or attention_query_rows not in (2, 4):
            raise ValueError("Gemma canonical attention query packing must be2 or4")
        if type(drafter_bits) is not int or drafter_bits not in (8, 16):
            raise ValueError("Gemma E2B DFlash2 proposal precision must be BF16(16) or private FP8(8)")
        assistant_options = {"assistant_attention_gqa": assistant_attention_gqa,
                             "assistant_device_chain": assistant_device_chain,
                             "assistant_sampled_q": assistant_sampled_q}
        if any(type(value) is not bool for value in (no_drafts, vision, vision_urls, fuse_prompt_norms,
                                                   prompt_sdpa_gqa, full_static_cache, *assistant_options.values())):
            raise ValueError("Gemma E2B feature switches must be boolean")
        factor, q_top_k = 1.0, None
        if (assistant_sampled_q or type(assistant_q_temperature_factor) not in (int, float)
                or assistant_q_temperature_factor != 1.0 or assistant_q_top_k is not None):
            from .assistant_sampled import validate_proposal_parameters
            factor, q_top_k = validate_proposal_parameters(assistant_q_temperature_factor, assistant_q_top_k)
        q_policy_changed = factor != 1.0 or q_top_k is not None
        if q_policy_changed and not assistant_sampled_q:
            raise ValueError("nondefault assistant proposal parameters require sampled_q=True")
        if any(assistant_options.values()) and (not drafter or no_drafts):
            raise ValueError("native assistant candidates require an enabled official assistant")
        if vision_urls and not vision:
            raise ValueError("Gemma E2B media URLs require --vision")
        if depth is not None and (type(depth) is not int or not 1 <= depth <= 7):
            raise ValueError("Gemma E2B DFlash2 permits one through seven proposals")
        if depth is not None and (not drafter or no_drafts):
            raise ValueError("Gemma E2B draft depth requires its enabled companion")
        raw = read_config(model_dir)
        cfg = E2BConfig.from_checkpoint(raw)
        from tensorfold.cuda.nvfp4.format import config_block
        packed_storage = None
        if config_block(raw) is not None:
            from tensorfold.cuda import precision
            if precision.mode() != precision.CHECKPOINT:
                raise ValueError("packed E4B requires --precision checkpoint with calibrated W4A4 arithmetic")
            from .dense_engine_storage import inspect_packed_storage
            packed_storage = inspect_packed_storage(model_dir)
        if prefill_rows != PREFILL_ROWS and cfg.profile != "E4B":
            raise ValueError("nondefault prefill rows require the E4B profile")
        self.prefill_rows = prefill_rows
        if full_static_cache and cfg.profile != "E4B":
            raise ValueError("static full donors currently require the qualified E4B profile")
        if assistant_sampled_q:
            validate_proposal_parameters(factor, q_top_k, vocab=cfg.vocab)
        if attention_query_rows == 4 and cfg.profile != "E4B":
            raise ValueError("four-query packing requires the E4B two-KV-head profile")
        if cfg.profile == '31B' and drafter:
            raise ValueError('Gemma31B currently requires serial --no-drafts execution')
        head_kind, head_cfg = None, None
        if cfg.profile == "E4B" and drafter:
            head_kind, head_cfg = companion_metadata(Path(drafter), cfg)
            head_cfg.memory_bytes(drafter_bits)
            if depth is not None and depth >= head_cfg.block:
                raise ValueError(f"Gemma {head_kind} permits at most {head_cfg.block-1} proposals")
        if any(assistant_options.values()) and head_kind != "assistant":
            raise ValueError("native assistant candidates require the official E4B assistant")
        draft_dir = Path(drafter) if drafter and not no_drafts else None
        default_rows = (7 if head_kind == "assistant" else head_cfg.block) if head_cfg else 8
        self.max_rows = (default_rows if depth is None else depth + 1) if draft_dir else 1
        self._tap_features = not (draft_dir and head_kind == "assistant")
        self._head_token_ids = bool(draft_dir and head_kind == "eagle3")
        self._assistant_sampled_q = assistant_sampled_q
        self._assistant_q_parameters = factor, q_top_k
        self.chat_template = prepare_chat_template(model_dir, template_file=chat_template_file,
                                                  template_name=chat_template_name)
        tokenizer = load_tokenizer(model_dir)
        self._turn = tokenizer.token_to_id("<|turn>")
        self.eos = resolve_eos(model_dir, raw.get("eos_token_id", raw["text_config"].get("eos_token_id")),
                               vocab_size=cfg.vocab, override=eos_token_ids, tokenizer=tokenizer)
        self.vision = None
        if vision:
            from tensorfold.vision.gemma4_e2b import GemmaE2BProcessor
            self.vision = GemmaE2BProcessor.from_directory(model_dir, allow_urls=vision_urls)
        torch.cuda.set_device(0)
        admission_options = {"assistant_sampled_q": True} if assistant_sampled_q else {}
        if packed_storage is not None:
            admission_options["packed_storage"] = packed_storage
        if q_policy_changed:
            admission_options["assistant_q_policy"] = True
        if prefill_rows != PREFILL_ROWS:
            admission_options["prefill_rows"] = prefill_rows
        if full_static_cache:
            admission_options["full_static_cache"] = True
        self.capacity_plan = admit(model_dir, cfg, context, context_explicit, torch, draft_dir,
                                   self.max_rows, vision, drafter_bits, **admission_options)
        if packed_storage is not None:
            self.capacity_plan["packed_storage"] = {
                "resident_bytes_estimate": packed_storage.resident,
                "device_loading_staging_bytes_estimate": packed_storage.device_staging,
                "host_loading_staging_bytes_estimate": packed_storage.host_staging,
                "native_nonpersistent_bytes_bound": packed_storage.native_buffer_bound,
                "scope": "ModelOpt NVFP4 W4A4 core; BF16 exceptions/KV",
                "qualification": "integrated numerical/media/native-context/HTTP/quality/performance pending",
                "bounds": "logical tensor owners; allocator/runtime/metadata overhead is not physical-capacity proof",
            }
        self.capacity_plan["media_encoder_workspace"] = "checked per request against live remaining pool"
        if cfg.bidirectional_vision:
            self.capacity_plan['bidirectional_visual_chunks'] = 'whole native blocks; oversized decoder chunks separately admitted before forward'
        self.context_window = self.capacity_plan["context_window"]
        if full_static_cache:
            self.capacity_plan["cache_policy"] = "fixed full donors; native dynamic SWA; compact logical-prefix snapshots"
            self.capacity_plan["permanent_full_slab_bytes"] = sum(
                self.context_window*cfg.kv_heads*cfg.global_head_dim*4
                for kind in cfg.donor_layers if kind == "full_attention")
        target_options = {"tap_layers": head_cfg.tap_layers} if self._head_token_ids else {}
        self.target = E2BTarget(model_dir, fuse_prompt_norms=fuse_prompt_norms,
                                prompt_sdpa_gqa=prompt_sdpa_gqa, attention_query_rows=attention_query_rows,
                                full_static_cache=full_static_cache,
                                **target_options)
        self.w = self.target
        self.head = None
        self._draft = None
        try:
            self._state = (self.target.make_state(self.context_window, assistant=True)
                           if not self._tap_features else self.target.make_state(self.context_window))
            if draft_dir is not None:
                if head_kind == "assistant":
                    from .assistant import Gemma4AssistantHead
                    q_options = ({"assistant_q_temperature_factor": factor, "assistant_q_top_k": q_top_k}
                                 if q_policy_changed else {})
                    self.head = Gemma4AssistantHead(draft_dir, self.target, self.context_window,
                                                   device=self.target.device, **assistant_options, **q_options)
                    self.head.bind_state(self._state)
                elif head_kind == "eagle3":
                    from .eagle3 import Gemma4Eagle3Head
                    self.head = Gemma4Eagle3Head(draft_dir, self.target, self.context_window,
                                                device=self.target.device)
                else:
                    from .e2b_draft import E2BDFlash2
                    self.head = E2BDFlash2(draft_dir, self.target.embedding_weight, self.target.head_weight,
                                          self.context_window, device=self.target.device, bits=drafter_bits)
                self._draft = self.head.make_state()
            self._prefix = None
            self._prefix_lock = threading.Lock()
            self._run_lock = threading.Lock()
            self._closed_lock = threading.Lock()
            self._closed = False
            self._disposed = False
            self._owner = None
            self._stream = torch.cuda.Stream(device=self.target.device)
            self.concurrent = False
            self.scheduler = None
            self._loaded = torch.cuda.Event()
            self._loaded.record()
            self._stream.wait_event(self._loaded)
        except BaseException as primary:
            self._draft = None
            self.head = None
            from tensorfold.cleanup import rollback
            rollback(self, primary, self.target.close)

    def generate(self, prompt: list[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None], draft: bool = True,
                 stop_eos: bool = True, *, vision=None, constraint=None,
                 background: bool = False, progress=None, penalties=None, cache_prompt: bool = True) -> dict[str, Any]:
        if type(cache_prompt) is not bool:
            raise ValueError("cache_prompt must be a bool")
        if self._assistant_sampled_q and draft and self.head is not None and self.head._reference_depth == 0:
            from tensorfold.engine.exact_sampling import Sampling
            if isinstance(sampling, Sampling) and sampling.temperature > 0:
                from .assistant_sampled import make_proposal_sampling
                make_proposal_sampling(sampling, *self._assistant_q_parameters, vocab=self.target.config.vocab)
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft=draft, stop_eos=stop_eos,
                                vision=vision, constraint=constraint, background=background,
                                progress=progress, penalties=penalties,
                                **({"cache_prompt": False} if not cache_prompt else {}))

    @staticmethod
    def _media_identity(media):
        return () if media is None else tuple(media.media_key)

    def _taps(self, result, count=None):
        if not self._tap_features:
            return result.hidden[:count]
        return super()._taps(result, count)

    def _append_head(self, state, result, input_ids, count=None, *, prompt=False):
        if self._head_token_ids:
            self.head.append(state, self._taps(result, count), token_ids=input_ids[:count], prompt=prompt)
        else:
            super()._append_head(state, result, input_ids, count, prompt=prompt)

    def _find_prefix(self, prompt, media, *, blocks=()):
        boundaries = set(prefill_boundaries(prompt, self._turn, self.prefill_rows, blocks=blocks))
        with self._prefix_lock:
            hit = self._prefix
            if hit is None or hit.media != media:
                return None
            for checkpoint in reversed(hit.checkpoints):
                position = checkpoint.position
                if (position in boundaries and position <= len(prompt)
                        and (position < len(prompt) or checkpoint.logits is not None)
                        and tuple(prompt[:position]) == hit.tokens[:position]):
                    return checkpoint
            return None

    def _prefill(self, prompt, vision, progress, started, *, cache_prompt=True):
        identity = self._media_identity(vision)
        blocks = tuple(vision.blocks) if vision is not None and self.target.config.bidirectional_vision else ()
        hit = (self._find_prefix(prompt, identity, blocks=blocks) if blocks else self._find_prefix(prompt, identity))
        state = self._state
        cached = hit.position if hit is not None else 0
        logits = None
        if hit is None:
            state.reset()
            if self._draft is not None:
                self._draft.reset()
        else:
            state.restore(hit.state)
            if self._draft is not None:
                self._draft.restore(hit.companion)
            logits = hit.logits
        encoded = None
        if vision is not None and cached < len(prompt):
            if self.vision is None:
                raise ValueError("Gemma E2B media requires --vision")
            media_ids = {self.target.raw_config[k] for k in ("image_token_id", "video_token_id", "audio_token_id")}
            # A media-key-matching prefix already owns these encoder results.
            # A text-only suffix needs no repeat tower pass or embedding staging.
            mask = vision.model_inputs.get("attention_mask")
            if cached == 0 or any(t in media_ids for t in prompt[cached:]) or (mask is not None and not bool(mask.all())):
                encoded = self.target.prepare_media(prompt, vision.model_inputs, trusted_tokens=True)
        boundary = self._turn_boundary(prompt, cached, blocks)
        kept = []
        start = cached
        try:
            for end in prefill_boundaries(prompt, self._turn, self.prefill_rows, blocks=blocks):
                if end <= start:
                    continue
                self._check_open()
                result = self.target.forward(
                    prompt[start:end], state, output_taps=self.head is not None and self._tap_features,
                    output_logits=end == len(prompt),
                    trusted_tokens=True, prompt=True, encoded=encoded)
                state.rollback(state.position)  # committed prefill trims recorded sliding history
                if self.head is not None:
                    self._append_head(self._draft, result, prompt[start:end], prompt=True)
                if end == len(prompt):
                    logits = result.logits
                elif end == boundary and cache_prompt:
                    kept.append(self._checkpoint(state))
                start = end
                if progress is not None and progress(len(prompt), cached, start,
                                                     (time.perf_counter() - started) * 1000):
                    from tensorfold.server.cancellation import RequestCancelled
                    raise RequestCancelled("request cancelled during Gemma E2B prefill")
        except BaseException:
            kept.clear()
            encoded = None
            result = None
            raise
        if logits is None:
            raise RuntimeError("Gemma E2B prefill produced no next-token logits")
        if cached < len(prompt) and cache_prompt:
            with self._prefix_lock:
                self._prefix = None
                kept.append(self._checkpoint(state, logits))
                self._prefix = _Prefix(tuple(prompt), identity, tuple(kept))
        return state, logits, cached

    def close(self):
        super().close()
        with self._run_lock:
            if self._disposed:
                return
            self._state = None
            self._draft = None
            self.head = None
            self.target.close()
            self._disposed = True
