"""Single-lane full BF16 Gemma E2B using TensorFold's existing request sampler."""
from __future__ import annotations

from pathlib import Path
import math
import threading
import time

from .engine import Gemma4Engine, _Prefix
from .e2b_metadata import E2BConfig
from .e2b_target import E2BTarget

PREFILL_ROWS = 512
PREFIX_SNAPSHOTS = 3


def prefill_boundaries(prompt, turn):
    """Canonical native-GEMM shapes: every turn, then at most512 rows.

    Extension may change the previous final partial chunk, so that checkpoint
    is reusable only when it is also a boundary of the extended prompt.
    """
    stops = [i for i, token in enumerate(prompt) if i and token == turn] if turn is not None else []
    stops.append(len(prompt))
    ends = []
    start = 0
    for stop in stops:
        while start < stop:
            start = min(start + PREFILL_ROWS, stop)
            ends.append(start)
    return tuple(ends)


def admit(model_dir, cfg, context, explicit, torch, draft_dir, max_rows, media, drafter_bits=16):
    """Account target/draft owners before allocating, using the shared CUDA policy."""
    from tensorfold.cuda.capacity import Geometry, Weights, admit as common, headers, itemsize
    from tensorfold.cuda.sampling import effective_workspace
    from tensorfold.cuda.speculative import chain_workspace

    entries = headers(model_dir)
    target_bytes = sum(itemsize(v, k) * math.prod(v["shape"]) for k, v in entries.items())
    draft = None
    draft_bytes = 0
    draft_peak = 0
    if draft_dir is not None:
        from ..e2b_dflash_metadata import checked_headers
        draft, _ = checked_headers(draft_dir)
        if draft.hidden != cfg.hidden or draft.vocab != cfg.vocab:
            raise ValueError("Gemma E2B target and DFlash2 widths/vocabulary differ")
        draft_bytes, draft_peak = draft.memory_bytes(drafter_bits)
    sampling = effective_workspace(max_rows, cfg.vocab, top_k=0) if draft else {"device_bytes": 0, "host_bytes": 0}
    proposal = chain_workspace(max_rows - 1, cfg.vocab, streams=1, penalized=True) if draft else {"device_bytes": 0, "host_bytes": 0}

    def geometry(slots):
        slots = min(slots, cfg.max_position)
        cache = cfg.cache_bytes(slots, PREFILL_ROWS)
        companion = draft.state_bytes(slots) if draft else 0
        # Live request plus bounded immutable prefix/snapshot owners.
        retained = (PREFIX_SNAPSHOTS + 1) * (cache + companion)
        # Conservative math-SDPA owner: scores and probabilities, plus causal mask.
        # A fused library provider may use less; no Flash dispatch is assumed here.
        attention = PREFILL_ROWS * (slots + PREFILL_ROWS) * cfg.heads * 9
        activations = PREFILL_ROWS * (cfg.hidden * 40 + cfg.intermediate * 8) * 4
        logits = max_rows * cfg.vocab * 4
        # Main embeddings before/after media replacement are retained through chunked prefill.
        embeddings = slots * cfg.hidden * 4 if media else 0
        draft_work = (draft.block * (draft.vocab * 8 + draft.hidden * 64)) if draft else 0
        return retained + attention + activations + logits + embeddings + draft_work

    resident = target_bytes + draft_bytes
    peak = max(target_bytes * 2, target_bytes + draft_peak)
    return common(
        model_dir, context, explicit, torch, Geometry(geometry, 0), lambda _name, _info: (0, 0),
        weight_estimator=lambda _: (Weights(resident, max(0, peak - resident)),
                                     max(target_bytes, draft.parameter_bytes if draft else 0)),
        device_workspace_bytes=sampling["device_bytes"] + proposal["device_bytes"],
        host_workspace_bytes=sampling["host_bytes"] + proposal["host_bytes"])


class E2BEngine(Gemma4Engine):
    """Own one target/draft transaction lane; reuse target-law rejection and API contracts."""

    graphs = None
    # Qualification candidate; enable only after an integrated P3 comparison.
    _clip_sampled_verification_to_room = False

    def __init__(self, model_dir: Path, *, drafter="", drafter_bits=16, depth=None, context=None,
                 context_explicit=None, streams=1, no_drafts=False, vision=False,
                 eos_token_ids=None, chat_template_file=None, chat_template_name=None, vision_urls=False):
        import torch
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.cuda.chat_template import prepare_chat_template
        from tensorfold.token_metadata import load_tokenizer, resolve_eos

        if type(streams) is not int or streams != 1:
            raise ValueError("Gemma E2B currently requires one exclusive request lane")
        if type(drafter_bits) is not int or drafter_bits not in (8, 16):
            raise ValueError("Gemma E2B DFlash2 proposal precision must be BF16(16) or private FP8(8)")
        if any(type(value) is not bool for value in (no_drafts, vision, vision_urls)):
            raise ValueError("Gemma E2B feature switches must be boolean")
        if vision_urls and not vision:
            raise ValueError("Gemma E2B media URLs require --vision")
        if depth is not None and (type(depth) is not int or not 1 <= depth <= 7):
            raise ValueError("Gemma E2B DFlash2 permits one through seven proposals")
        if depth is not None and (not drafter or no_drafts):
            raise ValueError("Gemma E2B draft depth requires its enabled companion")
        raw = read_config(model_dir)
        cfg = E2BConfig.from_dict(raw)
        draft_dir = Path(drafter) if drafter and not no_drafts else None
        self.max_rows = (8 if depth is None else depth + 1) if draft_dir else 1
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
        self.capacity_plan = admit(model_dir, cfg, context, context_explicit, torch, draft_dir,
                                   self.max_rows, vision, drafter_bits)
        self.capacity_plan["media_encoder_workspace"] = "checked per request against live remaining pool"
        self.context_window = self.capacity_plan["context_window"]
        self.target = E2BTarget(model_dir)
        self.w = self.target
        self.head = None
        self._draft = None
        try:
            if draft_dir is not None:
                from .e2b_draft import E2BDFlash2
                self.head = E2BDFlash2(draft_dir, self.target.embedding_weight, self.target.head_weight,
                                      self.context_window, device=self.target.device, bits=drafter_bits)
                self._draft = self.head.make_state()
            self._state = self.target.make_state(self.context_window)
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

    @staticmethod
    def _media_identity(media):
        return () if media is None else tuple(media.media_key)

    def _find_prefix(self, prompt, media):
        boundaries = set(prefill_boundaries(prompt, self._turn))
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

    def _prefill(self, prompt, vision, progress, started):
        identity = self._media_identity(vision)
        hit = self._find_prefix(prompt, identity)
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
        boundary = self._turn_boundary(prompt, cached, ())
        kept = []
        start = cached
        try:
            for end in prefill_boundaries(prompt, self._turn):
                if end <= start:
                    continue
                self._check_open()
                result = self.target.forward(
                    prompt[start:end], state, output_taps=self.head is not None, output_logits=end == len(prompt),
                    trusted_tokens=True, prompt=True, encoded=encoded)
                state.rollback(state.position)  # committed prefill trims recorded sliding history
                if self.head is not None:
                    self.head.append(self._draft, self._taps(result), prompt=True)
                if end == len(prompt):
                    logits = result.logits
                elif end == boundary:
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
        if cached < len(prompt):
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
