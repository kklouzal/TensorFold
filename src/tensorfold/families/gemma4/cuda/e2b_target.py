"""Owned TensorFold integration of Transformers' complete Gemma E2B target.

Transformers owns model math, PLE, modality encoders and shared-KV attention.
This adapter owns request/cache lifetime, the four draft taps, and bounded
prefill/decode results. The checkpoint remains BF16 without requantization.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .e2b_metadata import E2BConfig

# Speculators indices include embedding output0; decoder output i occupies index i+1.
TAP_LAYERS = (2, 12, 22, 32)


@dataclass(frozen=True)
class E2BForward:
    logits: Any
    hidden: Any
    taps: dict[int, Any]


@dataclass(frozen=True)
class E2BSnapshot:
    position: int
    cache: Any
    owner: Any
    limit: int
    attention_mask: Any = None


@dataclass(frozen=True)
class E2BEncodedMedia:
    """Prepared GPU owners retained only for this request's bounded prefill."""
    token_ids: tuple[int, ...]
    padded_ids: Any
    text_embeddings: Any
    embeddings: Any
    attention_mask: Any


class E2BState:
    """One lane owns all donor caches, including speculative sliding history.

    HF's public past-recording mode retains the previous window until crop.
    Every committed forward calls crop(0); rejected verifier rows use a negative
    crop. Snapshots are detached owners and restore never mutates the stored one.
    """

    def __init__(self, config, limit: int, *, owner=None):
        text = config.get_text_config()
        if type(limit) is not int or not 0 < limit <= text.max_position_embeddings:
            raise ValueError("Gemma E2B state context must be within the native model limit")
        self.config, self.limit = config, limit
        self.owner = object() if owner is None else owner
        self.position = 0
        self.poisoned = False
        self.attention_mask = None
        self.cache = self._empty()

    def _empty(self):
        from transformers import DynamicCache
        cache = DynamicCache(config=self.config)
        for layer in cache.layers:
            activate = getattr(layer, "activate_past_recording", None)
            if activate is not None:
                activate()
        return cache

    def reset(self):
        self.cache = self._empty()
        self.position = 0
        self.poisoned = False
        self.attention_mask = None

    def rollback(self, position: int):
        if self.poisoned or type(position) is not int or not 0 <= position <= self.position:
            raise ValueError("Gemma E2B rollback must remain within the tentative target history")
        self.cache.crop(-(self.position - position))
        self.position = position
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask[:, :position]

    def snapshot(self):
        if self.poisoned:
            raise RuntimeError("cannot snapshot failed Gemma E2B state")
        return E2BSnapshot(self.position, copy.deepcopy(self.cache), self.owner, self.limit,
                            None if self.attention_mask is None else self.attention_mask.clone())

    def restore(self, snapshot: E2BSnapshot):
        if (not isinstance(snapshot, E2BSnapshot) or snapshot.owner is not self.owner
                or snapshot.limit != self.limit or not 0 <= snapshot.position <= self.limit):
            raise ValueError("Gemma E2B snapshot has incompatible position ownership")
        cache = copy.deepcopy(snapshot.cache)
        mask = None if snapshot.attention_mask is None else snapshot.attention_mask.clone()
        self.cache = cache
        self.position = snapshot.position
        self.poisoned = False
        self.attention_mask = mask


class E2BTarget:
    precision = "checkpoint"
    fast_prefill = False

    def __init__(self, model_dir: Path, *, projection: str = "native", fuse_prompt_norms=False):
        import torch
        import transformers
        from importlib.metadata import version
        from tensorfold.checkpoint_metadata import read_config
        from .e2b_runtime import require_runtime

        self.raw_config = read_config(model_dir)
        self.config = self.cfg = E2BConfig.from_dict(self.raw_config)
        require_runtime(torch, transformers.__version__, version("triton"))
        from transformers import Gemma4ForConditionalGeneration
        from .e2b_policy import E2BBlasPolicy
        from .e2b_rows import E2BRowLinear, E2BRowPolicy, install_row_linears
        from .e2b_attention import install_attention
        from .e2b_norm import install_row_norms

        self.model = self.embedding_weight = self.head_weight = None
        self._collect_taps = False
        self._closed = False
        self._taps: dict[int, Any] = {}
        self._hooks = []
        self._blas_policy = None
        self._row_policy = E2BRowPolicy(projection, fuse_prompt_norms=fuse_prompt_norms)
        self.tap_layers = self.config.tap_layers
        try:
            self._blas_policy = E2BBlasPolicy(torch)
            self.model, info = Gemma4ForConditionalGeneration.from_pretrained(
                str(model_dir), local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
            errors = {k: info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                      if info.get(k)}
            if errors:
                raise ValueError(f"Gemma E2B checkpoint was not fully consumed: {errors}")
            # Public single-device placement, with CPU/GPU overlap in admission.
            self.model = self.model.eval().requires_grad_(False).to(torch.device("cuda:0"))
            self.device = next(self.model.parameters()).device
            self.embedding_weight = self.model.get_input_embeddings().weight
            self.head_weight = self.model.lm_head.weight
            install_attention(self.model.model.language_model, self._row_policy)
            install_row_norms(self.model.model.language_model, self._row_policy)
            if projection != "native":
                install_row_linears(self.model.model.language_model, self._row_policy)
                self.model.lm_head = E2BRowLinear(self.model.lm_head, self._row_policy)
            for layer in self.tap_layers:
                self._hooks.append(self.model.model.language_model.layers[layer].register_forward_hook(
                    self._tap_hook(layer)))
        except BaseException as primary:
            from tensorfold.cleanup import rollback
            rollback(self, primary, self.close)

    def _tap_hook(self, layer):
        def capture(_module, _args, output):
            if self._collect_taps:
                self._taps[layer] = output.squeeze(0).detach()
        return capture

    def make_state(self, context: int):
        return E2BState(self.model.config, context, owner=self)

    @contextmanager
    def native_reference(self):
        """Yield the native HF oracle under the caller's drained exclusive lane.

        No generation, Target.forward or configuration mutation may run in
        parallel. This explicit scope keeps whole-HF golden forwards native
        even when the production prompt norm candidate is enabled. The caller
        completes reference CUDA work before handing owners to another stream.
        """
        if self._closed:
            raise RuntimeError("Gemma E2B target is closed")
        with self._row_policy.native_reference():
            yield self.model

    def _feature_kwargs(self, model_inputs):
        import torch
        kwargs = {}
        for key, value in (model_inputs or {}).items():
            if not isinstance(value, torch.Tensor):
                raise ValueError("Gemma E2B media model inputs must be prepared tensors")
            # The library owns preprocessing casts. In particular vision first
            # normalizes FP32 pixels, then casts to its BF16 projection.
            if key != "audio_feature_lengths":
                kwargs[key] = value.to(device=self.device)
        return kwargs

    @staticmethod
    def _audio_lengths(model_inputs):
        """Validate the owned packed carrier before any device allocation."""
        import torch
        features = model_inputs.get("input_features")
        lengths = model_inputs.get("audio_feature_lengths")
        if features is None:
            if lengths is not None:
                raise ValueError("Gemma E2B audio lengths require packed features")
            return ()
        mask = model_inputs.get("input_features_mask")
        if (not isinstance(lengths, torch.Tensor) or lengths.device.type != "cpu"
                or lengths.dtype != torch.int64 or lengths.ndim != 1 or lengths.numel() == 0
                or features.ndim != 3 or features.shape[0] != 1
                or mask is None or tuple(mask.shape) != tuple(features.shape[:2])):
            raise ValueError("Gemma E2B audio requires packed features, masks and clip lengths")
        result = tuple(lengths.tolist())
        if any(n <= 0 for n in result) or sum(result) != features.shape[1]:
            raise ValueError("Gemma E2B packed audio clip lengths do not match features")
        return result

    def _require_media_workspace(self, ids, model_inputs, audio_lengths):
        """Check actual prepared input geometry before allocating encoder owners.

        Encoders run one image/frame/clip at a time, so batch count affects only
        retained input/output owners, never the encoder's attention working set.
        Library allocator workspace also retains the shared CUDA memory reserve.
        """
        import torch
        from tensorfold.cuda.capacity import available_bytes

        transfer = sum(t.numel() * t.element_size() for t in model_inputs.values())
        vision_work = 0
        v = self.raw_config["vision_config"]
        for key in ("pixel_values", "pixel_values_videos"):
            value = model_inputs.get(key)
            if value is not None:
                patches = value.shape[-2]
                # Three FP32 score/softmax owners plus conservative live layer activations.
                vision_work = max(vision_work, 12 * v["num_attention_heads"] * patches * patches
                                  + 64 * patches * (v["hidden_size"] + v["intermediate_size"]))
        audio_work = 0
        audio = model_inputs.get("input_features")
        if audio is not None:
            a = self.raw_config["audio_config"]
            frames = max(audio_lengths)
            conv = 2 * max(a["subsampling_conv_channels"]) * math.ceil(frames / 2) * audio.shape[-1] * 4
            audio_work = conv + frames * a["hidden_size"] * 128
        # Original padded-token embeddings remain alive for PLE. Row index_copy
        # modifies one separate merged owner, without masked_scatter's expanded
        # int64 prefix-sum workspace (8*N*hidden in the pinned CUDA implementation).
        embedding = len(ids) * self.config.hidden * 2
        has_media = any(model_inputs.get(key) is not None for key in
                        ("pixel_values", "pixel_values_videos", "input_features"))
        embeddings = embedding * (2 if has_media else 1)
        # Encoded features cannot outnumber placeholder tokens in an admitted
        # prompt. Concatenation can simultaneously retain source and output.
        projected_features = embedding * 2 if has_media else 0
        token_masks = len(ids) * (8 * 3 + 5)  # tokens, padded IDs, row indices, boolean masks
        needed = transfer + embeddings + projected_features + token_masks + max(vision_work, audio_work)
        if needed > available_bytes(torch):
            raise ValueError("Gemma E2B prepared media exceeds the remaining encoder workspace budget")

    def _image_features(self, pixels, positions):
        import torch
        if pixels.ndim != 3 or positions.ndim != 3 or pixels.shape[0] != positions.shape[0]:
            raise ValueError("Gemma E2B image/frame processor tensor geometry differs")
        outputs = []
        for i in range(pixels.shape[0]):
            features = self.model.model.get_image_features(
                pixels[i:i + 1], positions[i:i + 1], return_dict=True).pooler_output
            outputs.extend(features)
        return torch.cat(outputs, dim=0)

    def prepare_media(self, ids, model_inputs, *, trusted_tokens=False) -> E2BEncodedMedia:
        """Use the public HF encoder APIs, then preserve pre-merge PLE identity.

        HF5.17 has no public precomputed-media output argument. Its complete
        forward materializes decoder attention for the whole input. This small
        adapter keeps the same placeholder/encoder merge while allowing bounded
        decoder chunks. Encoder math remains entirely library-owned.
        """
        import torch
        if self._closed:
            raise RuntimeError("Gemma E2B target is closed")
        if (not isinstance(ids, (list, tuple)) or not ids or len(ids) > self.config.max_position
                or (not trusted_tokens and any(type(x) is not int or not 0 <= x < self.config.vocab for x in ids))):
            raise ValueError("Gemma E2B media requires valid prompt token IDs within native context")
        audio_lengths = self._audio_lengths(model_inputs)
        self._require_media_workspace(ids, model_inputs, audio_lengths)
        kwargs = self._feature_kwargs(model_inputs)
        model = self.model.model
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        masks = {kind: tokens == getattr(self.model.config, key)
                 for kind, key in (("image", "image_token_id"), ("video", "video_token_id"),
                                   ("audio", "audio_token_id"))}
        multimodal = masks["image"] | masks["video"] | masks["audio"]
        padded = torch.where(multimodal, self.model.config.text_config.pad_token_id, tokens)
        original = self.model.get_input_embeddings()(padded)
        embeddings = original
        features = []
        if kwargs.get("pixel_values") is not None:
            features.append(("image", self._image_features(kwargs["pixel_values"], kwargs["image_position_ids"])))
        if kwargs.get("pixel_values_videos") is not None:
            # Native HF video encoding flattens video/frame dimensions and uses
            # precisely the image tower/projector; no cross-frame operation exists.
            features.append(("video", self._image_features(
                kwargs["pixel_values_videos"].flatten(0, 1), kwargs["video_position_ids"].flatten(0, 1))))
        if kwargs.get("input_features") is not None:
            if kwargs.get("input_features_mask") is None:
                raise ValueError("Gemma E2B audio features require their mask")
            outputs = []
            start = 0
            for length in audio_lengths:
                end = start + length
                audio = model.get_audio_features(kwargs["input_features"][:, start:end],
                                                kwargs["input_features_mask"][:, start:end], return_dict=True)
                outputs.append(audio.pooler_output[audio.attention_mask.to(self.device)])
                start = end
            features.append(("audio", torch.cat(outputs, dim=0)))
        if features:
            embeddings = original.clone()
        for kind, feature in features:
            token = getattr(self.model.config, kind + "_token_id")
            rows = [i for i, value in enumerate(ids) if value == token]
            if feature.ndim != 2 or tuple(feature.shape) != (len(rows), self.config.hidden):
                raise ValueError(f"Gemma E2B {kind} placeholders do not match encoded feature rows")
            feature = feature.to(device=self.device, dtype=embeddings.dtype)
            # Enumerating token positions gives unique ordered indices; feature
            # order is precisely the native flattened placeholder order.
            index = torch.tensor(rows, dtype=torch.long, device=self.device)
            embeddings[0].index_copy_(0, index, feature)
        mask = kwargs.get("attention_mask")
        # Native unpadded single prompts have an all-one mask. Prove that on
        # the already-owned CPU tensor so text decode needs no growing mask.
        original_mask = model_inputs.get("attention_mask")
        if original_mask is not None and bool(original_mask.all()):
            mask = None
        return E2BEncodedMedia(tuple(ids), padded, original, embeddings, mask)

    def project_logits(self, hidden):
        import torch
        if self._closed:
            raise RuntimeError("Gemma E2B target is closed")
        logits = self.model.lm_head(hidden)
        return torch.tanh(logits / self.config.softcap) * self.config.softcap

    def forward(self, ids, state: E2BState, *, output_taps=False, output_logits=True,
                trusted_tokens=False, prompt=False, model_inputs=None,
                encoded: E2BEncodedMedia | None = None):
        import torch

        if self._closed:
            raise RuntimeError("Gemma E2B target is closed")
        if self._row_policy.reference_depth:
            raise RuntimeError("native_reference requires direct whole-HF forward, not Target.forward")
        if output_taps and not self.tap_layers:
            raise ValueError("Gemma E4B has no admitted speculative hidden-state tap contract")
        if not isinstance(state, E2BState) or state.owner is not self:
            raise ValueError("Gemma E2B cache state belongs to a different target")
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("Gemma E2B forward needs nonempty token IDs")
        if not trusted_tokens and any(type(x) is not int or not 0 <= x < self.config.vocab for x in ids):
            raise ValueError("Gemma E2B forward token is outside the target vocabulary")
        if state.poisoned:
            raise RuntimeError("Gemma E2B state failed; reset or restore it before reuse")
        if state.position + len(ids) > state.limit:
            raise ValueError("Gemma E2B forward exceeds admitted context")
        if model_inputs is not None:
            raise ValueError("Gemma E2B media requires prepare_media and bounded encoded chunks")
        kwargs = self._feature_kwargs(model_inputs)
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        positions = torch.arange(state.position, state.position + len(ids), device=self.device).unsqueeze(0)
        self._row_policy.prompt = bool(prompt)
        self._row_policy.position = state.position
        self._collect_taps = bool(output_taps)
        self._taps = {}
        try:
            if encoded is None:
                mask = kwargs.get("attention_mask")
                if mask is None and state.attention_mask is not None:
                    mask = torch.cat((state.attention_mask[:, :state.position],
                                      state.attention_mask.new_ones((1, len(ids)))), dim=-1)
                    kwargs["attention_mask"] = mask
                out = self.model.model(input_ids=tokens, position_ids=positions, past_key_values=state.cache,
                                       use_cache=True, return_dict=True, **kwargs)
            else:
                start, end = state.position, state.position + len(ids)
                if tuple(ids) != encoded.token_ids[start:end]:
                    raise ValueError("Gemma E2B media chunk does not belong to the prepared prompt")
                # Identity PLE sees padded original token IDs, never replaced media embeddings.
                ple = self.model.model.language_model.get_per_layer_inputs(
                    encoded.padded_ids[:, start:end], encoded.text_embeddings[:, start:end])
                mask = encoded.attention_mask
                out = self.model.model.language_model(
                    inputs_embeds=encoded.embeddings[:, start:end], per_layer_inputs=ple,
                    attention_mask=None if mask is None else mask[:, :end],
                    position_ids=positions, past_key_values=state.cache, use_cache=True, return_dict=True)
            state.position += len(ids)
            state.attention_mask = None if mask is None else mask[:, :state.position]
            hidden = out.last_hidden_state.squeeze(0)
            if output_taps and set(self._taps) != set(self.tap_layers):
                raise RuntimeError("Gemma E2B target did not publish the required decoder outputs")
            logits = self.project_logits(hidden[-1:] if prompt else hidden) if output_logits else None
            result = E2BForward(logits, hidden, dict(self._taps))
        except BaseException:
            state.poisoned = True
            raise
        finally:
            self._row_policy.prompt = True
            self._collect_taps = False
            self._taps = {}
        return result

    def close(self):
        from tensorfold.cleanup import raise_failures
        self._closed = True
        errors = []
        for hook in self._hooks:
            try:
                hook.remove()
            except BaseException as error:
                errors.append(error)
        self._hooks.clear()
        self.model = self.embedding_weight = self.head_weight = None
        if self._blas_policy is not None:
            try:
                self._blas_policy.close()
            except BaseException as error:
                errors.append(error)
        raise_failures(None, errors)
