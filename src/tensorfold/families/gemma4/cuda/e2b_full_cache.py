"""Unconnected full-donor StaticLayer candidate with compact prefix ownership.

Native HF owns GPU position arithmetic, index_copy_ and dynamic sliding-window
recording. This adapter adds bounded logical views, negative crop and compact
snapshots. It is an eager candidate, not a claim of whole-cache graph support.
One exclusive target lane owns updates, restoration and all borrowed views.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer, StaticLayer, get_layer_types_and_kwargs


class _FullLayer(StaticLayer):
    # Logical tensor views change shape, and this mixed candidate retains
    # dynamic SWA. Do not inherit the parent's broader compileability claim.
    is_compileable = False
    is_croppable = True

    def __init__(self, capacity):
        super().__init__(max_cache_len=capacity)
        self.logical_length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        end = self.logical_length + key_states.shape[-2]
        if end > self.max_cache_len:
            raise ValueError("full donor update exceeds its admitted capacity")
        keys, values = super().update(key_states, value_states, *args, **kwargs)
        self.logical_length = end
        return keys[:, :, :end], values[:, :, :end]

    def get_seq_length(self):
        # Eager mask/view metadata only. Native update still uses its device
        # cumulative_length tensor, with no .item() or host readback.
        return self.logical_length

    def get_mask_sizes(self, query_length):
        length = self.logical_length + query_length
        if length > self.max_cache_len:
            raise ValueError("full donor mask exceeds its admitted capacity")
        return length, 0

    def crop(self, tokens_to_remove):
        if type(tokens_to_remove) is not int or not -self.logical_length <= tokens_to_remove <= 0:
            raise ValueError("full donor crop accepts only bounded negative counts or zero")
        if tokens_to_remove:
            self.cumulative_length.add_(tokens_to_remove)
            self.logical_length += tokens_to_remove

    def reset(self):
        # Only initialized logical rows are ever returned, so stale backing
        # storage is not an operand after reset. Keep completed slab addresses.
        # A failed lazy allocation has no published static-address contract.
        if not self.is_initialized:
            self.keys = self.values = None
        self.cumulative_length.zero_()
        self.logical_length = 0


@dataclass(frozen=True)
class _FullPrefix:
    keys: object
    values: object


@dataclass(frozen=True)
class FullCacheSnapshot:
    owner: object
    contract: tuple
    position: int
    layers: tuple


def _copy_sliding(layer):
    # deepcopy(TensorView) may duplicate its entire old cat backing storage.
    # Clone only visible native rows, then let HF/Python own all other fields.
    memo = {}
    try:
        if layer.keys is not None:
            memo[id(layer.keys)] = layer.keys.clone()
            memo[id(layer.values)] = layer.values.clone()
        return copy.deepcopy(layer, memo)
    except BaseException:
        memo.clear()
        raise


class FullStaticCache(Cache):
    """Mixed native sliding donors and fixed-address full donors, opt-in only.

    Snapshot tensors are private immutable owners. Returned KV views are valid
    until the next cache mutation, serialized on the caller's existing stream.
    Parent state owners must prepare mask/assistant snapshot clones before
    calling restore; this cache never copies a full-prefix snapshot twice.
    """

    def __init__(self, config, capacity, *, owner=None):
        text = config.get_text_config(decoder=True)
        if type(capacity) is not int or not 0 < capacity <= text.max_position_embeddings:
            raise ValueError("full donor capacity must fit native context")
        kinds, kwargs = get_layer_types_and_kwargs(text)
        if any(kind not in ("sliding_attention", "full_attention") for kind in kinds) or not kinds:
            raise ValueError("full donor candidate requires the native Gemma attention cache profile")
        self.capacity = capacity
        self.owner = object() if owner is None else owner
        self.window = kwargs.get("sliding_window")
        self.kinds = tuple(kinds)
        self.geometry = tuple((text.per_layer_config[i].num_key_value_heads, text.per_layer_config[i].head_dim)
                              for i in range(len(kinds)))
        self.contract = capacity, self.kinds, self.geometry, self.window
        self.poisoned = False
        layers = [_FullLayer(capacity) if kind == "full_attention" else self._new_sliding() for kind in kinds]
        super().__init__(layers=layers)

    def _new_sliding(self):
        layer = DynamicSlidingWindowLayer(sliding_window=self.window)
        layer.activate_past_recording()
        return layer

    def _healthy(self):
        if self.poisoned:
            raise RuntimeError("full donor cache failed; restore a snapshot or reset before reuse")

    def _position(self):
        positions = [layer.get_seq_length() for layer in self.layers]
        if any(position != positions[0] for position in positions):
            raise RuntimeError("full donor cache has an incomplete forward")
        return positions[0]

    def get_seq_length(self, layer_idx=0):
        self._healthy()
        return super().get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx):
        self._healthy()
        return super().get_mask_sizes(query_length, layer_idx)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self._healthy()
        if type(layer_idx) is not int or not 0 <= layer_idx < len(self.layers):
            raise ValueError("full donor layer index differs from the admitted profile")
        heads, width = self.geometry[layer_idx]
        if (not isinstance(key_states, torch.Tensor) or not isinstance(value_states, torch.Tensor)
                or key_states.ndim != 4 or tuple(key_states.shape[:2]) != (1, heads)
                or key_states.shape[-1] != width or value_states.shape != key_states.shape
                or key_states.dtype != torch.bfloat16 or value_states.dtype != key_states.dtype
                or key_states.device != value_states.device or key_states.shape[-2] < 1):
            raise ValueError("full donor input geometry, precision or device differs")
        layer = self.layers[layer_idx]
        if layer.get_seq_length() + key_states.shape[-2] > self.capacity:
            raise ValueError("full donor update exceeds admitted context")
        if layer.is_initialized and (key_states.device != layer.keys.device or key_states.dtype != layer.keys.dtype):
            raise ValueError("full donor cache device/precision changed")
        try:
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        except BaseException:
            self.poisoned = True
            raise

    def crop(self, tokens_to_remove):
        self._healthy()
        position = self._position()
        if type(tokens_to_remove) is not int or not -position <= tokens_to_remove <= 0:
            raise ValueError("full donor crop requires a bounded negative count or zero")
        self.poisoned = True
        for layer in self.layers:
            if layer.is_initialized:
                layer.crop(tokens_to_remove)
        self.poisoned = False

    def snapshot(self):
        self._healthy()
        position = self._position()
        for kind, layer in zip(self.kinds, self.layers):
            if (kind == "sliding_attention" and layer.is_initialized
                    and layer.keys.shape[-2] != min(position, self.window-1)):
                raise RuntimeError("full donor snapshots require committed native sliding history")
        layers = []
        saved = None
        try:
            for kind, layer in zip(self.kinds, self.layers):
                if kind == "full_attention":
                    saved = (_FullPrefix(layer.keys[:, :, :position].clone(), layer.values[:, :, :position].clone())
                             if position else _FullPrefix(None, None))
                else:
                    saved = _copy_sliding(layer)
                layers.append(saved)
            return FullCacheSnapshot(self.owner, self.contract, position, tuple(layers))
        except BaseException:
            saved = None
            layers.clear()
            raise

    def __deepcopy__(self, memo):
        raise TypeError("full donor cache requires compact snapshot()/restore(), not whole-slab deepcopy")

    def _validate_snapshot(self, snapshot):
        if (not isinstance(snapshot, FullCacheSnapshot) or snapshot.owner is not self.owner
                or snapshot.contract != self.contract or type(snapshot.position) is not int
                or not 0 <= snapshot.position <= self.capacity or len(snapshot.layers) != len(self.layers)):
            raise ValueError("full donor snapshot belongs to a different cache contract")
        for i, (kind, saved, live) in enumerate(zip(self.kinds, snapshot.layers, self.layers)):
            if kind == "full_attention":
                if not isinstance(saved, _FullPrefix):
                    raise ValueError("full donor snapshot kind differs")
                if not snapshot.position:
                    if saved.keys is not None or saved.values is not None:
                        raise ValueError("empty full donor snapshot must not retain a slab")
                    continue
                keys, values = saved.keys, saved.values
                rows = snapshot.position
            else:
                if (not isinstance(saved, DynamicSlidingWindowLayer) or saved.get_seq_length() != snapshot.position
                        or saved.sliding_window != self.window or not saved.record_past):
                    raise ValueError("sliding donor snapshot position differs")
                keys, values = saved.keys, saved.values
                rows = min(snapshot.position, self.window-1)
                if not rows and not saved.is_initialized:
                    continue
            heads, width = self.geometry[i]
            if (not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor)
                    or tuple(keys.shape) != (1, heads, rows, width) or values.shape != keys.shape
                    or keys.dtype != torch.bfloat16 or values.dtype != keys.dtype or values.device != keys.device
                    or (live.is_initialized and keys.device != live.keys.device)):
                raise ValueError("full/sliding donor snapshot tensor contract differs")

    def restore(self, snapshot):
        self._validate_snapshot(snapshot)
        # All potentially allocating native SWA clones happen before live
        # publication. Full snapshots are immutable copy sources, never cloned.
        sliding = {i: _copy_sliding(saved) for i, (kind, saved) in enumerate(zip(self.kinds, snapshot.layers))
                   if kind == "sliding_attention"}
        self.poisoned = True
        prepared = None
        try:
            for kind, saved, live in zip(self.kinds, snapshot.layers, self.layers):
                if kind != "full_attention":
                    continue
                if snapshot.position:
                    if not live.is_initialized:
                        live.lazy_initialization(saved.keys, saved.values)
                    live.keys[:, :, :snapshot.position].copy_(saved.keys)
                    live.values[:, :, :snapshot.position].copy_(saved.values)
                live.cumulative_length.fill_(snapshot.position)
                live.logical_length = snapshot.position
            for index, prepared in sliding.items():
                self.layers[index] = prepared
        except BaseException:
            prepared = None
            sliding.clear()
            raise
        self.poisoned = False

    def reset(self):
        # Native dynamic reset retains tensor shapes, which is not the empty
        # history needed here. Fresh small SWA owners preserve its update API.
        sliding = {i: self._new_sliding() for i, kind in enumerate(self.kinds) if kind == "sliding_attention"}
        self.poisoned = True
        for kind, layer in zip(self.kinds, self.layers):
            if kind == "full_attention":
                layer.reset()
        for index, prepared in sliding.items():
            self.layers[index] = prepared
        self.poisoned = False
