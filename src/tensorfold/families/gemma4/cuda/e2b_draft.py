"""BF16 Speculators DFlash2 proposals over committed Gemma E2B hidden states.

The verifier alone owns output probabilities. Every sampled proposal carries its
actual sparse q row into TensorFold's standard rejection sampler. Proposal K/V
never enters committed state: append receives only accepted verifier taps.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from tensorfold.cuda.tensor_file import checkpoint_path
from tensorfold.engine.rejection import ChainPacket, draw, proposal_row
from ..e2b_dflash_metadata import E2BDraftConfig, checked_headers


def _conv(x, delta, base, branch: int, group_size: int):
    """Two-tap, one-anchor-block convolution with the checkpoint's BF16 rounding."""
    rows, hidden = x.shape
    groups = hidden // group_size
    current = x.reshape(rows, groups, group_size)
    previous = torch.cat((torch.zeros_like(current[:1]), current[:-1]), dim=0)
    k0 = base[branch, 0].reshape(1, groups, group_size) + delta[:, branch, 0, :, None]
    k1 = base[branch, 1].reshape(1, groups, group_size) + delta[:, branch, 1, :, None]
    # BF16 products round before addition, as grouped_dynamic_conv in Speculators.
    return (current * k0 + previous * k1).reshape(rows, hidden)


def _rope(x, positions, inv_freq):
    phase = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((phase, phase), dim=-1)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos[None] + rotated * sin[None]


class E2BDraftState:
    """One lane's owned committed context; sliding rings plus a full-history layer."""

    def __init__(self, cfg: E2BDraftConfig, device, context_limit: int):
        self.cfg, self.context_limit = cfg, context_limit
        self.slots = tuple(min(cfg.window, context_limit) if k == "sliding_attention"
                           else context_limit for k in cfg.layer_types)
        self.keys, self.values, self.work_keys, self.work_values = [], [], [], []
        for kind, slots in zip(cfg.layer_types, self.slots):
            # Full-history cache has a disposable tail for this round's noise K/V.
            size = slots if kind == "sliding_attention" else slots + cfg.block
            key = torch.empty((cfg.kv_heads, size, cfg.head_dim), dtype=torch.bfloat16, device=device)
            value = torch.empty_like(key)
            self.keys.append(key)
            self.values.append(value)
            if kind == "sliding_attention":
                self.work_keys.append(torch.empty((cfg.kv_heads, slots + cfg.block, cfg.head_dim),
                                                  dtype=torch.bfloat16, device=device))
                self.work_values.append(torch.empty_like(self.work_keys[-1]))
            else:
                self.work_keys.append(key)
                self.work_values.append(value)
        self.length = 0

    @property
    def nbytes(self):
        owned = self.keys + self.values + [
            t for i, t in enumerate(self.work_keys + self.work_values)
            if self.cfg.layer_types[i % self.cfg.layers] == "sliding_attention"]
        return sum(t.numel() * t.element_size() for t in owned)

    def reset(self):
        self.length = 0

    def _index(self, layer: int):
        count = min(self.length, self.slots[layer])
        return torch.arange(self.length - count, self.length, device=self.keys[layer].device) % self.slots[layer]

    def snapshot(self):
        """Detached chronological tensors; retained snapshots are caller-read-only."""
        held = tuple((k.index_select(1, self._index(i)),
                      v.index_select(1, self._index(i)))
                     for i, (k, v) in enumerate(zip(self.keys, self.values)))
        return self.length, held

    def restore(self, snapshot):
        length, held = snapshot
        if type(length) is not int or not 0 <= length <= self.context_limit or len(held) != self.cfg.layers:
            raise ValueError("Gemma E2B DFlash2 snapshot geometry differs")
        for i, (key, value) in enumerate(held):
            count = min(length, self.slots[i])
            expected = (self.cfg.kv_heads, count, self.cfg.head_dim)
            if (tuple(key.shape) != expected or tuple(value.shape) != expected
                    or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16
                    or key.device != self.keys[i].device or value.device != self.values[i].device):
                raise ValueError("Gemma E2B DFlash2 snapshot tensors differ")
        self.length = length
        for i, (key, value) in enumerate(held):
            index = self._index(i)
            self.keys[i].index_copy_(1, index, key)
            self.values[i].index_copy_(1, index, value)

    def attention_kv(self, layer: int, new_key, new_value):
        """Publish tentative noise only to scratch, never committed cache rows."""
        count = min(self.length, self.slots[layer])
        key, value = self.work_keys[layer], self.work_values[layer]
        if self.cfg.layer_types[layer] == "sliding_attention":
            index = self._index(layer)
            torch.index_select(self.keys[layer], 1, index, out=key[:, :count])
            torch.index_select(self.values[layer], 1, index, out=value[:, :count])
        key[:, count:count + new_key.shape[1]].copy_(new_key)
        value[:, count:count + new_value.shape[1]].copy_(new_value)
        return key[:, :count + new_key.shape[1]], value[:, :count + new_value.shape[1]]


class E2BChains:
    """One normal readback; exact full-row repair only for ambiguous unary ties.

    The exclusive lane finishes a packet on its producing CUDA stream. Immutable
    logits and selector owners remain live until membership repair completes.
    """

    def __init__(self, logits, hidden, weights, anchor: int, top_k: int, history, sampled: bool):
        from tensorfold.cuda.sampling import candidate_fields

        self.rows, self.top_k = logits.shape[0], top_k
        self.history, self.sampled = history, sampled
        self.full_width = logits.shape[1]
        # A lower guard edge proves complete membership without relying on topk's
        # undocumented tie order. Small exact sorts operate only on the guard set.
        values, provisional_ids, flags = candidate_fields(logits, top_k + 31)
        self.provisional_width = values.shape[1]
        id_order = torch.argsort(provisional_ids, dim=-1, stable=True)
        ordered_values = values.gather(1, id_order)
        order = torch.argsort(ordered_values, dim=-1, descending=True, stable=True)[:, :top_k]
        ids = provisional_ids.gather(1, id_order.gather(1, order))
        scores = self._selector_scores(logits, hidden, weights, anchor, ids)
        invalid = (flags.any() | torch.isnan(scores).any() | torch.isposinf(scores).any())
        self.words = torch.cat((
            invalid.reshape(1).to(torch.int64), provisional_ids.reshape(-1),
            values.contiguous().view(torch.int32).to(torch.int64).reshape(-1), ids.reshape(-1),
            scores.float().contiguous().view(torch.int32).to(torch.int64).reshape(-1)))
        self._source = logits, hidden, weights, anchor
        self.host = None
        self.repaired_support = False

    @staticmethod
    def _selector_scores(logits, hidden, weights, anchor, ids):
        # Unary membership is fixed before this predecessor-conditioned correction.
        values = logits.gather(1, ids)
        top_k = ids.shape[1]
        predecessor_ids = torch.cat((ids.new_full((1, top_k), anchor), ids[:-1]), dim=0)
        projected = F.linear(hidden, weights["candidate_selector.hidden_projection.weight"])
        pred = weights["candidate_selector.predecessor_codebook"][predecessor_ids]
        succ = weights["candidate_selector.successor_codebook"][ids]
        contexts = pred * projected[:, None, :]
        transitions = (contexts[:, :, None, :] * succ[:, None, :, :]).sum(-1)
        return values[:, None, :] + transitions

    def _readback(self):
        from tensorfold.cuda.sampling import canonical_candidates

        if self.host is not None:
            return
        words = self.words.cpu().numpy()
        if words[0]:
            raise ValueError("Gemma E2B DFlash2 proposal contains NaN or positive infinity")
        split = 1 + self.rows * self.provisional_width
        provisional_ids = words[1:split].reshape(self.rows, self.provisional_width)
        next_split = split + self.rows * self.provisional_width
        provisional_values = words[split:next_split].astype(np.int32).view(np.float32).reshape(
            self.rows, self.provisional_width)
        split = next_split
        next_split = split + self.rows * self.top_k
        ids = words[split:next_split].reshape(self.rows, self.top_k)
        scores = words[next_split:].astype(np.int32).view(np.float32).reshape(
            self.rows, self.top_k, self.top_k)
        logits, hidden, weights, anchor = self._source
        complete_ids = None

        def complete_row(row):
            nonlocal complete_ids
            if complete_ids is None:
                complete_ids = np.arange(self.full_width, dtype=np.int64)
            return logits[row].float().cpu().numpy(), complete_ids

        _, canonical_ids = canonical_candidates(
            provisional_values, provisional_ids, self.top_k, self.full_width, complete_row)
        if not np.array_equal(canonical_ids, ids):
            # A changed row also changes the next row's possible predecessors,
            # so rebuild the bounded table over the final complete support.
            ids_gpu = torch.tensor(canonical_ids, device=logits.device, dtype=torch.int64)
            corrected = self._selector_scores(logits, hidden, weights, anchor, ids_gpu)
            invalid = torch.isnan(corrected).any() | torch.isposinf(corrected).any()
            repaired = torch.cat((invalid.reshape(1).to(torch.int64),
                                  corrected.float().contiguous().view(torch.int32).to(torch.int64).reshape(-1)))
            readback = repaired.cpu().numpy()
            if readback[0]:
                raise ValueError("Gemma E2B DFlash2 repaired selector contains invalid scores")
            scores = readback[1:].astype(np.int32).view(np.float32).reshape(
                self.rows, self.top_k, self.top_k)
            self.repaired_support = True
        if not np.isfinite(scores).any(axis=-1).all():
            raise ValueError("Gemma E2B DFlash2 proposal has no finite candidate")
        self.host = canonical_ids, scores
        self.words = None
        self._source = None

    def _row(self, row, previous_slot, tokens, history):
        ids, scores = self.host
        values = scores[row, previous_slot].astype(np.float64)
        if history is not None:
            values = history.candidates(values[None], ids[row][None], tokens)[0]
        return values, ids[row]

    def finish(self, row: int, max_nodes: int, history=None, *, stop=None):
        if row != 0 or history is not self.history or self.sampled:
            raise ValueError("Gemma E2B DFlash2 greedy packet policy differs")
        if type(max_nodes) is not int or max_nodes < 0:
            raise ValueError("Gemma E2B DFlash2 proposal count must be nonnegative")
        self._readback()
        tokens, confidences, previous_slot, cumulative = [], [], 0, 0.0
        for i in range(min(max_nodes, self.rows)):
            values, ids = self._row(i, previous_slot, tokens, history)
            order = np.lexsort((ids, -values))
            previous_slot = int(order[0])
            cumulative += float(np.log(np.exp(values - values[previous_slot]).sum()))
            if stop is not None and tokens and cumulative > stop:
                break
            tokens.append(int(ids[previous_slot]))
            confidences.append(cumulative)
        return tokens, list(range(-1, len(tokens) - 1)), confidences

    def sample(self, row: int, max_nodes: int, sampling, first_position: int, history=None):
        if row != 0 or history is not self.history or not self.sampled:
            raise ValueError("Gemma E2B DFlash2 sampled packet policy differs")
        if type(max_nodes) is not int or max_nodes < 0:
            raise ValueError("Gemma E2B DFlash2 proposal count must be nonnegative")
        self._readback()
        tokens, probabilities, previous_slot = [], [], 0
        for i in range(min(max_nodes, self.rows)):
            values, ids = self._row(i, previous_slot, tokens, history)
            q = proposal_row(values, ids, sampling)
            token = draw(q, sampling.seed, first_position + i)
            previous_slot = int(np.flatnonzero(ids == token)[0])
            tokens.append(token)
            probabilities.append(q)
        return ChainPacket(tuple(tokens), tuple(probabilities), first_position)


class E2BDFlash2:
    """Original BF16 DFlash2 with raw borrowed verifier embedding and uncapped head."""

    def __init__(self, draft_dir: str | Path, embedding_weight, head_weight,
                 context_limit: int, device=None, bits: int = 16):
        from safetensors.torch import load_file
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        if type(bits) is not int or bits not in (8, 16):
            raise ValueError("Gemma E2B proposal precision is BF16(16) or private FP8(8)")
        self.cfg, _ = checked_headers(draft_dir)
        cfg = self.cfg
        self.context_limit = context_limit
        cfg.state_bytes(context_limit)  # validates the declared position envelope before allocation
        self.device = torch.device(device) if device is not None else embedding_weight.device
        if bits == 8 and self.device.type != "cuda":
            raise ValueError("Gemma E2B private FP8 proposals require CUDA")
        if (tuple(embedding_weight.shape) != (cfg.vocab, cfg.hidden)
                or tuple(head_weight.shape) != (cfg.vocab, cfg.hidden)
                or embedding_weight.dtype != torch.bfloat16 or head_weight.dtype != torch.bfloat16
                or embedding_weight.device != self.device or head_weight.device != self.device):
            raise ValueError("Gemma E2B DFlash2 borrowed verifier tensors differ")
        self.embedding_weight, self.head_weight = embedding_weight, head_weight
        self.bits = bits
        self.resident_bytes, self.loading_peak_bytes = cfg.memory_bytes(bits)
        self.weights = load_file(str(checkpoint_path(draft_dir, "model.safetensors")), device=str(self.device))
        self.norms = {}
        for name, weight in self.weights.items():
            if name in ("hidden_norm.weight", "norm.weight") or name.endswith(
                    ("input_layernorm.weight", "post_attention_layernorm.weight",
                     "q_norm.weight", "k_norm.weight")):
                norm = Qwen3RMSNorm(weight.numel(), eps=cfg.eps)
                norm.weight = torch.nn.Parameter(weight, requires_grad=False)
                self.norms[name] = norm
        self.linears = {}
        self.private_head = None
        if bits == 8:
            from tensorfold.cuda.requant import Head, requantize

            for name in cfg.projection_shapes():
                weight = self.weights.pop(name)
                self.linears[name] = requantize(weight, "fp8")
                del weight
            # This is the drafter's private copy, never the target projection.
            # Speculators applies no Gemma softcap to its proposal head.
            self.private_head = Head(SimpleNamespace(weight=head_weight), "fp8", softcap=None)
            from tensorfold.cuda.nvfp4.linear import _ext, _prompt_ext

            _prompt_ext()
            _ext()
        self.inv_freq = 1.0 / cfg.theta ** (
            torch.arange(cfg.head_dim // 2, device=self.device, dtype=torch.float32) * 2 / cfg.head_dim)
        self.tap_layers = cfg.tap_layers
        self.block = cfg.block
        self.rows = cfg.block
        self.parameter_bytes = cfg.parameter_bytes

    def make_state(self):
        return E2BDraftState(self.cfg, self.device, self.context_limit)

    def _norm(self, x, name):
        return self.norms[name](x)

    def _linear(self, x, name, *, prompt=False):
        if name in self.linears:
            op = self.linears[name]
            return op.prefill(x.contiguous()) if prompt else op(x.contiguous())
        return F.linear(x, self.weights[name])

    @torch.inference_mode()
    def append(self, state: E2BDraftState, taps, *, prompt: bool = False):
        cfg = self.cfg
        if (taps.ndim != 2 or taps.shape[1] != len(cfg.tap_layers) * cfg.hidden
                or taps.dtype != torch.bfloat16 or taps.device != self.device
                or state.cfg != cfg or state.length + taps.shape[0] > state.context_limit):
            raise ValueError("Gemma E2B DFlash2 committed taps differ")
        if not taps.shape[0]:
            return
        projected = self._norm(self._linear(taps, "fc.weight", prompt=prompt), "hidden_norm.weight")
        positions = torch.arange(state.length, state.length + len(taps), device=self.device)
        for i, kind in enumerate(cfg.layer_types):
            prefix = f"layers.{i}.self_attn."
            key = self._linear(projected, prefix + "k_proj.weight", prompt=prompt).view(-1, cfg.kv_heads, cfg.head_dim)
            key = _rope(self._norm(key, prefix + "k_norm.weight").transpose(0, 1), positions, self.inv_freq)
            value = self._linear(projected, prefix + "v_proj.weight", prompt=prompt).view(-1, cfg.kv_heads, cfg.head_dim).transpose(0, 1)
            slots = state.slots[i]
            count = min(len(taps), slots)
            index = positions[-count:] % slots
            state.keys[i].index_copy_(1, index, key[:, -count:])
            state.values[i].index_copy_(1, index, value[:, -count:])
        state.length += len(taps)

    def _layer(self, i, x, state, positions):
        cfg = self.cfg
        base = f"layers.{i}."
        residual = x
        normalized = self._norm(x, base + "input_layernorm.weight")
        conv = base + "attention_conv."
        delta = self._linear(normalized, conv + "kernel_projection.weight").view(
            len(x), 2, 2, cfg.hidden // cfg.group_size)
        prepared = _conv(normalized, delta, self.weights[conv + "base_kernel"], 0, cfg.group_size)
        prefix = base + "self_attn."
        q = self._linear(prepared, prefix + "q_proj.weight").view(-1, cfg.heads, cfg.head_dim)
        k = self._linear(prepared, prefix + "k_proj.weight").view(-1, cfg.kv_heads, cfg.head_dim)
        v = self._linear(prepared, prefix + "v_proj.weight").view(-1, cfg.kv_heads, cfg.head_dim).transpose(0, 1)
        q = _rope(self._norm(q, prefix + "q_norm.weight").transpose(0, 1), positions, self.inv_freq)
        k = _rope(self._norm(k, prefix + "k_norm.weight").transpose(0, 1), positions, self.inv_freq)
        k, v = state.attention_kv(i, k, v)
        attended = F.scaled_dot_product_attention(q[None], k[None], v[None],
                                                is_causal=False, enable_gqa=True)
        attended = attended[0].transpose(0, 1).reshape(len(x), cfg.hidden)
        attended = self._linear(attended, prefix + "o_proj.weight")
        x = residual + _conv(attended, delta, self.weights[conv + "base_kernel"], 1, cfg.group_size)
        residual = x
        normalized = self._norm(x, base + "post_attention_layernorm.weight")
        conv = base + "mlp_conv."
        delta = self._linear(normalized, conv + "kernel_projection.weight").view(
            len(x), 2, 2, cfg.hidden // cfg.group_size)
        prepared = _conv(normalized, delta, self.weights[conv + "base_kernel"], 0, cfg.group_size)
        mlp = self._linear(F.silu(self._linear(prepared, base + "mlp.gate_proj.weight")) *
                           self._linear(prepared, base + "mlp.up_proj.weight"), base + "mlp.down_proj.weight")
        return residual + _conv(mlp, delta, self.weights[conv + "base_kernel"], 1, cfg.group_size)

    @torch.inference_mode()
    def propose(self, state: E2BDraftState, anchor: int, count: int, history=None, *,
                sampled: bool = False, sampling=None):
        cfg = self.cfg
        if type(anchor) is not int or not 0 <= anchor < cfg.vocab:
            raise ValueError("Gemma E2B DFlash2 anchor is outside the verifier vocabulary")
        if type(count) is not int or count < 0 or state.cfg != cfg:
            raise ValueError("Gemma E2B DFlash2 proposal geometry differs")
        if sampled and (sampling is None or sampling.temperature <= 0):
            raise ValueError("Sampled Gemma E2B proposals require positive-temperature policy")
        length = min(count + 1, cfg.block, min(cfg.max_position, state.context_limit) - state.length)
        if length < 2 or state.length < 1:
            return None
        tokens = torch.tensor([anchor] + [cfg.mask_id] * (length - 1), device=self.device, dtype=torch.long)
        x = F.embedding(tokens, self.embedding_weight)  # raw rows: no Gemma embedding multiplier
        positions = torch.arange(state.length, state.length + length, device=self.device)
        for i in range(cfg.layers):
            x = self._layer(i, x, state, positions)
        hidden = self._norm(x[1:], "norm.weight")
        logits = (F.linear(hidden, self.head_weight) if self.private_head is None else self.private_head(hidden))
        return E2BChains(logits, hidden, self.weights, anchor, cfg.selector_top_k, history, sampled)
