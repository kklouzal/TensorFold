"""EAGLE3 adapter to native HF Llama components, with committed shifted pairs.

The first-layer boundary follows Speculators v0.8.0 commit894e71f9. Library
attention, masks, RMSNorm, rotary and MLP own all mathematical algorithms. The
head owns its checkpoint embedding/head and independent sliding KV, never the
verifier's parameters or caches. Only append commits accepted target inputs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import DynamicCache, LlamaConfig
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

from tensorfold.cuda.tensor_file import checkpoint_path
from tensorfold.engine.rejection import ChainPacket, ProbabilityRow
from ..eagle3_metadata import checked_headers


class Eagle3FirstLayer(LlamaDecoderLayer):
    """Public Llama layer with the checkpoint's fused embed/feature input gap."""
    def __init__(self, config):
        super().__init__(config, 0)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for name, heads in (('q_proj', config.num_attention_heads),
                            ('k_proj', config.num_key_value_heads), ('v_proj', config.num_key_value_heads)):
            setattr(self.self_attn, name, nn.Linear(2 * config.hidden_size, heads * config.head_dim, bias=False))

    def forward(self, inputs, *, attention_mask, position_embeddings, past_key_values):
        embeds, hidden = inputs.chunk(2, dim=-1)
        embeds, hidden = self.input_layernorm(embeds), self.hidden_norm(hidden)
        residual = hidden  # norm_before_residual=True in this exact checkpoint
        attended, _ = self.self_attn(torch.cat((embeds, hidden), dim=-1),
            attention_mask=attention_mask, position_embeddings=position_embeddings, past_key_values=past_key_values)
        hidden = residual + attended
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Eagle3Model(nn.Module):
    """Checkpoint-shaped module; CPU construction also supports reduced test configs."""
    def __init__(self, config, draft_vocab=32000, num_aux=3):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.input_norm = LlamaRMSNorm(h * num_aux, eps=config.rms_norm_eps)
        self.fc = nn.Linear(h * num_aux, h, bias=False)
        self.layers = nn.ModuleList([Eagle3FirstLayer(config)])
        self.norm = LlamaRMSNorm(h, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(h, draft_vocab, bias=False)
        self.rotary = LlamaRotaryEmbedding(config)
        self.register_buffer('d2t', torch.zeros(draft_vocab, dtype=torch.int64))
        self.register_buffer('t2d', torch.zeros(config.vocab_size, dtype=torch.bool))

    def project(self, taps):
        return self.fc(self.input_norm(taps))

    def step(self, ids, hidden, cache, positions):
        embedded = self.embed_tokens(ids)
        inputs = torch.cat((embedded, hidden), dim=-1)
        mask = create_sliding_window_causal_mask(self.config, inputs, None, cache,
                                               position_ids=positions, allow_is_causal_skip=False)
        out = self.layers[0](inputs, attention_mask=mask,
                            position_embeddings=self.rotary(inputs, positions), past_key_values=cache)
        # norm_output=True: normalized output is recurrent feature authority.
        return self.norm(out)


@dataclass(frozen=True)
class EagleSnapshot:
    owner: object
    length: int
    cache: object
    last_taps: object


class EagleState:
    def __init__(self, owner):
        self.owner = owner
        self.reset()

    def reset(self):
        self.length = 0
        self.poisoned = False
        self.cache = DynamicCache(config=self.owner.model.config)
        self.last_taps = None

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for layer in self.cache.layers
                   for t in (layer.keys, layer.values) if t is not None) + (
                       0 if self.last_taps is None else self.last_taps.numel() * self.last_taps.element_size())

    def snapshot(self):
        if self.poisoned:
            raise RuntimeError('EAGLE3 state failed; reset or restore before reuse')
        return EagleSnapshot(self.owner, self.length, copy.deepcopy(self.cache),
                             None if self.last_taps is None else self.last_taps.clone())

    def restore(self, snapshot):
        if not isinstance(snapshot, EagleSnapshot) or snapshot.owner is not self.owner:
            raise ValueError('EAGLE3 snapshot belongs to another head')
        cache = copy.deepcopy(snapshot.cache)
        last_taps = None if snapshot.last_taps is None else snapshot.last_taps.clone()
        self.length, self.cache, self.last_taps = snapshot.length, cache, last_taps
        self.poisoned = False


class EagleChains:
    def __init__(self, tokens, history, sampled):
        self.tokens, self.history, self.sampled = tuple(tokens), history, sampled

    def _tokens(self, row, count, history, sampled):
        if row != 0 or history is not self.history or sampled is not self.sampled:
            raise ValueError('EAGLE3 proposal policy differs')
        if type(count) is not int or count < 0:
            raise ValueError('EAGLE3 proposal count must be nonnegative')
        return self.tokens[:count]

    def finish(self, row, max_nodes, history=None, *, stop=None):
        tokens = self._tokens(row, max_nodes, history, False)
        return list(tokens), list(range(-1, len(tokens) - 1)), [0.] * len(tokens)

    def sample(self, row, max_nodes, sampling, first_position, history=None):
        tokens = self._tokens(row, max_nodes, history, True)
        if sampling is None or sampling.temperature <= 0:
            raise ValueError('EAGLE3 sampled verifier requires positive temperature')
        return ChainPacket(tokens, tuple(ProbabilityRow([t], [1.]) for t in tokens), first_position)


class Gemma4Eagle3Head:
    feature_source = 'taps'
    tap_layers = (1, 20, 38)
    needs_token_ids = True
    bits = 16

    def __init__(self, directory: str | Path, target, context_limit, device=None):
        from safetensors.torch import load_file
        self.cfg, _ = checked_headers(directory)
        self.cfg.state_bytes(context_limit)
        self.device = target.device if device is None else torch.device(device)
        if target.config.hidden != self.cfg.hidden or target.config.vocab != self.cfg.vocab:
            raise ValueError('EAGLE3 requires E4B verifier geometry')
        if self.device != target.device:
            raise ValueError('EAGLE3 and target must share a device')
        weights = model = None
        self.model = None
        try:
            weights = load_file(str(checkpoint_path(directory, 'model.safetensors')), device='cpu')
            ordering = torch.arange(self.cfg.draft_vocab, dtype=torch.int64) + weights['d2t']
            selected = weights['t2d'].nonzero().flatten()
            if not torch.equal(ordering, selected):
                raise ValueError('EAGLE3 vocabulary offset mapping must exactly match target membership')
            config = LlamaConfig(hidden_size=self.cfg.hidden, intermediate_size=self.cfg.width,
                num_hidden_layers=1, num_attention_heads=self.cfg.heads, num_key_value_heads=self.cfg.kv_heads,
                head_dim=self.cfg.head_dim, vocab_size=self.cfg.vocab, hidden_act='silu',
                max_position_embeddings=self.cfg.max_position, rms_norm_eps=self.cfg.eps,
                rope_parameters={'rope_type': 'default', 'rope_theta': self.cfg.theta},
                sliding_window=self.cfg.window, layer_types=['sliding_attention'], attention_bias=False, mlp_bias=False)
            config._attn_implementation = 'sdpa'
            # Meta avoids allocating and initializing a second 1.7GB CPU parameter owner.
            with torch.device('meta'):
                model = Eagle3Model(config, self.cfg.draft_vocab)
            model.load_state_dict(weights, strict=True, assign=True)
            # Rotary inv_freq is a nonpersistent meta buffer, created from public HF.
            model.rotary = LlamaRotaryEmbedding(config)
            self.model = model.eval().requires_grad_(False).to(self.device)
            self.context_limit = context_limit
            self.block = self.rows = self.cfg.block
            self.parameter_bytes = self.cfg.parameter_bytes
            self.resident_bytes, self.loading_peak_bytes = self.cfg.memory_bytes()
        except BaseException:
            # Drop only our aliases. Native exception/library frames remain
            # intact and may retain partial consumers until their owner retires.
            self.model = None
            raise
        finally:
            weights = model = None

    def make_state(self):
        return EagleState(self)

    @torch.inference_mode()
    def append(self, state, taps, *, token_ids, prompt=False):
        if state.poisoned:
            raise RuntimeError('EAGLE3 state failed; reset or restore before reuse')
        if (state.owner is not self or taps.ndim != 2 or taps.shape[1] != self.cfg.hidden * 3
                or taps.dtype != torch.bfloat16 or taps.device != self.device
                or len(token_ids) != len(taps) or any(type(t) is not int or not 0 <= t < self.cfg.vocab for t in token_ids)
                or state.length + len(taps) > self.context_limit):
            raise ValueError('EAGLE3 committed target taps/token alignment differs')
        if not len(taps):
            return
        begin = 1 if state.last_taps is None else 0
        # Bound native attention/MLP work; causal chunks preserve the full prompt.
        prior = torch.cat((state.last_taps, taps[:-1]), dim=0) if begin == 0 else taps[:-1]
        ids = token_ids[begin:]
        try:
            for start in range(0, len(ids), 128):
                end = min(len(ids), start + 128)
                hidden = self.model.project(prior[start:end].unsqueeze(0))
                positions = torch.arange(state.length + begin + start, state.length + begin + end,
                                         device=self.device).unsqueeze(0)
                self.model.step(torch.tensor([ids[start:end]], device=self.device), hidden, state.cache, positions)
            state.last_taps = taps[-1:].clone()
            state.length += len(taps)
        except BaseException:
            state.poisoned = True
            raise

    @torch.inference_mode()
    def propose(self, state, anchor, count, history=None, *, sampled=False, sampling=None):
        if state.poisoned:
            raise RuntimeError('EAGLE3 state failed; reset or restore before reuse')
        if (state.owner is not self or state.last_taps is None or type(anchor) is not int
                or not 0 <= anchor < self.cfg.vocab or type(count) is not int
                or not 0 <= count < self.block or not 0 < state.length <= self.context_limit):
            raise ValueError('EAGLE3 proposal exceeds committed state/context')
        # Verification also feeds the anchor: only a prefix that leaves that
        # row plus all proposed input rows inside native context is admissible.
        count = min(count, max(0, self.context_limit - state.length - 1))
        if not count:
            return None
        cache = copy.deepcopy(state.cache)
        hidden = self.model.project(state.last_taps.unsqueeze(0))
        tokens, token = [], anchor
        for index in range(count):
            positions = torch.tensor([[state.length + index]], device=self.device)
            hidden = self.model.step(torch.tensor([[token]], device=self.device), hidden, cache, positions)
            draft_id = self.model.lm_head(hidden).argmax(dim=-1).reshape(())
            token = int((draft_id + self.model.d2t[draft_id]).item())
            tokens.append(token)
        return EagleChains(tokens, history, sampled)
