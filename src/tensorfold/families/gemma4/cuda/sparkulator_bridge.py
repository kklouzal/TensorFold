"""Opt-in Sparkulator conditioning, borrowing main caches without a raw-KV alias.

Local donors use the target's already transformed ring. Full donors apply the
public HF normalizer/RoPE to committed raw K=V. This is target-conditioned data,
not a claim to reproduce an independent HF main-model forward bit-for-bit.
"""

from dataclasses import dataclass

import torch
from torch import nn
from tensorfold.cleanup import rollback


@dataclass(frozen=True)
class BridgeSnapshot:
    position: int
    hidden: object
    owner_token: object = None


class SparkulatorState:
    def __init__(self, owner, target_state):
        self._initialize(owner.target, target_state, owner.context_limit, owner)

    @classmethod
    def from_target(cls, target, target_state, context_limit):
        """Real-main-only recorder/conditioner; no assistant model or fake head."""
        value = cls.__new__(cls)
        value._initialize(target, target_state, context_limit, None)
        return value

    def _initialize(self, target, target_state, context_limit, owner):
        from .forward import Gemma4Target
        from .state import State

        if not isinstance(target, Gemma4Target) or not isinstance(target_state, State):
            raise ValueError("Sparkulator conditioning requires actual main target/state owners")
        from transformers import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextRotaryEmbedding

        self.owner, self.target_state = owner, target_state
        self.device, self.context_limit = target.device, context_limit
        self.window = target.config.window
        self._snapshot_token = object()
        c = target.config
        if (c.hidden, c.kv_heads, c.global_kv_heads, c.head_dim, c.global_head_dim, c.window) != (
            2816,
            8,
            2,
            256,
            512,
            1024,
        ):
            raise ValueError("Sparkulator requires its actual main cache geometry")
        if (
            target_state.config != c
            or torch.device(target_state.device) != target.device
            or target_state.context != context_limit
        ):
            raise ValueError("Sparkulator main state belongs to another cache contract")
        self.local_layer = max(i for i, k in enumerate(c.layer_types) if k == "sliding_attention")
        self.full_layer = max(i for i, k in enumerate(c.layer_types) if k == "full_attention")
        self.local_slot = target_state.slot[self.local_layer][1]
        self.full_slot = target_state.slot[self.full_layer][1]
        self.keys = self.values = self.rope = self.knorm = self.vnorm = None
        self.last_hidden = None
        self.position = self.valid_full = 0
        self.poisoned = False
        try:
            self.keys = torch.empty((context_limit, 2, 512), dtype=torch.bfloat16, device=target.device)
            self.values = torch.empty_like(self.keys)
            config = Gemma4TextConfig.from_dict(target.weights.raw_config["text_config"])
            with torch.device(target.device):
                self.rope = Gemma4TextRotaryEmbedding(config)
            with torch.device("meta"):
                self.knorm = Gemma4RMSNorm(512, eps=c.eps)
            self.knorm.weight = nn.Parameter(target.weights.layers[self.full_layer].k_norm, requires_grad=False)
            self.vnorm = Gemma4RMSNorm(512, eps=c.eps, with_scale=False)
        except BaseException as primary:

            def retire():
                torch.cuda.synchronize(target.device)
                self.keys = self.values = self.rope = self.knorm = self.vnorm = None

            _native_cause = BaseException.__cause__.__get__(primary)
            _native_context = BaseException.__context__.__get__(primary)
            try:
                rollback((self, target), primary, retire)
            except BaseException as transport:
                if transport is not primary:
                    raise primary from transport
                raise
            raise

    @property
    def nbytes(self):
        return (
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
            + sum(x.numel() * x.element_size() for x in self.rope.buffers())
            + (0 if self.last_hidden is None else self.last_hidden.numel() * self.last_hidden.element_size())
        )

    def reset(self):
        self.last_hidden = None
        self.position = self.valid_full = 0
        self.poisoned = False

    def snapshot(self):
        if self.poisoned or self.target_state.poisoned or self.target_state.position != self.position:
            raise RuntimeError("Sparkulator snapshot needs matching committed main state")
        return BridgeSnapshot(
            self.position, None if self.last_hidden is None else self.last_hidden.clone(), self._snapshot_token
        )

    def restore(self, snapshot):
        if (
            not isinstance(snapshot, BridgeSnapshot)
            or snapshot.owner_token is not self._snapshot_token
            or snapshot.position != self.target_state.position
            or not 0 <= snapshot.position <= self.context_limit
        ):
            raise ValueError("Sparkulator snapshot position differs from restored main state")
        if (snapshot.position == 0) != (snapshot.hidden is None) or (
            snapshot.hidden is not None
            and (
                snapshot.hidden.shape != (1, 1, 2816)
                or snapshot.hidden.dtype != torch.bfloat16
                or snapshot.hidden.device != self.device
            )
        ):
            raise ValueError("Sparkulator snapshot hidden geometry differs")
        # Main state is restored first; equal positions can describe different
        # branches. A failed hidden clone must not leave the old pair usable.
        self.poisoned = True
        value = None if snapshot.hidden is None else snapshot.hidden.clone()
        self.last_hidden = value
        self.position = snapshot.position
        self.valid_full = 0  # old derived bytes may belong to a different restored branch
        self.poisoned = False

    def append(self, hidden):
        if (
            self.poisoned
            or self.target_state.poisoned
            or hidden.ndim != 2
            or hidden.shape[1] != 2816
            or hidden.shape[0] <= 0
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.device
            or self.target_state.position != self.position + hidden.shape[0]
        ):
            raise ValueError("Sparkulator append must publish exactly committed normalized hidden rows")
        value = hidden[-1:].unsqueeze(0).clone()
        self.last_hidden = value
        self.position = self.target_state.position

    @torch.inference_mode()
    def view(self):
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

        state = self.target_state
        if (
            self.poisoned
            or state.poisoned
            or state.position != self.position
            or self.position < 1
            or self.last_hidden is None
        ):
            raise RuntimeError("Sparkulator needs a matching healthy committed main prefix")
        p = self.position
        try:
            # Small fixed chunks bound transient RMS/rotary owners. Derived
            # prefix is reused on this branch; target raw/inv caches are never changed.
            while self.valid_full < p:
                begin = self.valid_full
                end = min(p, begin + 512)
                raw = state.raw[self.full_slot, begin:end].unsqueeze(0)
                pos = torch.arange(begin, end, dtype=torch.long, device=self.device)[None]
                cos, sin = self.rope(raw, pos, "full_attention")
                key = apply_rotary_pos_emb(self.knorm(raw), cos, sin, unsqueeze_dim=2)
                value = self.vnorm(raw)
                self.keys[begin:end].copy_(key[0])
                self.values[begin:end].copy_(value[0])
                self.valid_full = end
            count = min(p, self.window)
            if p - count < max(0, state.high_water - state.ring):
                raise RuntimeError("Sparkulator needs restored sliding history, not overwritten ring rows")
            indices = torch.arange(p - count, p, device=self.device, dtype=torch.long) % state.ring
            local_k = state.keys[self.local_slot].index_select(0, indices).transpose(0, 1).unsqueeze(0)
            local_v = state.values[self.local_slot].index_select(0, indices).transpose(0, 1).unsqueeze(0)
            return {
                "position": p,
                "last_hidden_state": self.last_hidden,
                "attention_mask": None,
                "shared_kv_states": {
                    "sliding_attention": (local_k, local_v),
                    "full_attention": (
                        self.keys[:p].transpose(0, 1).unsqueeze(0),
                        self.values[:p].transpose(0, 1).unsqueeze(0),
                    ),
                },
            }
        except BaseException:
            self.poisoned = True
            raise
        finally:
            raw = pos = cos = sin = key = value = local_k = local_v = None
