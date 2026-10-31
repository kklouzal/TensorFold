"""Metadata boundary for the original BF16 E4B Speculators EAGLE3 head."""
from dataclasses import dataclass
import math

from tensorfold.checkpoint_metadata import read_json


@dataclass(frozen=True)
class Eagle3Config:
    hidden: int
    width: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    draft_vocab: int
    window: int
    max_position: int
    tap_hidden_indices: tuple[int, ...]
    eps: float
    theta: float
    block: int = 4

    @property
    def tap_layers(self):
        return tuple(index - 1 for index in self.tap_hidden_indices)

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get('architectures') != ['Eagle3DraftModel']:
            raise ValueError('EAGLE3 requires Eagle3DraftModel metadata')
        if (raw.get('speculators_model_type') != 'eagle3' or raw.get('dtype') != 'bfloat16'
                or raw.get('quantization_config') not in (None, {})
                or raw.get('torch_dtype') not in (None, 'bfloat16')):
            raise ValueError('EAGLE3 requires original unquantized BF16 weights')
        text = raw.get('transformer_layer_config', {})
        profile = {'model_type': 'llama', 'hidden_size': 2560, 'intermediate_size': 10240,
                   'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 256,
                   'num_hidden_layers': 1, 'vocab_size': 262144, 'sliding_window': 2048,
                   'max_position_embeddings': 131072, 'hidden_act': 'silu',
                   'attention_bias': False, 'mlp_bias': False, 'use_sliding_window': True,
                   'layer_types': ['sliding_attention']}
        if any(text.get(key) != value or type(text.get(key)) is not type(value)
               for key, value in profile.items()):
            raise ValueError('EAGLE3 requires the declared native E4B Llama draft profile')
        if (raw.get('target_hidden_size') != 2560 or raw.get('vocab_size') != 262144
                or raw.get('draft_vocab_size') != 32000
                or raw.get('eagle_aux_hidden_state_layer_ids') != [2, 21, 39]
                or raw.get('norm_before_fc') is not True or raw.get('fc_norm') is not False
                or raw.get('norm_before_residual') is not True or raw.get('norm_output') is not True
                or raw.get('embed_requires_grad') is not False or text.get('attention_dropout') != 0.0):
            raise ValueError('EAGLE3 auxiliary state, vocabulary or normalization contract differs')
        if text.get('rope_parameters') != {'rope_theta': 10000.0, 'rope_type': 'default'}:
            raise ValueError('EAGLE3 requires its unscaled full-head rotary schedule')
        eps = text.get('rms_norm_eps')
        if type(eps) not in (float, int) or not math.isfinite(eps) or eps != 1e-6:
            raise ValueError('EAGLE3 normalization epsilon differs')
        declared = raw.get('speculators_config', {})
        if declared.get('algorithm') != 'eagle3' or declared.get('default_proposal_method') != 'greedy':
            raise ValueError('EAGLE3 requires its actual deterministic greedy proposal law')
        if declared.get('proposal_methods') != [dict(accept_tolerance=0.0, proposal_type='greedy',
                                                     speculative_tokens=3, verifier_accept_k=1)]:
            raise ValueError('EAGLE3 requires its declared three-proposal greedy contract')
        return cls(2560, 10240, 8, 2, 256, 262144, 32000, 2048, 131072, (2, 21, 39), 1e-6, 10000.)

    @classmethod
    def read(cls, directory):
        return cls.from_dict(read_json(directory, 'config.json', required=True))

    def shapes(self):
        h, d = self.hidden, self.head_dim
        return {'d2t': (self.draft_vocab,), 't2d': (self.vocab,),
                'embed_tokens.weight': (self.vocab, h), 'fc.weight': (h, h * 3),
                'input_norm.weight': (h * 3,), 'norm.weight': (h,),
                'lm_head.weight': (self.draft_vocab, h),
                'layers.0.hidden_norm.weight': (h,), 'layers.0.input_layernorm.weight': (h,),
                'layers.0.post_attention_layernorm.weight': (h,),
                'layers.0.self_attn.q_proj.weight': (self.heads * d, h * 2),
                'layers.0.self_attn.k_proj.weight': (self.kv_heads * d, h * 2),
                'layers.0.self_attn.v_proj.weight': (self.kv_heads * d, h * 2),
                'layers.0.self_attn.o_proj.weight': (h, self.heads * d),
                'layers.0.mlp.gate_proj.weight': (self.width, h),
                'layers.0.mlp.up_proj.weight': (self.width, h),
                'layers.0.mlp.down_proj.weight': (h, self.width)}

    @property
    def parameter_bytes(self):
        return sum(math.prod(shape) * (8 if key == 'd2t' else 1 if key == 't2d' else 2)
                   for key, shape in self.shapes().items())

    def memory_bytes(self, bits=16):
        if type(bits) is not int or bits != 16:
            raise ValueError('EAGLE3 candidate retains original BF16 proposal weights')
        return self.parameter_bytes, self.parameter_bytes * 2

    def state_bytes(self, context):
        if type(context) is not int or not 1 <= context <= self.max_position:
            raise ValueError('EAGLE3 context exceeds native position range')
        # Committed cache, snapshot and disposable proposal copy; bounded native
        # encoder/chunk work is separately charged by the owning target engine.
        return 3 * (min(context, self.window) + self.block) * self.kv_heads * self.head_dim * 4 + 3 * self.hidden * 2


def checked_headers(directory):
    from tensorfold.cuda.capacity import headers
    cfg = Eagle3Config.read(directory)
    found, expected = headers(directory), cfg.shapes()
    if set(found) != set(expected):
        raise ValueError('EAGLE3 tensor inventory differs')
    for key, shape in expected.items():
        dtype = 'I64' if key == 'd2t' else 'BOOL' if key == 't2d' else 'BF16'
        if tuple(found[key]['shape']) != shape or found[key]['dtype'] != dtype:
            raise ValueError('EAGLE3 tensor geometry/dtype differs: ' + key)
    return cfg, found
