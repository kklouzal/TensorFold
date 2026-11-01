"""Private local-head all-eligible mask policies; no target/global mask change.

This module is unconnected until its owner explicitly prepares an assistant
policy/module pair. Default 'native' changes no mask/backend. Eligibility uses
only an exclusive per-propose owner frame (original held mask is None), never a
GPU all()/host readback. The forced-efficient policy additionally requires
process-exclusive SDPA ownership: pinned PyTorch backend flags are shared across
threads, and the public context restores them on exit. Proposal arithmetic can change with the selected SDPA
provider; actual q/delta law and unchanged target p remain mandatory.
"""
from contextlib import contextmanager

from torch.nn.attention import SDPBackend, sdpa_kernel

from .e2b_attention import E2BTextAttention

LOCAL_MASK_POLICIES = ('native', 'none-default', 'none-efficient')


class LocalMaskFrame:
    """One head lane's explicit borrowed-validity scope, never a prefix cache.

    The caller owns the committed target views until dependent recurrence and
    packet readback retire. Nested scopes restore immediate prior metadata;
    failure clears this frame exactly as success. No Tensor is retained here.
    """
    def __init__(self, selection='native'):
        if type(selection) is not str or selection not in LOCAL_MASK_POLICIES:
            raise ValueError('Assistant local mask policy must be native, none-default or none-efficient')
        self.selection = selection
        self.active = False
        self.unpadded = False

    @contextmanager
    def bind(self, original_attention_mask):
        previous = self.active, self.unpadded
        self.active, self.unpadded = True, original_attention_mask is None
        try:
            yield
        finally:
            self.active, self.unpadded = previous


class AssistantLocalAttention(E2BTextAttention):
    """Reuse native owned projection/attention glue, vary LOCAL mask/provider only."""
    def __init__(self, native, attention_policy, frame):
        if (not native.is_kv_shared_layer or native.config.num_attention_heads != 4
                or native.num_key_value_groups != 2 or not isinstance(frame, LocalMaskFrame)):
            raise ValueError('Assistant local mask policies require native shared-KV Q4/KV2 attention')
        super().__init__(native, attention_policy)
        self.local_frame = frame

    def forward(self, hidden_states, position_embeddings, attention_mask, shared_kv_states,
                past_key_values=None, **kwargs):
        frame = self.local_frame
        eligible = (frame.selection != 'native' and frame.active and frame.unpadded
                    and self.policy.prompt and not self.policy.reference_depth
                    and self.layer_type == 'sliding_attention' and hidden_states.shape[-2] == 1
                    and shared_kv_states['sliding_attention'][0].shape[-2] <= self.sliding_window)
        mask = None if eligible else attention_mask
        if eligible and frame.selection == 'none-efficient':
            # Scope exactly this LOCAL call; globalD512 retains native dispatch.
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                return super().forward(hidden_states, position_embeddings, mask, shared_kv_states,
                                       past_key_values=past_key_values, **kwargs)
        return super().forward(hidden_states, position_embeddings, mask, shared_kv_states,
                               past_key_values=past_key_values, **kwargs)
