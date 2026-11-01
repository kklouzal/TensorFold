"""Unconnected native-BF16 assistant synchronization candidate.

This owned adapter borrows the native centroid/ordering authorities. It keeps
native dense finite-floor logits and full-vocabulary lowest-ID argmax. Selection
requires GPU numerical and integrated performance evidence; no runtime enables
this candidate merely by importing it.
"""
from __future__ import annotations

import torch
from torch import nn


class TensorMaskedEmbedder(nn.Module):
    """Native ordered-vocabulary arithmetic with a tensor-valued BF16 floor.

    For every finite BF16 scalar, FP32(value)-1 -> BF16 equals native Python
    binary64(value)-1 -> BF16 (exhaustive65280 finite-pattern CPU proof).
    NaN/Inf are retained as invalid logits for the shipping packet validator;
    they are never clamped into acceptable proposal scores.
    """
    def __init__(self, native):
        super().__init__()
        if native.centroids.weight.dtype != torch.bfloat16:
            raise ValueError("Tensor Gemma assistant floor requires native BF16 weights")
        self.centroids = native.centroids
        self.register_buffer("token_ordering", native.token_ordering)
        self.centroid_intermediate_top_k = native.centroid_intermediate_top_k
        self.num_centroids = native.num_centroids
        self.vocab_size = native.vocab_size
        self.vocab_size_per_centroid = native.vocab_size_per_centroid
        self.hidden_size = native.hidden_size

    def forward(self, hidden_states, lm_head_weight):
        if hidden_states.dtype != torch.bfloat16 or lm_head_weight.dtype != torch.bfloat16:
            raise ValueError("Tensor Gemma assistant floor requires native BF16 operands")
        batch, seq_len = hidden_states.shape[:2]
        centroid_logits = self.centroids(hidden_states)
        _, top_k_indices = torch.topk(centroid_logits, k=self.centroid_intermediate_top_k, dim=-1)
        token_ordering = self.token_ordering.long()
        canonical_positions = token_ordering.view(self.num_centroids, self.vocab_size_per_centroid)
        selected_canonical = canonical_positions[top_k_indices]
        selected_flat = selected_canonical.reshape(-1)
        selected_embeddings = lm_head_weight[selected_flat].view(
            batch, seq_len, self.centroid_intermediate_top_k * self.vocab_size_per_centroid, self.hidden_size)
        selected_logits = (hidden_states.unsqueeze(-2) @ selected_embeddings.transpose(-1, -2)).squeeze(-2)
        # Global reduction across all B/L rows, exactly as the native scalar min.
        floor = (selected_logits.float().amin() - 1.0).to(torch.bfloat16)
        output = floor.expand(batch, seq_len, self.vocab_size).clone()
        return output.scatter_(dim=-1, index=selected_canonical.view(batch, seq_len, -1), src=selected_logits)


class DeviceGreedyChain:
    """Owned tiny GPU packet; complete all score checks before publishing IDs.

    The caller keeps the producing exclusive CUDA stream current through this
    readback. words owns [token,invalid-domain,finite-winner] for every step.
    Returned tuple retains no GPU/logit/hidden/cache storage.
    """
    def __init__(self, words):
        self.words = words
        self._tokens = None

    def readback(self):
        if self._tokens is None:
            fields = self.words.cpu().numpy()  # sole chain host synchronization
            if fields[:, 1].any():
                raise ValueError("sampling scores contain NaN or positive infinity")
            if not fields[:, 2].all():
                raise ValueError("a complete vocabulary row needs at least one finite score")
            self._tokens = tuple(int(token) for token in fields[:, 0])
            self.words = None
        return self._tokens


@torch.inference_mode()
def device_greedy_chain(model, embedding_module, hidden, shared_kv_states, attention_mask,
                        anchor: int, count: int, position: int):
    """Queue a native recurrent chain without per-step token/flag readbacks.

    Caller has already validated target context and donor ownership. The anchor
    position, mask and shared KV remain frozen for every recurrent model call.
    This candidate expects model.masked_embedding=TensorMaskedEmbedder, installed
    only by its explicit owner after numerical/performance admission.
    """
    from tensorfold.cuda.sampling import launch_streams

    if type(count) is not int or not 1 <= count <= 7:
        raise ValueError("Device Gemma assistant permits one through seven steps")
    if type(anchor) is not int or not 0 <= anchor < model.config.get_text_config().vocab_size:
        raise ValueError("Device Gemma assistant anchor is outside its vocabulary")
    if type(position) is not int or position < 0:
        raise ValueError("Device Gemma assistant position must be nonnegative")
    token = torch.tensor([[anchor]], dtype=torch.long, device=hidden.device)
    positions = torch.tensor([[position]], dtype=torch.long, device=hidden.device)
    words = []
    for _ in range(count):
        embedding = embedding_module(token)
        result = model(inputs_embeds=torch.cat((embedding, hidden), dim=-1), position_ids=positions,
                       attention_mask=attention_mask, shared_kv_states=shared_kv_states, use_cache=False)
        logits = result.logits.reshape(1, -1)
        # Public producer validates CUDA geometry and owns the existing greedy
        # packet algorithm. Exactly one known greedy row/group is launched.
        queued = launch_streams(logits, [0, 1], [[position]], [None])
        packet = queued.launched[0][3]
        token = packet[:, :1]
        hidden = result.last_hidden_state
        words.append(packet)
    return DeviceGreedyChain(torch.cat(words, dim=0))
