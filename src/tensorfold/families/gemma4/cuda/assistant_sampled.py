"""Unconnected opt-in stochastic native assistant proposal candidate.

The existing greedy/delta-q head remains the baseline. This candidate draws each
actual recurrent token from its recorded q, using established effective_rows and
PROPOSAL-domain randomness. The target alone owns p/grammar/history/rejection.
No training-calibration or performance claim follows from this implementation.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import ChainPacket, draw


def validate_proposal_parameters(temperature_factor=1.0, top_k=None, *, vocab=None):
    """Validate private configuration once, before loading/allocating on CUDA."""
    if type(temperature_factor) not in (int, float):
        raise ValueError("Assistant proposal temperature factor must be finite and positive")
    try:
        factor = float(temperature_factor)
    except OverflowError as error:
        raise ValueError("Assistant proposal temperature factor must be finite and positive") from error
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Assistant proposal temperature factor must be finite and positive")
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError("Assistant proposal top_k override must be a nonnegative integer or None")
    if vocab is not None:
        if type(vocab) is not int or vocab <= 0:
            raise ValueError("Assistant proposal vocabulary must be positive")
        if top_k is not None and top_k > vocab:
            raise ValueError("Assistant proposal top_k override exceeds its vocabulary")
    return factor, top_k


def make_proposal_sampling(target_sampling, temperature_factor=1.0, top_k=None, *, vocab):
    """Derive actual q policy while retaining the ORIGINAL target policy authority."""
    from tensorfold.cuda.sampling import validate_policy

    factor, top_k = validate_proposal_parameters(temperature_factor, top_k, vocab=vocab)
    if not isinstance(target_sampling, Sampling) or target_sampling.temperature <= 0:
        raise ValueError("Assistant proposal policy requires a positive-temperature target")
    validate_policy([], target_sampling)
    temperature = float(target_sampling.temperature) * factor
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Assistant proposal temperature product must be finite and positive")
    if factor == 1.0 and top_k is None:
        return target_sampling
    return replace(target_sampling, temperature=temperature,
                   top_k=target_sampling.top_k if top_k is None else top_k)


class RecordedSampledChain:
    """Immutable actual draws/laws: sample() transports, never draws again."""
    def __init__(self, packet, sampling, history, proposal_sampling=None):
        self.packet, self.sampling, self.history = packet, sampling, history
        self.proposal_sampling = sampling if proposal_sampling is None else proposal_sampling

    def sample(self, row, max_nodes, sampling, first_position, history=None):
        if row != 0 or history is not self.history or sampling != self.sampling:
            raise ValueError("Sampled Gemma assistant packet policy differs")
        if type(first_position) is not int or first_position != self.packet.first_position:
            raise ValueError("Sampled Gemma assistant packet position differs from actual draws")
        if type(max_nodes) is not int or max_nodes < 0:
            raise ValueError("Sampled Gemma assistant packet count must be nonnegative")
        return ChainPacket(self.packet.tokens[:max_nodes], self.packet.q_rows[:max_nodes], first_position)

    def finish(self, *args, **kwargs):
        raise ValueError("Sampled Gemma assistant proposal must use its recorded q packet")


@torch.inference_mode()
def sampled_assistant_chain(model, embedding_module, hidden, shared_kv_states, attention_mask,
                            anchor, count, position, sampling, history=None, *, enabled=False,
                            temperature_factor=1.0, proposal_top_k=None):
    """Draw q through the existing exact effective categorical sampler.

    Inputs are borrowed from the caller's exclusive committed target state.
    Caller validates context capacity and clips unreachable descendants before
    entry. Position/KV/mask remain frozen, while each ACTUAL drawn token and
    assistant projected hidden feed the next recurrent step. Histories are
    deliberately not mutated or applied to q; exact target p handles them.
    """
    from tensorfold.cuda.sampling import effective_rows

    if type(enabled) is not bool or not enabled:
        raise ValueError("Stochastic Gemma assistant q is an explicit opt-in candidate")
    if not isinstance(sampling, Sampling) or sampling.temperature <= 0:
        raise ValueError("Stochastic Gemma assistant q requires positive-temperature Sampling")
    if type(count) is not int or not 1 <= count <= 7:
        raise ValueError("Stochastic Gemma assistant permits one through seven proposals")
    vocab = model.config.get_text_config().vocab_size
    if type(anchor) is not int or not 0 <= anchor < vocab:
        raise ValueError("Stochastic Gemma assistant anchor is outside the vocabulary")
    if type(position) is not int or not 0 <= position <= min((1 << 63) - 1, (1 << 64) - count - 1):
        raise ValueError("Stochastic Gemma assistant draw positions exceed unsigned64")
    proposal_sampling = make_proposal_sampling(sampling, temperature_factor, proposal_top_k, vocab=vocab)
    positions = torch.tensor([[position]], device=hidden.device, dtype=torch.long)
    token, tokens, rows = anchor, [], []
    for index in range(count):
        ids = torch.tensor([[token]], device=hidden.device, dtype=torch.long)
        embedding = embedding_module(ids)
        result = model(inputs_embeds=torch.cat((embedding, hidden), dim=-1), position_ids=positions,
                       attention_mask=attention_mask, shared_kv_states=shared_kv_states, use_cache=False)
        # Includes the full native finite floor and existing canonical top-k /
        # nucleus selection. No selected-centroid-only support reinterpretation.
        q = effective_rows(result.logits.reshape(1, -1), proposal_sampling)[0]
        token = draw(q, proposal_sampling.seed, position + 1 + index)
        rows.append(q)
        tokens.append(token)
        hidden = result.last_hidden_state
    packet = ChainPacket(tuple(tokens), tuple(rows), position + 1)
    return RecordedSampledChain(packet, sampling, history, proposal_sampling)
