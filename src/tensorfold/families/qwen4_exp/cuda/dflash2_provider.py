"""Shared-target DFlash2 backbone and conditional-selector proposal execution.

No full draft vocabulary is gathered. Each target head keeps its actual shard;
only local top-16 scores/IDs are exchanged. Proposal q is conditioned on the
real selected predecessor and retained verbatim for standard rejection sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import ChainPacket, ProbabilityRow, PROPOSAL, draw, proposal_row, verify

from . import dflash2_ops as ops


@dataclass(frozen=True)
class TargetBinding:
    """Prepared target sharing/communication facts owned through loading."""
    target: object
    config: object
    tp: tuple
    local: dict
    comm: object
    embedding: object
    head: object


def bind_target(config, target, tp):
    config.check_target(target)
    rank, world = tp
    local = config.partition(rank, world)
    comm = target.comm
    if (world == 1) != (comm is None) or comm is not None and (comm.world, comm.rank) != (world, rank):
        raise ValueError("DFlash2 weight/target communicator placement differs")
    expected_rows = config.vocab // world
    actual_rows = getattr(target.head, "n", None)
    if actual_rows is None and isinstance(getattr(target.head, "weight", None), torch.Tensor):
        actual_rows = target.head.weight.shape[0]
    if actual_rows is not None and actual_rows != expected_rows:
        raise ValueError("DFlash2 must share the actual complete rank-local target head shard")
    if target.meta.get("vocab_offset", rank * expected_rows) != rank * expected_rows:
        raise ValueError("DFlash2 target head needs its declared contiguous global token-ID shard")
    return TargetBinding(target, config, tp, local, comm, target.embed, target.head)


def selector_lattice(candidate_ids, unary_logits, projected, predecessor, successor, anchors):
    """FP32 transition scores for all candidates; codebooks stay on device."""
    batch, slots, top = candidate_ids.shape
    before = torch.cat((anchors[:, None, None].expand(batch, 1, top), candidate_ids[:, :-1]), 1)
    predecessors = predecessor[before].float()
    successors = successor[candidate_ids].float()
    return unary_logits.float().unsqueeze(2) + torch.einsum(
        "bspr,bscr->bspc", predecessors * projected.float().unsqueeze(2), successors)


def selector_chain(candidate_ids, lattice, *, sampling, first_position):
    """Walk true predecessor-conditioned sparse q; immutable rows feed rejection."""
    ids = np.asarray(candidate_ids, dtype=np.int64)
    scores = np.asarray(lattice, dtype=np.float64)
    if ids.ndim != 2 or scores.shape != (len(ids), ids.shape[1], ids.shape[1]):
        raise ValueError("DFlash2 selector lattice does not match its candidate paths")
    tokens, rows, previous_index = [], [], 0
    for step in range(len(ids)):
        values = scores[step, previous_index]
        if not np.isfinite(values).all():
            raise ValueError("DFlash2 selector has nonfinite conditional scores")
        if sampling is None or sampling.temperature <= 0:
            order = np.lexsort((ids[step], -values))
            token = int(ids[step, order[0]])
            q = ProbabilityRow([token], [1.0])
        else:
            # Selector q normalizes its top16 support with temperature. Target
            # top-k/top-p/penalties/constraints belong to p. This sparse q need
            # not imitate them for exact standard rejection sampling.
            law = Sampling(sampling.seed, sampling.temperature, top_k=0, top_p=1.0, min_p=0.0)
            q = proposal_row(values, ids[step], law)
            token = draw(q, sampling.seed, first_position + step, domain=PROPOSAL)
        rows.append(q)
        tokens.append(token)
        previous_index = int(np.flatnonzero(ids[step] == token)[0])
    return ChainPacket(tuple(tokens), tuple(rows), first_position)


class DFlash2Provider:
    def __init__(self, weights, target, *, embedding=None, head=None, _binding=None):
        self.weights, self.target = weights, target
        self.config, self.tp = weights.config, weights.tp
        binding = bind_target(self.config, target, self.tp) if _binding is None else _binding
        if binding.target is not target or binding.config != self.config or binding.tp != self.tp:
            raise ValueError("DFlash2 prepared target binding belongs to another profile")
        self.local, self.comm = binding.local, binding.comm
        self.device = next(iter(weights.tensors.values())).device
        # Store references to the selected target objects, never decoded copies.
        self.target_embedding, self.target_head = binding.embedding, binding.head
        self.embedding = embedding
        self.head = head

    def _embedding(self, ids):
        if self.embedding is not None:
            return self.embedding(ids)
        face = self.target_embedding
        if len(face) != 1 or not hasattr(face[0], "lookup"):
            raise ValueError("DFlash2 needs the target's shared embedding lookup adapter")
        output = torch.empty((len(ids), self.config.hidden), dtype=torch.float16, device=self.device)
        return face[0].lookup(ids, copies=1, out=output)

    def _head(self, hidden):
        if self.head is not None:
            return self.head(hidden)
        return self.target_head.matmul(hidden, f32=True)

    def materialize(self, features, positions):
        c, w = self.config, self.weights.tensors
        projected = ops.norm(ops.linear(features, w["fc.weight"]), w["hidden_norm.weight"], c.eps)
        result = torch.empty((2, c.layers, len(features), self.local["kv_heads"], c.head_dim),
                             dtype=torch.float16, device=self.device)
        for layer in range(c.layers):
            prefix = f"layers.{layer}.self_attn."
            keys = ops.linear(projected, w[prefix + "k_proj.weight"]).reshape(len(features), self.local["kv_heads"], c.head_dim)
            result[0, layer].copy_(ops.rope(ops.norm(keys, w[prefix + "k_norm.weight"], c.eps), positions, c.theta))
            result[1, layer].copy_(ops.linear(projected, w[prefix + "v_proj.weight"]).reshape_as(keys))
        return result

    def backbone(self, states, anchors):
        c, weights = self.config, self.weights.tensors
        tokens = [token for anchor in anchors for token in (anchor, *([c.mask] * (c.block - 1)))]
        ids = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        positions = torch.tensor([position for state in states for position in range(state.end, state.end + c.block)],
                                 dtype=torch.int64, device=self.device)
        hidden = self._embedding(ids)
        for layer in range(c.layers):
            p = f"layers.{layer}."
            residual = hidden
            normalized = ops.norm(hidden, weights[p + "input_layernorm.weight"], c.eps)
            conv = p + "attention_conv."
            value, coeff = ops.conv_prepare(normalized, weights[conv + "kernel_projection.weight"], weights[conv + "base_kernel"], c)
            a = p + "self_attn."
            q = ops.linear(value, weights[a + "q_proj.weight"]).reshape(len(value), self.local["heads"], c.head_dim)
            k = ops.linear(value, weights[a + "k_proj.weight"]).reshape(len(value), self.local["kv_heads"], c.head_dim)
            v = ops.linear(value, weights[a + "v_proj.weight"]).reshape_as(k)
            q = ops.rope(ops.norm(q, weights[a + "q_norm.weight"], c.eps), positions, c.theta)
            k = ops.rope(ops.norm(k, weights[a + "k_norm.weight"], c.eps), positions, c.theta)
            pieces = []
            for index, state in enumerate(states):
                lo, hi = index * c.block, (index + 1) * c.block
                past, past_positions = state.layer(layer)
                pieces.append(ops.attention(q[lo:hi], k[lo:hi], v[lo:hi], past[0], past[1],
                                            positions[lo:hi], past_positions, window=c.window))
            attention = torch.cat(pieces)
            attention = ops.tp_sum(ops.linear(attention, weights[a + "o_proj.weight"], fp32=True), self.comm)
            attention = ops.grouped_conv(attention, coeff, weights[conv + "base_kernel"], side=1,
                                         block=c.block, group_size=c.group_size)
            hidden = (residual.float() + attention.float()).half()
            residual = hidden
            normalized = ops.norm(hidden, weights[p + "post_attention_layernorm.weight"], c.eps)
            conv = p + "mlp_conv."
            value, coeff = ops.conv_prepare(normalized, weights[conv + "kernel_projection.weight"], weights[conv + "base_kernel"], c)
            value = ops.mlp(value, weights[p + "mlp.gate_proj.weight"], weights[p + "mlp.up_proj.weight"],
                            weights[p + "mlp.down_proj.weight"], self.comm)
            value = ops.grouped_conv(value, coeff, weights[conv + "base_kernel"], side=1, block=c.block, group_size=c.group_size)
            hidden = (residual.float() + value.float()).half()
        return ops.norm(hidden, weights["norm.weight"], c.eps).reshape(len(states), c.block, c.hidden)

    def candidates(self, hidden):
        logits = self._head(hidden).float()
        c = self.config
        rank, world = self.tp
        offset = int(self.target.meta.get("vocab_offset", rank * (c.vocab // world)))
        valid = min(logits.shape[1], c.vocab - offset)
        logits = logits[:, :valid]
        # Raw projection top-k has global-ID ascending boundary ties. The pinned
        # compute_candidates selects this support BEFORE unary transforms; a
        # saturating softcap does not reselect from transformed-score ties.
        values, columns = torch.sort(logits, dim=-1, descending=True, stable=True)
        values, ids = values[:, :c.top_k].contiguous(), columns[:, :c.top_k].to(torch.int64).add(offset).contiguous()
        if self.comm is not None:
            all_values = torch.empty((world, *values.shape), dtype=torch.float32, device=self.device)
            all_ids = torch.empty((world, *ids.shape), dtype=torch.int64, device=self.device)
            self.comm.all_gather(values, all_values.flatten())
            self.comm.all_gather(ids, all_ids.flatten())
            merged_values = all_values.transpose(0, 1).flatten(1)
            merged_ids = all_ids.transpose(0, 1).flatten(1)
            # Rank shards are disjoint and in global ID order; ties within each
            # local sorted list are already canonical.
            order = torch.argsort(merged_values, dim=-1, descending=True, stable=True)[:, :c.top_k]
            values, ids = merged_values.gather(1, order), merged_ids.gather(1, order)
        values = values * c.output_multiplier
        if c.logit_softcap is not None:
            values = torch.tanh(values / c.logit_softcap) * c.logit_softcap
        return ids, values

    def propose_many(self, states, anchors, proposal_counts, samplings, first_positions):
        c = self.config
        if not (len(states) == len(anchors) == len(proposal_counts) == len(samplings) == len(first_positions)):
            raise ValueError("DFlash2 batch state/anchor/proposal metadata lengths differ")
        if not states:
            return []
        for state, anchor, count, position in zip(states, anchors, proposal_counts, first_positions):
            state.check()
            if type(count) is not int or not 1 <= count < c.block:
                raise ValueError("DFlash2 chooses one/two/three proposals before a complete block-four draw")
            if (state.end + c.block > c.context or type(anchor) is not int or not 0 <= anchor < c.vocab
                    or position != state.end + 1):
                raise ValueError("DFlash2 anchor/frontier/fixed block exceeds its supported absolute positions")
        hidden = self.backbone(states, anchors)[:, 1:].reshape(-1, c.hidden)
        ids, values = self.candidates(hidden)
        projected = ops.linear(hidden, self.weights.tensors["candidate_selector.hidden_projection.weight"])
        ids = ids.reshape(len(states), c.block - 1, c.top_k)
        values = values.reshape(len(states), c.block - 1, c.top_k)
        projected = projected.reshape(len(states), c.block - 1, c.selector_rank)
        lattice = selector_lattice(ids, values, projected,
            self.weights.tensors["candidate_selector.predecessor_codebook"],
            self.weights.tensors["candidate_selector.successor_codebook"],
            torch.tensor(anchors, dtype=torch.int64, device=self.device))
        # One compact transfer for IDs and one for conditional transition scores;
        # never one codebook readback per sequence/proposal.
        host_ids, host_lattice = ids.cpu().numpy(), lattice.cpu().numpy()
        packets = []
        for index, (count, sampling, position) in enumerate(zip(proposal_counts, samplings, first_positions)):
            packets.append(selector_chain(host_ids[index, :count], host_lattice[index, :count],
                                           sampling=sampling, first_position=position))
        return packets

    def propose(self, state, anchor, proposals, sampling, first_position):
        return self.propose_many([state], [anchor], [proposals], [sampling], [first_position])[0]

    @staticmethod
    def verify(packet, target_probabilities, sampling, *, room, eos=()):
        if sampling is None or sampling.temperature <= 0:
            if any(row is not None and len(row.ids) != 1 for row in target_probabilities):
                raise ValueError("greedy DFlash2 verification requires actual target-argmax point masses")
        seed = 0 if sampling is None else sampling.seed
        return verify(packet, target_probabilities, seed, room=room, eos=eos)
