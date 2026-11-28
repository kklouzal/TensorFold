"""Private preparation of one DSpark verification owner on the measured profile."""

import torch

import math

from . import dspark_weights as dw


def sampled_draftable(sampling) -> bool:
    """A positive finite temperature: rejection keeps the target sampler's law (``decide_sampled``)."""
    return sampling is not None and math.isfinite(sampling.temperature) and sampling.temperature > 0


def prepare_128k_verifier(engine, weights: dw.DSparkWeights, *, bucket: int, depth: int):
    """Build a detached eager owner from preloaded weights for a stopped TP8 scheduler.

    The private caller owns draft residency, startup budgeting and any later
    graph capture. This function never installs the verifier on the engine.
    """
    target = engine.target
    if (engine.context != 131072 or engine.yarn_factor != 16 or target.rotor_cache is None or engine.graphs is None
            or engine._closed or engine.scheduler is not None or engine.part is not None or target.pool.owned
            or getattr(engine, "_draft_verify", None) is not None):
        raise ValueError("DSpark preparation requires an idle private 128K H64 target with decode graphs")
    if (type(bucket) is not int or not 1 <= bucket <= min(3, engine.streams)
            or type(depth) is not int or not 1 <= depth <= dw.BLOCK):
        raise ValueError("DSpark preparation requires an explicit supported bucket and depth")
    return prepare_verifier(engine, weights, (bucket,), depth)


def prepare_verifier(engine, weights: dw.DSparkWeights, buckets: tuple[int, ...], depth: int):
    """Construct the shared head and rollback owner before capture and admission."""
    from .dspark_graph import FixedVerifyOwner, SpecShape
    from .dspark_head import DSparkHead
    from .speculation import Speculation

    target = engine.target
    vocab_first = target.w.embed_first
    layout = dw.ft_layout(target.g, rank=engine.rank, world=engine.world, vocab_first=vocab_first)
    if (weights.layout != layout or len(target.w.layers) != target.g.layers
            or [stage.index for stage in weights.stages] != list(range(dw.STAGES))
            or weights.markov_head.shape != (dw.VOCAB // layout.world, dw.MARKOV_RANK)
            or weights.markov_embed.shape != (dw.VOCAB, dw.MARKOV_RANK)
            or any(tensor.device != target.device or tensor.dtype != torch.bfloat16
                   for tensor in (weights.markov_head, weights.markov_embed))):
        raise ValueError("DSpark weights must match the loaded 1017 target and this rank's head")
    head = DSparkHead(weights, slots=engine.slots, comm=engine.coll, device=target.device,
                     vocab_first=vocab_first, target_embed=target.embed, window_rope=target.window_rope)
    spec = Speculation(target)
    shapes = tuple(SpecShape(engine, head, spec, bucket, depth, engine.context, use_prepared_embedding=True)
                   for bucket in buckets)
    if len(shapes) == 1:
        return FixedVerifyOwner(shapes[0])
    return FixedVerifyOwner(shapes[-1], narrower=shapes[:-1])


def decide(choices, anchor, proposed, room, eos):
    from tensorfold.engine.draft_acceptance import accepted_prefix

    if len(choices) != len(proposed) + 1 or room <= 0:
        raise ValueError("invalid DSpark verification result geometry")

    def reachable():
        for token, error in choices:
            if error is not None:
                raise error
            yield token

    return accepted_prefix(reachable(), [anchor, *proposed], room, eos)


def decide_sampled(sampling, logits, anchor, proposed, position, room, eos, constraint=None):
    """Standard rejection over a deterministic (Markov argmax) proposal chain.

    Each proposal's actual q row is a point mass, so it survives with probability p(token) and
    a rejection draws from p without it: the emitted law is exactly the target sampler's
    (Leviathan et al. 2023). ``logits`` holds the target's full rows for the anchor and each
    proposal; ``position`` is the anchor's. A reply grammar first drops proposals no path can
    keep, then masks each retained row by its path; descendants it drops are unreachable.
    """
    from tensorfold.engine.rejection import ChainPacket, ProbabilityRow, verify

    from tensorfold.cuda.sampling import effective_rows

    if len(logits) != len(proposed) + 1 or room <= 0:
        raise ValueError("invalid DSpark verification result geometry")
    tokens = [anchor, *proposed]
    rows = logits
    if constraint is not None:
        window = constraint.window(tokens, list(range(-1, len(proposed))))
        tokens = window.tokens
        rows = constraint.mask(logits[:len(tokens)], window) if window.rows else logits[:len(tokens)]
    chain = ChainPacket(tuple(int(t) for t in tokens[1:]),
                        tuple(ProbabilityRow([int(t)], [1.0]) for t in tokens[1:]), position + 1)
    p_rows = effective_rows(rows, sampling)
    path, terminal = verify(chain, p_rows, sampling.seed, room=room, eos=eos)
    return [*chain.tokens[:len(path) - 1], terminal]


def decide_constrained(constraint, logits, choices, anchor, proposed, room, eos):
    """Read only reachable target choices in the reply grammar's current state."""
    from .round_sampling import compact_packet, greedy_tokens

    window = constraint.window([anchor, *proposed], list(range(-1, len(proposed))))
    # A linear proposal chain can only lose a suffix. The target has already
    # evaluated all its rows, but only the retained prefix can be selected.
    width = len(window.tokens)
    if window.rows:
        rows = constraint.mask(logits[:width], window)
        choices = greedy_tokens(compact_packet(rows, 0).cpu().unsqueeze(0), width)
    return decide(choices[:width], anchor, window.tokens[1:], room, eos)
