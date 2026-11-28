"""Private preparation of one DSpark verification owner on the measured profile."""

import torch

from . import dspark_weights as dw


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
