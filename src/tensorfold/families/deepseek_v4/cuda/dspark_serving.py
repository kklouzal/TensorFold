"""Private preparation of one DSpark verification owner on the measured profile."""

import torch

from . import dspark_weights as dw


def prepare_128k_verifier(engine, weights: dw.DSparkWeights, *, bucket: int, depth: int):
    """Build a detached eager owner from preloaded weights for a stopped TP8 scheduler.

    The private caller owns draft residency, startup budgeting and any later
    graph capture. This function never installs the verifier on the engine.
    """
    from .dspark_graph import FixedVerifyOwner, SpecShape
    from .dspark_head import DSparkHead
    from .speculation import Speculation

    target = engine.target
    if (engine.context != 131072 or engine.yarn_factor != 16 or target.rotor_cache is None or engine.graphs is None
            or engine._closed or engine.scheduler is not None or engine.part is not None or target.pool.owned
            or getattr(engine, "_draft_verify", None) is not None):
        raise ValueError("DSpark preparation requires an idle private 128K H64 target with decode graphs")
    if (type(bucket) is not int or not 1 <= bucket <= min(3, engine.streams)
            or type(depth) is not int or not 1 <= depth <= dw.BLOCK):
        raise ValueError("DSpark preparation requires an explicit supported bucket and depth")
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
    shape = SpecShape(engine, head, spec, bucket, depth, engine.context, use_prepared_embedding=True)
    return FixedVerifyOwner(shape)
