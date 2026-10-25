"""Actual target sampler distributions for the selected DFlash conditional chain.

The target's masked logits govern rejection. A retained packet keeps the actual
selector-conditioned q rows and proposal existence, including grammar-pruned
descendants whose target inputs are unreachable. Output-room cuts happen in the
standard verifier and never manufacture a different proposal distribution.
"""

from __future__ import annotations

from typing import NamedTuple

from tensorfold.engine.rejection import ChainPacket, ProbabilityRow


class VerifiedPrefix(NamedTuple):
    path: tuple[int, ...]
    terminal: int
    outputs: tuple[int, ...]


class BoundChain(NamedTuple):
    chain: ChainPacket
    slot_generation: int
    window_profile: str
    window_owner: object


def bind_packet(state, chain):
    """Bind an actual proposal to the current sequence and runtime-owned window."""
    if not isinstance(chain, ChainPacket) or chain.first_position != state.pos + 1:
        raise ValueError("DFlash2 proposal is not aligned with the current pending anchor")
    window = state.dflash2
    if window is None:
        raise ValueError("DFlash2 proposals require a live draft-window owner")
    window.check()
    if window.generation != state.slot_generation or window.end != state.pos:
        raise ValueError("DFlash2 proposal window differs from the current sequence generation")
    return BoundChain(chain, state.slot_generation, window.profile_identity, window)


def verify_batch(weights, logits, segments, packets, samplings, rooms, ends, *, plan=None):
    """Verify one frozen stream-major batch; callers apply stops before commit.

    ``segments`` carry target input rows and pre-commit State positions. ``rooms``
    count further outputs. Every rank uses the same packet/segment/sampling order.
    The caller has already applied grammar and supported history transforms to
    logits. A missing tail row is allowed only when the standard verifier proves
    that the corresponding descendant cannot be visited.
    """
    from tensorfold.cuda.sampling import effective_rows, effective_shards
    from .decode import tp_sample_rows
    from .dflash2_provider import DFlash2Provider

    size = len(segments)
    if any(len(items) != size for items in (packets, samplings, rooms, ends)):
        raise ValueError("DFlash2 verification metadata differs from the planned batch")
    if len({id(state) for state, _, _ in segments}) != size:
        raise ValueError("DFlash2 verification repeats a logical sequence")
    previous = 0
    for (state, begin, end), owned, room in zip(segments, packets, rooms):
        if (not isinstance(owned, BoundChain) or owned.slot_generation != state.slot_generation
                or owned.window_owner is not state.dflash2
                or owned.window_profile != state.dflash2.profile_identity
                or state.dflash2.generation != state.slot_generation):
            raise ValueError("DFlash2 proposal belongs to a retired sequence or runtime window")
        # Target forward has legitimately published pending accepted-prefix
        # features. They must remain unresolved until sampling/stops decide the
        # prefix; closed/poisoned windows still invalidate this live capability.
        state.dflash2.check(require_resolved=False)
        packet = owned.chain
        if (type(begin) is not int or type(end) is not int or begin != previous
                or end <= begin or end > logits.shape[0] or not isinstance(packet, ChainPacket)
                or packet.first_position != state.pos + 1 or end - begin > len(packet.tokens) + 1
                or len(packet.tokens) > 3
                or type(room) is not int or room < 1):
            raise ValueError("DFlash2 verification lost its frozen positions, rows or output allowance")
        previous = end
    if previous != logits.shape[0]:
        raise ValueError("DFlash2 verification logits contain unowned rows")
    if not size:
        return []
    if weights.comm is not None and plan is None:
        raise ValueError("TP DFlash2 verification requires the frozen global vocabulary shard owner")
    if plan is not None:
        plan.check_model(weights, logits)
    results = []
    for (state, begin, end), owned, sampling, room, eos in zip(segments, packets, samplings, rooms, ends):
        packet = owned.chain
        rows = logits[begin:end]
        if sampling is not None and sampling.temperature > 0:
            probability = (effective_rows(rows, sampling) if weights.comm is None else
                           effective_shards(rows, sampling, plan=plan))
        else:
            positions = list(range(state.pos + 1, state.pos + 1 + end - begin))
            if weights.comm is None:
                from tensorfold.cuda.sampling import sample_rows

                chosen = sample_rows(rows, positions, sampling)
            else:
                chosen = tp_sample_rows(weights, rows, positions, sampling, plan=plan)
            probability = [ProbabilityRow([token], [1.0]) for token in chosen]
        if len(probability) != end - begin:
            raise ValueError("target sampler returned a different DFlash2 verification row count")
        probability.extend([None] * (len(packet.tokens) + 1 - len(probability)))
        path, terminal = DFlash2Provider.verify(packet, probability, sampling, room=room, eos=eos)
        # Input root zero consumes the pending anchor. Accepted child inputs
        # correspond to preceding proposed outputs, followed by one new anchor.
        outputs = tuple(packet.tokens[row - 1] for row in path[1:]) + (int(terminal),)
        results.append(VerifiedPrefix(tuple(path), int(terminal), outputs))
    return results
