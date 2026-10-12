"""Which kept prompt prefixes Flash Next retains: cost-aware slots, a geometric ladder of points, in-place resumes.

Under the old rule (first in, first out over eight entries, and a victim slot chosen by its oldest entry) three short
side requests evicted a long main conversation, a fork copy without room fell back to a cold prefill of a prefix that
was fully kept, and a long tool loop pushed out every point before the user message the next run diverges at.
"""

import pytest

from tensorfold.families.qwen4_exp.cuda import prefixes


class Slot:
    def __init__(self, name):
        self.name, self.copied = name, None

    def __repr__(self):
        return self.name

    def copy_prefix(self, source, pos, mtp_len):
        self.copied = (source, pos)


class Owner:
    """The decoder surface prefixes.py reads: slots, kept entries and growth that fits ``room`` (or always)."""

    def __init__(self, slots=4, keep=8, room=True):
        self.free = [Slot(f"S{i}") for i in range(slots)]
        self.kept, self.keep, self.depth, self.used, self.clock = [], keep, 3, {}, 0
        self.busy, self.room = set(), room

    def _busy(self):
        return set(self.busy)

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st, rows, *, protect=None):
        return self.room

    def _shrink(self, st, force=False):
        pass


def serve(owner, prompt, points=()):
    """Admit ``prompt``, keep its points and its end (one token early), finish: (slot, cached tokens)."""

    st, _, cached = prefixes.slot_for(owner, prompt, True)
    owner.busy.add(id(st))
    for p in [*points, len(prompt) - 1]:
        if p > cached:
            prefixes.remember(owner, list(prompt[:p]), st, {"pos": p, "mtp_len": p - 1}, None)
    owner.busy.discard(id(st))
    if not any(k[1] is st for k in owner.kept) and all(f is not st for f in owner.free):
        owner.free.append(st)
    return st, cached


def tokens(n, seed):
    return [seed * 100_000 + i for i in range(n)]


def test_short_side_requests_do_not_evict_a_long_main_chain():
    owner = Owner()
    main = tokens(5_000, 1)
    serve(owner, main)
    for k in range(3):                                       # the other three slots
        serve(owner, tokens(200, 10 + k))
    main = main[:-1] + tokens(4_000, 2)
    st, cached = serve(owner, main)                          # used again: the most valuable chain
    assert cached == 4_999
    serve(owner, tokens(200, 20))                            # a fourth side request needs a slot: a side one goes
    assert serve(owner, main[:-1] + tokens(300, 3)) == (st, len(main) - 1)


def test_a_fork_without_room_for_its_copy_resumes_the_chain_in_place():
    owner = Owner(room=False)
    prompt = tokens(20_000, 1) + tokens(5, 2)
    st, _ = serve(owner, prompt, points=[20_000])
    retry = prompt[:20_000] + tokens(200, 3)                 # a longer sibling exists: a fork, but no room to copy
    assert serve(owner, retry) == (st, 20_000)


def test_a_chain_worth_more_than_the_prompt_reuses_is_kept_and_the_prompt_goes_elsewhere():
    owner = Owner(room=False)
    system = tokens(3_000, 1)
    main = system + tokens(60_000, 2)
    st, _ = serve(owner, main, points=[3_000])
    other, cached = serve(owner, system + tokens(500, 3))    # shares only the system block: no copy room
    assert other is not st and cached == 0
    assert any(k[1] is st and len(k[0]) == len(main) - 1 for k in owner.kept)
    lone = Owner(slots=1, room=False)                        # no other slot: resuming in place beats a cold prefill
    st, _ = serve(lone, main, points=[3_000])
    assert serve(lone, system + tokens(500, 3)) == (st, 3_000)


def test_a_chain_a_cold_prefill_would_evict_anyway_is_resumed_in_place():
    """No free slot, and the long chain is the idle one worth least: a cold prefill would evict it, so the prompt
    resumes its shared part in place instead, and the newer chain stays."""

    owner = Owner(slots=2, room=False)
    system = tokens(3_000, 1)
    old, _ = serve(owner, system + tokens(60_000, 2), points=[3_000])
    owner.clock = 100_000                                    # evictions since raised the clock
    new, _ = serve(owner, tokens(5_000, 9))                  # worth more than the old chain now
    assert serve(owner, system + tokens(500, 3)) == (old, 3_000)
    assert any(k[1] is new and len(k[0]) == 4_999 for k in owner.kept)


@pytest.mark.parametrize("turns, step", [(40, 2_500), (120, 2_500), (300, 1_500), (60, 9_000)])
def test_past_keep_a_chain_keeps_its_end_its_root_and_a_ladder_a_divergence_resumes_near(turns, step):
    """A long tool loop on one chain: past ``keep`` points it keeps its end, its system block and about one point a
    doubling of distance from its end, so a prompt that rewrites any earlier turn (as a client that rewrites its last
    user message each run does) resumes within a few times its distance from the end, never cold."""

    owner, system = Owner(slots=1, keep=8), 20_000
    convo, starts = tokens(system, 1), []
    for turn in range(turns):
        starts.append(len(convo))
        convo = convo + tokens(step, 10 + turn)
        serve(owner, convo, points=[system])
    end = len(convo) - 1
    lengths = sorted(len(k[0]) for k in owner.kept)
    assert len(lengths) == 8 and lengths[0] == system and lengths[-1] == end
    for diverged in starts[1:]:
        resume = max(n for n in lengths if n <= diverged)
        assert diverged - resume <= 4 * (end - diverged), (diverged, resume)


def test_evict_frees_the_least_valuable_idle_chain_and_raises_the_clock():
    owner = Owner()
    big, small = serve(owner, tokens(9_000, 1))[0], serve(owner, tokens(300, 2))[0]
    owner.busy.add(id(big))
    assert prefixes.evict(owner, None) is small and owner.clock == 299
    assert prefixes.evict(owner, None) is None               # the big chain is busy
    owner.busy.clear()
    assert prefixes.evict(owner, big) is None                # and never the slot being grown
    assert prefixes.evict(owner, None) is big and owner.kept == []


def test_no_free_slot_resumes_in_place_when_going_cold_would_lose_more():
    """No spare slot: going cold would prefill the shared tokens again and evict another whole chain, so a prompt that
    discards fewer kept tokens than that resumes in place (the concurrent decoder's fork test on the GPU)."""
    owner = Owner(slots=2, room=False)
    shared = tokens(600, 1)
    one, nxt = shared + tokens(10, 2), None
    a, _ = serve(owner, one, points=[305])
    nxt = one[:-1] + tokens(8, 3)
    assert serve(owner, nxt) == (a, len(one) - 1)
    b, _ = serve(owner, shared + tokens(10, 4))              # a fork: the other slot, cold (no room to copy)
    assert b is not a
    other, cached = serve(owner, shared + tokens(10, 5))
    assert (other, cached) == (a, 305)                       # discards 311 of a's tokens, rather than b's 609 + 305
