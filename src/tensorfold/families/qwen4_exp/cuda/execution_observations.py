"""Bounded host observations owned by one serialized adaptive decoder.

Counts describe submitted logical rows and causal indexer geometry. They are
neither device traffic/completion measurements nor timing estimates. A failed
forward can submit only part of a model; that work stays explicitly unknown.
The worker publishes an immutable snapshot at an operation boundary and a
monotonic failure latch. HTTP readers never inspect mutable counters or CUDA
objects; a failed publication invalidates completeness even if it cannot allocate
another snapshot.
"""

from __future__ import annotations

from tensorfold.cleanup import raise_failures


_COUNTERS = (
    "request_admissions", "logical_prompt_tokens", "replay_admissions",
    "request_prefix_hit_tokens", "replay_prefix_hit_tokens",
    "target_prefill_rows_submitted", "target_decode_rows_submitted", "target_replay_rows_submitted",
    "mtp_prefill_rows_submitted", "mtp_refresh_rows_submitted", "mtp_proposal_rows_submitted",
    "mtp_zero_proposal_refresh_rows", "indexer_query_rows_submitted",
    "dflash_prefill_conditioning_rows_submitted", "dflash_decode_conditioning_rows_submitted",
    "dflash_fixed_block_rows_submitted", "dflash_proposals_drawn", "dflash_zero_proposal_conditioning_rows",
    "indexer_visible_pooled_groups", "indexer_visible_sparse_group_scores",
    "attention_selected_key_rows", "incomplete_forward_submissions", "indexer_unknown_rows",
    "verification_windows", "verified_candidates", "accepted_candidates_before_text_limits",
)
_LIMIT = (1 << 63) - 1


def _floors(end, ratio):
    if end < 0:
        return 0
    complete, tail = divmod(end, ratio)
    return ratio * complete * (complete - 1) // 2 + complete * (tail + 1)


def _remainders(end, ratio):
    if end < 0:
        return 0
    complete, tail = divmod(end, ratio)
    return complete * ratio * (ratio - 1) // 2 + tail * (tail + 1) // 2


def indexer_counts(position, rows, ratio, budget):
    """Causal geometry for query ends position+1..position+rows, in O(1)."""
    if (any(type(value) is not int for value in (position, rows, ratio, budget))
            or position < 0 or rows < 0 or ratio <= 0 or budget < ratio):
        raise ValueError("indexer observation needs nonnegative rows and a complete-group budget")
    first, end = position + 1, position + rows
    threshold = (budget // ratio + 1) * ratio
    sparse_first = max(first, threshold)
    dense_end = min(end, threshold - 1)
    dense = max(0, dense_end - first + 1)
    sparse = max(0, end - sparse_first + 1)
    visible = _floors(end, ratio) - _floors(position, ratio)
    scored = _floors(end, ratio) - _floors(sparse_first - 1, ratio) if sparse else 0
    selected = dense * (first + dense_end) // 2
    if sparse:
        selected += sparse * (budget // ratio * ratio)
        selected += _remainders(end, ratio) - _remainders(sparse_first - 1, ratio)
    return visible, scored, selected


class ExecutionObservations:
    """One worker writes; readers borrow only a separately published snapshot."""

    def __init__(self, proposal_maximum):
        if type(proposal_maximum) is not int or proposal_maximum < 0:
            raise ValueError("observation proposals need the allocated nonnegative maximum")
        self.proposal_maximum = proposal_maximum
        self.geometry = None
        self.reset()

    def configure_indexer(self, *, ratio, budget, heads, target_caches):
        if (any(type(value) is not int for value in (ratio, budget, heads, target_caches))
                or ratio <= 0 or budget < ratio or heads <= 0 or target_caches < 0):
            raise ValueError("invalid observation indexer geometry")
        geometry = (ratio, budget, heads, target_caches)
        if self.geometry is not None and self.geometry != geometry:
            raise ValueError("observation geometry requires a retired profile before replacement")
        self.geometry = geometry
        self.publish()

    def reset(self):
        self.totals = dict.fromkeys(_COUNTERS, 0)
        self.windows_by_offered = [0] * (self.proposal_maximum + 1)
        self.accepted_by_offered = [0] * (self.proposal_maximum + 1)
        self.accepted_histogram = [0] * (self.proposal_maximum + 1)
        self.overflowed = False
        self.failed_observation = False
        self.publish()

    def _add(self, name, amount):
        value = self.totals[name] + amount
        if value > _LIMIT:
            self.overflowed = True
            return
        self.totals[name] = value

    def admission(self, stream, *, prefix_mtp_position=None, zero_proposals=False):
        # Publication allocates. A failed diagnostic must neither claim an
        # admission nor leave a partially updated accumulation plane behind.
        previous = self.totals.copy()
        published, overflowed = self._published, self.overflowed
        try:
            replay = stream.carry is not None
            self._add("replay_admissions" if replay else "request_admissions", 1)
            if not replay:
                self._add("logical_prompt_tokens", len(stream.prompt))
            self._add("replay_prefix_hit_tokens" if replay else "request_prefix_hit_tokens", stream.cached)
            if prefix_mtp_position is not None:
                self.mtp([(stream.st, 0, 1)], role="refresh", position=prefix_mtp_position,
                         zero_proposal_rows=int(zero_proposals))
            self.publish()
        except BaseException:
            self.totals, self._published, self.overflowed = previous, published, overflowed
            self.failed_observation = True
            raise

    def indexer(self, segments, *, mtp=False, position=None):
        if position is not None and len(segments) != 1:
            raise ValueError("an observed completed span needs exactly one sequence")
        rows = sum(last - first for _state, first, last in segments)
        if self.geometry is None:
            self._add("indexer_unknown_rows", rows)
            return
        ratio, budget, heads, target_caches = self.geometry
        caches = 1 if mtp else target_caches
        self._add("indexer_query_rows_submitted", rows * caches)
        visible = scored = selected = 0
        for state, first, last in segments:
            before = position if position is not None else state.mtp_len if mtp else state.pos
            counts = indexer_counts(before, last - first, ratio, budget)
            visible += counts[0]
            scored += counts[1]
            selected += counts[2]
        self._add("indexer_visible_pooled_groups", visible * caches)
        self._add("indexer_visible_sparse_group_scores", scored * heads * caches)
        self._add("attention_selected_key_rows", selected * caches)

    def target(self, segments, *, prefill=False, replay_rows=0):
        self._add("target_prefill_rows_submitted" if prefill else "target_decode_rows_submitted",
                  sum(last - first for _state, first, last in segments))
        self._add("target_replay_rows_submitted", replay_rows)
        self.indexer(segments)

    def mtp(self, segments, *, role, zero_proposal_rows=0, position=None):
        if role not in ("prefill", "refresh", "proposal"):
            raise ValueError("unknown MTP observation role")
        self._add("mtp_" + role + "_rows_submitted", sum(last - first for _state, first, last in segments))
        self._add("mtp_zero_proposal_refresh_rows", zero_proposal_rows)
        self.indexer(segments, mtp=True, position=position)

    def incomplete(self, primary):
        try:
            self._add("incomplete_forward_submissions", 1)
        except BaseException as observation:
            raise_failures(primary, [observation])

    def dflash_proposals(self, counts):
        self._add("dflash_fixed_block_rows_submitted", 4 * sum(count > 0 for count in counts))
        self._add("dflash_proposals_drawn", sum(counts))

    def dflash_conditioning(self, rows, *, prefill, zero_proposal_rows=0):
        self._add("dflash_prefill_conditioning_rows_submitted" if prefill else
                  "dflash_decode_conditioning_rows_submitted", rows)
        self._add("dflash_zero_proposal_conditioning_rows", zero_proposal_rows)

    def verification(self, offered, accepted):
        if (type(offered) is not int or type(accepted) is not int
                or not 0 <= accepted <= offered <= self.proposal_maximum):
            raise ValueError("acceptance observation exceeds its allocated proposal contract")
        for values, index, amount in ((self.windows_by_offered, offered, 1),
                                      (self.accepted_by_offered, offered, accepted),
                                      (self.accepted_histogram, accepted, 1)):
            if values[index] + amount > _LIMIT:
                self.overflowed = True
            else:
                values[index] += amount
        self._add("verification_windows", 1)
        self._add("verified_candidates", offered)
        self._add("accepted_candidates_before_text_limits", accepted)

    def publish(self):
        self._published = (tuple(self.totals.items()), tuple(self.windows_by_offered),
                           tuple(self.accepted_by_offered), tuple(self.accepted_histogram), self.overflowed,
                           self.geometry)

    def record(self):
        totals, offered, accepted, histogram, overflowed, geometry = self._published
        return {"scope": "logical_successful_forward_submissions_not_device_traffic_or_completion",
                "indexer_geometry": None if geometry is None else
                dict(zip(("ratio", "budget", "heads", "target_caches"), geometry)),
                "totals": dict(totals), "windows_by_offered_candidates": list(offered),
                "accepted_candidates_by_offered_length": list(accepted),
                "accepted_prefix_histogram_before_text_limits": list(histogram),
                "expert_group_max": None, "counter_overflow": overflowed,
                "observation_failure": self.failed_observation,
                "acceptance_scope": "verifier_prefix_after_reply_and_eos_before_text_stop_cut",
                "incomplete_submission_contract": "partial_failed_forward_work_is_unknown_not_estimated",
                "counts_complete": not self.failed_observation and not overflowed
                and not dict(totals)["incomplete_forward_submissions"]
                and not dict(totals)["indexer_unknown_rows"]}
