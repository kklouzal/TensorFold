"""Private exact-row local candidate graphs over prepaid external packets.

One serialized actor owns both roles. All graph-private intermediates are
temporary; outputs are the original external candidate packets. Graphs never
consume another graph's private outputs or run concurrently, permitting the
independent-inference pool contract documented by PyTorch2.14.
"""
import os
import threading

import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sm70.router_blas import PROFILE

from .candidate_region_graphs import CandidateRegion, _binding
from .capture_lifetime import host_memory

ENV = 'TENSORFOLD_SM70_CANDIDATE_REGION_GRAPHS'
ROWS = tuple(range(1, 33))
HOST_ALLOWANCE = 1 << 30
MAX_COUNT = (1 << 63) - 1


def requested(profile):
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError('private candidate graph choice must be0 or1')
    if value == '1' and profile != PROFILE:
        raise ValueError('private candidate graphs require the admitted batched-router profile')
    return value == '1'


def funding():
    return {'rows': ROWS, 'roles': ('target', 'mtp'), 'maximum_graphs': 64,
            'future_host_bytes_per_rank': HOST_ALLOWANCE,
            'device_charge': 'one candidate-only pool charged ONCE through RoundGraphs physical/settled budget',
            'host_charge': '1GiB/rank operating allowance includes graph/runtime metadata and construction temporaries',
            'operands': 'original FP32 head outputs, immutable ID maps and candidate packets; no extra weight or matrix copy'}


class Catalogue:
    def __init__(self, parent):
        self.parent, self.dec = parent, parent.dec
        self.weights = self.dec.w
        self.regions, self.pools = {}, {}
        self.ready = self.closed = self.retiring = False
        self.failure = None
        self.lock = threading.Lock()
        self.notification, self.notified = [None], False
        self.roles = {}
        self.frame_counts = {role: {rows: 0 for rows in ROWS} for role in ('target', 'mtp')}
        self.counter_overflowed = False
        self.plan = funding()
        self.host_before = self.host_after = None
        self.host_retained_growth = self.host_peak_growth = 0

    def _policy(self):
        w = self.weights
        return (w.meta.get('sm70_profile'), w.meta.get('execution_provider'),
                w.meta.get('execution_dtype'), w.meta.get('history_layout'),
                w.meta.get('unique_capture_contract'), w.meta.get('sm70_candidate_region_graphs_trial'),
                w.meta.get('world'), w.meta.get('vocab_offset'), w.comm,
                w.head, w.draft_head, w.draft_ids, w.draft_filter,
                torch.are_deterministic_algorithms_enabled(), torch.is_autocast_enabled('cuda'))

    def _check_policy(self):
        current = self._policy()
        # Tensor equality would read device data; identities own this boundary.
        if any(now is not old if index in (8, 9, 10, 11, 12) else now != old
               for index, (now, old) in enumerate(zip(current, self.policy))):
            raise RuntimeError('candidate graph profile or original model binding changed')

    def _host_gate(self):
        now = host_memory()
        self.host_after = now
        self.host_retained_growth = max(self.host_retained_growth, now['VmRSS'] - self.host_before['VmRSS'])
        self.host_peak_growth = max(self.host_peak_growth, now['VmHWM'] - self.host_before['VmHWM'])
        if max(self.host_retained_growth, self.host_peak_growth) > HOST_ALLOWANCE:
            raise MemoryError('candidate graph construction exceeded its admitted host operating allowance')

    @torch.no_grad()
    def prepare(self):
        from tensorfold.cuda.capacity import admit_host_commitment

        if self.ready or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError('candidate catalogue is one-time cold preparation')
        w, dec = self.weights, self.dec
        if (dec.streams or dec.filling or dec.kept or len(dec.free) != dec.round_tables.slots
                or len(dec.free) < 24 or self.parent.allocator_backend != 'native'):
            raise RuntimeError('candidate preparation requires quiescent24-slot native-allocator ownership')
        if (w.meta.get('sm70_profile') != PROFILE or w.meta.get('execution_provider') != 'sm70'
                or w.meta.get('execution_dtype') != 'float16' or w.meta.get('history_layout') != 'unique'
                or w.meta.get('sm70_candidate_region_graphs_trial') is not True
                or w.comm is None or w.comm.world != 8 or w.mtp is None or dec.mbuf is None
                or getattr(w, 'dspark', None) is not None or getattr(w, 'dflash2', None) is not None):
            raise ValueError('candidate catalogue requires its explicit original TP8 stock-MTP profile')
        self.policy = self._policy()
        self.offset = int(w.meta['vocab_offset'])
        self.host_before = host_memory()
        try:
            self.host_preflight = admit_host_commitment(8 * HOST_ALLOWANCE)
            for role, buffers, width, mapping in (
                    ('target', dec.buf, w.head.n, None),
                    ('mtp', dec.mbuf, w.head.n if w.draft_head is None else w.draft_head.n, w.draft_ids)):
                if buffers.prefill or buffers.rows != 32 or buffers.input_preparation_context is None:
                    raise ValueError('candidate catalogue requires canonical row32 decode buffer roles')
                self.roles[role] = buffers, width, mapping
            torch.cuda.synchronize(w.device)  # Prior warmup/head/collective consumers positively retired.
            # Every cold capture row is valid and fully initialized. Actual
            # forwards overwrite the complete exact-R prefix before replay.
            for buffers, _, _ in self.roles.values():
                buffers.logits.zero_()
            self.pools['candidate-finish'] = torch.cuda.graph_pool_handle()
            for role, (buffers, width, mapping) in self.roles.items():
                for rows in ROWS:
                    before = self.parent.capture_observation()
                    region = object.__new__(CandidateRegion)
                    self.regions[role, rows] = region  # Retain partial acquisition before any submission.
                    logits = buffers.logits.view(-1)[:rows * width].view(rows, width)
                    region.__init__(w, buffers, logits, rows, id_map=mapping, offset=self.offset,
                                    side=self.parent.side, pool=self.pools['candidate-finish'])
                    region.prepare()
                    region.replay()  # Materialize runtime resources cold; no State/sampling/RNG effects.
                    self.parent.charge_capture(before, ['candidate', role, rows])
                    self._host_gate()
            self._check_policy()
            self.host_final = admit_host_commitment(8 * HOST_ALLOWANCE)
            self.ready = True
        except BaseException as error:
            self.failure, self.retiring = error, True
            raise

    def _poison(self, error):
        if self.failure is None:
            self.failure = error
        self.retiring = True
        self.dec._memory_error = self.failure

    def _notify(self, buffers, error):
        if not self.notified:
            self.notified = True
            try:
                buffers.input_preparation_context.fail(error)
            except BaseException as failed:
                self.notification[0] = failed
        if self.notification[0] is not None:
            raise_failures(error, self.notification)
        raise error

    def local_packet(self, weights, buffers, logits, rows, id_map, offset):
        role = next((name for name, (original, _, _) in self.roles.items() if buffers is original), None)
        if role is None or rows not in ROWS:
            return False  # Prefill/other Buffers and unsupported rows retain original local math.
        if not self.ready or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError('candidate graph catalogue is not reusable') from self.failure
        if torch.cuda.is_current_stream_capturing():
            return False  # Preserve the original math in existing N1/other enclosing captures.
        if not self.parent.admitted:
            raise RuntimeError('candidate graphs have not passed aggregate physical admission')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('candidate graph owner already has a packet borrower')
        failure = None
        try:
            self._check_policy()
            region = self.regions[role, rows]
            if weights is not self.weights or id_map is not region.id_map or offset != self.offset:
                raise ValueError('candidate packet lost its original role or vocabulary map')
            if _binding(logits) != _binding(region.logits):
                raise ValueError('candidate packet input differs from its exact original head prefix')
            region.replay()  # Same original stream; lock release does not promise GPU completion.
            if self.frame_counts[role][rows] < MAX_COUNT:
                self.frame_counts[role][rows] += 1
            else:
                self.counter_overflowed = True
            return True
        except BaseException as error:
            failure = error
            self._poison(error)
        finally:
            self.lock.release()
        self._notify(buffers, failure)

    def close(self):
        if self.closed:
            return
        self.retiring = True
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('candidate graph packet borrower has not retired')
        try:
            torch.cuda.synchronize(self.weights.device)
            for region in self.regions.values():
                if getattr(region, 'graph', None) is not None:
                    region._release_fenced()
            self.regions.clear()
            self.pools.clear()
            self.roles.clear()
            self.closed = True
        except BaseException as error:
            self._poison(error)
            raise
        finally:
            self.lock.release()

    def record(self):
        return {'enabled': True, 'rows': ROWS, 'roles': tuple(self.roles), 'graphs': len(self.regions),
                'frames': None if self.counter_overflowed else {role: dict(counts) for role, counts in self.frame_counts.items()},
                'funding': self.plan, 'ready': self.ready, 'closed': self.closed,
                'host_retained_growth_bytes_observed': self.host_retained_growth,
                'host_peak_growth_bytes_observed': self.host_peak_growth,
                'host_observation': 'boundary RSS/HWM deltas; historical HWM can mask smaller temporary peaks'}
