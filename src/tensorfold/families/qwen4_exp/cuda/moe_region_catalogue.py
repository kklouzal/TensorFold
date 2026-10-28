"""Private target-only exact-row pure-MoE graph catalogue.

All State, attention, GDN, PLE, partial reductions and NCCL remain canonical.
The model's one coordinator borrows this bounded owner for a complete forward.
"""
import os
import threading

import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sm70.router_blas import PROFILE as PROFILE, PROFILES, RolePlans, role_bytes
from .moe_region_graphs import MoERegion, validate_region
from .capture_lifetime import host_memory as _host_memory, on_stream

ENV = 'TENSORFOLD_SM70_MOE_REGION_GRAPHS'
HC_ENV = 'TENSORFOLD_SM70_HC_MOE_REGION_GRAPHS'
SHARED_ENV = 'TENSORFOLD_SM70_HC_MOE_SHARED_POOL'
ROWS = (30, 31, 32)
HOT_ROWS = (4, 5, 6, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 30, 31, 32)
LAYERS = 48
HOST_ALLOWANCE = 1 << 30
MAX_COUNT = (1 << 63) - 1


def requested(profile):
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError('private MoE graph choice must be0 or1')
    if value == '1' and profile not in PROFILES:
        raise ValueError('private MoE regions require the admitted batched-router profile')
    return value == '1'


def requested_hc(profile):
    value = os.environ.get(HC_ENV, '0')
    if value not in ('0', '1'):
        raise ValueError('private HC/MoE graph choice must be0 or1')
    if value == '1' and (profile not in PROFILES or not requested(profile)):
        raise ValueError('private HC/MoE graphs require the admitted MoE graph catalogue')
    return value == '1'


def requested_shared(profile):
    from tensorfold.cuda.sm70.execution_policy import SHARED_FUSION_TRIAL

    value = os.environ.get(SHARED_ENV, '0')
    if value not in ('0', '1'):
        raise ValueError('private HC/MoE shared-pool choice must be0 or1')
    if value == '1' and (profile != SHARED_FUSION_TRIAL or not requested_hc(profile)):
        raise ValueError('private shared HC/MoE pool requires its selected fused HC catalogue')
    return value == '1'


def funding(layers=LAYERS, *, independent=False):
    if type(independent) is not bool:
        raise ValueError('shared pool funding requires an explicit boolean policy')
    if type(layers) is not int or layers != LAYERS:
        raise ValueError('private MoE catalogue requires its48-layer model')
    gpu, host = role_bytes(32, layers)
    rows = HOT_ROWS if independent else ROWS
    return {'rows': rows, 'maximum_graphs': layers * len(rows),
            'pool_policy': ('one-independent-no-private-output-pool' if independent else 'ordered-pool-per-exact-R'),
            'capture_role_gpu_rounded_bytes': gpu, 'capture_role_host_bound_bytes': host,
            'future_host_bytes_per_rank': HOST_ALLOWANCE,
            'device_charge': 'new GPU active/pool/library bytes charged ONCE by max physical-drop/corrected-general-available-drop plus aggregate settlement; all private inactive excluded from reusable spare',
            'host_charge': '1GiB operating allowance INCLUDES capture-role host, catalogue/bindings/graph/runtime metadata and construction temporaries; no separate role-host addition',
            'baseline': 'canonical Buffers, model weights and eager router roles are already funded; no copies charged again'}


def _span(tensor):
    storage = tensor.untyped_storage()
    return (tensor.data_ptr(), storage.data_ptr(), storage.nbytes(), tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device, tensor.requires_grad)


class Bindings:
    """Retain original Storage objects even if a Tensor is later rebound."""
    def __init__(self, weights, buffers):
        self.weights, self.buffers = weights, buffers
        self.layers = tuple(weights.layers)
        self.tensors, self.objects, self.scalars = [], [], []
        self.object(buffers, 'moe')
        self.object(buffers, 'router_blas')
        self.tensor(buffers, 'mixed')
        for name in ('logits', 'pick', 'wts', 'y', 'act'):
            self.tensor(buffers.moe, name)
        for layer in self.layers:
            self.scalar(layer, ('index',))
            self.object(layer, 'moe')
            self.object(layer.moe, 'experts')
            self.tensor(layer.moe, 'router')
            bank = layer.moe.experts
            self.scalar(bank, ('count', 'width', 'dims', 'implementation', 'packed_rows', 'limit', 'swiglu'))
            self.object(bank, 'shared')
            for name in ('gate', 'up', 'down'):
                self.object(bank, name)
                packed = getattr(bank, name)
                self.tensor(packed, 'data')
                self.scalar(packed, ('ggml_type', 'n', 'k', 'column_offset'))
            for dense in bank.shared:
                self.tensor(dense, 'weight')
                self.scalar(dense, ('n', 'k', 'implementation'))
        self.policy = self.current_policy()

    def object(self, owner, name):
        self.objects.append((owner, name, getattr(owner, name)))

    def scalar(self, owner, names):
        self.scalars.append((owner, names, tuple(getattr(owner, name) for name in names)))

    def tensor(self, owner, name):
        value = getattr(owner, name)
        self.tensors.append((owner, name, value, value.untyped_storage(), _span(value)))

    def current_policy(self):
        from tensorfold.cuda import pdl
        from tensorfold.cuda.sm70 import family_ops, routing_native
        from .gguf_weights import GGUFDense, GGUFExperts, GGUFConcat

        w = self.weights
        return (id(w.cfg), w.cfg.experts, w.cfg.top_k,
                tuple(id(layer) for layer in w.layers), w.meta.get('sm70_profile'),
                w.meta.get('execution_provider'), w.meta.get('sm70_moe_region_graphs_trial'),
                w.meta.get('sm70_hc_moe_region_graphs_trial'),
                w.meta.get('sm70_hc_moe_shared_pool_trial'),
                w.meta.get('sm70_family_trial', {}).get('numerical_policy'),
                w.meta.get('sm70_family_trial', {}).get('experts'),
                GGUFDense.implementation, GGUFExperts.implementation, GGUFConcat.implementation,
                family_ops._NATIVE_FAMILY, routing_native._NATIVE_LOGITS, pdl.ON)

    def validate(self):
        if self.current_policy() != self.policy:
            raise RuntimeError('MoE graph effective profile/layer catalogue changed')
        for owner, name, value in self.objects:
            if getattr(owner, name) is not value:
                raise RuntimeError('MoE graph original model/buffer owner changed')
        for owner, names, values in self.scalars:
            if tuple(getattr(owner, name) for name in names) != values:
                raise RuntimeError('MoE graph captured scalar/layout policy changed')
        for owner, name, value, _storage_owner, signature in self.tensors:
            current = getattr(owner, name)
            if current is not value or _span(current) != signature:
                raise RuntimeError('MoE graph captured Tensor storage/span/layout changed')


class Catalogue:
    def __init__(self, parent):
        self.parent, self.dec = parent, parent.dec
        self.weights, self.buffers = self.dec.w, self.dec.buf
        self.layers = tuple(self.weights.layers)
        self.regions, self.pools, self.measured = {}, {}, []
        self.capture_router = self.bindings = None
        self.hc = self.weights.meta.get('sm70_hc_moe_region_graphs_trial') is True
        self.independent = self.weights.meta.get('sm70_hc_moe_shared_pool_trial') is True
        if self.independent:
            from tensorfold.cuda.sm70.execution_policy import SHARED_FUSION_TRIAL

            if not self.hc or self.weights.meta.get('sm70_profile') != SHARED_FUSION_TRIAL:
                raise ValueError('shared HC/MoE graph owner lacks its sealed explicit admission')
        self.row_keys = HOT_ROWS if self.independent else ROWS
        self.pending_hc = None
        self.ready = self.closed = self.retiring = self.active = False
        self.failure = None
        self.notified = False
        self.notification = [None]
        self.rows = self.cursor = self.frames = self.hits = 0
        self.counter_overflowed = False
        self.row_frames = {row: 0 for row in self.row_keys}
        self.frame_lock = threading.Lock()
        self.lease = False
        self.stream_id = None
        self.host_before = self.host_after = None
        self.host_observed_peak_growth = self.host_observed_retained_growth = 0
        self.role_and_warm_bytes = 0
        self.host_preflight = self.host_final = None
        self.plan = funding(len(self.layers), independent=self.independent)
        self.plan['boundary'] = 'MLP-HC-mode3+pure-MoE' if self.hc else 'pure-MoE'


    def _host_gate(self):
        now = _host_memory()
        self.host_after = now
        retained = max(0, now['VmRSS'] - self.host_before['VmRSS'])
        peak = max(0, now['VmHWM'] - self.host_before['VmHWM'])
        self.host_observed_retained_growth = max(self.host_observed_retained_growth, retained)
        self.host_observed_peak_growth = max(self.host_observed_peak_growth, peak)
        if max(retained, peak) > HOST_ALLOWANCE:
            raise MemoryError('MoE graph construction exceeded its admitted host operating allowance')

    def _account(self, before, label):
        row = self.parent.charge_capture(before, ["moe", *label])
        self.measured.append(row)
        self._host_gate()
        return row['charged_bytes']

    def prepare(self):
        from tensorfold.cuda.capacity import admit_host_commitment

        if self.ready or self.closed or self.failure is not None:
            raise RuntimeError('MoE catalogue is one-time cold preparation')
        if (self.dec.streams or self.dec.filling or self.dec.kept
                or len(self.dec.free) != self.dec.round_tables.slots or len(self.dec.free) < 24):
            raise RuntimeError('MoE capture requires quiescent24-slot cold ownership')
        if self.parent.allocator_backend != 'native':
            raise ValueError('private MoE pools require the qualified native allocator accounting')
        if getattr(self.weights, 'dspark', None) is not None or getattr(self.weights, 'dflash2', None) is not None:
            raise ValueError('MoE graph catalogue requires the stock MTP execution contract')
        if self.weights.meta.get('sm70_profile') not in PROFILES or self.buffers.router_blas is None:
            raise ValueError('MoE capture requires the selected eager batched-router role')
        if self.buffers.prefill or self.buffers.rows != 32:
            raise ValueError('MoE capture requires canonical target decode row32 buffers')
        self.host_before = _host_memory()
        try:
            self.host_preflight = admit_host_commitment(int(self.weights.meta.get('world', 1)) * HOST_ALLOWANCE)
            if tuple(layer.index for layer in self.layers) != tuple(range(LAYERS)):
                raise ValueError('MoE capture requires the original ordered48-layer catalogue')
            for layer in self.layers:
                validate_region(self.weights, layer, self.buffers, 32)
            region_type = MoERegion
            if self.hc:
                from .hc_moe_region import HCModelBindings, HCReadoutMoERegion, validate_hc_region

                region_type = HCReadoutMoERegion

                for layer in self.layers:
                    validate_hc_region(self.weights, layer, self.buffers, 32)
                self.bindings = HCModelBindings(self.weights, self.buffers)
            else:
                self.bindings = Bindings(self.weights, self.buffers)
            torch.cuda.synchronize(self.weights.device)  # Prior warm/copy/NCCL consumers positively retired.
            before = self.parent.capture_observation()
            remaining = before['general_available_bytes'] - self.parent.future_bytes
            if (self.parent.bytes + self.plan['capture_role_gpu_rounded_bytes'] > self.parent.budget
                    or remaining < self.plan['capture_role_gpu_rounded_bytes']):
                raise MemoryError('capture router role has no room after required service')
            self.stream_id = torch.cuda.current_stream(self.weights.device).cuda_stream
            def prepare_role():
                self.capture_router = object.__new__(RolePlans)
                self.capture_router.__init__(tuple(layer.moe.router for layer in self.layers), 32)
                # Every cold input row is valid initialized FP16; old N1warm only covered <=3.
                self.buffers.mixed[:32].zero_()
                if self.hc:
                    # Old bound warmup covers <=3 rows; every capture operand
                    # must be initialized for all exact30/31/32 cold graphs.
                    self.buffers.h[:32].fill_(0.5)
                    self.buffers.pss[:32].zero_()
                    self.buffers.g_branch.zero_()
                    self.buffers.inj_a[:32].fill_(1)

            on_stream(torch.cuda, self.parent.side, prepare_role)
            self.parent.side.synchronize()
            self.role_and_warm_bytes = self._account(before, ['capture-role-and-initial-residency'])
            if self.independent:
                # All graph-private intermediates die in their own pure body.
                # Only external operands survive, and the exclusive same-stream
                # forward lease preserves complete48-layer semantic order.
                self.pools['shared'] = torch.cuda.graph_pool_handle()
            for rows in self.row_keys:
                if not self.independent:
                    self.pools[rows] = torch.cuda.graph_pool_handle()
                pool = self.pools['shared'] if self.independent else self.pools[rows]
                for layer in self.layers:  # Fixed48-layer replay order for every exactR cycle.
                    before = self.parent.capture_observation()
                    key = rows, layer.index
                    region = object.__new__(region_type)
                    self.regions[key] = region
                    region.__init__(self.weights, layer, self.buffers, rows, side=self.parent.side,
                                    pool=pool, capture_router=self.capture_router)
                    region.prepare()
                    if self.hc:
                        region.replay(3, region.branch, region.injection)
                    else:
                        region.replay()  # Materialize graph/runtime resources cold, noState orRNG mutation.
                    self._account(before, ['target', rows, layer.index])
            self.bindings.validate()
            self._host_gate()
            self.host_final = admit_host_commitment(int(self.weights.meta.get('world', 1)) * HOST_ALLOWANCE)
            self.ready = True
        except BaseException as error:
            self.failure = error
            self.retiring = True
            raise

    def _poison(self, error):
        if getattr(self, 'failure', None) is None:
            self.failure = error
        self.retiring = True
        self.dec._memory_error = self.failure

    def _notify(self, error):
        # One fatal publication attempt, after releasing the forward lease.
        # No outcome-container allocation can replace the named operation error.
        if not self.notified:
            self.notified = True
            context = getattr(self.buffers, 'input_preparation_context', None)
            if context is not None:
                try:
                    context.fail(error)
                except BaseException as notification:
                    self.notification[0] = notification
        if self.notification[0] is not None:
            raise_failures(error, self.notification)
        raise error

    def begin(self, weights, buffers, rows):
        if not self.ready or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError('MoE graph catalogue is not reusable') from self.failure
        if rows not in self.row_keys:
            return False
        if not self.parent.admitted:
            raise RuntimeError('MoE graph catalogue has not passed aggregate physical admission')
        if not self.frame_lock.acquire(blocking=False):
            raise RuntimeError('MoE graph owner already has a forward borrower')
        self.lease = self.active = True
        self.rows, self.cursor = rows, 0
        self.pending_hc = None
        try:
            if weights is not self.weights or buffers is not self.buffers or buffers.prefill:
                raise ValueError('MoE graph forward lost its canonical target owner')
            if torch.cuda.current_stream(self.weights.device).cuda_stream != self.stream_id:
                raise RuntimeError('MoE graph forward requires the original ordered serving stream')
            self.bindings.validate()  # Once perforward; never perlayer catalogue walks.
            return True
        except BaseException as error:
            self._poison(error)
            self.active = self.lease = False
            self.frame_lock.release()
            self._notify(error)

    def defer_hc(self, layer, buffers, rows, mode, branch, injection):
        """Issue the original HC cut to the immediately following MoE replay."""
        if not self.hc or not self.active:
            return False
        if (buffers is not self.buffers or rows != self.rows or self.cursor >= len(self.layers)
                or layer is not self.layers[self.cursor] or self.pending_hc is not None):
            raise RuntimeError('HC/MoE original cut order or owner changed')
        region = self.regions[rows, layer.index]
        if (type(mode) is not int or mode != 3 or type(branch) is not torch.Tensor
                or type(injection) is not torch.Tensor or _span(branch) != region.branch_signature
                or _span(injection) != region.injection_signature):
            raise RuntimeError('HC/MoE requires its original mode3 branch/injection spans')
        self.pending_hc = layer
        return True

    def run(self, layer, buffers, rows):
        if not self.active:
            return False
        if (buffers is not self.buffers or rows != self.rows or self.cursor >= len(self.layers)
                or layer is not self.layers[self.cursor]):
            raise RuntimeError('MoE graph exactRpool replay order changed within a forward')
        region = self.regions[rows, layer.index]
        if self.hc:
            if self.pending_hc is not layer:
                raise RuntimeError('HC/MoE replay has no issued original HC cut')
            region.replay_prevalidated()
            self.pending_hc = None
        else:
            region.replay()
        self.cursor += 1
        if self.hits < MAX_COUNT:
            self.hits += 1
        else:
            self.counter_overflowed = True
        return True

    def finish(self, primary=None):
        failure = primary
        if self.active:
            try:
                if primary is None:
                    if self.cursor != len(self.layers) or self.pending_hc is not None:
                        raise RuntimeError('MoE graph forward did not consume its full ordered layer cycle')
                    if self.frames < MAX_COUNT and self.row_frames[self.rows] < MAX_COUNT:
                        self.frames += 1
                        self.row_frames[self.rows] += 1
                    else:
                        self.counter_overflowed = True
            except BaseException as error:
                failure = error
            finally:
                self.active = self.lease = False
                self.frame_lock.release()
        if failure is not None:
            self._poison(failure)  # Partial cycles never reuse any pool.
            self._notify(failure)

    def close(self):
        if getattr(self, 'closed', False):
            return
        self.retiring = True
        lock = getattr(self, 'frame_lock', None)
        if lock is not None and not lock.acquire(blocking=False):
            raise RuntimeError('MoE graph forward borrower has not retired')
        try:
            weights = getattr(self, 'weights', None)
            if weights is not None:
                torch.cuda.synchronize(weights.device)
            regions = getattr(self, 'regions', None)
            if regions is not None:
                for region in regions.values():
                    if getattr(region, 'graph', None) is not None:
                        region._release_fenced()
                regions.clear()
            pools = getattr(self, 'pools', None)
            if pools is not None:
                pools.clear()
            # Every graph reset precedes shared pointer-table/storage release.
            self.capture_router = self.bindings = None
            self.closed = True
        except BaseException as error:
            self._poison(error)
            raise
        finally:
            if lock is not None:
                lock.release()

    def record(self, *, detail=False):
        row_charges = {row: sum(item['charged_bytes'] for item in self.measured
                               if len(item['key']) > 2 and item['key'][1] == 'target' and item['key'][2] == row)
                       for row in self.row_keys}
        return {'enabled': True, 'numeric_profile': self.weights.meta.get('sm70_profile'),
                'graph_contract': ('mlp-hc-mode3-moe-hot16-independent-pool-v1' if self.independent else
                    'mlp-hc-mode3-moe-target-exactR-ordered-pools-v1' if self.hc else 'pure-moe-target-exactR-ordered-pools-v1'), 'rows': self.row_keys, 'graphs': len(self.regions), 'frames': None if self.counter_overflowed else self.frames,
                'replays': None if self.counter_overflowed else self.hits,
                'row_frames': None if self.counter_overflowed else dict(self.row_frames), 'funding': self.plan,
                'capture_role_and_initial_residency_charge_bytes': self.role_and_warm_bytes,
                'host_retained_growth_bytes_observed': self.host_observed_retained_growth,
                'host_peak_growth_bytes_observed': self.host_observed_peak_growth,
                'host_before': self.host_before, 'host_after': self.host_after,
                'host_preflight': self.host_preflight, 'host_final': self.host_final,
                'host_observation': 'boundary RSS and HWM deltas, not universal vendor/transient bound; historical HWM can mask smaller temporary peaks',
                'ready': self.ready, 'closed': self.closed,
                'capture_measurements': self.measured if detail else {
                    'count': len(self.measured), 'target_charge_by_exact_rows': row_charges}}
