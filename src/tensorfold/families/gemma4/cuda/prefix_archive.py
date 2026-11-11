"""Gemma canonical GPU prefix archive over the ordinary graph-bound working State.

Only prompt-role, jointly committed target/companion rows are published. Payload
stays on its original device. The Torch-free radix owns physical group credits;
CUDA producer/consumer leases retire only after their actual completion event.
This is Architecture A: GPU imports and canonical response work are paid work,
not a claim that metadata indexing alone eliminates model/copy costs.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys

from tensorfold.engine.prefix_blocks import PrefixBlocks, PrefixCapacity, WriteLease
from tensorfold.cleanup import raise_failures


@dataclass(frozen=True, slots=True)
class PrefixIdentity:
    position: int
    backing: tuple


@dataclass(slots=True)
class PublicationIntent:
    owner: object
    mode: str
    closed: bool = False


@dataclass(slots=True, eq=False)
class _Pending:
    event: object
    leases: list
    errors: list
    data: object = None
    recorded: bool = False
    effects: bool = False


class _MetadataCredit:
    __slots__ = ('archive', 'credit', 'nbytes', 'closed')

    def __init__(self, archive, credit, nbytes):
        self.archive, self.credit, self.nbytes, self.closed = archive, credit, nbytes, False

    def close(self):
        with self.archive.store._lock:
            if self.closed:
                return
            try:
                self.credit.close()
            except BaseException as primary:
                if self.archive.metadata_failure is None:
                    self.archive.metadata_failure = primary
                raise
            del self.archive.host_metadata[id(self)]
            self.closed = True



class _MetadataRegistration:
    __slots__ = ('archive', 'owner', 'close_owner', 'credit', 'failure', 'retiring',
                 'consumer_done', 'closed', 'attempted', 'close_error')

    @property
    def failed(self):
        return self.failure is not None

    def close(self):
        """Normal owner cleanup declares CPU completion, never refunds failure."""
        with self.archive.store._lock:
            if self.closed:
                return
            self.consumer_done = True
            if self.failed and not self.retiring:
                return
            if self.close_error is not None:
                raise self.close_error
            try:
                self.credit.close()
            except BaseException as error:
                self.close_error = error  # owner already retired; never retry a torn registry close
                self.attempted = True
                self.archive.metadata_failed(error, self.owner)
                raise
            del self.archive.metadata_owners[id(self.owner)]
            self.owner = self.close_owner = self.failure = None
            self.closed = True



class _RawResponseWork:
    """Paid CPU tape prepared before model effects; no large post-decode copy."""
    __slots__ = ('archive', 'credit', 'registration', 'data', 'prompt_rows',
                 'publish_started', 'retained', 'closed', 'errors')

    def __init__(self, archive, prompt, output_rows):
        self.archive, self.data, self.credit, self.registration = archive, None, None, None
        self.prompt_rows, self.publish_started, self.retained, self.closed = len(prompt), False, False, False
        self.errors = [None]
        slots = len(prompt) + output_rows
        pointer = sys.getsizeof([None]) - sys.getsizeof([])
        # Tape, original owned input list and worst reply-ID/list wrappers can
        # all survive an exception frame; reserve their explicit CPU extent.
        amount = 3 * sys.getsizeof([]) + (2 * slots + len(prompt) + 16) * pointer
        amount += output_rows * sys.getsizeof(archive.engine.target.config.vocab - 1)
        self.credit = archive.reserve_metadata(amount)
        try:
            self.registration = archive.register_metadata_owner(self)
        except BaseException as primary:
            try:
                self.credit.close()  # no optional tape was allocated
            except BaseException as cleanup:
                self.errors[0] = cleanup
                archive._raise_failures(primary, self.errors)
            raise
        try:
            self.data = [None] * slots
            for index, token in enumerate(prompt):
                self.data[index] = token
        except BaseException as primary:
            self.retained = True
            archive.metadata_failed(primary, self)
            self.registration.close()
            raise

    def __enter__(self):
        return self

    def tokens(self, output):
        for index, token in enumerate(output, self.prompt_rows):
            self.data[index] = token
        del self.data[self.prompt_rows + len(output):]
        self.publish_started = True  # a returned CPU tape may enter a retained caller frame
        return self.data

    def close(self):
        if self.closed:
            return
        self.data = None  # no detached CPU tape borrow survives credit return
        if self.retained and not self.registration.retiring:
            self.registration.close()
            return
        try:
            self.credit.close()
        except BaseException as primary:
            self.retained = True
            self.archive.metadata_failed(primary, self, close_failed=True)
            self.registration.close()
            raise
        self.registration.close()
        self.closed = True

    def __exit__(self, _kind, primary, _trace):
        if primary is not None and self.publish_started:
            # Canonical native/record frames may borrow the actual CPU tape.
            self.retained = True
            self.archive.metadata_failed(primary, self)
            self.registration.close()
            return False
        try:
            self.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            self.errors[0] = cleanup
            self.archive._raise_failures(primary, self.errors)
        return False



class GemmaPrefixArchive:
    def __init__(self, engine, budget_bytes):
        import torch
        self.engine, self.state, self.draft = engine, engine._state, engine._draft
        self.device, self.torch = torch.device(self.state.device), torch
        self.store = PrefixBlocks(budget_bytes, block_tokens=64)
        self.failure, self.closed = None, False
        self.pending = []
        self.metadata_owners, self.host_metadata, self.metadata_failure = {}, {}, None
        self.host_admission = None
        self._raise_failures = raise_failures
        self.spark = engine._prefix_draft_kind == 'sparkulator'
        one = self.state.cache_group_bytes(1)
        self.windows = {name: None if name == 'target_full' else self.state.ring
                        for name, size in one.items() if size}
        if self.draft is not None and not self.spark:
            self.draft_bytes = self.draft.cache_group_bytes(1)
            rings = [r for r, kind in zip(self.draft.rings, self.draft.layer_types)
                     if kind == 'sliding_attention']
            for name, size in self.draft_bytes.items():
                if size:
                    self.windows[name] = None if name == 'draft_full' else max(rings)
        else:
            self.draft_bytes = {}
        if self.spark:
            self.windows['hidden'] = 1  # companion bridge needs final hidden even at a partial hit
        self.store.configure_retention(self.windows, require_end_group='hidden')
        self.hidden_width = engine.target.config.hidden
        self._last = {'requested_tokens': 0, 'matched_tokens': 0, 'materialized_tokens': 0}

    def check_live(self):
        if self.failure is None and self.store._failure is not None:
            self.failure = self.store._failure
        if self.failure is not None:
            raise RuntimeError('Gemma shared prefix archive failed; retain its owners') from self.failure
        if self.closed:
            raise RuntimeError('Gemma shared prefix archive is closed')

    def reserve_metadata(self, nbytes):
        """Nonblocking metadata credit; no model/copy work or lane wait.

        Unpinned, positively retired payload eviction may run Torch allocator
        retirement internally; it is not a zero-cost/no-CUDA guarantee.
        """
        self.check_live()
        with self.store._lock:
            if self.metadata_failure is not None:
                raise PrefixCapacity('canonical cache metadata is terminally unavailable') from self.metadata_failure
            if type(nbytes) is not int or nbytes < 0:
                raise ValueError('cache metadata requires nonnegative native integer bytes')
            from tensorfold.cuda.capacity import admit_host_commitment, HostCommitmentCapacityError
            # Serialize every outstanding engine-issued host claim. Until the
            # caller proves actual materialization we grant ZERO credit against
            # physical availability; no allocator/UMA reservation is credited.
            overhead = (sys.getsizeof(object.__new__(_MetadataCredit)) + sys.getsizeof({0: None})
                        + sys.getsizeof((0, None)) + sys.getsizeof(0))
            funded = nbytes + overhead
            total = funded + sum(lease.nbytes for lease in self.host_metadata.values())
            try:
                admission = admit_host_commitment(total, materialized_owned_bytes=0)
            except HostCommitmentCapacityError as capacity:
                raise PrefixCapacity('canonical cache metadata has insufficient physical host credit') from capacity
            # Allocate/register the empty owner before accepting Core credit;
            # no late dict/wrapper allocation can strand an untracked lease.
            lease = _MetadataCredit(self, None, funded)
            self.host_metadata[id(lease)] = lease
            try:
                lease.credit = self.store.reserve_temporary(funded)
            except BaseException:
                del self.host_metadata[id(lease)]  # no credit/callback accepted
                raise
            self.host_admission = admission
            return lease

    def register_metadata_owner(self, owner):
        self.check_live()
        with self.store._lock:
            if self.metadata_failure is not None:
                raise PrefixCapacity('canonical cache metadata is terminally unavailable') from self.metadata_failure
            previous = self.metadata_owners.get(id(owner))
            if previous is not None:
                if previous.owner is not owner:
                    raise RuntimeError('cache metadata owner generation differs')
                return previous
            close = getattr(owner, 'close', None)
            if not callable(close):
                raise TypeError('registered metadata owner needs its actual close operation')
            record = object.__new__(_MetadataRegistration)
            # Per-owner registry/table/retirement-header extent; data itself is
            # charged by its existing HP claim or reserve_metadata, never both.
            header = sys.getsizeof(record) + sys.getsizeof({0: None}) + 4 * sys.getsizeof(0) + 5 * sys.getsizeof(None)
            record.archive, record.owner, record.close_owner, record.credit = self, owner, close, None
            record.failure = record.close_error = None
            record.retiring = record.consumer_done = record.closed = record.attempted = False
            self.metadata_owners[id(owner)] = record
            try:
                record.credit = self.reserve_metadata(header)
            except BaseException:
                del self.metadata_owners[id(owner)]  # no registry credit accepted
                record.owner = record.close_owner = None
                raise
            return record

    def metadata_failed(self, primary, owner, *, close_failed=False):
        if not isinstance(primary, BaseException) or type(close_failed) is not bool:
            raise TypeError('authentic metadata primary and exact close status required')
        with self.store._lock:
            record = self.metadata_owners.get(id(owner))
            if record is None or record.owner is not owner or record.closed:
                raise RuntimeError('metadata failure owner was not preregistered') from primary
            if close_failed:
                record.attempted = record.consumer_done = True
                if record.close_error is None:
                    record.close_error = primary
            if record.failure is None:
                record.failure = primary
            if self.metadata_failure is None:
                self.metadata_failure = primary

    def _retire_metadata(self):
        # No wait for frontend consumers while owning the lane they may need.
        with self.store._lock:
            records = tuple(self.metadata_owners.values())
            if any(not record.consumer_done for record in records):
                raise RuntimeError('cache metadata still has active frontend consumers')
            errors = [None] * len(records)  # before any cleanup failure
            first = None
            for index, record in enumerate(records):
                if record.closed:
                    continue
                record.retiring = True
                if record.close_error is not None:
                    errors[index] = record.close_error
                elif not record.attempted:
                    record.attempted = True
                    try:
                        record.close_owner()
                        record.close()
                    except BaseException as error:
                        record.close_error = errors[index] = error
                if errors[index] is not None and first is None:
                    first = errors[index]
            if first is not None:
                native_cause = BaseException.__cause__.__get__(first)
                native_context = BaseException.__context__.__get__(first)
                try:
                    for index in range(len(errors)):
                        if errors[index] is None:
                            errors[index] = first
                    self._raise_failures(first, errors)
                except BaseException as transport:
                    if native_cause is first:
                        native_cause = None
                    if native_context is first:
                        native_context = None
                    if transport is not first:
                        raise first from transport
                    raise
            self.metadata_failure = None

    def prepare_raw_response(self, prompt, output_rows):
        if self.metadata_failure is not None:
            return None
        work = object.__new__(_RawResponseWork)
        try:
            work.__init__(self, prompt, output_rows)
            return work
        except PrefixCapacity as primary:
            self._capacity_declined(primary)
            if work.errors[0] is not None:
                raise  # expected refusal cannot hide a required cleanup failure
            return None  # valid generation, explicit optional nonpublication

    def tags(self, prepared_media):
        if prepared_media is None or prepared_media == ():
            return ()
        if getattr(prepared_media, 'cache_metadata_complete', True) is not True:
            raise ValueError('unfunded media metadata cannot authorize a text-only cache entry')
        from .vision import prefix_media_tags
        return prefix_media_tags(prepared_media)

    def metadata_complete(self, prepared_media):
        return prepared_media is None or getattr(prepared_media, 'cache_metadata_complete', True) is True

    def _journal(self, slots, scratch_bytes):
        # Journal storage and its retirement slots exist before producer effects.
        event = self.torch.cuda.Event() if self.device.type == 'cuda' else None
        pending = _Pending(event, [None] * slots, [None] * slots)
        charged = scratch_bytes + sys.getsizeof(pending) + sys.getsizeof(pending.leases) + sys.getsizeof(pending.errors)
        if event is not None:
            charged += sys.getsizeof(event)
        # Register the empty journal before accepting credit. A list-growth
        # failure then owns no lease; a refused grant removes this no-effects
        # record, rather than stranding paid credit in an exception traceback.
        self.pending.append(pending)
        try:
            pending.leases[0] = self.store.reserve_temporary(charged)
        except BaseException:
            self.pending.remove(pending)
            raise
        return pending

    def _record(self, pending):
        if pending.event is not None:
            pending.event.record(self.torch.cuda.current_stream(self.device))
        pending.recorded = True
        self.reap()

    def _retire(self, pending):
        first = None
        # Every accepted lease is attempted independently. Error slots were paid
        # before its native producer/consumer was accepted.
        for index, lease in enumerate(pending.leases):
            if pending.errors[index] is not None:
                if first is None:
                    first = pending.errors[index]
                continue  # an uncertain failed close is never retried
            if lease is None:
                continue
            try:
                if type(lease) is WriteLease:
                    if lease._committed:
                        lease.finish()
                    else:
                        lease.abort()
                else:
                    lease.close()
            except BaseException as error:
                pending.errors[index] = error
                if first is None:
                    first = error
        if first is not None:
            native_cause = BaseException.__cause__.__get__(first)
            native_context = BaseException.__context__.__get__(first)
            try:
                for index in range(len(pending.errors)):
                    if pending.errors[index] is None:
                        pending.errors[index] = first
                self._raise_failures(first, pending.errors)
            except BaseException as transport:
                # Named original/status references remain in this retained
                # frame even if neutral transport itself cannot allocate.
                if native_cause is first:
                    native_cause = None
                if native_context is first:
                    native_context = None
                if transport is not first:
                    raise first from transport
                raise
        pending.data = None

    def reap(self, *, positive_retirement=False):
        """No flag/time/thread-state inference: query the actual event or drain."""
        try:
            while self.pending:
                pending = self.pending[0]
                if not positive_retirement and (not pending.recorded or
                        pending.event is not None and not pending.event.query()):
                    break
                self._retire(pending)
                del self.pending[0]
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise

    def _wait_oldest(self):
        if not self.pending:
            return False
        pending = self.pending[0]
        if not pending.recorded:
            return False
        try:
            if pending.event is not None:
                pending.event.synchronize()
            self._retire(pending)
            del self.pending[0]
            return True
        except BaseException as primary:
            if self.failure is None:
                self.failure = primary
            raise

    def _capacity_declined(self, primary):
        # A capacity primary with failed cleanup is retained by Core poison;
        # it must never be converted to an ordinary optional-cache miss.
        if self.store._failure is not None:
            if self.failure is None:
                self.failure = self.store._failure
            raise primary

    def _credit(self, acquire):
        while True:
            try:
                return acquire()
            except PrefixCapacity as primary:
                self._capacity_declined(primary)
                # Backpressure only on this archive's actually accepted GPU
                # consumers, never on frontend owners that need this lane.
                if not self._wait_oldest():
                    raise

    def _no_effects(self, pending):
        self._retire(pending)
        self.pending.remove(pending)

    def _scratch(self, rows):
        # Target arange+remainder; companion's previous layer index can remain
        # live while the next two indices are constructed. No full-prefix index.
        return 3 * rows * 8 + (self.hidden_width * 2 if self.spark else 0)

    def capture(self, tokens, start, stop, hidden, prepared_media=None, *, checkpoint_input=False):
        """Export native prefill batches;64 is index granularity, not model M."""
        self.check_live()
        if type(checkpoint_input) is not bool:
            raise TypeError('checkpoint_input must be a bool')
        if checkpoint_input and stop != len(tokens):
            raise ValueError('input checkpoint requires complete native tape EOF')
        if self.metadata_failure is not None or not self.metadata_complete(prepared_media):
            return False
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(tokens)
                or hidden.shape != (stop - start, self.hidden_width)
                or hidden.dtype != self.state.keys.dtype or hidden.device != self.state.keys.device):
            raise ValueError('canonical hidden/frontier does not match its committed State rows')
        tags = self.tags(prepared_media)
        limit = min(self.state.rows, 1024)
        batches = -(-(stop - start) // limit)
        try:
            pending = self._credit(lambda: self._journal(1 + 6 * batches, self._scratch(limit)))
        except PrefixCapacity as primary:
            self._capacity_declined(primary)
            return False
        complete, data, writer = True, None, None
        try:
            slot = 1
            for begin in range(start, stop, limit):
                end = min(stop, begin + limit)
                sizes = self.state.cache_group_bytes(end - begin)
                if self.draft is not None and not self.spark:
                    if self.draft.length != self.state.position:
                        raise ValueError('target and canonical companion frontiers differ')
                    sizes.update(self.draft.cache_group_bytes(end - begin))
                sizes['hidden'] = (end - begin) * self.hidden_width * hidden.element_size()
                sizes = {name: size for name, size in sizes.items() if size}
                missing = dict(sizes)
                try:
                    for group in sizes:
                        existing = self.store.pin_range(tokens, group, begin, end, tags=tags)
                        if existing is not None:
                            pending.leases[slot] = existing
                            slot += 1
                            del missing[group]
                    if not missing:
                        if checkpoint_input and end == stop:
                            complete = self.checkpoint(tokens,prepared_media) and complete
                        else:
                            self.store.retain_frontier(tokens, end, tags=tags)
                        continue
                    writer = self._credit(lambda: self.store.begin(tokens, begin, end, missing, tags=tags,
                                                                 retain_frontier=True,
                                                                 checkpoint=checkpoint_input and end == stop))
                except PrefixCapacity as primary:
                    self._capacity_declined(primary)
                    complete = False
                    break  # no current-batch producer effects were accepted
                pending.leases[slot] = writer
                slot += 1
                pending.effects = True  # before any fallible native producer
                target_groups = tuple(name for name in missing if name.startswith('target_'))
                data = self.state.cache_export_rows(begin, end, groups=target_groups) if target_groups else {}
                draft_groups = tuple(name for name in missing if name.startswith('draft_'))
                if draft_groups:
                    data.update(self.draft.cache_export_rows(begin, end, groups=draft_groups))
                if 'hidden' in missing:
                    data['hidden'] = (hidden[begin - start:end - start].clone(),)
                pending.data = data
                writer.commit(data)
            pending.data = data = writer = None
            if pending.effects:
                self._record(pending)
            else:
                self._no_effects(pending)
            return complete
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise

    def checkpoint(self, tokens, prepared_media=None):
        """Metadata-only complete-tape boundary; no model/import/projection/wait.

        Funding may evict retired inactive payload. False is an honest metadata,
        availability or capacity decline; terminal ownership failures propagate.
        Callers must recheck authoritative identity after this operation.
        """
        self.check_live()
        if self.metadata_failure is not None or not self.metadata_complete(prepared_media):
            return False
        try:
            return self.store.retain_frontier(tokens,len(tokens),tags=self.tags(prepared_media),checkpoint=True)
        except PrefixCapacity as primary:
            self._capacity_declined(primary)
            return False

    def restore(self, tokens, prepared_media=None, *, project_logits=True, checkpoint_input=False):
        """Return (actual restored rows, optional raw exact-end logits)."""
        self.check_live()
        if type(checkpoint_input) is not bool:
            raise TypeError('checkpoint_input must be a bool')
        self._last = {'requested_tokens': len(tokens), 'matched_tokens': 0, 'materialized_tokens': 0}
        self.reap()
        if self.metadata_failure is not None or not self.metadata_complete(prepared_media):
            self.state.reset()
            if self.draft is not None:
                self.draft.reset()
            return 0, None
        try:
            # Physical producer groups may contain1024 rows even though radix
            # labels have64 rows. Fund the actual largest import index batch.
            pending = self._credit(lambda: self._journal(2, self._scratch(min(self.state.rows, 1024))))
        except PrefixCapacity as primary:
            self._capacity_declined(primary)
            self.state.reset()
            if self.draft is not None:
                self.draft.reset()
            return 0, None
        try:
            try:
                read = self.store.match(tokens, self.windows, tags=self.tags(prepared_media))
            except PrefixCapacity as primary:
                self._capacity_declined(primary)
                # Lookup metadata/read pins are optional, just like capture
                # payload. No native restore was accepted at this boundary.
                self._no_effects(pending)
                self.state.reset()
                if self.draft is not None:
                    self.draft.reset()
                return 0, None
            if read is None:
                self._no_effects(pending)
                self.state.reset()
                if self.draft is not None:
                    self.draft.reset()
                return 0, None
            pending.leases[1] = read
            cut = read.length
            if checkpoint_input and cut == len(tokens) and not self.checkpoint(tokens,prepared_media):
                # No GPU consumer has started. Retire this actual CPU read pin
                # before declining the optional funded restore boundary.
                self._no_effects(pending)
                self.state.reset()
                if self.draft is not None:
                    self.draft.reset()
                return 0, None
            selected_last = {'requested_tokens': len(tokens), 'matched_tokens': cut, 'materialized_tokens': cut}
            pending.effects = True
            self.state.poisoned = True
            for group, window in self.windows.items():
                if group == 'hidden':
                    continue
                start = 0 if window is None else max(0, cut - window)
                position = start
                for tensors, offset, rows in read.spans(group, start, cut):
                    if group.startswith('target_'):
                        self.state.cache_import_rows(group, position, tensors, offset, rows)
                    else:
                        self.draft.cache_import_rows(group, position, tensors, offset, rows)
                    position += rows
            row = None
            if project_logits and cut == len(tokens) or self.spark:
                spans = read.spans('hidden', cut - 1, cut)
                if len(spans) != 1 or spans[0][2] != 1:
                    raise ValueError('exact-end hidden has no complete row authority')
                tensors, offset, _rows = spans[0]
                row = tensors[0][offset:offset + 1]
            if self.draft is not None:
                if self.spark:
                    from .sparkulator_bridge import BridgeSnapshot
                    self.state.position = cut  # still poisoned: bridge validates target frontier, no model reads
                    self.draft.restore(BridgeSnapshot(cut, row.unsqueeze(0), self.draft._snapshot_token))
                else:
                    self.draft.length = cut
            self.state.cache_finish_restore(cut)
            logits = self.engine.target.project_logits(row) if project_logits and cut == len(tokens) else None
            self._last = selected_last
            # Drop every actual span/row borrow BEFORE releasing read pins;
            # concurrent metadata grants may then evict its backing safely.
            spans = tensors = row = None
            self._record(pending)
            return cut, logits
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise

    def identity(self, tokens, media=None):
        self.check_live()
        try:
            found = self.store.identity(tokens, self.windows, tags=self.tags(media))
        except PrefixCapacity as primary:
            self._capacity_declined(primary)
            return None
        return None if found is None else PrefixIdentity(found[0], found[1])

    def clear(self):
        """Caller already positively drained the actual working stream."""
        self.reap(positive_retirement=True)
        # Prepared frontend owners may already exist before a maintenance cold
        # clear. Their independent host credit/authority must survive it.
        self.store.clear(preserve_temporary=tuple(lease.credit for lease in self.host_metadata.values()))

    def close(self):
        self.reap(positive_retirement=True)
        self._retire_metadata()
        if self.host_metadata:
            raise RuntimeError('cache metadata credits still have unretired owners')
        self.store.close()
        self.failure = None
        self.closed = True

    def stats(self):
        data = self.store.stats
        uncertain = sum(lease._record.groups[name].nbytes
                        for pending in self.pending if pending.effects and self.failure is not None
                        for lease in pending.leases if type(lease) is WriteLease and not lease._committed
                        for name in lease._record.groups)
        return {'entries': None, 'max_entries': None, 'capacity': None, 'tokens': None, 'slots': None,
                'nodes': data['nodes'], 'chunks': data['chunks'],
                'indexed_tokens': data['indexed_unique_tokens'],
                'bytes': data['payload_bytes'], 'reserved_payload_bytes': data['reserved_payload_bytes'],
                'uncertain_payload_bytes': uncertain,
                'charged_bytes': data['charged_bytes'], 'byte_budget': data['budget_bytes'],
                'temporary_bytes': data['temporary_bytes'],
                'store_schema': 'gemma-canonical-block-archive-v1', 'groups': data['groups'],
                'pins': data['pins'], 'pending_batches': len(self.pending), 'evictions': data['evictions'],
                'eviction_policy': data['eviction_policy'],
                'retained_frontiers': data['retained_frontiers'],
                'checkpoint_frontiers': data['checkpoint_frontiers'],
                'provisional_frontiers': data['provisional_frontiers'],
                'selected_payload_bytes': data['selected_payload_bytes'],
                'eviction_history': data['eviction_history'], 'eviction_history_capacity': 32,
                'last_lookup': dict(self._last),
                'metadata_status': 'failed' if self.metadata_failure is not None else 'ready',
                'metadata_failure_type': type(self.metadata_failure).__name__ if self.metadata_failure is not None else None,
                'metadata_owners': len(self.metadata_owners),
                'host_metadata_claim_bytes': sum(lease.nbytes for lease in self.host_metadata.values()),
                'host_admission_observed': self.host_admission is not None,
                'scope': 'published payload versus reserved/uncertain obligations; indexed tokens are not a restorable hit; allocator/RSS measured separately'}
