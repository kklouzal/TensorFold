"""Torch-free immutable prefix payloads under one byte credit.

The adapter owns device ordering and supplies exact unique-storage payload bytes.
Metadata uses conservative logical CPython charges (not a physical RSS claim).
Tokens/tags are borrowed stable only during a call; stored radix labels and tags
are owned and charged. Data objects must be immutable after producer completion.
Commit publishes all groups together; a writer pin survives until finish after
positive producer retirement. Read close likewise declares consumer retirement.
No finalizer releases a lease. A poisoned store retains its first failure.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import heapq
import sys
from bisect import bisect_right, bisect_left
from tensorfold.cleanup import raise_failures

EVICTION_POLICY = "complete-prefix-availability-aging-bytes-tiebreak-v2"

_BASE, _NODE, _TOKEN, _VIEW, _RECORD, _GROUP, _LEASE, _SPAN = 18432, 1024, 48, 256, 512, 512, 512, 512


def validate_prefix_budget(budget_bytes):
    """Validate the existing positive store domain without allocating a store.

    Zero disables an archive at its owning admission boundary; it is not a
    constructible empty-index budget. This single rule governs all callers.
    """
    if type(budget_bytes) is not int or budget_bytes <= _BASE + _NODE:
        raise ValueError('positive budget must fund the empty index metadata')
    return budget_bytes


@dataclass(slots=True)
class _Eviction:
    component: object = None
    start: int = 0
    stop: int = 0
    nbytes: int = 0
    reason: str = 'byte-pressure'


class PrefixCapacity(MemoryError):
    """No evictable byte credit remains; no producer effects are authorized."""


@dataclass(slots=True, eq=False)
class _Node:
    label: tuple
    depth: int
    parent: object
    children: dict
    views: list
    alive: bool = True


@dataclass(slots=True, eq=False)
class _Group:
    name: str
    nbytes: int
    generation: int
    data: object = None
    pins: int = 1
    published: bool = False
    used: int = 0
    window: bool = False
    benefit: int = 1
    fanout: int = 1


@dataclass(slots=True, eq=False)
class _Record:
    start: int
    stop: int
    groups: dict
    tags: tuple
    tag_bytes: int
    node: object = None


def _identity_charge(value, budget):
    total, stack, seen = 0, [value], set()
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        kind = type(item)
        if kind in (str, bytes):
            total += 128 + 4 * len(item)
        elif kind in (int, bool) or item is None:
            total += 64 + (item.bit_length() // 8 if kind is int else 0)
        elif kind is tuple:
            total += 128 + 8 * len(item)
            if total > budget:
                raise PrefixCapacity('tag metadata exceeds byte credit')
            stack.extend(item)
        else:
            raise TypeError('tag identities must be closed immutable scalar/tuple values')
        if total > budget:
            raise PrefixCapacity('tag metadata exceeds byte credit')
    return total


def _tokens(tokens, budget):
    if type(tokens) not in (tuple, list) or not tokens:
        raise ValueError('nonempty stable native token sequence required')
    if len(tokens) > budget // _TOKEN:
        raise PrefixCapacity('token metadata exceeds byte credit')
    if any(type(t) is not int or not 0 <= t < 2**31 for t in tokens):
        raise ValueError('token IDs must be native nonnegative int32')


def _tags(tags, limit, budget):
    if type(tags) is not tuple:
        raise TypeError('atomic media tags must be a tuple')
    charge, previous = 0, -1
    if len(tags) > budget // 128:
        raise PrefixCapacity('tag count exceeds byte credit')
    for tag in tags:
        if (type(tag) is not tuple or len(tag) != 3 or type(tag[0]) is not int or type(tag[1]) is not int
                or not 0 <= tag[0] < tag[1] <= limit or tag[0] < previous):
            raise ValueError('ordered nonoverlapping bounded media spans required')
        previous = tag[1]
        charge += 256 + _identity_charge(tag[2], budget)
        if charge > budget:
            raise PrefixCapacity('tag metadata exceeds byte credit')
    return charge


def _identity_equal(left, right):
    pending, seen = [(left,right)], set()
    while pending:
        a,b = pending.pop()
        if a is b:
            continue
        key = id(a),id(b)
        if key in seen:
            continue
        seen.add(key)
        if type(a) is not type(b):
            return False
        if type(a) is tuple:
            if len(a) != len(b):
                return False
            pending.extend(zip(a,b))
        elif a != b:
            return False
    return True


def _tags_equal(left, right):
    return len(left) == len(right) and all(a[:2] == b[:2] and _identity_equal(a[2],b[2]) for a,b in zip(left,right))


def _tag_prefix(tags, stop):
    return tuple(tag for tag in tags if tag[0] < stop)


class TemporaryLease:
    __slots__ = ('_store', 'nbytes', '_closed', '_retire_failure', '_registered', '_charged')

    def __init__(self, store, nbytes):
        self._store, self.nbytes, self._closed = store, nbytes, False
        self._retire_failure = None
        self._registered = self._charged = False

    def close(self):
        """Caller has positively retired every temporary consumer."""
        with self._store._lock:
            if self._retire_failure is not None:
                raise RuntimeError('temporary retirement is uncertain; no retry') from self._retire_failure
            if self._closed:
                return
            try:
                charged = self._store._charged-(_LEASE+self.nbytes) if self._charged else self._store._charged
                if self._registered or self in self._store._temporary:
                    self._store._temporary.remove(self)
                self._store._charged = charged
                self._registered = self._charged = False
                self._closed = True
            except BaseException as primary:
                self._retire_failure = primary
                self._store._poison(primary)
                raise


class WriteLease:
    __slots__ = ('_store', '_record', '_closed', '_committed', '_incoming', '_registered', '_charged', '_retire_failure')

    def __init__(self, store, record):
        self._store, self._record = store, record
        self._closed = self._committed = False
        self._incoming = None
        self._registered = self._charged = False
        self._retire_failure = None

    def commit(self, data_by_group):
        """Publish all declared groups atomically; writer ownership remains."""
        with self._store._lock:
            self._store._healthy()
            if self._closed or self._committed:
                raise RuntimeError('writer is closed or already committed')
            # Retain accepted producer owners before any fallible validation.
            self._incoming = data_by_group
            try:
                if type(data_by_group) is not dict or set(data_by_group) != set(self._record.groups):
                    raise ValueError('commit must supply every declared group exactly once')
                if any(data is None for data in data_by_group.values()):
                    raise ValueError('None is not an immutable payload owner')
                for name, group in self._record.groups.items():
                    group.data = data_by_group[name]
                for group in self._record.groups.values():
                    group.published = True
                # A full prefix component supports all current descendants;
                # charge its value by the longest published dependent frontier.
                node = self._record.node
                while node is not None:
                    for record, _lo, _hi in node.views:
                        if not _tags_equal(_tag_prefix(record.tags,min(record.stop,self._record.stop)), _tag_prefix(self._record.tags,min(record.stop,self._record.stop))):
                            continue
                        for group in record.groups.values():
                            if group.published and not group.window and group.name != "hidden":
                                group.benefit = max(group.benefit, self._record.stop)
                                group.fanout = max(group.fanout,len(node.children),1)
                    node = node.parent
                self._committed = True
                self._incoming = None
            except BaseException as error:
                self._store._poison(error)
                raise

    def finish(self):
        """Positive producer retirement releases only the writer pin."""
        with self._store._lock:
            if self._retire_failure is not None:
                raise RuntimeError('writer retirement is uncertain; no retry') from self._retire_failure
            if self._closed:
                return
            if not self._committed:
                raise RuntimeError('uncommitted writer must abort after retirement')
            try:
                pins = tuple((g,g.pins-1) for g in self._record.groups.values())
                charged = self._store._charged-_LEASE
                self._store._writers.remove(self)
                for group,count in pins:
                    group.pins = count
                self._store._charged = charged
                self._registered = self._charged = False
                self._closed = True
                self._incoming = self._record = None
            except BaseException as primary:
                self._retire_failure = primary
                self._store._poison(primary)
                raise

    close = finish

    def abort(self):
        """Caller declares no producer effects, or positive failure retirement."""
        with self._store._lock:
            if self._retire_failure is not None:
                raise RuntimeError('writer retirement is uncertain; no retry') from self._retire_failure
            if self._closed:
                return
            if any(g.pins != 1 for g in self._record.groups.values()):
                raise RuntimeError('published writer still has live readers')
            try:
                if self._charged:
                    self._store._discard(self._record)
                    self._store._charged -= _LEASE
                    self._charged = False
                else:
                    self._record.groups.clear()
                if self._registered or self in self._store._writers:
                    self._store._writers.remove(self)
                    self._registered = False
                self._closed = True
                self._incoming = self._record = None
            except BaseException as primary:
                self._retire_failure = primary
                self._store._poison(primary)
                raise


class ReadLease:
    __slots__ = ('_store', 'length', '_coverage', '_groups', '_closed', '_retire_failure')

    def __init__(self, store, length, coverage, groups):
        self._store, self.length, self._coverage, self._groups = store, length, coverage, groups
        self._closed = False
        self._retire_failure = None

    def spans(self, group, start, stop):
        """Ordered (actual immutable owner, row offset, row count) slices."""
        with self._store._lock:
            if self._closed:
                raise RuntimeError('read lease is closed')
            if type(group) is not str or group not in self._coverage or type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= self.length:
                raise ValueError('span differs from the pinned prefix domain')
            if start < stop and (start < self._coverage[group][0][0] or stop > self._coverage[group][-1][1]):
                raise ValueError('requested span lies outside the pinned group window')
            try:
                rows, cursor = [], start
                for lo, hi, owner, offset in self._coverage[group]:
                    begin, end = max(start, lo), min(stop, hi)
                    if begin < end:
                        if begin != cursor:
                            raise ValueError('requested span lies outside the pinned group window')
                        rows.append((owner.data, offset + begin - lo, end - begin))
                        cursor = end
                if cursor != stop:
                    raise ValueError('requested span lies outside the pinned group window')
                return tuple(rows)

            except BaseException as primary:
                self._store._poison(primary)
                raise

    def close(self):
        """Positive consumer retirement; caller also relinquishes span objects."""
        with self._store._lock:
            if self._retire_failure is not None:
                raise RuntimeError('read retirement is uncertain; no retry') from self._retire_failure
            if self._closed:
                return
            try:
                pins = tuple((g,g.pins-1) for g in self._groups)
                charged = self._store._charged-(_LEASE+_SPAN*sum(map(len,self._coverage.values())))
                self._store._readers.remove(self)
                for group,count in pins:
                    group.pins = count
                self._store._charged = charged
                self._coverage = self._groups = None
                self._closed = True
            except BaseException as primary:
                self._retire_failure = primary
                self._store._poison(primary)
                raise


class PrefixBlocks:
    """Compressed token radix with independently resident immutable groups.

    Store-per-model/runtime/principal is the caller's authority boundary. A
    payload charge must include unique retained physical storage, not view size.
    Window groups are independently evicted; hidden is exact-end-only. Metadata
    is conservatively charged, including leases/pins and caller temporaries.
    """

    def __init__(self, budget_bytes, block_tokens=64):
        validate_prefix_budget(budget_bytes)
        if type(block_tokens) is not int or block_tokens <= 0:
            raise ValueError('positive native block token bound required')
        self.budget_bytes, self.block_tokens = budget_bytes, block_tokens
        self._lock = threading.RLock()
        self._root = _Node((), 0, None, {}, [])
        self._records, self._writers, self._readers, self._temporary = [], [], [], []
        self._serial = self._clock = self._evictions = 0
        self._failure = None
        self._closed = False
        self._history = [_Eviction() for _ in range(32)]
        self._history_next = self._history_count = 0
        self._charged = _BASE + _NODE
        self._validated_tokens = None
        self._validated_charge = 0
        self._cursor = None
        self._path_cache = None
        self._path_charge = 0

    def _healthy(self):
        if self._closed:
            raise RuntimeError("prefix store is permanently closed")
        if self._failure is not None:
            raise RuntimeError('prefix store is terminally poisoned') from self._failure

    def _poison(self, error):
        if self._failure is None:
            self._failure = error

    def _nodes(self):
        stack = [self._root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def _used(self):
        return self._charged

    def _validate_tokens(self, tokens):
        if tokens is self._validated_tokens:
            return
        _tokens(tokens, self.budget_bytes)
        if type(tokens) is tuple:
            charge = 64 + _TOKEN * len(tokens)
            self._ensure(charge)
            self._charged += charge-self._validated_charge
            self._drop_path()
            self._validated_tokens, self._validated_charge = tokens, charge
            self._cursor = None

    def _drop_path(self):
        self._charged -= self._path_charge
        self._path_cache, self._path_charge = None, 0

    def _prune(self):
        for node in reversed(list(self._nodes())):
            previous = len(node.views)
            node.views[:] = [v for v in node.views if v[0].groups]
            self._charged -= _VIEW*(previous-len(node.views))
            if node is not self._root and not node.views and not node.children:
                self._drop_path()
                before = sys.getsizeof(node.parent.children)
                del node.parent.children[node.label[0]]
                self._charged += sys.getsizeof(node.parent.children)-before
                self._charged -= _NODE + _TOKEN*len(node.label) + sys.getsizeof(node.children)-64
                if self._cursor is node:
                    self._cursor = None
                node.alive = False
                node.parent = None

    def _discard(self, record):
        if record in self._records:
            self._records.remove(record)
        self._charged -= _RECORD + record.tag_bytes + sum(_GROUP+4*len(n)+g.nbytes for n,g in record.groups.items())
        for group in record.groups.values():
            group.data = None
        record.groups.clear()
        record.node = None
        self._prune()

    def _ensure(self, extra):
        if type(extra) is not int or extra < 0:
            raise ValueError('nonnegative byte obligation required')
        try:
            while self._used() + extra > self.budget_bytes:
                if self._path_cache is not None:
                    self._drop_path()
                    if self._used()+extra <= self.budget_bytes:
                        return
                if self._validated_tokens is not None:
                    self._charged -= self._validated_charge
                    self._validated_tokens, self._validated_charge, self._cursor = None, 0, None
                    if self._used()+extra <= self.budget_bytes:
                        return
                best = None
                for record in self._records:
                    for name, group in record.groups.items():
                        if group.published and group.pins == 0:
                            key = (group.used+group.benefit*group.fanout, -group.nbytes,
                                   0 if group.window else 1 if name == 'hidden' else 2, group.generation)
                            if best is None or key < best[0]:
                                best = key, record, name
                if best is None:
                    record = group = None
                    raise PrefixCapacity('all remaining bytes are pinned, reserved or required metadata')
                key, record, name = best
                self._clock = max(self._clock, key[0])
                group = record.groups.pop(name)
                event = self._history[self._history_next]
                self._charged += 4*len(name)-(4*len(event.component) if event.component is not None else 0)
                event.component,event.start,event.stop,event.nbytes = name,record.start,record.stop,group.nbytes
                self._history_next = (self._history_next+1)%32
                self._history_count = min(32,self._history_count+1)
                self._charged -= _GROUP+4*len(name)+group.nbytes
                group.data = None
                del group  # Last unpinned payload reference retires before next grant.
                self._evictions += 1
                if not record.groups:
                    self._records.remove(record)
                    self._charged -= _RECORD+record.tag_bytes
                self._prune()

        except PrefixCapacity:
            raise
        except BaseException as primary:
            self._poison(primary)
            raise

    def _insert(self, tokens, stop, node=None):
        node = self._root if node is None else node
        at = node.depth
        while at < stop:
            child = node.children.get(tokens[at])
            if child is None:
                end = min(stop, at + self.block_tokens)
                child = _Node(tuple(tokens[at:end]), end, node, {}, [])
                before = sys.getsizeof(node.children)
                node.children[child.label[0]] = child
                self._drop_path()
                self._charged += _NODE + _TOKEN*len(child.label) + sys.getsizeof(node.children)-before
            else:
                same = 0
                while same < len(child.label) and at + same < stop and child.label[same] == tokens[at + same]:
                    same += 1
                if same < len(child.label):
                    depth = at + same
                    middle = _Node(child.label[:same], depth, node, {}, [])
                    self._drop_path()
                    child.label = child.label[same:]
                    child.parent = middle
                    middle.children[child.label[0]] = child
                    node.children[middle.label[0]] = middle
                    # Row views split; the immutable backing and its charge do not.
                    retained = []
                    for record, lo, hi in child.views:
                        if lo < depth:
                            middle.views.append((record, lo, min(hi, depth)))
                        if hi > depth:
                            retained.append((record, max(lo, depth), hi))
                    self._charged += _NODE + sys.getsizeof(middle.children)-64 + _VIEW*(len(middle.views)+len(retained)-len(child.views))
                    # The original label bytes move to two nodes without duplication.
                    child.views = retained
                    child = middle
            node, at = child, child.depth
        return node

    def _write_charge(self, tokens, start, stop, group_bytes, tag_bytes):
        fixed = _RECORD+_LEASE+tag_bytes+sum(_GROUP+4*len(k)+v for k,v in group_bytes.items())
        cursor = self._cursor
        if (tokens is self._validated_tokens and cursor is not None and cursor.alive and cursor.depth == start
                and not cursor.children.get(tokens[start])):
            nodes = (stop-start+self.block_tokens-1)//self.block_tokens
            return fixed+_TOKEN*(stop-start)+nodes*(_NODE+_VIEW)+2*sys.getsizeof(cursor.children)+256
        path,longest = self._path(tokens,stop)
        missing = max(0,stop-longest)
        nodes = (missing+self.block_tokens-1)//self.block_tokens
        splits = [node for node,matched in path if node.parent.depth < start < node.depth
                  or node.parent.depth < stop < node.depth or matched < node.depth]
        existing = sum(node.parent.depth < stop and matched > start for node,matched in path)
        return (fixed+_TOKEN*missing+nodes*(_NODE+_VIEW)+len(splits)*(_NODE+2*_VIEW)
                +_VIEW*(existing+sum(len(node.views) for node in splits))
                +2*max((sys.getsizeof(node.children) for node,_matched in path),default=64)+256)

    def begin(self, tokens, start, stop, group_bytes, *, tags=()):
        with self._lock:
            self._healthy()
            self._validate_tokens(tokens)
            tag_bytes = _tags(tags, len(tokens), self.budget_bytes)
            if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(tokens)
                    or stop-start > 1024 or type(group_bytes) is not dict or not group_bytes):
                raise ValueError('bounded nonempty writer interval/groups required')
            if any(type(k) is not str or not k or type(v) is not int or v <= 0 or v > self.budget_bytes for k, v in group_bytes.items()):
                raise ValueError('group names and exact positive storage byte charges required')
            # Recompute after eviction: pruning may change the new-label
            # obligation. No producer effects are possible before begin returns.
            if self._serial > (1 << 63)-1-len(group_bytes):
                raise PrefixCapacity('process-local payload generation exhausted')
            while True:
                obligation = self._write_charge(tokens,start,stop,group_bytes,tag_bytes)
                if self._used()+obligation <= self.budget_bytes:
                    break
                self._ensure(obligation)
            record = _Record(start, stop, {}, _tag_prefix(tags, stop), tag_bytes)
            for name, nbytes in group_bytes.items():
                self._serial += 1
                record.groups[name] = _Group(name, nbytes, self._serial, used=self._clock,
                                             window=name == "window" or name.endswith("_window"),
                                             benefit=stop)
            cleanup_errors = [None, None]
            cleanup_view = filter(lambda value: value is not None, cleanup_errors)
            writer = WriteLease(self, record)
            charge = _RECORD + _LEASE + tag_bytes + sum(_GROUP+4*len(n)+g.nbytes for n,g in record.groups.items())
            try:
                self._writers.append(writer)
                writer._registered = True
                self._charged += charge
                writer._charged = True
                self._records.append(record)
                cursor = self._cursor
                start_node = (cursor if tokens is self._validated_tokens and cursor is not None and cursor.alive and cursor.depth == start
                              else self._insert(tokens, start))
                node = self._insert(tokens, stop, start_node)
                record.node = node
                self._cursor = node if tokens is self._validated_tokens else None
                while node is not self._root and node.depth > start:
                    lo = max(start, node.parent.depth)
                    node.views.append((record, lo, node.depth))
                    self._charged += _VIEW
                    node = node.parent
                if self._used() > self.budget_bytes:
                    raise RuntimeError('metadata reservation underestimated the admitted index')
                return writer
            except BaseException as error:
                # Only a journaled or index-mutating failure is terminal. A
                # rejected registry allocation never authorized a producer.
                if writer._registered or writer._charged:
                    self._poison(error)
                # No GPU effects were possible: the lease has not escaped.
                try:
                    writer.abort()
                except BaseException as cleanup:
                    cleanup_errors[0] = cleanup
                    self._poison(error)
                try:
                    self._drop_path()
                    self._charged -= self._validated_charge
                    self._validated_tokens, self._validated_charge, self._cursor = None, 0, None
                except BaseException as cleanup:
                    cleanup_errors[1] = cleanup
                    self._poison(error)
                raise_failures(error, cleanup_view)

    def reserve_temporary(self, nbytes):
        with self._lock:
            self._healthy()
            if type(nbytes) is not int or nbytes < 0:
                raise ValueError('nonnegative temporary storage charge required')
            self._ensure(nbytes + _LEASE)
            lease = TemporaryLease(self,nbytes)
            errors = [None]
            error_view = filter(lambda value:value is not None,errors)
            charged = self._charged+nbytes+_LEASE
            try:
                self._temporary.append(lease)
                lease._registered = True
                self._charged = charged
                lease._charged = True
                return lease
            except BaseException as primary:
                if lease._registered or lease._charged:
                    self._poison(primary)
                try:
                    lease.close()  # No caller could have begun a GPU consumer.
                except BaseException as cleanup:
                    errors[0] = cleanup
                    self._poison(primary)
                raise_failures(primary,error_view)

    def _scan_path(self, tokens, limit):
        result, node, at = [], self._root, 0
        while at < limit:
            child = node.children.get(tokens[at])
            if child is None:
                break
            same = 0
            while same < len(child.label) and at+same < limit and child.label[same] == tokens[at+same]:
                same += 1
            result.append((child, at+same))
            at += same
            if same != len(child.label):
                break
            node = child
        return result, at

    def _path(self, tokens, limit=None, *, start=0):
        limit = len(tokens) if limit is None else limit
        cache = self._path_cache
        if cache is None or tokens is not self._validated_tokens:
            # An immutable input is validated/charged once. Scan once through
            # its currently resident path; later batch ranges use binary cuts.
            scan_limit = len(tokens) if tokens is self._validated_tokens else limit
            path, longest = self._scan_path(tokens, scan_limit)
            charge = 256*len(path)+128
            if tokens is self._validated_tokens and self._used()+charge <= self.budget_bytes:
                self._drop_path()
                cache = (tuple(path), tuple(end for _node,end in path), longest)
                self._path_cache, self._path_charge = cache, charge
                self._charged += charge
            else:
                return [(n,min(end,limit)) for n,end in path if n.parent.depth < limit and end > start], min(longest,limit)
        path, ends, longest = cache
        first = bisect_right(ends,start)
        last = bisect_left(ends,limit)+1
        return [(node,min(end,limit)) for node,end in path[first:last]], min(longest,limit)

    def _coverage(self, path, group, begin, cut, tags):
        """One sweep per group, not one path scan for every candidate token."""
        pieces = []
        for node, matched in path:
            for record, lo, hi in node.views:
                owner = record.groups.get(group)
                if owner is None or not owner.published:
                    continue
                left, right = record.tags, _tag_prefix(tags, record.stop)
                limit = record.stop
                for i in range(max(len(left), len(right))):
                    a = left[i] if i < len(left) else None
                    b = right[i] if i < len(right) else None
                    if a is None or b is None or not _tags_equal((a,),(b,)):
                        limit = min(a[0] if a else limit, b[0] if b else limit)
                        break
                lo, hi = max(lo, begin), min(hi, matched, cut, limit)
                if lo < hi:
                    pieces.append((lo, hi, owner, record.start))
        pieces.sort(key=lambda row: (row[0], row[1], row[2].generation))
        result, active, index, cursor = [], [], 0, 0
        while index < len(pieces) or active:
            if not active:
                cursor = pieces[index][0]
            while index < len(pieces) and pieces[index][0] <= cursor:
                lo, hi, owner, origin = pieces[index]
                heapq.heappush(active, (-owner.generation, index, hi, origin, owner))
                index += 1
            while active and active[0][2] <= cursor:
                heapq.heappop(active)
            if not active:
                continue
            _generation, _ordinal, end, origin, owner = active[0]
            if index < len(pieces):
                end = min(end, pieces[index][0])
            offset = cursor-origin
            if result and result[-1][2] is owner and result[-1][1] == cursor and result[-1][3]+result[-1][1]-result[-1][0] == offset:
                previous = result[-1]
                result[-1] = previous[0], end, owner, previous[3]
            else:
                result.append((cursor, end, owner, offset))
            cursor = end
        return tuple(result)

    @staticmethod
    def _intervals(rows):
        result = []
        for lo, hi, _owner, _offset in rows:
            if result and result[-1][1] == lo:
                result[-1] = result[-1][0], hi
            else:
                result.append((lo, hi))
        return result

    @staticmethod
    def _clip(rows, begin, end):
        return tuple((max(lo,begin), min(hi,end), owner, offset+max(begin-lo,0))
                     for lo,hi,owner,offset in rows if max(lo,begin) < min(hi,end))

    def _select(self, tokens, windows, tags, require_end_group):
        if tokens is not self._validated_tokens:
            _tokens(tokens, self.budget_bytes)
        _tags(tags, len(tokens), self.budget_bytes)
        if type(windows) is not dict or not windows or any(type(k) is not str or (v is not None and (type(v) is not int or v <= 0)) for k,v in windows.items()):
            raise ValueError('explicit full/window group requirements required')
        path, longest = self._path(tokens)
        required = dict(windows)
        if require_end_group is not None:
            required[require_end_group] = 1
        rows = {g:self._coverage(path,g,0,longest,tags) for g in required}
        intervals = {g:self._intervals(r) for g,r in rows.items()}
        cut = longest
        # Full groups cap at the first missing interval. Window groups define
        # disjoint valid-cut intervals [lo+window, hi], with a zero-root case.
        for group, window in windows.items():
            if window is None:
                found = intervals[group]
                cut = min(cut, found[0][1] if found and found[0][0] == 0 else 0)
            else:
                for _lo,_hi,owner,_offset in rows[group]:
                    owner.window = True
        while cut > 0:
            atomic = next((begin for begin,end,_identity in tags if begin < cut < end), None)
            if atomic is not None:
                cut = atomic
                continue
            changed = False
            for group, window in windows.items():
                if window is None:
                    continue
                valid = max((min(cut,hi) for lo,hi in intervals[group]
                             if (lo == 0 or lo+window <= min(cut,hi))), default=0)
                if valid < cut:
                    cut, changed = valid, True
                    break
            if changed:
                continue
            if cut == len(tokens) and require_end_group is not None:
                if not any(lo <= cut-1 and hi >= cut for lo,hi in intervals[require_end_group]):
                    cut -= 1
                    continue
            coverage = {g:self._clip(rows[g],0 if w is None else max(0,cut-w),cut) for g,w in windows.items()}
            if cut == len(tokens) and require_end_group is not None:
                coverage[require_end_group] = self._clip(rows[require_end_group],cut-1,cut)
            return cut, coverage
        return None

    def _validate_lookup(self,tokens,windows,tags,require_end_group,span):
        if tokens is not self._validated_tokens:
            _tokens(tokens,self.budget_bytes)
        _tags(tags,len(tokens),self.budget_bytes)
        if span is not None:
            group,start,stop = span
            if (type(group) is not str or not group or type(start) is not int or type(stop) is not int
                    or not 0 <= start < stop <= len(tokens)):
                raise ValueError('exact bounded independent group range required')
        elif (type(windows) is not dict or not windows
              or any(type(g) is not str or not g or (w is not None and (type(w) is not int or w <= 0)) for g,w in windows.items())
              or (require_end_group is not None and (type(require_end_group) is not str or not require_end_group))):
            raise ValueError('explicit full/window/end group requirements required')

    def _borrow(self, tokens, windows, tags, require_end_group, span=None):
        with self._lock:
            self._healthy()
            self._validate_lookup(tokens,windows,tags,require_end_group,span)
            try:
                while True:
                    selected = (self._select(tokens,windows,tags,require_end_group) if span is None
                                else self._range_select(tokens,*span,tags))
                    if selected is None:
                        return None
                    needed = _LEASE + _SPAN*sum(map(len,selected[1].values()))
                    if self._used()+needed <= self.budget_bytes:
                        break
                    # Never advertise bytes as reclaimed while this planning
                    # result still retains the old immutable payload owners.
                    selected = None
                    self._ensure(needed)
                length, coverage = selected
                groups = tuple(sorted(set(row[2] for rows in coverage.values() for row in rows), key=lambda g:g.generation))
                old_pins = tuple(g.pins for g in groups)
                new_pins = tuple(count+1 for count in old_pins)
                old_charge, new_charge = self._charged,self._charged+needed
                errors = [None]*(len(groups)+1)
                error_view = filter(lambda value:value is not None,errors)
                lease = ReadLease(self,length,coverage,groups)
                registered = False
                try:
                    self._readers.append(lease)
                    registered = True
                    self._charged = new_charge
                    for group,count in zip(groups,new_pins):
                        group.pins = count
                        group.used = self._clock
                        group.benefit = max(group.benefit,length)
                    return lease
                except BaseException as primary:
                    self._poison(primary)
                    for ordinal,(group,old) in enumerate(zip(groups,old_pins)):
                        try:
                            group.pins = old
                        except BaseException as cleanup:
                            errors[ordinal] = cleanup
                    try:
                        if registered:
                            self._readers.remove(lease)
                            self._charged = old_charge
                        lease._coverage = lease._groups = None
                        lease._closed = True
                    except BaseException as cleanup:
                        errors[-1] = cleanup
                    raise_failures(primary,error_view)

            except PrefixCapacity:
                # Only the owned explicit capacity decision is recoverable;
                # _ensure clears loop payload borrows before that raise.
                raise
            except BaseException as primary:
                self._poison(primary)
                raise

    def match(self, tokens, windows, *, tags=(), require_end_group='hidden'):
        return self._borrow(tokens,windows,tags,require_end_group)

    def _range_select(self, tokens, group, start, stop, tags):
        self._validate_tokens(tokens)
        _tags(tags,len(tokens),self.budget_bytes)
        if (type(group) is not str or not group or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= len(tokens)):
            raise ValueError('exact bounded independent group range required')
        cursor = self._cursor
        if (tokens is self._validated_tokens and cursor is not None and cursor.alive and cursor.depth == start
                and not cursor.children.get(tokens[start])):
            return None
        path, longest = self._path(tokens,stop,start=start)
        if longest < stop:
            return None
        rows = self._coverage(path,group,start,stop,tags)
        if self._intervals(rows) != [(start,stop)]:
            return None
        return stop,{group:rows}

    def pin_range(self, tokens, group, start, stop, *, tags=()):
        """Pin independently resident rows, not a complete continuation State."""
        return self._borrow(tokens,None,tags,None,(group,start,stop))

    def identity(self, tokens, windows, *, tags=()):
        with self._lock:
            self._healthy()
            self._validate_lookup(tokens,windows,tags,'hidden',None)
            try:
                selected = self._select(tokens, windows, tags, 'hidden')
                if selected is None or selected[0] != len(tokens):
                    return None
                length, coverage = selected
                return length, tuple((name, tuple((g.generation, offset, hi-lo) for lo,hi,g,offset in spans))
                                     for name,spans in sorted(coverage.items()))

            except BaseException as primary:
                self._poison(primary)
                raise

    def clear(self, *, preserve_temporary=()):
        """Retired data clear; only explicitly proved host metadata may stay.

        The adapter must first retire/reap ALL GPU temporaries. This Torch-free
        index can prove exact lease ownership, not that opaque bytes are host.
        Default clear and final close still require every temporary retired.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("prefix store is permanently closed")
            if self._writers or self._readers:
                raise RuntimeError('cannot clear live prefix read/write leases')
            if (type(preserve_temporary) is not tuple
                    or len(preserve_temporary) != len(self._temporary)
                    or len({id(t) for t in preserve_temporary}) != len(preserve_temporary)
                    or any(type(t) is not TemporaryLease or t._store is not self or t._closed
                           or t not in self._temporary for t in preserve_temporary)):
                raise RuntimeError('clear requires every active host metadata lease explicitly')
            new_root = _Node((),0,None,{},[])
            nodes = list(self._nodes())  # Prepare before the first release.
            self._drop_path()
            for record in self._records:
                for group in record.groups.values():
                    group.data = None
                record.groups.clear()
                record.node = None
            for node in nodes:
                node.views.clear()
                node.children.clear()
                node.parent = None
                node.alive = False
            self._root = new_root
            self._records.clear()
            self._validated_tokens, self._validated_charge, self._cursor = None,0,None
            for event in self._history:
                event.component = None
            self._history_next = self._history_count = 0
            self._charged = _BASE+_NODE+sum(_LEASE+t.nbytes for t in self._temporary)

    def close(self):
        """Positive caller retirement, final data release, never reopen."""
        with self._lock:
            if self._closed:
                return
            self.clear()
            self._failure = None
            self._closed = True

    @property
    def stats(self):
        with self._lock:
            try:
                groups = [g for r in self._records for g in r.groups.values()]
                return {'budget_bytes': self.budget_bytes, 'charged_bytes': self._used(),
                        'eviction_policy': EVICTION_POLICY,
                        'payload_bytes': sum(g.nbytes for g in groups if g.published),
                        'reserved_payload_bytes': sum(g.nbytes for g in groups if not g.published),
                        'chunks': len(self._records),
                        'indexed_unique_tokens': sum(len(n.label) for n in self._nodes()),
                        'groups': len(groups),
                        'nodes': sum(1 for _ in self._nodes()), 'pins': sum(g.pins for g in groups),
                        'read_leases': len(self._readers), 'write_leases': len(self._writers),
                        'temporary_leases': len(self._temporary), 'temporary_bytes': sum(t.nbytes for t in self._temporary),
                        'evictions': self._evictions, 'poisoned': self._failure is not None, 'closed': self._closed,
                        'eviction_history': tuple({'component':e.component,'start':e.start,'stop':e.stop,'bytes':e.nbytes,'reason':e.reason}
                            for e in [self._history[(self._history_next-self._history_count+i)%32] for i in range(self._history_count)])}
            except BaseException as primary:
                self._poison(primary)
                raise
