"""Bounded CPU counters at explicit FAST cohort boundaries, never in its core."""
import hashlib
import json
import os
from pathlib import Path
import time

from tensorfold.cleanup import raise_failures, rollback

PROC = Path('/proc')
CGROUP = Path('/sys/fs/cgroup')
MAX_TIDS = 4096
PHASES = 12
LIVE_BYTES = 2 << 20
DESCRIPTOR_BYTES = 64 << 10
CPU_FIELDS = ('usage_usec', 'user_usec', 'system_usec', 'nr_periods', 'nr_throttled', 'throttled_usec')
FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
DIR_FLAGS = FLAGS | os.O_DIRECTORY


def reserved_bytes(phase_bytes):
    if type(phase_bytes) is not int or phase_bytes <= 0:
        raise ValueError("CPU counter byte allowance must be a positive integer")
    return PHASES * phase_bytes + LIVE_BYTES + DESCRIPTOR_BYTES


def fd_operation(acquire, operation, *, missing=False):
    errors = [None]
    primary = cause = context = None
    suppressed, result = False, None
    try:
        fd = acquire()
    except (FileNotFoundError, ProcessLookupError):
        if missing:
            return None
        raise
    try:
        result = operation(fd)
    except BaseException as error:
        primary = error
        cause = BaseException.__cause__.__get__(error)
        context = BaseException.__context__.__get__(error)
        suppressed = BaseException.__suppress_context__.__get__(error)
    try:
        os.close(fd)
    except BaseException as error:
        errors[0] = error
    if primary is not None:
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        BaseException.__suppress_context__.__set__(primary, suppressed)
    if errors[0] is not None:
        raise_failures(primary, errors)
    if primary is not None:
        raise_failures(primary, ())
    return result


def read_at(directory, name, limit=4096, *, missing=False):
    def read(fd):
        try:
            return _read_fd(fd, limit)
        except (FileNotFoundError, ProcessLookupError):
            if missing:
                return None
            raise
    return fd_operation(lambda: os.open(name, FLAGS, dir_fd=directory), read, missing=missing)


def path_bytes(path, limit=4096):
    return fd_operation(lambda: os.open(path, FLAGS), lambda fd: _read_fd(fd, limit))


def _read_fd(fd, limit):
    chunks, size = [], 0
    while True:
        raw = os.read(fd, limit + 1 - size)
        if not raw:
            return b''.join(chunks)
        chunks.append(raw)
        size += len(raw)
        if size > limit:
            raise ValueError('CPU counter input exceeds bounded size')


def parse_stat(raw, expected):
    head, rest = raw.rsplit(b') ', 1)
    if int(head.split(b' (', 1)[0]) != expected:
        raise ValueError('proc stat PID identity differs')
    fields = rest.split()
    if len(fields) < 22:
        raise ValueError('incomplete proc stat')
    values = [int(fields[i]) for i in (11, 12, 19, 21)]
    if any(value < 0 for value in values):
        raise ValueError('invalid CPU/process counters')
    return dict(zip(('utime_ticks', 'stime_ticks', 'start_ticks', 'rss_pages'), values, strict=True))


def cgroup_values(directory):
    raw = read_at(directory, 'cpu.stat')
    rows = {}
    for line in raw.splitlines():
        key, value = line.split()
        key = key.decode('ascii')
        if key in rows or int(value) < 0:
            raise ValueError('invalid cumulative cgroup CPU counter')
        rows[key] = int(value)
    if any(name not in rows for name in CPU_FIELDS):
        raise ValueError('CPU counter diagnostic requires cgroup-v2 CPU accounting fields')
    quota = read_at(directory, 'cpu.max', 128).decode('ascii').strip().split()
    if len(quota) != 2 or quota[0] != 'max' and not quota[0].isdecimal() or not quota[1].isdecimal():
        raise ValueError('invalid cgroup CPU quota provider')
    if int(quota[1]) <= 0 or quota[0] != 'max' and int(quota[0]) <= 0:
        raise ValueError('nonpositive cgroup CPU quota provider')
    pids_max = read_at(directory, 'pids.max', 128).decode('ascii').strip()
    if not pids_max.isdecimal() or not 0 < int(pids_max) <= MAX_TIDS:
        raise ValueError('CPU counter census requires the held bounded PID controller')
    members = {int(x) for x in read_at(directory, 'cgroup.procs', 65536).split()}
    return {'cpu_stat': rows, 'cpu_max': quota, 'pids_max': int(pids_max), 'members': sorted(members)}


def preflight():
    def inspect(directory):
        values = cgroup_values(directory)
        if os.getpid() not in values['members']:
            raise ValueError('CPU cgroup root does not own this model process; no ancestor-scope fallback')
        return {'cgroup_inode': os.fstat(directory).st_ino, 'cgroup_device': os.fstat(directory).st_dev,
                'cpu_max': values['cpu_max'], 'unified_membership': path_bytes(PROC / 'self/cgroup').decode('ascii').strip(),
                'source': str(CGROUP), 'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'),
                'scope': 'cgroup-v2 namespace root, validated membership; no counter reset or settings change'}
    return fd_operation(lambda: os.open(CGROUP, DIR_FLAGS), inspect)


class Collector:
    def __init__(self, output, *, phase_bytes, bootstrap_sha):
        reserved_bytes(phase_bytes)
        self.output, self.phase_bytes = Path(output), phase_bytes
        self.fds, self.specs, self.labels = [None] * 10, [], []
        self.closed = False
        self.close_errors = [None] * 10
        self.close_attempted = False
        self.close_failure = None
        close = self.close  # Reserve the rollback callback before acquisition.
        try:
            provider = preflight()
            self.fds[0] = os.open(CGROUP, DIR_FLAGS)
            self.cgroup_identity = (os.fstat(self.fds[0]).st_dev, os.fstat(self.fds[0]).st_ino)
            self.quota = provider['cpu_max']
            self.membership = provider['unified_membership']
            parent = None
            for rank in range(8):
                raw = path_bytes(self.output / f'rank-{rank}-affinity-bootstrap.json', 256 << 10)
                receipt = json.loads(raw)
                if (receipt['rank'] != rank or receipt['mode'] != 'bind' or receipt['bootstrap_sha256'] != bootstrap_sha
                        or receipt['identity']['pid'] != receipt['identity']['session']):
                    raise ValueError('CPU counters require actual held required-BIND bootstrap identities')
                identity = receipt['identity']
                pid, started = identity['pid'], identity['start_ticks']
                if type(pid) is not int or pid <= 0 or type(started) is not int or started <= 0:
                    raise ValueError('invalid held process identity')
                if parent is not None and parent != identity['parent_pid']:
                    raise ValueError('CPU counter ranks do not share their owned supervisor')
                parent = identity['parent_pid']
                self.specs.append((rank, pid, started))
            if parent in [s[1] for s in self.specs] or type(parent) is not int or parent <= 0:
                raise ValueError('invalid CPU counter supervisor identity')
            parent_start = parse_stat(path_bytes(PROC / str(parent) / 'stat'), parent)['start_ticks']
            self.specs.append(('supervisor', parent, parent_start))
            for index, (_role, pid, started) in enumerate(self.specs, 1):
                self.fds[index] = os.open(PROC / str(pid), DIR_FLAGS)
                if parse_stat(read_at(self.fds[index], 'stat'), pid)['start_ticks'] != started:
                    raise ValueError('CPU process PID was reused before its directory was pinned')
            self.clock = {'clock': 'CLOCK_MONOTONIC', 'implementation': time.get_clock_info('monotonic').implementation, 'resolution_s': time.get_clock_info('monotonic').resolution,
                          'ticks_per_second': os.sysconf('SC_CLK_TCK'), 'page_size': os.sysconf('SC_PAGE_SIZE'),
                          'boot_id': path_bytes(PROC / 'sys/kernel/random/boot_id', 128).decode('ascii').strip(),
                          'observer_pid': os.getpid(), 'observer_tid': __import__('threading').get_native_id()}
            sched = PROC / 'sys/kernel/sched_schedstats'
            sched_raw = fd_operation(lambda: os.open(sched, FLAGS), lambda fd: _read_fd(fd, 32), missing=True)
            self.clock['sched_schedstats'] = (sched_raw.decode('ascii').strip() if sched_raw is not None
                else 'unavailable; schedstat values may be unsupported/disabled')
        except BaseException as primary:
            rollback(self, primary, close)

    def close(self):
        if self.closed:
            return
        if self.close_attempted:
            raise self.close_failure
        self.close_attempted = True
        for index, fd in enumerate(self.fds):
            if fd is None:
                continue
            try:
                os.close(fd)
            except BaseException as error:
                self.close_errors[index] = error
                if self.close_failure is None:
                    self.close_failure = error
            else:
                self.fds[index] = None
        self.closed = self.close_failure is None
        try:
            errors = [e for e in self.close_errors if e is not None]
        except BaseException as diagnostic:
            if self.close_failure is not None:
                raise self.close_failure from diagnostic
            raise
        if errors:
            raise_failures(None, errors)
        self.closed = True

    def capture(self, count, phase):
        label = f'n{count:02d}-{phase}'
        if self.closed or count not in (1, 2, 4, 8, 16, 24) or phase not in ('before', 'after') or label in self.labels or len(self.labels) >= PHASES:
            raise ValueError('invalid or repeated bounded CPU counter phase')
        self.labels.append(label)
        path = self.output / ('cpu-counters-' + label + '.jsonl')
        started = time.monotonic_ns()
        digest, size, total_tids = hashlib.sha256(), 0, 0
        def write(fd):
            nonlocal size, total_tids
            def row(value):
                nonlocal size
                raw = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
                if size + len(raw) > self.phase_bytes:
                    raise ValueError('CPU counter phase exceeds prepaid serialized allowance')
                offset = 0
                while offset < len(raw):
                    written = os.write(fd, memoryview(raw)[offset:])
                    if written <= 0:
                        raise OSError('short CPU counter output write')
                    offset += written
                digest.update(raw)
                size += len(raw)
            row({'schema': 'tensorfold.cpu-cohort-counters-v1', 'phase': label, 'clock': self.clock,
                 'units': {'proc_cpu': 'SC_CLK_TCK ticks', 'schedstat': ['runtime_ns', 'runqueue_wait_ns', 'timeslices'],
                           'cgroup_cpu': 'microseconds; nr_* are counts', 'rss': 'pages; not CPU time'},
                 'scope': 'outside-core wholecohort endpoints, not exact10s CPU utilization'})
            cg_start = time.monotonic_ns()
            if (os.fstat(self.fds[0]).st_dev, os.fstat(self.fds[0]).st_ino) != self.cgroup_identity:
                raise ValueError('CPU cgroup identity changed')
            group = cgroup_values(self.fds[0])
            if group['cpu_max'] != self.quota:
                raise ValueError('CPU cgroup quota changed during diagnostic')
            row({'kind': 'cgroup', 'identity': self.cgroup_identity, 'started_ns': cg_start,
                 'ended_ns': time.monotonic_ns(), **group})
            for index, (role, pid, expected_start) in enumerate(self.specs, 1):
                directory = self.fds[index]
                first = time.monotonic_ns()
                raw = read_at(directory, 'stat', missing=True)
                membership = read_at(directory, 'cgroup', missing=True)
                if raw is None or membership is None:
                    row({'kind': 'process', 'role': role, 'pid': pid, 'expected_start_ticks': expected_start,
                         'status': 'vanished_before_read', 'started_ns': first, 'ended_ns': time.monotonic_ns()})
                    continue
                process = parse_stat(raw, pid)
                if process['start_ticks'] != expected_start or membership.decode('ascii').strip() != self.membership:
                    raise ValueError('CPU process identity or cgroup membership changed')
                row({'kind': 'process', 'role': role, 'pid': pid, 'started_ns': first,
                     'ended_ns': time.monotonic_ns(), **process})
                def list_tasks(task):
                    try:
                        return sorted(int(x) for x in os.listdir(task) if x.isdecimal())
                    except (FileNotFoundError, ProcessLookupError):
                        return None
                tids = fd_operation(lambda: os.open('task', DIR_FLAGS, dir_fd=directory), list_tasks, missing=True)
                if tids is None:
                    row({'kind': 'task_census', 'pid': pid, 'status': 'vanished_before_list'})
                    continue
                total_tids += len(tids)
                if total_tids > MAX_TIDS:
                    raise ValueError('CPU census exceeds held container total PID limit')
                for tid in tids:
                    sample_start = time.monotonic_ns()
                    def read_thread(thread_dir):
                        first_raw = read_at(thread_dir, 'stat', missing=True)
                        if first_raw is None:
                            return None
                        before = parse_stat(first_raw, tid)
                        raw_sched = read_at(thread_dir, 'schedstat', 128, missing=True)
                        if raw_sched is None:
                            schedstat, status = None, 'unavailable'
                        else:
                            schedstat = [int(x) for x in raw_sched.split()]
                            if len(schedstat) != 3 or any(x < 0 for x in schedstat):
                                raise ValueError('invalid schedstat provider')
                            status = 'raw; capability flag recorded, zero does not prove zero wait'
                        last_raw = read_at(thread_dir, 'stat', missing=True)
                        if last_raw is None:
                            return None
                        after = parse_stat(last_raw, tid)
                        if before['start_ticks'] != after['start_ticks']:
                            raise ValueError('CPU TID identity changed during read')
                        return {'stat': after, 'schedstat': schedstat, 'schedstat_status': status}
                    value = fd_operation(lambda: os.open('task/' + str(tid), DIR_FLAGS, dir_fd=directory),
                                         read_thread, missing=True)
                    record = {'kind': 'thread', 'pid': pid, 'tid': tid, 'started_ns': sample_start,
                              'ended_ns': time.monotonic_ns()}
                    if value is None:
                        record['status'] = 'vanished_before_read'
                    else:
                        record.update(value)
                    row(record)
                last = read_at(directory, 'stat', missing=True)
                if last is None:
                    row({'kind': 'process_end', 'pid': pid, 'status': 'vanished_after_census'})
                elif parse_stat(last, pid)['start_ticks'] != expected_start:
                    raise ValueError('CPU process identity changed after census')
        fd_operation(lambda: os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600), write)
        return {'path': path.name, 'bytes': size, 'sha256': digest.hexdigest(), 'count': count, 'phase': phase,
                'started_monotonic_ns': started, 'ended_monotonic_ns': time.monotonic_ns(),
                'TIDs_seen': total_tids, 'scope': 'outside10s core; wholecohort CPU endpoint, not exactcoreCPU'}
