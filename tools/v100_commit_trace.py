"""Finite diagnostic commit-batch timestamps; never a performance observer.

The caller admits ``resources`` before installation and drains the decoder
before close. Records contain integers only, not tokens, Tensors or Streams.
Several MTP tokens may share one observed batch; no individual token timing is
inferred. Diagnostic exhaustion never suppresses or retries model work.
"""
from array import array
import sys
import time


GRAPH_COLUMNS = ('role', 'total_rows', 'participants', 'outcome', 'reason', 'replay_calls', 'replay_returns')
GRAPH_ROLES = ('verify', 'mtp')
GRAPH_OUTCOMES = ('miss', 'hit', 'error')
GRAPH_REASONS = ('replay-returned', 'not-captured', 'device-greedy', 'rotor', 'media',
                 'error-before-replay', 'error-after-replay-call', 'closed-owner', 'failed-owner', 'invalid-metadata')


class _ReplayWitness:
    __slots__ = ('graph', 'trace')

    def __init__(self, graph, trace):
        self.graph, self.trace = graph, trace

    def replay(self):
        trace = self.trace
        if trace._graph_active:
            trace._replay_calls += 1
        else:
            trace._fail('replay-outside-attempt', 'RuntimeError')
        result = self.graph.replay()  # Exactly once; no new CUDA operation or synchronization.
        if trace._graph_active:
            trace._replay_returns += 1
        return result

    def reset(self):
        return self.graph.reset()  # Existing Engine retirement still owns this operation.


COLUMNS = ('stream_sid', 'begin_ns', 'end_ns', 'committed_before', 'committed_after', 'committed_delta')


def resources(max_events, *, graph_attempts=False):
    if type(graph_attempts) is not bool:
        raise ValueError('graph attempt selection must be explicit boolean')
    columns = GRAPH_COLUMNS if graph_attempts else COLUMNS
    if type(max_events) is not int or not 0 < max_events <= sys.maxsize // (len(columns) * 8):
        raise ValueError('finite positive bounded diagnostic event count required')
    if graph_attempts:
        # At most one proxy/binding tuple per event-credit slot, including the
        # preallocated reference slot and a bounded native index integer.
        binding_bytes = (sys.getsizeof(_ReplayWitness(None, None)) + sys.getsizeof((None, 0, None, None))
                         + sys.getsizeof(sys.maxsize) + (sys.getsizeof([None]) - sys.getsizeof([])))
        return {'maximum_events': max_events, 'maximum_graph_bindings': max_events,
                'retained_event_bytes': max_events * (len(columns) * 8 + binding_bytes),
                'serialization_bytes': max_events * (len(columns) * 160 + 1024) + 16384,
                'graph_binding_slot_bytes': binding_bytes,
                'scope': 'diagnostic host rows/bindings/histogram storage;no GPU allowance'}
    return {'maximum_events': max_events, 'retained_event_bytes': max_events * len(COLUMNS) * 8,
            # Native signed integers take at most20 digits. Reserve a second
            # materialization of Python rows plus bounded JSON/failure metadata.
            'serialization_bytes': max_events * (len(COLUMNS) * 80 + 256) + 8192,
            'scope': 'diagnostic host storage; no model workload or GPU allowance'}


class CommitTrace:
    def __init__(self, max_events, *, clock=None, marker=None, graph_attempts=False):
        if graph_attempts and marker is not None:
            raise ValueError('graph attempt mode does not issue CUDA/NVTX markers')
        self.resources = resources(max_events, graph_attempts=graph_attempts)
        self._graph_attempts = graph_attempts
        self._columns = GRAPH_COLUMNS if graph_attempts else COLUMNS
        self._rows = array('q', [0]) * (max_events * len(self._columns))
        self._graph_bindings = [None] * max_events if graph_attempts else None
        self._graph_binding_count = 0
        self._graph_bound = None
        self._graph_active = False
        self._replay_calls = self._replay_returns = 0
        self._clock = time.monotonic_ns if clock is None else clock
        self._marker = marker
        self._count = 0
        self._failure = None
        self._failure_record = {'stage': None, 'error_type': None}
        self._decoder = self._original = self._wrapper = None
        self._had_take = False
        self._previous_take = None
        self._closed = False
        self._anchors = {'host_monotonic_ns': time.monotonic_ns(), 'host_wall_ns': time.time_ns(),
                         'monotonic_resolution_s': time.get_clock_info('monotonic').resolution}

    def _fail(self, stage, error_type):
        if self._failure is None:
            self._failure_record['stage'] = stage
            self._failure_record['error_type'] = (error_type if len(error_type) <= 128
                                                   else 'oversized-exception-type')
            self._failure = self._failure_record

    def install(self, decoder):
        if self._closed or self._decoder is not None:
            raise RuntimeError('commit trace has one installation lifetime')
        if self._graph_attempts:
            return self._install_graphs(decoder)
        original = decoder._take
        if not callable(original):
            raise ValueError('commit trace requires the actual callable decoder boundary')
        self._had_take = '_take' in vars(decoder)
        self._previous_take = vars(decoder).get('_take')
        self._decoder, self._original = decoder, original

        def take(stream, new):
            before = begin = None
            if self._failure is None:
                try:
                    before, begin = stream.committed_output_tokens, self._clock()
                except BaseException as error:
                    self._fail('before-observation', type(error).__name__)
            try:
                result = original(stream, new)  # Always exactly once, including exhaustion.
            except BaseException as error:
                try:
                    self._fail('model-take', type(error).__name__)
                except BaseException as diagnostic:
                    raise error from diagnostic
                raise
            if self._failure is None:
                try:
                    end, after, sid = self._clock(), stream.committed_output_tokens, stream.sid
                    row = (sid, begin, end, before, after, after - before)
                    if (any(type(v) is not int or not 0 <= v <= sys.maxsize for v in row)
                            or end < begin):
                        raise ValueError('invalid commit observation metadata')
                    if self._count == self.resources['maximum_events']:
                        self._fail('event-capacity', 'OverflowError')
                    else:
                        offset = self._count * len(COLUMNS)
                        for column, value in enumerate(row):
                            self._rows[offset + column] = value
                        self._count += 1
                        if self._marker is not None:
                            self._marker(sid, row[-1], self._count)
                except BaseException as error:
                    self._fail('after-observation', type(error).__name__)
            return result

        self._wrapper = take
        decoder._take = take
        return self

    def close(self):
        if self._graph_attempts:
            return self._close_graphs()
        if not self._closed:
            if self._decoder is not None:
                if self._decoder._take is not self._wrapper:
                    raise RuntimeError('commit observer binding changed before drained retirement')
                if self._had_take:
                    self._decoder._take = self._previous_take
                else:
                    del self._decoder._take
            self._decoder = self._original = self._wrapper = self._previous_take = None
            self._closed = True
        return self.report()

    def report(self):
        if self._graph_attempts:
            return self._graph_report()
        width = len(COLUMNS)
        return {'schema': 'tensorfold.commit-batch-diagnostic-v1', 'closed': self._closed,
                'complete': self._closed and self._failure is None, 'failure': self._failure,
                'diagnostic_only': True, 'P3_valid': False,
                'resources': dict(self.resources), 'clock_anchors': dict(self._anchors),
                'columns': list(COLUMNS),
                'events': [list(self._rows[i:i + width]) for i in range(0, self._count * width, width)],
                'scope': 'Host intervals around committed model-output batches; includes publication/backpressure. '
                         'Not GPU completion, per-token spacing, proposal acceptance or P3 evidence.'}


    def _install_graphs(self, decoder):
        graphs = getattr(decoder, 'graphs', None)
        bound = None if graphs is None else getattr(graphs, 'bound', None)
        if (bound is None or bound.closed or bound.preparing or not callable(bound.forward)
                or set(bound.serving_hits) != set(GRAPH_ROLES) or set(bound.serving_misses) != set(GRAPH_ROLES)):
            raise ValueError('graph attempt diagnostic requires the actual prepared idle exact graph owner')
        if any(type(pieces.graphs) is not list for table in (bound.owner.verify, bound.owner.mtp) for pieces in table.values()):
            raise ValueError('graph diagnostic requires the original mutable piece lists')
        count = sum(len(pieces.graphs) for table in (bound.owner.verify, bound.owner.mtp) for pieces in table.values())
        if count > self.resources['maximum_graph_bindings']:
            raise ValueError('graph bindings exceed admitted diagnostic event credit')
        self._decoder, self._graph_bound, self._original = decoder, bound, bound.forward
        self._had_take, self._previous_take = 'forward' in vars(bound), vars(bound).get('forward')
        try:
            for table in (bound.owner.verify, bound.owner.mtp):
                for pieces in table.values():
                    for index, graph in enumerate(pieces.graphs):
                        witness = _ReplayWitness(graph, self)
                        self._graph_bindings[self._graph_binding_count] = (pieces, index, graph, witness)
                        self._graph_binding_count += 1
                        pieces.graphs[index] = witness
            def attempt(kind, segs, **kwargs):
                original = self._original
                if self._graph_active:
                    self._fail('reentrant-graph-attempt', 'RuntimeError')
                    return original(kind, segs, **kwargs)
                role = rows = participants = 0
                reason = 9
                before_hit = before_miss = 0
                self._graph_active = True
                self._replay_calls = self._replay_returns = 0
                if self._failure is None:
                    try:
                        role = GRAPH_ROLES.index(kind)
                        participants, rows = len(segs), segs[-1][2]
                        if type(rows) is not int or not 0 < participants <= rows <= sys.maxsize:
                            raise ValueError('invalid graph shape metadata')
                        before_hit, before_miss = bound.serving_hits[kind], bound.serving_misses[kind]
                        source = bound.dec.buf if kind == 'verify' else bound.dec.mbuf
                        reason = (2 if kwargs.get('mtp_positions') is not None or kwargs.get('device_tokens') is not None
                                  else 3 if source.rotor_lookup else 4 if any(state.image_positions is not None for state, _, _ in segs)
                                  else 1)
                    except BaseException as error:
                        self._fail('graph-before-observation', type(error).__name__)
                try:
                    result = original(kind, segs, **kwargs)  # Always exactly once, including exhaustion.
                except BaseException as primary:
                    try:
                        if self._failure is None:
                            reason = 6 if self._replay_calls else 7 if bound.closed else 8 if bound.owner.failure is not None else 5
                            self._append_graph(role, rows, participants, 2, reason)
                            self._fail('model-graph-attempt', type(primary).__name__)
                    except BaseException as diagnostic:
                        self._fail('graph-error-observation', type(diagnostic).__name__)
                        raise primary from diagnostic
                    raise
                else:
                    if self._failure is None:
                        try:
                            hits, misses = bound.serving_hits[kind] - before_hit, bound.serving_misses[kind] - before_miss
                            if (bound.preparing or (hits, misses) not in ((1, 0), (0, 1))
                                    or hits and (not self._replay_calls or self._replay_calls != self._replay_returns)
                                    or misses and self._replay_calls):
                                raise ValueError('graph attempt did not produce one matching serving/replay outcome')
                            self._append_graph(role, rows, participants, hits, 0 if hits else reason)
                        except BaseException as error:
                            self._fail('graph-after-observation', type(error).__name__)
                    return result
                finally:
                    self._graph_active = False
            self._wrapper = attempt
            bound.forward = attempt
        except BaseException:
            for offset in range(self._graph_binding_count):
                pieces, index, original, _ = self._graph_bindings[offset]
                pieces.graphs[index] = original
                self._graph_bindings[offset] = None
            self._graph_binding_count = 0
            self._decoder = self._graph_bound = self._original = self._wrapper = self._previous_take = None
            raise
        return self

    def _append_graph(self, role, rows, participants, outcome, reason):
        row = (role, rows, participants, outcome, reason, self._replay_calls, self._replay_returns)
        if (any(type(value) is not int or not 0 <= value <= sys.maxsize for value in row)
                or self._replay_returns > self._replay_calls):
            self._fail('graph-observation', 'ValueError')
            return
        if self._count == self.resources['maximum_events']:
            self._fail('event-capacity', 'OverflowError')
            return
        offset = self._count * len(GRAPH_COLUMNS)
        for column, value in enumerate(row):
            self._rows[offset + column] = value
        self._count += 1

    def _close_graphs(self):
        if not self._closed:
            bound = self._graph_bound
            if bound is not None:
                if bound.forward is not self._wrapper or self._graph_active:
                    raise RuntimeError('graph observer binding changed or decoder was not drained')
                for offset in range(self._graph_binding_count):
                    pieces, index, original, witness = self._graph_bindings[offset]
                    if pieces.graphs[index] is not witness:
                        raise RuntimeError('graph replay witness binding changed before retirement')
                if self._had_take:
                    bound.forward = self._previous_take
                else:
                    del bound.forward
                for offset in range(self._graph_binding_count):
                    pieces, index, original, _ = self._graph_bindings[offset]
                    pieces.graphs[index] = original
                    self._graph_bindings[offset] = None
            self._decoder = self._graph_bound = self._original = self._wrapper = self._previous_take = None
            self._closed = True
        return self.report()

    def _graph_report(self):
        width = len(GRAPH_COLUMNS)
        events = [list(self._rows[i:i + width]) for i in range(0, self._count * width, width)]
        histogram = graph_histogram(events)
        return {'schema': 'tensorfold.graph-attempt-diagnostic-v1', 'closed': self._closed,
                'complete': self._closed and self._failure is None, 'failure': self._failure,
                'diagnostic_only': True, 'P3_valid': False, 'resources': dict(self.resources),
                'columns': list(GRAPH_COLUMNS), 'events': events, 'histogram': histogram,
                'roles': list(GRAPH_ROLES), 'outcomes': list(GRAPH_OUTCOMES), 'reasons': list(GRAPH_REASONS),
                'graph_bindings': self._graph_binding_count,
                'scope': 'Host graph attempts and replay API calls/returns;not GPU completion/activity or P3. '
                         'C1/C8 plan N>=2 belongs to C8;N1 mixes C1 warmup/timed and C8 tails.'}


def graph_histogram(events):
    grouped = {}
    for row in events:
        key = tuple(row[:5])
        count, calls, returned = grouped.get(key, (0, 0, 0))
        grouped[key] = (count + 1, calls + row[5], returned + row[6])
    histogram = [{'role': GRAPH_ROLES[key[0]], 'total_rows': key[1], 'participants': key[2],
                  'outcome': GRAPH_OUTCOMES[key[3]], 'reason': GRAPH_REASONS[key[4]],
                  'attempts': values[0], 'replay_calls': values[1], 'replay_returns': values[2]}
                 for key, values in sorted(grouped.items())]
    return histogram
