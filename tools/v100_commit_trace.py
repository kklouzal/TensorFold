"""Finite diagnostic commit-batch timestamps; never a performance observer.

The caller admits ``resources`` before installation and drains the decoder
before close. Records contain integers only, not tokens, Tensors or Streams.
Several MTP tokens may share one observed batch; no individual token timing is
inferred. Diagnostic exhaustion never suppresses or retries model work.
"""
from array import array
import sys
import time


COLUMNS = ('stream_sid', 'begin_ns', 'end_ns', 'committed_before', 'committed_after', 'committed_delta')


def resources(max_events):
    if type(max_events) is not int or not 0 < max_events <= sys.maxsize // (len(COLUMNS) * 8):
        raise ValueError('finite positive bounded diagnostic event count required')
    return {'maximum_events': max_events, 'retained_event_bytes': max_events * len(COLUMNS) * 8,
            # Native signed integers take at most20 digits. Reserve a second
            # materialization of Python rows plus bounded JSON/failure metadata.
            'serialization_bytes': max_events * (len(COLUMNS) * 80 + 256) + 8192,
            'scope': 'diagnostic host storage; no model workload or GPU allowance'}


class CommitTrace:
    def __init__(self, max_events, *, clock=None, marker=None):
        self.resources = resources(max_events)
        self._rows = array('q', [0]) * (max_events * len(COLUMNS))
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
        width = len(COLUMNS)
        return {'schema': 'tensorfold.commit-batch-diagnostic-v1', 'closed': self._closed,
                'complete': self._closed and self._failure is None, 'failure': self._failure,
                'diagnostic_only': True, 'P3_valid': False,
                'resources': dict(self.resources), 'clock_anchors': dict(self._anchors),
                'columns': list(COLUMNS),
                'events': [list(self._rows[i:i + width]) for i in range(0, self._count * width, width)],
                'scope': 'Host intervals around committed model-output batches; includes publication/backpressure. '
                         'Not GPU completion, per-token spacing, proposal acceptance or P3 evidence.'}
