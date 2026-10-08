"""ReadAhead instrumentation retaining its constructor and ownership contract."""


def read_ahead_recorder(base,records):
    class ObservedReadAhead(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.observed_futures = set()
            self.observed_threads = set()
            self.observed_closed = False
            records.append(self)

        def queue(self,*args,**kwargs):
            result = super().queue(*args,**kwargs)
            self.observed_futures.update(self.ahead.values())
            if self.pool is not None:
                self.observed_threads.update(self.pool._threads)
            return result

        def close(self):
            if self.pool is not None:
                self.observed_threads.update(self.pool._threads)
            super().close()
            self.observed_closed = True

    return ObservedReadAhead
