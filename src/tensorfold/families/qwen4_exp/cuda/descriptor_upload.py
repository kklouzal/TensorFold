"""Bounded host descriptor DMA leases; GPU descriptor addresses remain canonical."""
import os
import torch

PROFILE = "source-owner-descriptor-ring-trial"
SLOTS_ENV = "TENSORFOLD_OWNER_DESCRIPTOR_UPLOAD_SLOTS"
METADATA_BOUND = 16384


def selected_slots():
    if os.environ.get("TENSORFOLD_SM70_PROFILE", "reference") != PROFILE:
        if SLOTS_ENV in os.environ:
            raise ValueError("descriptor upload slots require the explicit ring trial profile")
        return 1
    raw = os.environ.get(SLOTS_ENV, "4")
    if raw not in ("2", "4", "8"):
        raise ValueError("descriptor upload ring must contain2,4or8 slots")
    return int(raw)


def host_commitment(rows, slots):
    if type(rows) is not int or not 0 < rows <= 256 or type(slots) is not int or slots not in (1, 2, 4, 8):
        raise ValueError("invalid descriptor upload ring envelope")
    return slots * rows * 16 * 8 + (METADATA_BOUND if slots > 1 else 0)


class DescriptorUploadRing:
    """One actor writes; pending host bytes remain borrowed until their event."""
    __slots__ = ("slots", "hosts", "views", "events", "pending", "cursor", "active",
                 "latest_event", "latest_stream", "error", "uncertain")

    def __init__(self, rows, slots, device, *, event_factory=None):
        host_commitment(rows, slots)
        self.slots, self.cursor, self.active = slots, 0, None
        self.hosts, self.views, self.events = None, (), ()
        self.pending = ()
        self.latest_event = self.latest_stream = self.error = None
        self.uncertain = False
        self.pending = [False] * slots
        cuda = torch.device(device).type == "cuda"
        self.hosts = torch.zeros((slots, rows, 16), dtype=torch.int64, pin_memory=cuda)
        self.views = tuple(self.hosts[index] for index in range(slots))
        factory = torch.cuda.Event if event_factory is None and cuda else event_factory
        if factory is not None:
            events = []
            self.events = events  # Retain partial construction before another event can fail.
            for _ in range(slots):
                events.append(factory())
            self.events = tuple(events)
            if cuda:
                stream = torch.cuda.current_stream(device)
                for event in self.events:
                    event.record(stream)
                for event in self.events:
                    event.synchronize()  # Materialize every handle before initial model admission.
                self.latest_event, self.latest_stream = self.events[0], stream.cuda_stream
        self.latest_event = self.events[0] if self.events else None

    def poison(self, error):
        if self.error is None:
            self.error = error
        self.uncertain = True

    def acquire(self):
        if self.error is not None:
            raise RuntimeError("descriptor upload owner is poisoned; retire the model") from self.error
        if self.active is not None:
            raise RuntimeError("descriptor upload host lease is already active")
        try:
            selected = None
            for offset in range(self.slots):
                index = (self.cursor + offset) % self.slots
                if not self.pending[index] or self.events[index].query():
                    self.pending[index] = False
                    selected = index
                    break
            if selected is None:
                selected = self.cursor
                self.events[selected].synchronize()
                self.pending[selected] = False
            self.active = selected
            return self.views[selected]
        except BaseException as error:
            self.poison(error)
            raise

    def before_gpu_write(self, stream):
        if self.latest_event is not None and self.latest_stream != stream.cuda_stream:
            stream.wait_event(self.latest_event)

    def publish(self, stream):
        if self.active is None:
            raise RuntimeError("descriptor upload needs its acquired host lease")
        index = self.active
        try:
            if self.events:
                self.pending[index] = True  # Unknown record completion cannot permit reuse.
                self.events[index].record(stream)
                self.latest_event, self.latest_stream = self.events[index], stream.cuda_stream
            self.active, self.cursor = None, (index + 1) % self.slots
            return self.latest_event
        except BaseException as error:
            self.poison(error)
            raise

    def drain(self, *, globally_fenced=False):
        if self.uncertain and not globally_fenced:
            raise RuntimeError("uncertain descriptor upload retirement requires a successful global fence") from self.error
        try:
            if not globally_fenced:
                for index, pending in enumerate(self.pending):
                    if pending:
                        self.events[index].synchronize()
            for index in range(len(self.pending)):
                self.pending[index] = False
            self.active = None
        except BaseException as error:
            self.poison(error)
            raise
